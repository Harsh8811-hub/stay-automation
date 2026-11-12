"""
Room Query Handler Node — handles selected room details, feature lookups, and booking summaries.
"""

import logging
from app.workflows.state import ConversationState
from app.repositories.room_repository import get_all_rooms
from app.prompts.templates import text_card
from app.services.gemini_service import _get_model, translate_cards_if_needed
from app.utils.intent_overrides import extract_nights_from_message
from app.services.llm import compat as genai

logger = logging.getLogger(__name__)

def _append_booking_reminder(text: str, state: dict) -> str:
    from app.utils.booking_resume import append_booking_resume
    return append_booking_resume(text, state)


def _effective_max_occupancy(occ) -> int:
    """
    The most guests one room of this type holds, from parts that may disagree.

    Every room at hotel 108959 stores max_occupancy=2 next to max_adults=2 and
    max_children=1. The stored total therefore contradicts the parts it is made
    of, and trusting it fed the model "up to 8 adults and 4 children, i.e. 8
    guests in total" — a sentence that refutes itself, describing an option that
    really holds 12. Taking the larger of the two is the only reading that never
    tells a guest their own offered room holds fewer people than its bed count
    allows.

    `max` rather than `or`: a missing total falls back to adults + children, and
    a total that is merely too small is corrected rather than believed.
    """
    return max(
        occ.max_occupancy or 0,
        (occ.max_adults or 0) + (occ.max_children or 0),
    )


def _count_rooms_per_option(allocation_options: list) -> dict[str, dict]:
    """
    Map `room_id` -> {quantity, option_number, assignments} for what a set of
    allocation options actually offers.

    Counting is the whole point. An option of "4 x Superior Family Room" holds
    FOUR rooms, and the guest's party has to be judged against their combined
    capacity — not one room's. Dropping this count is what produced defect #8:
    the model was handed max 2 adults / 1 child next to a party of 5 adults +
    3 children and told the guest their own selection did not fit them.

    The first option containing a room_id wins, matching the "Option N" label
    the guest is shown.
    """
    out: dict[str, dict] = {}
    for opt_idx, opt in enumerate(allocation_options or [], 1):
        for rm_data in (opt.get("rooms") or []):
            rid = rm_data.get("room_id")
            if not rid:
                continue
            entry = out.get(rid)
            if entry is None:
                entry = out[rid] = {
                    "quantity": 0,
                    "option_number": opt_idx,
                    "assignments": [],
                }
            if entry["option_number"] != opt_idx:
                continue  # already counted under an earlier option
            entry["quantity"] += 1
            entry["assignments"].append({
                "assigned_adults": rm_data.get("assigned_adults"),
                "assigned_children": rm_data.get("assigned_children"),
            })
    return out


async def handle_room_query(state: ConversationState) -> ConversationState:
    """
    Process SELECTED_ROOM_DETAILS, BOOKING_SUMMARY, and ROOM_FEATURE_LOOKUP.
    Resolves references to 'this room', 'the room I booked', etc. using selected room state.
    """
    intent = state.get("intent", "UNKNOWN")
    hotel_id = state.get("hotel_id", 0)
    selected_rooms = state.get("selected_rooms", [])
    allocation_options = state.get("allocation_options", [])
    query_room_type = state.get("pricing_query_room_type")
    query_nights = state.get("pricing_query_nights")

    # Room-type pricing query (e.g. "deluxe rooms price for 2 nights")
    if intent == "BOOKING_SUMMARY" and not allocation_options and not selected_rooms:
        pref = query_room_type or state.get("room_type_preference")
        nights = query_nights or extract_nights_from_message(state.get("user_message", ""))
        num_rooms = state.get("pricing_query_num_rooms") or state.get("num_rooms")
        if not num_rooms:
            from app.utils.intent_overrides import extract_num_rooms_from_message
            num_rooms = extract_num_rooms_from_message(state.get("user_message", ""))
        if pref and nights:
            from app.services.room_service import build_room_type_pricing_card

            adults = state.get("adults") or (num_rooms if num_rooms else 2)
            card = await build_room_type_pricing_card(
                hotel_id=hotel_id,
                room_type_keyword=pref,
                num_nights=nights,
                adults=adults,
                children=state.get("children") or 0,
                num_rooms=num_rooms,
                meal_plan_preference=state.get("meal_plan_preference"),
                # Read-only: the card renders its own options into `text`, and
                # needs the guest's language to do it.
                state=state,
            )
            if card:
                footer = text_card(
                    "Would you like to book one of these? Share your **check-in and check-out dates** and I'll get you booked! 📅"
                )
                return {
                    **state,
                    "bot_response": [card] + footer,
                    "next_action": "room_query_answered",
                }
            label = pref.strip().title()
            return {
                **state,
                "bot_response": text_card(
                    f"I'm sorry, I couldn't find any **{label}** rooms right now. Would you like to see all available room types?"
                ),
                "next_action": "room_query_answered",
            }

    # Pricing totals while room options are on screen (no selection yet)
    if intent == "BOOKING_SUMMARY" and not selected_rooms and allocation_options:
        from app.services.booking_engine import calculate_num_nights
        from app.services.room_service import (
            build_room_options_card,
            enrich_options_with_stay_totals,
            format_stay_totals,
        )

        num_nights = calculate_num_nights(state.get("check_in", ""), state.get("check_out", ""))
        options = await enrich_options_with_stay_totals(
            allocation_options,
            num_nights,
            hotel_id,
            state.get("check_in"),
            state.get("check_out"),
        )
        # Answer the question from the options already on screen, rather than
        # putting the same options on screen a second time.
        #
        # `_packing_price` prints per-night rates and nothing else, on purpose —
        # the card's rows were changed to rates-only on 2026-09-24 — so "what's
        # the total?" is a question the card cannot answer, and re-sending it is
        # a non-answer that hands the guest the rates they are already reading.
        # A live conversation on 2026-09-26 asked three times and got the same
        # card three times, then asked a fourth time. `format_stay_totals` does
        # the multiplication against the option's priced stay figure, under the
        # card's own numerals, so the numbers in the reply are the numbers in
        # the question the guest is answering.
        #
        # `rooms_shown` on both returns below, not `room_query_answered`: the
        # *card* above is still the live question, and its numbered reply is
        # resolved by `room_card_reply`, which gates on exactly this action.
        # With the old value the guest answered a question the re-shown card had
        # just asked and nothing heard them — the resolver declined, and the
        # digits fell through to the generic numeric block, which reads "3" as
        # "the third priced option". `room_query_answered` had no reader
        # anywhere in app/; this is the only place it was load-bearing, and it
        # was load-bearing in the wrong direction. `handle_room_query` has a
        # plain edge to `save_state` (graph.py:211), so neither return can reach
        # `route_after_search` and re-run the search.
        totals = format_stay_totals(options, num_nights, state)
        if totals:
            return {
                **state,
                "bot_response": await translate_cards_if_needed(text_card(totals), state),
                "next_action": "rooms_shown",
            }

        # Nothing could be totalled — no priced row, or the engine returned no
        # figure for any of them. The card is then the honest reply: it is the
        # options themselves, and the guest can still see the rates.
        #
        # Render the catalogue into `text`, the way every other producer of a
        # room_options card does. WhatsApp's card renderer prints a
        # room_options card's `text` and never walks `data.options`, so a card
        # whose text is only a preamble reaches the guest as a bare
        # "Please select an option:" with nothing underneath it — which is
        # exactly what Chat A 06:35:26 and Chat B 06:48:50 received.
        # `data.options` is kept for clients that read it structurally.
        # Rendered by the same builder the search uses, so the re-shown card
        # letters its combinations the same way as the one already above it in
        # the chat. This call site used `format_s4_room_options`, whose options
        # are numbered 1..N — a guest who asked "what's the total?" mid-browse
        # got a numbered card under a lettered one, and "1" meant two different
        # things in the same conversation.
        #
        # `rooms` is not passed: this handler has the priced options but not the
        # catalogue, and the combinations section — which is what a total
        # question is about — renders correctly without it. The reply range on
        # the card is derived from the options, so the letters still match.
        card = build_room_options_card(options, state=state)
        card["data"]["options"] = options
        card["data"]["num_nights"] = num_nights
        return {
            **state,
            "bot_response": await translate_cards_if_needed([card], state),
            "next_action": "rooms_shown",
        }

    # Gather rooms to describe for feature lookups and details.
    #
    # `quantities` is the point of this block. An option like
    # "4 x Superior Family Room" is FOUR rooms, and the guest's party has to be
    # judged against their combined capacity. The code here used to dedupe by
    # room_id and throw that count away, so generate_room_details_response was
    # handed one room's occupancy (max 2 adults / 1 child) next to a party of
    # 5 adults + 3 children, and the model told the guest their own selection
    # would not fit them — Chat B 06:47:39, defect #8.
    #
    # Deduping also stops the selected_rooms branch from appending the same room
    # once per SelectedRoom entry, which used to emit one identical photo card
    # per room in `target_rooms` below.
    from app.repositories.room_repository import get_all_rooms
    all_rooms = await get_all_rooms(hotel_id)
    target_rooms = []
    quantities: dict[str, int] = {}
    room_assignments: dict[str, list[dict]] = {}
    option_numbers: dict[str, int] = {}

    def _assignment(entry: dict) -> dict:
        return {
            "assigned_adults": entry.get("assigned_adults"),
            "assigned_children": entry.get("assigned_children"),
        }

    if selected_rooms:
        for sel_rm in selected_rooms:
            room_match = next((rm for rm in all_rooms if rm.room_id == sel_rm.get("room_id") or rm.name.lower() == sel_rm.get("room_name", "").lower()), None)
            if not room_match:
                continue
            rid = room_match.room_id
            if rid not in quantities:
                quantities[rid] = 0
                room_assignments[rid] = []
                target_rooms.append(room_match)
            quantities[rid] += 1
            room_assignments[rid].append(_assignment(sel_rm))
    elif allocation_options:
        for rid, meta in _count_rooms_per_option(allocation_options).items():
            room_match = next((rm for rm in all_rooms if rm.room_id == rid), None)
            if room_match:
                quantities[rid] = meta["quantity"]
                option_numbers[rid] = meta["option_number"]
                room_assignments[rid] = meta["assignments"]
                target_rooms.append(room_match)
    else:
        pref = state.get("room_type_preference") or query_room_type
        if pref:
            from app.services.room_alias_resolver import (
                filter_rooms_by_reference,
                aliases_from_hotel_config,
            )
            aliases = aliases_from_hotel_config(state.get("hotel_config"))
            threshold = 90.0
            cfg = state.get("hotel_config") or {}
            if isinstance(cfg, dict) and cfg.get("room_match_threshold") is not None:
                try:
                    threshold = float(cfg["room_match_threshold"])
                except (TypeError, ValueError):
                    pass
            target_rooms = filter_rooms_by_reference(
                pref, all_rooms, aliases=aliases, fuzzy_threshold=threshold
            )
        if not target_rooms:
            target_rooms = all_rooms

    from app.services.room_service import _friendly_rate_plan_name
    selected_rooms_details = []
    for room_match in target_rooms:
        opt_num = option_numbers.get(room_match.room_id)
        opt_num_str = f"Option {opt_num}" if opt_num else ""
        quantity = max(quantities.get(room_match.room_id, 1), 1)
        from app.services.pricing_engine import PricingEngine
        rate_plans_info = [{"name": _friendly_rate_plan_name(rp.name), "price": PricingEngine.estimate_inclusive_price(rp.base_price)} for rp in room_match.rate_plans if rp.active]
        occ = room_match.occupancy
        max_adults = occ.max_adults or 0
        max_children = occ.max_children or 0
        max_occupancy = _effective_max_occupancy(occ)
        details_dict = {
            "option_number": opt_num_str,
            "room_id": room_match.room_id,
            "name": room_match.name,
            "room_type": room_match.room_type,
            "description": room_match.description,
            "room_size": f"{room_match.room_size} Sq Ft" if room_match.room_size else "",
            "beds": room_match.beds,
            "bathroom_count": room_match.bathroom_count,
            "amenities": [k for k, v in room_match.amenities.items() if v],
            "rate_plans": rate_plans_info,
            # How many rooms of this type the guest is actually being offered,
            # and what they add up to. Anything that judges the party against
            # this room must read combined_occupancy, not occupancy.
            "quantity": quantity,
            "occupancy": {
                "base_adults": occ.base_adults,
                "max_adults": max_adults,
                "base_children": occ.base_children,
                "max_children": max_children,
                "max_occupancy": max_occupancy,
            },
            "combined_occupancy": {
                "rooms": quantity,
                "max_adults": max_adults * quantity,
                "max_children": max_children * quantity,
                "max_occupancy": max_occupancy * quantity,
            },
            "room_assignments": room_assignments.get(room_match.room_id, []),
            "extra_adult_rate": occ.extra_adult_rate,
            "extra_child_rate": occ.extra_child_rate,
        }
        selected_rooms_details.append(details_dict)

    booking_summary = {
        "check_in": state.get("check_in"),
        "check_out": state.get("check_out"),
        "adults": state.get("adults"),
        "children": state.get("children"),
        "num_rooms": state.get("num_rooms"),
        "total_price": state.get("total_price"),
        "price_breakdown": state.get("price_breakdown"),
    }

    if intent == "BOOKING_SUMMARY":
        response_text = _generate_booking_summary_text(booking_summary, selected_rooms)
        response_text = _append_booking_reminder(response_text, state)
        return {
            **state,
            "bot_response": text_card(response_text),
            "next_action": "room_query_answered",
        }

    # Otherwise, selected_room_details or room_feature_lookup
    answer, wants_photos = await generate_room_details_response(
        question=state.get("user_message", ""),
        room_details=selected_rooms_details,
        booking_summary=booking_summary,
    )

    answer = _append_booking_reminder(answer, state)
    
    media_cards = []
    
    if wants_photos and target_rooms:
        from app.services.hotel_media_service import media_delivery_card
        for room in target_rooms:
            media_cards.append(media_delivery_card(
                hotel_id=state.get("hotel_id"),
                entity_type="room",
                entity_id=room.room_id,
                tags=[room.name.lower()],
                aliases=room.aliases if hasattr(room, "aliases") and room.aliases else [],
                caption=room.name,
                limit=3,
                explicit_request=True,
            ))

    _reply_cards = [*text_card(answer), *media_cards]
    bot_response = await translate_cards_if_needed(_reply_cards, state)

    return {
        **state,
        "bot_response": bot_response,
        "next_action": "room_query_answered",
    }


def _generate_booking_summary_text(summary: dict, selected_rooms: list) -> str:
    check_in = summary.get("check_in")
    check_out = summary.get("check_out")
    # `.get(key, default)` only fires when the key is *absent*. The summary is
    # built from `state.get("adults")`, so on a fresh session the key is
    # present and None — and "None Adults, None Children" reached a guest.
    # Coerce the value, not the lookup.
    adults = summary.get("adults") or 2
    children = summary.get("children") or 0
    total = summary.get("total_price")
    breakdown = summary.get("price_breakdown") or {}
    tax_percent = breakdown.get("tax_percent", 0.0) or 0.0
    
    lines = [
        "Sure! Here are your current booking details:",
    ]
    if check_in and check_out:
        try:
            from datetime import datetime
            ci_dt = datetime.strptime(check_in, "%Y-%m-%d").strftime("%d/%m/%Y")
            co_dt = datetime.strptime(check_out, "%Y-%m-%d").strftime("%d/%m/%Y")
        except ValueError:
            ci_dt, co_dt = check_in, check_out
        lines.append(f"📅 **Dates:** {ci_dt} to {co_dt}")
    else:
        lines.append("📅 **Dates:** Not yet provided — please share your check-in and check-out dates.")
    lines += [
        f"👥 **Guests:** {adults} Adults, {children} Children",
        "🏨 **Selected Rooms:**"
    ]
    from app.services.room_service import _friendly_rate_plan_name
    from collections import defaultdict

    breakdown_rooms = breakdown.get("rooms") or []
    if breakdown_rooms:
        for item in breakdown_rooms:
            if isinstance(item, dict):
                r_name = item.get("room_name", "Room")
                plan = _friendly_rate_plan_name(item.get("rate_plan", ""))
                qty = item.get("quantity", 1)
                ppn = item.get("price_per_night", 0)
                prefix = f"{qty}× " if qty > 1 else ""
                lines.append(f"   • {prefix}{r_name} ({plan}) - ₹{round(ppn):,.0f} / night (incl. GST)")
    else:
        room_counts: dict[tuple, int] = defaultdict(int)
        room_samples: dict[tuple, dict] = {}
        for rm in selected_rooms:
            key = (rm.get('room_name'), rm.get('rate_plan_name'), rm.get('price_per_night', 0))
            room_counts[key] += 1
            room_samples[key] = rm

        for key, count in room_counts.items():
            rm = room_samples[key]
            friendly_plan = _friendly_rate_plan_name(rm.get('rate_plan_name', ''))
            base_pn = rm.get('price_per_night', 0)
            from app.services.pricing_engine import PricingEngine
            inc_pn = PricingEngine.estimate_inclusive_price(base_pn)
            prefix = f"{count}× " if count > 1 else ""
            lines.append(f"   • {prefix}{rm.get('room_name')} ({friendly_plan}) - ₹{inc_pn:,.0f} / night (incl. GST)")
        
    if total is not None:
        lines.append(f"\n💳 **Total Price (including GST):** ₹{total:,.0f}")
    
    return "\n".join(lines)


async def generate_room_details_response(
    question: str,
    room_details: list[dict],
    booking_summary: dict,
) -> tuple[str, bool]:
    model = _get_model()

    # ── MEAL PLAN GUARD & AUDIT LOGGING ──
    question_lower = (question or "").lower()
    meal_keywords = ["breakfast", "dinner", "lunch", "meal", "cp", "map", "ap", "food", "half board", "full board", "all inclusive"]
    meal_plan_requested = any(kw in question_lower for kw in meal_keywords)

    if meal_plan_requested:
        logger.info("MEAL_PLAN_SHOWN | trigger=guest_request | plan=CP/MAP")
    else:
        logger.info("MEAL_PLAN_SUPPRESSED | reason=not_requested")

    context_str = "SELECTED ROOM DETAILS FROM DATABASE:\n"
    for r in room_details:
        opt_str = f" ({r.get('option_number')})" if r.get('option_number') else ""
        qty = int(r.get("quantity") or 1)
        qty_str = f" — this option includes {qty} of this room type" if qty > 1 else ""
        context_str += f"- Room Name: {r.get('name')}{opt_str}{qty_str}\n"
        context_str += f"  Type: {r.get('room_type')}\n"
        context_str += f"  Description: {r.get('description')}\n"
        if r.get('room_size'):
            context_str += f"  Size: {r.get('room_size')}\n"
        
        # Beds
        beds = r.get("beds", [])
        if beds:
            bed_str = ", ".join(f"{b.get('count')}x {b.get('type')}" for b in beds)
            context_str += f"  Beds: {bed_str}\n"
        else:
            context_str += f"  Beds: Not available in database\n"
            
        # Bathroom
        context_str += f"  Bathrooms: {r.get('bathroom_count')} bathroom(s)\n"
        
        # Amenities
        amenities = r.get("amenities", [])
        if amenities:
            context_str += f"  Amenities: {', '.join(amenities)}\n"
        else:
            context_str += f"  Amenities: Not available in database\n"
            
        # Rate Plans (Meal Plans) — Guarded by meal_plan_requested
        rate_plans = r.get("rate_plans", [])
        if rate_plans:
            if meal_plan_requested:
                rp_str = ", ".join(f"{rp['name']} (₹{float(rp['price']):,.0f}/night incl. taxes)" for rp in rate_plans)
            else:
                # EP / Room Only rate plan only
                ep_plan = next((rp for rp in rate_plans if "room only" in rp['name'].lower() or "ep" in rp['name'].lower()), rate_plans[0])
                rp_str = f"{ep_plan['name']} (₹{float(ep_plan['price']):,.0f}/night incl. taxes)"
            context_str += f"  Rate Plans: {rp_str}\n"
            
        # Occupancy
        occ = r.get("occupancy", {})
        if occ:
            context_str += (
                f"  Occupancy of ONE such room: Max {occ.get('max_occupancy')} guests "
                f"(Max {occ.get('max_adults')} Adults, Max {occ.get('max_children')} Children). "
                f"Base occupancy: {occ.get('base_adults')} Adults, {occ.get('base_children')} Children.\n"
            )
            comb = r.get("combined_occupancy") or {}
            if qty > 1 and comb:
                context_str += (
                    f"  COMBINED CAPACITY OF THIS OPTION ({qty} rooms of this type): "
                    f"up to {comb.get('max_adults')} adults and {comb.get('max_children')} children, "
                    f"i.e. {comb.get('max_occupancy')} guests in total.\n"
                )
            assigned = r.get("room_assignments") or []
            if qty > 1 and assigned:
                per_room = ", ".join(
                    f"{a.get('assigned_adults') or 0} adults/{a.get('assigned_children') or 0} children"
                    for a in assigned
                )
                context_str += (
                    f"  THIS OPTION ALREADY PLACES THE PARTY ACROSS ITS {qty} ROOMS AS: {per_room}. "
                    f"It was built for this party and holds them.\n"
                )
            if occ.get('extra_adult_rate'):
                context_str += f"  Extra Adult Rate: ₹{float(r.get('extra_adult_rate', 0)):,.0f}/night\n"
            if occ.get('extra_child_rate'):
                context_str += f"  Extra Child Rate: ₹{float(r.get('extra_child_rate', 0)):,.0f}/night\n"

    total_price_val = booking_summary.get('total_price')
    formatted_total = f"₹{float(total_price_val):,.0f}" if total_price_val is not None else "Not set"
    from app.utils.helpers import format_display_date
    # Same None trap as the summary text above: an unset date must read as
    # "not set" to the model, never as the string "None".
    ci_disp = format_display_date(booking_summary.get('check_in')) or booking_summary.get('check_in') or "Not set"
    co_disp = format_display_date(booking_summary.get('check_out')) or booking_summary.get('check_out') or "Not set"
    adults_ctx = booking_summary.get('adults') or 2
    children_ctx = booking_summary.get('children') or 0

    prompt = f"""You are a helpful hotel receptionist.

    RULES:
    1. Answer the guest's question naturally, conversationally, and accurately based ONLY on the room details and booking state provided below.
    2. Keep your response concise (1-3 sentences max). Do NOT dump full lists of amenities or Sq Ft unless the guest explicitly asked for them.
    3. Format all prices with ₹ symbol and clean integer formatting (e.g., ₹1,550). Never wrap prices in double quotes.
    4. By default, ONLY mention the Room Only (EP) price. Do NOT list CP or MAP meal plan upgrade prices unless the guest explicitly asked about breakfast or meal plans. If meal plan was NOT asked, include a single line: "Breakfast and meal plan upgrades available — ask me for details."
    5. Never use abbreviations like EP, CP, MAP, or AP. Always use plain English (e.g., 'Room Only', 'Includes Breakfast').
    6. When a room is offered in multiples (e.g. "4 x Superior Family Room"), the guest is looking at ALL of those rooms together. Judge whether their party fits against the "COMBINED CAPACITY OF THIS OPTION" line, NEVER against the occupancy of a single room. If that line says the option holds the party, confirm it does. Never tell a guest that an option they were just offered cannot accommodate them when the combined capacity covers them.
    7. Reply in JSON only, with exactly these two keys: {{"answer": "<your reply to the guest>", "wants_photos": <true or false, true if the guest is asking to see pictures>}}.

    ROOM DETAILS FROM DATABASE:
    {context_str}

    CURRENT BOOKING SUMMARY:
    - Check-in: {ci_disp}
    - Check-out: {co_disp}
    - Guests: {adults_ctx} adults, {children_ctx} children
    - Total Price: {formatted_total}

    GUEST QUESTION:
    {question}

    Answer:"""

    try:
        from pydantic import BaseModel, Field
        class RoomResponseSchema(BaseModel):
            answer: str = Field(description="Your conversational response to the guest's question.")
            wants_photos: bool = Field(description="True if the guest's question implies they want to see pictures, photos, or visual media of the room. True for any language (e.g. 'tasvir dhikav', 'photo', 'show me').")

        import asyncio
        loop = asyncio.get_running_loop()
        # `response_mime_type="application/json"` reaches the provider as
        # `response_format: {"type": "json_object"}` (see
        # app/services/llm/genai_compat.py). DeepSeek rejects that with HTTP 400
        # unless the *prompt* contains the word "json" — Gemini did not care.
        # That is why rule 7 above spells out the exact shape: it is both the
        # shape instruction and the literal the provider looks for. Removing it
        # brought back "I'm having trouble fetching its detailed features"
        # (prod 2026-10-23 06:13:38).
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        response_mime_type="application/json",
                        response_schema=RoomResponseSchema,
                    ),
                )
            ),
            timeout=15.0
        )
        import json
        data = json.loads(response.text.strip())
        return data.get("answer", "Here are the details."), data.get("wants_photos", False)
    except Exception as e:
        logger.error(f"Failed to generate room details response: {e}")
        return "I can confirm you have selected the room, but I'm having trouble fetching its detailed features right now. Please let me know if you'd like to proceed with the booking.", False
