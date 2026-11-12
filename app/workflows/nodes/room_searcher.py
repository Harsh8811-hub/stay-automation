"""
Room Searcher Node — queries MongoDB for matching rooms.
"""

import logging

from app.config.settings import get_settings
from app.services.room_service import clean_room_type_name, find_suitable_rooms
from app.utils.room_requirement_parse import is_an_explicit_split
from app.workflows.state import ConversationState
from app.services.gemini_service import generate_room_recommendations, translate_cards_if_needed  # Kept in case used elsewhere, but will use Python engine instead
from app.repositories.room_repository import get_all_rooms
from typing import cast

logger = logging.getLogger(__name__)


def _room_preview_media_cards(options: list[dict], state: dict | None = None) -> list[dict]:
    """Request one primary image for each suggested room category, with price matching the text exactly."""
    from app.services.hotel_media_service import MAX_IMAGES_PER_RESPONSE, media_delivery_card

    from app.services.booking_engine import calculate_num_nights
    state = state or {}
    ci = state.get("check_in")
    co = state.get("check_out")
    num_nights = calculate_num_nights(ci, co) if (ci and co) else (state.get("num_nights") or 1)

    cards: list[dict] = []
    seen_room_ids: set[str] = set()
    hotel_id = state.get("hotel_id")

    for i, option in enumerate(options[:4], 1):
        rms = option.get("rooms", [])
        price_pn = float(option.get("total_price_per_night_inclusive") or option.get("total_price_per_night") or 0)
        stay_val = option.get("total_price_for_stay")
        stay_total = float(stay_val) if stay_val is not None else (price_pn * num_nights)

        for room in rms:
            room_id = str(room.get("room_id") or "").strip()
            room_name = str(room.get("room_name") or "Room").strip()
            if not room_id or room_id in seen_room_ids:
                continue
            seen_room_ids.add(room_id)

            caption = room_name

            cards.append(media_delivery_card(
                hotel_id=hotel_id,
                entity_type="room",
                entity_id=room_id,
                tags=[room_name],
                aliases=room.get("aliases", []),
                caption=caption,
                limit=1,
                is_preview=True,
            ))
            if len(cards) >= MAX_IMAGES_PER_RESPONSE:
                return cards
    return cards


async def _room_class_gap_response(
    state: ConversationState,
    hotel_id: int,
    check_in: str | None,
    check_out: str | None,
    effective_features: list[str] | None,
) -> dict | None:
    """
    Name the AC class that emptied the search, and what is actually free.

    AC / Non-AC is a hard filter — `allocate_rooms` returns nothing rather than
    quietly showing the other class — so an empty result can mean "no non-AC
    room is free these nights" while several AC rooms sit open. The generic
    "we don't have rooms available that match your requirements" card is the
    only thing that reaches the guest in that case, and it hides both facts:
    which part of the request failed and what the hotel does have.

    Live cost, guest 918017904770 on 2026-10-17: one non-AC room for 18–21 Oct.
    Both non-AC types were closed on the 20th, two AC types were free on all
    three nights, and the guest received the same sentence three times before a
    staff member answered by hand with a room at another property.

    Returns the state updates for a better answer, or None to leave the
    existing no-rooms card in place. None is the right answer whenever there is
    nothing to offer — no other room is free, or the dates are missing — so the
    caller falls through to exactly today's behaviour.
    """
    from app.services.allocation_engine import is_hard_inventory_feature

    hard = [f for f in (effective_features or []) if is_hard_inventory_feature(f)]
    if not hard or not (check_in and check_out):
        return None

    from app.services import inventory_service
    try:
        open_rooms = await inventory_service.get_available_rooms(hotel_id, check_in, check_out)
    except Exception as e:
        # The guest is about to be told "nothing available"; a failed lookup is
        # not evidence for that, but it also must not raise out of this node and
        # turn a no-rooms turn into a 500. Fall back to the old card.
        logger.warning("ROOM_CLASS_GAP | availability lookup failed: %s", e)
        return None

    # The rooms that survived the stay — i.e. the ones the guest could book now.
    from app.services.allocation_engine import room_matches_feature

    if any(all(room_matches_feature(r, f) for f in hard) for r in open_rooms or []):
        # A room of the requested class IS open, so the class is not what failed
        # — occupancy, budget or a room count emptied the search. Saying "we
        # have no AC rooms free" while listing the free AC room would be worse
        # than the message this replaces, so leave the existing card alone.
        return None

    wanted = "Non-AC" if any("non" in f.lower() for f in hard) else "AC"
    from app.utils.helpers import format_display_date
    stay = f"{format_display_date(check_in)} to {format_display_date(check_out)}"

    # `clean_room_type_name` maps "STANDARD NON AC ROOM" to "Standard Room" — it
    # drops the class marker this message is about. When the alternative is the
    # opposite class the guest did NOT ask for, put the marker back, so an AC
    # guest offered a non-AC room is told it is one.
    names: list[str] = []
    classes: list[str] = []
    for room in open_rooms or []:
        clean = clean_room_type_name(room.name)
        if not clean:
            continue
        cls = "Non-AC" if room_matches_feature(room, "Non-AC") else "AC"
        if cls not in classes:
            classes.append(cls)
        label = clean if cls == wanted else f"{clean} ({cls})"
        if label not in names:
            names.append(label)
    if not names:
        return None

    # Try building priced options with ready combinations & buttons for the alternative class
    adults = int(state.get("effective_adults") if state.get("effective_adults") is not None else (state.get("adults") or 2))
    children = int(state.get("effective_children") if state.get("effective_children") is not None else (state.get("children") or 0))
    num_rooms = state.get("num_rooms")
    budget = state.get("budget")
    meal_pref = state.get("meal_plan_preference")
    from app.services.booking_engine import calculate_num_nights
    try:
        num_nights = calculate_num_nights(check_in, check_out) if check_in and check_out else 1
    except Exception:
        num_nights = 1

    alt_class = classes[0] if classes else ("AC" if wanted == "Non-AC" else "Non-AC")
    alt_features = [f for f in (effective_features or []) if not is_hard_inventory_feature(f)]
    if alt_class not in alt_features:
        alt_features.append(alt_class)

    allocations = []
    from app.services.room_service import find_suitable_rooms, enrich_options_with_stay_totals, build_room_options_card
    try:
        # If requested num_rooms > adults, allocate_rooms returns empty because each room needs at least 1 adult.
        # Relax room count constraint if num_rooms > adults so party gets valid multi-room alternatives.
        req_rooms = num_rooms if (num_rooms and num_rooms <= adults) else None
        allocations, _ = await find_suitable_rooms(
            hotel_id=hotel_id,
            adults=adults,
            children=children,
            num_rooms=req_rooms,
            budget=budget,
            check_in=check_in,
            check_out=check_out,
            meal_plan_preference=meal_pref,
            feature_preferences=alt_features,
            paid_children=int(state.get("paid_child_count") or 0),
            wants_bed=bool(state.get("wants_bed")),
            max_adults_per_room=state.get("max_adults_per_room"),
        )
    except Exception as e:
        logger.warning("ROOM_CLASS_GAP | find_suitable_rooms failed: %s", e)
        allocations = []

    if allocations:
        options_dicts = [opt.model_dump() for opt in allocations]
        try:
            available = await enrich_options_with_stay_totals(
                options_dicts, num_nights, hotel_id, check_in=check_in, check_out=check_out
            )
        except Exception:
            available = options_dicts

        room_card = build_room_options_card(
            available,
            state={
                **state,
                "ac_preference": classes[0] if len(classes) == 1 else "BOTH",
                "assumptions": [alt_class, "Room Only"],
            },
            rooms=open_rooms,
        )
        notice = (
            f"I'm sorry, we don't have *{wanted}* rooms free for {stay}. 😔\n\n"
            f"These alternative *{alt_class}* options are available for the same dates:\n\n"
        )
        room_card["text"] = notice + room_card["text"]
        card_options = room_card["data"].get("options") or []

        return cast(ConversationState, {
            **state,
            "ac_preference": classes[0] if len(classes) == 1 else "BOTH",
            "feature_preferences": [
                f for f in (state.get("feature_preferences") or [])
                if not is_hard_inventory_feature(f)
            ],
            "assumptions": [alt_class, "Room Only"],
            "allocation_options": card_options,
            "room_type_not_available": False,
            "booking_stage": "rooms_shown",
            "room_previews_sent": True,
            "quote_presented": True,
            "next_action": "no_rooms_class_gap",
            "last_prompt_action": "rooms_shown",
            "bot_response": [room_card],
        })

    listing = "\n".join(f"• *{n}*" for n in names[:6])

    from app.prompts import templates
    text = (
        f"I'm sorry, we don't have *{wanted}* rooms free for {stay}. 😔\n\n"
        f"These rooms are available for the same dates:\n{listing}\n\n"
        f"Would you like me to price any of them?"
    )
    fallback_options = []
    for i, room in enumerate((open_rooms or [])[:4]):
        r_name = getattr(room, "name", "")
        r_id = getattr(room, "room_id", getattr(room, "id", str(i + 1)))
        r_price = float(getattr(room, "price", 0.0) or 0.0)
        fallback_options.append({
            "option_id": f"opt_{i + 1}",
            "rooms": [{
                "room_id": str(r_id),
                "room_name": r_name,
                "price_per_night": r_price,
                "quantity": 1,
            }],
            "total_price_per_night": r_price,
            "total_price_for_stay": r_price * num_nights,
        })

    return cast(ConversationState, {
        **state,
        "ac_preference": classes[0] if len(classes) == 1 else "BOTH",
        "feature_preferences": [
            f for f in (state.get("feature_preferences") or [])
            if not is_hard_inventory_feature(f)
        ],
        "assumptions": state.get("assumptions") or [],
        "allocation_options": fallback_options,
        "room_type_not_available": False,
        "booking_stage": "rooms_shown",
        "next_action": "no_rooms_class_gap",
        "bot_response": templates.text_card(text),
    })


def _reply_prepend(state: ConversationState) -> list[dict]:
    """Cards to put in front of this node's own reply.

    `response_generator` honours `bot_response_prepend`; this node does not pass
    through it, so it has to read the same key itself. A dict is accepted
    because two producers write it in two shapes — see `queue_manager`.
    """
    prepend = state.get("bot_response_prepend") or []
    if isinstance(prepend, dict):
        return [prepend]
    return [c for c in prepend if isinstance(c, dict)]


def may_ask_ac_question(
    hotel_cfg: dict,
    has_mixed_ac: bool | None,
    ac_pref: str,
    feature_preferences: list | None,
    state: ConversationState,
) -> bool:
    """Should this turn ask "AC or Non-AC?".

    Pure, so the switch and its guards can be read and tested in one place.

    `ASK_AC_PREFERENCE` is the master switch and is OFF today (Product,
    2026-11-03): at Sample Hotel both answers led to the same rooms, so the
    question cost a turn and bought nothing. While it is off the bot assumes AC,
    says so in the footer, and the guest can still say "non-AC" themselves —
    that arrives as `ac_preference` from their own words and filters the search
    exactly as before, without passing through this gate.

    With it on, the remaining guards apply as they always have: the hotel must
    have a config and not have declared single-class inventory, the guest must
    not have answered already, and no other feature preference may be in play.
    """
    if not get_settings().ASK_AC_PREFERENCE:
        return False
    if not hotel_cfg or has_mixed_ac is False:
        return False
    if ac_pref or feature_preferences or state.get("ac_qualified"):
        return False
    return True


async def search_rooms(state: ConversationState) -> ConversationState:
    """
    Search the normalized room catalog (room_types/rate_plans) for rooms matching the guest requirements.
    Uses multi-room allocation engine.
    """
    # A guest we already know is confirmed is never handed a fresh card. The stage
    # routing above should have sent this turn elsewhere; when it has not, this is
    # the node that would build the card, so this is where it stops. Returns before
    # any search, allocation, or option state is written — a refused search must
    # not leave `allocation_options` behind for the next turn to re-render.
    from app.services.confirmed_guard import confirmed_guest_reply, guest_should_not_be_sold_to

    if guest_should_not_be_sold_to(state):
        logger.info(
            "SEARCH_ROOMS_SKIPPED_CONFIRMED_GUEST | hotel=%s — guest is already confirmed; "
            "no search, no card",
            state.get("hotel_id"),
        )
        return {
            **state,
            "bot_response": confirmed_guest_reply(state),
            "next_action": "confirmed_guest_no_sell",
        }

    hotel_id = state.get("hotel_id", 0)
    adults = int(state.get("effective_adults") if state.get("effective_adults") is not None else (state.get("adults") or 2))
    children = int(state.get("effective_children") if state.get("effective_children") is not None else (state.get("children") or 0))
    # How many of those children the hotel's OWN policy makes chargeable — the
    # band between free_till and paid_till. The validator works it out, because
    # it is the only place holding both the ages and the policy. 0 here is the
    # honest default: a hotel with no policy configured charges no child charge,
    # exactly as it did before this existed.
    paid_children = int(state.get("paid_child_count") or 0)
    num_rooms = state.get("num_rooms")
    room_type_pref = state.get("room_type_preference")
    feature_preferences = state.get("feature_preferences")

    budget = state.get("budget")
    check_in = state.get("check_in")
    check_out = state.get("check_out")
    meal_pref = state.get("meal_plan_preference")

    # Staff inventory/booking overrides (conversation only)
    from app.services.override_resolver import resolve
    staff_room = resolve(state, "inventory", "room_type") or resolve(state, "booking", "room_type")
    if staff_room:
        room_type_pref = staff_room
    if resolve(state, "booking", "num_rooms") is not None:
        num_rooms = resolve(state, "booking", "num_rooms")
    if resolve(state, "booking", "adults") is not None and state.get("effective_adults") is None:
        adults = int(resolve(state, "booking", "adults") or 2)
    if resolve(state, "booking", "children") is not None and state.get("effective_children") is None:
        children = int(resolve(state, "booking", "children") or 0)

    from app.services.room_alias_resolver import aliases_from_hotel_config

    hotel_cfg = state.get("hotel_config") or {}
    if hasattr(hotel_cfg, "model_dump"):
        hotel_cfg = hotel_cfg.model_dump()
    elif hasattr(hotel_cfg, "dict"):
        hotel_cfg = hotel_cfg.dict()
    elif not isinstance(hotel_cfg, dict):
        hotel_cfg = {}
    aliases = aliases_from_hotel_config(hotel_cfg)
    threshold = 90.0
    if isinstance(hotel_cfg, dict) and hotel_cfg.get("room_match_threshold") is not None:
        try:
            threshold = float(hotel_cfg["room_match_threshold"])
        except (TypeError, ValueError):
            pass

    # S3 Qualifier (Spec §5 / Ticket H4):
    # Only ask when property has mixed inventory AND both AC & Non-AC are available
    #
    # `ac_qualified` means the GUEST HAS ANSWERED — not "we have asked". This
    # gate is what re-asks, and the block below deliberately does not set the
    # flag when it asks. Setting it there was how an unread answer became a
    # silent AC: the question was asked once, the flag went up, and every later
    # turn walked past this gate and fell to the "default assumption is AC"
    # branch below without a word to the guest.
    #
    # The flag means OFF ONLY WHEN IT SAYS OFF. An absent key used to mean "no"
    # here, while `hotel_config` and `validator` read the same absence as "yes"
    # — three answers to one question, and which one a hotel got depended on
    # which module asked. `True` and absent now do the same thing: fall through
    # to the two probes below, which ask the inventory itself and are the real
    # answer to "is there a choice worth offering here". A hotel that sets the
    # key false is switching the question off and is still obeyed.
    #
    # Two cases are NOT the same, and the difference is load. An absent key
    # inside a config is a hotel that never set it — unknown, so the probes
    # decide. NO CONFIG AT ALL is a caller that told us nothing about the
    # hotel, and there we do not ask: the probes are two database round trips,
    # and spending them on a state that carries no hotel config is paying for
    # a guess. In production `context_loader` always stores the resolved
    # config, so the second case is tests and hand-built states — but it is the
    # difference between reading the absence and inventing an answer.
    biz_feats = (hotel_cfg.get("business_config") or {}).get("features", {}) if isinstance(hotel_cfg, dict) else {}
    has_mixed_ac = biz_feats.get("has_mixed_ac_inventory")
    ac_pref = (state.get("ac_preference") or "").upper()
    may_offer_ac_choice = may_ask_ac_question(hotel_cfg, has_mixed_ac, ac_pref, feature_preferences, state)

    if may_offer_ac_choice:
        ac_alloc, _ = await find_suitable_rooms(
            hotel_id=hotel_id, adults=adults, children=children, num_rooms=num_rooms,
            budget=budget, check_in=check_in, check_out=check_out, meal_plan_preference=meal_pref,
            room_aliases=aliases, room_match_threshold=threshold, feature_preferences=["AC"],
            wants_bed=bool(state.get("wants_bed")),
        )
        non_ac_alloc, _ = await find_suitable_rooms(
            hotel_id=hotel_id, adults=adults, children=children, num_rooms=num_rooms,
            budget=budget, check_in=check_in, check_out=check_out, meal_plan_preference=meal_pref,
            room_aliases=aliases, room_match_threshold=threshold, feature_preferences=["Non-AC"],
            wants_bed=bool(state.get("wants_bed")),
        )
        # A non-AC option only makes the question worth asking if it really is a
        # non-AC room. `find_suitable_rooms(feature_preferences=["Non-AC"])`
        # matches on the room's NAME, and a name outlives the configuration: at
        # Sample Hotel both "STANDARD ROOM (NON-AC)" and "STANDARD TRIPPLE BED
        # (NON-AC)" now carry Air Conditioning on their amenity rows, so the
        # second answer would quote the first answer's rooms. See
        # `allocation_engine.ac_choice_is_real`.
        _real_choice = True
        if ac_alloc and non_ac_alloc and get_settings().AC_QUESTION_HONOURS_ROOM_CONFIG:
            from app.services.allocation_engine import ac_choice_is_real

            try:
                _catalogue = await get_all_rooms(hotel_id)
            except Exception as e:
                # The search itself just read this catalogue, so a failure here
                # is a transient DB fault. Asking is the pre-existing behaviour
                # and the recoverable answer, so keep it rather than 500.
                logger.warning("AC_QUESTION_CATALOGUE_READ_FAILED | hotel=%s | %s", hotel_id, e)
                _catalogue = []
            _real_choice = ac_choice_is_real(non_ac_alloc, _catalogue)
            if not _real_choice:
                logger.warning(
                    "AC_QUESTION_SKIPPED_NO_NON_AC_INVENTORY | hotel=%s | the "
                    "non-AC-named rooms are configured as AC; not asking",
                    hotel_id,
                )

        if ac_alloc and non_ac_alloc and _real_choice:
            from app.prompts import templates
            ci_str = check_in or ""
            co_str = check_out or ""
            pax_str = f"{adults} adults"
            if children and children > 0:
                pax_str += f", {children} children"
            rooms_str = f"{num_rooms or 1} room"
            dates_str = f"{ci_str} to {co_str}" if ci_str else ""
            lang = state.get("preferred_language") or "English"
            resp = templates.ask_ac_preference(dates_str, pax_str, rooms_str, lang=lang)
            
            child_notes = state.get("child_age_notes") or []
            if child_notes:
                notes_str = "\n".join(child_notes)
                if resp and resp[0].get("text"):
                    resp[0]["text"] = f"_{notes_str}_\n\n{resp[0]['text']}"

            # Note what is NOT set: `ac_qualified`. Asking is not answering,
            # and the gate above reads that flag as "the guest has told us".
            # Left unset, an answer we cannot read brings the guest back here
            # and the question is put again; setting it here closed the door
            # on the question and opened the silent AC default instead.
            return {
                **state,
                "booking_stage": "collecting",
                "next_action": "need_ac_preference",
                "last_prompt_action": "need_ac_preference",
                "bot_response": resp,
            }

    # Determine meal assumption from guest preference
    meal_pref = (state.get("meal_plan_preference") or "").upper()
    if meal_pref in ("CP", "BREAKFAST"):
        meal_assumption = "Breakfast Included"
    elif meal_pref in ("MAP", "HALF_BOARD", "DINNER"):
        meal_assumption = "Breakfast & Dinner"
    else:
        meal_assumption = "Room Only"
        # ...and the allocation has to be priced on that same basis. Left as
        # None, `get_preferred_rate_plan` falls back to the cheapest active
        # plan — which is not always Room Only. Hotel 111111's own calendar has
        # Deluxe King room-only at Rs 3,000 against Rs 2,950 with breakfast, so
        # the cheapest plan is CP and the card sold breakfast under a
        # "Room only" footer. Naming EP here is what makes the assumption line
        # true; a hotel with no EP plan still falls back to its cheapest.
        meal_pref = "EP"

    # Determine AC preference vs default assumption
    if state.get("room_requirements"):
        # A split's lines each name an AC class when the model managed to fill
        # one in, and that is the most specific signal. When none of them did —
        # the common case, because the deterministic reader never sets per-line
        # `ac` at all — the guest's own message-level preference still stands.
        #
        # Reading only the lines was the 2026-09-26 defect. A guest wrote "Non
        # Ac room … 2 room chahiye 1- 2 bed 2- 4 bed"; `understand` recorded
        # `ac_preference = NON_AC` and `feature_preferences = ["Non-AC"]`, and
        # this branch discarded both. `effective_features = []` meant the
        # allocator never filtered, so the card quoted AC rooms while the
        # confirmation directly above it said "room features to **Non-AC**" —
        # and the empty assumption list made the footer re-stamp "Assuming: AC"
        # from the template default. Two halves of one reply contradicted each
        # other and neither matched what the guest asked for.
        split_ac = None
        for _item in state.get("room_requirements") or []:
            _value = (
                _item.get("ac") if isinstance(_item, dict) else getattr(_item, "ac", None)
            )
            if _value in ("AC", "NON_AC"):
                split_ac = _value
                break

        if split_ac is None:
            # `NO_PREFERENCE` is not a class — it means "show me both" — and
            # resolves to None so the card stays unfiltered, which is what the
            # guest asked for. Only an explicit AC or Non-AC reaches here.
            if ac_pref == "AC":
                split_ac = "AC"
            elif ac_pref == "NON_AC":
                split_ac = "NON_AC"
            else:
                # `feature_preferences` holds display labels — "Non-AC", "AC" —
                # and carries the class when `ac_preference` itself is unset.
                # Non-AC is tested first: "non ac" contains "ac", so the reverse
                # order would read a Non-AC request as an AC one.
                _feats = [str(p).strip().lower() for p in (feature_preferences or [])]
                if any("non" in f and "ac" in f for f in _feats):
                    split_ac = "NON_AC"
                elif any(f == "ac" for f in _feats):
                    split_ac = "AC"

        if split_ac:
            # Carry the class into allocation as a hard filter, so the room the
            # guest asked for is the room that gets priced. Without this the
            # footer and the card can disagree, which is worse than either one
            # being wrong on its own.
            _ac_label = "AC" if split_ac == "AC" else "Non-AC"
            effective_features = [_ac_label]
            assumptions = [_ac_label, meal_assumption]
        elif ac_pref in ("NO_PREFERENCE", "BOTH", "SHOW_BOTH") or any(
            p.lower() in ("no_preference", "both", "show both", "dono", "all")
            for p in (feature_preferences or [])
        ):
            effective_features = []
            assumptions = [meal_assumption]
        else:
            # Default assumption for the hotel is AC when guest did not qualify or request Non-AC
            effective_features = ["AC"]
            assumptions = ["AC", meal_assumption]
    else:
        effective_features = list(feature_preferences or [])
        if ac_pref == "NON_AC" or (not ac_pref and any("non" in p.lower() for p in effective_features)):
            if "Non-AC" not in effective_features:
                effective_features.append("Non-AC")
            assumptions = ["Non-AC", meal_assumption]
        elif ac_pref == "AC" or (not ac_pref and any(p.upper() == "AC" for p in effective_features)):
            effective_features = [f for f in effective_features if "non" not in f.lower()]
            if "AC" not in effective_features:
                effective_features.append("AC")
            assumptions = ["AC", meal_assumption]
        elif ac_pref in ("NO_PREFERENCE", "BOTH", "SHOW_BOTH") or any(p.lower() in ("no_preference", "both", "show both", "dono", "all") for p in effective_features):
            # User explicitly wants to see BOTH AC and Non-AC rooms — do not filter
            effective_features = [f for f in effective_features if f not in ("AC", "Non-AC", "NO_PREFERENCE", "BOTH", "SHOW_BOTH")]
            assumptions = [meal_assumption]
        else:
            # Default assumption is AC when user has not qualified AC preference
            if not any("non" in p.lower() for p in effective_features) and "AC" not in effective_features:
                effective_features.append("AC")
            assumptions = ["AC", meal_assumption]

    try:
        import asyncio
        allocations, fell_back = await asyncio.wait_for(
            find_suitable_rooms(
                hotel_id=hotel_id,
                adults=adults,
                children=children,
                num_rooms=num_rooms,
                room_type_preference=room_type_pref,
                budget=budget,
                check_in=check_in,
                check_out=check_out,
                meal_plan_preference=meal_pref,
                room_aliases=aliases,
                room_match_threshold=threshold,
                feature_preferences=effective_features,
                room_requirements=state.get("room_requirements"),
                occupancy_preference=state.get("occupancy_preference"),
                grouping_mode=state.get("grouping_mode"),
                paid_children=paid_children,
                wants_bed=bool(state.get("wants_bed")),
                max_adults_per_room=state.get("max_adults_per_room"),
            ),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        from app.prompts import templates
        return cast(ConversationState, {
            **state,
            "escalation_required": True,
            "escalation_issue_type": "NORMAL",
            "escalation_reason": "availability_gap_timeout",
            "booking_stage": "collecting",
            "next_action": "availability_timeout",
            "bot_response": templates.text_card("Main hotel se confirm karke 10 min mein batata hoon."),
        })

    # Fallback if no rooms matched AC assumption
    if not allocations and "AC" in effective_features and not ac_pref:
        effective_features = [f for f in effective_features if f != "AC"]
        allocations, fell_back = await find_suitable_rooms(
            hotel_id=hotel_id,
            adults=adults,
            children=children,
            num_rooms=num_rooms,
            room_type_preference=room_type_pref,
            budget=budget,
            check_in=check_in,
            check_out=check_out,
            meal_plan_preference=meal_pref,
            room_aliases=aliases,
            room_match_threshold=threshold,
            feature_preferences=effective_features,
            room_requirements=state.get("room_requirements"),
            occupancy_preference=state.get("occupancy_preference"),
            grouping_mode=state.get("grouping_mode"),
            paid_children=paid_children,
            wants_bed=bool(state.get("wants_bed")),
            max_adults_per_room=state.get("max_adults_per_room"),
        )
        assumptions = ["Non-AC", "Room Only"]

    # ── STOP-SELL ENFORCEMENT & INSISTENT GUEST ESCALATION ──
    if fell_back or not allocations:
        from app.services.stop_sell_guard import check_and_handle_stop_sell_escalation
        stop_sell_res = await check_and_handle_stop_sell_escalation(
            state=state,
            preferred_room=room_type_pref,
            check_in=check_in,
            check_out=check_out,
        )
        if stop_sell_res:
            return cast(ConversationState, {
                **state,
                **stop_sell_res,
            })

    if not allocations:
        # Check if the search yielded no options because party requested an excessive room count.
        # Only a handoff when the hotel does *not* sell group bookings itself —
        # otherwise the empty result is a real "no rooms" and says so.
        if (
            not get_settings().GROUP_BOOKING_SELF_SERVE
            and not state.get("group_handoff_sent")
            and ((num_rooms is not None and num_rooms >= 5) or adults >= 15)
        ):
            from app.prompts import templates
            return cast(ConversationState, {
                **state,
                "assumptions": assumptions,
                "allocation_options": [],
                "room_type_not_available": fell_back,
                "booking_stage": "collecting",
                "next_action": "group_booking_lead",
                "group_handoff_sent": True,
                "bot_response": templates.button_card(
                    "For group bookings of 5+ rooms or 10+ guests, our reservations team will assist you directly with special group rates and dedicated coordination.\n\nPlease share your contact details and group requirements, or connect with our front desk below.",
                    buttons=[{"id": "menu_reception", "title": "Talk to reception"}],
                ),
            })

        # ── NAME THE HALF THAT FAILED ──
        # Reached only when nothing can be allocated, so it cannot change a turn
        # that already succeeds. Group bookings are handled above because a big
        # party needs the reservations desk whatever its AC class.
        if get_settings().ROOM_CLASS_GAP_HINT_ENABLED:
            gap = await _room_class_gap_response(
                state, hotel_id, check_in, check_out, effective_features
            )
            if gap is not None:
                return gap

        # ── MULTI-BOOKING QUEUE CONTINUATION (FAILURE) ──
        from app.utils.queue_manager import advance_booking_queue
        q_updates = advance_booking_queue(state, "failed")
        bot_response = []
        if q_updates.get("queue_advanced"):
            bot_response = q_updates.pop("bot_response_prepend", [])
            
        return cast(ConversationState, {
            **state,
            "assumptions": assumptions,
            "allocation_options": [],
            "room_type_not_available": fell_back,
            "booking_stage": "collecting",
            "next_action": "no_rooms",
            "bot_response": bot_response,
            **q_updates
        })

    from app.services.room_service import enrich_options_with_stay_totals
    from app.services.booking_engine import calculate_num_nights
    try:
        num_nights = calculate_num_nights(check_in, check_out) if check_in and check_out else 1
    except Exception:
        num_nights = 1

    # Convert AllocationOption objects to dictionaries for state persistence
    options_dicts = [opt.model_dump() for opt in allocations]
    available = await enrich_options_with_stay_totals(
        options_dicts, num_nights, hotel_id, check_in=check_in, check_out=check_out
    )

    from app.prompts import templates

    # ── AC / NON-AC / BOTH: PRICE (client, 2026-09-19; revised 2026-09-19) ──
    #
    # There used to be a room-description card here, sent when the guest
    # answered the AC question and built from an LLM call over the filtered
    # catalogue. The client has dropped it: the priced card below answers the
    # same question in three lines, and the description contradicted it on
    # price (static catalogue rates against the card's calendar rates), on
    # capacity (`_effective_max_occupancy` against the booking system's
    # `max_occupancy`) and on which rooms were in the guest's AC class
    # (`recommendation_engine._matches_feature` against
    # `allocation_engine.room_matches_feature`, since unified).
    #
    # `ac_details_pending` and `build_ac_details_message` went with it.

    # If the user explicitly selected a custom composite configuration, auto-select those specific rooms.
    # Otherwise, ALWAYS present room options to the guest so they can choose.
    #
    # A guest-stated per-room split is the second case, and it is here rather
    # than in `booking_stage` for a reason worth keeping: the stage is NOT a
    # reliable signal for it. `understand` sets `custom_rooms_selected`, and
    # then its own hard-reset block assigns `booking_stage = "collecting"`
    # afterwards whenever the change touched a capacity field — which a split
    # always does, because accepting one writes `num_rooms`. The requirements
    # survive that reset (`understand.py`, the wipe is guarded on
    # `extracted.room_requirements is None`); the stage does not. So the
    # condition reads the requirements, which are the thing that is actually
    # true, instead of a stage that was true one node ago.
    #
    # This is the whole fix and not a detail: the ordinary card cannot deliver a
    # split. Its rows are re-derived cheapest-first and any packing it cannot
    # reproduce is dropped, so an explicit set reaches `card_rows` and returns
    # **zero** rows — the guest gets today's generic card back, silently. The
    # only path that quotes an explicit set verbatim is this one.
    #
    # `available` is a guard, not decoration: `best_option = available[0]` below
    # is unguarded, and the split condition is what newly makes an empty list
    # reachable here.
    is_split = is_an_explicit_split(state.get("room_requirements"))
    if state.get("room_requirements") and not fell_back and available and (
        is_split
        or state.get("booking_stage") in ("custom_rooms_selected", "composite_selected")
    ):
        # Logged because this failure mode is silent by nature: when routing is
        # missed the guest gets a plausible card, and nothing anywhere says a
        # split was read and then not delivered.
        logger.info(
            "ROOM_ROUTING | sess=%s | auto-select | split=%s stage=%s rooms=%d",
            state.get("session_id"),
            is_split,
            state.get("booking_stage"),
            len(available[0].get("rooms", [])),
        )
        best_option = available[0]
        rooms = best_option.get("rooms", [])
        total_price = best_option.get("total_price_for_stay", best_option.get("total_price_per_night", 0) * num_nights)
        
        room_names = [r.get("room_name") for r in rooms]
        if len(room_names) == 1:
            name_str = f"**{room_names[0]}**"
        else:
            counts = {}
            for name in room_names:
                counts[name] = counts.get(name, 0) + 1
            summary = ", ".join(f"{cnt}x {n}" for n, cnt in counts.items())
            name_str = f"**{summary}**"
            
        msg = f"I have selected the {name_str} for you."
        
        return {
            **state,
            "assumptions": assumptions,
            "allocation_options": available[:2],
            "selected_rooms": rooms,
            "total_price": total_price,
            "room_type_not_available": fell_back,
            "booking_stage": "rooms_selected",
            "bot_response": [*templates.text_card(msg)],
            "next_action": "show_quote",
        }

    # ── S5.5 COMPOSITION TURN (FIT QUESTION FOR A LARGE PARTY) ──
    #
    # Trigger: five or more ADULTS, with no upper bound (client, 2026-09-19).
    #
    # It used to be `5 <= total_pax < 8` AND more than three viable
    # arrangements. Both bounds are gone. The ceiling silently stopped asking
    # for exactly the parties that need it most — the ones where "fewer rooms
    # with beds" and "more rooms" diverge by thousands a night — and
    # "more than three arrangements" measured the four composition modes, which
    # no longer exist.
    #
    # Adults, not headcount, and that is a deliberate line rather than the
    # leftover it looks like. Five adults is where the arrangement space
    # explodes — one triple, two doubles, a family room plus a single — and the
    # guest needs the question to narrow it. Children do not add arrangements
    # in the same way: they are assumed to settle in with the adults, so 4
    # adults + 3 children is a room-count question the search answers on its
    # own. Kept as `> 4` to say five, since that is the number the client named.
    #
    # `adults` here is `effective_adults`, NOT the raw count. That is the half
    # that fixes the reported case: 4 adults + one 16-year-old at Sample Hotel is
    # five adults once the hotel's own child policy is applied, so it asks. The
    # gate never had to change for that — the number reaching it did.
    #
    # A guest who has named a room TYPE or a budget has answered it, and asking
    # again is the thing that reads as a bot not listening. The surviving
    # `viable_count > 1` is the old intent kept intact: the question is only
    # worth asking when the guest actually has a choice, and with one
    # arrangement both answers lead to the same card.
    # `num_rooms` is deliberately NOT in this list any more. It used to be, on
    # the theory that a guest who named a room count had answered the question.
    # They have not: `room_service.find_suitable_rooms` throws the count away
    # for every party of five or more before the search runs, so the count is
    # not honoured anywhere downstream — it only ever suppressed this question.
    # A guest who says "5 people, 1 TBR & 1 DBR" was therefore neither asked how
    # to split nor given the two rooms they named; they got a generic
    # combinations card. Asking is the honest half of that pair.
    has_skip_conditions = bool(
        room_type_pref
        or budget
        or state.get("composition_asked")
        or state.get("occupancy_preference")
        or state.get("grouping_mode")
    )
    if int(adults or 0) > 4 and not has_skip_conditions:
        viable_count = len(available)
        if viable_count > 1:
            from app.utils.helpers import format_quote_date_range
            dates_str = format_quote_date_range(check_in, check_out) if (check_in and check_out) else ""
            night_word = "night" if num_nights == 1 else "nights"
            nights_str = f"{num_nights} {night_word}"
            adult_word = "adult" if adults == 1 else "adults"
            if children and children > 0:
                child_word = "child" if children == 1 else "children"
                pax_str = f"{adults} {adult_word} + {children} {child_word}"
            else:
                pax_str = f"{adults} {adult_word}"

            lang = state.get("preferred_language") or "English"
            comp_response = templates.ask_composition_preference(dates_str, nights_str, pax_str, lang=lang)
            child_notes = state.get("child_age_notes") or []
            if child_notes:
                notes_str = "\n".join(child_notes)
                if comp_response and comp_response[0].get("text"):
                    comp_response[0]["text"] = f"_{notes_str}_\n\n{comp_response[0]['text']}"

            return {
                **state,
                "assumptions": assumptions,
                "composition_asked": True,
                "composition_options_count": viable_count,
                "booking_stage": "collecting",
                "next_action": "need_composition_preference",
                "bot_response": [*comp_response],
            }

    total_guests = int(adults or 2) + int(children or 0)
    is_multi_room = total_guests >= 5

    from app.services.room_service import build_room_options_card

    state_with_assumptions = {**state, "assumptions": assumptions}
    all_rooms = await get_all_rooms(hotel_id)

    if not is_multi_room:
        from app.services.recommendation_engine import rank_and_select_options
        available, _ = rank_and_select_options(available, state, all_rooms)

    # No truncation. The card's three lines are chosen by the packer, not by
    # position, so `available[:4]` only ever hid rooms the packer wanted: the
    # cheapest set for a party of seven sits at position 13 of 80, and position
    # 13 of a four-item list does not exist. Ranking still runs — it decides
    # which options lead when the packer has nothing to say.

    # The ROOMS section has to quote the prices the combinations beneath it were
    # priced on, and those went through `find_suitable_rooms` →
    # `get_available_rooms`, which overlays the rate calendar. `get_all_rooms`
    # returns the static catalogue and knows nothing about the calendar, so the
    # two halves of one card disagreed: Deluxe King was listed at its static Rs
    # 2,550 while every combination charged the calendar's Rs 3,000.
    #
    # Falls back to the static catalogue, because a card whose list is slightly
    # stale beats a card with no list at all.
    from app.services import inventory_service

    card_rooms = (
        await inventory_service.get_available_rooms(hotel_id, check_in, check_out)
    ) or all_rooms

    # One renderer for every party size. The old fork — the S4 formatter below
    # five guests, the structured multi-room card at and above — is what let the
    # two drift: they disagreed about the capacity number, about whether the bed
    # was a separate line, and about which rooms they showed at all.
    room_card = build_room_options_card(
        available, state=state_with_assumptions, rooms=card_rooms
    )

    # The card's own lines, in the card's own order. A reply of "1" is resolved
    # by position against `allocation_options` a turn later, so this has to be
    # the list the guest is looking at rather than the allocator's full output.
    card_options = room_card["data"].get("options") or []

    # Attach room preview media cards ONLY on the first time room suggestions are sent,
    # or if the guest explicitly requested photos (wants_photos=True).
    # On subsequent customization / re-searches, omit duplicate media cards.
    already_sent_previews = bool(state.get("room_previews_sent") or state.get("quote_presented"))
    wants_explicit_photos = bool(state.get("wants_photos"))

    preview_cards: list[dict] = []
    if not already_sent_previews or wants_explicit_photos:
        preview_cards = _room_preview_media_cards(card_options, state)

    return {
        **state,
        "assumptions": assumptions,
        "allocation_options": card_options,
        "room_type_not_available": fell_back,
        "booking_stage": "rooms_shown",
        "room_previews_sent": True,
        "quote_presented": True,
        "wants_photos": False,
        # Translated here because this node returns to the guest directly and
        # never passes through response_generator's translation choke point.
        #
        # `bot_response_prepend` is carried through by hand for the same reason:
        # this node returns straight to `save_state`, so anything queued for the
        # reply (the multi-booking acknowledgement `understand` writes when one
        # message holds several stays) would never be rendered. Cleared here so
        # it is said once, on the turn the queue was created.
        "bot_response": _reply_prepend(state)
        + await translate_cards_if_needed([room_card, *preview_cards], state),
        "bot_response_prepend": [],
        "next_action": "rooms_shown",
        "last_prompt_action": "rooms_shown",
    }
