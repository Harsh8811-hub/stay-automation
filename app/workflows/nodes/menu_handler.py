"""
Menu Handler Node — handles all menu-driven intents:

  SHOW_PRICES        → fetch room_types/rate_plans and format a price list
  SHOW_LOCATION      → fetch hotel_details and format address/directions
  SHOW_AMENITIES     → fetch hotel_amenities and request tagged media delivery
  CHECK_EXISTING_BOOKING → lookup confirmed booking from DB (phone + hotel), fallback to draft
  SOMETHING_ELSE     → friendly redirect to open-ended help
"""

import logging

from app.prompts import templates
from app.repositories import hotel_repository, room_repository
from app.services.room_service import _friendly_rate_plan_name
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

def _append_booking_reminder_to_cards(cards: list[dict], state: dict) -> list[dict]:
    from app.utils.booking_resume import get_booking_resume_nudge
    text = get_booking_resume_nudge(state)
    if text:
        cards = list(cards) + [{"type": "text", "text": text}]
    return cards


def _room_name_from_booking(doc: dict) -> str:
    rooms = doc.get("rooms") or doc.get("selected_rooms") or []
    if not rooms:
        return ""
    first = rooms[0] if isinstance(rooms[0], dict) else {}
    return first.get("room_name") or first.get("name") or ""


def _summary_from_booking_doc(doc: dict, *, source: str = "confirmed") -> list[dict]:
    return templates.existing_booking_summary(
        booking_id=doc.get("booking_id") or "",
        check_in=doc.get("check_in") or "",
        check_out=doc.get("check_out") or "",
        adults=int(doc.get("adults") or 0),
        children=int(doc.get("children") or 0),
        num_rooms=int(doc.get("num_rooms") or len(doc.get("rooms") or []) or 1),
        room_name=_room_name_from_booking(doc),
        total_price=doc.get("total_price"),
        payment_status=doc.get("payment_status"),
        booking_status=doc.get("booking_status"),
        source=source,
    )


def _date_phrase_in(message: str) -> str:
    """The guest's own date words, if their message held any.

    Used only to show the guest they were heard. It must never narrow or
    widen the lookup: a booking found by date alone is somebody else's.
    """
    from app.utils.date_masking import date_spans

    spans = date_spans(message)
    # "Check in 3rd oct Check out 5th oct" -> "3rd oct to 5th oct", not
    # "3rd oct 5th oct", which reads like one garbled date.
    phrases = [message[start:end].strip() for start, end in spans[:2]]
    return " to ".join(phrases).strip()


def _has_draft_progress(state: dict) -> bool:
    stage = state.get("booking_stage", "new")
    return bool(
        state.get("check_in")
        or state.get("check_out")
        or state.get("selected_rooms")
        or state.get("allocation_options")
        or state.get("total_price")
        or stage in ("collecting", "rooms_shown", "quote_shown", "awaiting_payment", "ready_for_payment")
    )


async def _resolve_existing_booking_response(state: ConversationState) -> list[dict]:
    """
    Production lookup order:
    1) Session booking_id → DB
    2) Latest active booking by phone + hotel
    3) Latest booking including cancelled (\"what was my last booking?\")
    4) In-progress draft in this chat
    5) Not found
    """
    from app.repositories import booking_repository

    hotel_id = state.get("hotel_id") or 0
    phone = state.get("customer_identifier") or state.get("guest_phone") or ""
    booking_id = (state.get("booking_id") or "").strip()

    # 1) Explicit session booking_id
    if booking_id:
        try:
            doc = await booking_repository.get_booking(booking_id)
        except Exception as e:
            logger.error(f"get_booking failed for {booking_id}: {e}")
            doc = None
        if doc:
            # Authorization: phone must match when both present
            doc_phone = "".join(ch for ch in str(doc.get("customer_phone") or "") if ch.isdigit())
            sess_phone = "".join(ch for ch in str(phone) if ch.isdigit())
            hotel_ok = doc.get("hotel_id") in (None, hotel_id, str(hotel_id))
            phone_ok = (
                not doc_phone
                or not sess_phone
                or doc_phone.endswith(sess_phone[-10:])
                or sess_phone.endswith(doc_phone[-10:] if len(doc_phone) >= 10 else doc_phone)
            )
            if hotel_ok and phone_ok:
                return _summary_from_booking_doc(doc)

    # 2) Confirmed (then other active) booking by hotel + phone
    doc = None
    if phone and hotel_id:
        try:
            doc = await booking_repository.get_latest_booking(
                hotel_id=hotel_id,
                customer_phone=phone,
                booking_status="confirmed",
            )
            if not doc:
                doc = await booking_repository.get_latest_booking(
                    hotel_id=hotel_id,
                    customer_phone=phone,
                    booking_status=booking_repository.ACTIVE_BOOKING_STATUSES,
                )
        except Exception as e:
            logger.error(f"get_latest_booking failed: {e}")
            doc = None
    if doc:
        return _summary_from_booking_doc(doc)

    # 3) Last booking at this hotel including cancelled
    if phone and hotel_id:
        try:
            doc = await booking_repository.get_latest_booking(
                hotel_id=hotel_id,
                customer_phone=phone,
                booking_status=None,
            )
        except Exception as e:
            logger.error(f"get_latest_booking (any status) failed: {e}")
            doc = None
    if doc:
        return _summary_from_booking_doc(doc)

    # 4) Guest claims they already booked (e.g., via OTA / MMT / Agoda or front desk)
    user_msg = (state.get("user_message") or "").lower()
    is_claiming_existing = bool(state.get("claimed_existing_booking")) or any(
        phrase in user_msg
        for phrase in (
            "already booked", "already have a booking", "already book",
            "pehle se book", "pehle se booking", "booking ho chuki",
            "book ho gaya", "booking hai meri", "humari booking hai",
            "kitni baar btaya", "kitni bar bataya", "already done"
        )
    )
    if is_claiming_existing:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Guest Claims Existing Booking ({phone})",
                description=f"Guest states '{state.get('user_message')}'. No direct DB record found for phone {phone}. Check PMS / OTA portals (MMT, Agoda, Booking.com) for their reservation.",
                task_type="FRONT_DESK",
                subtype="CHECK_IN_INQUIRY",
                department="FRONT_DESK",
                priority="HIGH",
                customer_phone=phone,
                metadata={
                    "guest_message": state.get("user_message"),
                    "claimed_existing": True,
                },
            )
        except Exception as e:
            logger.warning(f"Error raising existing booking lookup task: {e}")

        is_hi = str(state.get("preferred_language") or "").lower() in ("hinglish", "hindi", "hi")
        if is_hi:
            return templates.text_card(
                "Namaste! 🙏 Agar aapne direct ya kisi travel portal (jaise MakeMyTrip, Agoda, Goibibo, Booking.com) se booking ki hai, to kripya apna *Booking ID* ya reservation par diya gaya *Guest Name* share kar dijiye.\n\n"
                "Main turant front desk se aapki booking details check kar leta hoon. 🏨"
            )
        return templates.text_card(
            "Namaste! 🙏 If you have a confirmed booking (direct or via portals like MakeMyTrip, Agoda, Booking.com), please share your *Booking ID* or the *Guest Name* on the reservation.\n\n"
            "I will verify your details with our front desk right away. 🏨"
        )

    # 5) Draft in this session (not yet confirmed in DB)
    if _has_draft_progress(state) and state.get("check_in") and state.get("check_out"):
        selected = state.get("selected_rooms") or []
        room_name = selected[0].get("room_name", "") if selected else ""
        cards = templates.existing_booking_summary(
            booking_id=state.get("booking_id") or "",
            check_in=state.get("check_in") or "",
            check_out=state.get("check_out") or "",
            adults=int(state.get("adults") or 0),
            children=int(state.get("children") or 0),
            num_rooms=int(state.get("num_rooms") or 1),
            room_name=room_name,
            total_price=state.get("total_price"),
            payment_status=state.get("payment_status"),
            booking_status=state.get("booking_stage"),
            source="in_progress",
        )
        return cards + templates.draft_booking_summary_note()

    if _has_draft_progress(state):
        return templates.draft_booking_summary_note()

    lang = state.get("preferred_language") or "English"
    # The guest may be answering our own question with a date ("October 20").
    # Repeating the lookup prompt verbatim leaves them no way forward, so say
    # which date we could not match and ask for the one thing that works.
    dated = _date_phrase_in(str(state.get("user_message") or ""))
    if dated:
        return templates.existing_booking_not_found_dated(dated, lang=lang)
    return templates.existing_booking_not_found(lang=lang)


async def handle_menu(state: ConversationState) -> ConversationState:
    intent = state.get("intent", "UNKNOWN")
    hotel_id = state.get("hotel_id", 0)

    # ── Option 3: Room prices & availability ─────────────────────
    if intent == "SHOW_PRICES":
        # Price from the calendar, not the static rate_plans row. `get_all_rooms`
        # returns `rate_plans.base_price` verbatim — a field nothing keeps in sync
        # with `daily_rates` — so this list quoted 2550 for a room the calendar
        # prices at 3000. With no dates it prices from today; it never filters
        # availability, so the same rooms appear as before.
        from app.services import inventory_service

        check_in = state.get("check_in")
        check_out = state.get("check_out")
        rooms = await inventory_service.get_available_rooms(hotel_id, check_in, check_out)
        if not rooms:
            response = templates.text_card("I'm sorry, I couldn't fetch room pricing right now. Please contact the front desk!")
        else:
            lines: list[str] = []
            keycaps = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            for i, room in enumerate(rooms, 1):
                num_icon = keycaps[i-1] if i <= len(keycaps) else f"*{i}*"
                lines.append(f"{num_icon} *{room.name.upper()}*")
                lines.append(
                    f"👥 {room.occupancy.base_adults} Adults, {room.occupancy.base_children} Children"
                )
                if room.rate_plans:
                    from app.services.pricing_engine import PricingEngine
                    # Only show Room Only / EP price by default
                    ep_plan = next(
                        (rp for rp in room.rate_plans if "room only" in rp.name.lower() or "ep" in rp.name.lower()),
                        room.rate_plans[0]
                    )
                    incl_price = PricingEngine.estimate_inclusive_price(ep_plan.base_price)
                    lines.append(f"💰 ₹{incl_price:,.0f}/night (incl. taxes)")
                lines.append("")  # blank separator

            lines.append("Meal plan upgrades (breakfast/dinner) available — ask me for details.")
            rooms_text = "\n".join(lines).strip()
            response = templates.rooms_pricing_response(rooms_text)
            from app.services.hotel_media_service import MAX_IMAGES_PER_RESPONSE, media_delivery_card

            for room in rooms[:MAX_IMAGES_PER_RESPONSE]:
                if not room.room_id:
                    continue
                caption = room.name
                response.append(media_delivery_card(
                    entity_type="room",
                    entity_id=room.room_id,
                    caption=caption,
                    limit=1,
                ))

        response = _append_booking_reminder_to_cards(response, state)
        return {**state, "bot_response": response, "next_action": "menu_handled"}

    # ── Paid early check-in / late check-out, asked about but not timed ──
    # The guest named a direction and no clock, so there is nothing to price.
    # The hotel's own policy line says what is on sale and asks for the time —
    # the number follows on the quote, from the real bands. A hotel that has not
    # configured this (or has it switched off) gets the availability answer
    # without the "at an extra charge", because inventing a charge for a hotel
    # that never set one is the one thing this must not do.
    if intent == "SHOW_CHECKIN_POLICY":
        from app.config.settings import get_settings
        from app.repositories.early_late_checkout_repository import (
            format_early_late_checkout_policy,
            get_early_late_checkout_config,
        )

        policy = ""
        if get_settings().EARLY_LATE_CHECKOUT_FEE_ENABLED:
            try:
                policy = format_early_late_checkout_policy(
                    await get_early_late_checkout_config(hotel_id)
                )
            except Exception:
                # A config read must never cost the guest an answer they asked for.
                logger.warning("early/late policy lookup failed for hotel %s", hotel_id, exc_info=True)
                policy = ""

        if not policy:
            policy = (
                "Early check-in and late check-out are subject to availability and "
                "must be confirmed by the front desk. Share your arrival or departure "
                "time and I'll check for you."
            )

        # This is the first thing the bot says about *times*, and an early
        # check-in charge means nothing without the standard check-in it is early
        # against — so name both, before the policy. Read-only, never raises, and
        # omitted entirely when the hotel has not configured them (this repo has
        # two different hardcoded fallbacks for a missing check-in time, so
        # inventing one here would contradict both).
        times_line = ""
        try:
            check_in_time, check_out_time = await hotel_repository.get_standard_stay_times(hotel_id)
            parts = []
            if check_in_time:
                parts.append(f"Check-in: {check_in_time}")
            if check_out_time:
                parts.append(f"Check-out: {check_out_time}")
            if parts:
                times_line = "⏰ *Standard timings* — " + " · ".join(parts) + "\n\n"
        except Exception:
            logger.warning("standard times lookup failed for hotel %s", hotel_id, exc_info=True)

        response = templates.text_card(times_line + policy)
        response = _append_booking_reminder_to_cards(response, state)
        return {**state, "bot_response": response, "next_action": "menu_handled"}

    # ── Option 4: Hotel location & directions ────────────────────
    if intent == "SHOW_LOCATION":
        doc = await hotel_repository.get_hotel_location(hotel_id)
        response = templates.text_card(hotel_repository.format_location(doc))
        response = _append_booking_reminder_to_cards(response, state)
        return {**state, "bot_response": response, "next_action": "menu_handled"}

    # ── Option 5: Room photos & amenities (hotel-level) ─────────
    if intent == "SHOW_AMENITIES":
        response = []
        photo_tag = state.get("photo_tag_preference")
        # A request for one named room's photo is answered with that room's
        # photo. The booking-resume nudge is for a guest who has *stalled* in
        # front of the options card, so on a named-room request it both
        # contradicts the answer and repeats verbatim on the second request
        # (measured 2026-10-28, guest 917631666364: two photo requests, two
        # identical "you still have room options waiting" replies and no
        # picture). Generic "show me photos" and amenity requests keep it.
        targeted_photo_request = False

        last_msg = state.get("user_message", "").lower()
        is_photo_request = bool(photo_tag) or any(k in last_msg for k in ["photo", "pic", "image", "img", "gallery", "foto", "view", "send", "show"])
        is_amenity_request = any(k in last_msg for k in ["amenit", "facilit", "option 5"])
        
        # If user just typed "5" or something generic, show BOTH amenities and photos
        if not is_photo_request and not is_amenity_request:
            is_photo_request = True
            is_amenity_request = True
        
        # If user typed "5", show both (the old check only matched amenities for "5")
        if last_msg.strip() == "5":
            is_photo_request = True
            is_amenity_request = True
            
        if is_amenity_request:
            doc = await hotel_repository.get_hotel_amenities(hotel_id)
            response.extend(templates.text_card(hotel_repository.format_amenities(doc)))
        
        if is_photo_request:
            from app.services.hotel_media_service import MAX_IMAGES_PER_RESPONSE, media_delivery_card

            generic_photo_tags = {
                "room", "rooms", "hotel", "all", "photos", "photo", "pics", "pic",
                "picture", "pictures", "gallery", "see rooms", "view", "image", "images",
            }
            feature_tags = {
                "washroom", "bathroom", "reception", "lobby", "facade",
                "entrance", "pool", "gym", "restaurant", "dining", "balcony",
            }
            is_generic = not photo_tag or photo_tag.strip().lower() in generic_photo_tags

            if not is_generic:
                targeted_photo_request = True
                pt_clean = photo_tag.strip().lower()
                entity_type = None if pt_clean in feature_tags else "room"
                response.append(media_delivery_card(
                    entity_type=entity_type,
                    tags=[pt_clean],
                    caption=photo_tag.title(),
                    limit=2 if entity_type is None else 1,
                ))
            elif state.get("selected_rooms") and not any(k in last_msg for k in ["all", "other", "catalog", "different", "all rooms"]):
                # Deliver photos for the selected room(s) that the guest has chosen
                seen_room_ids = set()
                for room in state.get("selected_rooms", []):
                    r_id = str(room.get("room_id") or "").strip()
                    r_name = str(room.get("room_name") or "Room").strip()
                    if not r_id or r_id in seen_room_ids:
                        continue
                    seen_room_ids.add(r_id)
                    response.append(media_delivery_card(
                        entity_type="room",
                        entity_id=r_id,
                        tags=[r_name.lower()],
                        aliases=room.get("aliases", []),
                        caption=r_name,
                        limit=2,
                    ))
            elif state.get("allocation_options"):
                # Deliver 1 photo for each suggested room category in the current search/quote
                seen_room_ids = set()
                for opt in state.get("allocation_options", []):
                    rms = opt.get("rooms", [])
                    opt_rate = float(opt.get("total_price_per_night_inclusive") or opt.get("total_price_per_night") or 0)
                    for room in rms:
                        r_id = str(room.get("room_id") or "").strip()
                        r_name = str(room.get("room_name") or "Room").strip()
                        if not r_id or r_id in seen_room_ids:
                            continue
                        seen_room_ids.add(r_id)
                        caption = r_name
                        response.append(media_delivery_card(
                            entity_type="room",
                            entity_id=r_id,
                            tags=[r_name.lower()],
                            aliases=room.get("aliases", []),
                            caption=caption,
                            limit=1,
                        ))
            else:
                # Deliver 1 photo for each available room category in the catalog
                rooms = await room_repository.get_all_rooms(hotel_id)
                if rooms:
                    for room in rooms[:MAX_IMAGES_PER_RESPONSE]:
                        if not room.room_id:
                            continue
                        caption = room.name
                        response.append(media_delivery_card(
                            entity_type="room",
                            entity_id=room.room_id,
                            tags=[room.name.lower()],
                            aliases=room.aliases if hasattr(room, "aliases") and room.aliases else [],
                            caption=caption,
                            limit=1,
                        ))
                else:
                    response.append(media_delivery_card(
                        tags=[],
                        caption="Here are photos of our rooms.",
                        limit=1,
                    ))
            
        if not targeted_photo_request:
            response = _append_booking_reminder_to_cards(response, state)
        preserved_action = state.get("next_action")
        if preserved_action in (
            "need_guest_name_for_payment", "need_custom_rooms", "need_room_choice",
            "need_dates", "need_guests", "need_ac_preference", "need_child_ages"
        ):
            final_action = preserved_action
        else:
            final_action = "menu_handled"

        return {**state, "bot_response": response, "next_action": final_action}

    # ── Option 2: Check existing / last booking ──────────────────
    if intent == "CHECK_EXISTING_BOOKING":
        try:
            response = await _resolve_existing_booking_response(state)
        except Exception as e:
            logger.exception(f"CHECK_EXISTING_BOOKING failed: {e}")
            response = templates.existing_booking_not_found(lang=state.get("preferred_language") or "English")
        user_msg = (state.get("user_message") or "").lower()
        is_claiming = bool(state.get("claimed_existing_booking")) or any(
            p in user_msg for p in ("already", "pehle", "kitni", "done")
        )
        if not is_claiming:
            response = _append_booking_reminder_to_cards(response, state)
        return {**state, "bot_response": response, "next_action": "menu_handled"}

    # ── Option 6: Something else ─────────────────────────────────
    if intent == "SOMETHING_ELSE":
        if not state.get("check_in") or not state.get("check_out"):
            response = templates.text_card("Could you please share your check-in and check-out dates so I can check availability for you?")
            return {**state, "bot_response": response, "next_action": "menu_handled", "awaiting": "stay_dates"}
        elif not state.get("adults"):
            response = templates.text_card("How many guests will be staying with us?")
        elif state.get("booking_stage") in ("rooms_shown", "quote_shown"):
            # Range derived from the options actually offered — see the sibling
            # prompt in response_generator. The old literal "*1*, *2*, or *3*"
            # offered numbers that were not on screen.
            #
            # Numbers, not letters, since 2026-10-14, to match the card — which
            # stopped being lettered on 2026-09-23. The letter formatter stays in
            # `room_service` so an old lettered card in the guest's scrollback
            # still resolves.
            from app.services.room_service import format_option_numbers, option_number_count
            n = option_number_count(state.get("allocation_options"), state)
            response = templates.text_card(
                f"Please reply with a combination number ({format_option_numbers(n)}) to continue with your booking, or say *change* to adjust details."
            )
        else:
            response = templates.something_else_response()
        response = _append_booking_reminder_to_cards(response, state)
        return {**state, "bot_response": response, "next_action": "menu_handled"}

    # Fallthrough — shouldn't happen
    logger.warning(f"handle_menu called with unexpected intent: {intent}")
    return {**state, "bot_response": templates.fallback_response(), "next_action": "menu_handled"}
