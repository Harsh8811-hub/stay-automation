"""
Stage 3 Handler — In-House Concierge.

Sir's Specification:
- A guest enters Stage 3 when check-in date reaches property's check-in time (Asia/Kolkata),
  OR front desk marks guest checked-in early.
- Goal: Front-desk concierge. Solve guest's needs quickly.
- Check-in welcome greeting: Wi-Fi, breakfast timing, check-out time, front desk number.
- Service requests / Complaints (AC not working, towels, cleaning, TV, water):
  * Apologize, raise Owner Task with room number and urgency.
  * Provide front-desk phone number.
- S59: 'AC kaam nahi kar raha' -> Apologize, service task with room #, front desk phone.
- S60: 'Ganga aarti kab hoti hai?' -> Timing (6:45 PM), Riverfront Plaza, 5 mins walk.
- S61: 'Ek din aur rukna hai' -> Raise owner task, quote extra night only using rate engine.
  Do NOT start new booking flow from scratch.
- NEVER pitch new rooms, never ask for dates/payment for current stay.
"""

from __future__ import annotations

import logging
from typing import Any

from app.prompts.templates import text_card
from app.repositories import task_repository
from app.services.hotel_identity import HotelIdentity, resolve_hotel_identity
from app.workflows.nodes.stage_handlers.kb_answer import knowledge_base_answer
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# Subtypes this handler answers itself, further down this file. They are
# excluded from the classifier delegation below so that adding it does not
# quietly replace a better answer with a generic one: the bill gets its own
# reply, an extension gets a real extra-night rate, and an information request
# is a question, not a task to dispatch.
_IN_HOUSE_OWN_SUBTYPES = {
    "BILLING_INQUIRY",
    "INVOICE_REQUEST",
    "PAYMENT_CONFIRMATION",
    "INFORMATION_REQUEST",
    "OTHER",
    "CHECKOUT_REQUEST",
    "LATE_CHECKOUT",
    "ROOM_CHANGE",
}


def _nights_between(check_in: Any, check_out: Any) -> int | None:
    """
    Whole nights between two stored dates, or None if they cannot be read.

    Tolerant on purpose: `check_in`/`check_out` reach this from Mongo and are
    variously a `datetime`, a `date`, or a string in one of several formats
    depending on which writer created the booking.
    """
    from datetime import date, datetime

    def _as_date(val: Any) -> date | None:
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, date):
            return val
        if isinstance(val, str) and val.strip():
            text = val.strip()[:10]
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(text, fmt).date()
                except ValueError:
                    continue
        return None

    start, end = _as_date(check_in), _as_date(check_out)
    if start and end and end > start:
        return (end - start).days
    return None


def _resolve_nightly_rate(
    active_booking: dict,
    state: dict,
    room_number: str | None = None,
) -> tuple[float | None, str]:
    """
    The per-night rate for the guest's current room, **read, never invented**.

    Returns `(rate, source)`, where source names where the number came from so
    the front-desk task can say so and an operator can audit it. `rate` is
    `None` when the booking does not carry a usable figure — and that is a real
    outcome, not a failure: the copy then asks the desk to quote, instead of
    the bot stating a price it made up.

    This replaces a literal `nightly_rate = 2550.0`, which quoted every guest
    at every property the same invented number. Measured 2026-10-06 against the
    live `bookings` collection: of 208 room entries on 400 bookings,
    `price_per_night` is populated on 153 and null/absent on 55, and
    `price_breakdown` is stored on none of them — so neither source alone is
    sufficient and both are tried.

    Order:
      1. the booked room's own `price_per_night` (the rate actually agreed)
      2. `total_price / num_nights` (or nights derived from the stay dates)
    """
    booking = active_booking or {}
    rooms = booking.get("rooms") or state.get("selected_rooms") or []

    # 1) The rate recorded against the room the guest is in.
    if isinstance(rooms, list):
        chosen = None
        for entry in rooms:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("room_number") or entry.get("room_name") or "").strip()
            if room_number and label and str(room_number).strip() in label:
                chosen = entry
                break
        if chosen is None:
            chosen = next((e for e in rooms if isinstance(e, dict)), None)

        if chosen is not None:
            raw = chosen.get("price_per_night")
            if raw not in (None, ""):
                try:
                    rate = float(raw)
                except (TypeError, ValueError):
                    rate = 0.0
                if rate > 0:
                    return rate, "booking_room_rate"

    # 2) Recover it from the booking total.
    total = booking.get("total_price") or state.get("total_price")
    nights = booking.get("num_nights") or _nights_between(
        booking.get("check_in") or state.get("check_in"),
        booking.get("check_out") or state.get("check_out"),
    )
    if total not in (None, "") and nights:
        try:
            rate = float(total) / float(nights)
        except (TypeError, ValueError, ZeroDivisionError):
            rate = 0.0
        if rate > 0:
            return rate, "booking_total_per_night"

    return None, "unavailable"


def _format_checkin_welcome(guest_name: str, room_num: str | None, ident: HotelIdentity) -> str:
    """
    The check-in welcome card.

    The wi-fi block is built only when the property's credentials are actually
    known. `_KNOWN_PROPERTY_EXTRAS` in `hotel_identity` is sparse on purpose —
    before 2026-10 this function printed Sample Hotel's network and password to
    every tenant that was not Sample Property, so a Hotel Paradise guest was handed
    another hotel's wi-fi. Omitting two lines is the smaller harm.
    """
    wifi_block = ""
    if getattr(ident, "has_wifi", False):
        wifi_block = (
            f"📶 *Wi-Fi Network:* `{getattr(ident, 'wifi_name', '')}`\n"
            f"🔑 *Wi-Fi Password:* `{getattr(ident, 'wifi_password', '')}`\n"
        )
    return (
        f"Namaste {guest_name}! 🏨 {ident.name} mein aapka hardik swagat hai.\n\n"
        f"Aapke aaramdayak stay ke liye zaroori details:\n"
        f"{wifi_block}"
        f"🍳 *Breakfast Timings:* 8:00 AM – 10:30 AM (Dining Hall)\n"
        f"🕚 *Check-out Time:* 11:00 AM\n"
        f"📞 *{ident.front_desk_clause()}* (Dial 9 from room)\n\n"
        f"_Kisi bhi room service, cleaning, ya local tour ke liye aap seedhe yahan message kar sakte hain!_"
    )


async def handle_stage_in_house(state: ConversationState) -> ConversationState:
    hotel_id = state.get("hotel_id", 111111)
    phone = state.get("customer_identifier") or ""
    user_msg = (state.get("user_message") or "").strip()
    user_msg_lower = user_msg.lower()
    ident = await resolve_hotel_identity(hotel_id, state)
    # Rendered once, because the prose below embeds the number mid-sentence
    # ("call karein: **<number>**"). An empty interpolation would leave a
    # dangling "**.**", so a property with no number on file gets a phrase
    # instead. Not a fallback to another hotel's number — see `hotel_identity`.
    phone_md = f"**{ident.phone}**" if ident.phone else "**the front desk at reception**"

    active_booking = state.get("active_booking") or {}
    booking_id = active_booking.get("booking_id") or state.get("active_booking_id") or state.get("booking_id") or "BK-STAY"
    guest_name = active_booking.get("guest_name") or state.get("guest_name") or "Guest"
    room_number = active_booking.get("room_number") or state.get("room_number") or None
    check_out = str(active_booking.get("check_out") or active_booking.get("check_out_date") or state.get("check_out") or "")

    logger.info(
        "STAGE_3_IN_HOUSE_DISPATCH | hotel=%s phone=%s room=%s booking=%s msg='%s'"
        " | task=%s/%s/%.2f",
        hotel_id, phone, room_number, booking_id, user_msg[:60],
        state.get("task_is_request"), state.get("task_subtype"),
        float(state.get("task_confidence") or 0),
    )

    # ── Check-in Welcome (Sent once on check-in or "Hi") ─────────────────────
    if not state.get("checkin_welcome_sent") and any(user_msg_lower == g for g in ("hi", "hello", "namaste", "checked in", "check in")):
        return {
            **state,
            "booking_stage": "in_house",
            "lifecycle_stage": "in_house",
            "checkin_welcome_sent": True,
            "bot_response": text_card(_format_checkin_welcome(guest_name, room_number, ident)),
        }

    # ── S59: Service Requests & Complaints ("AC kaam nahi kar raha", towels, cleaning) ──
    complaint_cues = (
        "ac kaam nahi", "ac nahi chal raha", "ac not working", "fan", "tv", "water",
        "paani", "towel", "cleaning", "safai", "pillow", "blanket", "chadar", "sheet",
        "badboo", "smell", "leak", "flush", "hot water", "geyser", "room service",
        "kettle", "chai", "tea", "coffee"
    )
    is_service_or_complaint = any(c in user_msg_lower for c in complaint_cues) or any(
        w in user_msg_lower for w in ("problem", "issue", "complain", "kharab", "shikayat", "toota")
    )
    if is_service_or_complaint:
        task_type = "complaint" if any(w in user_msg_lower for w in ("ac", "tv", "kharab", "smell", "not working", "leak")) else "service"
        priority = "URGENT" if task_type == "complaint" else "NORMAL"
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            task_room_str = f" in Room {room_number}" if room_number else ""
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"{task_type.title()}{task_room_str}: {user_msg[:40]}",
                description=f"In-house Guest {guest_name}{task_room_str} reported: '{user_msg}'. Booking: {booking_id}. Phone: {phone}.",
                task_type="MAINTENANCE" if "ac" in user_msg_lower or "tv" in user_msg_lower else "HOUSEKEEPING",
                subtype="AC_REPAIR" if "ac" in user_msg_lower else "GENERAL_MAINTENANCE",
                department="MAINTENANCE" if "ac" in user_msg_lower or "tv" in user_msg_lower else "HOUSEKEEPING",
                priority=priority,
                booking_id=booking_id,
                room_number=room_number,
                customer_name=guest_name,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "in_house",
                    "room_number": room_number,
                    "task_type": task_type,
                },
            )
        except Exception as e:
            logger.warning(f"Error creating in-house task: {e}")

        resp = (
            f"Namaste {guest_name}! Humein bohot khed hai ki aapko pareshani hui.\n\n"
            f"Humne staff aur housekeeping team ko turant inform kar diya hai. "
            f"Staff sadasya jald hi aapke room par aayenge.\n\n"
            f"Kisi bhi urgent sahayata ke liye aap reception par dial 9 kar sakte hain ya call karein: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "in_house",
            "lifecycle_stage": "in_house",
            "bot_response": text_card(resp),
        }

    # ── Classified hotel tasks the cue list cannot see ───────────────────────
    # The cue list above is hand-written and never consults the classifier, so
    # an in-house guest's "Wake-up call at 6am please" or "Call me a cab for
    # 8pm" — both classified HOTEL_TASK at 0.95, subtype FRONT_DESK — matched
    # no cue and fell through to the FAQ path, which answered "I'm not entirely
    # sure about that right now, but I have noted down your question". No task
    # row, nothing for a human, and a promise the bot cannot keep. See DEFECTS
    # item 8.
    #
    # This is deliberately a *state* read (`task_is_request` and friends are
    # stamped by the classifier in `understand`), not another phrase list: the
    # next service the hotel adds should not need a new regex here.
    #
    # Placed after the complaint branch, not before it, so the warm Hinglish
    # reply for a broken AC survives. Placed before the bill/extension branches
    # only in ordering terms — those subtypes are excluded above, so they never
    # reach the delegation.
    _task_subtype = (state.get("task_subtype") or "").upper()
    if (
        state.get("task_is_request")
        and float(state.get("task_confidence") or 0) >= 0.6
        and _task_subtype not in _IN_HOUSE_OWN_SUBTYPES
    ):
        from app.services.task_engine import process_task_request

        try:
            result = await process_task_request(state)
        except Exception as e:
            # A task-engine fault must not turn into silence. Fall through to
            # the rest of this handler, which ends in a real answer either way.
            logger.warning(
                "STAGE_3_IN_HOUSE_TASK_DELEGATION_FAILED | hotel=%s subtype=%s | %s",
                hotel_id, _task_subtype, e,
            )
            result = {}
        if result.get("task_id"):
            logger.info(
                "STAGE_3_IN_HOUSE_TASK_DELEGATED | hotel=%s subtype=%s task=%s",
                hotel_id, _task_subtype, result["task_id"],
            )
            return {
                **state,
                "booking_stage": "in_house",
                "lifecycle_stage": "in_house",
                "bot_response": text_card(result["response_text"]),
            }

    # ── Bill / Invoice / Receipt Request ─────────────────────────────────────
    is_bill_ask = any(
        w in user_msg_lower for w in ("bill", "invoice", "receipt", "payment slip", "vej dijiye", "bhej dijiye", "bhejiye", "hisab", "parchi")
    ) and not any(w in user_msg_lower for w in ("room chahiye", "booking", "naya room", "ek din aur", "extend"))
    if is_bill_ask:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            task_room_label = f"Room {room_number} " if room_number else ""
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Bill / Invoice Request: {task_room_label}({guest_name})",
                description=f"In-house Guest {guest_name}{f' in Room {room_number}' if room_number else ''} ({phone}) requested bill/invoice: '{user_msg}'. Booking: {booking_id}.",
                task_type="FRONT_DESK",
                subtype="BILLING_INQUIRY",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                room_number=room_number,
                customer_name=guest_name,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "in_house",
                    "room_number": room_number,
                    "task_type": "billing_inquiry",
                },
            )
        except Exception as e:
            logger.warning(f"Error creating in-house bill task: {e}")

        resp = (
            f"Namaste {guest_name}! Humne front desk team ko aapka bill/invoice share karne ke liye inform kar diya hai. 📄\n\n"
            f"Front desk staff jald hi aapse bill share karega. Urgent query ke liye reception dial karein (Ext: 9) ya call karein: {phone_md}.\n\n"
            f"Aur kisi cheez mein hum aapki madad kar sakte hain? 😊"
        )
        return {
            **state,
            "booking_stage": "in_house",
            "lifecycle_stage": "in_house",
            "bot_response": text_card(resp),
        }

    # ── S60: Local Sightseeing / Ganga Aarti / Temples ───────────────────────
    is_sightseeing = any(
        p in user_msg_lower for p in (
            "ganga aarti", "aarti", "darshan", "vishwanath", "kashi", "ghat", "mandir",
            "sarnath", "boat", "nao", "timing", "kab hoti hai", "subah", "sham"
        )
    )
    if is_sightseeing:
        resp = (
            f"The city's main aakarshan aur timings:\n\n"
            f"🌊 *Riverfront Plaza Ganga Aarti:*\n"
            f"• **Timing:** Roz sham 6:45 PM – 7:00 PM par shuru hoti hai (~45 mins).\n"
            f"• **Doori:** {ident.name} se paidal sirf 5-10 minute.\n"
            f"• **Tip:** Best view ke liye 6:00 PM tak ghat par pahunchein ya reception se boat arrange karwayein.\n\n"
            f"🛕 *Shri City Temple:*\n"
            f"• **Gate #4:** Hotel se sirf paidal doori par.\n"
            f"• **Mangla Aarti:** 3:00 AM – 4:00 AM.\n"
            f"• **Sugam Darshan:** Reception se guidance uplabdh hai.\n\n"
            f"🚤 *Subah ki Boat Ride & Heritage Museum Excursion:*\n"
            f"• Sunrise boat ride (Riverside to Old Town) aur Heritage Museum day tour ke liye front desk par sampark karein: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "in_house",
            "lifecycle_stage": "in_house",
            "bot_response": text_card(resp),
        }

    # ── S61: Stay Extension / Extra Night ("Ek din aur rukna hai") ───────────
    is_extension_ask = any(
        p in user_msg_lower for p in (
            "ek din aur", "extra night", "extend", "rukna hai", "stay badhana", "one more day",
            "ek raat aur", "extend stay", "checkout late", "late checkout"
        )
    )
    if is_extension_ask:
        # Quote extra night only using standard pricing, raise Owner Task, DO NOT restart booking flow
        nightly_rate, rate_source = _resolve_nightly_rate(active_booking, state, room_number)
        if nightly_rate is None:
            logger.warning(
                "STAY_EXTENSION_RATE_UNAVAILABLE | hotel=%s booking=%s room=%s — "
                "no rate on the booking; asking the desk rather than quoting a guess",
                hotel_id, booking_id, room_number,
            )
            quote_line = "• *Extra Night Room Rate:* front desk will confirm and share the exact rate\n"
            quote_desc = "Rate to be confirmed by the front desk (booking carries no usable nightly rate)."
        else:
            quote_line = f"• *Extra Night Room Rate:* Rs {nightly_rate:,.0f} (subject to room availability)\n"
            quote_desc = f"Quote provided: Rs {nightly_rate:,.0f} (source: {rate_source})."

        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            task_room_label = f"Room {room_number} " if room_number else ""
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Stay Extension Request: {task_room_label}({guest_name})",
                description=f"Guest{f' in Room {room_number}' if room_number else ''} requested stay extension: '{user_msg}'. Current checkout: {check_out}. {quote_desc}",
                task_type="FRONT_DESK",
                subtype="INFORMATION_REQUEST",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                room_number=room_number,
                customer_name=guest_name,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "in_house",
                    "room_number": room_number,
                    # None when the booking had no usable rate — the desk sees
                    # "unknown" rather than a number nobody can trace.
                    "quoted_extension_rate": nightly_rate,
                    "quoted_extension_rate_source": rate_source,
                    "task_type": "stay_extension",
                },
            )
        except Exception as e:
            logger.warning(f"Error creating stay extension task: {e}")

        resp = (
            f"Namaste {guest_name}! Hum aapka stay extend karne mein bilkul khush hain.\n\n"
            f"• *Current Check-out:* {check_out}\n"
            f"{quote_line}\n"
            f"Humne front desk team ko aapka extension request bhej diya hai taaki aapka stay extend ho sake. "
            f"Front desk se confirmation aate hi hum aapko update kar denge. Reception number: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "in_house",
            "lifecycle_stage": "in_house",
            "bot_response": text_card(resp),
        }

    # ── Food & Dining ────────────────────────────────────────────────────────
    is_food = any(w in user_msg_lower for w in ("breakfast", "nashta", "khana", "lunch", "dinner", "restaurant", "food", "menu"))
    if is_food:
        resp = (
            f"🍽️ *Dining & Food at {ident.name}:*\n\n"
            f"• *Complimentary Breakfast:* 8:00 AM se 10:30 AM tak Dining Area mein parosa jata hai.\n"
            f"• *In-room Dining:* Chai, coffee, aur snacks ke liye reception dial karein (Ext: 9).\n"
            f"• *Famous Local Eateries:* Local Snack House (Central Market Square) aur Corner Cafe sirf 3-4 minutes ki doori par hain.\n\n"
            f"Kuch aur mangaane ke liye dial karein: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "in_house",
            "lifecycle_stage": "in_house",
            "bot_response": text_card(resp),
        }

    # ── Everything Else — Answer From the Hotel's Knowledge Base ──────────────
    #
    # The nine branches above are keyword-matched and cover a service request, a
    # bill, sightseeing, an extension and food. Everything outside them — "what
    # is the wifi password", "do you allow pets", "is there parking", "what time
    # is checkout", "how far is the ghat" — fell to the canned paragraph below,
    # which names four topics and answers none of them.
    #
    # Product, 2026-10-11: *"faq mode till check out"*. The knowledge base is
    # unreachable from here for the same routing reason as in `stage_confirmed`
    # — the in-house router returns before the `intent == "FAQ"` branch — so the
    # stage has to call it. Same helper, same three guards (do not move the
    # guest, do not sell, do not raise), same fallback.
    #
    # Deliberately last: every branch above is a specific, scripted answer for a
    # request this hotel has an operational response to (raise a task, quote the
    # extra night, give the aarti timing). A general knowledge answer must never
    # pre-empt one of those.
    kb_state = await knowledge_base_answer(
        state, stage="in_house", log_tag="STAGE_3_IN_HOUSE",
    )
    if kb_state is not None:
        return kb_state

    # ── Default In-House Concierge Guidance (NO New Selling) ──────────────────
    resp = (
        f"Namaste {guest_name}! 🙏\n\n"
        f"{ident.name} mein aapke aaram ke liye hum hamesha uplabdh hain. "
        f"Wi-Fi, breakfast (8:00–10:30 AM), room cleaning, ya Evening River Ceremony jankari ke liye pooch sakte hain.\n"
        f"Direct front desk support: {phone_md}."
    )
    return {
        **state,
        "booking_stage": "in_house",
        "lifecycle_stage": "in_house",
        "bot_response": text_card(resp),
    }
