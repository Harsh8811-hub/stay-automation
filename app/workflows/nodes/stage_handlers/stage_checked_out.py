"""
Stage 4 Handler — Checked Out Guest & Post-Stay Feedback.

Sir's Specification:
- A guest enters Stage 4 when check-out time passes, OR front desk marks checked-out.
- Goal: End stay warmly, capture feedback, retain guest for future stay.
- S62: Send ONE thank-you message with feedback question (1–5 rating).
  * Rating 4-5: Share Google review link once.
  * Rating 1-3: Apologize, raise Owner Task, do NOT ask for public review.
- S63: 'Next month phir aana hai' -> Switch to Stage 1 for new inquiry.
  Greet by name, do NOT reuse old dates/room!
- Lost and found: Raise owner task.
- Invoices / GST bills: Direct to front desk.
- Never pitch booking unless guest explicitly raises a new stay.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.prompts.templates import text_card
from app.repositories import task_repository
from app.services.hotel_identity import HotelIdentity, resolve_hotel_identity
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

#: Intents that mean "I want to stay again", whatever the words were. S63's
#: "never pitch unless the guest raises a new stay" and this are the same rule
#: read from both ends.
#
# These do not go through routing: a recent check-out still owns the turn until
# the guest has said what they want, and the switch below is what ends it. The
# set exists so that a message the classifier *did* understand counts. Measured
# on prod 2026-10-22: "2 people tomorrow" arrived here as BOOK_ROOM and was
# answered with the rating prompt, because only the phrase list was consulted.
_NEW_STAY_INTENTS = frozenset({
    "SHOW_PRICES",
    "BOOK_ROOM",
    "CHECK_AVAILABILITY",
    "SELECT_ROOM",
    "SELECTED_ROOM_DETAILS",
    "SHOW_PRICES_FOR_DATES",
    "BOOKING_SUMMARY",
    "ROOM_FEATURE_LOOKUP",
})


def _format_checkout_thank_you(guest_name: str, ident: HotelIdentity) -> str:
    return (
        f"Namaste {guest_name}! 🙏\n\n"
        f"{ident.name} mein rukne ke liye aapka bohot-bohot dhanyawad. "
        f"Hum aasha karte hain aapki city trip shubh aur aaramdayak rahi hogi.\n\n"
        f"Kripya humein 1 se 5 ke beech mein rate karke batayein aapka stay kaisa raha?\n"
        f"⭐⭐⭐⭐⭐ (Reply with 1 to 5)"
    )


async def handle_stage_checked_out(state: ConversationState) -> ConversationState:
    hotel_id = state.get("hotel_id", 111111)
    phone = state.get("customer_identifier") or ""
    user_msg = (state.get("user_message") or "").strip()
    user_msg_lower = user_msg.lower()
    intent = state.get("intent") or "UNKNOWN"
    ident = await resolve_hotel_identity(hotel_id, state)
    phone_md = f"**{ident.phone}**" if ident.phone else "**the front desk at reception**"
    # No review link on file for this property -> drop the line and the arrow
    # rather than print Sample Hotel's review page to another hotel's guest.
    review_block = (
        "Kripya 1 minute nikaal kar humare Google review page par apna anubhav zaroor share karein:\n"
        f"👉 {ident.review_link}\n\n"
    ) if ident.has_review_link else (
        "Aapka yeh anubhav humein aur behtar seva dene ke liye prerit karta hai.\n\n"
    )

    active_booking = state.get("active_booking") or {}
    booking_id = active_booking.get("booking_id") or state.get("active_booking_id") or "BK-PAST"
    guest_name = active_booking.get("guest_name") or state.get("guest_name") or "Valued Guest"

    logger.info(
        "STAGE_4_CHECKED_OUT_DISPATCH | hotel=%s phone=%s booking=%s msg='%s'",
        hotel_id, phone, booking_id, user_msg[:60],
    )

    # ── S63: Future Stay Inquiry ("Next month phir aana hai") ─────────────────
    # Two ways in: the phrases guests use, and the intents the graph already
    # resolved. The phrase list alone missed the most common shape of all — a
    # booking enquiry the classifier *did* understand. Measured on prod
    # 2026-10-22: "2 people tomorrow" arrived here as BOOK_ROOM and was answered
    # with the rating prompt, because none of these phrases matched it.
    is_future_stay = (
        intent in _NEW_STAY_INTENTS
        or any(
            p in user_msg_lower for p in (
                "phir aana hai", "next month", "agle mahine", "future", "again", "dobara", "next stay",
                "naya booking", "new booking", "book again", "fir aana", "agle hafte", "next year"
            )
        )
    )
    # A bare number is an answer to the question the bot just asked — the rating
    # prompt — not a new booking enquiry. On 2026-10-28 a guest replied "5" to
    # "how was your stay?" and the classifier read it as BOOK_ROOM, so the
    # future-stay branch below answered "which dates?" instead of thanking them.
    #
    # Deliberately `fullmatch` and deliberately narrower than the rating regex
    # further down: "5 people next month" contains a 5 and IS an enquiry, while
    # "5", "5/5" and "4 stars" are not. Same reason the enquiry check is not
    # simply moved below the rating block — its regex is positional and would
    # claim any message carrying a digit.
    _bare_rating = re.fullmatch(
        r"\s*([1-5])\s*(?:/\s*5|stars?|out of 5)?\s*[.!]?\s*", user_msg, re.IGNORECASE
    )

    if is_future_stay and not _bare_rating:
        # Switch to Stage 1 Lead behavior: greet by name, DO NOT reuse old dates/room
        resp = (
            f"Namaste {guest_name}! 🙏 Dobara aane ke liye aapka hamesha swagat hai.\n\n"
            f"Aap agle stay ke liye kis date par check-in karna chahenge aur kitne log honge? "
            f"Main aapke liye best available rooms aur special direct rates check kar deta hoon."
        )
        return {
            **state,
            "booking_stage": "lead",
            "lifecycle_stage": "lead",
            "check_in": None,
            "check_out": None,
            "selected_rooms": [],
            "allocation_options": [],
            "price_breakdown": None,
            "total_price": None,
            "next_action": "need_dates",
            # Without this the switch lasts exactly one turn: `context_loader`
            # re-resolves the stage from the past booking row before `understand`
            # runs, so the guest's dates would arrive back here and be wiped by
            # the `check_in: None` above — a loop that asks for dates forever.
            # The same latch a "start fresh" writes makes the resolver drop this
            # row from the next turn on, so the enquiry runs as an ordinary lead.
            "dismissed_booking_id": (
                state.get("active_booking_id") or state.get("booking_id")
                or state.get("dismissed_booking_id")
            ),
            # A stay staff confirmed in the CRM has no row of ours, so the latch
            # above is empty for it. Name its check-out instead, the same way
            # `understand`'s fresh-start path does.
            "crm_confirmation_dismissed_check_out": (
                state.get("crm_confirmed_check_out")
                or state.get("crm_confirmation_dismissed_check_out")
            ),
            "bot_response": text_card(resp),
        }

    # ── Rating Response (1 to 5) ─────────────────────────────────────────────
    # Check if guest sent a 1-5 rating
    rating_match = re.search(r'\b([1-5])\b', user_msg)
    if rating_match and not state.get("feedback_recorded"):
        score = int(rating_match.group(1))
        if score >= 4:
            resp = (
                f"Bohot bohot dhanyawad {guest_name}! 🌟\n\n"
                f"Humein behad khushi hai ki aapka anubhav shaandaar raha. "
                f"{review_block}"
                f"Aapke shubh vichar humein prerit karte hain. Shubh yatra! 🙏"
            )
        else:
            # Low rating: apologize, raise owner task, do NOT ask for Google review
            try:
                from app.workflows.nodes.stage_handlers import create_stage_task
                await create_stage_task(
                    hotel_id=hotel_id,
                    session_id=state.get("session_id") or "",
                    title=f"Low Stay Feedback ({score}/5) from {guest_name}",
                    description=f"Checked-out Guest {guest_name} rated stay {score}/5. Message: '{user_msg}'. Booking: {booking_id}. Follow up with guest.",
                    task_type="FRONT_DESK",
                    subtype="INFORMATION_REQUEST",
                    department="FRONT_DESK",
                    priority="HIGH",
                    booking_id=booking_id,
                    customer_name=guest_name,
                    customer_phone=phone,
                    metadata={"rating": score, "stage": "checked_out", "task_type": "complaint"},
                )
            except Exception as e:
                logger.warning(f"Error logging feedback task: {e}")

            resp = (
                f"Namaste {guest_name}. Humein khed hai ki aapka anubhav 5-star nahi raha. "
                f"Humne aapka feedback senior hotel management ko bhej diya hai taaki hum turant sudhaar kar sakein.\n\n"
                f"Aapke feedback ke liye dhanyawad. Kisi bhi sujhav ya shikayat ke liye aap humari team se seedhe sampark kar sakte hain: {phone_md}."
            )

        return {
            **state,
            "booking_stage": "checked_out",
            "lifecycle_stage": "checked_out",
            "feedback_recorded": True,
            "bot_response": text_card(resp),
        }

    # ── Lost and Found ───────────────────────────────────────────────────────
    is_lost_and_found = any(w in user_msg_lower for w in ("bhool gaya", "chhoot gaya", "lost", "left", "saman", "bag", "charger", "clothes", "ghadi", "watch"))
    if is_lost_and_found:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Lost & Found Inquiry from {guest_name} ({booking_id})",
                description=f"Guest reported possible lost item: '{user_msg}'. Booking: {booking_id}. Phone: {phone}.",
                task_type="SECURITY",
                subtype="LOST_AND_FOUND",
                department="SECURITY",
                priority="HIGH",
                booking_id=booking_id,
                customer_name=guest_name,
                customer_phone=phone,
                metadata={"guest_message": user_msg, "stage": "checked_out", "task_type": "service"},
            )
        except Exception as e:
            logger.warning(f"Error creating lost and found task: {e}")

        resp = (
            f"Humne aapki lost & found request housekeeping aur front desk team ko note karwa di hai.\n\n"
            f"Room check karke humari team aapse turant is number par sampark karegi. "
            f"Emergency inquiry ke liye reception par call karein: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "checked_out",
            "lifecycle_stage": "checked_out",
            "bot_response": text_card(resp),
        }

    # ── Invoice / Bill / Tax Document Ask ────────────────────────────────────
    is_invoice_ask = any(w in user_msg_lower for w in ("bill", "invoice", "receipt", "gst bill", "tax invoice", "parcha", "rasid"))
    if is_invoice_ask:
        resp = (
            f"Official GST invoice aur final bill ke liye kripya hotel reception / front desk team se "
            f"call ya WhatsApp karein: {phone_md}.\n\n"
            f"Team aapko official computerized tax bill turant provide kar degi."
        )
        return {
            **state,
            "booking_stage": "checked_out",
            "lifecycle_stage": "checked_out",
            "bot_response": text_card(resp),
        }

    # ── Initial Post-Checkout Thank You ──────────────────────────────────────
    return {
        **state,
        "booking_stage": "checked_out",
        "lifecycle_stage": "checked_out",
        "bot_response": text_card(_format_checkout_thank_you(guest_name, ident)),
    }
