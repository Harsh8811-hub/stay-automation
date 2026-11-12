"""The confirmed-reservation router, out of `understand`.

Extracted from `app/workflows/nodes/understand.py` on 2026-10-28 (DEFECTS.md
#25). Every moved statement is the original line, in the same order; the only
edits are on the four `return` statements, which now hand back `state` as well
as the reply. Proof: the body is AST-identical to the pre-change block once the
same wrapper is applied to the original in memory.

Two things this block does that the earlier slices do not:

**It rebinds `state`.** On the ghost-confirmed path it replaces `state` with
`demoted_state` and falls through, so the rest of the node runs against the
demoted dict. A local cannot cross a function boundary, so the function returns
`(reply, state)` and the caller rebinds. This is the same problem as
`_meal_plan_from_button` in `button_tap.py`, solved the same way.

**It reads two names that belong to its caller.** `_PAYMENT_CLAIM_RE` is a
constant compiled inside `understand`, and `_is_confirmed_side_query` is a
module-level helper there. Both are passed in and bound to their original names
at the top of the body, rather than imported: importing either from `understand`
would close an import cycle (this module is imported at `understand`'s module
level). The aliases are two extra statements and they keep every moved line
byte-identical.

The contract:
  * return `(reply, state)`; `reply` is the update dict to hand back to the
    graph, or `None` for "not the confirmed router's business, carry on";
  * `state` is the dict the rest of the node should now use — usually the one
    that was passed in, but on the ghost path the demoted copy;
  * no `await`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable

from app.config.settings import get_settings

# Same logger object as the node, by name: importing it would be a cycle.
logger = logging.getLogger("app.workflows.nodes.understand")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# A guest who is *asking* about the booking they already hold can also type
# its dates ("my booking 28 Nov to 1 Dec, is breakfast included?"). That is a
# question, not a new stay, and it must not open a draft.
_QUESTION_START = re.compile(
    r"^\s*(?:what|when|which|where|how|is|are|do|does|can|could|kya|kab|kitna|kaise)\b",
    re.IGNORECASE,
)

_ROOM_COUNT = re.compile(r"\b\d+\s*(?:rooms?|kamar\w*)\b", re.IGNORECASE)


def _answers_with_a_booking(message: str) -> bool:
    """True when the message states a stay, not a statement about one.

    Used only while S57's "is that a separate booking?" question is open. The
    bar is a check-in date the guest typed, plus either a check-out or a room
    count — one of the two is enough because the guest writes in typos that
    the regex cannot read ("5 decemeber" parses as no check-out at all, and
    the model reads it fine on the turn that follows).
    """
    text = message or ""
    # The refusal answer wins. "no, my existing booking is 28 Nov to 1 Dec"
    # carries a date pair and is the opposite of a yes.
    from app.workflows.nodes.stage_handlers.stage_confirmed import (
        _SEPARATE_BOOKING_NO,
    )

    if _SEPARATE_BOOKING_NO.match(text):
        return False
    if "?" in text or _QUESTION_START.match(text):
        return False

    from app.services.date_extractor import extract_stay_dates
    from app.utils.helpers import today_str

    today = datetime.strptime(today_str(), "%Y-%m-%d").date()
    found = extract_stay_dates(text, today=today)
    if not (found.found and found.check_in):
        return False
    return bool(found.check_out) or bool(_ROOM_COUNT.search(text))


def handle_confirmed_reservation(
    state: dict[str, Any],
    user_msg_clean: str,
    stage: str,
    payment_claim_re: re.Pattern[str],
    is_confirmed_side_query: Callable[[str], bool],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Route a turn from a guest whose booking is already confirmed."""
    # aliases, so the moved statements below are byte-identical to the originals
    _PAYMENT_CLAIM_RE = payment_claim_re
    _is_confirmed_side_query = is_confirmed_side_query
    # ── S8: CONFIRMED RESERVATION ROUTER & MULTI-BOOKING MEMORY ──
    if stage == "confirmed" or (state.get("booking_id") and stage in ("confirmed", "checked_in")):
        # 0. S57's question was open, and the guest answered with the booking
        #    itself rather than the word "yes".
        #
        #    Measured 2026-11-05 (guest 917631666364, evaled as
        #    `week-sahu-confirmed-guest-new-booking`): asked "kya aap ek naya
        #    alag booking karna chahte hain?", the guest replied
        #    "6 rooms needed, 28 november to 5 decemeber". That message is a
        #    yes by any reading — nobody gives dates for a booking they are not
        #    making — but it matched neither the yes words nor the room-browse
        #    pattern, so the stage handler answered it as a general question and
        #    the dates were dropped on the floor. The next turn ("10", the guest
        #    count) had nothing to attach to and got "your message may have come
        #    through incomplete".
        #
        #    Returning `None` is the point: this opens the draft and then hands
        #    the turn back to extraction, so the dates the guest just typed land
        #    in it. Answering here with a "please share your dates" template —
        #    which is what the yes branch in stage_confirmed does — would drop
        #    the very details that proved this was a yes.
        if state.get("separate_booking_clarification_asked") and _answers_with_a_booking(
            user_msg_clean
        ):
            from app.services.second_booking import second_booking_gate_enabled

            logger.info(
                "S57_ANSWERED_WITH_DATES | opening the second-booking draft and "
                "handing the turn to extraction | msg=%r",
                user_msg_clean,
            )
            opened: dict[str, Any] = {
                **state,
                "booking_stage": "lead",
                "lifecycle_stage": "lead",
                "separate_booking_clarification_asked": False,
                "is_separate_booking_inquiry": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "next_action": "need_dates",
            }
            if second_booking_gate_enabled(state):
                # Same durable latch stage_confirmed writes: without it the
                # resolver finds the confirmed row again next turn and the draft
                # dies one message after it opened.
                opened["separate_booking_active"] = True
                opened["separate_booking_started_at"] = _utc_now_iso()
            return (None, opened)

        # 1. If user explicitly requests a new booking
        if any(w in user_msg_clean for w in ("make a new booking", "new booking", "another booking", "book another room", "book one more room", "1 make a new booking", "book another")):
            completed = list(state.get("completed_bookings") or [])
            if state.get("booking_id") and state.get("booking_id") not in [b.get("booking_id") if isinstance(b, dict) else b for b in completed]:
                completed.append({
                    "booking_id": state.get("booking_id"),
                    "guest_name": state.get("guest_name"),
                    "check_in": state.get("check_in"),
                    "check_out": state.get("check_out"),
                    "room_name": state.get("room_type_preference"),
                    "total_price": state.get("total_price"),
                })
            return ({
                **state,
                "completed_bookings": completed,
                "booking_id": None,
                "check_in": None,
                "check_out": None,
                "adults": None,
                "children": None,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "payment_details": None,
                "payment_status": None,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "need_dates",
                "bot_response": "Great! Let's arrange your new booking. Please share your **stay dates** and **number of guests**! 🏨",
                "extracted_data": None,
                "extraction_result": None,
            }, state)

        # 2. General conversation / greetings / payment confirmations in confirmed state
        is_greeting_or_ack = user_msg_clean in ("hi", "hello", "namaste", "hey", "thanks", "thank you", "ok", "okay", "done", "not now", "no thanks")
        is_payment_conf = bool(_PAYMENT_CLAIM_RE.search(user_msg_clean))
        raw_b_id = state.get("booking_id") or state.get("active_booking_id")
        # A stay confirmed by staff in the CRM has no booking row of ours and so no
        # `booking_id`. Without this the ghost guard below would demote a guest who has
        # already paid back to a lead and start selling to them again — which is the
        # whole bug the stage resolver's CRM branch exists to fix.
        from app.utils.helpers import guest_is_confirmed

        has_real_confirmed_booking = guest_is_confirmed(state, raw_b_id)

        if not has_real_confirmed_booking:
            logger.warning("UNDERSTAND_GHOST_CONFIRMED_GUARD | Confirmed stage with no real booking_id; demoting to lead")
            demoted_state = {
                **state,
                "booking_stage": "lead",
                "lifecycle_stage": "lead",
                "booking_id": None,
                "active_booking_id": None,
            }
            if user_msg_clean in ("no thanks", "no thank you", "nahi", "not now", "cancel", "not interested"):
                return ({
                    **demoted_state,
                    "intent": "UNKNOWN",
                    "next_action": None,
                    "bot_response": "Thank you! If you plan a trip to the city in the future, feel free to reach out. Have a wonderful day! 🙏",
                    "extracted_data": None,
                    "extraction_result": None,
                }, state)
            # Fall through to general flow for inquiry/dates
            state = demoted_state
        elif is_greeting_or_ack or is_payment_conf:
            b_id = raw_b_id
            g_name = state.get("guest_name") or "Guest"
            ci = state.get("check_in") or ""
            co = state.get("check_out") or ""
            dates_disp = f" ({ci} to {co})" if ci and co else ""
            # A CRM-confirmed guest has no booking id of ours. Naming no reference
            # reads better than printing the literal "(`None`)" at them.
            id_disp = f" (`{b_id}`)" if b_id else ""
            if user_msg_clean in ("no thanks", "no thank you", "not now", "ok", "okay", "thanks", "thank you"):
                ack_reply = f"Bilkul {g_name}! Swagat ki poori taiyari hai. Kisi bhi sahayata ke liye hum yahan hain. Shubh din! 🙏"
            else:
                ack_reply = f"Hello **{g_name}**! Your reservation{id_disp}{dates_disp} is confirmed with us! 🏨\n\nHow can I help you today? You can share your ID card for express check-in, ask for directions, WiFi info, or anything else."
            return ({
                **state,
                "intent": "PAYMENT_STATUS" if is_payment_conf else "GREETING",
                "booking_stage": "confirmed",
                "next_action": None,
                "bot_response": ack_reply,
                "extracted_data": None,
                "extraction_result": None,
            }, state)
        elif (
            get_settings().CONFIRMED_BOOKING_STOPS_THE_BOT
            # Deliberately still `next_action`, which is None by the time a turn runs — so
            # this clause never blocks the suppression. It is a "is anything pending at
            # all" check, not a "which prompt am I waiting on" dispatch, and the
            # confirmed-booking suppression around it (2026-10-13) is the delicate one
            # with the ghost guards. Left as found rather than widened to
            # `active_prompt_action` on the back of a pointer fix; flagged, not changed.
            and not state.get("next_action")
            and not _is_confirmed_side_query(user_msg_clean)
        ):
            b_id = raw_b_id
            g_name = state.get("guest_name") or "Guest"
            id_disp = f" (`{b_id}`)" if b_id else ""
            return ({
                **state,
                "intent": "PAYMENT_STATUS",
                "booking_stage": "confirmed",
                "next_action": None,
                "bot_response": (
                    f"Your booking{id_disp} is confirmed, **{g_name}** — you're all set! ✅\n\n"
                    "I'll leave you to it. If you need anything, just ask: "
                    "*WiFi, directions, check-in time, amenities, or your invoice*.\n\n"
                    "To book another room, say **\"new booking\"**."
                ),
                "extracted_data": None,
                "extraction_result": None,
            }, state)

    return (None, state)
