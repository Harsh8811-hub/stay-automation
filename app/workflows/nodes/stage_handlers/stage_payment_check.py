"""
Stage 1b Handler — Payment Under Check.

Sir's Specification:
- A guest enters Stage 1b when sending a payment screenshot OR claiming payment ("maine payment kar diya").
- The bot detects the payment claim/proof, thanks the guest, and explains front desk is verifying.
- Raises an owner task containing payment message or image.
- Booking remains PENDING status.
- The bot MUST NOT say the booking is confirmed.
- The bot MUST NOT build a receipt-style booking card.
- Do NOT ask upsell questions, meal plans, or add-ons.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.prompts.templates import text_card
from app.repositories import booking_repository, task_repository
from app.repositories.booking_panel_fields import status_aliases
from app.utils.payment_escalation import escalation_due, hours_waiting
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# A booking in any of these states has already been settled (or killed) and must
# never be walked backwards into "pending_payment" by a payment claim. Kept as
# the *union* of the two vocabularies, because `stage_resolver` reads both
# `booking_status` and `stage` and either one being set to pending_payment /
# payment_check overrides everything else.
TERMINAL_BOOKING_STATUSES = ("confirmed", "checked_in", "checked_out", "cancelled")
TERMINAL_STAGES = ("confirmed", "in_house", "checked_out", "cancelled")

# ── What the guest is actually asking, while their payment is verified ───────
#
# These are deliberately narrow, and they only ever choose between replies that
# all state the same fact — the payment is still under verification, the
# booking is not confirmed yet. None of them can promote a booking; a wrong
# match costs a slightly less relevant sentence, never a false confirmation.
_CLAIM_RE = re.compile(
    r"\b(paid|payment|pay\s*kar|paisa|paise|utr|txn|transaction|screenshot|"
    r"upi|transferred?|bhej\s*diya|kar\s*diya|done)\b",
    re.IGNORECASE,
)
_CONFIRM_ASK_RE = re.compile(
    r"\b(confirmation|confirmed?|receipt|voucher|invoice|slip|booking\s+details|"
    r"booking\s+id|booking\s+reference)\b",
    re.IGNORECASE,
)
_URGENT_RE = re.compile(
    r"\b(today|aaj|jaldi|urgent|abhi|right\s+now|check\s*-?\s*in|checkout|"
    r"check\s*-?\s*out|baje|tonight)\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|namaste|namaskar|salaam|good\s+\w+)\s*[!.🙏]*\s*$", re.IGNORECASE)


class _NoClaimOnThisTurn(Exception):
    """Control flow inside step 1: this turn does not claim payment, so the
    booking record is left alone. Never escapes the handler."""


def _status_reply(user_msg: str, booking_id: str | None = None, is_hi: bool = True) -> str | None:
    """
    A reply that answers the guest's turn instead of repeating the same paragraph.

    Returns None when the message is a payment claim (or unrecognised), so the
    caller keeps the original, deliberately fixed wording.
    """
    msg = (user_msg or "").strip()
    if not msg:
        return None

    from app.utils.booking_resume import is_real_booking_id
    has_real_ref = is_real_booking_id(booking_id)
    b_id = str(booking_id).strip() if has_real_ref else ""
    ref_line = f"• *Booking Reference:* *{b_id}*\n" if has_real_ref else ""

    if _GREETING_RE.match(msg):
        if is_hi:
            booking_part = f"Aapki booking *{b_id}* ka payment" if has_real_ref else "Aapka payment"
            return (
                f"Namaste! 🙏 {booking_part} abhi front desk verify kar rahi hai.\n\n"
                "• *Current Status:* ⏳ *PENDING VERIFICATION*\n\n"
                "Verify hote hi confirmation slip bhej di jayegi. Kuch aur chahiye to bata dijiye."
            )
        else:
            booking_part = f"for booking *{b_id}*" if has_real_ref else ""
            return (
                f"Hello! 🙏 Our front desk is currently verifying the payment {booking_part}.\n\n"
                "• *Current Status:* ⏳ *PENDING VERIFICATION*\n\n"
                "We will send your confirmation slip as soon as it is verified. Please let us know if you need anything else."
            )

    if _URGENT_RE.search(msg):
        if is_hi:
            return (
                "Aapka message front desk ko bhej diya hai — wo aapke check-in/"
                "check-out ki timing dekh rahe hain. 🙏\n\n"
                f"{ref_line}"
                "• *Payment Status:* ⏳ *PENDING VERIFICATION*\n"
                "• *Booking Status:* verification ke baad hi confirm hoga\n\n"
                "Payment verify hote hi wo aapko isi chat par confirmation bhej "
                "denge, taaki aap check-in kar sakein."
            )
        else:
            return (
                "We have forwarded your message to the front desk — they are checking your check-in/check-out timing. 🙏\n\n"
                f"{ref_line}"
                "• *Payment Status:* ⏳ *PENDING VERIFICATION*\n"
                "• *Booking Status:* will be confirmed only after verification\n\n"
                "Payment will be confirmed once verified and we will send your confirmation on this chat."
            )

    if _CONFIRM_ASK_RE.search(msg):
        if is_hi:
            return (
                "Confirmation abhi isliye nahi bhej sakte ki aapka payment front "
                "desk verify kar rahi hai. 🙏\n\n"
                f"{ref_line}"
                "• *Current Status:* ⏳ *PENDING VERIFICATION*\n\n"
                "Verify hote hi official confirmation slip isi chat par aa jayegi. "
                "Aapko kuch aur karne ki zaroorat nahi hai."
            )
        else:
            return (
                "We cannot send the confirmation yet as our front desk is currently verifying your payment. 🙏\n\n"
                f"{ref_line}"
                "• *Current Status:* ⏳ *PENDING VERIFICATION*\n\n"
                "As soon as verified, your official confirmation slip will arrive on this chat. You do not need to do anything else."
            )

    return None


async def _escalate_if_stale(
    hotel_id: str,
    phone: str,
    booking_id: str | None,
    session_id: str | None,
    task: dict[str, Any] | None = None,
) -> float | None:
    """
    Escalate a payment that has been waiting past the threshold.

    Returns the hours waited when this call did the escalating (so the reply can
    say so), otherwise None. Latched by `payment_escalated_at` on the booking,
    or by `escalated_at` on the task: a guest who writes ten messages gets one
    escalation, not ten.

    Two clocks, because not every waiting guest has a booking row. The guest
    product flagged (919999000111) had none — they paid off a quote, before any
    booking document existed — so a booking-only clock would have done nothing
    for exactly the case that caused the complaint. When there is no booking,
    the open front-desk task's `created_at` is the start of the wait; it is
    written on the first claim turn, which is the same moment.

    Both artifacts are written in one direction only — nothing here can confirm
    a booking. The worst case is a task the front desk did not need.
    """
    from app.db.mongodb import bookings_collection

    if not phone and not booking_id:
        return None

    settings = get_settings()
    coll = bookings_collection()
    variants = [hotel_id, int(hotel_id)] if hotel_id.isdigit() else [hotel_id]
    flt_or: list[dict[str, Any]] = []
    if phone:
        flt_or.append({"customer_phone": phone})
    if booking_id:
        flt_or.append({"booking_id": booking_id})

    doc = await coll.find_one(
        {
            "hotel_id": {"$in": variants},
            "$or": flt_or,
            # A booking that has since been settled must not produce an
            # "overdue" escalation. Confirmation can arrive from a staff path
            # that leaves `payment_claimed_at` behind, and the front desk's
            # queue is not the place to discover that.
            "booking_status": {"$nin": TERMINAL_BOOKING_STATUSES},
            "stage": {"$nin": TERMINAL_STAGES},
        }
    )
    if doc and doc.get("payment_escalated_at"):
        return None

    now = datetime.now(timezone.utc)
    threshold = float(settings.PAYMENT_CHECK_ESCALATION_HOURS)

    if task is None:
        task = await task_repository.find_open_payment_task(
            hotel_id=hotel_id,
            booking_id=booking_id,
            session_id=session_id,
            customer_phone=phone,
        )

    if doc:
        claimed_at = doc.get("payment_claimed_at")
        if claimed_at is None:
            # Rows written before this feature have no start time. Start the
            # clock now rather than escalating a wait whose length nobody can
            # state — the guest's wait ends up longer, but the number staff
            # read is true.
            await coll.update_one({"_id": doc["_id"]}, {"$set": {"payment_claimed_at": now}})
            return None
        if not escalation_due(claimed_at, now, threshold):
            return None
        waited = hours_waiting(claimed_at, now)
    else:
        # No booking row: the task is the only record of this wait.
        if not task or task.get("escalated_at"):
            return None
        if not escalation_due(task.get("created_at"), now, threshold):
            return None
        waited = hours_waiting(task.get("created_at"), now)

    if task:
        await task_repository.escalate_payment_task(task["task_id"], waited)
    else:
        # Staff closed the task without the booking ever being confirmed. That
        # is exactly the case where the guest falls silent and nobody notices,
        # so the escalation raises a fresh task rather than a note on a closed
        # one.
        from app.workflows.nodes.stage_handlers import create_stage_task

        await create_stage_task(
            hotel_id=hotel_id,
            session_id=session_id or "",
            title="Payment Verification Overdue",
            description=(
                f"Payment verification has been pending {waited:.1f}h for "
                f"{phone or booking_id}. Guest has not been confirmed. "
                "Please verify the payment and confirm the booking."
            ),
            task_type="FRONT_DESK",
            subtype="PAYMENT_CONFIRMATION",
            department="FRONT_DESK",
            priority="URGENT",
            booking_id=booking_id,
            customer_phone=phone,
            metadata={"stage": "payment_check", "escalated": True},
        )

    if doc:
        await coll.update_one(
            {"_id": doc["_id"]},
            {"$set": {"payment_escalated_at": now}},
        )
    # With no booking row the latch is the task's own `escalated_at`, written by
    # `escalate_payment_task` — there is nowhere else to keep it.
    logger.warning(
        "STAGE_1B_PAYMENT_ESCALATED | hotel=%s phone=%s booking=%s waited=%.1fh",
        hotel_id, phone or "NONE", booking_id or "NONE", waited,
    )
    return waited


async def handle_stage_payment_check(state: ConversationState) -> ConversationState:
    hotel_id = state.get("hotel_id", 111111)
    str_hid = str(hotel_id)
    phone = state.get("customer_identifier") or ""
    user_msg = state.get("user_message") or ""
    from app.utils.booking_resume import is_real_booking_id, is_hindi_or_hinglish, get_booking_resume_nudge

    raw_booking_id = state.get("active_booking_id") or state.get("booking_id")
    has_real_ref = is_real_booking_id(raw_booking_id)
    booking_id = str(raw_booking_id).strip() if has_real_ref else ""
    is_hi = is_hindi_or_hinglish(state, user_msg)
    ref_line = f"• *Booking Reference:* *{booking_id}*\n" if has_real_ref else ""

    logger.info(
        "STAGE_1B_PAYMENT_CHECK_TRIGGERED | hotel=%s phone=%s booking_id=%s msg='%s'",
        str_hid, phone, booking_id or "NONE", user_msg[:60],
    )

    # 1. Ensure booking in DB is flagged as payment_proof_submitted and stays pending
    #
    # Only when this turn actually carries a claim. The write below sets
    # `claimed_paid: True`, and `stage_resolver` reads that field as proof that
    # the guest said they paid — so writing it on every turn the handler serves
    # pinned a guest who merely said "hi" to `payment_check` for good. The guest
    # is already in this stage; a turn that does not claim payment has nothing
    # to add to the booking record.
    _is_claim_turn = bool(
        state.get("payment_proof_submitted") or _CLAIM_RE.search(user_msg)
    )
    try:
        from app.config.settings import get_settings
        from app.db.mongodb import bookings_collection

        if get_settings().PAYMENT_CHECK_INTENT_ROUTING_ENABLED and not _is_claim_turn:
            raise _NoClaimOnThisTurn

        # With neither a booking reference nor a phone there is nothing to
        # identify the guest by, and the `$or` below becomes
        # `[{"customer_phone": ""}]` — which matches an arbitrary booking in
        # this hotel that happens to carry an empty phone field, and marks IT
        # as claimed and pending. Prit's #2: "booking_id=None with null payment
        # details". Ungated on purpose: this only ever prevents a write the
        # handler has no identity to justify, it never adds one.
        if not phone and not booking_id:
            logger.error(
                "STAGE_1B_NO_IDENTITY | hotel=%s session=%s — refusing to write a "
                "booking with neither a booking_id nor a phone",
                str_hid, state.get("session_id"),
            )
            raise _NoClaimOnThisTurn

        coll = bookings_collection()
        flt_or: list[dict[str, Any]] = [{"customer_phone": phone}]
        if booking_id:
            flt_or.append({"booking_id": booking_id})

        result = await coll.update_one(
            # NEVER demote a stay that is already settled. The `$set` below
            # writes `booking_status: pending_payment` and `stage: payment_check`,
            # and `stage_resolver` treats both as overriding everything else — so
            # without this guard a guest whose booking is CONFIRMED who says
            # "already paid" has their confirmation erased by the bot itself, and
            # every later turn resolves to payment_check. The bot can then no
            # longer see the booking it confirmed, and re-enters the payment flow.
            #
            # Guarding inside the FILTER keeps this a single atomic update: a
            # settled booking simply does not match, so nothing is written and
            # the row is left exactly as it was.
            {"hotel_id": {"$in": [str_hid, int(str_hid) if str_hid.isdigit() else str_hid]},
             "booking_status": {"$nin": TERMINAL_BOOKING_STATUSES},
             "stage": {"$nin": TERMINAL_STAGES},
             "$or": flt_or},
            {"$set": {
                "stage": "payment_check",
                "payment_check_pending": True,
                "payment_proof_submitted": True,
                "claimed_paid": True,
                "booking_status": "pending_payment",
                # The dashboard's own status field, kept in step with the
                # canonical one above — this is a direct write, so it does not
                # pass through `create_booking`.
                **status_aliases("pending_payment"),
            },
            # The guest has already paid and sent proof — this booking is now
            # waiting on a human, not on the guest. The TTL index (see
            # booking_repository.ensure_indexes) deletes any pending_payment
            # hold 300s past hold_expires_at, so leaving the field set would
            # erase the booking, the proof and the payment record while the
            # front desk is still verifying it.
            "$unset": {
                "hold_expires_at": "",
            },
            # When the wait started. `$min` rather than `$set`: every claim turn
            # re-runs this update, and `$set` would push the clock forward on
            # each one, so a guest who keeps talking could never reach the
            # escalation threshold. `$min` keeps the earliest claim.
            "$min": {
                "payment_claimed_at": datetime.now(timezone.utc),
            }}
        )

        # Matched nothing? That is not "already handled" — it means the claim
        # landed nowhere. Guest 919999000111: staff booked him in the hotel's
        # own PMS and sent a PDF voucher, so no row existed here, this update
        # wrote nothing, and the result was discarded — while eight
        # PAYMENT_CONFIRMATION tasks asked the front desk to verify a payment
        # against a booking that was never in our database.
        #
        # A row that exists but is terminal is a different thing: the guest is
        # already settled and this filter correctly protected him. Only a total
        # absence of a row is the orphan this reports.
        if (get_settings().PAYMENT_CLAIM_ORPHAN_TASK_ENABLED
                and getattr(result, "matched_count", 1) == 0):
            try:
                from app.services.orphan_payment_claim import booking_exists, handle_orphan_claim

                if not await booking_exists(hotel_id, phone, booking_id or None):
                    await handle_orphan_claim(
                        hotel_id=hotel_id,
                        session_id=state.get("session_id"),
                        phone=phone,
                        booking_id=booking_id or None,
                        guest_message=user_msg,
                    )
            except Exception as e:  # noqa: BLE001 — never fail the guest's turn for this
                logger.warning("ORPHAN_CLAIM_HANDLER_FAILED | %s", e)
    except _NoClaimOnThisTurn:
        logger.info(
            "STAGE_1B_NO_CLAIM_ON_TURN | booking=%s msg=%r — booking left untouched",
            booking_id or "NONE", user_msg[:60],
        )
    except Exception as e:
        logger.warning(f"Error marking booking pending in stage 1b: {e}")

    # 2. Raise Owner Task for front desk verification (idempotent: avoid spamming duplicate open tasks)
    #
    # This used to read `list_tasks(hotel_id=..., status="OPEN", limit=5)` and
    # filter in Python. Five tasks is the whole hotel's newest five, so on a
    # busy hotel the guest's own existing payment task was never in the window
    # and a second one was created — measured 2026-10-27 in the mirror: 16 phone
    # numbers carry more than one simultaneously-open payment task, one of them
    # six. Three pairs were created 1–5 seconds apart, one with a `PAY-…`
    # reference and its twin with none, which is this code and
    # `create_payment_confirmation_task` racing each other.
    #
    # `find_open_payment_task` asks the database the right question instead:
    # same hotel, same subtype, still open/assigned/in-progress, matching on
    # booking, session or phone. It also sees ASSIGNED tasks, which the old
    # OPEN-only filter did not — a task staff had picked up still counted as
    # absent.
    open_task: dict[str, Any] | None = None
    try:
        existing_task = await task_repository.find_open_payment_task(
            hotel_id=hotel_id,
            booking_id=booking_id if has_real_ref else None,
            session_id=state.get("session_id"),
            customer_phone=phone,
        )
        # Handed to the escalation step so a turn asks the database once, not
        # twice. A task raised below is deliberately not passed on — it is
        # seconds old and cannot be overdue.
        open_task = existing_task

        if not existing_task:
            from app.workflows.nodes.stage_handlers import create_stage_task
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Payment Verification Needed ({booking_id or 'New Hold'})",
                description=f"Guest submitted payment proof / claim: '{user_msg}'. Phone: {phone}. Verify transaction in bank / gateway.",
                task_type="FRONT_DESK",
                subtype="PAYMENT_CONFIRMATION",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id if has_real_ref else None,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "payment_check",
                    "task_type": "payment_check",
                },
            )
            logger.info("STAGE_1B_OWNER_TASK_RAISED | booking=%s", booking_id or "NONE")
    except Exception as e:
        # A warning in the log is not tracking: this turn still tells the guest
        # "front desk is verifying", so nothing downstream notices that no task
        # was ever raised. Put the swallow where ops can see it.
        from app.observability.failure_store import record_failure

        record_failure(
            component="task.create.stage_payment_check",
            error=e,
            hotel_id=hotel_id,
            session_id=state.get("session_id"),
            phone=phone,
            details={"booking_id": booking_id or None, "stage": "payment_check"},
            exc=e,
        )
        logger.warning(f"Error raising owner task in stage 1b: {e}")

    # 2c. Past the threshold, the wait stops being invisible. Runs after the
    # task step so there is a task to raise; a failure here must not cost the
    # guest their reply, so it is logged and the turn continues as before.
    escalated_hours: float | None = None
    if get_settings().PAYMENT_CHECK_ESCALATION_ENABLED:
        try:
            escalated_hours = await _escalate_if_stale(
                hotel_id=str_hid,
                phone=phone,
                booking_id=booking_id or None,
                session_id=state.get("session_id"),
                task=open_task,
            )
        except Exception as e:
            from app.observability.failure_store import record_failure

            record_failure(
                component="stage_payment_check.escalation",
                error=e,
                hotel_id=hotel_id,
                session_id=state.get("session_id"),
                phone=phone,
                details={"booking_id": booking_id or None},
                exc=e,
            )
            logger.warning(f"Error escalating stale payment check: {e}")

    # 3. Reassuring response matching Sir's exact requirements:
    # Thank guest, state front desk is verifying, keep pending, NO confirmed, NO upsell, NO receipt card
    if escalated_hours is not None:
        # Still no confirmation — the payment has not been verified. What
        # changed is that someone senior is now answerable for the wait.
        if is_hi:
            response_text = (
                "Namaste! 🙏 Aapka payment abhi bhi verify ho raha hai aur ismein "
                "2 ghante se zyada lag chuke hain.\n\n"
                f"{ref_line}"
                "• *Payment Status:* ⏳ *PENDING VERIFICATION*\n"
                "• *Escalated:* 📣 *Manager / senior front desk*\n\n"
                "Humne aapka case senior team ko escalate kar diya hai — wo aapko isi "
                "chat par jaldi update denge. Booking verification ke baad hi confirm hogi. 🙏"
            )
        else:
            response_text = (
                "Hello! 🙏 Your payment is still being verified and it has now been "
                "waiting more than 2 hours.\n\n"
                f"{ref_line}"
                "• *Payment Status:* ⏳ *PENDING VERIFICATION*\n"
                "• *Escalated:* 📣 *Manager / senior front desk*\n\n"
                "We have escalated your case to our senior team — they will update you "
                "on this chat shortly. The booking will be confirmed only after verification. 🙏"
            )
    elif is_hi:
        response_text = (
            "Namaste! 🙏 Humne aapka payment confirmation note kar liya hai.\n\n"
            "Humari front desk team abhi aapka payment verify kar rahi hai. "
            "Jaise hi bank/UPI account se verify hota hai, hum aapko official confirmation slip bhej denge.\n\n"
            f"{ref_line}"
            "• *Current Status:* ⏳ *PENDING VERIFICATION*\n\n"
            "Aapko kuch aur karne ki zaroorat nahi hai, kripya thoda intezaar karein."
        )
    else:
        response_text = (
            "Hello! 🙏 We have noted your payment confirmation.\n\n"
            "Our front desk team is currently verifying your payment. "
            "As soon as it is verified with the bank/UPI records, we will send your official confirmation slip.\n\n"
            f"{ref_line}"
            "• *Current Status:* ⏳ *PENDING VERIFICATION*\n\n"
            "You do not need to take any further action, please wait a moment."
        )

    # The paragraph above answers a payment claim. Every other turn in this
    # stage — a greeting, "send me my confirmation", "my check-in is in a few
    # hours" — used to get it verbatim too, which reads as a bot that is not
    # listening. These variants answer the question that was asked and state
    # the same unconfirmed status; `_status_reply` returns None for a payment
    # claim, so the wording Sir specified is untouched for the case it was
    # written for.
    # The escalation wording wins over the tailored variants: the one fact the
    # guest most needs is that a person is now answerable, and every variant
    # here would replace it with the plain "still pending" paragraph.
    if (
        escalated_hours is None
        and get_settings().PAYMENT_CHECK_STATUS_REPLIES_ENABLED
        and not _CLAIM_RE.search(user_msg)
    ):
        tailored = _status_reply(user_msg, booking_id, is_hi=is_hi)
        if tailored:
            response_text = tailored
        else:
            from app.workflows.nodes.stage_handlers.kb_answer import knowledge_base_answer
            kb_state = await knowledge_base_answer(
                state, stage="payment_check", log_tag="STAGE_1B_PAYMENT_CHECK",
            )
            if kb_state is not None:
                kb_resp = list(kb_state.get("bot_response") or [])
                if kb_resp and isinstance(kb_resp[0], dict) and kb_resp[0].get("text"):
                    nudge = get_booking_resume_nudge(state)
                    if nudge and nudge not in kb_resp[0]["text"]:
                        kb_resp[0]["text"] += f"\n\n{nudge}"
                return {
                    **kb_state,
                    "booking_stage": "payment_check",
                    "lifecycle_stage": "payment_check",
                    "bot_response": kb_resp,
                    "next_action": "awaiting_payment_verification",
                }

    return {
        **state,
        "booking_stage": "payment_check",
        "lifecycle_stage": "payment_check",
        "bot_response": text_card(response_text),
        "next_action": "awaiting_payment_verification",
    }
