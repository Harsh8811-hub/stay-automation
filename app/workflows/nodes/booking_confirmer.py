"""
Booking Confirmer Node — creates the final booking after payment.
"""

import logging

from app.config.settings import get_settings
from app.prompts.templates import booking_confirmed, text_card
from app.repositories import booking_repository
from app.services.availability_guard import AvailabilityVerdict
from app.services.booking_engine import build_booking_document, generate_booking_id
from app.workflows.session_mapper import state_to_session
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


def _name_needed(state: ConversationState) -> bool:
    """Whether the slip has no real name on it.

    `payment/link.py` falls back to "Guest" when the guest never gave one, and
    `understand` writes "Guest" as a placeholder too — neither is a name.
    """
    name = str(state.get("guest_name") or "").strip()
    return not name or name.lower() == "guest"


def _ask_for_name_after_payment(bot_response: list[dict]) -> list[dict]:
    """Append the after-payment name ask to a confirmation reply.

    See `NAME_AFTER_PAYMENT_ENABLED`: the name is no longer asked before the
    payment link, so this is where we get it — on the turn the booking is
    created, when the guest already has a reason to answer.
    """
    return list(bot_response) + text_card(
        "One last thing — what name should we print on the confirmation slip? "
        "(as it appears on the ID for check-in)"
    )


def _clear_overrides_after_booking(state: ConversationState) -> dict:
    """Staff overrides and draft quote options end with this booking — must not leak to the next one."""
    from app.services.override_resolver import clear_overrides_in_state
    cleared = clear_overrides_in_state(dict(state), reason="booking_confirmed")
    cleared.update({
        "allocation_options": [],
        "payment_link": None,
        "payment_link_id": None,
        "payment_link_amount": None,
        "payment_check_pending": False,
        "payment_claim": None,
    })
    return cleared


async def _inventory_verdict(state: ConversationState) -> AvailabilityVerdict:
    """
    Re-check rooms are free before confirming (prevents post-hold-expiry overbook).

    Returns a verdict rather than a bool, because the caller's message says "the hold may
    have expired" — and it can only honestly say that if it knows the shortfall was real.
    The old bool could not tell a genuine shortfall from an availability lookup that came
    back empty or an id that never matched, which is the B2 defect in payment/link.py.
    """
    from app.services.availability_guard import (
        OK,
        AvailabilityVerdict,
        classify_availability,
    )

    selected_rooms = state.get("selected_rooms") or []
    if not selected_rooms:
        return AvailabilityVerdict(OK, "")

    booking_id = state.get("booking_id")
    existing = await booking_repository.get_booking(booking_id) if booking_id else None
    own_hold_active = False
    if existing and existing.get("booking_status") == "pending_payment":
        from datetime import datetime

        expires = existing.get("hold_expires_at")
        if expires is not None:
            # Motor may return naive UTC datetime
            now = datetime.utcnow()
            if getattr(expires, "tzinfo", None) is not None:
                expires = expires.replace(tzinfo=None)
            own_hold_active = expires > now

    # An active pending hold already reserves this inventory for us — allow confirm
    # without spending an availability lookup on it.
    if own_hold_active:
        return AvailabilityVerdict(OK, "own hold still active")

    from app.services.inventory_service import get_available_rooms

    available_rooms = await get_available_rooms(
        state.get("hotel_id", 0), state.get("check_in"), state.get("check_out")
    )
    return classify_availability(selected_rooms, available_rooms)


async def confirm_booking(state: ConversationState) -> ConversationState:
    """
    Create the confirmed booking in MongoDB.
    Only called AFTER payment is verified.
    """
    # A gateway-only payment never produced a screenshot, so this is the first
    # turn with proof of payment; mint the id here and carry the provisional
    # hold over to it. With BOOKING_ID_ON_PAYMENT_PROOF_ENABLED off this is the
    # identity function for a state that already carries an id.
    from app.services.booking_identity import mint_booking_id_on_payment_proof

    state, booking_id = await mint_booking_id_on_payment_proof(state)
    booking_id = booking_id or generate_booking_id()

    # Final inventory guard (especially after hold expiry)
    verdict = await _inventory_verdict(state)
    if not verdict.ok:
        from app.observability.metrics import metrics
        metrics.incr(
            "booking.confirm_total",
            outcome="inventory_unavailable",
            hotel_id=state.get("hotel_id"),
        )
        logger.warning(
            f"Confirm blocked after payment — {verdict.status}: {verdict.reason} "
            f"(booking={booking_id})"
        )
        # This guest has already paid. Until now nothing was raised and no one was told:
        # the reply promised "our team will contact you" with no ticket behind it.
        from app.services.availability_guard import (
            escalation_updates,
            raise_availability_ticket,
        )

        await raise_availability_ticket(state, verdict)
        return {
            **state,
            "booking_id": booking_id,
            "booking_stage": "collecting",
            # Kept, not cleared: staff resolving this need to know what was booked, and
            # the guest needs to be able to retry once the real cause is known.
            "payment_link": None,
            "payment_link_id": None,
            "payment_link_amount": None,
            "bot_response": text_card(
                "We received your payment, but I could not confirm the room "
                "automatically just now (the hold may have expired). I've flagged this to "
                "our team and they will contact you shortly to resolve it — please keep "
                f"your Booking ID **{booking_id}** handy. We're very sorry for the "
                "inconvenience."
            ),
            "next_action": "rooms_unavailable",
            **escalation_updates(verdict),
        }

    # Check if booking exists (it should, as a pending hold)
    existing_booking = await booking_repository.get_booking(booking_id)
    if existing_booking:
        current_status = existing_booking.get("booking_status")
        if current_status in ("confirmed", "partial", "cancelled"):
            logger.info(f"Booking {booking_id} is already {current_status}. Skipping creation.")
            return {
                **state,
                "booking_id": booking_id,
                "booking_stage": current_status,
                "bot_response": booking_confirmed(
                    booking_id=booking_id,
                    check_in=state.get("check_in", ""),
                    check_out=state.get("check_out", ""),
                ),
                "next_action": "confirmed",
            }

        total = float(state.get("total_price") or 0.0)
        amount_paid = float(state.get("amount_paid") or 0.0)
        if total > 0 and amount_paid > 0:
            amount_due = max(0.0, total - amount_paid)
        else:
            amount_due = float(state.get("amount_due") or 0.0)
        payment_status = "paid" if amount_due <= 1.0 else "partial" 

        success = await booking_repository.confirm_pending_booking(
            booking_id, payment_status, amount_paid, amount_due
        )
        if success:
            logger.info(f"Confirmed pending booking {booking_id}")
            bot_response = booking_confirmed(
                booking_id=booking_id,
                check_in=state.get("check_in", ""),
                check_out=state.get("check_out", ""),
            )
            if payment_status == "partial" and amount_due > 0:
                total = state.get("total_price", 0)
                bot_response += text_card(
                    f"💰 **Payment Summary:**\n"
                    f"• Total: ₹{total:,.0f}\n"
                    f"• Advance Paid: ₹{amount_paid:,.0f}\n"
                    f"• Due at Check-in: ₹{amount_due:,.0f}"
                )
            
            # ── MULTI-BOOKING QUEUE CONTINUATION ──
            from app.utils.queue_manager import advance_booking_queue
            q_updates = advance_booking_queue(state, "confirmed")
            if q_updates.get("queue_advanced"):
                bot_response += q_updates.pop("bot_response_prepend", [])

            ask_for_name = get_settings().NAME_AFTER_PAYMENT_ENABLED and _name_needed(state)
            if ask_for_name:
                bot_response = _ask_for_name_after_payment(bot_response)

            return {
                **state,
                "booking_id": booking_id,
                "booking_stage": "confirmed",
                "bot_response": bot_response,
                "next_action": "confirmed",
                **({"name_after_payment_pending": True} if ask_for_name else {}),
                **_clear_overrides_after_booking(state),
                **q_updates
            }

    # No pending hold — create a fresh confirmed booking
    # A row with no name at all reads as blank in the CRM. The placeholder is
    # what the guest is asked to replace on this same turn (NAME_AFTER_PAYMENT).
    if not state.get("guest_name"):
        state = {**state, "guest_name": "Guest"}

    session = state_to_session(
        state,
        booking_id=booking_id,
        payment_status=state.get("payment_status", "paid"),
    )

    booking_doc = build_booking_document(session)
    await booking_repository.create_booking(booking_doc)

    total = float(state.get("total_price") or 0.0)
    amount_paid = float(state.get("amount_paid") or 0.0)
    if total > 0 and amount_paid > 0:
        amount_due = max(0.0, total - amount_paid)
    else:
        amount_due = float(state.get("amount_due") or 0.0)
    payment_status = "paid" if amount_due <= 1.0 else "partial" 

    bot_response = booking_confirmed(
        booking_id=booking_id,
        check_in=state.get("check_in", ""),
        check_out=state.get("check_out", ""),
    )

    if payment_status == "partial" and amount_due > 0:
        total = state.get("total_price", 0)
        bot_response += text_card(
            f"💰 **Payment Summary:**\n"
            f"• Total: ₹{total:,.0f}\n"
            f"• Advance Paid: ₹{amount_paid:,.0f}\n"
            f"• Due at Check-in: ₹{amount_due:,.0f}"
        )

    # ── MULTI-BOOKING QUEUE CONTINUATION ──
    from app.utils.queue_manager import advance_booking_queue
    q_updates = advance_booking_queue(state, "confirmed")
    if q_updates.get("queue_advanced"):
        bot_response += q_updates.pop("bot_response_prepend", [])

    ask_for_name = get_settings().NAME_AFTER_PAYMENT_ENABLED and _name_needed(state)
    if ask_for_name:
        bot_response = _ask_for_name_after_payment(bot_response)

    return {
        **state,
        "booking_id": booking_id,
        "booking_stage": "confirmed",
        "payment_status": payment_status,
        "bot_response": bot_response,
        "next_action": "confirmed",
        **({"name_after_payment_pending": True} if ask_for_name else {}),
        **_clear_overrides_after_booking(state),
        **q_updates
    }
