import logging
import re
from datetime import datetime

from app.repositories import booking_repository
from app.workflows.state import ConversationState
from app.prompts.templates import button_card, text_card
from app.rag.chroma_service import query_knowledge
from app.services.lifecycle_manager import (
    check_human_takeover,
    invalidate_payment_links,
    log_lifecycle_event,
)

logger = logging.getLogger(__name__)


def check_cancellation_confirmation(message: str) -> str | None:
    """Analyze the user's message to check if they confirm or abort cancellation."""
    message_clean = message.strip().lower()
    
    # Check abort first to prevent false matches (e.g. "don't cancel" containing "cancel")
    abort_patterns = [
        r"\bno\b", r"\bkeep\b", r"\bdo(?:n't|nt)\s+cancel\b", r"\bdo\s+not\s+cancel\b",
        r"\babort\b", r"\bnever\s*mind\b", r"\bback\b", r"\bkeep\s+it\b",
        r"\bno\s+keep\b", r"\bno\s+don't\b", r"\bno\s+dont\b"
    ]
    for pattern in abort_patterns:
        if re.search(pattern, message_clean):
            return "abort"
            
    confirm_patterns = [
        r"\byes\b", r"\byep\b", r"\byeah\b", r"\by\b", r"\bproceed\b",
        r"\bcancel\b", r"\bcancel\s+it\b", r"\bconfirm\b", r"\bsure\b",
        r"\bok\b", r"\bokay\b", r"\bdo\s+it\b"
    ]
    for pattern in confirm_patterns:
        if re.search(pattern, message_clean):
            return "confirm"
            
    return None


async def handle_cancellation(state: ConversationState) -> ConversationState:
    """
    Handle booking cancellation with a multi-step confirmation flow.
    
    Phase 1: Booking identification and warning/policy display.
    Phase 2: Confirmation or abortion of the cancellation.
    """
    user_message = state.get("user_message", "")
    booking_stage = state.get("booking_stage", "new")
    hotel_id = state.get("hotel_id", 0)

    # ── Phase 2: Confirmation / Abort Check ──────────────────────────
    if booking_stage == "cancellation_pending":
        booking_id = state.get("booking_id")
        if not booking_id:
            # Fallback if stage got desynced
            return {
                **state,
                "booking_stage": "new",
                "bot_response": text_card("I couldn't find a pending booking to cancel. Could you please provide your booking ID?"),
                "next_action": "need_booking_id"
            }

        confirm_status = check_cancellation_confirmation(user_message)

        if confirm_status == "confirm":
            # Invalidate any open payment links before cancelling
            payment_clears = invalidate_payment_links(state)
            # Actually cancel in DB
            updated = await booking_repository.cancel_booking(booking_id)
            if updated:
                await log_lifecycle_event(
                    "BOOKING_CANCELLED",
                    state,
                    description=f"Booking {booking_id} cancelled by guest.",
                    metadata={"booking_id": booking_id},
                )
                bot_response = text_card(f"Your booking **{booking_id}** has been successfully cancelled. ❌ If a refund is applicable, it will be processed according to our cancellation policy.")
                
                # ── MULTI-BOOKING QUEUE CONTINUATION ──
                from app.utils.queue_manager import advance_booking_queue
                q_updates = advance_booking_queue(state, "cancelled")
                if q_updates.get("queue_advanced"):
                    bot_response += q_updates.pop("bot_response_prepend", [])
                    
                from app.services.override_resolver import clear_overrides_in_state
                return {
                    **state,
                    "booking_stage": "cancelled",
                    "bot_response": bot_response,
                    "next_action": "booking_cancelled",
                    **clear_overrides_in_state(dict(state), reason="booking_cancelled"),
                    **q_updates
                }
            else:
                return {
                    **state,
                    "bot_response": text_card(f"I was unable to cancel booking **{booking_id}**. It may already be cancelled or not exist in our database."),
                    "next_action": "cancellation_failed"
                }

        elif confirm_status == "abort":
            # Abort and restore to confirmed stage
            await log_lifecycle_event(
                "CANCELLATION_ABORTED",
                state,
                description=f"Guest declined cancellation for booking {booking_id}.",
            )
            bot_response_str = f"Okay, your booking **{booking_id}** remains active and has not been cancelled. \u2705 Let me know if you need any other assistance!"
            return {
                **state,
                "booking_stage": "confirmed",
                "bot_response": text_card(bot_response_str),
                "next_action": "cancellation_aborted"
            }

        else:
            # Ambiguous response
            bot_response_str = (
                f"I'm not sure if you want to proceed. Please reply with **'Yes'** to cancel booking **{booking_id}**, "
                f"or **'No'** to keep it active."
            )
            return {
                **state,
                "bot_response": text_card(bot_response_str),
                "next_action": "need_cancellation_confirmation"
            }

    # ── Phase 1: Initial Request & Booking Retrieval ─────────────
    
    # ── Check human takeover first — block cancellation ops if active ────────────
    under_takeover = await check_human_takeover(state)
    if under_takeover:
        await log_lifecycle_event(
            "LIFECYCLE_BLOCKED",
            state,
            description="Cancellation blocked: human takeover active",
        )
        msg = (
            "A staff member is currently assisting you. "
            "Cancellations will resume once the session is handed back to me.\n\n"
            "If you need immediate help, please let the staff member know directly."
        )
        return {
            **state,
            "bot_response": text_card(msg),
            "next_action": "lifecycle_blocked_takeover",
        }

    booking_id = None

    # Try regex match on message
    match = re.search(r'(BK-\d{8}-[A-Z0-9]+)', user_message, re.IGNORECASE)
    if match:
        booking_id = match.group(1).upper()
        logger.info(f"Extracted booking ID from message: {booking_id}")

    # Fall back to session booking_id
    if not booking_id:
        booking_id = state.get("booking_id")
        if booking_id:
            logger.info(f"Using booking ID from session: {booking_id}")

    # Fall back to latest booking by phone
    if not booking_id:
        phone = state.get("customer_identifier") or state.get("guest_phone")
        if phone:
            booking = await booking_repository.get_latest_booking(
                hotel_id=hotel_id,
                customer_phone=phone,
                booking_status=booking_repository.ACTIVE_BOOKING_STATUSES,
            )
            if booking:
                booking_id = booking.get("booking_id")
                logger.info(f"Found latest active booking: {booking_id} for phone: {phone}")

    if not booking_id:
        return {
            **state,
            "bot_response": text_card("I couldn't find a booking to cancel. Could you please provide your booking ID (e.g., BK-YYYYMMDD-XXXXXX)?"),
            "next_action": "need_booking_id"
        }

    # Check database for booking details and status
    booking = await booking_repository.get_booking(booking_id)
    if not booking:
        return {
            **state,
            "bot_response": text_card(f"I couldn't find a booking with ID **{booking_id}** to cancel. Could you please check the ID and try again?"),
            "next_action": "invalid_booking_id"
        }

    # ── SECURITY: Authorization Check (IDOR Prevention) ──
    # Ensure the user requesting cancellation actually owns the booking.
    booking_phone = booking.get("customer_phone")
    session_phone = state.get("customer_identifier")
    
    # If both exist and they don't match, deny access.
    # (In a real system, you might format the phones first, e.g., strip '+91')
    if booking_phone and session_phone:
        # Basic normalization for comparison
        clean_b_phone = booking_phone.replace("+", "").replace(" ", "").strip()
        clean_s_phone = session_phone.replace("+", "").replace(" ", "").strip()
        
        # If the session phone doesn't match the booking phone (and isn't a substring just in case of country code mismatches)
        if clean_s_phone not in clean_b_phone and clean_b_phone not in clean_s_phone:
            logger.warning(f"SECURITY ALERT: Unauthorized cancellation attempt. Session {session_phone} tried to cancel booking {booking_id} owned by {booking_phone}.")
            return {
                **state,
                "bot_response": text_card("⚠️ Security Alert: You are not authorized to modify this booking. The phone number on the booking does not match your current phone number."),
                "next_action": "unauthorized_access"
            }

    current_status = booking.get("booking_status", "").lower()
    if current_status == "cancelled":
        return {
            **state,
            "booking_stage": "cancelled",
            "bot_response": text_card(f"Your booking **{booking_id}** is already cancelled. ❌"),
            "next_action": "already_cancelled"
        }

    # Policy text comes from the one shared reader (2026-10-10). This handler
    # and the confirmed-stage canceller must never quote different percentages:
    # a refund table with two implementations is one fix away from disagreeing
    # with itself in front of a guest who is about to lose money.
    from app.services.cancellation_policy_service import get_cancellation_policy_parts

    cancel_policy, refund_policy = await get_cancellation_policy_parts(hotel_id)

    # Show warning and policies, and ask for confirmation
    warning_str = (
        f"⚠️ **Cancellation Request Received**\n\n"
        f"You have requested to cancel your booking **{booking_id}**.\n\n"
    )
    
    # Mention pending modifications if any exist
    if state.get("requested_values") or state.get("modification_metadata"):
        warning_str += "🟡 *Any pending changes to this booking will also be discarded.*\n\n"
        
    warning_str += (
        f"🔴 *Please note that cancellation is irreversible.*\n\n"
        f"📋 **Cancellation Policy:**\n"
        f"{cancel_policy}\n\n"
        f"💳 **Refund Policy:**\n"
        f"{refund_policy}\n\n"
        f"Do you have any query regarding the refund, or would you like to proceed with the cancellation?\n\n"
        f"• Reply with **'Yes, cancel'** to proceed.\n"
        f"• Reply with **'No, keep booking'** to abort."
    )

    await log_lifecycle_event(
        "BOOKING_CANCEL_REQUESTED",
        state,
        description=f"Guest initiated cancellation flow for booking {booking_id}.",
        metadata={"booking_id": booking_id},
    )

    return {
        **state,
        "booking_id": booking_id,
        "booking_stage": "cancellation_pending",
        "bot_response": button_card(
            warning_str,
            [
                {"id": "cancel_confirm", "title": "Yes, cancel"},
                {"id": "cancel_abort", "title": "No, keep booking"},
            ],
        ),
        "next_action": "warned_and_awaiting_confirmation"
    }
