"""Check payment status (gateway + manual approval) and handle paid outcomes."""

from __future__ import annotations

import logging
import re
from typing import cast

from app.config.hotel_config import HotelConfig
from app.models.activity_schemas import ActivityLog
from app.prompts.templates import payment_pending, payment_verified, text_card
from app.repositories import payment_receipt_repository
from app.repositories.activity_repository import log_activity
from app.services.feature_gate import FeatureGate
from app.services.payment_service import PaymentVerifyResult, verify_payment_status
from app.services import qr_payment_service
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

_AMOUNT_TOLERANCE = 10.0  # INR


async def _maybe_auto_upgrade(state: ConversationState, hotel_id: int) -> None:
    """Complimentary upgrade after full payment when feature is enabled."""
    config_dict = state.get("hotel_config")
    config = HotelConfig(**config_dict) if config_dict else HotelConfig(hotel_id=hotel_id)
    gate = FeatureGate(config, session_id=state.get("session_id"))

    if gate.can_auto_upgrade() and state.get("selected_rooms"):
        from app.services.inventory_service import get_available_rooms
        from app.services.allocation_engine import get_preferred_rate_plan

        avail_rooms = await get_available_rooms(hotel_id, state.get("check_in"), state.get("check_out"))

        booked_rooms = state.get("selected_rooms", [])
        highest_current_price = 0
        for br in booked_rooms:
            highest_current_price = max(highest_current_price, br.get("price_per_night", 0))

        target_upgrade = None
        for ar in avail_rooms:
            rp = get_preferred_rate_plan(ar, state.get("meal_plan_preference"))
            if rp and rp.base_price > highest_current_price and ar.number_of_rooms >= len(booked_rooms):
                if not target_upgrade or rp.base_price < target_upgrade["price"]:
                    target_upgrade = {"room": ar, "rate_plan": rp, "price": rp.base_price}

        if target_upgrade:
            for br in booked_rooms:
                br["room_id"] = target_upgrade["room"].room_id
                br["room_name"] = target_upgrade["room"].name
            state["selected_rooms"] = booked_rooms

            await log_activity(ActivityLog(
                hotel_id=str(hotel_id),
                activity_type="SYSTEM_EVENT",
                reference_id=state.get("booking_id") or "UNKNOWN",
                booking_id=state.get("booking_id") or "UNKNOWN",
                session_id=state.get("session_id"),
                description=f"UPGRADE_GRANTED: Complimentarily upgraded to {target_upgrade['room'].name}"
            ))
        else:
            await log_activity(ActivityLog(
                hotel_id=str(hotel_id),
                activity_type="SYSTEM_EVENT",
                reference_id=state.get("booking_id") or "UNKNOWN",
                booking_id=state.get("booking_id") or "UNKNOWN",
                session_id=state.get("session_id"),
                description="UPGRADE_SKIPPED: Higher Tier Unavailable or Multi-room Restriction"
            ))
    elif not gate.can_auto_upgrade():
        await log_activity(ActivityLog(
            hotel_id=str(hotel_id),
            activity_type="SYSTEM_EVENT",
            reference_id=state.get("booking_id") or "UNKNOWN",
            booking_id=state.get("booking_id") or "UNKNOWN",
            session_id=state.get("session_id"),
            description="UPGRADE_SKIPPED: Feature Disabled"
        ))


def _credited_amount(
    *,
    expected_link_amount: float,
    gateway_amount: float | None,
) -> tuple[float | None, str | None]:
    """
    Decide how much to credit.

    If gateway reports an amount, it must match expected (±tolerance).
    If gateway omits amount, credit the expected link amount.
    """
    if gateway_amount is None:
        return expected_link_amount, None

    # Backend may return paise (e.g. 10000 for ₹100)
    candidates = [gateway_amount]
    if gateway_amount >= 100 and abs(gateway_amount / 100.0 - expected_link_amount) <= _AMOUNT_TOLERANCE:
        candidates.append(gateway_amount / 100.0)

    for amt in candidates:
        if abs(float(amt) - expected_link_amount) <= _AMOUNT_TOLERANCE:
            return float(amt) if abs(float(amt) - expected_link_amount) > 0.01 else expected_link_amount, None

    return None, (
        f"Payment received (₹{gateway_amount:,.0f}) doesn't match the expected "
        f"amount for this link (₹{expected_link_amount:,.0f}). "
        f"Please contact the front desk for help."
    )


async def resolve_payment_status(
    state: ConversationState,
    *,
    hotel_id: int,
    total: float,
    amount_paid: float,
    amount_due: float,
    existing_link: str | None,
    intent: str,
    claims_paid: bool,
    payment_details,
) -> ConversationState | None:
    """
    Verify gateway / manual approval and return a terminal state if resolved.
    Returns None if payment is not yet verified and flow should continue.
    """
    is_paid = False
    expected_link_amount = float(
        state.get("payment_link_amount")
        or (amount_due if amount_due > 0 else total)
        or 0
    )
    link_amount = expected_link_amount
    payment_link_id = state.get("payment_link_id")
    qr_order_id = state.get("qr_order_id")
    qr_id = state.get("qr_id")

    if (existing_link or payment_link_id or qr_order_id or qr_id) and (
        intent == "PAYMENT_STATUS" or claims_paid
    ):
        verify = None
        if existing_link or payment_link_id:
            verify = await verify_payment_status(
                hotel_id,
                payment_link_id=payment_link_id,
                payment_link=existing_link,
            )
            if verify.error and not verify.is_paid:
                logger.warning(f"Payment verify error: {verify.error}")

        # The QR is a separate Razorpay order, and a guest may have paid it
        # while the link sits untouched. Ask about it too whenever the link did
        # not already answer "paid" — this is what makes a guest's "I've paid"
        # confirm on its own instead of queuing a front-desk task.
        if (verify is None or not verify.is_paid) and (qr_order_id or qr_id):
            qr_verify = await qr_payment_service.verify_guest_qr(
                hotel_id,
                order_id=qr_order_id,
                qr_id=qr_id,
            )
            if qr_verify.error and not qr_verify.is_paid:
                logger.warning(f"QR verify error: {qr_verify.error}")
            if qr_verify.is_paid:
                verify = PaymentVerifyResult(
                    is_paid=True,
                    amount_paid=qr_verify.amount_paid,
                    status=qr_verify.status,
                    raw=qr_verify.raw,
                )
                logger.info(
                    "QR_PAYMENT_VERIFIED | hotel=%s order=%s payment=%s",
                    hotel_id, qr_order_id or qr_id, qr_verify.payment_id,
                )

        if verify is not None and verify.is_paid:
            credited, amount_err = _credited_amount(
                expected_link_amount=expected_link_amount,
                gateway_amount=verify.amount_paid,
            )
            if amount_err:
                return cast(ConversationState, {
                    **state,
                    "next_action": "payment_pending",
                    "bot_response": text_card(amount_err),
                })
            is_paid = True
            link_amount = credited if credited is not None else expected_link_amount

    if not is_paid and not payment_details and (intent == "PAYMENT_STATUS" or claims_paid):
        latest_receipt = await payment_receipt_repository.get_latest_receipt_by_session(
            state.get("session_id")
        )
        if latest_receipt:
            is_approved = latest_receipt.get("is_approved")
            if is_approved is None:
                from app.services.task_engine import create_payment_confirmation_task
                receipt_details = latest_receipt.get("payment_details", {})
                task_res = await create_payment_confirmation_task(
                    dict(state),
                    payment_info={
                        "amount": receipt_details.get("amount") or link_amount,
                        "payment_mode": "Receipt Upload / OCR Review",
                        "payer_name": receipt_details.get("sender_name") or "From Receipt",
                        "utr_number": receipt_details.get("transaction_id") or "From Receipt",
                        "details": "Payment receipt uploaded requiring front desk verification",
                    },
                    source="receipt_review",
                )
                task_id = task_res.get("task_id", "")
                return cast(ConversationState, {
                    **state,
                    "next_action": "payment_pending",
                    "bot_response": text_card(
                        f"Your payment receipt is under review by our front desk team. ✅\n\n"
                        f"🔧 **Task ID:** {task_id}\n\n"
                        f"We will verify your payment and confirm your reservation shortly. 🙏"
                    ),
                })
            if is_approved is False:
                return cast(ConversationState, {
                    **state,
                    "next_action": "payment_pending",
                    "bot_response": text_card(
                        "Your payment was not approved. For further assistance, please contact the hotel front desk."
                    ),
                })
            if is_approved is True:
                is_paid = True
                receipt_details = latest_receipt.get("payment_details", {})
                if receipt_details.get("amount") is not None:
                    link_amount = float(receipt_details.get("amount"))

    if is_paid:
        new_amount_paid = amount_paid + link_amount
        new_amount_due = max(0, total - new_amount_paid)

        updated_state = {
            **state,
            "payment_link": None,
            "payment_link_id": None,
            "payment_link_amount": None,
            # Cleared with the link, and for the same reason: a QR order stays
            # `paid` forever, so leaving it on the session would let a second
            # "I've paid" verify the same payment again and credit the advance
            # twice. One payment, one credit.
            "qr_order_id": None,
            "qr_id": None,
            "qr_link": None,
            "amount_paid": new_amount_paid,
            "amount_due": new_amount_due,
        }

        if new_amount_due <= 0:
            await _maybe_auto_upgrade(state, hotel_id)

            await log_activity(ActivityLog(
                hotel_id=hotel_id,
                activity_type="PAYMENT_RECEIVED",
                reference_id=state.get("booking_id") or "UNKNOWN",
                booking_id=state.get("booking_id") or "UNKNOWN",
                customer_phone=state.get("customer_identifier"),
                description=f"Full payment of ₹{link_amount:,.0f} received."
            ))

            return cast(ConversationState, {
                **updated_state,
                "payment_status": "paid",
                "booking_stage": "confirmed",
                "next_action": "payment_verified",
                "bot_response": payment_verified(),
            })

        booking_id = state.get("booking_id", "UNKNOWN")
        await log_activity(ActivityLog(
            hotel_id=hotel_id,
            activity_type="PAYMENT_RECEIVED",
            reference_id=booking_id or "UNKNOWN",
            booking_id=booking_id or "UNKNOWN",
            customer_phone=state.get("customer_identifier"),
            description=f"Advance payment of ₹{link_amount:,.0f} received. Due: ₹{new_amount_due:,.0f}"
        ))

        return cast(ConversationState, {
            **updated_state,
            "payment_status": "partial",
            "booking_stage": "confirmed",
            "next_action": "payment_verified",
            "bot_response": text_card(
                f"✅ Advance payment of ₹{link_amount:,.0f} received! Thank you.\n\n"
                f"💰 **Payment Summary:**\n"
                f"• Total: ₹{total:,.0f}\n"
                f"• Paid: ₹{new_amount_paid:,.0f}\n"
                f"• Due: ₹{new_amount_due:,.0f}\n\n"
                f"Your booking is being confirmed. The remaining ₹{new_amount_due:,.0f} "
                f"can be paid at the time of check-in. 🏨"
            ),
        })

    if claims_paid:
        from app.services.task_engine import create_payment_confirmation_task
        user_msg = state.get("user_message", "")
        msg_lower = user_msg.lower()

        # Classify mode from user message
        mode = "Online Other Party / Other Modes"
        if any(w in msg_lower for w in ("brother", "bhai", "friend", "father", "wife", "husband", "other party", "another account", "different account", "other account", "relative")):
            mode = "Online Third-Party / Family Account"
        elif any(w in msg_lower for w in ("neft", "rtgs", "imps", "bank transfer", "net banking")):
            mode = "Direct Bank Transfer (NEFT/RTGS/IMPS)"
        elif any(w in msg_lower for w in ("cash", "counter", "reception")):
            mode = "Cash at Counter / Reception"
        elif any(w in msg_lower for w in ("qr", "scanner", "pos")):
            mode = "Offline QR Scan / POS"

        # Detect UTR or reference number if mentioned
        utr_match = re.search(r"\b(?:utr|rrn|txn|ref|transaction\s*id)?\s*[:#-]?\s*([0-9]{10,18}|[a-zA-Z0-9]{12,22})\b", user_msg, re.IGNORECASE)
        utr_num = utr_match.group(1) if utr_match else "Claimed in chat"

        task_res = await create_payment_confirmation_task(
            dict(state),
            payment_info={
                "amount": expected_link_amount,
                "payment_mode": mode,
                "utr_number": utr_num,
                "details": user_msg[:300],
            },
            source="guest_claims_paid",
        )
        task_id = task_res.get("task_id", "")

        return cast(ConversationState, {
            **state,
            "next_action": "payment_pending",
            "bot_response": text_card(
                f"Thank you for informing us! We have created a payment verification request for our front desk team. ✅\n\n"
                f"🔧 **Task ID:** {task_id}\n\n"
                f"Our team will verify your payment ({mode}) and confirm your reservation shortly. "
                f"If you have a payment screenshot or transaction ID, please share it here! 🙏"
            ),
        })

    if intent == "PAYMENT_STATUS" and existing_link:
        return cast(ConversationState, {
            **state,
            "next_action": "payment_pending",
            "bot_response": payment_pending(existing_link),
        })

    return None
