"""Validate uploaded payment screenshots / OCR details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from app.config.hotel_config import HotelConfig
from app.prompts.templates import payment_pending, text_card
from app.repositories import payment_receipt_repository
from app.services.feature_gate import FeatureGate
from app.workflows.nodes.payment.constants import SUCCESS_STATUSES
from app.workflows.state import ConversationState


@dataclass
class ReceiptValidation:
    state: ConversationState
    total: float
    amount_due: float
    screenshot_valid: bool = False
    invalid_payment_msg: str | None = None
    early_return: ConversationState | None = None


async def validate_payment_receipt(
    state: ConversationState,
    payment_details: dict,
    *,
    hotel_id: int,
    total: float,
    amount_due: float,
    existing_link: str | None,
) -> ReceiptValidation:
    """
    Validate an uploaded payment screenshot.

    May return early_return for hard stops (gate blocked, duplicate, etc.).
    Otherwise sets screenshot_valid / invalid_payment_msg for the orchestrator.
    """
    config_dict = state.get("hotel_config")
    config = HotelConfig(**config_dict) if config_dict else HotelConfig(hotel_id=hotel_id)
    gate = FeatureGate(config, session_id=state.get("session_id"))

    pv_check = gate.check_payment_verification()
    if not pv_check.allowed:
        return ReceiptValidation(
            state=state,
            total=total,
            amount_due=amount_due,
            early_return=cast(ConversationState, {
                **state,
                "next_action": "payment_pending",
                "bot_response": text_card(pv_check.response),
                "payment_details": None,
            }),
        )

    status = payment_details.get("status", "").upper()
    notes = payment_details.get("notes") or []
    txn_id = payment_details.get("transaction_id")
    ocr_booking_id = payment_details.get("booking_id")
    ocr_customer_phone = payment_details.get("customer_phone")

    from app.repositories import booking_repository

    booking_id_to_lookup = ocr_booking_id or state.get("booking_id")
    customer_phone_to_lookup = ocr_customer_phone or state.get("customer_identifier")

    booking_record = None
    if booking_id_to_lookup:
        booking_record = await booking_repository.get_booking(booking_id_to_lookup)
    elif customer_phone_to_lookup:
        booking_record = await booking_repository.get_latest_booking(
            hotel_id=state.get("hotel_id") or 0,
            customer_phone=customer_phone_to_lookup,
            booking_status=booking_repository.ACTIVE_BOOKING_STATUSES,
        )

    if not booking_record and not (total > 0):
        return ReceiptValidation(
            state=state,
            total=total,
            amount_due=amount_due,
            early_return=cast(ConversationState, {
                **state,
                "next_action": "need_info",
                "bot_response": text_card(
                    "I received your payment receipt, but I couldn't find a Customer ID, "
                    "Phone Number, or Booking ID on it to locate your reservation. "
                    "Could you please provide your Booking ID or Phone Number?"
                ),
            }),
        )

    if booking_record:
        total = booking_record.get("total_price", total)
        amount_due = booking_record.get("amount_due", amount_due)
        # An unrenamed hold is a row, not a booking. Adopting its `PAY-…` id
        # would hand the guest a reference that is not their booking id and
        # stop `mint_booking_id_on_payment_proof` from minting the real one.
        # (`payment_reference` survives the rename as provenance, so the id
        # matching it — not the field merely being present — is the test.)
        unrenamed_hold = (
            booking_record.get("booking_status") == "pending_payment"
            and booking_record.get("booking_id") == booking_record.get("payment_reference")
        )
        if not unrenamed_hold:
            state["booking_id"] = booking_record.get("booking_id", state.get("booking_id"))

    expected_amount = amount_due if amount_due > 0 else total
    screenshot_valid = False
    invalid_payment_msg = None

    if status == "INVALID":
        reasons = ". ".join(notes) if notes else "verification failure"
        if total > 0:
            invalid_payment_msg = (
                f"The uploaded screenshot appears to be invalid or altered ({reasons}). "
                f"Please upload a valid payment receipt of ₹{expected_amount:,.0f} to confirm your room booking."
            )
        else:
            invalid_payment_msg = (
                f"The uploaded image does not appear to be a valid payment receipt for our hotel ({reasons}). "
                f"If you are looking to book a room with us, please let me know your check-in dates and "
                f"room preferences, and I'll find the best options for you! 🏨"
            )
    elif status in SUCCESS_STATUSES:
        amount = payment_details.get("amount")

        if txn_id:
            is_duplicate = await payment_receipt_repository.check_duplicate_transaction(txn_id)
            if is_duplicate:
                return ReceiptValidation(
                    state=state,
                    total=total,
                    amount_due=amount_due,
                    early_return=cast(ConversationState, {
                        **state,
                        "next_action": "payment_pending",
                        "bot_response": text_card(
                            "This payment screenshot/transaction ID has already been submitted and processed. "
                            "Please upload a new and valid payment receipt."
                        ),
                    }),
                )

        if total > 0 and amount is None:
            return ReceiptValidation(
                state=state,
                total=total,
                amount_due=amount_due,
                early_return=cast(ConversationState, {
                    **state,
                    "next_action": "need_info",
                    "bot_response": text_card(
                        "I see your payment receipt, but I can't read the amount clearly. "
                        "Could you please confirm the amount paid?"
                    ),
                }),
            )

        if total > 0 and amount is not None and (abs(float(amount) - expected_amount) <= 10.0):
            screenshot_valid = True
            await payment_receipt_repository.save_payment_receipt({
                "session_id": state.get("session_id"),
                "hotel_id": hotel_id,
                "customer_identifier": state.get("customer_identifier", ""),
                "payment_details": payment_details,
                "is_approved": None,
            })
        elif total == 0:
            await payment_receipt_repository.save_payment_receipt({
                "session_id": state.get("session_id"),
                "hotel_id": hotel_id,
                "customer_identifier": state.get("customer_identifier", ""),
                "payment_details": payment_details,
            })
            return ReceiptValidation(
                state=state,
                total=total,
                amount_due=amount_due,
                early_return=cast(ConversationState, {
                    **state,
                    "next_action": "need_info",
                    "bot_response": text_card(
                        f"I see a successful transaction of ₹{amount if amount else 'some amount'} on this receipt. "
                        f"However, you don't have an active room booking with us yet. "
                        f"Would you like to book a room? Please share your check-in dates and room preferences, "
                        f"and I'll check the best available rooms for you! 🏨"
                    ),
                }),
            )
        else:
            invalid_payment_msg = (
                f"I see you uploaded a payment screenshot, but the amount (₹{amount}) doesn't match "
                f"your expected payment (₹{expected_amount:,.0f}). Please ensure you've paid the correct "
                f"amount or contact the front desk."
            )
    else:
        invalid_payment_msg = (
            "The payment screenshot you uploaded doesn't show a successful status. "
            "Please ensure the payment is complete."
        )

    if invalid_payment_msg and existing_link:
        return ReceiptValidation(
            state=state,
            total=total,
            amount_due=amount_due,
            early_return=cast(ConversationState, {
                **state,
                "next_action": "payment_pending",
                "bot_response": payment_pending(existing_link, prepend_text=invalid_payment_msg),
            }),
        )

    if invalid_payment_msg and (state.get("total_price") or 0) <= 0:
        return ReceiptValidation(
            state=state,
            total=total,
            amount_due=amount_due,
            early_return=cast(ConversationState, {
                **state,
                "next_action": "need_info",
                "bot_response": text_card(invalid_payment_msg),
            }),
        )

    # No existing link but have a quote — continue so a link can be generated,
    # with invalid_payment_msg prepended later.
    return ReceiptValidation(
        state=state,
        total=total,
        amount_due=amount_due,
        screenshot_valid=screenshot_valid,
        invalid_payment_msg=invalid_payment_msg,
    )
