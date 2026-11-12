"""
Grounded Business Logic Validators for StayInn Vision Pipeline.

Validates:
1. Payment Proof: against active booking amount, duplicate UTR in DB, and tamper detection.
2. Room Reference: against MongoDB room catalog, matching beds/amenities without hallucination.
3. OTA Price Match: against hotel owner discount policy, seasonal limits, and hard floor price.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.activity_schemas import ActivityLog
from app.repositories import payment_receipt_repository, room_repository
from app.repositories.activity_repository import log_activity
from app.repositories.season_repository import get_season_negotiation_limit
from app.services import price_match_service
from app.services.vision.schemas import (
    BookingNoteExtracted,
    IDCardExtracted,
    OTAPriceExtracted,
    PaymentProofExtracted,
    RoomReferenceExtracted,
    ValidationResult,
    VisionAction,
)

logger = logging.getLogger(__name__)

AMOUNT_TOLERANCE = 10.0  # ₹10 tolerance


# ═════════════════════════════════════════════════════════════════════════════
# 1. PAYMENT PROOF VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

async def validate_payment(
    payment_data: PaymentProofExtracted,
    session_state: dict[str, Any],
    hotel_id: str | int,
) -> tuple[ValidationResult, VisionAction]:
    """
    Validates an extracted payment screenshot against current session/booking state.
    Enforces duplicate UTR prevention, multi-payment/partial payment handling,
    grounded MongoDB booking lookup, and Front Desk task escalation for ambiguous/3rd-party receipts.
    """
    from app.repositories import booking_repository
    from app.services.task_engine import create_payment_confirmation_task

    session_id = session_state.get("session_id", "")
    customer_phone = session_state.get("customer_identifier") or session_state.get("guest_phone", "")
    booking_id = session_state.get("booking_id")
    guest_name = session_state.get("guest_name") or session_state.get("customer_name") or "Guest"

    # Grounded Booking Lookup from MongoDB
    booking_record = None
    if booking_id:
        booking_record = await booking_repository.get_booking(str(booking_id))
    if not booking_record and customer_phone:
        booking_record = await booking_repository.get_latest_booking(
            hotel_id=hotel_id,
            customer_phone=customer_phone,
            booking_status=booking_repository.ACTIVE_BOOKING_STATUSES,
        )

    if booking_record:
        booking_id = booking_record.get("booking_id", booking_id)
        guest_name = booking_record.get("guest_name") or guest_name
        total_price = float(booking_record.get("total_price") or session_state.get("total_price") or 0.0)
        already_paid = float(booking_record.get("amount_paid") or session_state.get("amount_paid") or 0.0)
        if booking_record.get("amount_due") is not None:
            existing_amount_due = float(booking_record.get("amount_due"))
        else:
            existing_amount_due = max(0.0, total_price - already_paid)
    else:
        total_price = float(session_state.get("total_price") or 0.0)
        already_paid = float(session_state.get("amount_paid") or 0.0)
        if session_state.get("amount_due") is not None:
            existing_amount_due = float(session_state.get("amount_due"))
        else:
            existing_amount_due = max(0.0, total_price - already_paid)

    expected_amount = existing_amount_due if existing_amount_due > 0 else total_price
    utr = payment_data.transaction_id
    extracted_amt = payment_data.amount
    sender_name = payment_data.sender_name or ""
    payer_display = sender_name if sender_name else (guest_name or "Guest")
    payment_app = payment_data.payment_app or "UPI"

    # 1. Check for genuine visual tampering (not mere WhatsApp compression or camera photo of screen)
    if payment_data.is_suspicious or payment_data.status == "INVALID":
        reasons = "; ".join(payment_data.notes) if payment_data.notes else "Visual tampering or invalid payment structure detected"
        logger.warning("PAYMENT_TAMPER_FLAGGED | sess=%s | reasons=%s | utr=%s", session_id, reasons, utr)

        # If there is a valid UTR or recognized bank app, route to Front Desk task instead of a hostile rejection
        if utr or payment_data.payment_app:
            payment_info = {
                "booking_id": booking_id,
                "amount": extracted_amt or expected_amount,
                "payment_mode": f"{payment_app} (Manual Review Required)",
                "payer_name": payer_display,
                "transaction_id": utr,
                "details": f"Payment screenshot flagged for review ({reasons}). Front desk verification required.",
            }
            await create_payment_confirmation_task(
                state={
                    **session_state,
                    "hotel_id": hotel_id,
                    "booking_id": booking_id,
                    "guest_phone": customer_phone,
                    "guest_name": guest_name,
                },
                payment_info=payment_info,
                source="vision_tamper_fallback",
            )
            return (
                ValidationResult(
                    status="escalate",
                    reason=f"Payment screenshot routed for manual review ({reasons})",
                    discrepancies=[reasons],
                ),
                VisionAction(
                    type="escalate",
                    message=(
                        f"Thank you for sharing your payment receipt! 🙏 We've received your transfer details "
                        f"(Ref: {utr or 'Pending'}). Our front desk team has been notified to verify the transaction "
                        f"with our bank records and confirm your booking shortly."
                    ),
                    escalation_reason=f"Payment screenshot needs manual verification: {reasons}",
                ),
            )

        # Non-payment image or completely blank/invalid document
        return (
            ValidationResult(
                status="rejected",
                reason=f"Payment screenshot failed validation ({reasons})",
                discrepancies=[reasons],
            ),
            VisionAction(
                type="ask_clarification",
                message=(
                    "The uploaded image could not be verified as a valid bank payment receipt. "
                    "Please upload an official payment receipt or screenshot from your UPI/banking app showing the completed transfer."
                ),
                escalation_reason="Invalid payment screenshot uploaded",
            ),
        )

    # 2. Check for failed or declined payment status
    if payment_data.status in ("FAILED", "DECLINED"):
        return (
            ValidationResult(
                status="rejected",
                reason="Payment screenshot displays a FAILED or DECLINED transaction status.",
            ),
            VisionAction(
                type="ask_clarification",
                message="The payment screenshot you uploaded indicates that the transaction failed. Please retry the payment and share the successful receipt.",
            ),
        )

    # 3. Bank processing or pending
    if payment_data.status == "PENDING":
        if utr:
            payment_info = {
                "booking_id": booking_id,
                "amount": extracted_amt or expected_amount,
                "payment_mode": f"{payment_app} (Bank Processing)",
                "payer_name": payer_display,
                "transaction_id": utr,
                "details": f"Transaction {utr} marked as processing/pending by bank.",
            }
            await create_payment_confirmation_task(
                state={
                    **session_state,
                    "hotel_id": hotel_id,
                    "booking_id": booking_id,
                    "guest_phone": customer_phone,
                    "guest_name": guest_name,
                },
                payment_info=payment_info,
                source="vision_bank_pending",
            )
        return (
            ValidationResult(
                status="needs_clarification",
                reason="Payment is still processing / pending at the bank.",
            ),
            VisionAction(
                type="ask_clarification",
                message=(
                    f"Your payment of ₹{extracted_amt:,.0f} (Ref: {utr or 'Pending'}) is currently marked as processing by your bank. "
                    f"Our front desk has noted this, and once the bank completes the transfer, we will confirm your reservation!"
                ) if extracted_amt else (
                    "Your payment is currently marked as pending by your bank. Once the transaction completes successfully, please share the updated confirmation screenshot."
                ),
            ),
        )

    # 4. Duplicate UTR / Transaction ID Check
    if utr:
        is_duplicate = await payment_receipt_repository.check_duplicate_transaction(utr)
        if is_duplicate:
            logger.warning("DUPLICATE_UTR_DETECTED | utr=%s | sess=%s", utr, session_id)
            return (
                ValidationResult(
                    status="duplicate_utr",
                    reason=f"Transaction ID {utr} has already been registered in the system.",
                    discrepancies=["Duplicate transaction ID"],
                ),
                VisionAction(
                    type="escalate",
                    message=(
                        "This payment transaction reference has already been processed previously. "
                        "If you believe this is an error, our team will review your account."
                    ),
                    escalation_reason=f"Duplicate UTR submitted: {utr}",
                ),
            )

    # 5. Check if amount is missing or unreadable
    if extracted_amt is None:
        if utr:
            payment_info = {
                "booking_id": booking_id,
                "amount": expected_amount,
                "payment_mode": f"{payment_app} (Amount Unclear)",
                "payer_name": payer_display,
                "transaction_id": utr,
                "details": f"Amount unreadable from screenshot. Ref: {utr}.",
            }
            await create_payment_confirmation_task(
                state={
                    **session_state,
                    "hotel_id": hotel_id,
                    "booking_id": booking_id,
                    "guest_phone": customer_phone,
                    "guest_name": guest_name,
                },
                payment_info=payment_info,
                source="vision_amount_unclear",
            )
        from app.prompts import templates
        unclear_cards = templates.payment_screenshot_unclear()
        unclear_msg = unclear_cards[0].get("text", "The uploaded screenshot appears unclear or is missing the UTR/transaction ID and amount.")
        return (
            ValidationResult(
                status="needs_clarification",
                reason="Unable to extract payment amount from image.",
            ),
            VisionAction(
                type="ask_clarification",
                message=unclear_msg,
            ),
        )

    # ── Auto-verification gate (sits between step 5 and the financial steps) ──
    #
    # Product, 2026-10-16: "turn off UPI payment auto verification." With the
    # flag off, a screenshot never confirms anything by itself — it becomes a
    # front-desk task, the guest is told the receipt is with the team, and
    # nothing is recorded as paid. Every rejection above (tampering,
    # FAILED/DECLINED, PENDING, duplicate UTR, unreadable amount) has already
    # run, so this only intercepts screenshots that would otherwise have
    # *succeeded* — arms A and B below are the only behaviour it changes.
    from app.config.settings import get_settings
    if not get_settings().UPI_SCREENSHOT_AUTO_VERIFICATION:
        logger.info(
            "PAYMENT_AUTO_VERIFY_DISABLED | sess=%s | booking=%s | utr=%s | amt=%s",
            session_id, booking_id, utr, extracted_amt,
        )
        payment_info = {
            "booking_id": booking_id,
            "amount": extracted_amt or expected_amount,
            "payment_mode": f"{payment_app} (Front Desk Verification)",
            "payer_name": payer_display,
            "transaction_id": utr,
            "details": (
                f"Auto-verification disabled. Guest reports ₹{(extracted_amt or 0):,.0f} paid; "
                f"outstanding was ₹{existing_amount_due:,.0f}. Verify against bank records before confirming."
            ),
        }
        await create_payment_confirmation_task(
            state={
                **session_state,
                "hotel_id": hotel_id,
                "booking_id": booking_id,
                "guest_phone": customer_phone,
                "guest_name": guest_name,
            },
            payment_info=payment_info,
            source="vision_auto_verify_disabled",
        )
        return (
            ValidationResult(
                status="escalate",
                reason="Screenshot received and routed to the front desk (auto-verification disabled).",
            ),
            VisionAction(
                type="escalate",
                message=(
                    f"Thank you for sharing your payment receipt! 🙏 We've received your transfer details "
                    f"(Ref: {utr or 'Pending'}). Our front desk team will verify it against our bank records "
                    f"and confirm your booking shortly."
                ),
                escalation_reason="Payment screenshot awaiting front-desk verification",
                db_updates={
                    "booking_stage": "payment_check",
                },
            ),
        )

    # 6. Financial Evaluation (Full, Advance, Balance, or Task Escalation)
    is_balance_payment = already_paid > 0 and existing_amount_due > 0
    diff_expected = abs(extracted_amt - expected_amount)
    diff_total = abs(extracted_amt - total_price) if total_price > 0 else 999999.0

    # A) Exact Match on remaining balance or total price
    if diff_expected <= AMOUNT_TOLERANCE or diff_total <= AMOUNT_TOLERANCE:
        new_total_paid = (already_paid + extracted_amt) if is_balance_payment else extracted_amt
        new_amount_due = max(0.0, total_price - new_total_paid) if total_price > 0 else 0.0
        final_payment_status = "paid" if new_amount_due <= AMOUNT_TOLERANCE else "partial"

        if is_balance_payment:
            confirm_msg = (
                f"Payment of ₹{extracted_amt:,.0f} verified successfully! 🎉 "
                f"Your remaining balance is now fully cleared (Total Paid: ₹{new_total_paid:,.0f}). "
                f"Your booking is confirmed. We look forward to hosting you!"
            )
        else:
            confirm_msg = (
                f"Payment of ₹{extracted_amt:,.0f} verified successfully! 🎉 "
                f"Your booking is now confirmed. We look forward to hosting you!"
            )

    # B) Partial Advance Payment (e.g. ₹2,000 paid towards ₹12,200)
    elif total_price > 0 and 0 < extracted_amt < expected_amount:
        new_total_paid = already_paid + extracted_amt
        new_amount_due = max(0.0, total_price - new_total_paid)
        final_payment_status = "partial"
        confirm_msg = (
            f"Payment of ₹{extracted_amt:,.0f} verified successfully! 🎉 "
            f"(Total Paid: ₹{new_total_paid:,.0f}, Balance Due at check-in: ₹{new_amount_due:,.0f}). "
            f"Your booking is confirmed!"
        )

    # C) Amount Difference / Variance -> Create Front Desk Task
    else:
        logger.warning(
            "PAYMENT_AMOUNT_MISMATCH | expected=%.2f | extracted=%.2f | total=%.2f | paid=%.2f",
            expected_amount, extracted_amt, total_price, already_paid
        )
        payment_info = {
            "booking_id": booking_id,
            "amount": extracted_amt,
            "payment_mode": f"{payment_app} (Amount Variance)",
            "payer_name": payer_display,
            "transaction_id": utr or "Pending Verification",
            "details": f"Paid ₹{extracted_amt:,.0f}, expected ₹{expected_amount:,.0f} (Total: ₹{total_price:,.0f}, Prev Paid: ₹{already_paid:,.0f}).",
        }
        await create_payment_confirmation_task(
            state={
                **session_state,
                "hotel_id": hotel_id,
                "booking_id": booking_id,
                "guest_phone": customer_phone,
                "guest_name": guest_name,
            },
            payment_info=payment_info,
            source="vision_amount_variance",
        )
        discrepancies = [f"Expected: ₹{expected_amount:,.0f}, Paid: ₹{extracted_amt:,.0f}"] if expected_amount > 0 else []
        if expected_amount <= 0:
            clarify_msg = (
                f"Thank you for sharing your payment receipt! 🎉 We've received your transfer of ₹{extracted_amt:,.0f} (Ref: {utr or 'Pending'}). "
                f"Our front desk team has been notified to verify the transaction details and confirm your reservation shortly. 🏨"
            )
        else:
            from app.prompts import templates
            mismatch_cards = templates.payment_amount_mismatch(
                expected=expected_amount,
                received=extracted_amt,
                booking_id=booking_id,
            )
            clarify_msg = mismatch_cards[0].get("text", f"Payment screenshot shows ₹{extracted_amt:,.0f}, but the expected amount is ₹{expected_amount:,.0f}.")
        return (
            ValidationResult(
                status="amount_mismatch" if expected_amount > 0 else "needs_staff_review",
                reason=f"Amount paid (₹{extracted_amt:,.0f}) differs from expected balance (₹{expected_amount:,.0f})." if expected_amount > 0 else "Payment receipt received without prior balance record; queued for staff review.",
                discrepancies=discrepancies,
            ),
            VisionAction(
                # `escalate`, not `confirm_payment`: nothing has been confirmed here, and the
                # action actually taken is a front-desk task (created just above). Note that
                # `action.type` has no reader anywhere in the codebase today, so this is
                # honesty for the next person rather than a behaviour change.
                type="ask_clarification" if expected_amount > 0 else "escalate",
                message=clarify_msg,
                escalation_reason=f"Payment amount variance (Paid ₹{extracted_amt}, expected ₹{expected_amount})" if expected_amount > 0 else f"Payment receipt received (₹{extracted_amt}) for manual verification",
                # Only what is actually known: a receipt arrived and it needs a human. This
                # arm is reached precisely when there is no balance to validate against, so
                # there is nothing that could justify the two writes that used to be here —
                # `booking_stage: confirmed` settled a booking nobody confirmed, and
                # `amount_paid` recorded an amount read off an unverified screenshot, which
                # then surfaces to the model as "Amount Paid: ₹X" (schemas.py:819-820) and
                # reads as money genuinely received. `review_needed` is the word this codebase
                # already uses for this state (understand.py:2823, task_handler.py:71).
                db_updates={
                    "payment_status": "review_needed",
                    "booking_id": booking_id,
                } if expected_amount <= 0 else {},
            ),
        )

    # 7. Payment verified successfully!
    receipt_doc = {
        "session_id": session_id,
        "booking_id": booking_id,
        "hotel_id": hotel_id,
        "customer_identifier": customer_phone,
        "payment_details": {
            "amount": extracted_amt,
            "transaction_id": utr,
            "status": payment_data.status,
            "payment_app": payment_app,
            "sender_name": sender_name,
            "receiver_name": payment_data.receiver_name,
            "timestamp": payment_data.timestamp,
        },
        "is_approved": True,
    }
    await payment_receipt_repository.save_payment_receipt(receipt_doc)

    # Update MongoDB booking if booking exists.
    #
    # Both branches confirm the booking; only the outstanding balance differs.
    # The partial-payment branch used to write booking_status by hand with a raw
    # update_one, which skipped the hold release and the inventory reservation —
    # so every advance-paid booking confirmed a room that was never taken out of
    # `daily_inventory`. `confirm_pending_booking` takes `amount_due` as an
    # argument and does all three, so one call covers both cases.
    if booking_id:
        await booking_repository.confirm_pending_booking(
            booking_id=booking_id,
            payment_status=final_payment_status,
            amount_paid=new_total_paid,
            amount_due=0.0 if new_amount_due <= AMOUNT_TOLERANCE else new_amount_due,
        )

        # ── The slip this path has never sent ────────────────────────────────
        #
        # Every other route that confirms a booking also sends the guest the
        # confirmation slip — the Razorpay webhook (`payment_webhook.py:135`),
        # staff confirmation (`tasks.py:268,328`) and the reconciliation sweep
        # (`reconciliation_service.py:70`). This one does not, and it is the one
        # Sample Hotel's bookings actually come through: the guest pays by UPI,
        # photographs the receipt, and the screenshot is verified here.
        #
        # Measured 2026-10-11 on `BK-20261010-94330B97` (Sample Hotel, 11-12 Oct,
        # ₹460 of ₹1,550): the guest's whole confirmation was the one line below
        # — no dates, no room, no address, and the prices were right only because
        # `confirm_pending_booking` above had just written them. Only 5 of 412
        # bookings carry `confirmation_sent`, and the last slip of any kind went
        # out on 2026-09-01.
        #
        # This is additive. `confirm_msg` is still what the turn returns, so the
        # guest gets today's acknowledgement and then the slip — if the slip
        # fails they are no worse off than before this call existed.
        #
        # Three properties make it safe to call from here:
        #
        #   * Idempotent. `send_booking_confirmation_notification` claims
        #     `confirmation_sent` with one atomic `update_one` and returns early
        #     when another process already holds it, so a webhook or a
        #     reconciliation arriving later cannot send a second slip. It also
        #     RELEASES the claim when the send fails (`confirmation_service.py:349`),
        #     so a failed attempt here does not spend the booking's one chance.
        #   * Non-fatal. A raise is caught below; the booking is already confirmed
        #     by the call above, and the guest's reply does not depend on this.
        #   * Grounded. It re-reads the booking from Mongo, and `amount_paid` is
        #     passed so the slip's paid/balance figures are the ones just written
        #     rather than whatever the pre-call document held.
        #
        # The cost, stated plainly: `whatsapp_service.send_message` is synchronous
        # — a `requests.post` with a 10 s timeout — so a stalled send holds this
        # turn for up to ten seconds. Every existing caller already runs that way
        # (they are API routes and background jobs, where it does not matter);
        # this is the first caller on a live chat turn. Making the send async is a
        # change to `whatsapp_service`, not to this call.
        from app.config.settings import get_settings

        if get_settings().CONFIRMATION_SLIP_ON_SCREENSHOT_PAYMENT:
            try:
                from app.services.confirmation_service import (
                    send_booking_confirmation_notification,
                )

                sent = await send_booking_confirmation_notification(
                    booking_id=str(booking_id),
                    payment_id=utr,
                    amount_paid=new_total_paid,
                )
                logger.info(
                    "PAYMENT_CONFIRMATION_SLIP | booking_id=%s sent=%s utr=%s",
                    booking_id, sent, utr,
                )
            except Exception as e:
                # A slip that fails must not cost the guest the acknowledgement
                # they already get, so this deliberately does not re-raise.
                logger.warning(
                    "PAYMENT_CONFIRMATION_SLIP_FAILED | booking_id=%s error=%s",
                    booking_id, e,
                )

    payer_note = f" from {sender_name}" if sender_name else ""
    await log_activity(
        ActivityLog(
            hotel_id=str(hotel_id),
            activity_type="SYSTEM_ACTION",
            reference_id=str(booking_id or session_id or "PAYMENT"),
            customer_phone=customer_phone,
            description=f"Payment of ₹{extracted_amt:,.0f}{payer_note} verified via {payment_app} screenshot (UTR: {utr or 'N/A'}).",
        )
    )

    return (
        ValidationResult(
            status="valid",
            reason=f"Payment of ₹{extracted_amt:,.0f} verified successfully.",
        ),
        VisionAction(
            type="confirm_payment",
            message=confirm_msg,
            db_updates={
                "booking_stage": "confirmed",
                "payment_status": final_payment_status,
                "amount_paid": new_total_paid,
                "amount_due": new_amount_due,
                "booking_id": booking_id,
            },
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. ROOM REFERENCE VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

async def validate_room_reference(
    room_data: RoomReferenceExtracted,
    hotel_id: str | int,
    session_state: dict[str, Any],
    all_rooms: list[Any] | None = None,
) -> tuple[ValidationResult, VisionAction]:
    """
    Compares the visible room features (beds, arrangement, style) against the hotel's
    actual room catalog in MongoDB to suggest the closest genuine matching room.
    Strictly avoids hallucinating non-existent amenities or promising unavailable setups.
    """
    if room_data.confidence < 0.4:
        return (
            ValidationResult(
                status="needs_clarification",
                reason="Room reference photo confidence is too low to identify bed configuration.",
            ),
            VisionAction(
                type="ask_clarification",
                message=(
                    "I can see a room photo, but the bed layout is a bit unclear. "
                    "Are you looking for a King bed room, Twin beds, or a Family room with multiple beds?"
                ),
            ),
        )

    # Fetch actual hotel rooms from MongoDB if not pre-provided
    if all_rooms is None:
        all_rooms = await room_repository.get_all_rooms(hotel_id)
    if not all_rooms:
        return (
            ValidationResult(
                status="valid",
                reason="No rooms found in database for this hotel.",
            ),
            VisionAction(
                type="suggest_room",
                message="Thank you for sharing your room preference! Please let me know your travel dates so I can check which of our rooms are available for you.",
            ),
        )

    detected_beds = room_data.bed_count or 1
    detected_type = (room_data.bed_type or "unknown").lower()

    # 1. First priority: Grounded Multimodal Match from Gemini
    best_match = None
    if getattr(room_data, "matched_room_id", None):
        for room in all_rooms:
            if getattr(room, "room_id", "") == room_data.matched_room_id:
                best_match = room
                break

    if not best_match and getattr(room_data, "matched_room_name", None):
        for room in all_rooms:
            if room.name.strip().lower() == room_data.matched_room_name.strip().lower():
                best_match = room
                break

    # 2. Second priority: Comprehensive multi-factor scoring across ALL rooms (no premature break)
    if not best_match:
        inferred_cat = (room_data.room_category_inferred or "").lower()
        style_desc = (room_data.room_style or "").lower()

        scores: list[tuple[int, Any]] = []
        for room in all_rooms:
            r_name_lower = room.name.lower()
            score = 0

            # Bed count / layout matching
            if detected_beds >= 4 or detected_type == "family":
                if "four" in r_name_lower or "family" in r_name_lower or "4" in r_name_lower:
                    score += 60
            elif detected_beds == 3 or detected_type == "triple":
                if "triple" in r_name_lower or "tripple" in r_name_lower or "3" in r_name_lower:
                    score += 60
            elif detected_beds == 2 or detected_type == "twin":
                if "twin" in r_name_lower or "double" in r_name_lower:
                    score += 50
            elif detected_type in ("king", "queen", "single") or detected_beds == 1:
                if detected_type == "king" and "king" in r_name_lower:
                    score += 35
                elif detected_type == "queen" and "queen" in r_name_lower:
                    score += 35
                elif detected_type == "single" and "single" in r_name_lower:
                    score += 35

            # Tier & Aesthetic matching
            if "premier" in inferred_cat or "premier" in style_desc or "luxury" in style_desc or "wooden" in style_desc:
                if "premier" in r_name_lower:
                    score += 45
                elif "deluxe" in r_name_lower and "family" not in r_name_lower:
                    score += 20
            elif "deluxe" in inferred_cat or "deluxe" in style_desc:
                if "deluxe" in r_name_lower and "family" not in r_name_lower and "four" not in r_name_lower:
                    score += 35
            elif "standard" in inferred_cat or "standard" in style_desc or "budget" in style_desc:
                if "standard" in r_name_lower:
                    score += 35

            # Amenity overlap
            if isinstance(room.amenities, dict):
                r_amenities = [k.lower() for k, v in room.amenities.items() if v]
            elif isinstance(room.amenities, list):
                r_amenities = [str(a).lower() for a in room.amenities]
            else:
                r_amenities = []
            for vis_am in room_data.visible_amenities:
                if any(vis_am.lower() in a for a in r_amenities):
                    score += 5

            scores.append((score, room))

        if scores:
            scores.sort(key=lambda x: x[0], reverse=True)
            best_match = scores[0][1]

    target_room = best_match or all_rooms[0]

    # Extract base price
    base_price = 0.0
    if hasattr(target_room, "rate_plans") and target_room.rate_plans:
        base_price = float(target_room.rate_plans[0].base_price)
    elif hasattr(target_room, "base_price"):
        base_price = float(target_room.base_price)
    elif isinstance(target_room, dict):
        base_price = float(target_room.get("base_price") or 0.0)

    # Conversational reply message
    match_expl = getattr(room_data, "match_explanation", None)
    sim_score = getattr(room_data, "similarity_score", None) or 85.0
    if match_expl and match_expl.strip():
        reply_msg = (
            f"Based on your photo, our closest matching option is the *{target_room.name}*! 🏨\n\n"
            f"{match_expl.strip()}\n\n"
            f"Base rate starts from ₹{base_price:,.0f}/night.\n\n"
            f"Would you like to check availability for your dates?"
        )
    else:
        # Filter confirmed amenities present in both image and actual room DB
        raw_am = getattr(target_room, "amenities", [])
        if isinstance(raw_am, dict):
            actual_amenities = [k for k, v in raw_am.items() if v]
        elif isinstance(raw_am, list):
            actual_amenities = raw_am
        else:
            actual_amenities = []

        confirmed_amenities = []
        for vis_am in room_data.visible_amenities:
            for act_am in actual_amenities:
                if vis_am.lower() in act_am.lower() or act_am.lower() in vis_am.lower():
                    confirmed_amenities.append(act_am)

        amenity_str = f" with {', '.join(confirmed_amenities[:3])}" if confirmed_amenities else ""
        bed_desc = f"{detected_beds}-bed" if detected_beds > 1 else (detected_type.title() + " bed" if detected_type != "unknown" else "comfortable bed")

        reply_msg = (
            f"This looks like a lovely {bed_desc} room setup! 🏨\n\n"
            f"In our property, our closest matching option is the *{target_room.name}*{amenity_str}.\n"
            f"Base rate starts from ₹{base_price:,.0f}/night.\n\n"
            f"Would you like to check availability for your dates?"
        )

    return (
        ValidationResult(
            status="valid",
            reason=f"Matched photo to closest room in DB: {target_room.name}",
            metadata={
                "matched_room_id": target_room.room_id,
                "matched_room_name": target_room.name,
                "similarity_score": sim_score,
            },
        ),
        VisionAction(
            type="suggest_room",
            message=reply_msg,
            db_updates={
                "photo_tag_preference": target_room.name,
                "preferred_room_id": target_room.room_id,
                "preferred_room_name": target_room.name,
                "recommendation_preference": "MATCH_PHOTO",
            },
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. OTA PRICE MATCH VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

async def validate_ota_price_match(
    ota_data: OTAPriceExtracted,
    hotel_id: str | int,
    session_state: dict[str, Any],
) -> tuple[ValidationResult, VisionAction]:
    """
    Validates a competitor OTA screenshot against hotel owner policy, seasonal limits,
    tax normalization, and hard floor price bounds.
    """
    if ota_data.confidence < 0.5 or not ota_data.price:
        return (
            ValidationResult(
                status="needs_clarification",
                reason="Unable to clearly extract hotel room price or dates from OTA screenshot.",
            ),
            VisionAction(
                type="ask_clarification",
                message="I received your screenshot, but the price or room details are not clearly visible. Please upload a clear screenshot showing the total room rate and dates.",
            ),
        )

    # 1. Fetch hotel pricing & price match policy
    policy = await price_match_service.get_price_match_policy(hotel_id)
    if not policy.get("price_match_enabled", True):
        return (
            ValidationResult(
                status="rejected",
                reason="Price match is disabled for this property in settings.",
            ),
            VisionAction(
                type="ask_clarification",
                message="We currently offer our best direct booking rates with exclusive perks. Let me know your dates to see our direct offers!",
            ),
        )

    # 2. Tax normalization
    raw_price = float(ota_data.price)
    if ota_data.tax_included is False:
        # Add standard 12% GST estimation
        normalized_ota_price = round(raw_price * 1.12, 2)
    else:
        normalized_ota_price = raw_price

    platform = (ota_data.booking_platform or "Online Portal").strip()
    current_total = float(session_state.get("total_price") or 0.0)
    base_total = float(session_state.get("base_total_price") or current_total or normalized_ota_price * 1.1)

    # 3. Fetch seasonal negotiation limits
    check_in = ota_data.check_in or session_state.get("check_in") or ""
    check_out = ota_data.check_out or session_state.get("check_out") or ""
    max_discount_pct, season_name = await get_season_negotiation_limit(hotel_id, str(check_in), str(check_out))

    # Policy cap
    owner_cap_pct = float(policy.get("maximum_discount_percent") or 10.0)
    allowed_max_pct = min(max_discount_pct, owner_cap_pct)

    # 4. Calculate proposed direct booking offer (5% lower than OTA price or capped by policy)
    proposed_offer = float(round((normalized_ota_price * 0.95) / 10) * 10)
    discount_from_base = max(0.0, base_total - proposed_offer)
    discount_pct = (discount_from_base / base_total * 100.0) if base_total > 0 else 5.0

    # 5. Check Floor Price
    if policy.get("floor_price") is not None:
        hard_floor_price = float(policy["floor_price"])
    else:
        hard_floor_price = float(base_total * (1.0 - (allowed_max_pct / 100.0))) if base_total > 0 else 0.0

    if proposed_offer < hard_floor_price:
        logger.info(
            "PRICE_MATCH_FLOOR_REACHED | proposed=%.2f | floor=%.2f | ota=%.2f",
            proposed_offer,
            hard_floor_price,
            normalized_ota_price,
        )
        capped_offer = float(round(hard_floor_price / 10) * 10)
        return (
            ValidationResult(
                status="below_floor",
                reason=f"Competitor price ₹{normalized_ota_price:,.0f} is below our minimum floor price ₹{hard_floor_price:,.0f}.",
            ),
            VisionAction(
                type="negotiate_price",
                message=(
                    f"Thank you for sharing the *{platform}* screenshot! 🏨\n\n"
                    f"Our absolute best direct-booking rate for your stay is *₹{capped_offer:,.0f}* "
                    f"(includes direct front desk support, free WiFi, and priority early check-in).\n\n"
                    f"We cannot go below this rate, but we guarantee 100% direct hotel service. "
                    f"Shall I confirm your booking at *₹{capped_offer:,.0f}*?"
                ),
                db_updates={
                    "total_price": capped_offer,
                    "amount_due": capped_offer - float(session_state.get("amount_paid") or 0.0),
                    "discount_amount": base_total - capped_offer,
                    "discount_reason": f"Direct Rate Match Guarantee ({allowed_max_pct:.0f}% MAX)",
                    "booking_stage": "quote_shown",
                    "next_action": "need_guest_name_for_payment",
                },
                escalation_reason="Customer requested price match below policy floor price",
            ),
        )

    # 6. Safe match within allowed limits
    savings = float(round(normalized_ota_price - proposed_offer))
    reply_text = (
        f"🎉 *Exclusive Direct Booking Price Match!*\n\n"
        f"We verified your *{platform}* price of ₹{normalized_ota_price:,.0f}.\n"
        f"To give you the best value, here is our special direct rate (5% extra discount):\n\n"
        f"💰 *{platform} Price:* ₹{normalized_ota_price:,.0f}\n"
        f"🎁 *StayInn Direct Rate:* *₹{proposed_offer:,.0f}*\n"
        f"✨ *Your Direct Savings:* ₹{savings:,.0f}\n\n"
        f"🌟 *Direct Perks:* Free early check-in (subject to availability) & instant front desk assistance.\n\n"
        f"Would you like to confirm your reservation at *₹{proposed_offer:,.0f}*?"
    )

    return (
        ValidationResult(
            status="valid",
            reason=f"Price match approved against {platform}. Direct offer generated.",
        ),
        VisionAction(
            type="negotiate_price",
            message=reply_text,
            db_updates={
                "total_price": proposed_offer,
                "amount_due": proposed_offer - float(session_state.get("amount_paid") or 0.0),
                "discount_amount": discount_from_base,
                "discount_reason": f"Direct Booking Price Match vs {platform}",
                "booking_stage": "quote_shown",
                "next_action": "need_guest_name_for_payment",
            },
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. ID CARD / GUEST IDENTITY VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

async def validate_id_card(
    id_data: IDCardExtracted,
    hotel_id: str | int,
    session_state: dict[str, Any],
) -> tuple[ValidationResult, VisionAction]:
    """
    Validates a government ID card (Aadhaar, Passport, Driving License, Voter ID, PAN).
    Updates guest name in state and guides guest to the next conversational step.
    """
    if not id_data.is_valid_id or id_data.confidence < 0.3:
        return (
            ValidationResult(
                status="rejected",
                reason="Uploaded image could not be verified as a valid government ID document.",
            ),
            VisionAction(
                type="ask_clarification",
                message=(
                    "I could not clearly read the details on this ID card. "
                    "Please upload a clear, front-facing photo of your Government ID (Aadhaar card, Passport, Driving License, or Voter ID)."
                ),
            ),
        )

    doc_type_names = {
        "aadhaar": "Aadhaar Card",
        "passport": "Passport",
        "driving_license": "Driving License",
        "voter_id": "Voter ID",
        "pan_card": "PAN Card",
        "national_id": "National Identity Card",
        "other_id": "Government ID",
    }
    doc_label = doc_type_names.get(id_data.id_type, "Government ID")
    name = id_data.full_name or "Guest"
    masked_num = id_data.document_number_masked or ""

    stage = session_state.get("booking_stage") or "new"
    has_rooms = bool(session_state.get("selected_rooms"))
    booking_id = session_state.get("booking_id")

    # Accumulate into verified_ids list for multi-guest / bulk ID support
    existing_ids = list(session_state.get("verified_ids") or [])
    new_card = {
        "guest_name": name,
        "id_type": id_data.id_type,
        "doc_label": doc_label,
        "id_masked_number": masked_num,
    }
    is_duplicate = any(
        (masked_num and e.get("id_masked_number") == masked_num)
        or (name != "Guest" and e.get("guest_name", "").strip().lower() == name.strip().lower() and masked_num == e.get("id_masked_number"))
        for e in existing_ids
    )
    if not is_duplicate:
        existing_ids.append(new_card)

    total_count = len(existing_ids)
    cards_bullets = "\n".join([
        f"• **Guest {i+1}:** {c['guest_name']} — {c['doc_label']}" + (f" ({c['id_masked_number']})" if c.get('id_masked_number') else "")
        for i, c in enumerate(existing_ids)
    ])

    db_updates: dict[str, Any] = {
        "id_verified": True,
        "id_type": id_data.id_type,
        "id_masked_number": masked_num,
        "verified_ids": existing_ids,
    }
    if id_data.full_name:
        db_updates["guest_name"] = id_data.full_name

    # ── Database Persistence: Canonical Guest, Booking & Activity Log ──
    try:
        from app.repositories import guest_repository
        from app.db.mongodb import bookings_collection
        from app.repositories.activity_repository import log_activity
        from app.models.activity_schemas import ActivityLog
        from datetime import datetime, timezone

        phone = session_state.get("customer_identifier") or session_state.get("customer_phone")
        if phone:
            guest_record = await guest_repository.get_or_create_guest(
                hotel_id=hotel_id,
                phone=phone,
                name=name if name != "Guest" else None,
            )
            guest_updates = {
                "valid_id": True,
                "id_type": id_data.id_type,
                "id_masked_number": masked_num,
            }
            if id_data.full_name and id_data.full_name != "Guest":
                guest_updates["name"] = id_data.full_name
            if id_data.dob:
                guest_updates["dob"] = id_data.dob
            if id_data.gender:
                guest_updates["gender"] = id_data.gender
            if id_data.address:
                guest_updates["address"] = id_data.address
            await guest_repository.update_guest(guest_record["guest_id"], guest_updates)

            await log_activity(ActivityLog(
                hotel_id=str(hotel_id),
                activity_type="ID_VERIFIED",
                reference_id=guest_record["guest_id"],
                booking_id=booking_id,
                customer_phone=phone,
                customer_name=name,
                description=f"Identity document verified: {doc_label} ({masked_num}) for {name}",
                metadata={
                    "id_type": id_data.id_type,
                    "masked_number": masked_num,
                    "confidence": id_data.confidence,
                }
            ))

        if booking_id:
            coll = bookings_collection()
            now = datetime.now(timezone.utc)
            await coll.update_one(
                {"booking_id": booking_id},
                {"$set": {
                    "id_verified": True,
                    "id_details": {
                        "id_type": id_data.id_type,
                        "doc_label": doc_label,
                        "masked_number": masked_num,
                        "guest_name": name,
                        "verified_at": now.isoformat(),
                    },
                    "updated_at": now,
                }}
            )
    except Exception as e:
        logger.warning(f"Failed to persist ID verification to DB: {e}")

    doc_with_num = f"{doc_label} ({masked_num})" if masked_num else doc_label

    # Scenario 1: Quote is already shown, or waiting for guest name to create booking hold
    if stage in ("quote_shown", "rooms_shown") or (stage == "collecting" and has_rooms):
        db_updates["booking_stage"] = "quote_shown"
        db_updates["next_action"] = "confirm_with_name"
        
        name_part = f" for {name}" if name and name != "Guest" else ""
        doc_part = f" ({doc_with_num})" if doc_with_num else ""
        reply = f"Your ID verification has been completed successfully{name_part}{doc_part}. Reply *confirm* to proceed with your booking! 🏨"
        return (
            ValidationResult(
                status="valid",
                reason=f"ID verified for {name} ({doc_label})",
                metadata={"guest_name": name, "doc_label": doc_label, "masked_num": masked_num, "total_verified": total_count},
            ),
            VisionAction(
                type="verify_id_card",
                message=reply,
                db_updates=db_updates,
            ),
        )

    # Scenario 2: Already at awaiting_payment (Hold created, waiting for payment)
    elif stage == "awaiting_payment":
        reply = "Your ID verification has been completed successfully. Please complete the advance payment to finalize your booking! 💳"
        return (
            ValidationResult(
                status="valid",
                reason=f"ID attached to booking {booking_id}",
                metadata={"guest_name": name, "doc_label": doc_label, "masked_num": masked_num, "total_verified": total_count},
            ),
            VisionAction(
                type="verify_id_card",
                message=reply,
                db_updates=db_updates,
            ),
        )

    # Scenario 3: Booking already confirmed
    elif stage in ("confirmed", "paid"):
        reply = "Your ID verification has been completed successfully. We look forward to welcoming you! 🏨"
        return (
            ValidationResult(
                status="valid",
                reason=f"ID saved for confirmed booking",
                metadata={"guest_name": name, "doc_label": doc_label, "masked_num": masked_num, "total_verified": total_count},
            ),
            VisionAction(
                type="verify_id_card",
                message=reply,
                db_updates=db_updates,
            ),
        )

    # Scenario 4: New / Collecting stage (no room selected yet)
    else:
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_updates["id_received_date"] = now_str
        db_updates["next_action"] = "ask_booking_or_checkin"

        name_part = f" for {name}" if name and name != "Guest" else ""
        doc_part = f" ({doc_with_num})" if doc_with_num else ""
        reply = (
            f"Your ID verification has been completed successfully{name_part}{doc_part}. "
            f"Are you checking in with your Booking ID or looking for a new booking? 🏨"
        )
        return (
            ValidationResult(
                status="valid",
                reason=f"ID verified for {name} ({doc_label})",
                metadata={"guest_name": name, "doc_label": doc_label, "masked_num": masked_num, "total_verified": total_count},
            ),
            VisionAction(
                type="verify_id_card",
                message=reply,
                db_updates=db_updates,
            ),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. HANDWRITTEN / PAPER BOOKING NOTE VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

async def validate_booking_note(
    note_data: BookingNoteExtracted,
    hotel_id: int,
    session_state: dict[str, Any],
) -> tuple[ValidationResult, VisionAction]:
    """
    Validates extracted handwritten booking slip and converts it into a proper CRM Lead,
    persisting to MongoDB canonical leads collection and preparing room search.
    """
    from app.repositories import lead_repository, guest_repository

    contact_name = note_data.guest_name or session_state.get("guest_name") or "Guest"
    contact_phone = note_data.phone_number or session_state.get("customer_identifier") or session_state.get("customer_phone") or ""

    # Generate canonical Lead ID (LEAD-YYYYMMDD-NNNNNN)
    lead_id = None
    try:
        lead_id = await lead_repository.generate_lead_id(hotel_id)
        note_data.lead_id = lead_id

        lead_data = {
            "lead_id": lead_id,
            "hotel_id": str(hotel_id),
            "session_id": session_state.get("session_id", ""),
            "customer_phone": contact_phone,
            "guest_name": contact_name,
            "check_in": note_data.check_in,
            "check_out": note_data.check_out,
            "adults": note_data.adults or 2,
            "children": note_data.children or 0,
            "num_rooms": note_data.num_rooms or 1,
            "preferred_room": note_data.room_type_preference,
            "meal_plan": note_data.meal_plan_preference,
            "special_requests": note_data.special_requests,
            "budget": note_data.budget,
            "lead_type": "HANDWRITTEN_NOTE",
            "source": "handwritten_slip",
            "status": "NEW",
            "notes": note_data.raw_notes or "Handwritten booking slip inquiry",
        }
        await lead_repository.create_lead(lead_data)

        if contact_phone:
            await guest_repository.get_or_create_guest(
                hotel_id=hotel_id,
                phone=contact_phone,
                name=contact_name if contact_name != "Guest" else None,
                source="handwritten_slip",
            )
    except Exception as e:
        logger.warning(f"Failed to create CRM lead from handwritten note: {e}")
        if not lead_id:
            lead_id = f"LEAD-MANUAL-{hotel_id}"

    db_updates: dict[str, Any] = {
        "booking_stage": "collecting",
        "intent": "BOOK_ROOM",
        "next_action": "search",
        "force_new_search": True,
        "lead_id": lead_id,
        "lead_created": True,
    }
    if note_data.guest_name and note_data.guest_name != "Guest":
        db_updates["guest_name"] = note_data.guest_name
    if note_data.phone_number:
        db_updates["customer_identifier"] = note_data.phone_number
    if note_data.check_in:
        db_updates["check_in"] = note_data.check_in
    if note_data.check_out:
        db_updates["check_out"] = note_data.check_out
    if note_data.adults:
        db_updates["adults"] = note_data.adults
        db_updates["effective_adults"] = note_data.adults
    if note_data.children is not None:
        db_updates["children"] = note_data.children
        db_updates["effective_children"] = note_data.children
    if note_data.num_rooms:
        db_updates["num_rooms"] = note_data.num_rooms
    if note_data.room_type_preference:
        db_updates["room_type_preference"] = note_data.room_type_preference
    if note_data.meal_plan_preference:
        db_updates["meal_plan_preference"] = note_data.meal_plan_preference
    if note_data.special_requests:
        db_updates["special_requests"] = note_data.special_requests

    parts = []
    if contact_name and contact_name != "Guest":
        parts.append(f"👤 **Guest:** {contact_name}" + (f" ({contact_phone})" if contact_phone else ""))
    elif contact_phone:
        parts.append(f"📞 **Phone:** {contact_phone}")
    if note_data.check_in and note_data.check_out:
        parts.append(f"📅 **Stay Dates:** {note_data.check_in} to {note_data.check_out}")
    if note_data.adults:
        parts.append(f"👥 **Guests:** {note_data.adults} adults" + (f", {note_data.children} children" if note_data.children else ""))
    if note_data.num_rooms:
        rt = f" ({note_data.room_type_preference})" if note_data.room_type_preference else ""
        parts.append(f"🛏️ **Rooms:** {note_data.num_rooms} room{'s' if note_data.num_rooms > 1 else ''}{rt}")
    if note_data.meal_plan_preference:
        parts.append(f"🍽️ **Meal Plan:** {note_data.meal_plan_preference}")
    if note_data.special_requests:
        parts.append(f"📝 **Special Requests:** {note_data.special_requests}")
    if note_data.budget:
        parts.append(f"💰 **Rate / Budget:** ₹{note_data.budget:,.0f}")

    summary_str = "\n".join(f"• {p}" for p in parts) if parts else "your reservation requirements"

    msg = (
        f"📝 **Handwritten Note Converted to Lead!**\n\n"
        f"🎫 **Lead Reference:** `{lead_id}`\n\n"
        f"I have parsed the handwritten slip and created a qualified lead in our system:\n"
        f"{summary_str}\n\n"
        f"Our reservations team has logged this request. Let me check available room options for your stay dates! 🏨"
    )

    return (
        ValidationResult(
            status="valid",
            reason=f"Handwritten note converted to CRM lead {lead_id}",
            metadata={"lead_id": lead_id, "guest_name": contact_name, "phone": contact_phone},
        ),
        VisionAction(
            type="create_lead",
            message=msg,
            db_updates=db_updates,
        ),
    )

