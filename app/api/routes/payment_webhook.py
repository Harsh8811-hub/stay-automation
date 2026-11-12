"""
Razorpay Webhook Handler (Ticket 10)

Handles incoming payment webhooks from Razorpay with:
- HMAC-SHA256 signature verification
- 24-hour idempotency key check
- Transition booking status to CONFIRMED via set_booking_confirmed
- Dispatch confirmation message to guest
"""

import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from app.config.settings import get_settings
from app.db.mongodb import get_database
from app.repositories.booking_repository import get_booking, set_booking_confirmed
from app.services.confirmation_service import send_booking_confirmation_notification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["Payment Webhooks"])


def verify_razorpay_signature(payload_bytes: bytes, signature: str, secret: str) -> bool:
    """Verify Razorpay webhook HMAC-SHA256 signature."""
    if not signature or not secret:
        return False
    generated_sig = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(generated_sig, signature)


@router.post("/razorpay")
async def handle_razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(None, alias="X-Razorpay-Signature"),
):
    """
    POST /api/webhooks/razorpay
    Verifies signature, enforces 24h idempotency, and confirms booking.
    """
    body_bytes = await request.body()

    # 1. Signature Verification
    settings = get_settings()
    webhook_secret = getattr(settings, "RAZORPAY_WEBHOOK_SECRET", None) or getattr(settings, "RAZORPAY_KEY_SECRET", "")
    if not webhook_secret or not verify_razorpay_signature(body_bytes, x_razorpay_signature, webhook_secret):
        logger.warning("Razorpay webhook signature verification failed.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        )

    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to parse Razorpay webhook payload: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    event_id = data.get("event_id") or data.get("id")
    event_type = data.get("event")

    # 2. Idempotency Check (24h TTL)
    db = get_database()
    events_coll = db["webhook_events"]

    if event_id:
        existing = await events_coll.find_one({"event_id": event_id})
        if existing:
            logger.info(f"Duplicate Razorpay webhook event {event_id} ignored (idempotent).")
            return {"status": "success", "message": "Event already processed"}

    # 3. Process payment events
    #
    # `qr_code.credited` is the UPI-QR event: a guest scanned the Razorpay QR
    # and Razorpay captured the money. It is the event that lets a booking
    # confirm itself without a screenshot. Its payload carries the payment
    # under `payload.payment` and the code under `payload.qr_code`, and the
    # notes we set at create time ride on both — so notes are looked up across
    # every entity present rather than assuming one shape.
    if event_type in ("payment.captured", "order.paid", "payment_link.paid", "qr_code.credited"):
        payload = data.get("payload", {}) or {}
        payment_entity = (payload.get("payment") or {}).get("entity") or {}
        qr_entity = (payload.get("qr_code") or {}).get("entity") or {}
        link_entity = (payload.get("payment_link") or {}).get("entity") or {}
        order_entity = (payload.get("order") or {}).get("entity") or {}

        notes = {}
        for entity in (payment_entity, qr_entity, link_entity, order_entity):
            if isinstance(entity.get("notes"), dict) and entity["notes"]:
                notes = entity["notes"]
                break
        booking_id = (
            notes.get("booking_id")
            or notes.get("bookingId")
            or payment_entity.get("description")
            or qr_entity.get("description")
        )
        payment_id = (
            payment_entity.get("id")
            or notes.get("payment_id")
            or qr_entity.get("id")
        )

        # Razorpay amounts are in paise. The payment entity is authoritative
        # when present; the QR's `payment_amount` and the order's `amount` are
        # the fallbacks for an event that carries no payment object.
        amount_paise = (
            payment_entity.get("amount")
            or qr_entity.get("payment_amount")
            or order_entity.get("amount")
            or 0
        )
        try:
            amount = float(amount_paise) / 100.0
        except (TypeError, ValueError):
            amount = 0.0

        if booking_id:
            logger.info(
                f"Processing payment confirmation for Booking ID: {booking_id}, "
                f"Payment ID: {payment_id}, Amount: ₹{amount}"
            )
            success = await set_booking_confirmed(
                booking_id=booking_id,
                confirmation_source="razorpay_webhook_verified",
                payment_id=payment_id,
                amount_paid=amount,
            )
            if success:
                logger.info(f"✅ Booking {booking_id} successfully marked as CONFIRMED.")
                # Automatically complete any open front desk payment confirmation tasks
                try:
                    from app.repositories import task_repository
                    open_tasks, _ = await task_repository.list_tasks(
                        booking_id=booking_id,
                        status="OPEN",
                        subtype="PAYMENT_CONFIRMATION",
                        limit=5,
                    )
                    for ot in open_tasks:
                        await task_repository.update_task_status(
                            ot["task_id"],
                            "COMPLETED",
                            actor="razorpay_webhook",
                        )
                        await task_repository.add_task_note(
                            ot["task_id"],
                            f"Automatically completed via Razorpay webhook ({event_type}). Payment ID: {payment_id}",
                            author="system",
                        )
                except Exception as task_err:
                    logger.warning(f"Error completing task on webhook: {task_err}")

                try:
                    await send_booking_confirmation_notification(
                        booking_id=booking_id,
                        payment_id=payment_id,
                        amount_paid=amount,
                    )
                except Exception as notify_err:
                    logger.error(
                        f"Failed to dispatch booking confirmation notification for {booking_id}: {notify_err}",
                        exc_info=True,
                    )

    # 4. Record event for 24-hour idempotency
    if event_id:
        await events_coll.insert_one({
            "event_id": event_id,
            "event_type": event_type,
            "processed_at": datetime.now(timezone.utc),
            "expire_at": datetime.now(timezone.utc) + timedelta(hours=24),
        })

    return {"status": "success"}
