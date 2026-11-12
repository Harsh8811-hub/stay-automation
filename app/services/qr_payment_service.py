"""
Razorpay locked-amount UPI QR — create and verify via the StayInn backend.

Why this exists (2026-11-09, product): the bot used to draw its own QR from a
`upi://` string (`app/repositories/payment_gateway_repository.py` builds it with
api.qrserver.com). That image is a picture of a VPA. Nothing watches it, so a
guest who scans it pays the hotel and then has to send a screenshot, and a human
has to approve it before the booking is confirmed.

A Razorpay QR is an *order*: fixed amount, an expiry, and a `notes` bag we
control. Razorpay fires `qr_code.credited` at the webhook when it is paid, and
the notes carry the booking reference, so the booking confirms itself. The
`verifyqr_guest` endpoint is the pull-side fallback for a dropped webhook.

Two rules carried over from `payment_service.py`:

1. Never fabricate a QR. If creation fails, the caller keeps the payment link
   it already has; it must not show a dead QR.
2. Never verify a QR by its short `rzp.io` link. Verify by `order_id` (ord_...)
   or `qr_id` (qr_...), both of which come back at create time and must be
   persisted by the caller.

Endpoints (spec pasted by Product, 2026-11-09):
    POST /api/payment-collections/{hotel_id}/createqr_guest
    POST /api/payment-collections/{hotel_id}/verifyqr_guest
Both authenticated with `X-Order-Key`. The key is never put in a URL here: the
spec also offers `?order_key=`, but query strings land in access logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class QrCreateResult:
    """Outcome of a QR creation attempt."""

    order_id: str | None = None      # ord_...
    qr_id: str | None = None         # qr_...
    qr_link: str | None = None       # https://rzp.io/rzp/...
    image_url: str | None = None
    raw: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        # A QR is useless without a way to reach it. `order_id` is what we must
        # persist to verify later, so both are required for "ok".
        return bool(self.order_id and (self.qr_link or self.image_url))


@dataclass
class QrVerifyResult:
    """Outcome of a QR status check."""

    is_paid: bool = False
    amount_paid: float | None = None
    status: str | None = None
    qr_id: str | None = None
    payment_id: str | None = None
    error: str | None = None
    raw: dict | None = None


def enabled() -> bool:
    """Feature gate read, so one env change restores the old QR path."""
    return bool(getattr(get_settings(), "RAZORPAY_QR_ENABLED", False))


def _dig(data: dict, *keys: str):
    """First present key from `data` or `data['data']`, else None."""
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested and nested[key] not in (None, ""):
                return nested[key]
    return None


async def create_guest_qr(
    hotel_id: int | str,
    *,
    amount: float,
    booking_id: str,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    expires_in_minutes: int | None = None,
) -> QrCreateResult:
    """Create a fixed-amount UPI QR for this booking.

    `notes.booking_id` is the link the webhook follows back to the hold row. It
    is the booking/hold reference, never the guest's phone or name — the note is
    copied into Razorpay's dashboard and into any future webhook payload.
    """
    settings = get_settings()
    url = (
        f"{settings.STAYINN_BACKEND_URL.rstrip('/')}"
        f"/api/payment-collections/{hotel_id}/createqr_guest"
    )
    headers = {
        "X-Order-Key": settings.STAYINN_PAYMENT_ORDER_KEY,
        "Content-Type": "application/json",
    }
    payload: dict = {
        "amount": amount,
        "booking_id": booking_id,
        "expires_in_minutes": expires_in_minutes
        or getattr(settings, "RAZORPAY_QR_EXPIRY_MINUTES", 30),
        "notes": {"booking_id": booking_id, "hotel_id": str(hotel_id)},
    }
    if customer_name:
        payload["customer_name"] = str(customer_name)[:100]
    if customer_phone:
        # Razorpay accepts a bare 10-digit number; strip the country code.
        digits = "".join(ch for ch in str(customer_phone) if ch.isdigit())
        payload["customer_phone"] = digits[-10:]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=15.0)
            try:
                data = response.json()
            except Exception:
                data = {"raw_text": response.text}

            if response.status_code >= 400 or not isinstance(data, dict):
                err = (
                    _dig(data, "error", "message")
                    if isinstance(data, dict)
                    else response.text[:200]
                )
                logger.error(
                    "createqr_guest failed hotel=%s booking=%s status=%s err=%s",
                    hotel_id, booking_id, response.status_code, err,
                )
                return QrCreateResult(
                    error=str(err),
                    raw=data if isinstance(data, dict) else None,
                )

            if data.get("success") is False:
                err = _dig(data, "error", "message") or "success=false"
                logger.error(
                    "createqr_guest rejected hotel=%s booking=%s err=%s",
                    hotel_id, booking_id, err,
                )
                return QrCreateResult(error=str(err), raw=data)

            qr_details = data.get("qr_details") if isinstance(data.get("qr_details"), dict) else {}
            result = QrCreateResult(
                order_id=str(_dig(data, "order_id") or "") or None,
                qr_id=str(_dig(data, "qr_id") or qr_details.get("id") or "") or None,
                qr_link=_dig(data, "qr_link", "payment_link"),
                image_url=_dig(data, "image_url") or qr_details.get("image_url"),
                raw=data,
            )
            if not result.ok:
                logger.error(
                    "createqr_guest missing order/link hotel=%s booking=%s data=%s",
                    hotel_id, booking_id, data,
                )
                result.error = "Missing order_id or qr_link in createqr_guest response"
                return result

            logger.info(
                "QR created hotel=%s booking=%s order=%s qrid=%s amount=%s",
                hotel_id, booking_id, result.order_id, result.qr_id, amount,
            )
            return result
    except Exception as exc:  # network, timeout, DNS
        logger.error("createqr_guest error hotel=%s booking=%s err=%s", hotel_id, booking_id, exc)
        return QrCreateResult(error=str(exc))


async def verify_guest_qr(
    hotel_id: int | str,
    *,
    order_id: str | None = None,
    qr_id: str | None = None,
) -> QrVerifyResult:
    """Check whether a QR order has been paid.

    POST rather than GET so the order key travels in the header, not the query
    string. Either identifier is accepted by the backend; the caller should pass
    whatever it persisted at create time.
    """
    reference = order_id or qr_id
    if not reference:
        return QrVerifyResult(error="Missing qr order_id/qr_id")

    settings = get_settings()
    url = (
        f"{settings.STAYINN_BACKEND_URL.rstrip('/')}"
        f"/api/payment-collections/{hotel_id}/verifyqr_guest"
    )
    headers = {
        "X-Order-Key": settings.STAYINN_PAYMENT_ORDER_KEY,
        "Content-Type": "application/json",
    }
    payload = {"order_id": reference}
    if qr_id:
        payload["qr_id"] = qr_id

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=15.0)
            try:
                data = response.json()
            except Exception:
                data = {"raw_text": response.text}

            if response.status_code >= 400 or not isinstance(data, dict):
                err = (
                    _dig(data, "error", "message")
                    if isinstance(data, dict)
                    else response.text[:200]
                )
                logger.error(
                    "verifyqr_guest failed hotel=%s ref=%s status=%s err=%s",
                    hotel_id, reference, response.status_code, err,
                )
                return QrVerifyResult(error=str(err), raw=data if isinstance(data, dict) else None)

            status = _dig(data, "status", "qr_status")
            is_paid = (
                data.get("is_paid") is True
                or data.get("captured_payment") is True
                or (isinstance(status, str) and status.lower() in ("paid", "captured", "completed"))
            )

            amount_raw = _dig(data, "amount_paid", "payments_amount_received")
            amount_paid = None
            if amount_raw is not None:
                try:
                    amount_paid = float(amount_raw)
                except (TypeError, ValueError):
                    amount_paid = None
            # The spec reports `payments[].amount` in paise; `amount_paid` in
            # rupees. Only fall back to the array when the rupee field is absent.
            if amount_paid is None and isinstance(data.get("payments"), list):
                paise = sum(
                    float(p.get("amount") or 0)
                    for p in data["payments"]
                    if isinstance(p, dict)
                )
                if paise:
                    amount_paid = paise / 100.0

            return QrVerifyResult(
                is_paid=bool(is_paid),
                amount_paid=amount_paid,
                status=str(status) if status is not None else None,
                qr_id=_dig(data, "qr_id"),
                payment_id=_dig(data, "payment_id") or None,
                raw=data,
            )
    except Exception as exc:
        logger.error("verifyqr_guest error hotel=%s ref=%s err=%s", hotel_id, reference, exc)
        return QrVerifyResult(error=str(exc))
