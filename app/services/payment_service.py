"""
Payment Service — StayInn Backend create-order / verify-payment (Razorpay).

Critical: always persist and verify with payment_link_id (plink_...),
never scrape short URLs like rzp.io.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_PLINK_RE = re.compile(r"(plink_[A-Za-z0-9]+)")


@dataclass
class PaymentCreateResult:
    """Successful (or failed) payment-link creation."""
    url: str | None = None
    payment_link_id: str | None = None
    order_id: str | None = None
    raw: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.url and self.payment_link_id)


@dataclass
class PaymentVerifyResult:
    """Gateway verification outcome."""
    is_paid: bool = False
    amount_paid: float | None = None
    status: str | None = None
    error: str | None = None
    raw: dict | None = None


def extract_plink_id(value: str | None) -> str | None:
    """Extract a Razorpay payment link id (plink_...) from a URL or raw id."""
    if not value:
        return None
    value = value.strip()
    if value.startswith("plink_"):
        return value.split("?")[0].split("&")[0]
    match = _PLINK_RE.search(value)
    return match.group(1) if match else None


def _dig(data: dict, *keys: str):
    """Return first present key from data or nested data.data."""
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested and nested[key] not in (None, ""):
                return nested[key]
    return None


async def create_payment_link(
    hotel_id: int,
    booking_id: str,
    amount: float,
) -> PaymentCreateResult:
    """
    Create a Razorpay payment order via StayInn backend.

    Endpoint: POST /api/payment-collections/{hotel_id}/create-order
    """
    settings = get_settings()
    url = (
        f"{settings.STAYINN_BACKEND_URL.rstrip('/')}"
        f"/api/payment-collections/{hotel_id}/create-order"
    )
    headers = {
        "X-Order-Key": settings.STAYINN_PAYMENT_ORDER_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "amount": amount,
        "gateway": "razorpay",
        "booking_id": booking_id,
    }

    from app.observability.metrics import metrics
    outcome = "error"
    with metrics.timer("payment.create_latency_ms", hotel_id=hotel_id):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=15.0)
                try:
                    data = response.json()
                except Exception:
                    data = {"raw_text": response.text}

                if response.status_code >= 400:
                    err = _dig(data, "error", "message") or response.text[:200]
                    logger.error(
                        f"create-order failed hotel={hotel_id} booking={booking_id} "
                        f"status={response.status_code} err={err}"
                    )
                    result = PaymentCreateResult(error=str(err), raw=data if isinstance(data, dict) else None)
                elif not isinstance(data, dict):
                    result = PaymentCreateResult(error="Invalid create-order response")
                else:
                    payment_url = _dig(
                        data,
                        "payment_link",
                        "payment_url",
                        "short_url",
                        "link",
                    )
                    payment_link_id = extract_plink_id(
                        str(
                            _dig(
                                data,
                                "payment_link_id",
                                "razorpay_payment_link_id",
                                "order_id",
                                "id",
                            )
                            or ""
                        )
                    ) or extract_plink_id(str(payment_url or ""))

                    if payment_url and payment_link_id:
                        logger.info(
                            f"Payment link created hotel={hotel_id} booking={booking_id} "
                            f"plink={payment_link_id}"
                        )
                        outcome = "ok"
                        result = PaymentCreateResult(
                            url=str(payment_url),
                            payment_link_id=payment_link_id,
                            order_id=str(_dig(data, "order_id") or payment_link_id),
                            raw=data,
                        )
                    else:
                        logger.error(f"Failed to extract payment link/id from response: {data}")
                        result = PaymentCreateResult(
                            url=str(payment_url) if payment_url else None,
                            payment_link_id=payment_link_id,
                            error="Missing payment_link or payment_link_id in response",
                            raw=data,
                        )
        except Exception as e:
            logger.error(f"Error creating payment link via backend: {e}")
            result = PaymentCreateResult(error=str(e))

    # NO FABRICATED FALLBACK. This used to invent
    # `https://rzp.io/l/<booking_id>` and a `plink_...` id for *any* failure, and
    # because `PaymentCreateResult.ok` is just `bool(self.url and
    # self.payment_link_id)`, that made `ok` unconditionally true. Two things
    # followed, both bad:
    #
    #   1. The caller's `if not create_result.ok:` branch in
    #      `app/workflows/nodes/payment/link.py` was unreachable, so the
    #      guest-facing "I couldn't generate a payment link" message could never
    #      be shown.
    #   2. A guest whose gateway call genuinely failed received a link that goes
    #      nowhere. Nothing recorded that it was fake, so the failure was silent.
    #
    # Reporting the failure honestly is the whole point: `link.py` handles a real
    # failure by telling the guest to retry. Failing to a *dead link* was
    # strictly worse than failing loudly.
    if not result.ok:
        logger.error(
            f"PAYMENT_LINK_CREATE_FAILED | hotel={hotel_id} booking={booking_id} "
            f"err={result.error} — no link fabricated; caller must report the failure"
        )

    metrics.incr("payment.create_total", outcome=outcome, hotel_id=hotel_id)
    return result


async def verify_payment_status(
    hotel_id: int,
    *,
    payment_link_id: str | None = None,
    payment_link: str | None = None,
) -> PaymentVerifyResult:
    """
    Verify payment status via StayInn backend.

    Always prefers an explicit plink_ id. Falls back to extracting one from
    the payment URL only if it contains plink_ (short rzp.io URLs will fail —
    callers must persist payment_link_id at create time).
    """
    settings = get_settings()

    from app.observability.metrics import metrics

    plink_id = extract_plink_id(payment_link_id) or extract_plink_id(payment_link)
    if not plink_id:
        logger.error(
            f"verify-payment missing plink_ id hotel={hotel_id} "
            f"link_id={payment_link_id!r} link={payment_link!r}"
        )
        metrics.incr("payment.verify_total", outcome="missing_id", hotel_id=hotel_id)
        return PaymentVerifyResult(
            is_paid=False,
            error="Missing payment_link_id (plink_...). Cannot verify short URLs alone.",
        )

    url = (
        f"{settings.STAYINN_BACKEND_URL.rstrip('/')}"
        f"/api/payment-collections/{hotel_id}/verify-payment"
    )
    headers = {
        "X-Order-Key": settings.STAYINN_PAYMENT_ORDER_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        # Live StayInn backend expects order_id (= plink_...)
        "order_id": plink_id,
        "payment_link_id": plink_id,
    }

    outcome = "error"
    with metrics.timer("payment.verify_latency_ms", hotel_id=hotel_id):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=15.0)
                try:
                    data = response.json()
                except Exception:
                    data = {"raw_text": response.text}

                if response.status_code >= 400:
                    err = _dig(data, "error", "message") if isinstance(data, dict) else response.text[:200]
                    logger.error(
                        f"verify-payment failed hotel={hotel_id} plink={plink_id} "
                        f"status={response.status_code} err={err}"
                    )
                    result = PaymentVerifyResult(
                        is_paid=False,
                        error=str(err),
                        raw=data if isinstance(data, dict) else None,
                    )
                elif not isinstance(data, dict):
                    result = PaymentVerifyResult(is_paid=False, error="Invalid verify response")
                else:
                    status = _dig(data, "status", "payment_status")
                    is_paid = (
                        _dig(data, "is_paid") is True
                        or _dig(data, "captured_payment") is True
                        or (
                            isinstance(status, str)
                            and status.lower() in ("paid", "captured", "success", "completed")
                        )
                    )

                    amount_raw = _dig(
                        data,
                        "amount_paid",
                        "amount",
                        "paid_amount",
                        "payment_amount",
                    )
                    amount_paid = None
                    if amount_raw is not None:
                        try:
                            amount_paid = float(amount_raw)
                        except (TypeError, ValueError):
                            amount_paid = None

                    outcome = "paid" if is_paid else "unpaid"
                    result = PaymentVerifyResult(
                        is_paid=bool(is_paid),
                        amount_paid=amount_paid,
                        status=str(status) if status is not None else None,
                        raw=data,
                    )

        except Exception as e:
            logger.error(f"Error verifying payment status via backend: {e}")
            result = PaymentVerifyResult(is_paid=False, error=str(e))

    metrics.incr("payment.verify_total", outcome=outcome, hotel_id=hotel_id)
    return result
