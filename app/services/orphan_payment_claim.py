"""
A guest says he has paid, and there is no booking of ours to check it against.

Measured 2026-10-27, guest 919999000111. Staff took the booking in Yanolja — the
hotel's own PMS — and sent the voucher as a PDF. `stayinn_web.bookings` has no
row for him, no `payments` row and no `leads` row, so `stage_payment_check`'s
`update_one` matched nothing, its result was never inspected, and it logged
success regardless. Eight `PAYMENT_CONFIRMATION` tasks were raised over two days,
all with `booking_id: None`, all still open, all asking the front desk to verify
a payment against nothing.

The front desk did not need a fourth thing to verify. It needed one sentence:
*this guest says he paid and we have no booking for him.* This module writes it,
once per conversation, and records the dropped write so the next investigation
does not have to rediscover that the update went nowhere.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SOURCE = "orphan_payment_claim"


def _hotel_variants(hotel_id: Any) -> list[Any]:
    variants: list[Any] = [str(hotel_id)]
    if str(hotel_id).isdigit():
        variants.append(int(hotel_id))
    return variants


async def booking_exists(hotel_id: Any, phone: str | None, booking_id: str | None) -> bool:
    """
    Is there any booking row this claim could belong to?

    Deliberately unfiltered by status: a settled booking is a *different*
    problem (the guest is confirmed; nothing is orphaned), and only the total
    absence of a row is what this guard is for.
    """
    if not phone and not booking_id:
        return False
    from app.db.mongodb import bookings_collection

    flt_or: list[dict[str, Any]] = []
    if phone:
        flt_or.append({"customer_phone": phone})
    if booking_id:
        flt_or.append({"booking_id": booking_id})
    doc = await bookings_collection().find_one(
        {"hotel_id": {"$in": _hotel_variants(hotel_id)}, "$or": flt_or},
        {"_id": 1},
    )
    return doc is not None


async def handle_orphan_claim(
    hotel_id: Any,
    session_id: str | None,
    phone: str | None,
    booking_id: str | None,
    guest_message: str,
) -> str | None:
    """
    Record the dropped write and raise one front-desk task. Returns the task id.

    Deduped in the database rather than with a state flag: the flag would not
    survive to the next turn (LangGraph drops state keys not declared in
    `ConversationState`), and this handler runs on every message the guest
    sends while the payment sits unverified.
    """
    from app.db.mongodb import hotel_tasks_collection

    try:
        from app.observability.failure_store import record_failure
        record_failure(
            component="stage_payment_check.orphan_claim",
            error="payment claim matched no booking row — the $set wrote nothing",
            hotel_id=str(hotel_id),
            session_id=session_id,
            phone=phone,
            details={"booking_id": booking_id},
        )
    except Exception as e:  # noqa: BLE001 — observability must never block the task
        logger.warning("ORPHAN_CLAIM_RECORD_FAILED | %s", e)

    identity = []
    if session_id:
        identity.append({"session_id": session_id})
    if phone:
        identity.append({"customer_phone": phone})
    if not identity:
        identity.append({"metadata.orphan_phone": phone})

    existing = await hotel_tasks_collection().find_one(
        {
            "hotel_id": {"$in": _hotel_variants(hotel_id)},
            "metadata.source": SOURCE,
            "status": {"$in": ["OPEN", "ASSIGNED", "IN_PROGRESS"]},
            "$or": identity,
        }
    )
    if existing:
        return existing.get("task_id")

    from app.workflows.nodes.stage_handlers import create_stage_task

    task_id = await create_stage_task(
        hotel_id=hotel_id,
        session_id=session_id or "",
        title="Guest says he has paid — no booking record in StayInn",
        description=(
            "This guest claims a payment, but there is no booking row for the "
            f"phone {phone or '(none)'} or reference {booking_id or '(none)'} in "
            "our database, so the bot cannot verify anything. The stay may have "
            f"been booked in the hotel's own system. Guest's message: "
            f"'{guest_message[:300]}'. Check the PMS and the bank, then reply to "
            "the guest directly — do not ask them to pay again."
        ),
        task_type="FRONT_DESK",
        subtype="PAYMENT_CONFIRMATION",
        department="FRONT_DESK",
        priority="URGENT",
        booking_id=booking_id,
        customer_phone=phone,
        metadata={
            "source": SOURCE,
            "orphan_phone": phone,
            "guest_message": guest_message[:300],
        },
    )
    logger.warning(
        "ORPHAN_PAYMENT_CLAIM | hotel=%s phone=%s booking=%s task=%s — no booking "
        "row; front desk asked to reconcile",
        hotel_id, phone or "NONE", booking_id or "NONE", task_id,
    )
    return task_id
