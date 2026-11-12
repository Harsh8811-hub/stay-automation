"""Stage Handlers for StayInn Booking-Stage-Based Lifecycle Architecture."""

from __future__ import annotations

import logging
from typing import Any

from app.models.task_schemas import HotelTask
from app.repositories import task_repository

logger = logging.getLogger(__name__)


async def create_stage_task(
    *,
    hotel_id: int | str,
    session_id: str = "",
    title: str,
    description: str,
    task_type: str = "FRONT_DESK",
    subtype: str = "OTHER",
    department: str = "FRONT_DESK",
    priority: str = "NORMAL",
    booking_id: str | None = None,
    room_number: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Helper to generate readable task ID and create canonical HotelTask."""
    task_id = await task_repository.generate_task_id(hotel_id)
    task = HotelTask(
        task_id=task_id,
        hotel_id=hotel_id,
        session_id=session_id or f"sess_{customer_phone or 'anon'}",
        booking_id=booking_id if booking_id not in ("BK-PENDING", "BK-PAST", None) else None,
        room_number=room_number,
        customer_name=customer_name,
        customer_phone=customer_phone,
        department=department,
        task_type=task_type,
        subtype=subtype,
        priority=priority,
        status="OPEN",
        summary=title,
        customer_request=description,
        metadata={
            **(metadata or {}),
            "task_type": task_type,
        },
    )

    # This is the second of the two payment-task creators, and until now it
    # recorded nothing at all: no counter, no failure entry. Every caller wraps
    # it in a broad `except Exception` (see `stage_payment_check`), so a task
    # that failed to insert was silently absent from both the CRM queue and
    # `task.create_total` — the metric said "no tasks lost" because it never saw
    # the attempt. The same counter the task engine writes is used here, with
    # the same labels, so a created task counts once whichever path created it.
    from app.observability.metrics import metrics

    try:
        created = await task_repository.create_task(task)
    except Exception as e:
        from app.observability.failure_store import record_failure

        metrics.incr("task.create_total", outcome="error", hotel_id=hotel_id)
        record_failure(
            component="task.create",
            error=e,
            hotel_id=hotel_id,
            phone=customer_phone,
            details={"subtype": subtype, "session_id": task.session_id},
            exc=e,
        )
        raise

    metrics.incr(
        "task.create_total",
        outcome="ok",
        hotel_id=hotel_id,
        department=department,
    )
    return created
