"""
Tasks API Router — Operational hotel task endpoints.

Provides REST APIs for:
- Listing tasks with flexible filtering (status, department, priority, booking_id, phone)
- Getting full task details with audit log events and notes
- Creating operational tasks manually from staff dashboard / front desk
- Updating task status (OPEN -> ASSIGNED -> IN_PROGRESS -> COMPLETED / CANCELLED)
- Assigning tasks to staff members
- Adding internal staff notes
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.models.task_schemas import (
    HotelTask,
    SUBTYPE_TO_DEPARTMENT,
    TaskDepartment,
    TaskEvent,
    TaskPriority,
    TaskStatus,
    TaskSubtype,
    TaskType,
)
from app.repositories import task_repository
from app.services.task_notification_service import (
    dispatch_task_completed_notification,
    record_task_notified,
    resolve_booking_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


# ═══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class CreateTaskRequest(BaseModel):
    hotel_id: int | str | None = None
    session_id: str | None = None
    department: str | None = None
    task_type: str | None = None
    subtype: str = "OTHER"
    priority: str = "NORMAL"
    customer_request: str
    summary: str | None = None
    room_number: str | None = None
    booking_id: str | None = None
    customer_name: str | None = None
    customer_phone: str | None = None
    assigned_staff: str | None = None
    created_by: str = "staff"
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateTaskStatusRequest(BaseModel):
    status: str
    actor: str = "staff"
    note: str | None = None


class AssignTaskRequest(BaseModel):
    staff_name: str
    actor: str = "staff"


class AddTaskNoteRequest(BaseModel):
    note: str
    author: str = "staff"


class ConfirmPaymentTaskRequest(BaseModel):
    amount_paid: float | None = None
    actor: str = "staff"
    note: str | None = None


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("", summary="List hotel tasks with optional filtering")
@router.get("/", include_in_schema=False)
async def list_tasks(
    hotel_id: int | str | None = None,
    status: str | None = Query(None, description="OPEN, ASSIGNED, IN_PROGRESS, COMPLETED, CANCELLED"),
    department: str | None = Query(None, description="FRONT_DESK, HOUSEKEEPING, MAINTENANCE, ROOM_SERVICE, etc."),
    priority: str | None = Query(None, description="LOW, NORMAL, HIGH, URGENT"),
    subtype: str | None = Query(None, description="e.g. PAYMENT_CONFIRMATION, EXTRA_TOWELS, AC_REPAIR"),
    booking_id: str | None = Query(None, description="Filter by booking reference"),
    customer_phone: str | None = Query(None, description="Filter by customer phone"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
):
    """
    Fetch hotel operational tasks matching criteria with pagination and total count.
    """
    hid = hotel_id
    if hid is None:
        try:
            hid = get_settings().HOTEL_ID
        except Exception:
            hid = 111111

    tasks, total = await task_repository.list_tasks(
        hotel_id=hid,
        status=status,
        department=department,
        priority=priority,
        subtype=subtype,
        booking_id=booking_id,
        customer_phone=customer_phone,
        session_id=session_id,
        limit=limit,
        skip=skip,
    )
    return {
        "success": True,
        "total": total,
        "count": len(tasks),
        "limit": limit,
        "skip": skip,
        "tasks": tasks,
    }


@router.get("/{task_id}", summary="Get task details by ID")
async def get_task(task_id: str):
    """
    Fetch full details of a specific task including events, notes, and customer updates.
    """
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found",
        )
    return {
        "success": True,
        "task": task,
    }


@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a new hotel task")
@router.post("/", include_in_schema=False, status_code=status.HTTP_201_CREATED)
async def create_task(req: CreateTaskRequest):
    """
    Create a new operational task manually from staff portal or external integration.
    """
    hid = req.hotel_id
    if hid is None:
        try:
            hid = get_settings().HOTEL_ID
        except Exception:
            hid = 111111

    # Resolve department from subtype if not explicitly provided
    resolved_subtype = req.subtype.upper() if req.subtype else "OTHER"
    resolved_dept = req.department.upper() if req.department else SUBTYPE_TO_DEPARTMENT.get(resolved_subtype, "FRONT_DESK")
    resolved_type = req.task_type.upper() if req.task_type else resolved_dept
    resolved_priority = req.priority.upper() if req.priority else "NORMAL"

    task_id = await task_repository.generate_task_id(hid)
    session_id = req.session_id or f"manual_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    task_doc = HotelTask(
        task_id=task_id,
        hotel_id=str(hid),
        session_id=session_id,
        booking_id=req.booking_id,
        room_number=req.room_number,
        customer_name=req.customer_name,
        customer_phone=req.customer_phone,
        department=resolved_dept,
        task_type=resolved_type,
        subtype=resolved_subtype,
        priority=resolved_priority,
        status="ASSIGNED" if req.assigned_staff else "OPEN",
        summary=req.summary or f"{resolved_dept.replace('_', ' ').title()} task: {resolved_subtype.replace('_', ' ').title()}",
        customer_request=req.customer_request,
        assigned_staff=req.assigned_staff,
        created_by=req.created_by,
        events=[
            TaskEvent(
                event_type="CREATED",
                description=f"Task manually created by {req.created_by}: {resolved_subtype}",
                timestamp=now,
                actor=req.created_by,
                metadata={"manual_create": True},
            )
        ],
        created_at=now,
        updated_at=now,
        metadata=req.metadata,
    )

    created_id = await task_repository.create_task(task_doc)
    created_task = await task_repository.get_task(created_id)

    return {
        "success": True,
        "task_id": created_id,
        "message": "Task created successfully",
        "task": created_task,
    }


@router.patch("/{task_id}/status", summary="Update task status")
async def update_task_status(task_id: str, req: UpdateTaskStatusRequest):
    """
    Transition task status: OPEN, ASSIGNED, IN_PROGRESS, COMPLETED, CANCELLED.
    """
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found",
        )

    valid_statuses = {"OPEN", "ASSIGNED", "IN_PROGRESS", "COMPLETED", "CANCELLED"}
    new_status = req.status.upper()
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{req.status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
        )

    if req.note:
        await task_repository.add_task_note(task_id, req.note, author=req.actor)

    await task_repository.update_task_status(task_id, new_status, actor=req.actor)

    # When staff marks a task COMPLETED, tell the guest. The wording, the
    # booking confirmation and the dedupe ledger all live in the shared service,
    # because the CRM panel completes tasks by writing this collection directly
    # and never reaches this route — both doors must say the same thing.
    notification_dispatched = False
    if new_status == "COMPLETED":
        try:
            notification_dispatched = await dispatch_task_completed_notification(task)
            if notification_dispatched:
                await record_task_notified(task_id)
        except Exception as notify_err:
            logger.error(f"Failed to dispatch task completion notification: {notify_err}", exc_info=True)

    return {
        "success": True,
        "task_id": task_id,
        "status": new_status,
        "notification_dispatched": notification_dispatched,
        "message": f"Task status updated to {new_status}",
    }


@router.post("/{task_id}/confirm-payment", summary="Confirm payment task, verify booking, and send confirmation slip")
async def confirm_payment_task(task_id: str, req: ConfirmPaymentTaskRequest):
    """
    Staff / Reception confirms payment for a task.
    Marks task COMPLETED, updates booking to CONFIRMED, and sends official confirmation slip to guest via WhatsApp.
    """
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found",
        )

    actor = req.actor or "staff"
    if req.note:
        await task_repository.add_task_note(task_id, req.note, author=actor)

    await task_repository.update_task_status(task_id, "COMPLETED", actor=actor)

    # The dispatcher confirms the booking (with the amount staff stated, or the
    # one the task recorded when the receipt came in) and sends the slip. Same
    # code the sweep uses, so a panel completion and this button agree.
    booking_id = await resolve_booking_id(task, task.get("customer_phone"))
    notification_dispatched = False
    try:
        notification_dispatched = await dispatch_task_completed_notification(
            task, amount_override=req.amount_paid,
        )
        if notification_dispatched:
            await record_task_notified(task_id)
    except Exception as notify_err:
        logger.error(f"Failed to dispatch booking confirmation notification: {notify_err}", exc_info=True)

    return {
        "success": True,
        "task_id": task_id,
        "booking_id": booking_id,
        "status": "COMPLETED",
        "notification_dispatched": notification_dispatched,
        "message": "Payment verified, booking marked confirmed, and confirmation notification dispatched.",
    }


@router.post("/{task_id}/assign", summary="Assign task to a staff member")
async def assign_task(task_id: str, req: AssignTaskRequest):
    """
    Assign a task to a staff member and update status to ASSIGNED.
    """
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found",
        )

    await task_repository.assign_task(task_id, req.staff_name)

    return {
        "success": True,
        "task_id": task_id,
        "assigned_staff": req.staff_name,
        "status": "ASSIGNED",
        "message": f"Task assigned to {req.staff_name}",
    }


@router.post("/{task_id}/notes", summary="Add internal staff note to task")
async def add_task_note(task_id: str, req: AddTaskNoteRequest):
    """
    Add an internal staff note and audit event to a task.
    """
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found",
        )

    await task_repository.add_task_note(task_id, req.note, author=req.author)

    return {
        "success": True,
        "task_id": task_id,
        "message": "Note added successfully",
    }
