"""
Escalation & Support Tickets API Router.

Provides REST APIs for:
- Listing escalations / support tickets with filtering (status, priority, issue_type, phone)
- Getting full escalation details (conversation history, events, staff notes, smart replies)
- Creating escalation tickets manually or from external channels
- Updating status (OPEN -> IN_PROGRESS -> RESOLVED -> CLOSED)
- Quick resolution with resolution notes
- Assigning tickets to staff members
- Adding internal staff notes (never exposed to guests)
- Sending staff replies / conversation messages (with human takeover activation)
- Getting and regenerating staff smart replies

Exported routers:
- `router`: Mounted at `/api/escalations`
- `tickets_router`: Mounted at `/api/tickets`
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.models.escalation_schemas import (
    EscalationCategory,
    ReasonCode,
    StaffNote,
    SupportTicket,
    TicketEvent,
    TicketMessage,
    TicketPriority,
    TicketStatus,
)
from app.repositories import ticket_repository
from app.services.ticket_service import (
    create_or_update_ticket,
    generate_escalation_smart_replies,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/escalations", tags=["escalations"])
tickets_router = APIRouter(prefix="/api/tickets", tags=["tickets"])


# ═══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class CreateEscalationRequest(BaseModel):
    hotel_id: int | str | None = None
    session_id: str | None = None
    user_message: str
    issue_type: str = "OTHER"
    reason_code: str = "OTHER"
    priority: str = "NORMAL"
    summary: str | None = None
    customer_name: str | None = None
    customer_phone: str | None = None
    customer_email: str | None = None
    booking_id: str | None = None
    booking_state: dict[str, Any] = Field(default_factory=dict)


class UpdateEscalationStatusRequest(BaseModel):
    status: str
    resolution: str | None = None
    actor: str = "staff"


class ResolveEscalationRequest(BaseModel):
    resolution: str
    actor: str = "staff"


class AssignEscalationRequest(BaseModel):
    staff_name: str
    actor: str = "staff"


class AddEscalationNoteRequest(BaseModel):
    content: str
    author: str = "staff"


class ReplyEscalationRequest(BaseModel):
    message: str
    sender: str = "staff"
    notify_guest: bool = False


# ═══════════════════════════════════════════════════════════════════
# CORE ENDPOINTS (Registered on both /api/escalations and /api/tickets)
# ═══════════════════════════════════════════════════════════════════

@router.get("", summary="List support tickets / escalations with optional filtering")
@router.get("/", include_in_schema=False)
@tickets_router.get("", summary="List support tickets / escalations with optional filtering")
@tickets_router.get("/", include_in_schema=False)
async def list_escalations(
    hotel_id: int | str | None = None,
    status: str | None = Query(None, description="OPEN, IN_PROGRESS, RESOLVED, CLOSED, EXPIRED"),
    priority: str | None = Query(None, description="CRITICAL, HIGH, NORMAL, LOW"),
    issue_type: str | None = Query(None, description="PAYMENT_PROBLEM, BOOKING_DISPUTE, COMPLAINT, etc."),
    reason_code: str | None = Query(None, description="STOP_SOLD_ROOM_REQUEST, PAYMENT_PENDING, etc."),
    customer_phone: str | None = Query(None, description="Filter by customer phone"),
    booking_id: str | None = Query(None, description="Filter by booking ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
):
    """
    Fetch support tickets matching criteria with pagination and total count.
    """
    hid = hotel_id
    if hid is None:
        try:
            hid = get_settings().HOTEL_ID
        except Exception:
            hid = 111111

    tickets, total = await ticket_repository.list_tickets(
        hotel_id=hid,
        status=status,
        priority=priority,
        issue_type=issue_type,
        reason_code=reason_code,
        customer_phone=customer_phone,
        booking_id=booking_id,
        session_id=session_id,
        limit=limit,
        skip=skip,
    )
    return {
        "success": True,
        "total": total,
        "count": len(tickets),
        "limit": limit,
        "skip": skip,
        "tickets": tickets,
    }


@router.get("/{ticket_id}", summary="Get escalation ticket details by ID")
@tickets_router.get("/{ticket_id}", summary="Get escalation ticket details by ID")
async def get_escalation(ticket_id: str):
    """
    Fetch full details of a specific escalation ticket including conversation, notes, and events.
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )
    return {
        "success": True,
        "ticket": ticket,
    }


@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a new support ticket / escalation")
@router.post("/", include_in_schema=False, status_code=status.HTTP_201_CREATED)
@tickets_router.post("", status_code=status.HTTP_201_CREATED, summary="Create a new support ticket / escalation")
@tickets_router.post("/", include_in_schema=False, status_code=status.HTTP_201_CREATED)
async def create_escalation(req: CreateEscalationRequest):
    """
    Create a new escalation ticket or append to an existing open ticket for the same customer/session.
    """
    hid = req.hotel_id
    if hid is None:
        try:
            hid = get_settings().HOTEL_ID
        except Exception:
            hid = 111111

    sid = req.session_id or f"manual_{uuid.uuid4().hex[:12]}"
    summary_text = req.summary or f"Manual escalation: {req.issue_type} - {req.reason_code}"

    ticket_id, is_new = await create_or_update_ticket(
        hotel_id=int(hid) if str(hid).isdigit() else 111111,
        session_id=sid,
        user_message=req.user_message,
        issue_type=req.issue_type.upper(),
        reason_code=req.reason_code.upper(),
        priority=req.priority.upper(),
        summary=summary_text,
        booking_state=req.booking_state,
        customer_name=req.customer_name,
        customer_phone=req.customer_phone,
        customer_email=req.customer_email,
        booking_id=req.booking_id,
    )

    ticket = await ticket_repository.get_ticket(ticket_id)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "is_new": is_new,
        "message": "Ticket created successfully" if is_new else "Appended to existing open ticket",
        "ticket": ticket,
    }


async def _notify_escalation_resolved(ticket: dict, resolution: str | None = None) -> bool:
    """Automatically send WhatsApp confirmation to guest when escalation ticket is resolved.

    The message text is shared with the panel sweep
    (`escalation_notification_service.resolution_message`) so a ticket resolved
    through this route and one resolved in the panel read the same to the guest.

    The ledger marker is written only when the send actually went out. A failed
    send must stay unrecorded so the sweep retries it — in the October provider
    outage every message failed, and marking those "sent" would have turned an
    outage into a permanent silence. The sweep's claim is what stops a retry
    from becoming a duplicate once one of them succeeds.
    """
    from app.services import escalation_notification_service

    ticket_id = ticket.get("ticket_id")
    customer_phone = ticket.get("customer_phone")
    if not customer_phone:
        return False
    from app.services.whatsapp_service import whatsapp_service
    hotel_id = ticket.get("hotel_id", 111111)
    try:
        sent = bool(whatsapp_service.send_message(
            to_number=customer_phone,
            message_body=escalation_notification_service.resolution_message(ticket, resolution),
            hotel_id=hotel_id,
        ))
    except Exception as e:
        logger.error(f"Failed to dispatch escalation resolution notification: {e}")
        sent = False
    if sent:
        await escalation_notification_service.record_ticket_notified(ticket_id)
    else:
        logger.warning(
            "ESCALATION_RESOLVE_NOTIFY_FAILED | ticket=%s | left for the sweep to retry",
            ticket_id,
        )
    return sent


@router.patch("/{ticket_id}/status", summary="Update escalation ticket status")
@tickets_router.patch("/{ticket_id}/status", summary="Update escalation ticket status")
async def update_escalation_status(ticket_id: str, req: UpdateEscalationStatusRequest):
    """
    Transition ticket status: OPEN, IN_PROGRESS, RESOLVED, CLOSED, EXPIRED.
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )

    valid_statuses = {"OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED", "EXPIRED"}
    new_status = req.status.upper()
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{req.status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
        )

    await ticket_repository.update_ticket_status(
        ticket_id=ticket_id,
        new_status=new_status,
        resolution=req.resolution,
    )

    notification_dispatched = False
    if new_status == "RESOLVED":
        notification_dispatched = await _notify_escalation_resolved(ticket, req.resolution)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": new_status,
        "resolution": req.resolution,
        "notification_dispatched": notification_dispatched,
        "message": f"Ticket status updated to {new_status}",
    }


@router.post("/{ticket_id}/resolve", summary="Quick resolve an escalation ticket")
@tickets_router.post("/{ticket_id}/resolve", summary="Quick resolve an escalation ticket")
async def resolve_escalation(ticket_id: str, req: ResolveEscalationRequest):
    """
    Quick resolution endpoint: marks ticket as RESOLVED with explanation.
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )

    await ticket_repository.update_ticket_status(
        ticket_id=ticket_id,
        new_status="RESOLVED",
        resolution=req.resolution,
    )

    notification_dispatched = await _notify_escalation_resolved(ticket, req.resolution)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "RESOLVED",
        "resolution": req.resolution,
        "notification_dispatched": notification_dispatched,
        "message": "Ticket resolved successfully",
    }


@router.post("/{ticket_id}/assign", summary="Assign escalation ticket to staff")
@tickets_router.post("/{ticket_id}/assign", summary="Assign escalation ticket to staff")
async def assign_escalation(ticket_id: str, req: AssignEscalationRequest):
    """
    Assign ticket to staff member and transition to IN_PROGRESS.
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )

    await ticket_repository.assign_ticket(ticket_id, req.staff_name, actor=req.actor)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "assigned_to": req.staff_name,
        "status": "IN_PROGRESS",
        "message": f"Ticket assigned to {req.staff_name}",
    }


@router.post("/{ticket_id}/notes", summary="Add internal staff note")
@tickets_router.post("/{ticket_id}/notes", summary="Add internal staff note")
async def add_escalation_note(ticket_id: str, req: AddEscalationNoteRequest):
    """
    Add internal staff note (never displayed to guest).
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )

    await ticket_repository.add_staff_note(ticket_id, req.content, author=req.author)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "message": "Staff note added successfully",
    }


@router.post("/{ticket_id}/reply", summary="Send staff reply to guest on ticket")
@router.post("/{ticket_id}/messages", include_in_schema=False)
@tickets_router.post("/{ticket_id}/reply", summary="Send staff reply to guest on ticket")
@tickets_router.post("/{ticket_id}/messages", include_in_schema=False)
async def reply_escalation(ticket_id: str, req: ReplyEscalationRequest):
    """
    Append a staff reply to the ticket conversation.
    If notify_guest is true and a chat session exists, activates/extends Human Takeover.
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )

    await ticket_repository.add_ticket_message(
        ticket_id=ticket_id,
        role="staff",
        content=req.message,
        actor=req.sender,
    )

    takeover_activated = False
    session_id = ticket.get("session_id")
    hotel_id = ticket.get("hotel_id")

    if req.notify_guest and session_id and hotel_id:
        try:
            from app.services.takeover_engine import start_or_update_intervention
            hid = int(hotel_id) if str(hotel_id).isdigit() else 111111
            await start_or_update_intervention(
                hotel_id=hid,
                session_id=session_id,
                staff_id=req.sender,
                staff_message=req.message,
                reason="ESCALATION_REPLY",
            )
            takeover_activated = True
        except Exception as e:
            logger.warning(f"Could not trigger takeover on escalation reply: {e}")

    return {
        "success": True,
        "ticket_id": ticket_id,
        "message": "Staff reply recorded successfully",
        "takeover_active": takeover_activated,
    }


@router.get("/{ticket_id}/smart-replies", summary="Get suggested smart replies for staff")
@tickets_router.get("/{ticket_id}/smart-replies", summary="Get suggested smart replies for staff")
async def get_escalation_smart_replies(ticket_id: str):
    """
    Get 3 high-probability suggested replies for staff to answer or resolve this ticket quickly.
    """
    ticket = await ticket_repository.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID '{ticket_id}' not found",
        )

    existing_replies = ticket.get("smart_replies")
    if existing_replies and len(existing_replies) > 0:
        return {
            "success": True,
            "ticket_id": ticket_id,
            "smart_replies": existing_replies,
        }

    generated = generate_escalation_smart_replies(
        issue_type=ticket.get("issue_type", "OTHER"),
        reason_code=ticket.get("reason_code", "OTHER"),
        user_message=ticket.get("last_user_message", ""),
        summary=ticket.get("summary", ""),
        customer_name=ticket.get("customer_name"),
    )
    await ticket_repository.update_smart_replies(ticket_id, generated)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "smart_replies": generated,
    }
