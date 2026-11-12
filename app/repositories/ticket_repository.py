"""
Ticket Repository — MongoDB CRUD for canonical `support_tickets` collection.

Handles:
- Readable ticket ID generation (SUP-YYYYMMDD-NNNNNN)
- Duplicate ticket detection
- Append-only event logging
- Ticket status management
- Stale ticket expiry
- Canonical guest_id linkage
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db.mongodb import support_tickets_collection
from app.models.activity_schemas import ActivityLog
from app.models.escalation_schemas import (
    StaffNote,
    SupportTicket,
    TicketEvent,
    TicketMessage,
)
from app.repositories import guest_repository
from app.repositories.activity_repository import log_activity

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# INDEXES
# ═══════════════════════════════════════════════════════════════════

async def ensure_indexes() -> None:
    """Create indexes for support_tickets. Idempotent."""
    coll = support_tickets_collection()
    await coll.create_index("ticket_id", unique=True)
    await coll.create_index([("hotel_id", 1), ("session_id", 1)])
    await coll.create_index([("hotel_id", 1), ("guest_id", 1)])
    await coll.create_index([("hotel_id", 1), ("customer_phone", 1)])
    await coll.create_index([("hotel_id", 1), ("status", 1)])
    await coll.create_index("created_at")
    await coll.create_index("expires_at")
    logger.info("support_tickets indexes ensured")


# ═══════════════════════════════════════════════════════════════════
# TICKET ID GENERATION
# ═══════════════════════════════════════════════════════════════════

async def generate_ticket_id(hotel_id: int | str) -> str:
    """
    Generate a readable ticket ID: SUP-YYYYMMDD-NNNNNN
    Uses today's date + the highest sequence already issued today, across ALL hotels.

    The sequence is deliberately NOT scoped to `hotel_id`, even though the parameter is
    still accepted. The `ticket_id` unique index (see ensure_indexes above) is
    collection-wide, so a per-hotel sequence made the sequence domain and the uniqueness
    domain disagree. A hotel whose own sequence was empty asked for -000001, which any
    other hotel may already hold for the same day — the insert then failed with E11000 on
    the first ticket that hotel raised that day, every time, and because the insert never
    succeeded that hotel never advanced and never recovered. Using the global high-water
    mark is what makes the generated id consistent with the constraint that enforces it.

    `hotel_id` is kept in the signature because callers pass it and an id that encodes
    the tenant would break the format of the tickets already in the database.

    This is still a read-then-write, so two concurrent callers can compute the same
    sequence. `create_or_update_ticket` retries on DuplicateKeyError, which is what
    makes the assignment safe rather than merely likely — and because a losing insert
    raises the high-water mark, the retry recomputes a genuinely new id.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"SUP-{today}-"
    coll = support_tickets_collection()

    # Highest sequence already issued today — NOT the row count. Counting assumes the day's ids
    # are gap-free, so deleting a single row (a bad ticket, a cleanup) makes count+1 recompute an
    # id that already exists: with -000002 and -000003 left, count=2 asks for -000003 again. The
    # retry loop cannot rescue that, because count_documents returns the same value on every
    # attempt, so the id collides all five times and the guest's turn fails. max()+1 only ever
    # moves forward, so a gap is harmless.
    #
    # Sorting by ticket_id descending is safe because the sequence is zero-padded to a fixed
    # width, which makes lexicographic order match numeric order.
    latest = await coll.find_one(
        {"ticket_id": {"$regex": f"^{prefix}"}},
        {"ticket_id": 1},
        sort=[("ticket_id", -1)],
    )
    try:
        next_sequence = int(latest["ticket_id"][len(prefix):]) + 1
    except (TypeError, KeyError, ValueError):
        # No tickets today, or a row whose suffix is not a number. Either way, start at 1.
        next_sequence = 1

    return f"{prefix}{str(next_sequence).zfill(6)}"


# ═══════════════════════════════════════════════════════════════════
# DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════

async def find_open_ticket_by_session(hotel_id: int | str, session_id: str) -> dict[str, Any] | None:
    """Find existing OPEN/IN_PROGRESS ticket for this session."""
    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    coll = support_tickets_collection()
    doc = await coll.find_one({
        "hotel_id": {"$in": variants},
        "session_id": session_id,
        "status": {"$in": ["OPEN", "IN_PROGRESS"]},
    })
    if doc:
        doc.pop("_id", None)
    return doc


async def find_open_ticket_by_customer(hotel_id: int | str, customer_phone: str) -> dict[str, Any] | None:
    """Find existing OPEN/IN_PROGRESS ticket for a customer across sessions."""
    if not customer_phone:
        return None

    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    from app.utils.phone import phone_lookup_variants
    phone_vars = phone_lookup_variants(customer_phone) or [customer_phone]

    coll = support_tickets_collection()
    doc = await coll.find_one({
        "hotel_id": {"$in": variants},
        "customer_phone": {"$in": phone_vars},
        "status": {"$in": ["OPEN", "IN_PROGRESS"]},
    })
    if doc:
        doc.pop("_id", None)
    return doc


# ═══════════════════════════════════════════════════════════════════
# CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════════════

async def create_ticket(ticket: SupportTicket) -> str:
    """
    Insert a new support ticket into MongoDB.
    Returns the ticket_id.
    """
    coll = support_tickets_collection()
    data = ticket.model_dump(mode="json")
    str_hid = str(ticket.hotel_id)
    data["hotel_id"] = str_hid

    # Link canonical guest if phone available
    if ticket.customer_phone and not data.get("guest_id"):
        guest = await guest_repository.get_or_create_guest(
            hotel_id=str_hid,
            phone=ticket.customer_phone,
            name=ticket.customer_name,
            email=ticket.customer_email,
        )
        data["guest_id"] = guest["guest_id"]

    await coll.insert_one(data)

    # Log Activity
    await log_activity(ActivityLog(
        hotel_id=int(str_hid) if str_hid.isdigit() else 111111,
        activity_type="TICKET_CREATED",
        reference_id=ticket.ticket_id,
        booking_id=ticket.booking_id,
        customer_phone=ticket.customer_phone,
        customer_name=ticket.customer_name,
        description=f"Support Ticket created for {ticket.issue_type.replace('_', ' ').title()}"
    ))

    logger.info(f"Ticket created: {ticket.ticket_id} | Type: {ticket.issue_type} | Priority: {ticket.priority}")
    return ticket.ticket_id


async def get_ticket(ticket_id: str) -> dict[str, Any] | None:
    """Fetch a ticket by its readable ID."""
    coll = support_tickets_collection()
    doc = await coll.find_one({"ticket_id": ticket_id})
    if doc:
        doc.pop("_id", None)
    return doc


async def append_customer_message(ticket_id: str, message: str) -> None:
    """Append a new customer message to the ticket conversation."""
    coll = support_tickets_collection()
    now = datetime.now(timezone.utc)

    msg = TicketMessage(role="customer", content=message, timestamp=now)
    event = TicketEvent(
        event_type="CUSTOMER_REPLIED",
        description=f"Customer added message: {message[:100]}...",
        timestamp=now,
        actor="customer",
    )

    await coll.update_one(
        {"ticket_id": ticket_id},
        {
            "$push": {
                "conversation": msg.model_dump(mode="json"),
                "events": event.model_dump(mode="json"),
            },
            "$set": {
                "updated_at": now,
                "last_user_message": message,
            },
        },
    )
    logger.info(f"Customer message appended to ticket: {ticket_id}")


async def add_staff_note(ticket_id: str, note: str, author: str = "system") -> None:
    """Add an internal staff note."""
    coll = support_tickets_collection()
    now = datetime.now(timezone.utc)

    staff_note = StaffNote(author=author, content=note, timestamp=now)
    event = TicketEvent(
        event_type="NOTE_ADDED",
        description=f"Staff note by {author}: {note[:100]}...",
        timestamp=now,
        actor=author,
    )

    await coll.update_one(
        {"ticket_id": ticket_id},
        {
            "$push": {
                "staff_notes": staff_note.model_dump(mode="json"),
                "events": event.model_dump(mode="json"),
            },
            "$set": {"updated_at": now},
        },
    )
    logger.info(f"Staff note added to ticket: {ticket_id}")


async def update_ticket_status(
    ticket_id: str,
    new_status: str,
    resolution: str | None = None,
) -> None:
    """Update ticket status with audit event."""
    coll = support_tickets_collection()
    now = datetime.now(timezone.utc)

    update_fields: dict[str, Any] = {
        "status": new_status,
        "updated_at": now,
    }
    if resolution:
        update_fields["resolution"] = resolution
    if new_status == "RESOLVED":
        update_fields["resolved_at"] = now

    event = TicketEvent(
        event_type="STATUS_CHANGED",
        description=f"Status changed to {new_status}",
        timestamp=now,
        actor="system",
        metadata={"new_status": new_status, "resolution": resolution},
    )

    await coll.update_one(
        {"ticket_id": ticket_id},
        {
            "$set": update_fields,
            "$push": {"events": event.model_dump(mode="json")},
        },
    )

    doc = await get_ticket(ticket_id)
    if doc:
        hid = doc.get("hotel_id", 111111)
        await log_activity(ActivityLog(
            hotel_id=int(hid) if str(hid).isdigit() else 111111,
            activity_type="TICKET_RESOLVED" if new_status == "RESOLVED" else "TICKET_UPDATED",
            reference_id=ticket_id,
            booking_id=doc.get("booking_id"),
            customer_phone=doc.get("customer_phone"),
            customer_name=doc.get("customer_name"),
            description=f"Ticket {ticket_id} status changed to {new_status}"
        ))

    logger.info(f"Ticket {ticket_id} status -> {new_status}")


async def expire_stale_tickets() -> int:
    """
    Mark tickets as EXPIRED if they have been OPEN/IN_PROGRESS
    for more than 7 days without activity.
    """
    coll = support_tickets_collection()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)

    result = await coll.update_many(
        {
            "status": {"$in": ["OPEN", "IN_PROGRESS"]},
            "updated_at": {"$lt": cutoff},
        },
        {
            "$set": {"status": "EXPIRED", "updated_at": now},
            "$push": {
                "events": TicketEvent(
                    event_type="STATUS_CHANGED",
                    description="Ticket expired due to 7 days of inactivity",
                    timestamp=now,
                    actor="system",
                ).model_dump(mode="json"),
            },
        },
    )

    if result.modified_count > 0:
        logger.info(f"Expired {result.modified_count} stale tickets")
    return result.modified_count


async def update_smart_replies(ticket_id: str, smart_replies: list[str]) -> None:
    """Update suggested quick replies for a ticket."""
    coll = support_tickets_collection()
    await coll.update_one(
        {"ticket_id": ticket_id},
        {"$set": {"smart_replies": smart_replies, "updated_at": datetime.now(timezone.utc)}},
    )


async def list_tickets(
    hotel_id: int | str | None = None,
    status: str | None = None,
    priority: str | None = None,
    issue_type: str | None = None,
    reason_code: str | None = None,
    customer_phone: str | None = None,
    booking_id: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    skip: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch support tickets matching optional filters with total matching count.
    Supports tenant isolation via hotel_id variants.
    """
    coll = support_tickets_collection()
    query: dict[str, Any] = {}

    if hotel_id is not None:
        str_hid = str(hotel_id)
        variants: list[Any] = [str_hid]
        if str_hid.isdigit():
            variants.append(int(str_hid))
        query["hotel_id"] = {"$in": variants}

    if status:
        query["status"] = status.upper()
    if priority:
        query["priority"] = priority.upper()
    if issue_type:
        query["issue_type"] = issue_type.upper()
    if reason_code:
        query["reason_code"] = reason_code.upper()
    if booking_id:
        query["booking_id"] = booking_id
    if session_id:
        query["session_id"] = session_id
    if customer_phone:
        from app.utils.phone import phone_lookup_variants
        phone_vars = phone_lookup_variants(customer_phone) or [customer_phone]
        query["customer_phone"] = {"$in": phone_vars}

    total = await coll.count_documents(query)
    tickets: list[dict[str, Any]] = []
    cursor = coll.find(query, sort=[("created_at", -1)]).skip(skip).limit(limit)
    async for doc in cursor:
        doc.pop("_id", None)
        tickets.append(doc)
    return tickets, total


async def assign_ticket(ticket_id: str, staff_name: str, actor: str = "system") -> None:
    """Assign a support ticket to a staff member."""
    coll = support_tickets_collection()
    now = datetime.now(timezone.utc)

    event = TicketEvent(
        event_type="ASSIGNED",
        description=f"Ticket assigned to {staff_name}",
        timestamp=now,
        actor=actor,
        metadata={"assigned_to": staff_name},
    )

    await coll.update_one(
        {"ticket_id": ticket_id},
        {
            "$set": {
                "assigned_to": staff_name,
                "status": "IN_PROGRESS",
                "updated_at": now,
            },
            "$push": {"events": event.model_dump(mode="json")},
        },
    )
    logger.info(f"Ticket {ticket_id} assigned to {staff_name}")


async def add_ticket_message(
    ticket_id: str,
    role: str,
    content: str,
    actor: str = "staff",
) -> None:
    """Add a message (from staff, guest, or system) to the ticket conversation."""
    coll = support_tickets_collection()
    now = datetime.now(timezone.utc)

    msg = TicketMessage(role=role, content=content, timestamp=now)
    event = TicketEvent(
        event_type="STAFF_REPLIED" if role == "staff" else "MESSAGE_ADDED",
        description=f"{role.capitalize()} message: {content[:100]}...",
        timestamp=now,
        actor=actor,
    )

    await coll.update_one(
        {"ticket_id": ticket_id},
        {
            "$push": {
                "conversation": msg.model_dump(mode="json"),
                "events": event.model_dump(mode="json"),
            },
            "$set": {
                "updated_at": now,
            },
        },
    )
    logger.info(f"Message added to ticket {ticket_id} by {role}")


