"""
Task Repository — MongoDB CRUD for canonical `hotel_tasks` collection.

Handles:
- Readable task ID generation (TASK-YYYYMMDD-NNNNNN)
- Duplicate detection (same booking + room + subtype)
- Append-only event logging
- Task status management
- Canonical guest_id linkage
"""

import logging
from datetime import datetime, timezone
from typing import Any

from app.db.mongodb import hotel_tasks_collection
from app.models.activity_schemas import ActivityLog
from app.models.task_schemas import HotelTask, TaskEvent
from app.repositories import guest_repository
from app.repositories.activity_repository import log_activity

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# INDEXES
# ═══════════════════════════════════════════════════════════════════

async def ensure_indexes() -> None:
    """Create indexes for hotel_tasks. Idempotent."""
    coll = hotel_tasks_collection()
    await coll.create_index("task_id", unique=True)
    await coll.create_index([("hotel_id", 1), ("session_id", 1)])
    await coll.create_index([("hotel_id", 1), ("guest_id", 1)])
    await coll.create_index([("hotel_id", 1), ("booking_id", 1), ("subtype", 1), ("status", 1)])
    await coll.create_index([("hotel_id", 1), ("department", 1), ("status", 1)])
    await coll.create_index([("hotel_id", 1), ("status", 1)])
    await coll.create_index("created_at")
    logger.info("hotel_tasks indexes ensured")


# ═══════════════════════════════════════════════════════════════════
# TASK ID GENERATION
# ═══════════════════════════════════════════════════════════════════

async def generate_task_id(hotel_id: int | str) -> str:
    """
    Generate a globally unique, readable task ID: TASK-YYYYMMDD-NNNNNN
    Uses today's date + global sequential counter per day to prevent E11000 duplicate key collisions.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    coll = hotel_tasks_collection()
    latest_task = await coll.find_one(
        {"task_id": {"$regex": f"^TASK-{today}-"}},
        sort=[("task_id", -1)],
    )
    if latest_task and latest_task.get("task_id"):
        try:
            last_seq = int(latest_task["task_id"].split("-")[-1])
            next_seq = last_seq + 1
        except (ValueError, IndexError):
            count = await coll.count_documents({"task_id": {"$regex": f"^TASK-{today}-"}})
            next_seq = count + 1
    else:
        next_seq = 1
    sequence = str(next_seq).zfill(6)
    return f"TASK-{today}-{sequence}"


# ═══════════════════════════════════════════════════════════════════
# DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════

async def find_duplicate_task(
    hotel_id: int | str,
    booking_id: str | None,
    room_number: str | None,
    subtype: str,
) -> dict[str, Any] | None:
    """
    Find existing OPEN/ASSIGNED/IN_PROGRESS task for the same
    booking + room + subtype combination.
    """
    if not booking_id:
        return None

    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    coll = hotel_tasks_collection()
    query: dict[str, Any] = {
        "hotel_id": {"$in": variants},
        "booking_id": booking_id,
        "subtype": subtype,
        "status": {"$in": ["OPEN", "ASSIGNED", "IN_PROGRESS"]},
    }
    if room_number:
        query["room_number"] = room_number

    doc = await coll.find_one(query, sort=[("created_at", -1)])
    if doc:
        doc.pop("_id", None)
    return doc


async def find_open_payment_task(
    hotel_id: int | str,
    booking_id: str | None = None,
    session_id: str | None = None,
    customer_phone: str | None = None,
) -> dict[str, Any] | None:
    """
    Find existing OPEN/ASSIGNED/IN_PROGRESS payment confirmation task for
    the same booking, session, or customer phone.
    """
    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    or_clauses: list[dict[str, Any]] = []
    if booking_id:
        or_clauses.append({"booking_id": booking_id})
    if session_id:
        or_clauses.append({"session_id": session_id})
    if customer_phone:
        # Variants, not the raw string. A phone is stored as `919…`, `+919…` or
        # a bare 10-digit number depending on which path wrote the row, and an
        # exact compare silently misses the task it was meant to find — which
        # is one of the ways a duplicate got created (see the note on
        # `stage_payment_check`). `list_tasks` already expands variants; this
        # is the same rule in the dedicated payment lookup.
        from app.utils.phone import phone_lookup_variants
        phone_variants = phone_lookup_variants(customer_phone) or [customer_phone]
        or_clauses.append({"customer_phone": {"$in": phone_variants}})

    if not or_clauses:
        return None

    coll = hotel_tasks_collection()
    query: dict[str, Any] = {
        "hotel_id": {"$in": variants},
        "subtype": "PAYMENT_CONFIRMATION",
        "status": {"$in": ["OPEN", "ASSIGNED", "IN_PROGRESS"]},
        "$or": or_clauses,
    }

    doc = await coll.find_one(query, sort=[("created_at", -1)])
    if doc:
        doc.pop("_id", None)
    return doc



# ═══════════════════════════════════════════════════════════════════
# CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════════════

async def create_task(task: HotelTask) -> str:
    """
    Insert a new hotel task into MongoDB.
    Returns the task_id.
    """
    coll = hotel_tasks_collection()
    data = task.model_dump(mode="json")
    str_hid = str(task.hotel_id)
    data["hotel_id"] = str_hid

    # Link canonical guest if phone available
    if task.customer_phone and not data.get("guest_id"):
        guest = await guest_repository.get_or_create_guest(
            hotel_id=str_hid,
            phone=task.customer_phone,
            name=task.customer_name,
        )
        data["guest_id"] = guest["guest_id"]

    await coll.insert_one(data)

    # Log Activity
    await log_activity(ActivityLog(
        hotel_id=int(str_hid) if str_hid.isdigit() else 111111,
        activity_type="TASK_CREATED",
        reference_id=task.task_id,
        booking_id=task.booking_id,
        customer_phone=task.customer_phone,
        customer_name=task.customer_name,
        description=f"Task requested: {task.subtype.replace('_', ' ').title()}"
    ))

    logger.info(
        f"Task created: {task.task_id} | Dept: {task.department} "
        f"| Type: {task.task_type}/{task.subtype} | Priority: {task.priority}"
    )
    return task.task_id


async def get_task(task_id: str) -> dict[str, Any] | None:
    """Fetch a task by its readable ID."""
    coll = hotel_tasks_collection()
    doc = await coll.find_one({"task_id": task_id})
    if doc:
        doc.pop("_id", None)
    return doc


async def append_customer_update(task_id: str, message: str) -> None:
    """Append a customer's latest message to an existing task."""
    coll = hotel_tasks_collection()
    now = datetime.now(timezone.utc)

    event = TaskEvent(
        event_type="CUSTOMER_UPDATED",
        description=f"Customer updated request: {message[:200]}",
        timestamp=now,
        actor="customer",
    )

    await coll.update_one(
        {"task_id": task_id},
        {
            "$push": {
                "events": event.model_dump(mode="json"),
                "notes": f"[{now.isoformat()}] Customer: {message[:500]}",
                "customer_updates": {
                    "message": message,
                    "time": now.isoformat()
                }
            },
            "$set": {
                "updated_at": now,
                "customer_request": message,
            },
        },
    )
    logger.info(f"Customer update appended to task: {task_id}")


async def escalate_payment_task(task_id: str, hours_waiting: float) -> None:
    """
    Raise an already-open payment task to URGENT because it has waited too long.

    Not the same event as a customer update: the note is written in the system's
    voice and the priority change is what moves the task up the front desk's
    queue. Called once per booking — `stage_payment_check` latches
    `payment_escalated_at` on the booking so a chatty guest cannot re-escalate
    every turn.
    """
    coll = hotel_tasks_collection()
    now = datetime.now(timezone.utc)

    event = TaskEvent(
        event_type="ESCALATED",
        description=f"Payment verification pending {hours_waiting:.1f}h — escalated to URGENT",
        timestamp=now,
        actor="system",
    )

    await coll.update_one(
        {"task_id": task_id},
        {
            "$push": {
                "events": event.model_dump(mode="json"),
                "notes": (
                    f"[{now.isoformat()}] System: payment verification pending "
                    f"{hours_waiting:.1f}h — priority raised to URGENT"
                ),
            },
            "$set": {
                "priority": "URGENT",
                "escalated_at": now,
                "updated_at": now,
            },
        },
    )
    logger.info(f"Payment task escalated to URGENT after {hours_waiting:.1f}h: {task_id}")


async def update_task_status(
    task_id: str,
    new_status: str,
    actor: str = "system",
) -> None:
    """Update task status with audit event."""
    coll = hotel_tasks_collection()
    now = datetime.now(timezone.utc)

    update_fields: dict[str, Any] = {
        "status": new_status,
        "updated_at": now,
    }
    if new_status == "COMPLETED":
        update_fields["completed_at"] = now

    event = TaskEvent(
        event_type="STATUS_CHANGED",
        description=f"Status changed to {new_status}",
        timestamp=now,
        actor=actor,
        metadata={"new_status": new_status},
    )

    await coll.update_one(
        {"task_id": task_id},
        {
            "$set": update_fields,
            "$push": {"events": event.model_dump(mode="json")},
        },
    )

    doc = await get_task(task_id)
    if doc:
        hid = doc.get("hotel_id", 111111)
        await log_activity(ActivityLog(
            hotel_id=int(hid) if str(hid).isdigit() else 111111,
            activity_type="TASK_COMPLETED" if new_status == "COMPLETED" else ("TASK_CANCELLED" if new_status == "CANCELLED" else "TASK_UPDATED"),
            reference_id=task_id,
            booking_id=doc.get("booking_id"),
            customer_phone=doc.get("customer_phone"),
            customer_name=doc.get("customer_name"),
            description=f"Task {task_id} status changed to {new_status}"
        ))

    logger.info(f"Task {task_id} status -> {new_status}")


async def assign_task(task_id: str, staff_name: str) -> None:
    """Assign a task to a staff member."""
    coll = hotel_tasks_collection()
    now = datetime.now(timezone.utc)

    event = TaskEvent(
        event_type="ASSIGNED",
        description=f"Task assigned to {staff_name}",
        timestamp=now,
        actor="system",
        metadata={"assigned_to": staff_name},
    )

    await coll.update_one(
        {"task_id": task_id},
        {
            "$set": {
                "assigned_staff": staff_name,
                "status": "ASSIGNED",
                "updated_at": now,
            },
            "$push": {"events": event.model_dump(mode="json")},
        },
    )
    logger.info(f"Task {task_id} assigned to {staff_name}")


async def get_tasks_by_department(
    hotel_id: int | str,
    department: str,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch tasks for a specific department."""
    coll = hotel_tasks_collection()
    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    query: dict[str, Any] = {"hotel_id": {"$in": variants}, "department": department}
    if status:
        query["status"] = status

    tasks = []
    async for doc in coll.find(query, sort=[("created_at", -1)]).limit(50):
        doc.pop("_id", None)
        tasks.append(doc)
    return tasks


async def get_tasks_by_booking(hotel_id: int | str, booking_id: str) -> list[dict[str, Any]]:
    """Fetch all tasks for a specific booking."""
    coll = hotel_tasks_collection()
    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    tasks = []
    async for doc in coll.find(
        {"hotel_id": {"$in": variants}, "booking_id": booking_id},
        sort=[("created_at", -1)],
    ):
        doc.pop("_id", None)
        tasks.append(doc)
    return tasks


async def list_tasks(
    hotel_id: int | str | None = None,
    status: str | None = None,
    department: str | None = None,
    priority: str | None = None,
    subtype: str | None = None,
    booking_id: str | None = None,
    customer_phone: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    skip: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch tasks matching optional filters with total matching count.
    Supports tenant isolation via hotel_id variants.
    """
    coll = hotel_tasks_collection()
    query: dict[str, Any] = {}

    if hotel_id is not None:
        str_hid = str(hotel_id)
        variants: list[Any] = [str_hid]
        if str_hid.isdigit():
            variants.append(int(str_hid))
        query["hotel_id"] = {"$in": variants}

    if status:
        query["status"] = status.upper()
    if department:
        query["department"] = department.upper()
    if priority:
        query["priority"] = priority.upper()
    if subtype:
        query["subtype"] = subtype.upper()
    if booking_id:
        query["booking_id"] = booking_id
    if session_id:
        query["session_id"] = session_id
    if customer_phone:
        from app.utils.phone import phone_lookup_variants
        phone_vars = phone_lookup_variants(customer_phone) or [customer_phone]
        query["customer_phone"] = {"$in": phone_vars}

    total = await coll.count_documents(query)
    tasks: list[dict[str, Any]] = []
    cursor = coll.find(query, sort=[("created_at", -1)]).skip(skip).limit(limit)
    async for doc in cursor:
        doc.pop("_id", None)
        tasks.append(doc)
    return tasks, total


async def add_task_note(task_id: str, note: str, author: str = "staff") -> None:
    """Add an internal staff note to an existing task."""
    coll = hotel_tasks_collection()
    now = datetime.now(timezone.utc)

    event = TaskEvent(
        event_type="NOTE_ADDED",
        description=f"Note by {author}: {note[:100]}",
        timestamp=now,
        actor=author,
    )

    await coll.update_one(
        {"task_id": task_id},
        {
            "$push": {
                "notes": f"[{now.isoformat()}] {author}: {note}",
                "events": event.model_dump(mode="json"),
            },
            "$set": {"updated_at": now},
        },
    )
    logger.info(f"Note added to task {task_id} by {author}")

