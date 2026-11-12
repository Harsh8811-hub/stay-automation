"""
Lead Repository — MongoDB CRUD for canonical `leads` collection.

Handles:
- Readable lead ID generation (LEAD-YYYYMMDD-NNNNNN)
- Lead creation with canonical guest_id and stay_intent structures
- Duplicate detection by session and guest
- Lead status lifecycle management
"""

import logging
from datetime import datetime, timezone
from typing import Any

from app.db.mongodb import leads_collection
from app.models.activity_schemas import ActivityLog
from app.repositories import guest_repository
from app.repositories.activity_repository import log_activity

logger = logging.getLogger(__name__)


# ── Indexes ──────────────────────────────────────────────────────

async def ensure_indexes() -> None:
    """Create indexes for leads collection. Idempotent."""
    coll = leads_collection()
    await coll.create_index("lead_id", unique=True)
    await coll.create_index([("hotel_id", 1), ("guest_id", 1)])
    await coll.create_index([("hotel_id", 1), ("session_id", 1)])
    await coll.create_index([("hotel_id", 1), ("customer_phone", 1)])
    await coll.create_index([("hotel_id", 1), ("status", 1)])
    await coll.create_index("created_at")
    logger.info("leads indexes ensured")


# ── Lead ID Generation ───────────────────────────────────────────

async def generate_lead_id(hotel_id: int | str) -> str:
    """
    Generate a readable lead ID: LEAD-YYYYMMDD-NNNNNN
    Uses today's date + a sequential counter per hotel per day.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    str_hid = str(hotel_id)
    coll = leads_collection()
    count = await coll.count_documents({
        "hotel_id": str_hid,
        "lead_id": {"$regex": f"^LEAD-{today}-"}
    })
    sequence = str(count + 1).zfill(6)
    return f"LEAD-{today}-{sequence}"


# ── Duplicate Detection ──────────────────────────────────────────

async def find_existing_lead(hotel_id: int | str, session_id: str) -> dict[str, Any] | None:
    """
    Find an existing active lead for this session.
    Prevents duplicate lead creation within the same conversation.
    """
    coll = leads_collection()
    str_hid = str(hotel_id)
    doc = await coll.find_one(
        {
            "hotel_id": str_hid,
            "$or": [
                {"session_id": session_id},
                {"conversation_id": session_id},
                {"metadata.session_id": session_id},
            ],
            "status": {"$in": ["NEW", "CONTACTED", "IN_PROGRESS"]},
        },
        sort=[("created_at", -1)],
    )
    if doc:
        doc.pop("_id", None)
    return doc


# ── CRUD Operations ──────────────────────────────────────────────

async def create_lead(lead_data: dict[str, Any]) -> str:
    """
    Insert a new lead into MongoDB canonical `leads` collection.
    Returns the lead_id.
    """
    coll = leads_collection()
    now = datetime.now(timezone.utc)

    hotel_id = str(lead_data.get("hotel_id") or "111111")
    lead_data["hotel_id"] = hotel_id
    if not lead_data.get("lead_id"):
        lead_data["lead_id"] = await generate_lead_id(hotel_id)
    lead_data.setdefault("status", "NEW")
    lead_data.setdefault("source", "whatsapp")
    lead_data.setdefault("created_at", now)
    lead_data.setdefault("updated_at", now)

    # Link to canonical guest
    phone = lead_data.get("customer_phone") or lead_data.get("phone") or ""
    name = lead_data.get("guest_name") or lead_data.get("customer_name")
    email = lead_data.get("customer_email")

    if phone and not lead_data.get("guest_id"):
        guest = await guest_repository.get_or_create_guest(
            hotel_id=hotel_id,
            phone=phone,
            name=name,
            email=email,
        )
        lead_data["guest_id"] = guest["guest_id"]

    # Structure stay_intent if not already structured
    if "stay_intent" not in lead_data:
        lead_data["stay_intent"] = {
            "check_in": lead_data.get("check_in"),
            "check_out": lead_data.get("check_out"),
            "adults": lead_data.get("adults"),
            "children": lead_data.get("children"),
            "num_rooms": lead_data.get("num_rooms") or 1,
            "preferred_room": lead_data.get("preferred_room"),
            "special_requests": lead_data.get("special_requests"),
        }

    await coll.insert_one(lead_data)

    # Log to activity timeline
    await log_activity(ActivityLog(
        hotel_id=str(hotel_id),
        activity_type="LEAD_CREATED",
        reference_id=lead_data["lead_id"],
        booking_id=None,
        customer_phone=phone,
        customer_name=name,
        description=(
            f"New booking lead: {lead_data.get('lead_type', 'INQUIRY')} | "
            f"Room: {lead_data.get('preferred_room', 'Not specified')} | "
            f"Dates: {lead_data.get('check_in', 'TBD')} to {lead_data.get('check_out', 'TBD')}"
        ),
        metadata={
            "lead_type": lead_data.get("lead_type"),
            "adults": lead_data.get("adults"),
            "children": lead_data.get("children"),
        },
    ))

    logger.info(
        f"Lead created: {lead_data['lead_id']} | "
        f"Type: {lead_data.get('lead_type')} | "
        f"Hotel: {hotel_id}"
    )
    return lead_data["lead_id"]


async def get_lead(lead_id: str) -> dict[str, Any] | None:
    """Fetch a lead by its readable ID."""
    coll = leads_collection()
    doc = await coll.find_one({"lead_id": lead_id})
    if doc:
        doc.pop("_id", None)
    return doc


async def update_lead_status(lead_id: str, new_status: str) -> None:
    """Update a lead's status."""
    coll = leads_collection()
    now = datetime.now(timezone.utc)
    await coll.update_one(
        {"lead_id": lead_id},
        {"$set": {"status": new_status, "updated_at": now}},
    )
    logger.info(f"Lead {lead_id} status -> {new_status}")
