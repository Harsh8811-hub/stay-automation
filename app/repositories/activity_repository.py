"""
Activity Repository — MongoDB CRUD for activity_logs / bot_activity_logs collection.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db.mongodb import activity_logs_collection
from app.models.activity_schemas import ActivityLog

logger = logging.getLogger(__name__)

async def ensure_indexes() -> None:
    """Create indexes for activity_logs / bot_activity_logs."""
    coll = activity_logs_collection()
    try:
        await coll.create_index("activity_id", unique=True)
    except Exception:
        pass
    await coll.create_index([("hotel_id", 1), ("booking_id", 1)])
    await coll.create_index([("hotel_id", 1), ("reference_id", 1)])
    await coll.create_index([("hotel_id", 1), ("customer_phone", 1)])
    await coll.create_index([("hotel_id", 1), ("created_at", -1)])
    logger.info("activity_logs indexes ensured")


async def log_activity(activity: ActivityLog) -> None:
    """
    Insert a new activity log entry into MongoDB.
    This is an append-only operation.
    """
    coll = activity_logs_collection()
    data = activity.model_dump(mode="json")
    now = datetime.now(timezone.utc)
    
    # Ensure activity_id for bot_activity_logs unique constraint
    today = now.strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    data.setdefault("activity_id", f"ACT-{today}-{short_uuid}")
    data.setdefault("hotel_id", str(data.get("hotel_id") or "111111"))
    data.setdefault("created_at", now)
    data.setdefault("updated_at", now)
    
    await coll.insert_one(data)
    logger.info(f"Activity Logged: [{activity.activity_type}] {data['activity_id']} - {activity.description}")


async def get_booking_timeline(hotel_id: int | str, booking_id: str) -> list[dict[str, Any]]:
    """
    Fetch the complete timeline for a specific booking.
    Sorted chronologically (oldest to newest).
    """
    coll = activity_logs_collection()
    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    timeline = []
    async for doc in coll.find(
        {"hotel_id": {"$in": variants}, "booking_id": booking_id},
        sort=[("created_at", 1), ("timestamp", 1)]
    ):
        doc.pop("_id", None)
        timeline.append(doc)
        
    return timeline
