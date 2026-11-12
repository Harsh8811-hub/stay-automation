"""
Price Match Repository for managing price match requests in MongoDB.
"""
from typing import Dict, Any, List
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def price_match_collection():
    from app.db.mongodb import price_matches_collection
    return price_matches_collection()

async def ensure_indexes() -> None:
    """Create indexes for price_matches collection. Idempotent."""
    coll = price_match_collection()
    await coll.create_index("session_id")
    await coll.create_index("hotel_id")
    await coll.create_index([("hotel_id", 1), ("status", 1)])
    await coll.create_index("created_at")
    logger.info("price_matches indexes ensured")

async def create_price_match_request(data: Dict[str, Any]) -> str:
    """Create a new price match request in price_matches."""
    coll = price_match_collection()
    now = datetime.now(timezone.utc)
    doc = {
        **data,
        "created_at": now,
        "updated_at": now,
        "events": [{"status": data.get("status", "PENDING"), "timestamp": now, "note": "Request created"}]
    }
    res = await coll.insert_one(doc)
    return str(res.inserted_id)

async def get_active_request_for_session(session_id: str) -> Dict[str, Any] | None:
    """Get any PENDING or MANUAL_REVIEW request for a session."""
    coll = price_match_collection()
    doc = await coll.find_one(
        {
            "session_id": session_id,
            "status": {"$in": ["PENDING", "MANUAL_REVIEW"]}
        },
        sort=[("created_at", -1)]
    )
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc

async def update_request_status(request_id: str, status: str, validation_errors: List[str] = None, final_price: float = None) -> bool:
    """Update the status of a price match request."""
    from bson import ObjectId
    coll = price_match_collection()
    now = datetime.now(timezone.utc)
    
    update_data = {
        "status": status,
        "updated_at": now,
    }
    if validation_errors is not None:
        update_data["validation_errors"] = validation_errors
    if final_price is not None:
        update_data["final_price"] = final_price
        
    res = await coll.update_one(
        {"_id": ObjectId(request_id)},
        {
            "$set": update_data,
            "$push": {"events": {"status": status, "timestamp": now, "note": "Status updated"}}
        }
    )
    return res.modified_count > 0
