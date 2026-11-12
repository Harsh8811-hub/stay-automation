import logging
from datetime import datetime, UTC
import uuid

from app.db.mongodb import room_service_carts_collection, room_service_menu_collection
from app.models.room_service_schemas import RoomServiceCart, RoomServiceMenuItem, RoomServiceCartStatus

logger = logging.getLogger(__name__)

async def ensure_indexes() -> None:
    """Create indexes for room service collections."""
    carts_coll = room_service_carts_collection()
    await carts_coll.create_index("cart_id", unique=True)
    await carts_coll.create_index([("hotel_id", 1), ("session_id", 1), ("status", 1)])
    
    menus_coll = room_service_menu_collection()
    await menus_coll.create_index([("hotel_id", 1), ("menu_item_id", 1)], unique=True)
    logger.info("✅ Room service indexes ensured")


async def get_menu_items(hotel_id: int) -> list[RoomServiceMenuItem]:
    coll = room_service_menu_collection()
    items = []
    async for doc in coll.find({"hotel_id": hotel_id, "active": True}):
        if "menu_item_id" not in doc or not doc["menu_item_id"]:
            if "_id" not in doc:
                logger.error(f"CORRUPT DATA: Menu item {doc.get('item_name')} is missing both menu_item_id and _id. Skipping.")
                continue
            doc["menu_item_id"] = str(doc["_id"])
            
        doc.pop("_id", None)
        try:
            items.append(RoomServiceMenuItem(**doc))
        except Exception as e:
            logger.error(f"Failed to parse menu item {doc.get('item_name')}: {e}")
    return items


async def get_active_cart(hotel_id: int, session_id: str) -> RoomServiceCart | None:
    coll = room_service_carts_collection()
    doc = await coll.find_one({
        "hotel_id": hotel_id,
        "session_id": session_id,
        "status": RoomServiceCartStatus.ACTIVE.value
    })
    if doc:
        doc.pop("_id", None)
        return RoomServiceCart(**doc)
    return None


async def generate_cart_id(hotel_id: int) -> str:
    today = datetime.now(UTC).strftime("%Y%m%d")
    coll = room_service_carts_collection()
    count = await coll.count_documents({
        "hotel_id": hotel_id,
        "cart_id": {"$regex": f"^CART-{today}-"}
    })
    sequence = str(count + 1).zfill(6)
    return f"CART-{today}-{sequence}"


async def save_cart(cart: RoomServiceCart) -> None:
    coll = room_service_carts_collection()
    data = cart.model_dump(mode="json")
    await coll.replace_one(
        {"cart_id": cart.cart_id},
        data,
        upsert=True
    )

async def mark_cart_checked_out(cart_id: str) -> None:
    coll = room_service_carts_collection()
    await coll.update_one(
        {"cart_id": cart_id},
        {"$set": {"status": RoomServiceCartStatus.CHECKED_OUT.value, "updated_at": datetime.now(UTC)}}
    )
