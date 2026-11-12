import logging
import time
from typing import List, Optional, Any, Dict, Tuple
from bson import ObjectId
from datetime import datetime
from pymongo import ReturnDocument

from app.db.mongodb import hotel_amenities_collection
from app.models.amenity_schemas import (
    AmenityCreate,
    AmenityUpdate,
    AmenityInDB,
    AccessType,
    AccessRules,
    Pricing,
    Scheduling,
    DaySchedule,
    OperationalInfo,
)

logger = logging.getLogger(__name__)

# In-memory cache for loaded hotel amenities: (timestamp, amenities_list)
_AMENITY_CACHE: Dict[str, Tuple[float, List[AmenityInDB]]] = {}
_CACHE_TTL_SECONDS = 300.0  # 5 minutes


def invalidate_amenity_cache(hotel_id: int | str | None = None) -> None:
    """Invalidate in-memory cache for a specific hotel or all hotels."""
    if hotel_id is None:
        _AMENITY_CACHE.clear()
    else:
        variants = _get_hotel_variants(hotel_id)
        for v in variants:
            prefix = str(v)
            for k in list(_AMENITY_CACHE.keys()):
                if k.startswith(f"{prefix}_"):
                    _AMENITY_CACHE.pop(k, None)


def _get_hotel_variants(hotel_id: int | str) -> list:
    """Build list of tenant ID representations including string and int."""
    str_hotel_id = str(hotel_id)
    if str_hotel_id in ("988273", "108959"):
        return ["108959", "988273", 108959, 988273]
    variants: list[int | str] = [str_hotel_id]
    if str_hotel_id.isdigit():
        variants.append(int(hotel_id))
    return list(dict.fromkeys(variants))


def _parse_container_amenities(doc: dict, hotel_id: int | str) -> List[AmenityInDB]:
    """
    Parse a hierarchical hotel_amenities container document into a list of AmenityInDB objects.
    Recursively scans categories and subcategories to extract each leaf amenity.
    """
    amenities_dict = doc.get("amenities") or {}
    items: List[AmenityInDB] = []

    def _recurse(curr: Any, cat_path: list[str]):
        if not isinstance(curr, dict):
            return

        # Check if curr is a leaf amenity definition
        if "name" in curr and "available" in curr:
            leaf_name = str(curr.get("name") or "").strip()
            amenity_id = cat_path[-1] if cat_path else leaf_name.lower().replace(" ", "_")
            cat_name = cat_path[0] if cat_path else "General"
            subcat_name = cat_path[1] if len(cat_path) > 1 else ""
            notes = str(curr.get("notes") or "").strip()
            metadata = curr.get("metadata") or {}
            is_avail = bool(curr.get("available", False))

            access_type = AccessType.PAID if (metadata.get("Paid") or metadata.get("paid")) else AccessType.FREE

            pricing = Pricing()
            access_rules = AccessRules(access_type=access_type)

            scheduling = Scheduling()
            sched_data = curr.get("schedule")
            if isinstance(sched_data, dict):
                def_sched = sched_data.get("default")
                if isinstance(def_sched, dict):
                    all_24 = True
                    for day_name, times in def_sched.items():
                        if isinstance(times, dict):
                            o_t = times.get("open") or ""
                            c_t = times.get("close") or ""
                            scheduling.default_schedule[day_name.lower()] = DaySchedule(open_time=o_t, close_time=c_t)
                            if not (o_t in ("00:00", "0:00") and c_t in ("23:59", "24:00")):
                                all_24 = False
                        else:
                            all_24 = False
                    if all_24 and len(scheduling.default_schedule) >= 7:
                        scheduling.is_24_hours = True

            op_info = OperationalInfo(
                notes=notes,
                metadata=metadata,
                location=subcat_name or None,
            )

            items.append(
                AmenityInDB(
                    _id=f"{doc.get('_id', 'amenity')}_{amenity_id}",
                    hotel_id=hotel_id,
                    amenity_id=amenity_id,
                    name=leaf_name,
                    category=cat_name,
                    description=notes or f"{leaf_name} ({cat_name})",
                    available=is_avail,
                    active=True,
                    access_rules=access_rules,
                    pricing=pricing,
                    scheduling=scheduling,
                    operational_info=op_info,
                )
            )
            return

        for k, v in curr.items():
            if k in ("_id", "hotel_id", "updated_at", "created_at", "restaurants"):
                continue
            if isinstance(v, dict):
                _recurse(v, cat_path + [k])

    _recurse(amenities_dict, [])
    return items


async def create_amenity(amenity_in: AmenityCreate) -> Optional[AmenityInDB]:
    """Create a new amenity for a hotel."""
    try:
        coll = hotel_amenities_collection()
        amenity_dict = amenity_in.model_dump()
        amenity_dict["created_at"] = datetime.utcnow()
        amenity_dict["updated_at"] = datetime.utcnow()

        result = await coll.insert_one(amenity_dict)
        if result.inserted_id:
            amenity_dict["_id"] = str(result.inserted_id)
            invalidate_amenity_cache(amenity_in.hotel_id)
            return AmenityInDB(**amenity_dict)
    except Exception as e:
        logger.error(f"Error creating amenity: {e}")
    return None


async def get_amenity(amenity_id: str, hotel_id: int | str) -> Optional[AmenityInDB]:
    """Get a specific amenity by its string amenity_id and hotel_id."""
    try:
        amenities = await list_amenities(hotel_id, active_only=False)
        q_id = amenity_id.lower().strip()
        for a in amenities:
            if a.amenity_id.lower() == q_id or a.name.lower() == q_id:
                return a
    except Exception as e:
        logger.error(f"Error fetching amenity {amenity_id}: {e}")
    return None


async def list_amenities(hotel_id: int | str, active_only: bool = True) -> List[AmenityInDB]:
    """
    List all amenities for a hotel.
    Supports both:
    1. Hierarchical container document (hotel_amenities has nested dict per hotel)
    2. Flat documents (one document per amenity)
    """
    cache_key = f"{hotel_id}_{active_only}"
    now_ts = time.time()
    if cache_key in _AMENITY_CACHE:
        cached_ts, cached_data = _AMENITY_CACHE[cache_key]
        if now_ts - cached_ts < _CACHE_TTL_SECONDS:
            return cached_data

    try:
        coll = hotel_amenities_collection()
        variants = _get_hotel_variants(hotel_id)
        cursor = coll.find({"hotel_id": {"$in": variants}})

        amenities: List[AmenityInDB] = []
        async for doc in cursor:
            # Check if this is a container document with nested "amenities"
            if "amenities" in doc and isinstance(doc["amenities"], dict):
                parsed = _parse_container_amenities(doc, hotel_id)
                amenities.extend(parsed)
            elif "amenity_id" in doc:
                # Flat document format
                if active_only and not doc.get("active", True):
                    continue
                doc["_id"] = str(doc["_id"])
                amenities.append(AmenityInDB(**doc))

        if amenities:
            _AMENITY_CACHE[cache_key] = (now_ts, amenities)
        return amenities
    except Exception as e:
        logger.error(f"Error listing amenities for hotel {hotel_id}: {e}")
    return []


async def update_amenity(amenity_id: str, hotel_id: int | str, update_data: AmenityUpdate) -> Optional[AmenityInDB]:
    """Update an amenity."""
    try:
        coll = hotel_amenities_collection()
        variants = _get_hotel_variants(hotel_id)

        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            return await get_amenity(amenity_id, hotel_id)

        update_dict["updated_at"] = datetime.utcnow()

        doc = await coll.find_one_and_update(
            {"amenity_id": amenity_id, "hotel_id": {"$in": variants}},
            {"$set": update_dict},
            return_document=ReturnDocument.AFTER,
        )
        if doc:
            invalidate_amenity_cache(hotel_id)
            doc["_id"] = str(doc["_id"])
            return AmenityInDB(**doc)
    except Exception as e:
        logger.error(f"Error updating amenity {amenity_id}: {e}")
    return None


async def delete_amenity(amenity_id: str, hotel_id: int | str) -> bool:
    """Soft delete an amenity (set active=False)."""
    try:
        coll = hotel_amenities_collection()
        variants = _get_hotel_variants(hotel_id)
        result = await coll.update_one(
            {"amenity_id": amenity_id, "hotel_id": {"$in": variants}},
            {"$set": {"active": False, "updated_at": datetime.utcnow()}},
        )
        if result.modified_count > 0:
            invalidate_amenity_cache(hotel_id)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting amenity {amenity_id}: {e}")
    return False


async def hard_delete_amenity(amenity_id: str, hotel_id: int | str) -> bool:
    """Permanently delete an amenity."""
    try:
        coll = hotel_amenities_collection()
        variants = _get_hotel_variants(hotel_id)
        result = await coll.delete_one({"amenity_id": amenity_id, "hotel_id": {"$in": variants}})
        if result.deleted_count > 0:
            invalidate_amenity_cache(hotel_id)
            return True
        return False
    except Exception as e:
        logger.error(f"Error hard deleting amenity {amenity_id}: {e}")
    return False


async def create_indexes():
    """Create indexes for the amenities collection."""
    try:
        coll = hotel_amenities_collection()
        await coll.create_index([("hotel_id", 1), ("amenity_id", 1)])
        await coll.create_index([("hotel_id", 1)])
        logger.info("Successfully created indexes for hotel_amenities collection.")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
