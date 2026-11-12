"""
Guest Repository — CRUD for canonical `guests` collection.

Single source of truth (SSOT) for guest identity across all channels.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from app.db.mongodb import guests_collection
from app.utils.phone import normalize_phone, phone_lookup_variants

logger = logging.getLogger(__name__)


def _generate_guest_id(hotel_id: str, phone: str) -> str:
    """Generate canonical guest_id: GUEST-{hotel_id}-{clean_phone_digits}."""
    digits = "".join(ch for ch in phone if ch.isdigit())
    clean_phone = digits[-10:] if len(digits) >= 10 else digits
    return f"GUEST-{hotel_id}-{clean_phone}"


async def ensure_indexes() -> None:
    """Create indexes for guests collection. Idempotent."""
    coll = guests_collection()
    await coll.create_index("guest_id", unique=True)
    await coll.create_index([("hotel_id", 1), ("phone", 1)], unique=True)
    await coll.create_index([("hotel_id", 1), ("email", 1)], sparse=True)
    await coll.create_index("phone")
    logger.info("guests indexes ensured")


async def get_guest_by_id(guest_id: str) -> dict[str, Any] | None:
    """Fetch a guest document by canonical guest_id."""
    if not guest_id:
        return None
    coll = guests_collection()
    doc = await coll.find_one({"guest_id": guest_id})
    if doc:
        doc.pop("_id", None)
    return doc


async def get_guest_by_phone(hotel_id: str | int, phone: str) -> dict[str, Any] | None:
    """
    Fetch a guest document by hotel_id and phone (supporting all phone variants).
    """
    if not hotel_id or not phone:
        return None

    str_hotel_id = str(hotel_id)
    variants = phone_lookup_variants(phone)
    if not variants:
        variants = [phone.strip()]

    coll = guests_collection()
    doc = await coll.find_one({
        "hotel_id": str_hotel_id,
        "phone": {"$in": variants}
    })
    if doc:
        doc.pop("_id", None)
        return doc

    # Fallback to guest_id pattern lookup
    guest_id = _generate_guest_id(str_hotel_id, phone)
    doc = await coll.find_one({"guest_id": guest_id})
    if doc:
        doc.pop("_id", None)
        return doc

    return None


async def get_or_create_guest(
    hotel_id: str | int,
    phone: str,
    name: str | None = None,
    email: str | None = None,
    source: str = "whatsapp",
    preferred_language: str = "en",
) -> dict[str, Any]:
    """
    Find existing guest or create a new guest document.
    Updates name/email if provided and not previously recorded.
    """
    str_hotel_id = str(hotel_id)
    norm_phone = normalize_phone(phone) or phone.strip()

    existing = await get_guest_by_phone(str_hotel_id, norm_phone)
    if existing:
        updates: dict[str, Any] = {}
        if name and not existing.get("name"):
            updates["name"] = name
        if email and not existing.get("email"):
            updates["email"] = email

        if updates:
            updates["updated_at"] = datetime.now(timezone.utc)
            coll = guests_collection()
            await coll.update_one(
                {"guest_id": existing["guest_id"]},
                {"$set": updates}
            )
            existing.update(updates)
        return existing

    # Create new guest
    guest_id = _generate_guest_id(str_hotel_id, norm_phone)
    now = datetime.now(timezone.utc)
    guest_doc = {
        "guest_id": guest_id,
        "hotel_id": str_hotel_id,
        "name": name,
        "email": email,
        "phone": norm_phone,
        "preferred_language": preferred_language,
        "country": None,
        "age": None,
        "rating": None,
        "valid_id": None,
        "contact_channel": source,
        "source": source,
        "tags": [],
        "notes": None,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }

    coll = guests_collection()
    try:
        await coll.insert_one(guest_doc)
    except Exception as e:
        # Handle concurrent race if another process inserted same guest
        logger.debug(f"Guest insert conflict handled for {guest_id}: {e}")
        found = await coll.find_one({"guest_id": guest_id})
        if found:
            found.pop("_id", None)
            return found

    guest_doc.pop("_id", None)
    return guest_doc


async def update_guest(guest_id: str, updates: dict[str, Any]) -> None:
    """Update fields on a guest document."""
    if not guest_id or not updates:
        return
    coll = guests_collection()
    updates["updated_at"] = datetime.now(timezone.utc)
    await coll.update_one({"guest_id": guest_id}, {"$set": updates})
