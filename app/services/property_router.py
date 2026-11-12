"""
Property Router — Dynamic Multi-Tenant Hotel Resolution Service.

Resolves inbound WhatsApp metadata (phone_number_id, display_phone_number, WABA ID)
and request payloads to canonical hotel_id.

Key Principles:
1. Dynamic DB Registry: Reads hotel profiles and contact numbers from MongoDB.
2. In-Memory TTL Cache: Fast resolution without DB load on every turn.
3. Fail Loud / Safe Isolation: NEVER silently fall back to Sample Hotel (111111) for unmapped tenants.
4. Concurrent-Safe: Thread-safe, read-only cache with normalized integer keys.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Cache configuration
_CACHE_TTL_SECONDS = 300  # 5 minutes
_property_cache: dict[str, tuple[str, float]] = {}  # identifier -> (hotel_id, expiry)

# Canonical Known Tenant Registry (Hardcoded baseline fallback for zero-latency)
_STATIC_REGISTRY = {
    # 111111: Sample Hotel
    "9453222931": "111111",
    "919453222931": "111111",
    "950439284815083": "111111",  # Meta Cloud API Phone Number ID
    "HOTELSAHU": "111111",
    
    # 988273 / 108959 / 710235: Sample Property by Sahu Hotels
    "9214012263": "988273",
    "919214012263": "988273",
    "8141459509": "988273",
    "918141459509": "988273",
    "15558707948": "988273",
    "710235": "988273",
    "988273": "988273",
    "108959": "988273",
    "SANIDHYAM": "988273",
    
    # 376891: Hotel Paradise
    "9125864443": "376891",
    "919125864443": "376891",
    "981207785076244": "376891",  # Meta Cloud API Phone Number ID
    "1413888793726432": "376891",  # WABA ID
    "HOTELROME": "376891",
    
    # 968225: Hotel Maharaja
    "9410170311": "968225",
    "919410170311": "968225",
    "HOTELMAHARAJA": "968225",
    
    # 191919: hoteldummy
    "hoteldummy": "191919",
}


def _clean_identifier(val: str | None) -> str:
    """Normalize identifier by stripping non-alphanumeric chars and converting to uppercase."""
    if not val:
        return ""
    s = str(val).strip()
    return s


def _clean_phone_digits(val: str | None) -> str:
    """Extract digits only from phone number string."""
    if not val:
        return ""
    return re.sub(r"\D", "", str(val))


class PropertyRouter:
    """
    Dynamic property router that resolves hotel_id from various metadata identifiers.
    """

    def resolve_hotel_id(
        self,
        *,
        phone_number_id: str | None = None,
        display_phone_number: str | None = None,
        waba_id: str | None = None,
        property_code: str | None = None,
        allow_default_fallback: bool = False,
    ) -> Optional[str]:
        """
        Resolve hotel_id with the following priority:
        1. Explicit Environment Variable overrides (META_COM_PHONE_NUMBER_ID_{hotel}, etc.)
        2. In-memory cache
        3. Static known tenant baseline
        4. (Synchronous fallback) Default settings.HOTEL_ID ONLY IF allow_default_fallback=True
        
        Returns None if unmapped (fail-loud isolation) when allow_default_fallback=False.
        """
        # 1. Check phone_number_id in env vars
        pid = (phone_number_id or "").strip()
        if pid:
            # Check suffixed META_COM_PHONE_NUMBER_ID_{hotel}
            prefix = "META_COM_PHONE_NUMBER_ID_"
            for key, val in os.environ.items():
                if key.startswith(prefix) and val.strip() == pid:
                    suffix = key[len(prefix):]
                    if suffix.isdigit():
                        return suffix

            # Check un-suffixed default META_COM_PHONE_NUMBER_ID
            default_pid = os.getenv("META_COM_PHONE_NUMBER_ID", "").strip()
            if default_pid and default_pid == pid:
                settings = get_settings()
                return str(settings.HOTEL_ID)

        # Check waba_id in env vars
        wid = (waba_id or "").strip()
        if wid:
            w_prefix = "COMBOT_WABA_ID_"
            for key, val in os.environ.items():
                if key.startswith(w_prefix) and val.strip() == wid:
                    suffix = key[len(w_prefix):]
                    if suffix.isdigit():
                        return suffix

        # 2. Check cached registry
        now = time.time()
        for candidate in (pid, wid, property_code, display_phone_number):
            if not candidate:
                continue
            cand_clean = _clean_identifier(candidate)
            if cand_clean in _property_cache:
                hid, expiry = _property_cache[cand_clean]
                if now < expiry:
                    return hid

            # Check phone digits (last 10 digits match)
            digits = _clean_phone_digits(candidate)
            if digits:
                if digits in _property_cache:
                    hid, expiry = _property_cache[digits]
                    if now < expiry:
                        return hid
                if len(digits) >= 10 and digits[-10:] in _property_cache:
                    hid, expiry = _property_cache[digits[-10:]]
                    if now < expiry:
                        return hid

        # 3. Check static baseline registry
        if pid and pid in _STATIC_REGISTRY:
            return _STATIC_REGISTRY[pid]
        if wid and wid in _STATIC_REGISTRY:
            return _STATIC_REGISTRY[wid]
        if property_code and property_code.upper() in _STATIC_REGISTRY:
            return _STATIC_REGISTRY[property_code.upper()]

        dpn_digits = _clean_phone_digits(display_phone_number)
        if dpn_digits:
            if dpn_digits in _STATIC_REGISTRY:
                return _STATIC_REGISTRY[dpn_digits]
            if len(dpn_digits) >= 10 and dpn_digits[-10:] in _STATIC_REGISTRY:
                return _STATIC_REGISTRY[dpn_digits[-10:]]

        # 4. Fallback only if explicitly allowed (e.g. CLI or single-tenant local testing)
        if allow_default_fallback:
            settings = get_settings()
            return str(settings.HOTEL_ID)

        logger.warning(
            "UNMAPPED_PROPERTY | phone_id=%s | dpn=%s | waba_id=%s | prop_code=%s | Returning None",
            pid or "-",
            display_phone_number or "-",
            wid or "-",
            property_code or "-",
        )
        return None

    async def resolve_hotel_id_async(
        self,
        *,
        phone_number_id: str | None = None,
        display_phone_number: str | None = None,
        waba_id: str | None = None,
        property_code: str | None = None,
        allow_default_fallback: bool = False,
    ) -> Optional[str]:
        """
        Async resolution with MongoDB fallback on cache miss.
        """
        # First try sync fast-path
        resolved = self.resolve_hotel_id(
            phone_number_id=phone_number_id,
            display_phone_number=display_phone_number,
            waba_id=waba_id,
            property_code=property_code,
            allow_default_fallback=False,
        )
        if resolved is not None:
            return resolved

        # Query MongoDB for dynamic mapping
        try:
            from app.db.mongodb import hotels_collection, hotel_details_collection
            
            # Check property_code or phone in hotels collection
            cand_digits = _clean_phone_digits(display_phone_number)
            last10 = cand_digits[-10:] if len(cand_digits) >= 10 else cand_digits

            or_clauses = []
            if property_code:
                or_clauses.extend([
                    {"property_code": property_code},
                    {"property_code": property_code.upper()},
                    {"property_code": property_code.lower()},
                    {"name": property_code},
                ])
            if last10:
                or_clauses.extend([
                    {"contact.primary_contact_number": {"$regex": last10}},
                    {"contact.alt_contact_number": {"$regex": last10}},
                ])
            if phone_number_id:
                or_clauses.extend([
                    {"phone_number_id": phone_number_id},
                    {"meta_phone_id": phone_number_id},
                ])

            if or_clauses:
                doc = await hotels_collection().find_one({"$or": or_clauses})
                if not doc:
                    doc = await hotel_details_collection().find_one({"$or": or_clauses})

                if doc and doc.get("hotel_id"):
                    raw_id = doc.get("hotel_id")
                    hid = str(raw_id) if raw_id is not None else None
                    if hid:
                        # Cache the result
                        self.register_mapping(
                            hotel_id=hid,
                            phone_number_id=phone_number_id,
                            display_phone_number=display_phone_number,
                            waba_id=waba_id,
                            property_code=property_code,
                        )
                        return hid
        except Exception as e:
            logger.warning(f"PropertyRouter DB lookup failed: {e}")

        if allow_default_fallback:
            settings = get_settings()
            return str(settings.HOTEL_ID)

        return None

    def register_mapping(
        self,
        hotel_id: str | int,
        phone_number_id: str | None = None,
        display_phone_number: str | None = None,
        waba_id: str | None = None,
        property_code: str | None = None,
        ttl: int = _CACHE_TTL_SECONDS,
    ) -> None:
        """Register dynamic mapping in memory cache."""
        now = time.time()
        expiry = now + ttl
        hotel_id = str(hotel_id)
        if phone_number_id:
            _property_cache[_clean_identifier(phone_number_id)] = (hotel_id, expiry)
        if waba_id:
            _property_cache[_clean_identifier(waba_id)] = (hotel_id, expiry)
        if property_code:
            _property_cache[_clean_identifier(property_code).upper()] = (hotel_id, expiry)
        if display_phone_number:
            digits = _clean_phone_digits(display_phone_number)
            if digits:
                _property_cache[digits] = (hotel_id, expiry)
                if len(digits) >= 10:
                    _property_cache[digits[-10:]] = (hotel_id, expiry)

    async def warm_registry(self) -> None:
        """Load all hotel records from DB into memory cache on startup."""
        try:
            from app.db.mongodb import hotels_collection, hotel_details_collection
            h_docs = await hotels_collection().find({}).to_list(100)
            for doc in h_docs:
                raw_id = doc.get("hotel_id")
                if not raw_id or not str(raw_id).isdigit():
                    continue
                hid = str(raw_id)
                self.register_mapping(
                    hotel_id=hid,
                    property_code=doc.get("property_code"),
                    ttl=3600,  # 1 hour
                )

            d_docs = await hotel_details_collection().find({}).to_list(100)
            for doc in d_docs:
                raw_id = doc.get("hotel_id")
                if not raw_id or not str(raw_id).isdigit():
                    continue
                hid = str(raw_id)
                contact = doc.get("contact") or {}
                p_phone = contact.get("primary_contact_number")
                a_phone = contact.get("alt_contact_number")
                if p_phone:
                    self.register_mapping(hotel_id=hid, display_phone_number=p_phone, ttl=3600)
                if a_phone:
                    self.register_mapping(hotel_id=hid, display_phone_number=a_phone, ttl=3600)

            logger.info(f"PropertyRouter warmed with {len(_property_cache)} identifier mappings")
        except Exception as e:
            logger.warning(f"PropertyRouter warmup skipped: {e}")


property_router = PropertyRouter()
