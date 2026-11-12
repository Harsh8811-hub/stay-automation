"""
Hotel Configuration Service — loads, caches, and provides typed hotel config.

Architecture:
- In-memory TTL cache (avoids MongoDB on every request)
- Returns typed HotelConfig with separate AI and Business sections
- Cache invalidation on config update
- Thread-safe singleton pattern

The AI only receives AIPromptConfig. BusinessConfig stays in Python.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.config.hotel_config import HotelConfig
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Cache Configuration ──────────────────────────────────────────
_DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes

# In-memory cache: hotel_id → (HotelConfig, expiry_timestamp)
_config_cache: dict[int, tuple[HotelConfig, float]] = {}


def _cache_key(hotel_id: int) -> int:
    """Normalize cache key."""
    return hotel_id


def _is_cache_valid(hotel_id: int) -> bool:
    """Check if cached config exists and has not expired."""
    key = _cache_key(hotel_id)
    if key not in _config_cache:
        return False
    _, expiry = _config_cache[key]
    return time.time() < expiry


def _get_cached(hotel_id: int) -> Optional[HotelConfig]:
    """Retrieve cached config if valid."""
    key = _cache_key(hotel_id)
    if _is_cache_valid(hotel_id):
        config, _ = _config_cache[key]
        return config
    return None


def _set_cached(hotel_id: int, config: HotelConfig, ttl: int = _DEFAULT_CACHE_TTL_SECONDS) -> None:
    """Store config in cache with TTL."""
    key = _cache_key(hotel_id)
    _config_cache[key] = (config, time.time() + ttl)


def invalidate_cache(hotel_id: int | None = None) -> None:
    """
    Invalidate cached config.

    Call this when hotel configuration is updated (e.g., via admin API).
    If hotel_id is None, clears entire cache.
    """
    if hotel_id is None:
        _config_cache.clear()
        logger.info("Hotel config cache fully cleared")
    else:
        key = _cache_key(hotel_id)
        _config_cache.pop(key, None)
        logger.info(f"Hotel config cache invalidated for hotel_id={hotel_id}")


async def get_hotel_config(hotel_id: int | str) -> HotelConfig:
    """
    Load hotel configuration with caching.
    Strictly keyed by normalized integer hotel_id.

    1. Check in-memory cache
    2. If miss → load from MongoDB Hotel-Settings
    3. Parse into typed HotelConfig
    4. Cache with TTL
    """
    if not hotel_id:
        settings = get_settings()
        hotel_id = settings.HOTEL_ID

    int_hotel_id = int(hotel_id) if str(hotel_id).isdigit() else 111111

    # 1. Check cache
    cached = _get_cached(int_hotel_id)
    if cached is not None:
        logger.debug(f"Hotel config cache HIT for hotel_id={int_hotel_id}")
        return cached

    logger.info(f"Hotel config cache MISS for hotel_id={int_hotel_id}. Loading from MongoDB...")

    # 2. Load from MongoDB
    settings = get_settings()
    hotel_name = ""

    try:
        from app.repositories.hotel_repository import get_hotel_settings, get_hotel_by_id
        from app.observability.metrics import metrics
        with metrics.timer("mongo.hotel_settings_latency_ms", hotel_id=int_hotel_id):
            doc = await get_hotel_settings(int_hotel_id)
        
        # Resolve real hotel name from hotel_settings or hotels profile
        if doc:
            hotel_name = (
                doc.get("property", {}).get("property_name")
                or doc.get("propertyName")
                or doc.get("property", {}).get("display_name")
                or doc.get("displayName")
                or doc.get("name")
                or ""
            )

        if not hotel_name:
            h_profile = await get_hotel_by_id(int_hotel_id)
            if h_profile:
                hotel_name = h_profile.get("name") or h_profile.get("property_name") or ""

        # ── The hotel's child age policy ──────────────────────────────
        #
        # It lives in `hotel_policies.child_age_policy` and until 2026-09-22
        # NOTHING in this codebase read it — `free_till` and `paid_till` appear
        # in no other file. Every hotel ran on the parser's hardcoded 10/11
        # defaults no matter what it had configured, so Sample Hotel, which says
        # free till 7 and paid till 12, was giving free stays to 8-to-10 year
        # olds and treating 11 and 12 year olds as adults.
        #
        # The two ages are surfaced under the key names the parser already
        # understands, so the policy reaches `BusinessLimits` without the
        # parser having to know where it came from.
        #
        # `paid_till` is inclusive — a child is charged up to and including
        # that age — so the adult threshold is one past it. Getting this wrong
        # by one turns a paid 12-year-old into an adult and re-prices the room.
        # Its own try. It shares the block above only by accident of layout:
        # the `except` down there sets `doc = None`, so a failure fetching the
        # policy — a different collection, a different query — would throw away
        # the entire hotel configuration and silently re-configure the hotel
        # from defaults. Losing the child policy is survivable; losing the
        # hotel is not.
        try:
            from app.repositories.hotel_repository import get_hotel_policies
            policies = await get_hotel_policies(int_hotel_id)
            child_policy = (policies or {}).get("child_age_policy") or {}
            free_till = child_policy.get("free_till")
            paid_till = child_policy.get("paid_till")
            if isinstance(free_till, (int, float)) and isinstance(paid_till, (int, float)):
                if 0 <= free_till <= paid_till <= 25:
                    doc = dict(doc or {})
                    # `setdefault`, not assignment: an explicit limits value
                    # still wins, and `limits_raw` is consulted before the top
                    # level anyway. This only fills a hole.
                    doc.setdefault("child_free_age_limit", int(free_till))
                    doc.setdefault("child_as_adult_age_limit", int(paid_till) + 1)
                else:
                    # A policy that cannot be true is worse than none: free_till
                    # above paid_till would put every child in no bucket at all.
                    logger.warning(
                        "hotel_id=%s has an unusable child_age_policy "
                        "(free_till=%r, paid_till=%r) — keeping the configured limits",
                        int_hotel_id, free_till, paid_till,
                    )
        except Exception as e:
            logger.warning(
                "hotel_id=%s: could not read child_age_policy (%s) — "
                "children fall back to the configured age limits",
                int_hotel_id, e,
            )
    except Exception as e:
        logger.error(f"Failed to load hotel settings from MongoDB for hotel_id={int_hotel_id}: {e}")
        doc = None

    if not hotel_name:
        hotel_name = settings.HOTEL_NAME

    # 3. Parse into typed config
    config = HotelConfig.from_mongo_doc(doc or {}, hotel_id=int_hotel_id, hotel_name=hotel_name)

    # 4. Cache
    _set_cached(int_hotel_id, config)
    logger.info(
        f"Hotel config loaded and cached for hotel_id={int_hotel_id} | "
        f"AI enabled: {config.business_config.features.ai_enabled} | "
        f"Tone: {config.ai_config.conversation_tone} | "
        f"Languages: {config.ai_config.supported_languages}"
    )

    return config


async def warm_cache(hotel_id: int) -> None:
    """
    Pre-load hotel config at startup.
    Called during application lifespan initialization.
    """
    await get_hotel_config(hotel_id)
    logger.info(f"Hotel config cache warmed for hotel_id={hotel_id}")
