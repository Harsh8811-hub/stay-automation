"""Reads a hotel's rate-threshold window from `hotel_settings`.

The Control Centre's "Rate Threshold Protection" page writes
`hotel_settings.default_rate_threshold` — two factors per hotel that bound how
far a computed nightly rate may sit from its reference. See
`app/utils/rate_threshold.py` for what a factor means and why the reference is
the pre-season calendar rate.

This module is only the reader: cache, tenant variants, and the fail-closed
`None`. The clamping itself is pure and lives in the utils module so it can be
tested without Mongo.

Reads `hotel_settings` **directly**, not through
`hotel_repository.get_hotel_settings`, for two reasons: that helper merges
`ai_config` over `hotel_settings` (an unrelated key of the same name in
`ai_config` would silently win), and it has no `try/except`, so a Mongo blip
during a quote would raise instead of leaving the price alone. The pattern here
is `early_late_checkout_repository`'s.

Nothing in this repo read this page before; the thresholds were stored and
displayed and never applied.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config.settings import get_settings
from app.db.mongodb import hotel_settings_collection
from app.repositories.hotel_repository import _get_hotel_variants
from app.utils.rate_threshold import RateThreshold, parse_rate_threshold

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300.0

# hotel_id (str) -> (RateThreshold or None, monotonic expiry)
_THRESHOLD_CACHE: dict[str, tuple[RateThreshold | None, float]] = {}


async def get_rate_threshold(hotel_id: int | str) -> RateThreshold | None:
    """The hotel's rate window, or None for "do not clamp anything".

    `None` is returned when the engineering kill switch is off, when the hotel
    has no `hotel_settings` row, when the row carries no
    `default_rate_threshold`, and when that block exists but does not describe a
    usable window. All four are the same thing to a caller and all four leave the
    guest's price exactly as it is today.

    `None` is cached alongside real values: an unconfigured hotel is the common
    case, and it must not hit Mongo on every priced night.
    """
    if not get_settings().RATE_THRESHOLD_PROTECTION_ENABLED:
        return None

    cache_key = str(hotel_id)
    now = time.monotonic()
    cached = _THRESHOLD_CACHE.get(cache_key)
    if cached is not None and now < cached[1]:
        return cached[0]

    threshold: RateThreshold | None = None
    try:
        # The alias helper is deliberate: hotel 988273 is stored under 108959's
        # settings row, and a threshold configured for either must reach both.
        variants = _get_hotel_variants(hotel_id)
        doc: dict[str, Any] | None = await hotel_settings_collection().find_one(
            {"hotel_id": {"$in": variants}}
        )
        if doc is not None:
            threshold = parse_rate_threshold(doc.get("default_rate_threshold"))
    except Exception as e:
        logger.warning(f"Error fetching rate threshold for hotel {hotel_id}: {e}")
        return None

    _THRESHOLD_CACHE[cache_key] = (threshold, now + _CACHE_TTL_SECONDS)
    return threshold
