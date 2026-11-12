"""
Pricing Engine — handles dynamic pricing, caching, and static fallback.

Implements the behavior for dynamicPricing feature toggle:
- Short-term caching (TTL based)
- Fallback to static pricing on failure/timeout
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
import json
import hashlib

from app.models.schemas import PriceBreakdown, SelectedRoom
from app.services.booking_engine import calculate_price_breakdown
from app.config.hotel_config import HotelConfig

logger = logging.getLogger(__name__)

# Simple in-memory cache for pricing requests
# Structure: { "cache_key": {"breakdown": PriceBreakdown, "timestamp": datetime} }
_pricing_cache: dict[str, dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

class PricingEngine:
    @staticmethod
    def estimate_inclusive_price(base_price: float, threshold: float = 7500.0, rate_low: float = 5.0, rate_high: float = 18.0) -> float:
        """
        Catalog prices are stored as final GST-inclusive rates.
        Returns the clean rounded price.
        """
        return float(round(base_price / 10) * 10)

    @staticmethod
    def _generate_cache_key(
        hotel_id: int, 
        selected_rooms: list[SelectedRoom], 
        check_in: str, 
        check_out: str, 
        adults: int, 
        children: int,
        extra_services: list[str] | None
    ) -> str:
        """
        Generate a unique cache key for a pricing request.

        Every field of a `SelectedRoom` that can move the price belongs in here,
        because these are the terms the entry is reused on. The charges and the
        nightly rate were missing until 2026-09-24, and the cost was a guest
        being billed for a mattress they had refused:

          `quote_generator.py:59` zeroes `extra_adult_charge` when the guest has
          said no to an extra bed, and then asks for a price. The four fields
          the key did cover were unchanged by that, so the lookup hit the entry
          the *card* turn had written — the one priced with the bed on it — and
          `calculate_price_breakdown` was never called. The guest saw "Extra bed
          Rs 600" on a bill for a bed they had explicitly declined, and the
          zeroing code had run correctly every time.

        `price_per_night` is in the key for a narrower reason: the calendar
        overwrites it inside `calculate_price_breakdown`, so it only decides the
        price on a night the calendar does not resolve. On such a night two
        rooms at different rates would otherwise share an entry.
        """
        # Sort rooms and extra services to ensure consistent keys
        rooms_data = sorted([
            (
                r.room_id,
                r.rate_plan_id,
                r.assigned_adults,
                r.assigned_children,
                r.extra_adult_charge,
                r.extra_child_charge,
                r.price_per_night,
            )
            for r in selected_rooms
        ])
        extras_data = sorted(extra_services) if extra_services else []
        
        key_dict = {
            "hotel_id": hotel_id,
            "check_in": check_in,
            "check_out": check_out,
            "adults": adults,
            "children": children,
            "rooms": rooms_data,
            "extras": extras_data
        }
        
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    @staticmethod
    async def _call_dynamic_pricing_api(
        selected_rooms: list[SelectedRoom],
        check_in: str,
        check_out: str,
        hotel_id: int,
        extra_services: list[str] | None
    ) -> PriceBreakdown | None:
        """
        Mock call to an external dynamic pricing engine.
        In a real scenario, this would make an HTTP request.
        For now, we simulate a failure or successful mock dynamic adjustment.
        """
        # For the sake of this mock, we will just use the static price but add a dynamic multiplier
        # representing demand/seasonality if we pretend the API succeeds.
        # We'll simulate that the API sometimes fails to trigger the fallback logic.
        
        # Simulate network delay
        await asyncio.sleep(0.5)
        
        # We base the price off the deterministic static calculation.
        # Arbitrary mock multipliers are removed to avoid discrepancies between list and photo prices.
        base_breakdown = await calculate_price_breakdown(
            selected_rooms, check_in, check_out, hotel_id, extra_services
        )
        base_breakdown.total = float(round(base_breakdown.total / 10) * 10)
        return base_breakdown

    @staticmethod
    async def get_price(
        selected_rooms: list[SelectedRoom],
        check_in: str,
        check_out: str,
        hotel_id: int,
        config: HotelConfig,
        adults: int,
        children: int,
        extra_services: list[str] | None = None,
        session_id: str | None = None
    ) -> PriceBreakdown:
        """
        Get the price breakdown. Uses dynamic pricing if enabled (with cache and fallback).
        Otherwise uses static rate card.
        """
        from app.services.feature_gate import FeatureGate
        gate = FeatureGate(config, session_id=session_id)
        
        if not gate.can_dynamic_price():
            # Static pricing only
            from app.repositories.activity_repository import log_activity, ActivityLog
            breakdown = await calculate_price_breakdown(
                selected_rooms, check_in, check_out, hotel_id, extra_services
            )
            breakdown.total = float(round(breakdown.total / 10) * 10)
            return breakdown

        int_hotel_id = int(hotel_id) if str(hotel_id).isdigit() else 111111

        # Dynamic pricing logic
        cache_key = PricingEngine._generate_cache_key(
            int_hotel_id, selected_rooms, check_in, check_out, adults, children, extra_services
        )

        # 1. Check cache
        if cache_key in _pricing_cache:
            cache_entry = _pricing_cache[cache_key]
            ts = cache_entry["timestamp"]
            ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts_aware).total_seconds()
            if age < CACHE_TTL_SECONDS:
                logger.info(f"DYNAMIC_PRICING_CACHE_HIT | hotel_id={int_hotel_id}")
                # We return a deep copy to prevent mutations if coupons are applied later
                res = cache_entry["breakdown"].model_copy(deep=True)
                res.total = float(round(res.total / 10) * 10)
                return res
            else:
                del _pricing_cache[cache_key]

        # 2. Call dynamic pricing API with timeout
        breakdown = None
        try:
            # Enforce 2 second timeout as requested
            breakdown = await asyncio.wait_for(
                PricingEngine._call_dynamic_pricing_api(
                    selected_rooms, check_in, check_out, int_hotel_id, extra_services
                ),
                timeout=2.0
            )
            logger.info(f"DYNAMIC_PRICING_USED | hotel_id={int_hotel_id}")
        except asyncio.TimeoutError:
            logger.warning(f"PRICING_ENGINE_TIMEOUT | hotel_id={int_hotel_id} | Falling back to static pricing")
            breakdown = None
        except Exception as e:
            logger.error(f"PRICING_ENGINE_ERROR | hotel_id={int_hotel_id} | Error: {str(e)} | Falling back to static pricing")
            breakdown = None

        # 3. Fallback to static if dynamic failed
        if breakdown is None:
            logger.info(f"PRICING_ENGINE_FALLBACK | hotel_id={int_hotel_id}")
            breakdown = await calculate_price_breakdown(
                selected_rooms, check_in, check_out, int_hotel_id, extra_services
            )

        if breakdown:
            breakdown.total = float(round(breakdown.total / 10) * 10)

        # 4. Update cache
        if breakdown:
            _pricing_cache[cache_key] = {
                "breakdown": breakdown.model_copy(deep=True),
                "timestamp": datetime.now(timezone.utc)
            }

        return breakdown
