"""
Extra Services / Upsell Collection Module.

Handles all logic related to identifying, calculating, and presenting extra hotel services
(e.g., airport pickup, late checkout, local sightseeing) either before, during, or after room booking.
Supports rich MongoDB schemas (UpsellItems, upsell_events, upsell_inventory) with TTL caching
and fallback to hardcoded default catalogs.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict

from app.services.payment_service import create_payment_link
from app.prompts.templates import text_card

logger = logging.getLogger(__name__)

# TTL for in-memory DB caching in seconds (5 minutes)
UPSELL_CACHE_TTL = 300


@dataclass
class ExtraService:
    service_id: str
    name: str
    category: str
    description: str
    synonyms: List[str]
    trigger_phrases: List[str]
    required_fields: List[str]
    pricing_type: str  # "fixed", "variable", "request_only", "custom"
    default_price: Optional[float]
    is_chargeable: bool
    is_request_only: bool
    supported_stages: List[str]  # e.g., ["pre_booking", "post_booking", "during_booking"]
    code: str = ""
    tax_rate: float = 12.0
    per_person: bool = False
    custom_pricing: List[Dict[str, Any]] = field(default_factory=list)
    availability_limit: Optional[int] = None
    lead_time_hours: Optional[int] = None
    supplier_info: Dict[str, Any] = field(default_factory=dict)
    cancellation_policy: Dict[str, Any] = field(default_factory=dict)
    payment_policy: Dict[str, Any] = field(default_factory=dict)
    image_url: str = ""
    active: bool = True
    raw_db_id: Optional[str] = None

    def __post_init__(self):
        if not self.code:
            self.code = re.sub(r'[^a-z0-9_]', '_', self.name.lower()).strip('_')
        if not self.raw_db_id:
            self.raw_db_id = self.service_id


@dataclass
class ExtraServiceEstimate:
    service_id: str
    name: str
    details: Dict[str, Any]
    estimate_amount: Optional[float]
    status: str  # "pending_details", "ready", "added_to_booking", "pending_payment", "pending_confirmation"


# Hardcoded Fallback Catalog of Services (Used if hotel DB catalog is empty)
SERVICE_CATALOG = [
    ExtraService(
        service_id="airport_pickup",
        name="Airport Pickup",
        category="Transport",
        description="Pickup from the nearest airport to the hotel.",
        synonyms=["airport pickup", "airport transfer", "cab from airport"],
        trigger_phrases=["airport se pickup", "airport pickup", "airport se", "cab from airport", "coming from airport", "station se pickup karva do", "airport pickup chahiye"],
        required_fields=["date", "time", "passengers"],
        pricing_type="fixed",
        default_price=1200.0,
        is_chargeable=True,
        is_request_only=False,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="airport_pickup",
        tax_rate=12.0
    ),
    ExtraService(
        service_id="airport_drop",
        name="Airport Drop",
        category="Transport",
        description="Drop from the hotel to the nearest airport.",
        synonyms=["airport drop", "drop to airport"],
        trigger_phrases=["airport drop", "airport jana hai", "drop to airport", "airport drop service"],
        required_fields=["date", "time", "passengers"],
        pricing_type="fixed",
        default_price=1200.0,
        is_chargeable=True,
        is_request_only=False,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="airport_drop",
        tax_rate=12.0
    ),
    ExtraService(
        service_id="railway_pickup",
        name="Railway Station Pickup",
        category="Transport",
        description="Pickup from the nearest railway station.",
        synonyms=["station pickup", "train station pickup"],
        trigger_phrases=["station se", "station pickup", "train se aa", "railway station pickup", "station lene aaoge", "station se pickup karva do"],
        required_fields=["date", "time", "passengers"],
        pricing_type="fixed",
        default_price=500.0,
        is_chargeable=True,
        is_request_only=False,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="railway_pickup",
        tax_rate=12.0
    ),
    ExtraService(
        service_id="local_sightseeing",
        name="Local Sightseeing",
        category="Local Experience",
        description="Arrangement for local city tour.",
        synonyms=["sightseeing", "city tour"],
        trigger_phrases=["sightseeing", "city tour", "ghumna hai", "local ghumna", "local tour", "darshan"],
        required_fields=["date", "passengers", "duration"],
        pricing_type="variable",
        default_price=None,  # Price to be confirmed
        is_chargeable=True,
        is_request_only=True,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="local_sightseeing"
    ),
    ExtraService(
        service_id="early_checkin",
        name="Early Check-in",
        category="Stay Convenience",
        description="Request for early check-in.",
        synonyms=["early check-in", "early check in", "come early"],
        trigger_phrases=["early check-in", "early checkin", "early check in", "jaldi aana", "subah aana"],
        required_fields=["arrival time"],
        pricing_type="request_only",
        default_price=None,
        is_chargeable=False,
        is_request_only=True,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="early_checkin"
    ),
    ExtraService(
        service_id="late_checkout",
        name="Late Checkout",
        category="Stay Convenience",
        description="Request for late checkout.",
        synonyms=["late checkout", "late check-out", "leave late"],
        trigger_phrases=["late checkout", "late check-out", "leave late", "der se jana", "late leave", "late checkout milega", "can i checkout at"],
        required_fields=["departure time"],
        pricing_type="request_only",
        default_price=None,
        is_chargeable=False,
        is_request_only=True,
        supported_stages=["during_booking", "post_booking"],
        code="late_checkout"
    ),
    ExtraService(
        service_id="breakfast_addon",
        name="Breakfast Add-on",
        category="Food & Beverage",
        description="Add breakfast to your stay.",
        synonyms=["breakfast", "nashta", "morning meal"],
        trigger_phrases=["breakfast add", "nashta add", "breakfast included", "want breakfast", "breakfast chahiye", "breakfast add kar do"],
        required_fields=["guests", "days"],
        pricing_type="fixed",
        default_price=350.0,
        is_chargeable=True,
        is_request_only=False,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="breakfast_addon",
        tax_rate=5.0
    ),
    ExtraService(
        service_id="pool_access",
        name="Swimming Pool Access",
        category="Wellness",
        description="Access to swimming pool.",
        synonyms=["swimming pool", "pool"],
        trigger_phrases=["swimming pool hai", "pool access", "pool hai", "do you have pool", "swimming pool", "pool access hai"],
        required_fields=[],
        pricing_type="request_only",
        default_price=0.0,
        is_chargeable=False,  # Assuming pool is generally an amenity unless otherwise configured
        is_request_only=True,
        supported_stages=["pre_booking", "during_booking", "post_booking"],
        code="pool_access"
    )
]


class ExtraServiceManager:
    """Manages detection, state, dynamic database catalog, pricing, and responses for Extra Services."""

    def __init__(self):
        self.catalog = {s.service_id: s for s in SERVICE_CATALOG}
        # In-memory cache mapping hotel_id_str -> (timestamp, List[ExtraService])
        self._cache: Dict[str, Tuple[float, List[ExtraService]]] = {}

    async def record_upsell_event(
        self,
        hotel_code: Any,
        session_id: str,
        upsell_item: str,
        response: str,
        booking_id: Optional[str] = None,
        revenue: float = 0.0
    ) -> bool:
        """
        Record an upsell event impression/acceptance/decline into upsell_events collection in MongoDB.
        """
        try:
            from app.db.mongodb import upsell_events_collection
            from datetime import datetime, timezone
            coll = upsell_events_collection()
            doc = {
                "hotel_code": hotel_code,
                "session_id": session_id,
                "booking_id": booking_id,
                "upsell_item": upsell_item,
                "response": response,
                "revenue": revenue,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await coll.insert_one(doc)
            logger.info(f"Recorded upsell_event: {upsell_item} -> {response} for hotel {hotel_code}")
            return True
        except Exception as e:
            logger.warning(f"Could not record upsell event ({upsell_item} -> {response}): {e}")
            return False

    async def get_services(
        self,
        hotel_id: Optional[Any] = None,
        stage: Optional[str] = None,
        active_only: bool = True
    ) -> List[ExtraService]:
        """
        Fetch active upsell items dynamically from MongoDB UpsellItems collection with TTL caching.
        If a hotel has no configured upsell items in DB, returns empty list (no hardcoded fallback).
        """
        if not hotel_id:
            services = []
        else:
            hotel_key = str(hotel_id)
            now = time.time()
            cached = self._cache.get(hotel_key)
            if cached and (now - cached[0]) < UPSELL_CACHE_TTL:
                services = cached[1]
            else:
                services = await self._fetch_from_db(hotel_id)
                if not services:
                    logger.debug(f"Upsell DB catalog empty for hotel {hotel_id}. No upsell items available.")
                    services = []
                self._cache[hotel_key] = (now, services)

        if active_only:
            services = [s for s in services if s.active]
        if stage:
            services = [s for s in services if stage in s.supported_stages]

        return services

    async def _fetch_from_db(self, hotel_id: Any) -> List[ExtraService]:
        """Query MongoDB UpsellItems collection and parse into ExtraService objects."""
        try:
            from app.db.mongodb import upsell_items_collection
            collection = upsell_items_collection()

            hotel_ids = [str(hotel_id)]
            cursor = collection.find({
                "hotel_id": {"$in": hotel_ids},
                "$or": [
                    {"active": True},
                    {"status": "active"},
                    {"active": {"$exists": False}, "status": {"$exists": False}}
                ]
            })
            docs = await cursor.to_list(length=100)
            result = []
            for doc in docs:
                try:
                    basic_info = doc.get("basic_info") or {}
                    name = basic_info.get("name") or doc.get("Name") or doc.get("name") or "Addon Service"
                    raw_id = str(doc["_id"]) if "_id" in doc else re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
                    code = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
                    category = basic_info.get("category") or doc.get("Category") or doc.get("category") or "General"
                    desc = basic_info.get("description") or doc.get("Description") or doc.get("description") or name
                    image_file = doc.get("image_file") or {}
                    image_url = doc.get("ImageUrl") or image_file.get("preview") or image_file.get("file") or ""

                    pricing = doc.get("Pricing") or doc.get("pricing") or {}
                    base_val = pricing.get("Base") if "Base" in pricing else pricing.get("base_price")
                    tax_val = pricing.get("Tax") if "Tax" in pricing else pricing.get("tax_percent")
                    
                    base_price = float(base_val) if base_val is not None and str(base_val).strip() != "" else None
                    tax_rate = float(tax_val) if tax_val is not None and str(tax_val).strip() != "" else 12.0
                    per_person = bool(pricing.get("PerPerson", False))
                    custom_pricing = doc.get("CustomPricing") or doc.get("custom_pricing") or []

                    if custom_pricing:
                        pricing_type = "custom"
                        if base_price is None and len(custom_pricing) > 0:
                            base_price = float(custom_pricing[0].get("price", 0.0))
                    elif base_price is not None and base_price > 0:
                        pricing_type = "fixed"
                    else:
                        pricing_type = "request_only"

                    avail = doc.get("Availability") or doc.get("inventory") or {}
                    limit = avail.get("Limit") if "Limit" in avail else avail.get("capacity_per_day")
                    lead_time = avail.get("LeadTime") if "LeadTime" in avail else avail.get("lead_time_hours")

                    supplier_info = doc.get("Supplier") or doc.get("supplier") or {}
                    cancellation_policy = doc.get("Cancellation") or doc.get("cancellation") or {}
                    payment_policy = doc.get("PaymentPolicy") or doc.get("payment_policy") or {}

                    active_status = doc.get("status")
                    if active_status is not None:
                        is_active = (active_status == "active")
                    else:
                        is_active = bool(doc.get("Active", True))

                    # Determine supported stages based on item characteristics
                    supported_stages = ["pre_booking", "during_booking", "post_booking"]
                    name_lower = name.lower()
                    if any(w in name_lower for w in ["late checkout", "decoration", "cake", "room setup"]):
                        supported_stages = ["during_booking", "post_booking"]

                    # Generate trigger phrases & synonyms from name and category
                    synonyms = [name.lower(), code.replace("_", " ")]
                    trigger_phrases = [name.lower()]
                    for cp in custom_pricing:
                        if "label" in cp:
                            trigger_phrases.append(f"{name.lower()} {cp['label'].lower()}")
                            trigger_phrases.append(cp['label'].lower())

                    srv = ExtraService(
                        service_id=code,
                        raw_db_id=raw_id,
                        name=name,
                        category=category,
                        description=desc,
                        synonyms=synonyms,
                        trigger_phrases=trigger_phrases,
                        required_fields=["date", "time"] if category.lower() in ["transport", "transfer"] else [],
                        pricing_type=pricing_type,
                        default_price=base_price,
                        tax_rate=tax_rate,
                        per_person=per_person,
                        custom_pricing=custom_pricing,
                        availability_limit=int(limit) if limit is not None and str(limit).strip() != "" else None,
                        lead_time_hours=int(lead_time) if lead_time is not None and str(lead_time).strip() != "" else None,
                        supplier_info=supplier_info,
                        cancellation_policy=cancellation_policy,
                        payment_policy=payment_policy,
                        image_url=image_url,
                        is_chargeable=(pricing_type != "request_only" or (base_price is not None and base_price > 0)),
                        is_request_only=(pricing_type == "request_only"),
                        supported_stages=supported_stages,
                        code=code,
                        active=is_active
                    )
                    result.append(srv)
                except Exception as e:
                    logger.warning(f"Error parsing MongoDB upsell doc {doc.get('_id')}: {e}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch upsell items from DB for hotel {hotel_id}: {e}")
            return []

    async def log_upsell_event(
        self,
        hotel_id: Any,
        session_id: str,
        booking_id: Optional[str] = None,
        room_no: Optional[str] = None,
        trigger_type: str = "stage_based",
        trigger_name: str = "pre_booking",
        upsell_item: str = "",
        upsell_price: float = 0.0,
        response: str = "OFFERED",
        revenue_generated: float = 0.0,
        suppressed: bool = False
    ) -> bool:
        """Log upsell impressions, acceptances, and declines into upsell_events MongoDB collection."""
        try:
            from app.db.mongodb import upsell_events_collection
            collection = upsell_events_collection()
            now_dt = datetime.now(timezone.utc)
            event_doc = {
                "hotel_code": str(hotel_id),
                "room_no": room_no or "",
                "booking_id": booking_id or "",
                "session_id": session_id,
                "trigger_type": trigger_type,
                "trigger_name": trigger_name,
                "upsell_item": upsell_item,
                "upsell_price": float(upsell_price),
                "shown_at": now_dt,
                "response": response,
                "response_at": now_dt,
                "revenue_generated": float(revenue_generated),
                "suppressed": suppressed
            }
            await collection.insert_one(event_doc)
            return True
        except Exception as e:
            logger.warning(f"Failed to log upsell event for {upsell_item}: {e}")
            return False

    async def check_inventory_availability(
        self,
        hotel_id: Any,
        upsell_item_id: str,
        date_str: str,
        quantity_needed: int = 1
    ) -> Dict[str, Any]:
        """Check upsell_inventory collection for daily capacity of limited add-on items."""
        try:
            from app.db.mongodb import upsell_inventory_collection
            collection = upsell_inventory_collection()
            hotel_ids = [hotel_id, str(hotel_id)]
            try:
                hotel_ids.append(int(hotel_id))
            except (ValueError, TypeError):
                pass

            doc = await collection.find_one({
                "upsell_item_id": upsell_item_id,
                "$or": [{"hotel_id": {"$in": hotel_ids}}, {"hotel_code": {"$in": hotel_ids}}]
            })
            if not doc or "availability" not in doc:
                return {"available": True, "capacity_unspecified": True}

            daily_info = doc["availability"].get(date_str)
            if not daily_info:
                return {"available": True, "capacity_unspecified": True}

            total_cap = int(daily_info.get("total_capacity", 999))
            booked = int(daily_info.get("booked", 0))
            remaining = total_cap - booked
            return {
                "available": remaining >= quantity_needed,
                "remaining": remaining,
                "total_capacity": total_cap,
                "booked": booked
            }
        except Exception as e:
            logger.warning(f"Error checking inventory for {upsell_item_id}: {e}")
            return {"available": True, "error": str(e)}

    def calculate_service_price_detail(
        self,
        service_query_name: str,
        services_catalog: List[ExtraService],
        tax_type: str = "exclusive",
        taxes_applicable: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate base, tax, and total breakdown for a requested extra service name or code.
        Handles matching against CustomPricing tiers (e.g. 'Airport Pickup (SUV)').
        """
        query_lower = service_query_name.lower().strip()
        matched_service: Optional[ExtraService] = None

        # 1. Exact or code match
        for s in services_catalog:
            if s.name.lower() == query_lower or s.code.lower() == query_lower or (s.raw_db_id and s.raw_db_id.lower() == query_lower):
                matched_service = s
                break

        # 2. Substring / Custom tier match
        selected_custom_tier: Optional[Dict[str, Any]] = None
        if not matched_service:
            for s in services_catalog:
                if s.name.lower() in query_lower or s.code.lower() in query_lower:
                    matched_service = s
                    for tier in s.custom_pricing:
                        label = tier.get("label", "").lower()
                        if label and label in query_lower:
                            selected_custom_tier = tier
                            break
                    if matched_service:
                        break

        # If still no match, try checking trigger phrases or custom tier labels directly
        if not matched_service:
            for s in services_catalog:
                for tier in s.custom_pricing:
                    label = tier.get("label", "").lower()
                    if label and label in query_lower:
                        matched_service = s
                        selected_custom_tier = tier
                        break
                if matched_service:
                    break

        if not matched_service:
            return None

        base_val = 0.0
        display_name = matched_service.name
        if selected_custom_tier:
            base_val = float(selected_custom_tier.get("price", 0.0))
            display_name = f"{matched_service.name} ({selected_custom_tier.get('label', '')})"
        elif matched_service.default_price is not None:
            base_val = float(matched_service.default_price)
        else:
            return None  # Request-only or unpriced item

        tax_rate = matched_service.tax_rate if taxes_applicable else 0.0

        if tax_type == "inclusive":
            u_base = round(base_val / (1 + tax_rate / 100), 2)
            u_tax = round(base_val - u_base, 2)
            u_final = base_val
        else:
            u_base = base_val
            u_tax = round(base_val * (tax_rate / 100), 2)
            u_final = round(base_val + u_tax, 2)

        return {
            "name": display_name,
            "item_name": display_name,
            "service_id": matched_service.code,
            "raw_db_id": matched_service.raw_db_id,
            "base_amount": u_base,
            "tax_amount": u_tax,
            "tax_rate": tax_rate,
            "tax_type": tax_type,
            "final_amount": u_final,
            "tax_details": {
                "base_amount": u_base,
                "tax_amount": u_tax,
                "tax_rate": tax_rate,
                "tax_type": tax_type,
                "final_amount": u_final
            }
        }

    def _detect_intent(self, text: str, available_services: Optional[List[ExtraService]] = None) -> List[ExtraService]:
        """Simple keyword/regex based detection for extra services using provided or active catalog."""
        detected = []
        text_lower = text.lower()
        services_pool = available_services if available_services is not None else list(self.catalog.values())
        for s in services_pool:
            for phrase in s.trigger_phrases + s.synonyms:
                if phrase.lower() in text_lower:
                    if s not in detected:
                        detected.append(s)
        return detected

    def _detect_cancellation(self, text: str) -> bool:
        """Detects if user wants to remove/cancel an extra service."""
        cancel_phrases = ["cancel", "remove", "hata do", "don't want", "delete"]
        return any(p in text.lower() for p in cancel_phrases)

    def _format_service_to_string(self, estimate: ExtraServiceEstimate) -> str:
        """Converts structured service estimate to a readable string for payment_details."""
        parts = [estimate.service_id]
        details_str = " | ".join(f"{k}: {v}" for k, v in estimate.details.items())
        if details_str:
            parts.append(details_str)
        if estimate.estimate_amount is not None:
            parts.append(f"estimate: ₹{estimate.estimate_amount}")
        else:
            parts.append("estimate: staff_confirmed")
        parts.append(f"status: {estimate.status}")
        return " | ".join(parts)

    def _parse_service_from_string(self, service_str: str) -> Optional[ExtraServiceEstimate]:
        """Parses a stored extra service string back to an estimate object."""
        try:
            parts = [p.strip() for p in service_str.split(" | ")]
            service_id = parts[0]
            service = self.catalog.get(service_id)
            if not service:
                return None
            details = {}
            estimate = None
            status = "added_to_booking"
            for part in parts[1:]:
                if ":" in part:
                    k, v = part.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k == "estimate":
                        if v.startswith("₹"):
                            try:
                                estimate = float(v[1:])
                            except ValueError:
                                pass
                    elif k == "status":
                        status = v
                    else:
                        details[k] = v
            return ExtraServiceEstimate(
                service_id=service_id,
                name=service.name,
                details=details,
                estimate_amount=estimate,
                status=status
            )
        except Exception as e:
            logger.warning(f"Failed to parse extra service string: {service_str} - {e}")
            return None

    def handle_message(
        self,
        user_message: str,
        session_state: dict,
        payment_details: dict,
        booking_stage: str
    ) -> dict:
        """
        Synchronous message handler for extra service flows (supported for unit tests & legacy flows).
        Returns a dict with modifications to state and possible bot response.
        """
        result = {
            "handled": False,
            "response": None,
            "session_updates": {},
            "payment_updates": {}
        }
        detected_services = self._detect_intent(user_message)
        is_cancel = self._detect_cancellation(user_message)
        pending_services = session_state.get("pending_extra_services", [])

        # 1. Handle Cancellation/Removal
        if is_cancel and detected_services:
            for ds in detected_services:
                pending_services = [s for s in pending_services if s.get("service_id") != ds.service_id]
                if "extra_service" in payment_details:
                    updated_extra = []
                    for stored_str in payment_details["extra_service"]:
                        parsed = self._parse_service_from_string(stored_str)
                        if parsed and parsed.service_id != ds.service_id:
                            updated_extra.append(stored_str)
                    payment_details["extra_service"] = updated_extra
                    result["payment_updates"]["extra_service"] = updated_extra
            result["session_updates"]["pending_extra_services"] = pending_services
            service_names = ", ".join(s.name for s in detected_services)
            result["handled"] = True
            result["response"] = [text_card(f"Okay, I have removed {service_names}.")[0]]
            return result

        # 2. Handle New/Ongoing Service Requests
        if detected_services:
            if any(s.service_id == "pool_access" for s in detected_services):
                result["handled"] = True
                result["response"] = [text_card("Yes, we have a swimming pool! It is generally complimentary for in-house guests.")[0]]
                return result

            service = detected_services[0]
            details = {}
            if "guests" in service.required_fields or "passengers" in service.required_fields:
                nums = [int(s) for s in user_message.split() if s.isdigit()]
                if nums:
                    details["passengers"] = nums[0]

            missing = [f for f in service.required_fields if f not in details]
            if missing:
                prompt_msg = f"Sure, we can help arrange {service.name}. Please share the {missing[0]}."
                result["handled"] = True
                result["response"] = text_card(prompt_msg)
                service_dict = {
                    "service_id": service.service_id,
                    "details": details,
                    "missing": missing
                }
                pending_services.append(service_dict)
                result["session_updates"]["pending_extra_services"] = pending_services
                return result
            else:
                estimate = ExtraServiceEstimate(
                    service_id=service.service_id,
                    name=service.name,
                    details=details,
                    estimate_amount=service.default_price,
                    status="added_to_booking" if booking_stage in ["collecting", "rooms_shown", "quote_shown"] else "pending_confirmation"
                )
                if booking_stage == "new" or booking_stage == "collecting":
                    prompt_msg = f"{service.name} can be arranged. Do you want only {service.name}, or are you also planning to book a room with us?"
                    result["handled"] = True
                    result["response"] = text_card(prompt_msg)
                    pending_services.append({"service_id": service.service_id, "details": details, "estimate": service.default_price})
                    result["session_updates"]["pending_extra_services"] = pending_services
                    return result
                else:
                    service_str = self._format_service_to_string(estimate)
                    if "extra_service" not in payment_details:
                        payment_details["extra_service"] = []
                    if not any(service.service_id in s for s in payment_details["extra_service"]):
                        payment_details["extra_service"].append(service_str)
                    result["payment_updates"]["extra_service"] = payment_details["extra_service"]
                    msg = f"{service.name} has been noted."
                    if service.default_price:
                        msg += f" The estimated charge is ₹{service.default_price}."
                    else:
                        msg += " The exact charge will be confirmed by the hotel team."
                    result["handled"] = True
                    result["response"] = text_card(msg)
                    pending_services = [s for s in pending_services if s.get("service_id") != service.service_id]
                    result["session_updates"]["pending_extra_services"] = pending_services
                    return result

        # 3. Check if waiting for details on pending service
        if pending_services:
            for pending in pending_services:
                if pending.get("missing"):
                    msg_lower = user_message.lower()
                    skip_intercept = False
                    if any(kw in msg_lower for kw in ["room", "book", "price", "cost", "cancel", "how much", "what is", "amenities", "wifi"]):
                        has_datetime = bool(re.search(r'\b(today|tomorrow|am|pm|morning|evening|night|\d{1,2}(:\d{2})?)\b', msg_lower))
                        if not has_datetime:
                            skip_intercept = True
                    if skip_intercept:
                        continue

                    field_name = pending["missing"].pop(0)
                    pending["details"][field_name] = user_message
                    if pending["missing"]:
                        next_missing = pending["missing"][0]
                        service = self.catalog.get(pending["service_id"])
                        prompt_msg = f"Got it. Now please share the {next_missing} for {service.name}."
                        result["handled"] = True
                        result["response"] = text_card(prompt_msg)
                        result["session_updates"]["pending_extra_services"] = pending_services
                        return result
                    else:
                        service = self.catalog.get(pending["service_id"])
                        estimate = ExtraServiceEstimate(
                            service_id=service.service_id,
                            name=service.name,
                            details=pending["details"],
                            estimate_amount=service.default_price,
                            status="added_to_booking"
                        )
                        service_str = self._format_service_to_string(estimate)
                        if "extra_service" not in payment_details:
                            payment_details["extra_service"] = []
                        if not any(service.service_id in s for s in payment_details["extra_service"]):
                            payment_details["extra_service"].append(service_str)
                        result["payment_updates"]["extra_service"] = payment_details["extra_service"]
                        msg = f"{service.name} has been added."
                        if service.default_price:
                            msg += f" The estimated charge is ₹{service.default_price}."
                        else:
                            msg += " The exact charge will be confirmed by the hotel team."
                        result["handled"] = True
                        result["response"] = [text_card(msg)[0]]
                        pending_services.remove(pending)
                        result["session_updates"]["pending_extra_services"] = pending_services
                        return result
        return result

    def generate_service_reservation_link(self, hotel_id: int, service_id: str, amount: float) -> Optional[str]:
        """Generates a payment link specifically for an extra service."""
        return None

    def attach_extra_services_to_payment_details(self, payment_details: dict, services: List[ExtraServiceEstimate]) -> dict:
        """Helper to safely append new services to payment details."""
        if "extra_service" not in payment_details:
            payment_details["extra_service"] = []
        for estimate in services:
            service_str = self._format_service_to_string(estimate)
            if not any(estimate.service_id in s for s in payment_details["extra_service"]):
                payment_details["extra_service"].append(service_str)
        return payment_details


# Singleton instance to be used across the app
extra_service_manager = ExtraServiceManager()
