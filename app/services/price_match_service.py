"""
Price Match Verification Engine.
Handles validation, policy enforcement, and discount calculation.
"""
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime, timezone

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Default policy if not found in DB
DEFAULT_POLICY = {
    "price_match_enabled": True,
    "maximum_discount_percent": 10,
    "minimum_price_difference": 100,
    "require_same_room": True,
    "require_same_meal_plan": True,
    "require_same_refund_policy": True
}

# Configurable supported platforms
SUPPORTED_PLATFORMS = ["booking.com", "agoda", "goibibo", "makemytrip", "expedia", "hotels.com"]

async def get_price_match_policy(hotel_id: str | int) -> Dict[str, Any]:
    """Fetch hotel price match policy from hotel_settings."""
    from app.db.mongodb import hotel_settings_collection
    coll = hotel_settings_collection()
    str_hid = str(hotel_id)
    config = await coll.find_one({"hotel_id": str_hid})
    if config:
        behavior = config.get("behavior") or {}
        limits = config.get("limits") or {}
        features = behavior.get("features") or {}
        return {
            "price_match_enabled": features.get("priceMatchEnabled", features.get("price_match_enabled", True)),
            "maximum_discount_percent": limits.get("maximumPriceMatchDiscount", limits.get("maximum_discount_percent", 10)),
            "minimum_price_difference": limits.get("minimumPriceDifference", 100),
            "require_same_room": True,
            "require_same_meal_plan": True,
            "require_same_refund_policy": True,
        }
    return DEFAULT_POLICY

async def get_supported_platforms() -> List[str]:
    """Fetch supported platforms from MongoDB."""
    return SUPPORTED_PLATFORMS

def _normalize_string(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().lower()

async def validate_price_match(
    extracted_data: Dict[str, Any],
    session_state: Dict[str, Any],
    hotel_id: str | int
) -> Tuple[bool, List[str], float, float]:
    """
    Validate the price match request against booking state and hotel policy.
    Returns: (is_valid, validation_errors, new_price, discount_applied)
    """
    errors = []
    
    # 1. Basic Confidence & Missing Fields
    confidence = extracted_data.get("confidence") or 0.0
    if confidence < 0.80:
        errors.append(f"Image confidence too low ({confidence}). Manual review required.")
        
    missing = extracted_data.get("missing_fields") or []
    if missing:
        errors.append(f"Missing required fields in image: {', '.join(missing)}")
        
    # 2. Check Supported Platform
    platform = _normalize_string(extracted_data.get("booking_platform"))
    supported = await get_supported_platforms()
    if not any(p in platform for p in supported):
        errors.append(f"Unsupported booking platform: {extracted_data.get('booking_platform')}")
        
    # 3. Policy Enforcement
    policy = await get_price_match_policy(hotel_id)
    if not policy.get("price_match_enabled"):
        errors.append("Price matching is currently disabled for this hotel.")
        return False, errors, 0.0, 0.0

    # 3.5 Check for Expired Screenshot
    current_year = str(datetime.now(timezone.utc).year)
    e_timestamp = str(extracted_data.get("screenshot_timestamp") or "").lower()
    e_checkin_str = str(extracted_data.get("check_in") or "").lower()
    
    # If the screenshot explicitly shows a past year, reject it
    past_years = [str(int(current_year) - i) for i in range(1, 5)]
    for py in past_years:
        if py in e_timestamp or py in e_checkin_str:
            errors.append(f"Booking Already Expired. The screenshot appears to be from {py}, but current year is {current_year}.")
            break

    # 4. Compare with Session State
    s_adults = session_state.get("adults", 0)
    s_children = session_state.get("children", 0)
    s_check_in = session_state.get("check_in")
    s_check_out = session_state.get("check_out")
    s_room_type = _normalize_string(session_state.get("room_type_preference"))
    s_meal_plan = _normalize_string(session_state.get("meal_plan", "Room Only"))
    our_price = session_state.get("total_price") or 0.0
    if not isinstance(our_price, (int, float)):
        our_price = 0.0
    
    e_adults = extracted_data.get("adults")
    if e_adults is not None and s_adults and e_adults != s_adults:
        errors.append(f"Guest count mismatch. You booked for {s_adults} adults, but image shows {e_adults}.")
        
    e_children = extracted_data.get("children")
    if e_children is not None and s_children and e_children != s_children:
        errors.append(f"Children count mismatch. You booked for {s_children}, but image shows {e_children}.")
        
    e_check_in = extracted_data.get("check_in")
    if e_check_in and s_check_in and s_check_in not in e_check_in:
        errors.append(f"Check-in date mismatch. You booked for {s_check_in}.")
        
    e_check_out = extracted_data.get("check_out")
    if e_check_out and s_check_out and s_check_out not in e_check_out:
        errors.append(f"Check-out date mismatch. You booked for {s_check_out}.")
        
    if policy.get("require_same_room"):
        e_room_type = _normalize_string(extracted_data.get("room_type"))
        if e_room_type and s_room_type and s_room_type not in e_room_type and e_room_type not in s_room_type:
            errors.append(f"Room type mismatch. You booked {session_state.get('room_type_preference')}, but image shows {extracted_data.get('room_type')}.")
            
    if policy.get("require_same_meal_plan"):
        e_meal_plan = _normalize_string(extracted_data.get("meal_plan"))
        if e_meal_plan and s_meal_plan and s_meal_plan not in e_meal_plan and e_meal_plan not in s_meal_plan:
            errors.append(f"Meal plan mismatch. Image shows {extracted_data.get('meal_plan')}.")

    e_price = extracted_data.get("price")
    if not isinstance(e_price, (int, float)) or e_price <= 0:
        errors.append("Invalid or missing price in the screenshot.")
        
    if our_price <= 0:
        errors.append("Please first select dates and a room with us so I can compare our rates against this offer.")
        
    # If any errors so far, return
    if errors:
        return False, errors, 0.0, 0.0
        
    # 5. Discount Calculation
    diff = our_price - e_price
    min_diff = policy.get("minimum_price_difference", 100)
    
    if diff < min_diff:
        errors.append(f"Price difference (₹{diff}) is less than the minimum required (₹{min_diff}) for a price match.")
        return False, errors, 0.0, 0.0
        
    max_discount_pct = policy.get("maximum_discount_percent", 10)
    max_discount_amount = our_price * (max_discount_pct / 100.0)
    
    calculated_discount = min(diff, max_discount_amount)
    final_price = float(round((our_price - calculated_discount) / 10) * 10)
    
    return True, [], final_price, calculated_discount


def generate_direct_booking_offer(
    extracted_ota: Dict[str, Any],
    direct_price: float = 0.0,
    hotel_policy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generates a 5%-lower direct booking offer based on an extracted OTA screenshot.
    Applies tax normalization, minimum savings check, and floor price protection.
    """
    policy = hotel_policy or DEFAULT_POLICY
    raw_price = float(extracted_ota.get("price") or extracted_ota.get("total_price") or 0.0)
    if raw_price <= 0:
        return {
            "is_valid": False,
            "error": "Missing or invalid price in the OTA screenshot.",
            "offer_price": 0.0,
            "savings": 0.0,
            "message": "We couldn't detect a valid total price in the screenshot. Please ensure the total amount is clearly visible.",
        }

    tax_included = extracted_ota.get("tax_included", True)
    if not tax_included:
        # Normalise an OTA's tax-exclusive total up to a GST-inclusive one, so it
        # is compared like-for-like against our own inclusive rates.
        #
        # This read 1.12 until 2026-09-29, and it is the same retired number that
        # was left behind in `booking_engine`'s `rate_high`: 12 was the LOW GST
        # band until the September 2025 reform cut it to 5.
        #
        # The `else` branch used to be `12.0`, which made it a trap rather than a
        # fallback: it is reached only when `GST_RATE_OVERRIDE_PERCENT` is
        # `None`, so restoring the per-hotel slabs would have quietly put this
        # back on the retired band. product set the override to `None` on
        # 2026-09-30, which is exactly that case — hence the low band, spelled
        # out, as the fallback. A screenshot offering Rs 10,000 + tax normalises
        # to Rs 10,500, not Rs 11,200.
        #
        # Overstating the OTA is not free: we then quote 5% below a price the
        # guest never saw, so the "we beat it" claim can be false and the guest
        # finds the OTA cheaper. Erring high protected revenue; erring accurate
        # keeps the promise.
        #
        # Deliberately the *low* band and not a slab: this normalises a whole-
        # stay total, and the Rs 7,500 threshold in `booking_engine` is a
        # per-night rate that this figure cannot be compared against.
        _gst_pct = get_settings().GST_RATE_OVERRIDE_PERCENT
        _fallback_gst_pct = 5.0
        ota_normalized = round(
            raw_price
            * (1 + (float(_gst_pct) if _gst_pct is not None else _fallback_gst_pct) / 100.0),
            2,
        )
    else:
        ota_normalized = raw_price

    platform_raw = extracted_ota.get("booking_platform") or extracted_ota.get("platform") or "OTA"
    platform = str(platform_raw).strip().title()

    # Calculate 5% lower offer
    discount_pct = float(policy.get("ota_direct_discount_percent", 5.0))
    discounted = ota_normalized * (1.0 - (discount_pct / 100.0))

    # Floor price guard (minimum 70% of standard direct price or explicit policy floor)
    floor_price = float(policy.get("floor_price", direct_price * 0.70 if direct_price > 0 else discounted * 0.85))
    offer_price = float(round(max(discounted, floor_price) / 10) * 10)

    savings = float(round((ota_normalized - offer_price) / 10) * 10)
    if savings < 50.0:
        # Minimum ₹50 direct booking incentive
        savings = 50.0
        offer_price = float(round(max(ota_normalized - 50.0, floor_price) / 10) * 10)

    reply_text = (
        f"🏨 *Exclusive Direct Booking Price Match!*\n\n"
        f"We checked your *{platform}* price of ₹{ota_normalized:,.0f}.\n"
        f"Book directly with us to get an extra *{discount_pct:.0f}% discount*:\n\n"
        f"💰 *{platform} Price:* ₹{ota_normalized:,.0f}\n"
        f"🎉 *StayInn Direct Price:* ₹{offer_price:,.0f}\n"
        f"✨ *Your Instant Savings:* ₹{savings:,.0f}\n\n"
        f"🌟 *Direct Booking Perks:*\n"
        f"• Free early check-in (subject to availability)\n"
        f"• Direct hotel support & instant confirmation\n\n"
        f"Would you like to book at ₹{offer_price:,.0f}? Reply **'Confirm'** or **'Yes'** to lock in this special rate!"
    )

    logger.info(
        "OTA_DIRECT_OFFER_GENERATED | platform=%s | ota_price=%s | offer_price=%s | savings=%s",
        platform,
        ota_normalized,
        offer_price,
        savings,
    )

    return {
        "is_valid": True,
        "platform": platform,
        "ota_price": ota_normalized,
        "offer_price": offer_price,
        "savings": savings,
        "message": reply_text,
    }
