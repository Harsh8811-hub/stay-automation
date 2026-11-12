"""
Hotel Repository — reads from normalized snake_case collections in bot_migration
(hotel_details, hotels, hotel_amenities, hotel_settings, hotel_policies).

All are READ-ONLY. Never write to these collections.
"""

import time
from app.db.mongodb import (
    get_database,
    hotel_amenities_collection,
    hotel_details_collection,
    hotel_policies_collection,
    hotel_settings_collection,
    hotels_collection,
)

# In-memory TTL caches (5 min) to eliminate redundant Mongo round-trips per message turn
_CACHE_TTL_SECONDS = 300.0
_POLICIES_CACHE: dict[str, tuple[dict | None, float]] = {}
_DETAILS_CACHE: dict[str, tuple[dict | None, float]] = {}


def _get_hotel_variants(hotel_id: int | str) -> list:
    """Build list of tenant ID representations including known cross-system aliases."""
    str_hotel_id = str(hotel_id)
    # Tenant aliases (e.g. Sample Property 108959 contains real Riverside City rooms for Sample Property by Sahu Hotels)
    if str_hotel_id in ("988273", "108959"):
        return ["108959", "988273", 108959, 988273]
    variants: list[int | str] = [str_hotel_id]
    if str_hotel_id.isdigit():
        variants.append(int(hotel_id))
    return list(dict.fromkeys(variants))


async def get_hotel_by_id(hotel_id: int | str) -> dict | None:
    """Fetch hotel basic profile document from the normalized hotels collection."""
    variants = _get_hotel_variants(hotel_id)
    return await hotels_collection().find_one({"hotel_id": {"$in": variants}})


async def get_hotel_contact(hotel_id: int | str) -> dict | None:
    """Fetch contact details from normalized hotel_details or the hotels profile."""
    variants = _get_hotel_variants(hotel_id)

    # 1. Try normalized hotel_details
    doc = await hotel_details_collection().find_one({"hotel_id": {"$in": variants}})
    if doc and doc.get("contact"):
        return doc.get("contact") or doc

    # 2. Try hotels profile
    h_doc = await hotels_collection().find_one({"hotel_id": {"$in": variants}})
    if h_doc and h_doc.get("contact"):
        return h_doc.get("contact") or h_doc

    return None


# ═══════════════════════════════════════════════════════════════════
# CANONICAL CONTACT FIELDS
# ═══════════════════════════════════════════════════════════════════
#
# `hotel_details.contact` stores snake_case keys — primary_contact_number, hotel_email,
# alt_contact_number. Readers used to reach into that document and probe for the older
# camelCase/legacy spellings instead (customerCareNumber, hotelPhone, hotelEmail). Every probe
# missed, nothing raised, and each caller fell back to placeholder copy: an escalation told the
# guest to "contact our front desk" with no number while the real number sat in the record
# unread, and the booking confirmation named "Our Hotel" instead of the property.
#
# These tables are the one place that decides what a contact record means. A reader that needs a
# field asks get_hotel_contact_info() for it by its canonical name. A reader that reaches into
# the document itself and guesses a key is the bug this exists to prevent.
#
# Both spellings are accepted, snake_case first, so a tenant still storing the old shape keeps
# working — this is strictly more permissive than the reader it replaces.
_CONTACT_PHONE_KEYS = (
    "primary_contact_number", "primaryContactNumber",
    "customerCareNumber", "customer_care_number",
    "hotelPhone", "hotel_phone",
    "hotelMobile", "hotel_mobile",
    "phone_number", "phoneNumber",
    "phoneList", "phone_list", "phone",
)
_CONTACT_ALT_PHONE_KEYS = ("alt_contact_number", "altContactNumber")
_CONTACT_EMAIL_KEYS = (
    "hotel_email", "hotelEmail",
    "contact_email", "contactEmail",
    "emailList", "email_list", "email",
)
_CONTACT_WHATSAPP_KEYS = ("whatsapp", "whatsappNumber", "whatsapp_number")
_CONTACT_WEBSITE_KEYS = ("website_url", "websiteUrl", "website")
_CONTACT_NAME_KEYS = ("primary_contact_name", "primaryContactName")


def _first_present(doc: dict, keys: tuple) -> str:
    """
    Return the first key that holds a non-empty value, as a trimmed string.

    A list-valued field (some tenants store `phoneList`) contributes its first entry that is
    non-blank *after trimming* — a whitespace-only entry is skipped, not selected, and the scan
    continues to the next entry rather than abandoning the whole key. Returns "" when nothing is
    set, so callers test truthiness rather than None.
    """
    for key in keys:
        val = doc.get(key)
        for candidate in (val if isinstance(val, (list, tuple)) else [val]):
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
    return ""


async def get_hotel_details(hotel_id: int | str) -> dict | None:
    """
    Fetch the whole `hotel_details` document — `property`, `contact` and `location` together.
    Uses an in-memory TTL cache to eliminate redundant Mongo round-trips per message turn.
    """
    key = str(hotel_id).strip()
    now = time.time()
    if key in _DETAILS_CACHE:
        val, expiry = _DETAILS_CACHE[key]
        if now < expiry:
            return val

    variants = _get_hotel_variants(hotel_id)
    doc = await hotel_details_collection().find_one({"hotel_id": {"$in": variants}})
    _DETAILS_CACHE[key] = (doc, now + _CACHE_TTL_SECONDS)
    return doc


async def get_hotel_contact_info(hotel_id: int | str) -> dict | None:
    """
    Fetch contact details under canonical keys:
    phone, alt_phone, email, whatsapp, website, contact_name.

    Returns None only when the hotel has no contact record at all. A record that exists but is
    missing a field returns that field as "", so a caller can tell "no record on file" apart from
    "record on file, no phone number" — the two need different copy.
    """
    doc = await get_hotel_contact(hotel_id)
    if not doc:
        return None
    return {
        "phone": _first_present(doc, _CONTACT_PHONE_KEYS),
        "alt_phone": _first_present(doc, _CONTACT_ALT_PHONE_KEYS),
        "email": _first_present(doc, _CONTACT_EMAIL_KEYS),
        "whatsapp": _first_present(doc, _CONTACT_WHATSAPP_KEYS),
        "website": _first_present(doc, _CONTACT_WEBSITE_KEYS),
        "contact_name": _first_present(doc, _CONTACT_NAME_KEYS),
    }


async def get_hotel_location(hotel_id: int | str) -> dict | None:
    """Fetch address and geospatial info for a hotel from normalized hotel_details."""
    variants = _get_hotel_variants(hotel_id)

    # 1. Try normalized hotel_details
    doc = await hotel_details_collection().find_one({"hotel_id": {"$in": variants}})
    if doc and doc.get("location"):
        loc = dict(doc.get("location"))
        if doc.get("transit_directions"):
            loc["transit_directions"] = doc.get("transit_directions")
        return loc

    # 2. Try hotels profile
    h_doc = await hotels_collection().find_one({"hotel_id": {"$in": variants}})
    if h_doc and h_doc.get("address"):
        return {"address": h_doc.get("address"), "name": h_doc.get("name")}

    return None


async def get_hotel_amenities(hotel_id: int | str) -> dict | None:
    """Fetch hotel-level amenity configuration from normalized hotel_amenities."""
    variants = _get_hotel_variants(hotel_id)
    return await hotel_amenities_collection().find_one({"hotel_id": {"$in": variants}})


async def get_hotel_settings(hotel_id: int | str) -> dict | None:
    """Fetch hotel settings, ai configuration, and business rules."""
    variants = _get_hotel_variants(hotel_id)

    # 1. Fetch AI prompt, behavior & feature toggles from ai_config
    ai_doc = await get_database()["ai_config"].find_one({"hotel_id": {"$in": variants}})

    # 2. Fetch business, tax & bank from hotel_settings
    settings_doc = await hotel_settings_collection().find_one({"hotel_id": {"$in": variants}})

    merged = {}
    if settings_doc:
        merged.update(settings_doc)
    if ai_doc:
        merged.update(ai_doc)
        # Ensure behavior and feature_toggles are populated at top level
        if "feature_toggles" in ai_doc and isinstance(ai_doc["feature_toggles"], dict):
            behavior = merged.get("behavior") or {}
            if isinstance(behavior, dict):
                behavior["features"] = {**ai_doc["feature_toggles"], **behavior.get("features", {})}
                merged["behavior"] = behavior

    return merged if merged else None


async def get_hotel_policies(hotel_id: int | str) -> dict | None:
    """Fetch check-in, check-out, and house rules with in-memory TTL caching."""
    key = str(hotel_id).strip()
    now = time.time()
    if key in _POLICIES_CACHE:
        val, expiry = _POLICIES_CACHE[key]
        if now < expiry:
            return val

    variants = _get_hotel_variants(hotel_id)
    doc = await hotel_policies_collection().find_one({"hotel_id": {"$in": variants}})
    if doc and not doc.get("notices") and not doc.get("Notices"):
        # Fallback to alternate collections if notices were updated in web clone / stayinn_web
        try:
            client = hotel_policies_collection().database.client
            for alt_db in ("stayinn_web_clone", "stayinn_web"):
                alt_doc = await client[alt_db]["hotel_policies"].find_one({"hotel_id": {"$in": variants}})
                if alt_doc and (alt_doc.get("notices") or alt_doc.get("Notices")):
                    doc["notices"] = alt_doc.get("notices") or alt_doc.get("Notices")
                    break
        except Exception:
            pass
    _POLICIES_CACHE[key] = (doc, now + _CACHE_TTL_SECONDS)
    return doc


async def get_formatted_hotel_policies(hotel_id: int | str) -> str:
    """Fetch and format hotel policies into canonical guest-facing facts."""
    doc = await get_hotel_policies(hotel_id)
    return format_policies(doc) if doc else ""


async def get_formatted_hotel_notices(hotel_id: int | str) -> str:
    """Fetch and format active property notices and announcements for guest responses."""
    doc = await get_hotel_policies(hotel_id)
    if not doc:
        return ""
    raw_notices = doc.get("notices") or doc.get("Notices") or []
    return format_notices(raw_notices)


async def get_standard_stay_times(hotel_id: int | str) -> tuple[str, str]:
    """The hotel's standard check-in and check-out times, as configured.

    Returns ``("", "")`` when the hotel has no policies row, or when either
    field is absent. Deliberately no fallback default: ``stage_resolver`` says
    12:00 PM and ``confirmation_service`` says 01:00 PM for the same missing
    field, so inventing a time here would print a number the hotel never set.
    A caller that gets two empty strings must render nothing about times.

    This is the raw configured string (e.g. ``"01:00 PM"``), not a parsed
    ``time`` — the guest is shown what the hotel typed, and parsing lives in
    ``early_late_checkout_service`` for the fee arithmetic.
    """
    doc = await get_hotel_policies(hotel_id)
    if not doc:
        return "", ""
    check_in = str(doc.get("check_in_time") or "").strip()
    check_out = str(doc.get("check_out_time") or "").strip()
    return check_in, check_out


def format_location(doc: dict) -> str:
    """Format location document into a clean guest-facing message."""
    if not doc:
        return "📍 Location information is not available right now. Please contact the front desk."

    lines = ["📍 *Hotel Location*\n"]

    # Support normalized location structure
    loc_dict = doc.get("location")
    if not isinstance(loc_dict, dict):
        loc_dict = doc
    
    # 1. Property address
    prop_addr = loc_dict.get("property_address") or loc_dict.get("propertyAddress")
    if prop_addr:
        lines.append(f"🏠 {prop_addr}")
    else:
        address = loc_dict.get("address") or loc_dict.get("Address") or {}
        if isinstance(address, dict):
            parts = []
            for key in ("line1", "line2", "street", "area", "city", "state", "pincode", "zip", "country"):
                val = address.get(key) or address.get(key.capitalize())
                if val:
                    parts.append(str(val))
            if parts:
                lines.append("🏠 " + ", ".join(parts))
        elif isinstance(address, str) and address:
            lines.append(f"🏠 {address}")

    # City / State
    city = loc_dict.get("city") or loc_dict.get("City")
    state = loc_dict.get("state") or loc_dict.get("State")
    if city and state:
        lines.append(f"🏙️ {city}, {state}")
    elif city:
        lines.append(f"🏙️ {city}")

    # Pincode / ZIP
    pincode = loc_dict.get("pincode") or loc_dict.get("zip") or loc_dict.get("postalCode")
    if pincode:
        lines.append(f"📮 Pincode: {pincode}")

    # Google Maps link
    maps_link = (
        loc_dict.get("google_map_link")
        or loc_dict.get("googleMapsUrl")
        or loc_dict.get("mapsUrl")
        or loc_dict.get("mapLink")
        or loc_dict.get("google_maps_url")
    )
    if maps_link:
        lines.append(f"\n🗺️ *Directions:* {maps_link}")

    # Landmark / Nearby
    landmark = loc_dict.get("landmark") or loc_dict.get("nearbyLandmark") or loc_dict.get("directions")
    if landmark:
        lines.append(f"\n📌 Near: {landmark}")

    # Transit Directions (Airport / Railway Station / Bus)
    transit_list = doc.get("transit_directions") or loc_dict.get("transit_directions") or []
    if isinstance(transit_list, list) and transit_list:
        transit_lines = ["\n🚗 *Transit & Commute:*"]
        for t in transit_list[:3]:
            cat = t.get("transit_category") or "Transit"
            origin = t.get("origin_name") or ""
            time_mins = t.get("approximate_commute_minutes")
            dist = t.get("approximate_distance_km")
            transport = t.get("recommended_transport") or "Taxi"
            dirs = t.get("directions") or ""
            map_url = t.get("google_maps_route_url") or ""

            detail_bits = []
            if time_mins:
                detail_bits.append(f"~{time_mins} mins")
            if dist:
                detail_bits.append(f"{dist} km")
            if transport:
                detail_bits.append(f"by {transport}")
            paren = f" ({', '.join(detail_bits)})" if detail_bits else ""

            origin_str = f" from {origin}" if origin else ""
            transit_lines.append(f"• *{cat}*{origin_str}{paren}")
            if dirs and dirs != origin:
                transit_lines.append(f"  _{dirs}_")
            if map_url:
                transit_lines.append(f"  Route: {map_url}")
        lines.append("\n".join(transit_lines))

    if len(lines) == 1:
        lines.append("Please contact our front desk for directions.")

    return "\n\n".join(lines)


def format_notices(notices: list) -> str:
    """Format active property notices and operational announcements into clean guest-facing facts."""
    if not notices or not isinstance(notices, list):
        return ""

    lines = ["📢 *Property Notices & Operational Announcements:*"]
    has_any = False

    for n in notices:
        if not isinstance(n, dict):
            continue
        # Skip expired notices
        if n.get("expired") is True or str(n.get("Action", "")).lower() in ("expired", "past"):
            continue

        title = (
            n.get("title")
            or n.get("Title")
            or n.get("category")
            or n.get("Category")
            or "Notice"
        ).strip()
        notes = (
            n.get("notes")
            or n.get("Notes")
            or n.get("content")
            or n.get("Content")
            or ""
        ).strip()
        applies = (
            n.get("associated_with")
            or n.get("associatedWith")
            or n.get("applies_to")
            or n.get("AppliesTo")
            or ""
        ).strip()

        # Dates
        start_date = str(n.get("start_date") or n.get("startDate") or "").strip()
        end_date = str(n.get("end_date") or n.get("endDate") or "").strip()
        stay_dates = str(n.get("stay_dates") or n.get("StayDates") or "").strip()
        if not stay_dates and (start_date or end_date):
            if start_date and end_date:
                stay_dates = f"{start_date} to {end_date}"
            else:
                stay_dates = start_date or end_date

        details = []
        if applies:
            details.append(f"Applies to: {applies}")
        if stay_dates:
            details.append(f"Stay Dates / Validity: {stay_dates}")

        paren = f" ({', '.join(details)})" if details else ""
        body = f": {notes}" if notes else ""
        lines.append(f"• *{title}*{paren}{body}")
        has_any = True

    return "\n".join(lines) if has_any else ""


def format_policies(doc: dict) -> str:
    """Format hotel_policies document into structured, authoritative policy facts for guest responses."""
    if not doc:
        return ""

    lines = ["Official Hotel Policies & House Rules:"]

    ci = doc.get("check_in_time")
    co = doc.get("check_out_time")
    if ci:
        lines.append(f"• Standard Check-in Time: {ci}")
    if co:
        lines.append(f"• Standard Check-out Time: {co}")

    early_ci = doc.get("early_check_in_available")
    late_co = doc.get("late_check_out_available")
    if early_ci is not None:
        lines.append(f"• Early Check-in: {'Available upon request (subject to room readiness)' if early_ci else 'Subject to availability / additional charge'}")
    if late_co is not None:
        lines.append(f"• Late Check-out: {'Available upon request' if late_co else 'Subject to availability / additional charge'}")

    fd_24 = doc.get("front_desk24_hr")
    if fd_24:
        lines.append("• Front Desk: 24-hour front desk service available.")

    rules = doc.get("property_rules") or {}
    if isinstance(rules, dict):
        # Guest profile
        gp = rules.get("guest_profile") or {}
        if isinstance(gp, dict):
            if gp.get("unmarried_couples_allowed") is False:
                lines.append("• Unmarried Couples: Not allowed (hotel policy requires valid proof of marriage/relationships where applicable).")
            elif gp.get("unmarried_couples_allowed") is True or gp.get("couple_friendly_tag") is True:
                lines.append("• Couple Friendly: Couples are welcome.")
            if gp.get("below18_allowed") is False:
                lines.append("• Age Requirement: Primary guest must be at least 18 years of age with valid ID.")

        # Accepted IDs
        ids = rules.get("accepted_i_ds") or rules.get("accepted_ids") or []
        if isinstance(ids, list) and ids:
            lines.append(f"• Accepted Photo IDs: {', '.join(str(i) for i in ids)}.")
        if rules.get("same_city_id_allowed") is False:
            lines.append("• Local / Same-city IDs: Not allowed for stay as per property rules.")

        # Restrictions
        restr = rules.get("restrictions") or {}
        if isinstance(restr, dict):
            if restr.get("smoking_allowed") is True:
                lines.append("• Smoking: Allowed in designated smoking areas.")
            elif restr.get("smoking_allowed") is False:
                lines.append("• Smoking: Non-smoking property throughout.")
            if restr.get("outside_visitors_allowed") is False:
                lines.append("• Outside Visitors: Outside visitors are not permitted in guest rooms.")
            if restr.get("wheelchair_accessible") is True:
                lines.append("• Accessibility: Wheelchair accessible property.")
            if restr.get("parties_events_allowed") is False:
                lines.append("• Events: Parties/events not allowed in rooms.")

        # Pet policy
        pet = rules.get("pet_policy") or {}
        if isinstance(pet, dict):
            if pet.get("pets_allowed") is True:
                max_p = pet.get("max_pets") or "1"
                extra = "Additional charges may apply." if pet.get("extra_charges") else "No extra pet fee."
                lines.append(f"• Pet Policy: Pets are allowed (Max: {max_p}). {extra}")
            elif pet.get("pets_allowed") is False:
                lines.append("• Pet Policy: Pets are not allowed on property.")

        # Extra bed / mattress policy
        extra_bed = rules.get("extra_bed_policy") or {}
        if isinstance(extra_bed, dict):
            rate = extra_bed.get("mattress_charge") or extra_bed.get("extra_bed_charge")
            bed_types_raw = extra_bed.get("bed_types") or ["Mattress"]
            if not isinstance(bed_types_raw, (list, tuple)):
                bed_types_raw = [str(bed_types_raw)]
            bed_types_list = [str(b).strip() for b in bed_types_raw if b and str(b).strip()]
            bed_types = ", ".join(bed_types_list) if bed_types_list else "Mattress"
            if rate:
                lines.append(f"• Extra Bed / Mattress: Available ({bed_types}) at ₹{rate} per night.")
            elif extra_bed.get("adult_bed") == "Yes":
                lines.append(f"• Extra Bed / Mattress: Available upon request ({bed_types}).")

        # Meal rates
        meals = rules.get("meal_prices") or {}
        if isinstance(meals, dict):
            meal_bits = []
            for m_name in ("breakfast", "lunch", "dinner"):
                if meals.get(m_name):
                    meal_bits.append(f"{m_name.title()}: ₹{meals[m_name]}")
            if meal_bits:
                lines.append(f"• Meal Prices: {', '.join(meal_bits)} per person.")

    # Property Notices & Operational Announcements
    raw_notices = doc.get("notices") or doc.get("Notices") or []
    if isinstance(raw_notices, list) and raw_notices:
        formatted_n = format_notices(raw_notices)
        if formatted_n:
            lines.append(f"\n{formatted_n}")

    return "\n".join(lines)


def format_amenities(doc: dict) -> str:
    """Format hotel_amenities document into a clean, guest-facing list."""
    if not doc:
        return "🏨 Amenity information is not available right now. Please contact the front desk."

    lines = ["🏨 *Hotel Amenities*\n"]

    amenities_raw = (
        doc.get("amenities")
        or doc.get("Amenities")
        or doc.get("hotelAmenities")
        or doc.get("hotel_amenities")
        or doc
    )

    found: list[str] = []

    def _extract_amenities(obj: object):
        if isinstance(obj, dict):
            if obj.get("available") is True or obj.get("is_available") is True:
                name = obj.get("name") or obj.get("amenityName") or obj.get("title")
                if name:
                    found.append(str(name).strip())
                return
            for k, v in obj.items():
                if k in ("_id", "hotel_id", "updated_at", "created_at"):
                    continue
                if v is True:
                    found.append(_humanize(k))
                elif isinstance(v, (dict, list)):
                    _extract_amenities(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    found.append(_humanize(item))
                elif isinstance(item, dict):
                    _extract_amenities(item)

    _extract_amenities(amenities_raw)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for a in found:
        if a.lower() not in seen:
            seen.add(a.lower())
            deduped.append(a)

    if deduped:
        for amenity in deduped[:15]:  # Top 15 amenities
            lines.append(f"✅ {amenity}")
    else:
        lines.append("Details not available. Please contact the front desk.")

    lines.append("\nWould you like to make a booking or need any other information?")
    return "\n\n".join(lines)


def _humanize(key: str) -> str:
    """Convert camelCase or snake_case amenity keys to readable strings."""
    import re
    # camelCase to words
    s = re.sub(r'([A-Z])', r' \1', key)
    # snake_case to words
    s = s.replace("_", " ")
    return s.strip().title()
