import logging
import datetime
from typing import Optional, Dict, Any, List
import difflib

from app.repositories import amenity_repository
from app.models.amenity_schemas import AmenityInDB, AccessType

logger = logging.getLogger(__name__)

# Common guest phrasing mapped to canonical amenity_id search keys
AMENITY_SYNONYMS: Dict[str, List[str]] = {
    "lift": ["elevator_lift", "lift", "elevator"],
    "elevator": ["elevator_lift", "elevator", "lift"],
    "stairs": ["elevator_lift", "step_free_entrance"],
    "wifi": ["wifi", "wi-fi", "internet", "lan"],
    "wi-fi": ["wifi", "wi-fi", "internet", "lan"],
    "internet": ["wifi", "wi-fi", "lan"],
    "lan": ["lan", "wifi"],
    "pool": ["swimming_pool", "pool"],
    "swimming": ["swimming_pool", "pool"],
    "swimming pool": ["swimming_pool", "pool"],
    "swimmingpool": ["swimming_pool", "pool"],
    "gym": ["gym_fitness_center", "gym", "fitness"],
    "fitness": ["gym_fitness_center", "gym"],
    "fitness center": ["gym_fitness_center", "gym"],
    "fitness centre": ["gym_fitness_center", "gym"],
    "workout": ["gym_fitness_center", "gym"],
    "ac": ["air_conditioning", "ac"],
    "air conditioning": ["air_conditioning", "ac"],
    "air conditioner": ["air_conditioning", "ac"],
    "cooler": ["air_conditioning"],
    "parking": ["parking", "car_parking"],
    "car parking": ["parking"],
    "gaadi": ["parking"],
    "gaadi parking": ["parking"],
    "gaadi park": ["parking"],
    "gaadi khadi": ["parking"],
    "vehicle parking": ["parking"],
    "two wheeler": ["parking"],
    "four wheeler": ["parking"],
    "ev": ["ev_charging_station"],
    "ev charging": ["ev_charging_station"],
    "charging": ["ev_charging_station"],
    "spa": ["spa", "massage"],
    "massage": ["spa"],
    "wellness": ["spa", "gym_fitness_center"],
    "laundry": ["laundry", "washing_machine", "laundromat"],
    "washing": ["laundry", "washing_machine"],
    "washing machine": ["washing_machine", "laundry"],
    "iron": ["laundry"],
    "ironing": ["laundry"],
    "press": ["laundry"],
    "dry cleaning": ["laundry"],
    "power backup": ["power_backup"],
    "generator": ["power_backup"],
    "power": ["power_backup"],
    "light": ["power_backup"],
    "electricity": ["power_backup"],
    "bijli": ["power_backup"],
    "doctor": ["doctor_on_call", "first_aid_services"],
    "doctor on call": ["doctor_on_call"],
    "medical": ["doctor_on_call", "first_aid_services", "medicare_centre"],
    "first aid": ["first_aid_services", "doctor_on_call"],
    "restaurant": ["restaurant", "dining_area", "in_room_dining"],
    "dining": ["dining_area", "restaurant", "in_room_dining"],
    "food": ["restaurant", "in_room_dining", "food_delivery_available", "breakfast"],
    "breakfast": ["breakfast", "restaurant"],
    "dinner": ["restaurant", "in_room_dining"],
    "lunch": ["restaurant", "in_room_dining"],
    "room service": ["room_service", "in_room_dining"],
    "housekeeping": ["housekeeping"],
    "cleaning": ["housekeeping"],
    "luggage": ["luggage_assistance", "cloak_room"],
    "luggage assistance": ["luggage_assistance"],
    "cloak room": ["cloak_room"],
    "non veg": ["non_vegetarian_food_allowed"],
    "non-veg": ["non_vegetarian_food_allowed"],
    "nonveg": ["non_vegetarian_food_allowed"],
    "chicken": ["non_vegetarian_food_allowed"],
    "meat": ["non_vegetarian_food_allowed"],
    "outside food": ["outside_food_allowed", "food_delivery_available"],
    "swiggy": ["food_delivery_available"],
    "zomato": ["food_delivery_available"],
    "alcohol": ["alcohol_served"],
    "liquor": ["alcohol_served"],
    "wine": ["alcohol_served"],
    "beer": ["alcohol_served"],
    "daru": ["alcohol_served"],
    "sharab": ["alcohol_served"],
    "wheelchair": ["wheelchair"],
    "ramp": ["ramp"],
    "accessible": ["wheelchair", "ramp", "step_free_entrance"],
    "handicapped": ["wheelchair", "ramp", "step_free_entrance"],
    "cctv": ["cctv", "security_guard"],
    "security": ["security_guard", "security_alarms", "cctv"],
    "guard": ["security_guard"],
    "security guard": ["security_guard"],
    "smoke": ["smoking_rooms"],
    "smoking": ["smoking_rooms"],
    "smoking room": ["smoking_rooms"],
    "smoking rooms": ["smoking_rooms"],
    "cigarette": ["smoking_rooms"],
    "sigret": ["smoking_rooms"],
    "driver": ["drivers_accommodation"],
    "driver room": ["drivers_accommodation"],
    "refrigerator": ["refrigerator"],
    "fridge": ["refrigerator"],
    "bar": ["bar"],
    "barbeque": ["barbeque"],
    "bbq": ["barbeque"],
    "reception": ["reception", "concierge"],
    "concierge": ["concierge", "reception"],
    "shuttle": ["shuttle_service", "airport_transfers", "railway_transfers"],
    "airport": ["airport_transfers", "pickup_drop"],
    "airport transfer": ["airport_transfers", "pickup_drop"],
    "cab": ["pickup_drop", "airport_transfers", "railway_transfers"],
    "taxi": ["pickup_drop", "airport_transfers", "railway_transfers"],
    "pickup": ["pickup_drop", "airport_transfers", "railway_transfers"],
    "drop": ["pickup_drop", "airport_transfers", "railway_transfers"],
    "railway": ["railway_transfers", "pickup_drop"],
    "station": ["railway_transfers", "bus_station_transfers"],
}


async def handle_amenity_query(hotel_id: int | str, amenity_name: str, query_types: List[str]) -> str:
    """
    Process an amenity query using Python business logic, returning a structured string
    for Gemini to format into conversational text.
    
    Checks:
    1. Structured hotel_amenities collection (hierarchical or flat)
    2. Meal / Breakfast rate plans (CP, MAP) from rate_plans
    3. Hotel Knowledge Base articles (parking, sightseeing, boats, cabs, policies)
    4. Room-level amenities (Wi-Fi, AC, Hot Water, etc.)
    """
    if not amenity_name:
        return "I need to know which specific amenity or facility you are asking about."

    q_clean = amenity_name.lower().strip()

    # A catalog row that matched but is flagged `available: false`, held back so
    # it can be the answer only if EVERY other source also comes up empty.
    #
    # This is the inconsistency fix. Step 1 used to return on any match,
    # including a negative one, and its structured catalog is the first source
    # checked — so a single `available: false` row ended the search and the
    # guest was told "UNAVAILABLE at this hotel" while step 4 held a positive
    # answer for the same facility. Measured on the live catalog: "gym" matched
    # `Gym/Fitness center (available=False)` and "doctor" matched `Doctor on
    # call (available=False)`, both of which short-circuited everything below.
    #
    # A negative is still a real answer — a genuinely absent facility must still
    # come back as unavailable — so it is deferred, not discarded.
    unavailable_match = None

    # ── 1. Check structured hotel_amenities collection ──
    amenities = await amenity_repository.list_amenities(hotel_id, active_only=False)
    if amenities:
        # Handle "all" or general facilities listing
        if q_clean in (
            "all", "facilities", "amenities", "services", "all amenities",
            "hotel amenities", "what facilities", "show all", "list", "available"
        ):
            available_amenities = [a for a in amenities if a.available]
            lines = ["Available Hotel Amenities & Facilities:\n"]
            by_cat: Dict[str, List[str]] = {}
            for a in available_amenities:
                cat = a.category or "General Facilities"
                desc = f"• **{a.name}**" + (f" — {a.operational_info.notes}" if a.operational_info.notes else "")
                by_cat.setdefault(cat, []).append(desc)
            for cat, items in by_cat.items():
                lines.append(f"*{cat}:*")
                lines.extend(items[:8])
                lines.append("")
            lines.append("Feel free to ask for more details about any facility or service! 🏨")
            return "\n".join(lines)

        best_match = _find_best_match(amenity_name, amenities)
        if best_match and best_match.available:
            return _generate_structured_response(best_match, query_types)
        if best_match:
            # Deferred, not returned — a later source may know better.
            unavailable_match = best_match
            logger.info(
                "AMENITY | step 1 matched %r but available=False for hotel_id=%s "
                "— holding the negative and trying the remaining sources",
                best_match.name, hotel_id,
            )

    # ── 2. Check Breakfast / Meal Plan Rate Plans ──
    meal_keywords = ("breakfast", "dining", "cp", "map", "dinner", "lunch", "restaurant")
    is_outside_food = any(w in q_clean for w in ("outside", "delivery", "zomato", "swiggy"))
    if not is_outside_food and (
        any(kw in q_clean for kw in meal_keywords)
        or any(w in q_clean.split() for w in ("food", "meal", "meals"))
    ):
        from app.repositories.room_repository import get_all_rooms
        rooms = await get_all_rooms(hotel_id)
        if rooms:
            meal_lines = ["Breakfast & Meal Plan Information:"]
            has_meal_plans = False
            for r in rooms:
                cp_plans = [rp for rp in r.rate_plans if "cp" in rp.name.lower() or "breakfast" in rp.name.lower()]
                map_plans = [rp for rp in r.rate_plans if "map" in rp.name.lower() or "dinner" in rp.name.lower() or "lunch" in rp.name.lower()]
                ep_plans = [rp for rp in r.rate_plans if "ep" in rp.name.lower() or "room only" in rp.name.lower()]

                if cp_plans or map_plans:
                    has_meal_plans = True
                    ep_price = ep_plans[0].base_price if ep_plans else 0
                    if cp_plans:
                        cp_price = cp_plans[0].base_price
                        diff = cp_price - ep_price if ep_price else 0
                        meal_lines.append(f"- {r.name}: CP (Breakfast Included) rate is ₹{cp_price:,.0f}/night (Room Only EP is ₹{ep_price:,.0f})")
                    if map_plans:
                        map_price = map_plans[0].base_price
                        meal_lines.append(f"- {r.name}: MAP (Breakfast + Dinner/Lunch) rate is ₹{map_price:,.0f}/night")

            if has_meal_plans:
                meal_lines.append("\nSummary: Breakfast is available as a Continental Plan (CP) package upgrade for our rooms. Guests can also choose MAP (Breakfast + Dinner).")
                return "\n".join(meal_lines)

    # ── 3. Check MongoDB Knowledge Base Articles (Parking, Cabs, Sightseeing, Policies) ──
    # Guard against generic keywords like "all" or short tokens dumping unrelated articles
    if len(q_clean) >= 3 and q_clean not in ("all", "any", "hotel", "amenity", "facilities", "services"):
        try:
            from app.db.mongodb import knowledge_base_collection
            kb_col = knowledge_base_collection()
            variants = [hotel_id, str(hotel_id)]
            if str(hotel_id).isdigit():
                variants.append(int(hotel_id))

            kb_docs = await kb_col.find({"hotel_id": {"$in": variants}}).to_list(50)
            matched_kb = []
            for doc in kb_docs:
                title = str(doc.get("title") or "").lower()
                content = str(doc.get("content") or "")
                content_lower = content.lower()

                # Specific search terms depending on query
                search_terms = []
                if "park" in q_clean:
                    search_terms.extend(["park", "parking", "parkin", "four-wheeler", "vehicle"])
                elif "cab" in q_clean or "taxi" in q_clean:
                    search_terms.extend(["taxi", "cab", "sightseeing", "tour", "transport"])
                elif "boat" in q_clean or "aarti" in q_clean:
                    search_terms.extend(["boat", "ghat", "ganga", "aarti"])
                elif "saree" in q_clean or "shopping" in q_clean:
                    search_terms.extend(["saree", "shopping"])
                elif "lift" in q_clean or "elevator" in q_clean:
                    search_terms.extend(["lift", "elevator"])
                else:
                    search_terms.append(q_clean)

                # Require match in title or strong match in content
                if any(term in title for term in search_terms):
                    matched_kb.append(f"**{doc.get('title', 'Information')}**\n{content}")
                elif any(f" {term} " in f" {content_lower} " for term in search_terms if len(term) >= 4):
                    matched_kb.append(f"**{doc.get('title', 'Information')}**\n{content}")

            if matched_kb:
                return "Hotel Knowledge Base Information:\n\n" + "\n\n---\n\n".join(matched_kb[:3])
        except Exception as e:
            logger.warning(f"Error querying knowledge base articles for amenity '{amenity_name}': {e}")

    # ── 4. Check Room Types Amenities ──
    try:
        from app.repositories.room_repository import get_all_rooms
        rooms = await get_all_rooms(hotel_id)
        room_amenities_found = []
        for r in rooms:
            for k, v in r.amenities.items():
                if v and _room_amenity_matches(q_clean, k):
                    room_amenities_found.append(f"- {k} is available in {r.name}")

        if room_amenities_found:
            return f"Amenity: {amenity_name.title()}\nStatus: AVAILABLE in hotel rooms.\n" + "\n".join(room_amenities_found)
    except Exception as e:
        logger.warning(f"Error checking room amenities: {e}")

    # ── 5. Courteous Unavailable Fallback for known hotel facility queries ──
    #
    # Every source has now been asked. If the structured catalog recognised the
    # facility and flagged it unavailable, that is the most specific negative we
    # hold — quote it rather than the generic line, so the guest gets the row's
    # own name and note.
    if unavailable_match is not None:
        return _generate_structured_response(unavailable_match, query_types)

    return f"Amenity: {amenity_name.title()}\nStatus: UNAVAILABLE at this hotel.\nNote: {amenity_name.title()} is not available at our property."


def _room_amenity_matches(query_lower: str, key: str) -> bool:
    """
    Whether a guest's phrasing names a room-level amenity key.

    Step 4 used bare substring containment both ways, which missed the pairs the
    catalog spells out in full: a guest asking "hot water" never reached
    `Hot & Cold Water`, because neither string contains the other. The room
    amenity list is the source that actually answers for Hotel 111111 — 21 keys
    against a structured catalog that holds no such row — so a miss here is the
    guest being told the hotel has no hot water.

    Same normalisation and token rule as `_find_best_match` step 4, so a phrase
    that matches in one source matches in the other. Deliberately no fuzzy
    matching: at the token level a 0.6 cutoff starts pairing unrelated words.
    """
    k_norm = key.lower().replace("/", " ").replace("-", " ").replace("_", " ")
    q_norm = query_lower.replace("/", " ").replace("-", " ").replace("_", " ")
    q_tokens = set(q_norm.split())
    k_tokens = set(k_norm.split())
    if not q_tokens or not k_tokens:
        return False
    if q_tokens.issubset(k_tokens) or k_tokens.issubset(q_tokens):
        return True
    # Fall back to the old substring rule for the single-word cases the token
    # rule cannot see ("wi" is not a token, "wifi" inside "wi-fi" is).
    return len(query_lower) >= 3 and (query_lower in k_norm or k_norm in query_lower)


def _find_best_match(query_name: str, amenities: List[AmenityInDB]) -> Optional[AmenityInDB]:
    query_lower = query_name.lower().strip()

    # 1. Check Synonym / Alias Dictionary First
    synonym_targets = AMENITY_SYNONYMS.get(query_lower)
    if synonym_targets:
        for target in synonym_targets:
            for a in amenities:
                if target == a.amenity_id.lower() or target == a.name.lower():
                    return a

    # 2. Exact match on name or ID
    for a in amenities:
        if query_lower == a.name.lower() or query_lower == a.amenity_id.lower():
            return a

    # 3. Normalized phrase match (strip punctuation, e.g. "elevator/lift")
    q_norm = query_lower.replace("/", " ").replace("-", " ").replace("_", " ")
    for a in amenities:
        a_norm = a.name.lower().replace("/", " ").replace("-", " ").replace("_", " ")
        if q_norm == a_norm:
            return a

    # 4. Word-boundary or token match
    q_tokens = set(q_norm.split())
    for a in amenities:
        a_tokens = set(a.name.lower().replace("/", " ").replace("-", " ").replace("_", " ").split())
        if q_tokens.issubset(a_tokens) or a_tokens.issubset(q_tokens):
            return a

    # 5. Substring match (preferring full token match)
    for a in amenities:
        if len(query_lower) >= 3 and (query_lower in a.name.lower() or query_lower in a.amenity_id.lower()):
            return a

    # 6. Fuzzy match
    names = {a.name: a for a in amenities}
    matches = difflib.get_close_matches(query_lower, [n.lower() for n in names.keys()], n=1, cutoff=0.6)
    if matches:
        match_lower = matches[0]
        for a in amenities:
            if a.name.lower() == match_lower:
                return a

    return None


def _generate_structured_response(amenity: AmenityInDB, query_types: List[str]) -> str:
    lines = [f"Amenity: {amenity.name}"]

    # General availability
    if not amenity.available:
        notes = amenity.operational_info.notes or ""
        notes_str = f"\nNote: {notes}" if notes else "\nNote: This facility is not available at our property."
        return f"Amenity: {amenity.name}\nStatus: UNAVAILABLE at this hotel.{notes_str}"

    now = datetime.datetime.now()
    today_str = now.strftime("%A").lower()
    date_str = now.strftime("%Y-%m-%d")

    # Check maintenance closures
    for period in amenity.scheduling.maintenance_periods:
        if period.start_date <= now <= period.end_date:
            return f"Amenity: {amenity.name}\nStatus: Temporarily unavailable due to maintenance."

    # Check schedule for today
    schedule_today = None
    is_open_now = True

    if amenity.scheduling.is_24_hours:
        schedule_today = "Open 24 hours"
        is_open_now = True
    elif not amenity.scheduling.default_schedule and not amenity.scheduling.special_schedule:
        # Standard hotel infrastructure / on-request service without restricted schedule
        schedule_today = "Available throughout stay"
        is_open_now = True
    else:
        # Check holiday override
        if date_str in amenity.scheduling.special_schedule:
            s = amenity.scheduling.special_schedule[date_str]
            if s.open_time and s.close_time:
                schedule_today = f"{s.open_time} to {s.close_time}"
            elif amenity.booking_rules.booking_required:
                schedule_today = "Available by appointment."
            else:
                schedule_today = "Schedule varies. Please check with the front desk."
        elif today_str in amenity.scheduling.default_schedule:
            s = amenity.scheduling.default_schedule[today_str]
            if s.open_time and s.close_time:
                schedule_today = f"{s.open_time} to {s.close_time}"
            elif amenity.booking_rules.booking_required:
                schedule_today = "Available by appointment."
            else:
                schedule_today = "Schedule varies. Please check with the front desk."
        else:
            schedule_today = "Closed today"
            is_open_now = False

        # Determine if currently open
        if schedule_today and " to " in schedule_today:
            try:
                open_t, close_t = schedule_today.split(" to ")
                open_dt = datetime.datetime.strptime(open_t, "%H:%M").time()
                close_dt = datetime.datetime.strptime(close_t, "%H:%M").time()
                current_t = now.time()

                if open_dt <= close_dt:
                    is_open_now = open_dt <= current_t <= close_dt
                else:  # crosses midnight
                    is_open_now = current_t >= open_dt or current_t <= close_dt
            except Exception as e:
                logger.error(f"Error parsing times: {e}")

    # Build detailed response based on query type
    has_type = lambda q: q in query_types or "general" in query_types

    lines.append(f"Status: AVAILABLE at this hotel.")
    if has_type("timings") or has_type("availability"):
        lines.append(f"Schedule: {schedule_today}")
        lines.append(f"Currently Open / Active: {'Yes' if is_open_now else 'No'}")

    if has_type("pricing") or has_type("access") or has_type("restrictions"):
        access = amenity.access_rules.access_type.value.replace("_", " ").title()
        lines.append(f"Access Type: {access}")

        if amenity.access_rules.access_type == AccessType.PAID and amenity.pricing.price:
            tax_str = " (Tax Inclusive)" if amenity.pricing.tax_inclusive else " (Plus Taxes)"
            lines.append(f"Price: {amenity.pricing.currency} {amenity.pricing.price}{tax_str}")

        if amenity.access_rules.allowed_room_types:
            rooms = ", ".join(amenity.access_rules.allowed_room_types)
            lines.append(f"Restricted to Room Types: {rooms}")

    if has_type("booking_requirement"):
        if amenity.booking_rules.booking_required:
            lines.append("Booking Required: Yes")
            if amenity.booking_rules.minimum_notice_minutes:
                lines.append(f"Minimum Notice: {amenity.booking_rules.minimum_notice_minutes} minutes")
            if amenity.booking_rules.slot_duration_minutes:
                lines.append(f"Slot Duration: {amenity.booking_rules.slot_duration_minutes} minutes")
        else:
            lines.append("Booking Required: No")

    if has_type("location") and (amenity.operational_info.location or amenity.category):
        loc = amenity.operational_info.location or amenity.category
        lines.append(f"Location / Category: {loc}")

    if has_type("description") and amenity.description:
        lines.append(f"Description: {amenity.description}")

    if has_type("restrictions") and amenity.operational_info.notes:
        lines.append(f"Notes / Details: {amenity.operational_info.notes}")

    return "\n".join(lines)
