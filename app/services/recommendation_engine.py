import logging
import re
from typing import List, Dict, Any, Tuple
from app.models.schemas import RoomType
from app.services.allocation_engine import room_matches_feature

logger = logging.getLogger(__name__)

#: The spellings that mean "the air-conditioning question". Answered by
#: `room_matches_feature` — see the note in `_matches_feature`.
_AC_FEATURES = ("ac", "air conditioning", "air-conditioning", "a/c")


def _matches_feature(feature: str, opt: Dict[str, Any], all_rooms: List[RoomType]) -> bool:
    """
    Whether this option satisfies `feature`.

    The AC / Non-AC question is delegated to `room_matches_feature`, which is the
    predicate the allocator filters candidates on and that the rooms card prices
    its options from. Answering it a second time here is how the two came to
    disagree, and the disagreement is visible to guests:

      * `STANDARD NON AC TRIPPLE BED` carries `amenities = {"Air Conditioning":
        True}` — the hotel's own data contradicting its own room name — so the
        name check below rejects it and the amenity loop then readmitted it. It
        appeared in an AC-only list with the word "NON AC" in its name.
      * `DELUXE ROOM FOUR BED` has no "ac" in its name and no AC amenity, so this
        function excluded it from an AC list — while the allocator and the card
        both treated it as AC, and the card's "Comfortable" row recommended it.

    `room_matches_feature` resolves a name that says non-AC by that name alone,
    and treats a room as AC unless something positively says otherwise. Both are
    the behaviour a guest would predict, and neither is re-derived here.

    Other features (meal plan, bed type, room type) keep the string match below.
    """
    pref_lower = feature.lower()
    synonyms = [pref_lower]

    if pref_lower in _AC_FEATURES or pref_lower in ("non ac", "non-ac", "nonac"):
        for selected_room in opt.get("rooms", []):
            orig_room = next(
                (r for r in all_rooms if r.room_id == selected_room.get("room_id")), None
            )
            if orig_room is not None and room_matches_feature(orig_room, pref_lower):
                return True
        return False

    # Everything else — meal plan, bed type, room type — is a plain word match
    # against the room's own name and its amenity keys.
    for selected_room in opt.get("rooms", []):
        room_id = selected_room.get("room_id")
        orig_room = next((r for r in all_rooms if r.room_id == room_id), None)
        if not orig_room:
            continue

        name_lower = orig_room.name.lower()
        if any(bool(re.search(r'\b' + re.escape(syn) + r'\b', name_lower)) for syn in synonyms):
            return True

        for k, v in orig_room.amenities.items():
            k_lower = str(k).lower()
            if v and any(bool(re.search(r'\b' + re.escape(syn) + r'\b', k_lower)) for syn in synonyms):
                return True
    return False

def _matches_room_type(pref: str, opt: Dict[str, Any], all_rooms: List[RoomType]) -> bool:
    pref_lower = pref.lower()
    prefs = [p.strip() for p in re.split(r'\bor\b|\band\b|,|/', pref_lower) if p.strip()]
    
    for selected_room in opt.get("rooms", []):
        room_name = selected_room.get("room_name", "").lower()
        room_id = selected_room.get("room_id")
        orig_room = next((r for r in all_rooms if r.room_id == room_id), None)
        orig_type = orig_room.room_type.lower() if (orig_room and orig_room.room_type) else ""
        
        for p in prefs:
            if p in room_name or p in orig_type:
                return True
    return False

def _matches_meal_plan(pref: str, opt: Dict[str, Any]) -> bool:
    pref_upper = pref.upper()
    for m in opt.get("meal_upgrades", []):
        if m.get("meal_plan", "").upper() == pref_upper:
            return True
    return False

def rank_and_select_options(
    options: List[Dict[str, Any]],
    state: Dict[str, Any],
    all_rooms: List[RoomType]
) -> Tuple[List[Dict[str, Any]], str]:
    if not options:
        return [], "I'm sorry, I couldn't find any rooms matching your criteria."

    feature_prefs = state.get("feature_preferences") or []
    room_type_pref = state.get("room_type_preference")
    meal_plan_pref = state.get("meal_plan_preference")
    budget = state.get("budget")
    requested_num_rooms = state.get("num_rooms")
    
    # Read structured preference ENUMs set by Gemini — no natural language parsing here
    recommendation_pref = (state.get("recommendation_preference") or "").upper()
    display_pref = (state.get("display_preference") or "").upper()
    room_count_pref = (state.get("room_count_preference") or "").upper()
    comfort_pref = (state.get("comfort_preference") or "").upper()
    
    wants_comfort = comfort_pref == "COMFORTABLE_STAY"
    wants_fewer_rooms = room_count_pref == "FEWER_ROOMS"
    wants_cheapest = recommendation_pref == "CHEAPEST"
    show_all = display_pref == "SHOW_ALL"

    scored_options = []
    
    for opt in options:
        score = 0.0
        labels = []
        
        # 1. Feature Matching
        for feat in feature_prefs:
            # Skip structural preferences from standard feature matching
            if feat.lower() in ["comfortable stay", "no extra beds", "comfort", "fewer rooms", "one room only", "minimum rooms"]:
                continue
            if _matches_feature(feat, opt, all_rooms):
                score += 50.0
                labels.append(f"Matches '{feat}'")
                
        # 2. Room Type Match
        if room_type_pref and _matches_room_type(room_type_pref, opt, all_rooms):
            score += 100.0
            labels.append(f"Matches {room_type_pref}")
            
        # Meal Plan Match
        if meal_plan_pref and _matches_meal_plan(meal_plan_pref, opt):
            score += 100.0
            labels.append(f"Matches {meal_plan_pref}")
            
        # 3. Budget Compliance
        total_price = opt.get("total_price_per_night", 0.0)
        if budget:
            if total_price <= budget:
                score += 100.0
                labels.append("Within Budget")
            else:
                score -= 100.0
                
        # 4. Structural & Capacity Fit Preferences
        alloc_type = opt.get("allocation_type", "COMFORT")
        extra_adults = opt.get("extra_adult_count", 0)
        extra_children = opt.get("extra_child_count", 0)
        total_extra = extra_adults + extra_children
        num_rooms = len(opt.get("rooms", []))

        total_base_a = opt.get("base_adults", 0)
        total_base_c = opt.get("base_children", 0)
        total_base_cap = total_base_a + total_base_c

        total_adults = state.get("adults", 2) or 2
        total_children = state.get("children", 0) or 0
        total_guests = total_adults + total_children

        delta = total_base_cap - total_guests
        if delta == 0:
            fit_tier = 0  # Exact base capacity fit (e.g. 4-person room for 2 adults + 2 children)
        elif delta < 0 and total_extra > 0:
            fit_tier = abs(delta)  # Needs extra beds/surcharges
        elif delta == 1:
            fit_tier = 1  # 1 step above base capacity (e.g. 2 double rooms for 3 guests - comfortable)
        elif delta > 1:
            fit_tier = 3 + (delta * 3)  # Large over-capacity (e.g. 2 Family rooms for 3 adults)
        else:
            fit_tier = 10 + abs(delta)

        # Penalize over-capacity (fit_tier >= 2) so exact-fit rooms rank ahead of oversized rooms
        if fit_tier >= 2:
            score -= (fit_tier * 300.0)
        
        if wants_comfort and alloc_type == "COMFORT":
            score += 150.0
        if wants_fewer_rooms:
            score -= (num_rooms * 150.0)
        if wants_cheapest:
            score -= (total_price * 10.0)  # Massive penalty for higher price overrides other preferences
            
        if requested_num_rooms:
            if num_rooms == requested_num_rooms:
                score += 100.0
            else:
                score -= (abs(num_rooms - requested_num_rooms) * 50.0)
                
        # 5. Base penalties
        score -= (total_price / 1000.0)  # Cheaper is slightly better
        score -= (extra_adults * 1.0)    # Minor penalty for extra beds (reduced to let price dominate)
        score -= (num_rooms * 2.0)       # Minor penalty for more rooms (reduced to let price dominate)
        
        if alloc_type == "MAX_OCCUPANCY":
            if extra_adults > 0:
                labels.append(f"Includes {extra_adults} Extra Adult{'s' if extra_adults > 1 else ''}")
        else:
            labels.append("No Extra Adult Charges")
            
        scored_options.append({"opt": opt, "score": score, "labels": labels})

    # Group by (allocation_type, room names tuple, meal plans tuple) for strict Pareto dominance
    groups = {}
    for item in scored_options:
        opt = item["opt"]
        alloc_type = opt.get("allocation_type", "COMFORT")
        room_names = tuple(sorted([r.get("room_name", "") for r in opt.get("rooms", [])]))
        meals = tuple(sorted([m.get("meal_plan", "") for m in opt.get("meal_upgrades", [])]))
        group_key = (alloc_type, room_names, meals)
        
        if group_key not in groups:
            groups[group_key] = item
        else:
            # Keep the highest scoring one in the group (which will be the cheaper one)
            if item["score"] > groups[group_key]["score"]:
                groups[group_key] = item
                
    diverse_options = list(groups.values())
    diverse_options.sort(key=lambda x: x["score"], reverse=True)
    
    # Global labels
    if diverse_options:
        diverse_options[0]["labels"].append("Recommended")
        
        min_price = min(item["opt"].get("total_price_per_night", 0.0) for item in diverse_options)
        for item in diverse_options:
            if item["opt"].get("total_price_per_night", 0.0) == min_price and "Lowest Price" not in item["labels"]:
                item["labels"].append("Lowest Price")
                break
                
        min_rooms = min(len(item["opt"].get("rooms", [])) for item in diverse_options)
        for item in diverse_options:
            if len(item["opt"].get("rooms", [])) == min_rooms and "Fewer Rooms" not in item["labels"]:
                # Only add if min_rooms is actually fewer than some other option
                if any(len(other["opt"].get("rooms", [])) > min_rooms for other in diverse_options):
                    item["labels"].append("Fewer Rooms")
                break
                
        comfort_options = [item for item in diverse_options if item["opt"].get("allocation_type") == "COMFORT"]
        if comfort_options:
            top_comfort = max(comfort_options, key=lambda x: x["score"])
            if "Most Comfortable" not in top_comfort["labels"]:
                top_comfort["labels"].append("Most Comfortable")

    # Selection: Priority slotting to guarantee MUST-HAVES don't get pushed out by padding.
    MAX_OPTIONS = len(diverse_options) if show_all else 6
    MIN_OPTIONS = len(diverse_options) if show_all else min(6, len(diverse_options))
    final_selection = []
    selected_indices = set()
    
    # Track what we NEED to show
    must_have_labels = set()
    if room_type_pref and not wants_cheapest:  # If they just want cheapest, skip forcing a specific room type
        must_have_labels.add(f"Matches {room_type_pref}")
    if meal_plan_pref:
        must_have_labels.add(f"Matches {meal_plan_pref}")
    for feat in feature_prefs:
        # feature_prefs only contains true amenity/feature preferences now — no structural strings
        must_have_labels.add(f"Matches '{feat}'")
            
    # Remove any must-haves that are literally impossible (not present in any diverse option)
    possible_labels = {lbl for item in diverse_options for lbl in item["labels"]}
    must_have_labels = must_have_labels.intersection(possible_labels)
    
    # PASS 1: Always include top option
    if diverse_options:
        final_selection.append(diverse_options[0])
        selected_indices.add(0)
        
    # PASS 2: Fulfill Must-Haves
    for lbl in must_have_labels:
        if any(lbl in diverse_options[idx]["labels"] for idx in selected_indices):
            continue
        for i, item in enumerate(diverse_options):
            if i not in selected_indices and lbl in item["labels"]:
                final_selection.append(item)
                selected_indices.add(i)
                break
                
    # PASS 3: Structural diversity
    desired_struct = []
    if not any(diverse_options[idx]["opt"].get("allocation_type") == "COMFORT" for idx in selected_indices):
        desired_struct.append(lambda item: item["opt"].get("allocation_type") == "COMFORT")
    if not any(diverse_options[idx]["opt"].get("allocation_type") == "MAX_OCCUPANCY" for idx in selected_indices):
        desired_struct.append(lambda item: item["opt"].get("allocation_type") == "MAX_OCCUPANCY")
    if not any("Lowest Price" in diverse_options[idx]["labels"] for idx in selected_indices):
        desired_struct.append(lambda item: "Lowest Price" in item["labels"])

    # Guarantee AC and Non-AC balance when user wants to see both types
    wants_both_ac_types = (state.get("ac_preference") or "").upper() in ("NO_PREFERENCE", "BOTH", "SHOW_BOTH") or not feature_prefs
    if wants_both_ac_types:
        if not any(_matches_feature("AC", diverse_options[idx]["opt"], all_rooms) for idx in selected_indices):
            desired_struct.append(lambda item: _matches_feature("AC", item["opt"], all_rooms))
        if not any(_matches_feature("Non-AC", diverse_options[idx]["opt"], all_rooms) for idx in selected_indices):
            desired_struct.append(lambda item: _matches_feature("Non-AC", item["opt"], all_rooms))
        
    for condition in desired_struct:
        if len(final_selection) >= MAX_OPTIONS:
            break
        for i, item in enumerate(diverse_options):
            if i not in selected_indices and condition(item):
                final_selection.append(item)
                selected_indices.add(i)
                break
                
    # PASS 4: Padding to reach minimum options
    for i, item in enumerate(diverse_options):
        if len(final_selection) >= MIN_OPTIONS:
            break
        if i not in selected_indices:
            final_selection.append(item)
            selected_indices.add(i)
            
    # Enforce maximum cap (preserves priority order: Top -> Must Haves -> Structural -> Padding)
    final_selection = final_selection[:MAX_OPTIONS]
    
    # Sort final selection by score
    final_selection.sort(key=lambda x: x["score"], reverse=True)
    
    # Assign labels back to options
    result_options = []
    for item in final_selection:
        opt = dict(item["opt"]) # Create copy
        # Deduplicate labels and keep order
        seen = set()
        clean_labels = []
        # Priority ordering of labels
        priority_labels = ["Recommended", "Lowest Price", "Most Comfortable", "Fewer Rooms", "Within Budget"]
        
        for p in priority_labels:
            if p in item["labels"]:
                clean_labels.append(p)
                seen.add(p)
                
        for lbl in item["labels"]:
            if lbl not in seen:
                clean_labels.append(lbl)
                seen.add(lbl)
                
        opt["recommendation_labels"] = clean_labels
        result_options.append(opt)
        
    # Generate text
    if budget and all(opt.get("total_price_per_night", 0) > budget for opt in result_options):
        text = f"I found some options for you. Note that all options exceed your budget of ₹{budget:,.0f}/night:"
    elif budget:
        text = f"I found some options for you. Options within your budget of ₹{budget:,.0f}/night are marked below:"
    else:
        text = "I found some great options for you. Please select an option:"
        
    return result_options, text
