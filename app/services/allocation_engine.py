"""
Multi-Room Allocation Engine

Generates combinations of rooms to satisfy guest capacity,
optimizes guest distribution to minimize extra guest charges,
and ranks the resulting packages by price.
"""

import itertools
import logging
from typing import List, Tuple
from app.config.settings import get_settings
from app.models.schemas import RoomType, SelectedRoom, AllocationOption
from app.utils.ac_text import ac_preference_from_text
from app.utils.meal_plan import classify_rate_plan, plan_charges_above_base

logger = logging.getLogger(__name__)


def plan_owes_for_an_extra_person(rate_type_id: str | None, plan_name: str | None) -> bool:
    """Does this room's *rate plan* owe for an adult the room does not seat?

    Wrapped so both charge sites ask the question the same way, and so the
    settings gate lives in exactly one place. `False` also when the flag is off,
    which is what makes the flag a one-push rollback rather than a second rule.
    """
    if not get_settings().EXTRA_PERSON_CHARGED_ON_CP_MAP:
        return False
    return plan_charges_above_base(classify_rate_plan(rate_type_id, plan_name))

_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "double": "2", "twin": "2", "triple": "3", "quad": "4",
}
_NUMBER_WORDS_RE = None


def normalize_bed_counts(text: str) -> str:
    """
    Rewrite number words as digits so "4 bed" and "four bed" are the same
    request: "DELUXE ROOM FOUR BED" -> "deluxe room 4 bed".

    Without this the two never meet. Hotel 111111 holds a room literally named
    "DELUXE ROOM FOUR BED" — the exact room a guest asking for a "4 bed" wanted —
    and `room_matches_feature` matched it zero times, because a word-boundary
    search for "4 bed" cannot see "four bed". The empty match then emptied the
    whole catalogue (defect #10, Chat A 06:36:55).
    """
    global _NUMBER_WORDS_RE
    import re
    if _NUMBER_WORDS_RE is None:
        _NUMBER_WORDS_RE = re.compile(
            r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE
        )
    return _NUMBER_WORDS_RE.sub(lambda m: _NUMBER_WORDS[m.group(1).lower()], text)


# A guest either has AC or does not; these are real inventory attributes that the
# hotel either has or cannot offer, so they stay hard filters even when nothing
# matches. Every other preference is a description harvested from a sentence.
_HARD_INVENTORY_FEATURES = frozenset({
    "ac", "a/c", "air conditioning", "air-conditioning", "air conditioned",
    "non ac", "non-ac", "nonac", "non a/c",
})


def is_hard_inventory_feature(pref: str) -> bool:
    """True for the AC / Non-AC pair, which must never be silently dropped."""
    return (pref or "").lower().strip() in _HARD_INVENTORY_FEATURES


# How many ranked combinations leave the allocator. The search itself stops at 80
# valid options, and every one of them used to be handed back, so the caller
# priced all 80 — each a serial `calculate_price_breakdown` call — before the
# card threw away everything past the first three. A large party could spend the
# whole turn's budget on prices no guest would ever see (measured 2026-10-25: 87
# pricing calls in one turn). The cap is applied *after* ranking, so the ten that
# survive are the ten the card would have drawn from anyway. Kept above
# `ROOM_CARD_MAX_COMBOS` (3) so the card still fills its slots when the best few
# combinations are duplicates of one another.
MAX_ALLOCATION_OPTIONS = 10


def resolve_honourable_features(
    rooms: List[RoomType], feature_preferences: list[str] | None
) -> list[str]:
    """
    The preferences this catalogue can actually act on.

    A preference nothing matches cannot be a filter. "4 bed" at a hotel whose
    room is spelled "DELUXE ROOM FOUR BED" used to match nothing and empty the
    entire catalogue, so the guest was told no rooms were available while the
    room sat in inventory (defect #10, Chat A 06:36:55) — and, because the
    preference stayed in state, every later turn repeated it.

    AC / Non-AC are exempt and always survive: they describe real inventory the
    hotel either has or cannot offer, so an all-non-AC hotel must still answer
    "no" to a guest who asked for AC rather than quietly showing them non-AC
    rooms.

    Resolved once and reused, so the upfront filter and the post-allocation
    filter cannot disagree — they were two copies of this logic, and only one of
    them was ever going to get fixed.
    """
    return [
        f for f in (feature_preferences or [])
        if is_hard_inventory_feature(f)
        or any(room_matches_feature(r, f) for r in rooms)
    ]


def _option_is_ac(opt: AllocationOption, rooms: List[RoomType]) -> bool | None:
    """
    Whether an option is entirely AC rooms, entirely Non-AC, or mixed.

    `None` means mixed. A set built from both classes is its own answer and must
    not be counted toward either side of the diversity rule below — the rule
    exists so that a guest who asked to see both classes sees a set from each,
    and a mixed set is not one of them.
    """
    by_id = {r.room_id: r for r in rooms}
    flags = {
        room_matches_feature(by_id.get(sr.room_id), "AC") for sr in (opt.rooms or [])
    }
    if not flags:
        return None
    if len(flags) > 1:
        return None
    return flags.pop()


def _keep_both_classes(
    options: List[AllocationOption],
    rooms: List[RoomType],
    limit: int,
    per_class: int = 3,
) -> List[AllocationOption]:
    """
    Cut the ranked list to `limit`, but never let one AC class take every slot.

    Why this exists, and why it is here rather than on the card: the ranking
    sorts every combination globally — fit first, then price — and then throws
    away everything past the tenth. On a hotel whose Non-AC rooms are a
    thousand rupees a room cheaper than its AC ones, and a party big enough that
    both classes need four rooms, the ten cheapest combinations are *all*
    Non-AC. The AC sets are not ranked low; they are ranked out before the rooms
    card is ever built, so no card-side change could have shown them. Live
    report, 2026-10-28: a party of 12 was offered two Non-AC combinations while
    the ROOMS list above them advertised four AC room types.

    The fix keeps the ordering and changes only *which* ten survive: the best
    `limit - per_class` overall, plus the best `per_class` of each class, in the
    original rank order. The top of the list — which is what an ordinary card
    reads — is therefore unchanged; the diversity is funded out of the tail.

    Does nothing when either class is absent or when the list is short enough to
    survive whole, so a hotel with one class, a one-room party, or a list of
    eight options all reach the card exactly as before. Gated by
    `ROOM_CARD_SHOW_BOTH_ENABLED`: with the flag off this is the plain `[:limit]`
    it replaced.
    """
    ranked = options[:limit]
    if len(options) <= limit or not get_settings().ROOM_CARD_SHOW_BOTH_ENABLED:
        return ranked

    ac_flags = {id(opt): _option_is_ac(opt, rooms) for opt in options}
    has_ac = any(flag is True for flag in ac_flags.values())
    has_non_ac = any(flag is False for flag in ac_flags.values())
    if not (has_ac and has_non_ac):
        return ranked

    kept = options[: max(limit - per_class, 0)]
    kept_ids = {id(opt) for opt in kept}
    for want in (True, False):
        taken = sum(1 for opt in kept if ac_flags[id(opt)] is want)
        for opt in options:
            # `len(kept)` is checked as well as `taken`: a mixed-class set counts
            # toward neither side, so two classes that both need topping up can
            # ask for more rows than the limit allows.
            if taken >= per_class or len(kept) >= limit:
                break
            if ac_flags[id(opt)] is want and id(opt) not in kept_ids:
                kept.append(opt)
                kept_ids.add(id(opt))
                taken += 1

    # Back into rank order: the card reads this list in order, and the extra
    # rows must not sit out of sequence behind the ones they rank above.
    order = {id(opt): index for index, opt in enumerate(options)}
    return sorted(kept, key=lambda opt: order[id(opt)])


def room_matches_feature(orig_room: RoomType | None, pref: str) -> bool:
    if not orig_room:
        return False
    import re
    pref_lower = normalize_bed_counts(pref.lower().strip())
    name_lower = normalize_bed_counts(orig_room.name.lower())

    if pref_lower in ("ac", "air conditioning", "air-conditioning", "a/c"):
        if "non ac" in name_lower or "non-ac" in name_lower or "nonac" in name_lower:
            return False
        for k, v in (orig_room.amenities or {}).items():
            k_lower = str(k).lower()
            if v and ("non ac" in k_lower or "non-ac" in k_lower or "nonac" in k_lower):
                return False
        if orig_room.amenities and any(
            (str(k).lower() in ("ac", "air conditioning", "air-conditioning", "a/c") and not v)
            for k, v in orig_room.amenities.items()
        ):
            return False
        return True
    elif pref_lower in ("non ac", "non-ac", "nonac"):
        if "non ac" in name_lower or "non-ac" in name_lower or "nonac" in name_lower:
            return True
        for k, v in (orig_room.amenities or {}).items():
            k_lower = str(k).lower()
            if v and ("non ac" in k_lower or "non-ac" in k_lower or "nonac" in k_lower):
                return True
        return False
    else:
        if bool(re.search(r'\b' + re.escape(pref_lower) + r'\b', name_lower)):
            return True
        for k, v in (orig_room.amenities or {}).items():
            k_lower = normalize_bed_counts(str(k).lower())
            if v and bool(re.search(r'\b' + re.escape(pref_lower) + r'\b', k_lower)):
                return True
        return False


# Amenity keys the catalogue uses for the two classes. `room_matches_feature`
# already carried these strings; they are named here so the configuration reader
# below and the name reader above cannot drift apart.
_AC_AMENITY_KEYS = (
    "ac", "a/c", "air conditioning", "air-conditioning",
    "air conditioned", "air-conditioned",
)
_NON_AC_AMENITY_KEYS = ("non ac", "non-ac", "nonac", "non a/c")


def configured_ac_state(room: RoomType | None) -> bool | None:
    """What the room's OWN amenity configuration says about air conditioning.

    `True`  — the hotel has configured AC for this room.
    `False` — the hotel has configured it as non-AC.
    `None`  — the amenities say nothing either way.

    This is a *different reader* from `room_matches_feature`, which reads the
    room's name first. Both are needed, because the two disagree in real data:

    Sample Hotel (111111), measured 2026-11-03. Two rooms are still named
    "STANDARD ROOM (NON-AC)" and "STANDARD TRIPPLE BED (NON-AC)", and the hotel
    has since configured Air Conditioning = yes on both — the amenity list on
    each row carries "Air Conditioning", and `amenity_details.room_essentials.
    air_conditioning.available` is true. The name never changed, so
    `room_matches_feature(r, "Non-AC")` still answers "yes", the search finds a
    non-AC option, and the bot keeps asking the AC / Non-AC question whose
    second answer the hotel no longer sells.

    An explicit Non-AC amenity outranks an AC one: a room whose configuration
    says both is not a room to sell as air-conditioned.
    """
    if not room:
        return None

    # `RoomType.amenities` is `dict[str, bool]`; `room_repository` has already
    # turned the stored list of amenity names into `{name: True}`.
    says_ac = None
    for key, value in (getattr(room, "amenities", None) or {}).items():
        key_lower = str(key).lower().strip()
        if key_lower in _NON_AC_AMENITY_KEYS or any(
            phrase in key_lower for phrase in _NON_AC_AMENITY_KEYS
        ):
            if value:
                return False
        if key_lower in _AC_AMENITY_KEYS and says_ac is None:
            says_ac = bool(value)

    return says_ac


def ac_choice_is_real(non_ac_options, catalogue_rooms) -> bool:
    """Is there a genuine Non-AC side to an AC vs Non-AC choice?

    The question exists so a guest can pick a class. When every room on the
    non-AC side of the search is, by its current configuration, an AC room,
    there is nothing to choose: the second answer would quote the first
    answer's rooms. Asking anyway is what a guest at Sample Hotel sees today.

    `non_ac_options` come from the search and carry only `room_id` /
    `room_name`, so the configuration is read from `catalogue_rooms` — the same
    `RoomType` objects the search was built from.

    Only *explicitly configured* AC rooms count toward "no choice". A room whose
    amenities say nothing keeps the name as its only evidence — that is every
    hotel that has not filled its amenity configuration in, and their behaviour
    is unchanged. Likewise a single unknown among the candidates keeps the
    question, because that room may genuinely have no AC.
    """
    if not non_ac_options:
        return False

    by_id = {room.room_id: room for room in (catalogue_rooms or [])}
    configured = [
        configured_ac_state(by_id.get(selected.room_id))
        for option in non_ac_options
        for selected in (getattr(option, "rooms", None) or [])
    ]
    if not configured or any(state is None for state in configured):
        return True
    return not all(configured)


def get_preferred_rate_plan(room: RoomType, meal_pref: str | None = None):
    """
    Get the preferred active rate plan based on meal plan preference, fallback to cheapest.

    Handles compound preferences like "breakfast and dinner" (MAP/Half-Board) by scoring each
    rate plan against all meal keywords extracted from the preference string, then picking the
    best match. Falls back to the cheapest active plan only when nothing matches.
    """
    logger.info(f"Inside get_preferred_rate_plan for {room.name} with meal_pref={meal_pref}")
    active_plans = [rp for rp in room.rate_plans if rp.active]
    if not active_plans:
        return None


    if not meal_pref:
        return min(active_plans, key=lambda rp: rp.base_price)

    pref_lower = meal_pref.lower()
    logger.info(f"DEBUG: meal_pref={pref_lower} for room={room.name}")

    no_breakfast = any(neg in pref_lower for neg in ["without breakfast", "no breakfast", "w/o breakfast", "excluding breakfast", "dont need breakfast", "don't need breakfast", "without bf", "no bf"])
    no_lunch = any(neg in pref_lower for neg in ["without lunch", "no lunch", "w/o lunch", "excluding lunch"])
    no_dinner = any(neg in pref_lower for neg in ["without dinner", "no dinner", "w/o dinner", "excluding dinner"])

    has_breakfast = ("breakfast" in pref_lower or "cp" in pref_lower) and not no_breakfast
    has_lunch     = ("lunch" in pref_lower) and not no_lunch
    has_dinner    = ("dinner" in pref_lower) and not no_dinner
    has_map       = ("map" in pref_lower) and not (no_breakfast or no_dinner)
    has_ap        = ("ap" in pref_lower or "all meals" in pref_lower) and not (no_breakfast or no_lunch or no_dinner)
    has_ep        = ("ep" in pref_lower or "room only" in pref_lower or no_breakfast)

    # Determine the desired plan tier:
    # AP  → breakfast + lunch + dinner
    # MAP → breakfast + dinner
    # CP  → breakfast only
    # EP  → room only
    wants_all_meals    = has_ap
    wants_map          = has_map or (has_breakfast and has_dinner and not has_ap)
    wants_breakfast    = has_breakfast and not wants_map and not wants_all_meals
    wants_lunch        = has_lunch and not wants_map and not wants_all_meals
    wants_room_only    = has_ep and not (has_breakfast or has_lunch or has_dinner or has_map or has_ap)

    def _score(rp) -> int:
        n = rp.name.lower()
        if wants_all_meals:
            if "ap" in n or "all meals" in n or "all inclusive" in n:
                return 3
            if "map" in n or ("breakfast" in n and "dinner" in n):
                return 2
            if "breakfast" in n:
                return 1
        elif wants_map:
            if "map" in n or ("breakfast" in n and "dinner" in n) or ("half board" in n):
                return 3
            if "breakfast" in n:
                return 1  # at least partial match
        elif wants_breakfast:
            if "breakfast" in n or "cp" in n:
                return 3
        elif wants_lunch:
            if "lunch" in n:
                return 3
        elif wants_room_only:
            if "ep" in n or "room only" in n:
                return 3
        return 0

    best_rp = max(active_plans, key=lambda rp: (_score(rp), -rp.base_price))
    if _score(best_rp) > 0:
        logger.info(f"DEBUG: Matched plan '{best_rp.name}' for {room.name} (pref={pref_lower})")
        return best_rp

    # No plan matched the preference → fallback to cheapest
    logger.info(f"DEBUG: No meal plan match for {room.name} (pref={pref_lower}), using cheapest.")
    return min(active_plans, key=lambda rp: rp.base_price)

def _apply_paid_child_charge(
    options: List[AllocationOption],
    room_pool: List[RoomType],
    paid_children: int,
    fold_into_price: bool,
) -> None:
    """Charge the hotel's child policy band: one rate, per chargeable child, per night.

    `paid_children` counts the children the HOTEL's own policy makes chargeable
    — the band strictly between `free_till` and `paid_till`. The validator works
    that out, because it is the only place the ages and the policy are both in
    hand; this function only turns the count into money.

    The rate is the room's `extra_child_rate`. Nothing sets it per hotel yet, so
    it is the catalogue-wide placeholder that `room_repository` puts on every
    room when the document has no rate of its own. When the real child rate
    lands on the room documents, this picks it up with no change here.

    **Not charged on top of the occupancy rule.** `distribute_guests` already
    charges a child who is past the room's `base_children` allowance, and one
    child can be both past the allowance and inside the policy band — two rules
    each billing the same child is the obvious way to get this wrong. The larger
    of the two charges wins and the other is dropped; a child is billed once.

    `fold_into_price` is a legacy of a split that no longer exists: the two
    public allocators used to differ, with `allocate_rooms_by_requirements`
    folding the charge in as well as recording it. Both keep it beside
    `price_per_night` now (2026-09-23), and both callers pass `False`.

    The flag is kept rather than removed because folding is what the *reader*
    assumes against: `calculate_price_breakdown` prices a room as
    `price_per_night + extra_adult_charge + extra_child_charge`, so a charge
    that is folded in *and* recorded is a charge the guest pays twice on any
    night the calendar does not resolve. Passing `True` does not raise — it
    silently overcharges.

    Mutates the options in place. Every total downstream — card, quote, booking
    — sums `extra_child_charge` across a set's rooms rather than recomputing it,
    so the charge has to live on a room for them all to see it.
    """
    if paid_children <= 0 or not options:
        return

    by_id = {r.room_id: r for r in room_pool if r is not None}

    for opt in options:
        rooms = getattr(opt, "rooms", None) or []
        if not rooms:
            continue

        rate = 0.0
        for sel in rooms:
            src = by_id.get(sel.room_id)
            if src is not None:
                rate = float(getattr(src.occupancy, "extra_child_rate", 0.0) or 0.0)
                if rate > 0:
                    break
        if rate <= 0:
            continue

        policy_charge = paid_children * rate
        already_charged = sum(float(s.extra_child_charge or 0.0) for s in rooms)
        if policy_charge <= already_charged:
            continue

        delta = policy_charge - already_charged

        # Put it on the first room the guest is actually paying for: a set can
        # hold an FOC room at Rs 0, and printing a child charge beside a free
        # room reads as a mistake.
        target = next(
            (s for s in rooms if float(s.price_per_night or 0) > 0), rooms[0]
        )
        if fold_into_price:
            target.price_per_night = float(target.price_per_night or 0.0) + delta
        else:
            target.extra_child_charge = float(target.extra_child_charge or 0.0) + delta

        opt.extra_child_count = int(getattr(opt, "extra_child_count", 0) or 0) + paid_children
        opt.total_price_per_night = float(opt.total_price_per_night or 0.0) + delta

    return None


def _nightly_price(room: RoomType, meal_pref: str | None) -> float:
    """The cheapest the room can be had for, for ranking candidates.

    Used by the per-room occupancy branch to pick the cheapest room that seats
    the guests the line named. A room with no rate plan prices at 0.0 rather
    than raising — it would otherwise sort first and be chosen for every line,
    which is the wrong failure for a pricing question. There is no plan to
    quote, and the caller's quote builder is where that has to be caught.
    """
    plan = get_preferred_rate_plan(room, meal_pref)
    return float(getattr(plan, "base_price", 0.0) or 0.0)


def allocate_rooms_by_requirements(
    rooms: List[RoomType],
    requirements: List[dict],
    meal_pref: str | None = None,
    adults: int = 0,
    children: int = 0,
    paid_children: int = 0,
) -> List[AllocationOption]:
    """
    Allocate rooms based on explicit multi-category requirements (e.g. 10 Double, 3 Triple, 1 FOC,
    or '1 deluxe room king bed with extra bed and 1 standard non ac room').
    """
    from app.services.room_alias_resolver import (
        filter_rooms_by_reference,
        rooms_matching_bed_terms,
    )

    selected: List[SelectedRoom] = []
    total_base_adults = 0
    total_max_adults = 0
    room_objects: List[RoomType] = []
    # Parallel to `selected`: did the requirement line that produced this room
    # ask for a bed? Kept per room rather than as one flag for the request,
    # because a request can mix lines — "2 deluxe, put an extra bed in one" —
    # and a single flag put the bed (and its charge) on every room.
    selected_wants_bed: List[bool] = []

    import re

    for req in requirements:
        raw_type = str(req.get("room_type") or "").strip()
        has_extra_in_name = bool(re.search(r'\b(?:extra\s*bed|extra\s*gadda|with\s*extra)\b', raw_type.lower()))
        clean_type = re.sub(r'\b(?:i\s+want|need|give\s+me|with\s+extra\s+bed|with\s+extra|extra\s+bed|please|book|chahiye)\b', '', raw_type, flags=re.IGNORECASE).strip()
        req_type = clean_type if clean_type else raw_type
        req_type_lower = req_type.lower()
        count = int(req.get("count") or 1)
        is_foc = bool(req.get("is_foc", False)) or ("foc" in req_type_lower or "free" in req_type_lower)
        with_extra_bed = bool(req.get("with_extra_bed", False)) or has_extra_in_name

        # 0. A per-room occupancy line, read by the extraction as `guests`.
        #
        # The guest said how many people go in this room, not what to call it —
        # "room 1 2 people, room 2 4 people" — so `room_type` is empty and every
        # branch below would match nothing, discard the whole request at the
        # `return []`, and answer with the generic card. This is the branch that
        # gives that sentence a meaning: the cheapest room that seats them,
        # inside the class the line named.
        #
        # "Seats" is `adult_seats()` — the adult ceiling. `max_occupancy` counts
        # adults *plus* children, and using it here would seat a party of four
        # in a room that holds three adults and call it a fit.
        #
        # AC comes from the line when the guest named a class per room, and from
        # the room type otherwise. "NO_PREFERENCE" and None both mean do not
        # filter — the shared matcher, not an `in` test on the string, because
        # "without ac" contains "ac".
        guests = int(req.get("guests") or 0)
        matched_room = None
        if guests > 0:
            line_pool = rooms
            ac_class = req.get("ac") or ac_preference_from_text(req_type)
            if ac_class == "NON_AC":
                line_pool = [r for r in rooms if room_matches_feature(r, "Non-AC")]
            elif ac_class == "AC":
                line_pool = [r for r in rooms if room_matches_feature(r, "AC")]
            seating = [r for r in line_pool if r.adult_seats() >= guests]
            if not seating:
                # Nothing in the catalogue seats this line. Same answer as an
                # unresolvable room type below, for the same reason: let the
                # caller's catalogue allocators seat the party rather than hand
                # back a selection that answers a different question.
                return []
            matched_room = min(seating, key=lambda r: _nightly_price(r, meal_pref))

        # 1. Exact alias/name matching first
        if not matched_room and req_type:
            matched_list = filter_rooms_by_reference(req_type, rooms)
            if matched_list:
                matched_room = matched_list[0]

        # 2. Category keyword fallbacks
        if not matched_room:
            if "triple" in req_type_lower or "tripple" in req_type_lower or "3 bed" in req_type_lower:
                matched_room = next((r for r in rooms if "tripple" in r.name.lower() or "triple" in r.name.lower()), None)
                if not matched_room:
                    matched_room = next((r for r in rooms if "double" in r.name.lower() or "king" in r.name.lower() or "deluxe" in r.name.lower() or (getattr(r, "occupancy", None) and r.occupancy.max_adults >= 3)), None)
                    if matched_room:
                        with_extra_bed = True
                        guests = max(guests, 3)
            elif "family" in req_type_lower or "four" in req_type_lower or "4 bed" in req_type_lower:
                matched_room = next((r for r in rooms if "family" in r.name.lower() or "four" in r.name.lower()), None)
            elif "king" in req_type_lower:
                matched_room = next((r for r in rooms if "king" in r.name.lower()), None)
            elif "deluxe" in req_type_lower:
                matched_room = next((r for r in rooms if "deluxe" in r.name.lower()), None)
            elif "non ac" in req_type_lower or "non-ac" in req_type_lower:
                matched_room = next((r for r in rooms if "non ac" in r.name.lower() or "non-ac" in r.name.lower()), None)
            elif "single" in req_type_lower:
                matched_room = next((r for r in rooms if "single" in r.name.lower()), None)
            elif "double" in req_type_lower or "twin" in req_type_lower:
                matched_room = next((r for r in rooms if "double" in r.name.lower() or "twin" in r.name.lower()), None)

        if not matched_room:
            # Last resort before declaring the line unresolvable. Every branch
            # above searches room NAMES, and a line that names a bed ("double
            # bed", "four bed") names no room — so a non-AC double bed request
            # matched nothing here and the `return []` below discarded the whole
            # request, including its other, perfectly good lines (2026-09-21).
            # Deliberately last: a name match is the stronger signal, and this
            # only ever fills a `None` that would otherwise give up.
            #
            # Scoped to the AC class the line named, for the same reason
            # `room_service` scopes its pool — the matcher is only as good as the
            # pool it is handed. Over the whole catalogue "double bed" returns
            # Sample Hotel's two NON-AC rooms, the only prose that says "double",
            # so "ac double bed" was answered with STANDARD NON AC ROOM: a
            # non-AC room quoted to a guest who asked for AC, silently, which is
            # worse than the `return []` it replaced — that at least fell through
            # to the catalogue allocators. Scoped to the AC rooms, tier 1 finds
            # no prose and tier 2 reaches their King beds, which is the room that
            # guest actually wanted.
            #
            # The class is read through the shared matcher rather than by looking
            # for "ac" in the string: "without ac" contains "ac", and reading it
            # literally puts an AC guest in a non-AC room.
            bed_pool = rooms
            ac_class = req.get("ac") or ac_preference_from_text(req_type)
            if ac_class == "NON_AC":
                bed_pool = [r for r in rooms if room_matches_feature(r, "Non-AC")]
            elif ac_class == "AC":
                bed_pool = [r for r in rooms if room_matches_feature(r, "AC")]
            bed_matches = rooms_matching_bed_terms(req_type, bed_pool)
            if bed_matches:
                matched_room = bed_matches[0]

        if not matched_room:
            # A requirement naming no room the catalogue has is not a room
            # request — it is a parse that went wrong, and nothing here can tell
            # the two apart. This used to fall back to `rooms[0]`, which is how a
            # bisected check-in date and "3 kashi classic Room in 4th floor"
            # became 29 x whichever room sorts first, at Rs 58,000/night, for a
            # party of twelve (2026-09-21).
            #
            # Returning nothing sends the caller on to the allocators below it
            # (`room_service.py:59`), which seat the party from the catalogue
            # rather than from the requirement. Same `return []` this function
            # already uses for a selection that does not hold the party, and for
            # the same reason: a selection that answers a different question must
            # not be handed back as an answer to this one.
            return []

        # How many adults this line's room is recorded as seating.
        #
        # For a named room that is its base occupancy, as it has always been.
        # For an occupancy line it is the `guests` the guest stated — and it has
        # to be, because `validator` totals `assigned_adults` over the selection
        # and wipes the whole thing when the total falls short of the party.
        # `SelectedRoom` carries no `max_adults`, so the validator falls back to
        # this exact number. Record a King (base 2) as 2 when the guest asked it
        # to hold 3, and the guest sees the quote once and then the generic card
        # comes back.
        line_seats = guests if guests > 0 else matched_room.occupancy.base_adults

        rate_plan = get_preferred_rate_plan(matched_room, meal_pref)
        base_price = rate_plan.base_price if rate_plan else 0.0
        raw_price = getattr(rate_plan, "raw_base_price", None) or base_price
        price_pn = 0.0 if is_foc else base_price
        raw_price_pn = 0.0 if is_foc else raw_price

        plan_name = rate_plan.name if rate_plan else "Room Only"
        if is_foc:
            plan_name = f"{plan_name} (FOC)"

        # The catalogue's own ceilings, carried onto the selection so `validator`
        # checks the set against the room rather than against `line_seats`.
        #
        # `line_seats` above is what this line *records as seated*, and for a
        # named room that is the base occupancy — so two Deluxe King Beds (base
        # 2, max 3, occupancy 4) came out of here totalling 4 seats against a
        # party of six. `validator` reads `max_adults` and `max_occupancy` off
        # each selected room first and only falls back to `assigned_adults` when
        # they are absent (`validator.py:282-283`), so filling them is the whole
        # fix: the room seats three and holds four, and the validator now says so.
        #
        # `or None`, not `or 0`: a document that never states `max_occupancy`
        # must keep the old fallback rather than declare a room that holds
        # nobody — the same reasoning as `RoomCandidate.head_capacity`.
        if get_settings().SELECTION_RECORDS_ROOM_CEILINGS:
            ceilings = {
                "max_adults": matched_room.adult_seats(),
                "max_occupancy": int(
                    getattr(matched_room.occupancy, "max_occupancy", 0) or 0
                )
                or None,
            }
        else:
            ceilings = {}

        for _ in range(count):
            selected.append(SelectedRoom(
                room_id=matched_room.room_id,
                room_name=matched_room.name,
                rate_plan_id=rate_plan.rate_plan_id if rate_plan else "",
                rate_plan_name=plan_name,
                # Carried, not re-derived: the plan's own identifier is what
                # tells EP from CP further down, and `plan_name` here has "(FOC)"
                # appended for a free room.
                rate_type_id=getattr(rate_plan, "rate_type_id", None),
                price_per_night=price_pn,
                raw_price_per_night=raw_price_pn,
                assigned_adults=line_seats,
                assigned_children=0,
                extra_adult_charge=0.0,
                extra_child_charge=0.0,
                bed_type=matched_room.beds[0].get("type") if matched_room.beds and isinstance(matched_room.beds[0], dict) else None,
                **ceilings,
            ))
            room_objects.append(matched_room)
            selected_wants_bed.append(with_extra_bed)
            # `line_seats`, so the option's own `base_adults` is the seat count
            # the guest asked for. Left at the rooms' base occupancy, an
            # occupancy line would come out with `max_adults > base_adults` and
            # `validator` would ask the guest for a mattress on a set that
            # already seats them.
            total_base_adults += line_seats
            # `max_adults`, not `max_occupancy`. This is the option's
            # `max_adults` field, and it was the only place in the module that
            # filled it with a total-bodies number — so a caller comparing
            # `option.max_adults` against a party of adults got a figure that
            # counted children, while the catalogue path at `_create_option`
            # filled the same field with `sum(max_adults)`. Two meanings for one
            # field; this is the one that was wrong.
            total_max_adults += matched_room.occupancy.max_adults

    if not selected:
        return []

    # 3. Extra beds — only where the guest asked for one.
    #
    # ONE fact decides this now, and it used to be two. The other was "Fact 1 —
    # the party's own surplus: adults beyond the adults the selection already
    # seats", counted against `base_adults`. That is the same wrong idea the
    # catalogue path carried (see `distribute_guests`), in its second copy:
    # adults between `base_adults` and `max_adults` are guests the hotel already
    # counts, not mattress berths. A party of three who asked for one Deluxe
    # King Bed was charged Rs 600 for a third adult the hotel seats, and the
    # room was recorded as sleeping three only because of the mattress.
    #
    # A party that genuinely does not fit the selection is caught below, by the
    # feasibility check against `max_occupancy` — which is what that check is
    # for, and it returns the caller to the allocators rather than stretching a
    # room with a mattress it cannot legally sell as a berth.
    #
    # The surviving fact — `with_extra_bed` — is set by the extraction when a
    # requirement line talks about beds ("put an extra bed in one of the deluxe
    # rooms"). It is read per LINE, and that is the part that matters: 46871e3
    # removed a global version of this trigger because it put a Rs 600 bed and
    # its charge on every room of a request that mentioned a bed once. A line
    # that asks for a bed still gets one on that line only.
    #
    # The bed still has to clear the hotel's ceiling, below.
    extra_adults_count = 0
    party_size = max(0, int(adults or 0)) + max(0, int(children or 0))

    # Adults this selection does not seat — the people a CP / MAP / AP plan owes
    # for. `total_base_adults` is the sum of what each line records as seated,
    # which for a named room is its base occupancy: "1 deluxe room" for a party
    # of three leaves one adult unseated, and on those plans one charge owed.
    adult_surplus = max(0, int(adults or 0) - total_base_adults)
    charged_above_base = 0

    for i, sel_room in enumerate(selected):
        orig_room = room_objects[i] if i < len(room_objects) else None

        # "Can take a mattress" is the hotel saying so, in the hotel's own
        # units: every room whose `max_adults` sits above its `base_adults`.
        #
        # This used to compare `max_occupancy` against the base *headcount*,
        # which at hotel 111111 is the same number for every room — a Deluxe is
        # 2A+2C=4 against a ceiling of 4 — so `4 > 4` was false and a bed could
        # not be added anywhere in the hotel. The guard was not wrong in
        # principle; it read the wrong pair of fields.
        #
        # The field pair is right and the meaning has changed: this no longer
        # says the room *needs* a mattress to seat its party (it does not — see
        # `distribute_guests`), only that the hotel offers one.
        #
        # Checked against all six hotels on 2026-09-22: every Sahu room offers a
        # bed (base 2 / max 3, 4 / 5, 3 / 4), every Sample Property room (base 2 /
        # max 2) offers none. No hotel needs a special case.
        wants_bed = selected_wants_bed[i] if i < len(selected_wants_bed) else False
        takes_bed = bool(
            orig_room
            and orig_room.occupancy.max_adults > orig_room.occupancy.base_adults
        )
        if takes_bed and wants_bed:
            extra_rate = orig_room.occupancy.extra_adult_rate or 600.0
            # Recorded beside `price_per_night`, never folded into it. This used
            # to do both — `price_per_night += extra_rate` as well — and that is
            # how the bed got charged twice (2026-09-23).
            #
            # `calculate_price_breakdown` prices a room as
            # `price_per_night + extra_adult_charge + extra_child_charge`
            # (booking_engine.py:163). It replaces `price_per_night` with the
            # calendar's own rate when the night resolves (lines 137-145), which
            # discarded the folded bed and hid the fault. On a night the calendar
            # does NOT resolve — any date the rate calendar does not cover — the
            # folded value survived and the bed was added a second time.
            #
            # Measured on hotel 111111, Deluxe Room King Bed, 3 adults, one room,
            # extra bed asked for: Rs 3,150/night inside calendar coverage and
            # Rs 3,750/night outside it, where 3,150 is the right answer in both.
            # The coverage there runs 2026-09-11 .. 2027-09-10, so the fault was
            # reachable for every date past it.
            #
            # "Beside" is also what every consumer already assumes — the sibling
            # `allocate_rooms` below, `room_service._per_room_price_line`, and the
            # no-breakdown fallbacks at `room_service.py:989/1102/1107`, which all
            # add `extra_adult_charge` to `price_per_night` themselves. The fold
            # made this one path the exception they had to guess about.
            sel_room.extra_adult_charge = extra_rate
            # A mattress really is on this room: the quote may print it, and
            # `validator._selection_carries_a_bed` may suppress the question.
            sel_room.extra_bed_count = 1
            extra_adults_count += 1
            charged_above_base += 1
        elif (
            orig_room
            and charged_above_base < adult_surplus
            and plan_owes_for_an_extra_person(sel_room.rate_type_id, sel_room.rate_plan_name)
        ):
            # The *other* half of the rule, and the half that was missing.
            #
            # Product, 2026-10-06: on EP an extra guest may stay in the room with
            # no bed and no charge, but "that SHOULD NOT be allowed on MAP and
            # CP". The EP half was already true by accident — nothing is charged
            # unless `wants_bed` above — but the CP half was true for the wrong
            # reason, so a CP guest who said "no extra bed" was charged nothing
            # for the adult eating breakfast in that room.
            #
            # The price is the room type's own rate, never the plan's: "Extra
            # person rate is for each room type and not for each plan". Only the
            # permission to seat them free is the plan's.
            #
            # `extra_bed_count = 0` matters as much as the money. Left unset it
            # would read as "nobody said", fall back to the charge itself
            # (`SelectedRoom.extra_bed_count`), and print "Extra bed" for a bed
            # this guest was never offered — the exact confusion the field was
            # added to end.
            sel_room.extra_adult_charge = orig_room.occupancy.extra_adult_rate or 600.0
            sel_room.extra_bed_count = 0
            extra_adults_count += 1
            charged_above_base += 1

    # Feasibility. If the party does not fit the selection's ceiling, this
    # selection is not an answer to the requirement and must not be returned as
    # one.
    #
    # Without this, removing the phantom bed made things *worse* rather than
    # better: the guard above had been quietly covering for a selection that
    # never held the party, so a request for one room and a party of five came
    # back as one room, two adults, no bed and no second room — and the caller
    # at `room_service.py:54` returns whatever it is handed. Returning nothing
    # instead lets that caller fall through to the multi-room allocator, which
    # is the path that can actually seat the party.
    #
    # Under correct inventory this is close to inert: a room that can take an
    # extra bed has a ceiling above its base, so the bed lifts it and the party
    # fits. It bites on the current data, where `max_occupancy` equals the base
    # headcount almost everywhere and no room can be stretched — which is the
    # data product is correcting.
    total_capacity = sum(r.occupancy.max_occupancy for r in room_objects)
    # Two ceilings, not one. `max_occupancy` counts adults **plus** children, so
    # on its own it says a room holding three adults and four people can take a
    # fourth adult — which the catalogue does not say. The Premier Room King Bed
    # at hotel Sahu is exactly that shape (base 2 / max_adults 3 / max_occupancy
    # 4), so "1 premier room" for four adults came back as one room seating
    # three: the card drew "1 x Premier Room — max 3" under a header of four
    # guests, `validator` saw the shortfall on the guest's next turn and wiped
    # the selection, the same requirement was re-allocated, and the guest got
    # the same card back — live loop, 2026-11-05. Refusing here lets the caller
    # fall through to the catalogue allocator, which answers with two rooms.
    #
    # `adult_seats()`, not `occupancy.max_adults`: the schema defaults that
    # field to 3, and a room stating a base of 4 must not be capped below what
    # it advertises. Same reader the `guests` line above uses.
    adult_capacity = sum(r.adult_seats() for r in room_objects)
    if party_size > total_capacity or int(adults or 0) > adult_capacity:
        # Silent until 2026-09-22, and that was the wrong way round: this
        # `return []` is the reason a guest whose request named rooms that
        # cannot hold them gets a generic card instead of their own answer, and
        # nothing in the logs said so. The caller falls through by design
        # (room_service.py:59); the fall-through is not the bug, its silence is.
        logger.info(
            "requirements: party of %s adults / %s total does not fit the "
            "selection (%s rooms, %s adult seats, capacity %s) — falling "
            "through to the catalogue allocators",
            adults,
            party_size,
            len(room_objects),
            adult_capacity,
            total_capacity,
        )
        return []

    # "Beside", like every other reader of these three fields: the bed and the
    # child charge live on their own fields and are added here. The sum is
    # unchanged from when the bed was folded in — that is the point. Folding it
    # is not a different way to reach the same total, it is a second copy of the
    # charge that survives every reader which adds the fields itself.
    total_price = sum(
        (r.price_per_night or 0.0) + (r.extra_adult_charge or 0.0) + (r.extra_child_charge or 0.0)
        for r in selected
    )

    option = AllocationOption(
        option_id="opt_1",
        rooms=selected,
        total_price_per_night=total_price,
        recommendation_labels=["Recommended", "Custom Combination"],
        base_adults=total_base_adults,
        max_adults=total_max_adults,
        extra_adult_count=extra_adults_count,
        extra_child_count=0,
    )

    # Children are given no berth on this path — every line is the guest's own
    # room request, and `assigned_children` stays 0 throughout — but the hotel's
    # policy still makes some of them chargeable, and until now that produced no
    # line and no money at all.
    # `False`, to match the bed above and every other reader of these fields.
    # This was `True`, which folded the child charge into `price_per_night` as
    # well as recording it — so a room with both a bed and a chargeable child
    # carried two folded charges, and `calculate_price_breakdown` added both
    # fields back on top of them on any night the calendar did not resolve. The
    # same double-count as the bed, reached by the same route.
    #
    # The option total is unaffected: `_apply_paid_child_charge` adds its delta
    # to `total_price_per_night` whichever convention is used, so the guest pays
    # the same. What changes is only whether the charge is *also* inside
    # `price_per_night`, which is the copy that gets counted twice.
    _apply_paid_child_charge([option], room_objects, paid_children, fold_into_price=False)

    return [option]


def distribute_guests(
    combination: List[RoomType],
    total_adults: int,
    total_children: int,
    meal_pref: str | None = None,
    wants_bed: bool = False,
) -> List[SelectedRoom] | None:
    """
    Seat a party across a combination of rooms, and price the mattress if asked.

    Returns a list of SelectedRoom objects if successful, or None if impossible.
    Every room must have at least 1 adult!

    **A mattress is not capacity.** `max_adults` is the hotel saying how many
    adults this room holds, and it holds that many with or without one — the
    field is literally `max_adult_occupancy`, the adult ceiling. `base_adults` is
    how many the room rate covers by default; the adults between `base_adults`
    and `max_adults` are ordinary guests the hotel already counts, not mattress
    berths.

    Until 2026-09-25 this function read the two fields the other way round: it
    filled to `base_adults`, then "extra" adults up to `max_adults` were seated
    by adding a mattress and charging for it. A party of three in a Deluxe King
    (base 2 / max 3) was therefore billed ₹600 for a third adult the hotel's own
    document says it seats. Worse, it taught every reader downstream that the
    third adult *was* the mattress — `room_service._card_candidates` derives the
    bed on a card row from `assigned_adults - base_adults`, so the row advertised
    "Sleeps 2" for a room that sleeps 3. A guest who declined the mattress then
    appeared not to fit, and the packer answered a party of three with two rooms
    ₹2,250 more expensive. That is the defect reported from hotel Sahu.

    So: adults fill to `max_adults` and stop. Nothing seats an adult above it,
    because the mattress never did. `wants_bed` is the guest asking for one
    outright — it prices the mattress and records it, and it changes no
    assignment, because a mattress moves nobody into the room who was not
    already in it.

    Children never take a mattress: the hotel's own policy says `kids_bed: "No"`,
    and a child above `base_children` borrows the room's unused adult headroom
    instead (see Pass 2).
    """
    if total_adults < len(combination):
        return None # Not enough adults to have 1 per room
        
    n_rooms = len(combination)
    adults_assigned = [1] * n_rooms
    children_assigned = [0] * n_rooms
    
    remaining_adults = total_adults - n_rooms
    remaining_children = total_children
    
    # Quick check: ensure initial assignment (1 adult per room) doesn't exceed any room's max_occupancy
    for i, room in enumerate(combination):
        if adults_assigned[i] + children_assigned[i] > room.occupancy.max_occupancy:
            return None
            
    # Pass 1: Fill up to the adult ceiling, then to base children.
    #
    # `max_adults`, not `base_adults` — see the docstring. The adults between the
    # two are guests the hotel counts, and seating them has never required a
    # mattress. Filling only to base is what made one load-bearing.
    for i, room in enumerate(combination):
        # Adults
        space_base_a = max(0, room.occupancy.max_adults - adults_assigned[i])
        add_a = min(space_base_a, remaining_adults)
        # Check max_occupancy constraint
        current_total = adults_assigned[i] + children_assigned[i]
        allowed_by_occ = max(0, room.occupancy.max_occupancy - current_total)
        add_a = min(add_a, allowed_by_occ)
        
        adults_assigned[i] += add_a
        remaining_adults -= add_a
        
        # Children
        space_base_c = max(0, room.occupancy.base_children - children_assigned[i])
        add_c = min(space_base_c, remaining_children)
        # Check max_occupancy constraint
        current_total = adults_assigned[i] + children_assigned[i]
        allowed_by_occ = max(0, room.occupancy.max_occupancy - current_total)
        add_c = min(add_c, allowed_by_occ)
        
        children_assigned[i] += add_c
        remaining_children -= add_c
        
    # Pass 2: children, into whatever room is left.
    #
    # There is no adult branch any more. There used to be — adults above
    # `base_adults` were seated here against `max_extra_beds` and charged for it,
    # which is what made the mattress load-bearing. Adults now fill to
    # `max_adults` in Pass 1 and stop, so by the time this loop runs
    # `remaining_adults` is already 0 for every combination that can seat the
    # party, and `_create_option` has already rejected the ones that cannot.
    for i, room in enumerate(combination):
        # Children can use remaining max_occupancy, substituting for missing adults.
        unused_adult_slots = max(0, room.occupancy.max_adults - adults_assigned[i])
        effective_max_children = room.occupancy.max_children + unused_adult_slots
        space_extra_c = max(0, effective_max_children - children_assigned[i])
        add_c = min(space_extra_c, remaining_children)
        # Check max_occupancy constraint (allow child sharing bed if room seats 2+ adults)
        effective_room_occ = max(room.occupancy.max_occupancy, room.occupancy.max_adults + (1 if room.occupancy.max_adults >= 2 else 0))
        current_total = adults_assigned[i] + children_assigned[i]
        allowed_by_occ = max(0, effective_room_occ - current_total)
        add_c = min(add_c, allowed_by_occ)
        
        children_assigned[i] += add_c
        remaining_children -= add_c
        
    if remaining_adults > 0 or remaining_children > 0:
        return None
        
    selected_rooms = []
    for i, room in enumerate(combination):
        rate_plan = get_preferred_rate_plan(room, meal_pref)
        if not rate_plan:
            return None
            
        a_assigned = adults_assigned[i]
        c_assigned = children_assigned[i]
        
        # Double check final occupancy check
        effective_room_occ = max(room.occupancy.max_occupancy, room.occupancy.max_adults + (1 if room.occupancy.max_adults >= 2 else 0))
        if a_assigned + c_assigned > effective_room_occ:
            return None
            
        # The mattress, charged only when the guest asked for one, and only where
        # the hotel permits one — `extra_bed_allowed` / `max_extra_beds` are the
        # policy fields, and a hotel that allows no mattress must not be sold one.
        #
        # Never `a_assigned - base_adults`: those are ordinary guests the hotel
        # counts, not mattresses. That expression is the defect — see the
        # docstring. The `or 600.0` matches the requirements path at
        # `allocate_rooms_by_requirements`, so the same room priced by either
        # path carries the same figure; a genuine zero there is a rate nobody
        # set, and `room_repository._room_extra_bed_rate` never returns one.
        bed_allowed = bool(getattr(room.occupancy, "extra_bed_allowed", True)) and int(
            getattr(room.occupancy, "max_extra_beds", 1) or 0
        ) > 0
        # A mattress, or an adult the plan does not seat. Same money, different
        # sentence — so they are decided together and recorded apart.
        extra_bed = bool(wants_bed and bed_allowed)
        extra_person = (
            a_assigned > room.occupancy.base_adults
            and plan_owes_for_an_extra_person(
                getattr(rate_plan, "rate_type_id", None), rate_plan.name
            )
        )
        extra_a_charge = (
            float(room.occupancy.extra_adult_rate or 600.0)
            if (extra_bed or extra_person)
            else 0.0
        )
        # `None`, not 0, when neither applies: it means "nothing was asked", and
        # leaves the old charge-based fallback in `SelectedRoom.extra_bed_count`.
        extra_bed_count = 1 if extra_bed else (0 if extra_person else None)
        extra_c_charge = max(0, c_assigned - room.occupancy.base_children) * room.occupancy.extra_child_rate

        selected_rooms.append(
            SelectedRoom(
                room_id=room.room_id,
                room_name=room.name,
                aliases=room.aliases,
                rate_plan_id=rate_plan.rate_plan_id,
                rate_plan_name=rate_plan.name,
                rate_type_id=getattr(rate_plan, "rate_type_id", None),
                price_per_night=rate_plan.base_price,
                raw_price_per_night=getattr(rate_plan, "raw_base_price", None) or rate_plan.base_price,
                currency=rate_plan.currency,
                assigned_adults=a_assigned,
                assigned_children=c_assigned,
                extra_adult_charge=extra_a_charge,
                extra_child_charge=extra_c_charge,
                extra_bed_count=extra_bed_count,
            )
        )
        
    return selected_rooms


def allocate_rooms(
    rooms: List[RoomType], 
    total_adults: int, 
    total_children: int,
    requested_num_rooms: int | None = None,
    max_depth: int = 4,
    budget: float | None = None,
    meal_pref: str | None = None,
    feature_preferences: list[str] | None = None,
    occupancy_preference: str | None = None,
    grouping_mode: str | None = None,
    paid_children: int = 0,
    wants_bed: bool = False,
    max_adults_per_room: int | None = None,
) -> List[AllocationOption]:
    """
    Find valid combinations of rooms to hold the guests.
    Returns ranked AllocationOptions.

    `wants_bed` prices a mattress on every room in the option and changes no
    assignment — a mattress is comfort, not capacity, and it seats nobody the
    rooms were not already seating. See `distribute_guests`.
    """
    valid_options: List[AllocationOption] = []
    
    # Filter out rooms with no active rate plans or 0 inventory
    available_rooms = [r for r in rooms if r.number_of_rooms > 0 and get_preferred_rate_plan(r, meal_pref)]
    if not available_rooms:
        return []

    # Upfront feature filtering: only consider rooms matching requested features (AC, Non-AC, Balcony, etc.)
    if feature_preferences:
        # Drop the preferences nothing in this catalogue can satisfy, so one
        # unmatched word cannot empty the whole inventory. Rebinding the
        # parameter means the post-allocation filter below works from the same
        # list and cannot re-introduce the dead end (defect #10, Chat A
        # 06:36:55: guest asked for a "4 bed", hotel holds "DELUXE ROOM FOUR
        # BED", guest was told no rooms matched).
        feature_preferences = resolve_honourable_features(available_rooms, feature_preferences)
        matching_feat_rooms = [r for r in available_rooms if all(room_matches_feature(r, f) for f in feature_preferences)]
        if matching_feat_rooms:
            available_rooms = matching_feat_rooms
        elif any(is_hard_inventory_feature(f) for f in feature_preferences):
            # AC/Non-AC could not be met by any room. That is a real answer.
            return []

    # Fast check: is it physically possible with all available inventory?
    max_possible_adults = sum(r.occupancy.max_adults * r.number_of_rooms for r in available_rooms)
    max_possible_occupancy = sum(max(r.occupancy.max_occupancy, r.occupancy.max_adults + (1 if r.occupancy.max_adults >= 2 else 0)) * r.number_of_rooms for r in available_rooms)
    
    if max_possible_adults < total_adults or max_possible_occupancy < (total_adults + total_children):
        return []

    # Dynamic target search range based on true inventory greedy check
    total_guests = total_adults + total_children
    
    # Calculate true min rooms for max capacity
    def calc_true_min(capacity_attr_a, capacity_attr_c, capacity_attr_occ):
        rooms_sorted = []
        for r in available_rooms:
            rooms_sorted.extend([r] * r.number_of_rooms)
        
        # We need to satisfy adults, children, and total guests.
        # A simple greedy approach: sort by the sum of these capacities.
        rooms_sorted.sort(key=lambda r: getattr(r.occupancy, capacity_attr_a) + getattr(r.occupancy, capacity_attr_c), reverse=True)
        
        a, c, occ = 0, 0, 0
        count = 0
        for r in rooms_sorted:
            a += getattr(r.occupancy, capacity_attr_a)
            c += getattr(r.occupancy, capacity_attr_c)
            occ += getattr(r.occupancy, capacity_attr_occ)
            count += 1
            effective_c = c + max(0, a - count)
            effective_occ = max(occ, a + min(total_children, count))
            if a >= total_adults and effective_c >= total_children and effective_occ >= total_guests and count <= total_adults:
                return count
        return count if (a >= total_adults and occ >= total_guests and count <= total_adults) else float('inf')

    min_rooms_needed = calc_true_min('max_adults', 'max_children', 'max_occupancy')
    min_rooms_base = calc_true_min('base_adults', 'base_children', 'max_occupancy')
    
    total_inventory = sum(r.number_of_rooms for r in available_rooms)
    
    if min_rooms_needed == float('inf'):
        return []
        
    min_single_base = min((r.occupancy.base_adults for r in available_rooms if r.occupancy.base_adults > 0), default=2)
    # T6 capped self-serve at four rooms so that a bigger party became a handoff
    # card. `GROUP_BOOKING_SELF_SERVE` lifts that ceiling: the party is searched
    # and priced like any other. What still bounds the work is the search's own
    # 2-second budget and 80-option cap, not a room count.
    self_serve_groups = get_settings().GROUP_BOOKING_SELF_SERVE
    MAX_SELF_SERVE_ROOMS = 4 if not self_serve_groups else max(total_inventory, 1)
    if requested_num_rooms is not None and requested_num_rooms >= min_rooms_needed:
        search_start = requested_num_rooms
        search_end = min(requested_num_rooms, total_inventory)
    elif requested_num_rooms is not None:
        search_start = min_rooms_needed
        search_end = max(requested_num_rooms, min(min_rooms_needed + 1, total_inventory))
    else:
        # Default reasonable allocation: at most 1 room per 2 adults, capped at 4 rooms max for self-serve
        max_reasonable_rooms = min(MAX_SELF_SERVE_ROOMS, max(min_rooms_needed, (total_guests + 1) // 2))
        search_start = min_rooms_needed
        search_end = min(max_reasonable_rooms, total_inventory)
        search_end = max(search_start, search_end)

    import time
    start_time = time.time()
    seen_combos = set()

    #: Set only by the last-resort pass below, and only for the minimum
    #: single-room-type combination that seats the party. Keeps T6's rule intact
    #: everywhere else.
    _oversize_rescue = [False]

    def _combo_key(combo):
        return tuple(sorted(r.room_id for r in combo))

    def _create_option(combo) -> AllocationOption | None:
        base_a = sum(r.occupancy.base_adults for r in combo)
        base_c = sum(r.occupancy.base_children for r in combo)
        max_a = sum(r.occupancy.max_adults for r in combo)
        max_c = sum(r.occupancy.max_children for r in combo)
        max_occ = sum(r.occupancy.max_occupancy for r in combo)
        
        # T6: Stop engine from generating 5+ room options for self-serve.
        # `_oversize_rescue` is the one exception, and only for the minimum
        # combination that seats the party — see the last-resort pass below.
        if (
            not self_serve_groups
            and requested_num_rooms is None
            and len(combo) >= 5
            and not _oversize_rescue[0]
        ):
            return None

        effective_max_c = max_c + max(0, max_a - len(combo))
        effective_max_occ = max(max_occ, max_a + min(total_children, len(combo)))
        if max_a >= total_adults and effective_max_c >= total_children and effective_max_occ >= total_guests and len(combo) <= total_adults:
            selected = distribute_guests(combo, total_adults, total_children, meal_pref, wants_bed=wants_bed)
            if selected:
                if max_adults_per_room is not None and any(s.assigned_adults > max_adults_per_room for s in selected):
                    return None
                total_price = sum(s.price_per_night + s.extra_adult_charge + s.extra_child_charge for s in selected)
                meal_upgrades = []
                current_plans = {s.rate_plan_name for s in selected}
                for alt_pref in ["EP", "CP", "MAP", "AP"]:
                    alt_selected = distribute_guests(combo, total_adults, total_children, alt_pref, wants_bed=wants_bed)
                    if alt_selected:
                        alt_price = sum(s.price_per_night + s.extra_adult_charge + s.extra_child_charge for s in alt_selected)
                        alt_plans = {s.rate_plan_name for s in alt_selected}
                        if alt_plans != current_plans:
                            from app.services.room_service import _friendly_rate_plan_name
                            friendly = _friendly_rate_plan_name(alt_pref)
                            if not any(u["plan_name"] == friendly for u in meal_upgrades):
                                meal_upgrades.append({
                                    "plan_name": friendly,
                                    "total_price_per_night": alt_price,
                                    "diff": alt_price - total_price
                                })

                # `extra_adult_count` is the MATTRESS count, not the adults
                # above base. It is read as "this option carries a bed" by
                # `room_service.format_allocations` (which appends "+ Extra Bed"
                # to the room name), `extra_bed_note`, `recommendation_engine`
                # and the validator — so it has to count the thing they are all
                # talking about, which since 2026-09-25 is a priced mattress and
                # not a third adult the hotel was always going to seat.
                extra_adults = sum(1 for s in selected if float(s.extra_adult_charge or 0) > 0)
                extra_children = sum(max(0, s.assigned_children - r.occupancy.base_children) for s, r in zip(selected, combo))
                # `allocation_type` deliberately still reads the old quantity —
                # adults above base. It is a ranking signal in
                # `recommendation_engine` (which needs a COMFORT option and a
                # MAX_OCCUPANCY one to offer a diverse set), and it means "this
                # set pushes its rooms harder", which is true of a Deluxe seating
                # three whether or not anyone bought a mattress. Sourcing it from
                # the mattress would have silently emptied the COMFORT bucket.
                above_base = sum(max(0, s.assigned_adults - r.occupancy.base_adults) for s, r in zip(selected, combo))
                allocation_type = "MAX_OCCUPANCY" if (above_base > 0 or extra_children > 0) else "COMFORT"

                return AllocationOption(
                    option_id=f"opt_{len(valid_options)+1}",
                    rooms=selected,
                    total_price_per_night=total_price,
                    base_adults=base_a,
                    base_children=base_c,
                    max_adults=max_a,
                    max_children=max_c,
                    meal_upgrades=meal_upgrades,
                    allocation_type=allocation_type,
                    extra_adult_count=extra_adults,
                    extra_child_count=extra_children
                )
        return None

    # 1. Guarantee Single-Room-Type Options for each room type (Comfort & Max Occupancy)
    for r in available_rooms:
        if r.occupancy.max_adults > 0:
            need_max_a = (total_adults + r.occupancy.max_adults - 1) // r.occupancy.max_adults
            need_max_occ = (total_guests + r.occupancy.max_occupancy - 1) // r.occupancy.max_occupancy if r.occupancy.max_occupancy > 0 else need_max_a
            need_max_calc = max(need_max_a, need_max_occ)
            if requested_num_rooms is not None:
                need_max = requested_num_rooms if requested_num_rooms >= need_max_calc else need_max_calc
            else:
                need_max = need_max_calc
            if need_max <= r.number_of_rooms and (
                self_serve_groups or requested_num_rooms is not None or need_max < 5
            ):
                combo_max = [r] * need_max
                ckey = _combo_key(combo_max)
                if ckey not in seen_combos:
                    opt = _create_option(combo_max)
                    if opt:
                        valid_options.append(opt)
                        seen_combos.add(ckey)

        if r.occupancy.base_adults > 0:
            need_base_a = (total_adults + r.occupancy.base_adults - 1) // r.occupancy.base_adults
            base_occ = r.occupancy.base_adults + r.occupancy.base_children
            need_base_occ = (total_guests + base_occ - 1) // base_occ if base_occ > 0 else need_base_a
            need_base_calc = max(need_base_a, need_base_occ)
            if requested_num_rooms is not None:
                need_base = requested_num_rooms if requested_num_rooms >= need_base_calc else need_base_calc
            else:
                need_base = need_base_calc
            if need_base <= r.number_of_rooms and (
                self_serve_groups or requested_num_rooms is not None or need_base < 5
            ):
                combo_base = [r] * need_base
                ckey = _combo_key(combo_base)
                if ckey not in seen_combos:
                    opt = _create_option(combo_base)
                    if opt:
                        valid_options.append(opt)
                        seen_combos.add(ckey)

    # 2. General multi-room combination search
    def generate_combos(idx, current_depth, current_counts, target_depth):
        if time.time() - start_time > 2.0:
            return  # Timeout to prevent exponential blowup
            
        if idx == len(available_rooms):
            if current_depth == target_depth:
                yield current_counts
            return
        
        room = available_rooms[idx]
        max_take = min(room.number_of_rooms, target_depth - current_depth)
        
        for take in range(max_take + 1):
            yield from generate_combos(idx + 1, current_depth + take, current_counts + [(room, take)], target_depth)

    for depth in range(search_start, search_end + 1):
        for combo_counts in generate_combos(0, 0, [], depth):
            combo = []
            for r, count in combo_counts:
                combo.extend([r] * count)
                
            if not combo:
                continue
                
            ckey = _combo_key(combo)
            if ckey in seen_combos:
                continue
                
            opt = _create_option(combo)
            if opt:
                valid_options.append(opt)
                seen_combos.add(ckey)
                
            if len(valid_options) >= 80:
                break


    # ── T6 last resort: a party too big for four rooms ──
    # T6 caps self-serve at 4 rooms so a 5+ room booking goes to a human. That is
    # still the rule, but below 5 rooms the *minimum* combination is the only
    # thing keeping the search alive, and once `min_rooms_needed` reaches 5 the
    # search range collapses to a single point whose every combo is rejected by
    # the `len(combo) >= 5` guard — so `valid_options` came back empty and the
    # guest got a room card with nothing in it.
    #
    # Measured on 111111: 21 adults in a room that seats 3 needs 7 rooms, and the
    # search returned `[]`. It also degraded to one or two options at 17-20.
    #
    # This pass runs ONLY when the search produced nothing at all, and offers the
    # smallest single-room-type combination that seats the party — the same shape
    # as the "Guarantee Single-Room-Type Options" block above, just without the
    # 4-room ceiling. It cannot change any result the engine already produces, so
    # T6's handoff behaviour for every other party is untouched.
    if not valid_options and requested_num_rooms is None and min_rooms_needed >= 5:
        _oversize_rescue[0] = True
        try:
            for r in available_rooms:
                seats = [
                    (r.occupancy.max_adults, r.occupancy.max_occupancy),
                    (r.occupancy.base_adults, r.occupancy.base_adults + r.occupancy.base_children),
                ]
                for per_room_a, per_room_occ in seats:
                    if per_room_a <= 0 or per_room_occ <= 0:
                        continue
                    need = max(
                        (total_adults + per_room_a - 1) // per_room_a,
                        (total_guests + per_room_occ - 1) // per_room_occ,
                    )
                    if need < 5 or need > r.number_of_rooms:
                        continue
                    combo = [r] * need
                    ckey = _combo_key(combo)
                    if ckey in seen_combos:
                        continue
                    opt = _create_option(combo)
                    if opt:
                        valid_options.append(opt)
                        seen_combos.add(ckey)
                    break  # The widest seating for this room type is enough.
        finally:
            _oversize_rescue[0] = False
        if valid_options:
            logger.warning(
                f"OVERSIZE_RESCUE | party={total_adults}a+{total_children}c needed "
                f"{min_rooms_needed} rooms; T6 self-serve cap is 4 — returning "
                f"{len(valid_options)} minimal option(s) instead of an empty card"
            )

    # ── HARD FEATURE FILTERING ──
    # If the user requested specific features (e.g. "AC", "Balcony"), exclude non-matching rooms
    # completely instead of just ranking them lower.
    if feature_preferences:
        import re
        # (the matcher used to be duplicated here, byte for byte, which is
        # exactly how the two copies drifted apart — see room_matches_feature)

        def _option_satisfies_all_features(opt: AllocationOption) -> bool:
            for pref in feature_preferences:
                for selected_room in opt.rooms:
                    orig = next((r for r in rooms if r.room_id == selected_room.room_id), None)
                    if not room_matches_feature(orig, pref):
                        return False
            return True

        matching_options = [opt for opt in valid_options if _option_satisfies_all_features(opt)]
        if matching_options:
            logger.info(
                f"FEATURE_HARD_FILTER | requested={feature_preferences} | "
                f"retained {len(matching_options)} of {len(valid_options)} options"
            )
            filtered_options = matching_options
        else:
            logger.warning(
                f"FEATURE_HARD_FILTER_NO_MATCH | requested={feature_preferences} | "
                f"no options met all features; returning empty options"
            )
            filtered_options = []
    else:
        filtered_options = valid_options

    # Ranking logic:
    # 1. Occupancy fit tier:
    #    - Tier 0: Exact fit (base_adults == total_adults)
    #    - Tier 1: Extra-bed / triple occupancy (base_adults == total_adults - 1 with 1 extra bed)
    #    - Tier 2: Minor over-capacity (base_adults == total_adults + 1)
    #    - Tier 3+: Major over-capacity (base_adults >= total_adults + 2, e.g. 5-bed room for 3 adults)
    # 2. Budget compliance
    # 3. Requested room count match
    # 4. Total price per night (cheaper is better)
    # 5. Number of rooms
    def sort_key(opt: AllocationOption):
        within_budget = opt.total_price_per_night <= budget if budget else True
        matches_rooms = len(opt.rooms) == requested_num_rooms if requested_num_rooms else True
        
        # Calculate occupancy fit tier across the option's rooms
        total_base_a = opt.base_adults
        extra_a = opt.extra_adult_count
        
        delta = total_base_a - total_adults
        occ_pref_upper = (occupancy_preference or "").upper()
        grouping_upper = (grouping_mode or "").upper()
        if grouping_upper == "SEGMENTED":
            # Priority: multi-room combinations with distinct room types (e.g. Family + Double) or dedicated rooms
            unique_types = len(set(r.room_name for r in opt.rooms))
            has_mixed_or_multi = unique_types > 1 or len(opt.rooms) >= 2
            if has_mixed_or_multi and extra_a == 0:
                fit_tier = 0
            elif has_mixed_or_multi:
                fit_tier = 1 + extra_a
            else:
                fit_tier = 5 + extra_a
        elif occ_pref_upper == "BASE":
            # Priority: fit in standard dedicated beds (no extra guest charges)
            if extra_a == 0:
                fit_tier = 0 if delta >= 0 else 1
            else:
                fit_tier = 10 + extra_a
        elif occ_pref_upper == "MAX":
            # Priority: utilize maximum room capacity with extra beds (fewer rooms)
            if extra_a > 0 or len(opt.rooms) == min_rooms_needed:
                fit_tier = 0
            else:
                fit_tier = len(opt.rooms)
        else:
            # T4 Gate 3: Dedicated-bed allocations rank strictly above extra-bed variants
            if extra_a == 0:
                if delta == 0:
                    fit_tier = 0  # Exact dedicated bed fit
                elif delta == 1:
                    fit_tier = 1  # 1 extra dedicated bed
                elif delta > 1:
                    fit_tier = 2 + delta
                else:
                    fit_tier = 5
            else:
                fit_tier = 10 + extra_a
            
        return (
            fit_tier,
            not within_budget,
            not matches_rooms,
            opt.total_price_per_night,
            len(opt.rooms)
        )

    filtered_options.sort(key=sort_key)
    # Rank first, then cut: the survivors are the best-ranked combinations, and
    # everything downstream (pricing each option, building the card) only ever
    # reads the first few. Truncating before the sort would spend the same work
    # on arbitrary combinations instead of the guest's best ones.
    #
    # The cut is not a plain slice any more: it keeps a few of each AC class, so
    # a "show both" card can actually draw one. See `_keep_both_classes` — with
    # the flag off it returns exactly the old top-N.
    final_options = _keep_both_classes(filtered_options, rooms, MAX_ALLOCATION_OPTIONS)

    # Re-assign clean option_ids after sorting
    for i, opt in enumerate(final_options):
        opt.option_id = f"opt_{i+1}"

    # The policy charge goes on after the sort, not before. It is the same money
    # for every candidate — it depends on the party and the rate, not on how the
    # rooms are split — so it cannot change which option is cheapest, and adding
    # it here keeps `sort_key` reading the prices the allocator actually built.
    _apply_paid_child_charge(final_options, available_rooms, paid_children, fold_into_price=False)

    # Structured GCP log for auditability & troubleshooting
    candidates_info = [
        {
            "option_id": opt.option_id,
            "rooms": [r.room_name for r in opt.rooms],
            "base_adults": opt.base_adults,
            "max_adults": opt.max_adults,
            "extra_adults": opt.extra_adult_count,
            "ep_price": opt.total_price_per_night,
        }
        for opt in final_options
    ]
    logger.info(
        f"ROOM_SUGGESTION_RANKED | party={total_adults}a+{total_children}c | "
        f"candidates={len(candidates_info)} | shown={[c['rooms'] for c in candidates_info[:5]]}"
    )

    return final_options
