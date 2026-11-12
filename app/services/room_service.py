"""
Room Service — smart room matching based on guest requirements.

Handles:
- Capacity matching (adults + children vs baseAdults/baseChildren)
- Room type filtering (by preference keyword)
- Multi-room distribution (e.g., 3 adults across 2 rooms)
- Availability check (numberOfRooms > 0)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from app.config.settings import get_settings
from app.models.schemas import AllocationOption, SelectedRoom
from app.services.allocation_engine import (
    allocate_rooms,
    resolve_honourable_features,
    room_matches_feature,
)
from app.services.room_packing import (
    RoomCandidate,
    ac_preference_from_state,
    both_class_rows,
    cheapest_and_comfy,
    class_can_seat,
    upgraded_row,
    wants_both_classes,
)

logger = logging.getLogger(__name__)


def room_names_from_options(options: list[dict] | None) -> list[str]:
    """
    The distinct room names in an allocation-option list, "ROOM" stripped.

    Every caller that names rooms to the guest needs this list, and three of
    them wrote their own copy. The copies agreed by luck, not by construction —
    and the custom-room card quotes this list back to the guest as the choices
    it is asking them to pick from, so two copies drifting apart means the
    guest is asked to choose from names the prompt never showed them.

    Order is the order the options came in, which is the order they were ranked
    in. Callers that need it sorted sort it themselves.
    """
    names: list[str] = []
    for opt in options or []:
        for r in opt.get("rooms", []) or []:
            short = str(r.get("room_name", "")).replace("ROOM", "").replace("Room", "").strip()
            if short and short not in names:
                names.append(short)
    return names


async def find_suitable_rooms(
    hotel_id: int,
    adults: int = 2,
    children: int = 0,
    num_rooms: int | None = None,
    room_type_preference: str | None = None,
    budget: float | None = None,
    check_in: str | None = None,
    check_out: str | None = None,
    meal_plan_preference: str | None = None,
    room_aliases: dict[str, str] | None = None,
    room_match_threshold: float = 90.0,
    feature_preferences: list[str] | None = None,
    room_requirements: list[dict] | None = None,
    occupancy_preference: str | None = None,
    grouping_mode: str | None = None,
    paid_children: int = 0,
    wants_bed: bool = False,
    max_adults_per_room: int | None = None,
) -> tuple[list[AllocationOption], bool]:
    """
    Find optimal room allocations for the requested guests.
    Returns (allocations, fallback_used)

    `paid_children` is how many of `children` the hotel's own policy makes
    chargeable (the validator's band count). It defaults to 0, so every caller
    that does not know about the policy prices exactly as it did before.
    """
    adults = adults if adults is not None else 2
    children = children if children is not None else 0

    # ── A room count is a preference, not a cage, once the party needs more ──
    #
    # `allocate_rooms` honours a requested room count by searching exactly that
    # depth (`search_start = search_end = requested_num_rooms`). For a 5-person
    # party that is one depth's worth of sets, and nothing else. The live
    # transcript is what that costs: the guest had said "single room" when the
    # party was still being collected, so `num_rooms=1` reached the allocator,
    # the search ran at depth 1, and the pool came back as eight single-type
    # sets — three Deluxe King, three Premier, two Family, and so on. The
    # packer's cheapest set for those five guests is `1 x Deluxe Family Room +
    # 1 x Deluxe Room King Bed` at Rs 6,100, which is a set of TWO rooms. No
    # option could ever match it, `_match_option` returned None, both rows were
    # dropped, and the card fell back to the allocator's own first two options:
    # "3 x Deluxe Room" and "3 x Premier Room", at Rs 7,650 and Rs 8,640
    # against a real cheapest of Rs 6,100.
    #
    # Dropping the count here is safe because the card, not the allocator, is
    # what the guest sees: `room_searcher` replaces `allocation_options` with
    # the card's own rows a few lines later, and `option_number_count` /
    # `option_letter_count` filter those on `state["num_rooms"]`. The guest's
    # "1 room" is still honoured downstream — it just no longer decides what
    # the allocator is allowed to think of.
    #
    # Scoped to 5+ because that is where the card shows combinations at all
    # (`room_searcher.is_multi_room`). Below five, a room count is a plain
    # request the allocator honours exactly as before.
    if num_rooms is not None and (int(adults or 0) + int(children or 0)) >= 5 and not room_requirements:
        num_rooms = None

    from app.services import inventory_service
    from app.services.room_alias_resolver import (
        filter_rooms_by_reference,
        rooms_matching_bed_terms,
    )
    from app.services.allocation_engine import allocate_rooms, allocate_rooms_by_requirements, room_matches_feature

    all_rooms = await inventory_service.get_available_rooms(hotel_id, check_in, check_out)

    if room_requirements:
        req_rooms = all_rooms
        if feature_preferences and "AC" in feature_preferences and not any("non" in p.lower() for p in feature_preferences):
            ac_rooms = [r for r in all_rooms if room_matches_feature(r, "AC")]
            if ac_rooms:
                req_rooms = ac_rooms
        elif feature_preferences and any("non" in p.lower() for p in feature_preferences):
            non_ac_rooms = [r for r in all_rooms if room_matches_feature(r, "Non-AC")]
            if non_ac_rooms:
                req_rooms = non_ac_rooms

        req_allocations = allocate_rooms_by_requirements(
            req_rooms, room_requirements, meal_pref=meal_plan_preference, adults=adults, children=children,
            paid_children=paid_children,
        )
        if req_allocations:
            return req_allocations, False

    aliases = room_aliases
    threshold = room_match_threshold
    if aliases is None:
        try:
            from app.services.hotel_config_service import get_hotel_config
            config = await get_hotel_config(hotel_id)
            aliases = config.room_aliases
            threshold = getattr(config, "room_match_threshold", 90.0)
        except Exception:
            aliases = {}

    if room_type_preference:
        matched = filter_rooms_by_reference(
            room_type_preference,
            all_rooms,
            aliases=aliases or {},
            fuzzy_threshold=threshold,
        )
        if not matched:
            # A preference that names a BED names no room, so name resolution
            # cannot answer it however good the fuzzy threshold is. Reaching only
            # for names is what told a guest who asked for a non-AC double bed
            # that no room matched, while the reply listed the non-AC room whose
            # own description says "double bed" (2026-09-21, Sample Hotel). This
            # does not replace the name match above — it runs only when that
            # found nothing, so no preference that resolves today resolves
            # differently.
            #
            # Handed the rooms the guest can actually be given, not the whole
            # catalogue. The matcher is only as good as its pool: at Sample Hotel
            # "double bed" over everything returns the two rooms whose prose
            # says "double", and both of them are non-AC — so an AC guest's
            # matches were removed by the AC preference one step later and the
            # dead end walked straight back in (verified 2026-09-22). Scoped
            # first, tier 1 searches the AC rooms' own prose and tier 2 falls
            # through to their King beds, which is the answer that guest wanted.
            #
            # `resolve_honourable_features` decides which preferences may filter
            # at all — reusing it here rather than testing the raw preferences
            # keeps the "4 bed" defect it was written for from reappearing on
            # this path.
            honourable = resolve_honourable_features(all_rooms, feature_preferences)
            pool = [
                room for room in all_rooms
                if all(room_matches_feature(room, f) for f in honourable)
            ]
            matched = rooms_matching_bed_terms(room_type_preference, pool)
        if matched:
            allocations = allocate_rooms(
                matched,
                adults,
                children,
                num_rooms,
                budget=budget,
                meal_pref=meal_plan_preference,
                feature_preferences=feature_preferences,
                occupancy_preference=occupancy_preference,
                grouping_mode=grouping_mode,
                paid_children=paid_children,
                wants_bed=wants_bed,
                max_adults_per_room=max_adults_per_room,
            )
            if allocations:
                return allocations, False

    return allocate_rooms(
        all_rooms,
        adults,
        children,
        num_rooms,
        budget=budget,
        meal_pref=meal_plan_preference,
        feature_preferences=feature_preferences,
        occupancy_preference=occupancy_preference,
        grouping_mode=grouping_mode,
        paid_children=paid_children,
        wants_bed=wants_bed,
        max_adults_per_room=max_adults_per_room,
    ), bool(room_type_preference)


def _dummy_dates_for_nights(num_nights: int) -> tuple[str, str]:
    """Synthetic check-in/out used only for multi-night price math."""
    start = datetime.utcnow().date()
    end = start + timedelta(days=max(num_nights, 1))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _normalize_option(opt: Any) -> dict:
    if hasattr(opt, "model_dump"):
        return opt.model_dump()
    if hasattr(opt, "dict"):
        return opt.dict()
    if isinstance(opt, dict):
        return opt
    return dict(opt)


def _per_room_nightly(total_nightly: float, num_rooms: int) -> float:
    """
    The per-room figure a guest actually compares on.

    ``total_price_per_night`` on an option is the price of the WHOLE set, not
    one room: the allocation engine computes it as
    ``sum(r.price_per_night for r in selected)``, and
    ``enrich_options_with_stay_totals`` divides the *stay* total by the nights
    and never by the rooms. Printed as-is, a five-room option reads
    "Rs 11,500/night" next to a single room's "Rs 2,300/night" — the same units
    carrying five times the number, which is what made multi-room options look
    like they cost five times as much per room.

    Dividing by the room count puts every price line on that one footing. A set
    of a single room type divides back to exactly that room's catalogue rate, so
    the single-room case needs no branch of its own — and "3 x Extra Beds"
    divides to the per-room cost of the room *and* its extra bed, which is what
    a guest comparing options is actually choosing between.
    """
    if num_rooms and num_rooms > 0:
        return total_nightly / num_rooms
    return total_nightly


def _extra_bed_rate(state: dict) -> float:
    """
    The per-night charge for one extra bed, for a formatter that only has state.

    HONEST NOTE — this returns a flat Rs 600, and the lookups around it do not
    work. It was lifted verbatim from `build_structured_multi_room_options`,
    where it read `business_config["extra_bed_rate"]`; `BusinessConfig` is
    `features` + `limits` and carries no such field, so that branch could never
    fire and 600.0 was always the answer. Replacing a dead lookup with a
    constant is the same behaviour, stated plainly, rather than a working
    lookup that looks like it does something.

    The rate a hotel actually charges is real and lives elsewhere: the hotel
    policy document's `extra_bed_rate` is read in `room_repository.py:49` and
    lands on `RoomOccupancy.extra_adult_rate` (Rs 600 at hotel 111111, Rs 500
    at 108959). The allocator uses it — `allocation_engine.py:319` and `:438`
    both read it — so a quote for 108959 is built from 500 and this fallback
    would print 600 for the same hotel. That divergence is left in place here
    on purpose: fixing it changes what this formatter prints, and this function
    is an extraction, not a fix. It is flagged for the extra-person card, which
    carries the option's own room dicts and can read the true charge per room.
    """
    hotel_cfg = (state or {}).get("hotel_config") or {}
    biz_cfg = hotel_cfg.get("business_config") or {}
    if "extra_bed_rate" in biz_cfg and biz_cfg["extra_bed_rate"] is not None:
        try:
            return float(biz_cfg["extra_bed_rate"])
        except (ValueError, TypeError):
            pass
    return 600.0


def _option_title(opt: dict) -> str:
    """
    The one-line name of a room set: "2 x Deluxe + 1 x Extra Bed".

    Room types are counted, not listed, and the " Room" suffix is stripped
    first — so a set of one type keeps the word ("2 x Deluxe Room") while a
    mixed set drops it ("1 x Deluxe + 1 x Premier"), which is what stops the
    line reading as one long run of "Room ... Room ... Room".

    The extra bed is appended last and always counted, because it is the part
    of the title the guest is actually choosing: the bed is a line in the set,
    not a property of any room in it. Singular and plural differ on purpose.

    Extracted from `build_structured_multi_room_options`, where it was inline
    and reachable only from that one formatter. The new room card needs the
    same title for every set it prints, so it lives here now and both callers
    share it rather than drifting apart.
    """
    rms = opt.get("rooms", [])
    extra_a = opt.get("extra_adult_count", 0)

    counts: dict[str, int] = {}
    for r in rms:
        nm = clean_room_type_name(str(r.get("room_name") or "Room"))
        short_nm = nm.replace(" Room", "").strip()
        counts[short_nm] = counts.get(short_nm, 0) + 1

    parts = []
    is_single = len(counts) == 1 and extra_a == 0
    for snm, cnt in counts.items():
        if is_single:
            parts.append(f"{cnt} x {snm} Room" if cnt > 1 else f"1 x {snm} Room")
        else:
            parts.append(f"{cnt} x {snm}" if cnt > 1 else f"1 x {snm}")

    if extra_a > 0:
        bed_word = "Extra Bed" if extra_a == 1 else "Extra Beds"
        parts.append(f"{extra_a} x {bed_word}")

    return " + ".join(parts)


def _option_tagline(opt: dict, adults: int = 0, children: int = 0) -> tuple[str, str]:
    """
    The one-line pitch for a room set — read off the set itself, never off its
    position in the card.

    The card used to label its options by slot: whichever set landed in the
    "family" bucket was told "Good if you want to stay grouped together", and
    whatever was left over took the first entry of a rotating list of generic
    lines. That is a lie whenever the slot and the set disagree, and a live card
    proved it. A guest asking for Family Rooms for 4 adults and 2 children was
    shown "2 x Family Room" and "3 x Family Room" — the same room at the same
    rate, differing only in how many — and the second was labelled "Bigger
    rooms, upgraded interiors." It is the same room. There are more of them.

    So every line here is a fact about the set in hand:

        upgraded stock   ->  the only claim worth making about a Premier
        an extra bed     ->  the bed is the thing being chosen
        one room         ->  the whole group in it
        more than one    ->  the whole group spread across them

    The group's size is read from the session (adults + children) so the line
    names the party the set is actually for. With no count in state it says
    "your group" rather than inventing one.
    """
    rms = opt.get("rooms", []) or []
    count = len(rms)
    guest_n = int(adults or 0) + int(children or 0)
    group = f"your group of {guest_n}" if guest_n > 0 else "your group"

    names = [clean_room_type_name(str(r.get("room_name") or "")) for r in rms]
    if any("premier" in nm.lower() or "suite" in nm.lower() for nm in names):
        return "✨", "Bigger rooms, upgraded interiors."

    if int(opt.get("extra_adult_count") or 0) > 0:
        icon, line = "💰", f"{group} with an extra bed."
    elif count <= 1:
        icon, line = "👨👩👧👦", f"{group} in one room."
    else:
        icon, line = "👨👩👧👦", f"{group} in {count} rooms."

    return icon, line[0].upper() + line[1:]


def _per_room_price_line(
    rms: list[dict],
    total_nightly: float,
    stay_total: float | None = None,
) -> str:
    """
    The price line for a room set — always per room, per night.

    Two shapes, because a mixed set has no single per-room rate:

        one room type   ->  Rs 2,550 per room / night
        a mixed set     ->  Rs 1,550 + Rs 2,950 per room / night

    An average is not used for the mixed case. The mean of 1,550 and 2,950 is
    2,250, which is neither room's price and matches nothing the guest is later
    quoted, so each distinct rate is listed instead. Rooms sharing a rate are
    listed once; how many there are is already in the title above
    ("2 x Deluxe Room").

    A room's rate here is what that room costs the guest, which is not
    ``price_per_night`` alone: an extra adult or child bed is charged on the room
    that holds it and recorded in a separate field, and ``allocate_rooms`` adds
    all three terms when it prices the set. Folding the bed charge into the room
    that carries it is what makes the parts sum to the set total — without it an
    8-guest option printed "Rs 3,880 + Rs 1,850" against a real Rs 6,330,
    understating the set by the Rs 600 extra bed it does not mention.

    The counts are not multiplied in: this is a rate, not a subtotal. And a
    breakdown is only shown for a set whose parts are *proved* to add up to that
    set's total, so both readings are tried and the one that reconciles with the
    total wins. Neither reconciling means the data is not a room set we
    understand, and the set total over the room count is then the safe line.

    The two readings exist because two callers used to disagree on where the bed
    charge lives: ``allocate_rooms`` kept it beside ``price_per_night``, while
    ``allocate_rooms_by_requirements`` folded it in *and* recorded it. They agree
    now — both keep it beside (2026-09-23, see ``allocation_engine``) — which
    makes ``with_bed`` the reading that wins on every set, including the one that
    used to reconcile against ``rate_only``. The second reading is left in place
    rather than deleted: it is the guard that keeps a card honest about a room
    set it cannot account for, and it costs one arithmetic pass.

    A free room is allowed through as ``Rs 0``: an FOC room genuinely costs
    nothing, and printing it is more honest than averaging its zero away.

    ``stay_total`` is optional and appends the whole-stay, whole-set figure
    after the rate. The rate above cannot express what separates two sets of
    the same room — "2 x Family Room" and "3 x Family Room" both read
    "Rs 3,550 per room / night", so the card showed the guest two options that
    looked identical and cost Rs 14,200 apart. The total is that difference.
    Defaults to None so every existing caller keeps the line it had.
    """
    # One entry per room, in order, for each convention — deduped only for
    # display, so the checks below still compare room-for-room against the total.
    with_bed = [
        float(r.get("price_per_night") or 0)
        + float(r.get("extra_adult_charge") or 0)
        + float(r.get("extra_child_charge") or 0)
        for r in rms
    ]
    rate_only = [float(r.get("price_per_night") or 0) for r in rms]

    per_room: list[float] | None = None
    for candidate in (with_bed, rate_only):
        if (
            candidate
            and any(rate > 0 for rate in candidate)
            and all(rate >= 0 for rate in candidate)
            and abs(sum(candidate) - total_nightly) < 1
        ):
            per_room = candidate
            break

    # No rooms, or no reading of them that adds up: the set total over the room
    # count is exact for the single-rate case and the only safe answer for the
    # rest.
    suffix = f" · Rs {stay_total:,.0f} total" if stay_total else ""

    if per_room is None:
        return f"💰 *Rs {_per_room_nightly(total_nightly, len(rms)):,.0f} per room / night{suffix}*"

    distinct: list[float] = []
    for rate in per_room:
        if rate not in distinct:
            distinct.append(rate)

    if len(distinct) == 1:
        return f"💰 *Rs {distinct[0]:,.0f} per room / night{suffix}*"

    return f"💰 *{' + '.join(f'Rs {rate:,.0f}' for rate in distinct)} per room / night{suffix}*"


async def enrich_options_with_stay_totals(
    options: list[Any],
    num_nights: int,
    hotel_id: int,
    check_in: str | None = None,
    check_out: str | None = None,
) -> list[dict]:
    options = [_normalize_option(o) for o in (options or [])]
    from app.services.hotel_config_service import get_hotel_config
    from app.services.pricing_engine import PricingEngine

    if not check_in or not check_out:
        check_in, check_out = _dummy_dates_for_nights(num_nights)

    config = await get_hotel_config(hotel_id)

    enriched: list[dict] = []
    for opt in options:
        rooms = [SelectedRoom(**r) for r in opt.get("rooms", [])]
        adults = sum((r.assigned_adults or 2) for r in rooms) or 2
        children = sum((r.assigned_children or 0) for r in rooms) or 0
        breakdown = await PricingEngine.get_price(
            rooms, check_in, check_out, hotel_id, config=config, adults=adults, children=children
        )
        opt_copy = dict(opt)
        opt_copy["total_price_for_stay"] = float(round(breakdown.total / 10) * 10)
        inclusive_nightly = float(round((breakdown.total / max(1, num_nights)) / 10) * 10)
        opt_copy["total_price_per_night_inclusive"] = inclusive_nightly
        opt_copy["total_price_per_night"] = inclusive_nightly
        opt_copy["num_nights"] = num_nights
        enriched.append(opt_copy)
    return enriched


async def build_room_type_pricing_card(
    hotel_id: int,
    room_type_keyword: str,
    num_nights: int,
    adults: int = 2,
    children: int = 0,
    num_rooms: int | None = None,
    meal_plan_preference: str | None = None,
    state: dict | None = None,
) -> dict | None:
    """
    Build a structured room_options card for a room-type + nights pricing query,
    e.g. "deluxe rooms price for 2 nights" or "book 2 deluxe rooms for 2 nights".

    ``state`` is optional and read-only: it supplies the guest's language for
    the renderer and, when the guest has given both, the check-in/check-out
    dates to allocate and price against. Nothing in it is ever written to.
    Everything this function computes itself (counts, nights) overrides what
    is already in it, and when the guest has not given both dates the
    synthetic ones this function invents are kept out of the header — see the
    render block at the end.
    """
    # At least 1 adult per requested room when guest count not specified
    if num_rooms and adults < num_rooms:
        adults = num_rooms

    # Price on the dates the guest actually gave. `state` carries them — the
    # caller passes the whole session state — and the sibling producer at
    # room_query_handler.py:140-149 already prices this way, so this is the
    # two paths converging rather than a new rule. Only when the guest has
    # given neither date do we fall back to synthetic ones; the card's own
    # footer asks for dates, so that path is reachable.
    real_check_in = (state or {}).get("check_in") or None
    real_check_out = (state or {}).get("check_out") or None
    priced_on_real_dates = bool(real_check_in and real_check_out)

    if priced_on_real_dates:
        # `enrich_options_with_stay_totals` divides the stay total by
        # `num_nights`, so the nights we bill for have to be the nights
        # between the dates we are billing — otherwise the per-night line
        # contradicts the stay total it came from.
        from app.services.booking_engine import calculate_num_nights
        try:
            num_nights = calculate_num_nights(real_check_in, real_check_out) or num_nights
        except Exception:
            pass  # unparseable dates: keep the caller's num_nights

    check_in, check_out = (
        (real_check_in, real_check_out)
        if priced_on_real_dates
        else _dummy_dates_for_nights(num_nights)
    )

    allocations, _ = await find_suitable_rooms(
        hotel_id=hotel_id,
        adults=adults,
        children=children,
        num_rooms=num_rooms,
        room_type_preference=room_type_keyword,
        meal_plan_preference=meal_plan_preference,
        # Allocate on the dates we price on. Offering a room that is not free
        # on the guest's dates is the same lie as mispricing it, and the
        # caller already has an honest branch for "found nothing" — it tells
        # the guest so instead of quoting a room they cannot have.
        check_in=check_in,
        check_out=check_out,
    )
    if not allocations:
        return None

    options = [a.model_dump() for a in allocations[:6]]
    options = await enrich_options_with_stay_totals(
        options, num_nights, hotel_id, check_in, check_out
    )

    label = room_type_keyword.strip().title()
    night_word = "night" if num_nights == 1 else "nights"
    if num_rooms and num_rooms > 1:
        room_phrase = f"**{num_rooms} {label} rooms**"
    elif num_rooms == 1:
        room_phrase = f"**1 {label} room**"
    else:
        room_phrase = f"**{label}** room options"

    guest_phrase = f"{adults} adult{'s' if adults != 1 else ''}"
    if children:
        guest_phrase += f", {children} child{'ren' if children != 1 else ''}"

    intro = (
        f"Here are options for {room_phrase} for **{num_nights} {night_word}** "
        f"(~{guest_phrase}):\n\nPlease select an option:"
    )

    # Render the options into `text` — the contract every other producer of a
    # room_options card already follows. WhatsApp's card renderer prints a
    # room_options card's `text` and reads `data.options` ONLY when the text is
    # empty (whatsapp.py:560-563), so a card whose text is a bare preamble
    # reaches the guest alone: "Please select an option:" with no options under
    # it. That is the live 2026-09-17 message in chat 919748533727.
    #
    # `format_s4_room_options` takes exactly the shape this function already
    # produces — `enrich_options_with_stay_totals` output.
    #
    # This is now the only caller left on that renderer. The search card and the
    # mid-browse total both moved to `build_room_options_card`, which letters its
    # combinations. This card numbers its options 1..N and keeps doing so: it is
    # reachable only when `allocation_options` is empty — a pre-search question
    # like "deluxe price for 2 nights" — so its numbers are never on screen next
    # to the search card's letters, and changing its vocabulary is a reply-parsing
    # change, not a rendering one.
    #
    # `check_in`/`check_out` reach the header only when they are the guest's
    # own, because S4 prints whatever dates it is handed. On the
    # synthetic-dates path they are withheld below: those are values this
    # function invented for the price math, and printing them would show the
    # guest dates they never supplied.
    render_state: dict = {**(state or {})}
    render_state.update(
        {
            "adults": adults,
            "children": children,
            # Set explicitly: S4 prefers `effective_*` over `adults`/`children`,
            # so a stale one carried in `state` would silently win over the
            # counts this function just computed.
            "effective_adults": adults,
            "effective_children": children,
            "num_rooms": num_rooms,
            "num_nights": num_nights,
        }
    )
    if priced_on_real_dates:
        # The price above was computed for exactly these dates, so the header
        # may name them.
        render_state["check_in"] = real_check_in
        render_state["check_out"] = real_check_out
    else:
        render_state.pop("check_in", None)
        render_state.pop("check_out", None)
    rendered = format_s4_room_options(options, render_state).strip()

    return {
        "type": "room_options",
        # Falling back to the preamble keeps the bot's own words rather than
        # an empty message if the renderer ever returns nothing.
        "text": rendered or intro,
        "data": {"options": options, "num_nights": num_nights, "budget": None},
        "buttons": None,
    }


def clean_room_type_name(name: str) -> str:
    n = str(name).strip()
    n = n.replace("DELUXE ROOM KING BED", "Deluxe Room")
    n = n.replace("PREMIER ROOM KING BED", "Premier Room")
    n = n.replace("STANDARD NON AC ROOM", "Standard Room")
    n = n.replace("DELUXE FAMILY ROOM", "Family Room")
    n = n.replace("DELUXE ROOM FOUR BED", "Four Bed Room")
    n = n.replace("STANDARD NON AC TRIPPLE BED", "Triple Bed Room")
    n = n.replace(" KING BED", "").replace(" NON AC", "").replace(" ROOM", " Room")
    return n.strip()


def _format_option_math_line(opt: dict, extra_bed_rate: float) -> str:
    rms = opt.get("rooms", [])
    extra_a = opt.get("extra_adult_count", 0)

    # Group rooms by name and nightly price
    room_groups: dict[str, dict] = {}
    for r in rms:
        nm = clean_room_type_name(str(r.get("room_name") or "Room"))
        p = float(r.get("price_per_night") or 0)
        k = f"{nm}_{p}"
        if k not in room_groups:
            room_groups[k] = {"name": nm, "price": p, "count": 0}
        room_groups[k]["count"] += 1

    terms = []
    total_calc = 0.0
    for grp in room_groups.values():
        cnt = grp["count"]
        prc = grp["price"]
        total_calc += prc * cnt
        if cnt > 1:
            terms.append(f"₹{prc:,.0f} × {cnt}")
        else:
            terms.append(f"₹{prc:,.0f}")

    if extra_a > 0:
        total_calc += extra_bed_rate * extra_a
        if extra_a > 1:
            terms.append(f"₹{extra_bed_rate:,.0f} × {extra_a}")
        else:
            terms.append(f"₹{extra_bed_rate:,.0f}")

    math_expr = " + ".join(terms)
    total_p = total_calc
    opt["total_price_per_night"] = total_calc
    return f"{math_expr} = ₹{total_p:,.0f}"


def format_option_numbers(count: int) -> str:
    """
    The numbers a guest can reply with, as Markdown: "*1*", "*1* or *2*",
    "*1*, *2*, or *3*". Never a number that does not exist.

    Shared with the "reply with an option number" prompts in response_generator
    and menu_handler, which used to carry a literal "*1*, *2*, or *3*".
    """
    numbers = [f"*{n}*" for n in range(1, max(int(count), 1) + 1)]
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return f"{numbers[0]} or {numbers[1]}"
    return ", ".join(numbers[:-1]) + f", or {numbers[-1]}"


def option_number_count(options: list | None, state: dict | None = None) -> int:
    """
    How many numbered options the guest can actually reply with.

    Both formatters cap the rendered catalogue at 3 entries and always append a
    trailing "Custom / Build your own", so the reply range is
    1..min(n, 3) + 1. Past three options the numbers are 1-4 whichever three
    happened to be rendered, which is why this cannot simply be a constant —
    and why the old literal "(e.g. *1*, *2*, or *3*)" was a lie whenever fewer
    than two options reached the guest (Chat B 06:47:45 and 06:51:00).

    The room-count filter mirrors build_room_options_card, the function that
    produces the cards these prompts sit next to.
    """
    options = [_normalize_option(o) for o in (options or [])]
    state = state or {}
    count = len(options)
    req_num_rooms = state.get("num_rooms")
    if req_num_rooms and req_num_rooms > 1:
        matching = [o for o in options if len(o.get("rooms", [])) == req_num_rooms]
        if matching:
            count = len(matching)
    return min(count, 3) + 1


def format_option_letters(count: int) -> str:
    """
    The letters a guest can reply with, as Markdown: "*A*", "*A* or *B*",
    "*A*, *B*, or *C*". The letter twin of `format_option_numbers`.

    Kept alongside the number formatter because the older screens still number
    their entries: the room card is lettered now (A, B, C plus the trailing
    "tell me how you want to split", which is why the letter count is one more
    than the number of priced combinations), and these prompts are used by
    paths that were not the card — so the two forms have to stay in step rather
    than one being deleted.
    """
    letters = [f"*{chr(ord('A') + i)}*" for i in range(max(int(count), 1))]
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def option_letter_count(options: list | None, state: dict | None = None) -> int:
    """
    How many letters the guest can actually reply with on a room card.

    Same cap and the same `num_rooms` filter as `option_number_count`, because
    both describe the same rendered card — but the letters are not the numbers.
    A card with two priced combinations prints A, B and then C for the
    build-your-own line, so the reply range is A-C: `min(n, 3) + 1` letters,
    where the numbers version counted `min(n, 3) + 1` numbers for a card that
    numbered its options instead of its rooms.
    """
    options = [_normalize_option(o) for o in (options or [])]
    state = state or {}
    count = len(options)
    req_num_rooms = state.get("num_rooms")
    if req_num_rooms and req_num_rooms > 1:
        matching = [o for o in options if len(o.get("rooms", [])) == req_num_rooms]
        if matching:
            count = len(matching)
    return min(count, ROOM_CARD_MAX_COMBOS) + 1


def _reply_number_hint(custom_idx: int) -> str:
    """
    The "reply with a number" line for the two option formatters below.

    They already know their own count, because they choose how many entries to
    render: options are numbered 1..custom_idx, where custom_idx is the trailing
    "Custom / Build your own" entry. The range therefore ends at custom_idx.

    This replaces a literal f"Reply *1*, *2*, *3*, or *{custom_idx}*", which
    ignored the real count: with a single option rendered, custom_idx is 2, so
    the guest was told "Reply *1*, *2*, *3*, or *2*" (Chat B, 2026-09-16
    06:36:28) — three of the four numbers did not exist.
    """
    return f"👉 Reply {format_option_numbers(custom_idx)} to continue"


def _drop_dominated(offers: list[dict]) -> list[dict]:
    """
    Drop a set that is only a more expensive way to buy the same room types.

    Asking for a room *type* rather than a room *count* makes `find_suitable_rooms`
    return one allocation per feasible room count, so a party of six asking for a
    family room was offered `2 x Family Room` and `3 x Family Room` as though
    those were two products. They are one product: same room, same per-room
    rate, the second set costing Rs 14,200 more for a room the group does not
    need. The guest sees two choices and no reason to prefer either.

    A set goes only when another set on the card is the same room types with no
    more rooms, no more extra beds and no higher price — the three axes a guest
    actually trades off. That is what keeps the card's central choice intact:
    bucket 1 (fewer rooms, extra beds) against bucket 2 (more rooms, everyone in
    a bed of their own) is a real decision, and neither dominates the other, so
    both survive. Only the twin disappears.

    A pair of sets that differ on no axis is not dominated either way — each
    needs one strictly-better axis to displace the other — so exact duplicates
    are left for the caller's `option_id` dedup to handle.

    Options carrying no rooms (the allocator fallback) share the empty
    signature and are never compared, so a card built from a fallback is
    returned untouched.
    """
    def rooms_of(opt: dict) -> list:
        return opt.get("rooms") or []

    def signature(opt: dict) -> frozenset:
        return frozenset(
            clean_room_type_name(str(r.get("room_name") or "")).strip().lower()
            for r in rooms_of(opt)
        )

    def price_of(opt: dict) -> float:
        return float(
            opt.get("total_price_per_night_inclusive")
            or opt.get("total_price_per_night")
            or 0
        )

    def cheaper_way(opt: dict, other: dict) -> bool:
        return (
            len(rooms_of(other)) <= len(rooms_of(opt))
            and int(other.get("extra_adult_count") or 0) <= int(opt.get("extra_adult_count") or 0)
            and price_of(other) <= price_of(opt)
            and (
                len(rooms_of(other)) < len(rooms_of(opt))
                or int(other.get("extra_adult_count") or 0) < int(opt.get("extra_adult_count") or 0)
                or price_of(other) < price_of(opt)
            )
        )

    kept: list[dict] = []
    for opt in offers:
        sig = signature(opt)
        if sig and any(
            other is not opt and signature(other) == sig and cheaper_way(opt, other)
            for other in offers
        ):
            logger.info("ROOM_CARD | dropped a dominated set | set=%s", _option_title(opt))
            continue
        kept.append(opt)
    return kept


def build_structured_multi_room_options(
    options: list[dict],
    state: dict | None = None,
    num_nights: int = 1,
    is_hindi_or_hinglish: bool = False,
) -> tuple[str, list[dict]]:
    """
    Format up to 5 curated room-set cards, matching product's shape:

    1️⃣ 5 Deluxe Rooms 👨👩👧👦
    💰 ₹2,300 per room / night · ₹46,000 total

    Each set is chosen into the card by the buckets below, which decide what
    the guest is offered and in what order. What the card then *says* about a
    set is read off that set at render time — see `_option_tagline` — so no
    bucket can describe its set as something it is not, and `_per_room_price_line`
    carries the whole-stay total that separates two sets sharing a rate.
    """
    state = state or {}
    options = [_normalize_option(o) for o in (options or [])]
    extra_bed_rate = _extra_bed_rate(state)

    # The party these sets are being chosen for, so each option can name the
    # group it is for. `effective_*` first, matching `format_s4_room_options`.
    adults = state.get("effective_adults")
    if adults is None:
        adults = state.get("adults") or 0
    children = state.get("effective_children")
    if children is None:
        children = state.get("children") or 0

    # Categorize options into 5 value buckets
    comfort_opts: list[dict] = []     # Dedicated beds, base room (Deluxe / Standard)
    value_opts: list[dict] = []       # With extra beds (cheapest per night)
    premium_opts: list[dict] = []     # Upgraded category (Premier / Suite)
    family_opts: list[dict] = []      # Family room combinations
    group_large_opts: list[dict] = [] # Four Bed / minimum room count combos
    other_combos: list[dict] = []

    for opt in options:
        rms = opt.get("rooms", [])
        extra_a = opt.get("extra_adult_count", 0)
        unique_rooms = [clean_room_type_name(r.get("room_name")) for r in rms]
        has_family = any("family" in n.lower() for n in unique_rooms)
        has_four_bed = any("four" in n.lower() or "4" in n.lower() for n in unique_rooms)
        has_premier = any("premier" in n.lower() or "suite" in n.lower() for n in unique_rooms)
        is_single_type = len(set(unique_rooms)) == 1

        if has_family:
            family_opts.append(opt)
        elif has_four_bed:
            group_large_opts.append(opt)
        elif has_premier:
            premium_opts.append(opt)
        elif is_single_type and extra_a == 0:
            comfort_opts.append(opt)
        elif extra_a > 0:
            value_opts.append(opt)
        else:
            other_combos.append(opt)

    # Sort each bucket by price
    for b in (comfort_opts, value_opts, premium_opts, family_opts, group_large_opts, other_combos):
        b.sort(key=lambda x: float(x.get("total_price_per_night") or 0))

    # The sets to show, in the order to show them. Only membership and order
    # are decided here; the icon and the line under each set are read off the
    # set itself at render time (see `_option_tagline`).
    curated: list[dict] = []

    # Bucket 1: Comfort (Dedicated beds)
    if comfort_opts:
        curated.append(comfort_opts.pop(0))
    elif other_combos and other_combos[0].get("extra_adult_count", 0) == 0:
        curated.append(other_combos.pop(0))

    # Bucket 2: Value (With extra bed)
    if value_opts:
        curated.append(value_opts.pop(0))
    elif other_combos and other_combos[0].get("extra_adult_count", 0) > 0:
        curated.append(other_combos.pop(0))

    # Bucket 3: Premier / Upgraded
    if premium_opts:
        curated.append(premium_opts.pop(0))

    # Bucket 4: Family Room combinations
    if family_opts:
        curated.append(family_opts.pop(0))

    # Bucket 5: Four Bed / Grouped
    if group_large_opts:
        curated.append(group_large_opts.pop(0))

    # Append all remaining options to allow multi-batch MORE pagination
    all_remaining = comfort_opts + value_opts + premium_opts + family_opts + group_large_opts + other_combos
    all_remaining.sort(key=lambda x: float(x.get("total_price_per_night") or 0))

    while all_remaining:
        rem_opt = all_remaining.pop(0)
        existing_ids = {c.get("option_id") for c in curated if c.get("option_id")}
        if rem_opt.get("option_id") and rem_opt.get("option_id") in existing_ids:
            continue
        curated.append(rem_opt)

    # Before pagination, so `MORE` pages over the sets that survived rather than
    # counting a twin that will never be drawn.
    curated = _drop_dominated(curated)

    num_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    lines = []
    flattened_options = []

    # T2 & T3: Capped at 3 options max per batch, paginated by quote_offset / show_more
    show_more = bool(state.get("show_more_combinations"))
    offset = int(state.get("quote_offset") or 0)
    if show_more and offset == 0:
        offset = 3
    if offset >= len(curated):
        offset = 0
    displayed_curated = curated[offset : offset + 3]
    if not displayed_curated:
        displayed_curated = curated[:3]

    for idx, opt in enumerate(displayed_curated, 1):
        badge = num_emojis[idx - 1] if idx <= len(num_emojis) else f"{idx}️⃣"
        opt["option_label"] = str(idx)
        flattened_options.append(opt)

        rms = opt.get("rooms", [])
        extra_a = opt.get("extra_adult_count", 0)

        title = _option_title(opt)

        total_nightly = float(
            opt.get("total_price_per_night_inclusive")
            or opt.get("total_price_per_night")
            or round((sum(float(r.get("price_per_night") or 0) for r in rms) + (extra_a * extra_bed_rate)) / 10) * 10
        )
        # The whole stay for the whole set — the figure that actually separates
        # two sets of the same room at the same rate. Absent only when the
        # caller skipped `enrich_options_with_stay_totals`, and multiplying out
        # then gives the same number that function would have written.
        stay_total = opt.get("total_price_for_stay") or (
            total_nightly * max(1, int(num_nights or 1))
        )
        price_line = _per_room_price_line(rms, total_nightly, float(stay_total))

        icon, tagline = _option_tagline(opt, adults, children)
        tag_str = f" — {icon} *{tagline}*" if tagline else ""
        lines.append(f"{badge} *{title}*{tag_str}")
        lines.append(f"{price_line}")
        bed_note = extra_bed_note(extra_a)
        if bed_note:
            lines.append(bed_note)
        lines.append("")

    curr_idx = len(displayed_curated)
    custom_idx = curr_idx + 1
    custom_badge = num_emojis[custom_idx - 1] if custom_idx <= len(num_emojis) else f"{custom_idx}️⃣"
    lines.append(f"{custom_badge} 🛠️ *Custom / Build your own*\n")

    lines.append("_Rates final · GST included_\n")
    lines.append(_reply_number_hint(custom_idx))
    lines.append("🔄 Reply *MORE* for other combinations")
    if is_hindi_or_hinglish:
        lines.append("📸 Photos dekhne ke liye *\"photo\"* likhiye")
    else:
        lines.append("📸 Reply *PHOTO* to see rooms")

    return "\n".join(lines), flattened_options


def _occupancy_header(dates_str: str, guest_str: str, night_str: str) -> str:
    """
    Join the header's three parts, skipping the ones we do not know.

    Unconditional joining printed a stray leading pipe when the date range was
    empty: a room-type pricing query ("3 bed price for 1 night") has nights but
    no dates, so the guest saw ``📅 * | 3 adults | 1 night*``. Every part here
    is one we actually have.
    """
    return " | ".join(p for p in (dates_str, guest_str, night_str) if p)


def format_s4_room_options(
    options: list[dict] | None,
    state: dict | None = None,
) -> str:
    """
    Format S4_SUGGEST room options template matching exact V4 specification.
    """
    options = [_normalize_option(o) for o in (options or [])]
    state = state or {}
    check_in = state.get("check_in", "")
    check_out = state.get("check_out", "")
    adults = state.get("effective_adults") if state.get("effective_adults") is not None else (state.get("adults") or 2)
    children = state.get("effective_children") if state.get("effective_children") is not None else (state.get("children") or 0)

    from app.utils.helpers import format_quote_date_range
    from app.prompts.templates import format_assumption_line

    dates_str = format_quote_date_range(check_in, check_out)
    if not dates_str and check_in and check_out:
        dates_str = f"{check_in} - {check_out}"

    num_nights = state.get("num_nights")
    if not num_nights and check_in and check_out:
        from app.services.booking_engine import calculate_num_nights
        try:
            num_nights = calculate_num_nights(check_in, check_out)
        except Exception:
            num_nights = 1
    num_nights = num_nights or 1

    night_str = f"{num_nights} night" if num_nights == 1 else f"{num_nights} nights"
    adult_str = f"{adults} adult" if adults == 1 else f"{adults} adults"
    guest_str = adult_str
    if children > 0:
        guest_str += f", {children} child" if children == 1 else f", {children} children"

    lang = (state.get("preferred_language") or "English").lower()
    is_hindi_or_hinglish = lang in ["hindi", "hinglish"]

    header = f"*{_occupancy_header(dates_str, guest_str, night_str)}*\n\n"
    total_guests = adults + children

    is_multi_room_group = total_guests >= 5

    if is_multi_room_group and options:
        cat_body, _ = build_structured_multi_room_options(
            options, state=state, num_nights=num_nights, is_hindi_or_hinglish=is_hindi_or_hinglish
        )
        full_text = header + cat_body
    else:
        # Standard format for room search suggestions (suggest 3-4 options whenever possible)
        req_num_rooms = state.get("num_rooms")
        if req_num_rooms:
            matching_room_count_opts = [o for o in options if len(o.get("rooms", [])) == req_num_rooms]
            effective_options = matching_room_count_opts if matching_room_count_opts else options
        else:
            effective_options = options

        num_badges = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣"}
        lines = [f"🏨 *Rooms Available for You*\n📅 *{_occupancy_header(dates_str, guest_str, night_str)}*\n"]
        if effective_options:
            displayed_opts = effective_options[:3]
            opt1 = displayed_opts[0]
            rms1 = opt1.get("rooms", [])
            extra_a1 = opt1.get("extra_adult_count", 0)
            nightly_price1 = float(opt1.get("total_price_per_night_inclusive") or opt1.get("total_price_per_night") or round((sum(float(r.get("price_per_night") or 0) for r in rms1) + (extra_a1 * 600.0)) / 10) * 10)

            for idx, opt in enumerate(displayed_opts, 1):
                rms = opt.get("rooms", [])
                extra_a = opt.get("extra_adult_count", 0)
                nightly_price = float(opt.get("total_price_per_night_inclusive") or opt.get("total_price_per_night") or round((sum(float(r.get("price_per_night") or 0) for r in rms) + (extra_a * 600.0)) / 10) * 10)
                badge = num_badges.get(idx, f"*{idx}*")
                
                parts = []
                counts = {}
                for r in rms:
                    nm = clean_room_type_name(str(r.get("room_name") or "Room"))
                    counts[nm] = counts.get(nm, 0) + 1
                for nm, count in counts.items():
                    parts.append(f"{count} x {nm}" if count > 1 else nm)
                if extra_a > 0:
                    parts.append(f"{extra_a} extra bed" if extra_a == 1 else f"{extra_a} extra beds")
                room_title = " + ".join(parts)

                tag = ""
                if idx == 1:
                    tag = " — ⭐ *Sabse Sahi*" if is_hindi_or_hinglish else " — ⭐ *Recommended*"
                elif idx > 1:
                    if nightly_price < nightly_price1:
                        tag = " — 🏷️ _(sasta)_" if is_hindi_or_hinglish else " — 🏷️ _(best value)_"
                    elif len(rms) > len(opt1.get("rooms", [])) or (extra_a == 0 and extra_a1 > 0):
                        tag = " — 🛋️ _(zyada jagah)_" if is_hindi_or_hinglish else " — 🛋️ _(more space)_"

                price_line = _per_room_price_line(rms, nightly_price)

                lines.append(f"{badge} *{room_title}*{tag}")
                lines.append(f"{price_line}")
                bed_note = extra_bed_note(extra_a)
                if bed_note:
                    lines.append(bed_note)
                lines.append("")

            curr_idx = len(displayed_opts)
            custom_idx = curr_idx + 1
            custom_badge = num_badges.get(custom_idx, f"*{custom_idx}*")
            lines.append(f"{custom_badge} 🛠️ *Custom / Build your own*\n")

            lines.append("_Rates final · GST included_\n")
            lines.append(_reply_number_hint(custom_idx))
            lines.append("🔄 Reply *MORE* for other combinations")
            if is_hindi_or_hinglish:
                lines.append("📸 Photos dekhne ke liye *\"photo\"* likhiye")
            else:
                lines.append("📸 Reply *PHOTO* to see rooms")

        full_text = "\n".join(lines)

    child_notes = state.get("child_age_notes") or []
    if child_notes:
        child_notes_str = " ".join(child_notes)
        full_text += f"\n\n_{child_notes_str}_"

    assumption_str = format_assumption_line(state.get("assumptions"))
    if assumption_str:
        full_text += f"\n\n{assumption_str}"

    return full_text


# ── The room card (client spec, 2026-09-19) ──────────────────────────────────
#
# Two sections, from two different datasets, and that is the point of it:
#
#   ROOMS 1-4     one room TYPE, its per-night Room Only rate, its capacity
#   COMBINATIONS  one room SET, priced for the whole stay
#
# Those are different grains and they were being served from one list. Every
# previous producer rendered a single set of priced options, which is why the
# multi-room and single-room cards forked on `total_guests >= 5` and why a
# party size changed which formatter ran at all. Splitting the sections means
# one renderer serves every party size.
#
# How many PRICED rows the card can print — three, not four. The "Build your
# own" line is not one of them: the resolver reads a reply of `n` as option `n`
# for n up to this number, and `this + 1` as build your own
# (`room_card_reply._choice`). Keeping the value at three is what makes the
# third tier land on row 3 and build-your-own on row 4 without the resolver
# moving at all.
#
# `room_card_reply` also caps a stale `allocation_options` list at this, so a
# reply past it resolves to nothing rather than to a row the guest never saw.
# The letters are kept as an alias for the numbers — a guest holding an older
# card in their scrollback still gets an answer, and the resolver is scoped on
# the active prompt, so those letters do not collide with the AC menu's own
# 1/2/3 (see app/utils/room_card_reply.py).
ROOM_CARD_MAX_COMBOS = 3

# What a card row is numbered with. Keycap emoji, not "1." — the card text is
# handed to `st.markdown` in chat_ui.py:264, and Markdown reads a line starting
# "1." as an ordered-list marker and renders the line without it. The code emits
# "1. Economical"; the guest sees "Economical" and is then asked to "Reply 1-2".
#
# Keycaps survive every renderer this card passes through: Markdown leaves them
# alone, and WhatsApp — which supports only *bold* and _italic_, no lists —
# shows them as digits. The rest of the bot already numbers with keycaps
# (`_card_party`'s menus above), so this is the house style, not a workaround.
_CARD_NUMERALS = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣")

def _card_numeral(position: int) -> str:
    """Kept total so a card can never raise on an unexpected row count."""
    if 1 <= position <= len(_CARD_NUMERALS):
        return _CARD_NUMERALS[position - 1]
    return f"{position}."


def _row_letter(position: int) -> str:
    """
    A, B, C…, and total past Z.

    **No longer called.** The card numbers its rows with `_card_numeral` as of
    2026-09-23; this is left in place rather than deleted because
    `room_card_reply` still answers a letter as a position, so a guest holding
    a lettered card in their scrollback is still served, and the two are easier
    to keep in step when they sit in one file. Delete it when no lettered card
    can plausibly still be on screen.
    """
    if 1 <= position <= 26:
        return chr(ord("A") + position - 1)
    return f"{position}."

# the product's template ends `{meal_footer} · +{gst}% GST`. That "+" is wrong for
# this inventory and is NOT reproduced: both hotels store GST-inclusive
# catalogue rates — 111111's own settings document says
# `taxConfig: {"enabled": true, "slab": "include"}`, and 108959 has no
# taxConfig at all, which `booking_engine.py:61-64` also reads as inclusive.
# Appending "+18% GST" to an inclusive Rs 2,550 would quote Rs 3,009. The
# footer says what is true today. When a hotel is configured exclusive, this is
# the one string that changes.
MEAL_FOOTER_EP = "Room only · GST included"


def _room_card_header(state: dict, adults: int, children: int, num_nights: int) -> str:
    """
    "{guest} ji / Hotel · dates (2N) · 3 guests".

    Children are named separately rather than folded into a total. The client's
    template says "{adults} guests", but a 3-adult/2-child party would then read
    "3 guests" on a card priced for five, and the capacity line underneath would
    look inconsistent with the header. Flagged for the client; this is the
    version that does not contradict itself.
    """
    cfg = state.get("hotel_config") or {}
    hotel_name = cfg.get("hotel_name") or state.get("hotel_name") or ""

    from app.utils.helpers import format_quote_date_range

    ci = state.get("check_in") or ""
    co = state.get("check_out") or ""
    dates = format_quote_date_range(ci, co) or (f"{ci}–{co}" if ci or co else "")

    bits = [b for b in (hotel_name, dates) if b]
    line = " · ".join(bits)
    if num_nights:
        line += f" ({num_nights}N)"

    guests = f"{adults} guests"
    if children:
        guests += f" + {children} children"
    line += f" · {guests}"

    parts = []
    guest_name = (state.get("guest_name") or "").strip()
    if guest_name:
        parts.append(f"{guest_name} ji 🙏")
    if line:
        parts.append(line)
    return "\n".join(parts)


def _card_party(state: dict) -> tuple[int, int]:
    """
    The (adults, children) the card is being built for.

    `effective_*` wins over the raw counts, matching room_searcher — a party
    whose size staff overrode must not be described by the number the guest
    first typed. One definition, because the header and the ROOMS selection
    would otherwise be computed from two different parties.
    """
    adults = state.get("effective_adults")
    if adults is None:
        adults = state.get("adults") or 2
    children = state.get("effective_children")
    if children is None:
        children = state.get("children") or 0
    return int(adults), int(children)


def _meal_footer(options: list[dict]) -> str:
    """
    The card's meal line, named from the plans the combinations were priced on.

    This was the constant `MEAL_FOOTER_EP` printed regardless of what the
    allocator had actually chosen — so a guest who asked for breakfast got
    CP-priced combinations under a "Room only" footer, the card contradicting
    itself about what was being sold. `room_searcher` now defaults the basis to
    EP, which is what makes the common case true; this makes the rest of them
    honest rather than assuming the basis it was written for.

    Falls back to the EP wording when no option carries a usable plan name,
    which is the same string the constant held, so a caller that passes
    plan-less options sees no change.
    """
    plans = {
        _friendly_rate_plan_name(str(room.get("rate_plan_name") or ""))
        for opt in options
        for room in (opt.get("rooms") or [])
    }
    plans.discard("")
    if not plans or plans == {"Room Only"}:
        return MEAL_FOOTER_EP
    return f"{' + '.join(sorted(plans))} · GST included"


def _pretty_room(name: str) -> str:
    """
    The room's own name, title-cased — deliberately *not* `clean_room_type_name`.

    That function is a matching normaliser, not a display one: it rewrites the
    six hotel-111111 names to shorter forms and, on its last line, deletes the
    words " NON AC" and " KING BED". Reusing it here printed a non-AC room as
    "Standard Room" — beside an AC room, on a card the guest reached by telling
    us which class they wanted. The one word that decided their choice was the
    one word dropped.

    "AC" is upper-cased as a word so a title-cased name does not read "Ac".
    """
    words = []
    for word in " ".join(str(name or "Room").split()).title().split():
        words.append("AC" if word.lower() == "ac" else word)
    return " ".join(words).replace("Non AC", "Non-AC")


def _packing_title(packing, beds_in_title: bool = True) -> str:
    """
    `2 x Deluxe Family Room`, or `1 x Deluxe + 1 x Premier` when the set is mixed.

    The " Room" suffix is dropped only for mixed sets, which is the same rule
    `_option_title` uses and for the same reason: one type reads better with the
    word, and three of them in a row read as noise.

    A mattress is named as its own thing — `1 x Deluxe Room King Bed + 1 Extra
    Bed` — and never folded into the room. `Extra Bed` is capitalised to match
    the room names it sits beside.

    `beds_in_title=False` drops it, and the caller sets that when the bed is
    priced on the line beneath instead. The two are one decision made in one
    place: naming the bed here AND quoting it there would sell the guest the
    same mattress twice, while naming it in neither would hide it. The title
    carries it only when there is no charge to show — a hotel that prices its
    bed at zero — because then the note has no price to print and the bed would
    otherwise disappear from the row entirely.
    """
    single = len(packing.counts) == 1
    parts = []
    for name, count in packing.counts:
        pretty = _pretty_room(name)
        if not single:
            pretty = pretty.replace(" Room", "").strip()
        parts.append(f"{count} x {pretty}")
    if packing.bed_count and beds_in_title:
        beds = "Extra Bed" if packing.bed_count == 1 else "Extra Beds"
        parts.append(f"{packing.bed_count} {beds}")
    return " + ".join(parts)


def _bed_charge(packing, price_by_name: dict[str, float]) -> float:
    """
    What the option charged for the mattresses in this set, per night.

    Derived rather than carried alongside the packing, because deriving it makes
    the line add up **by construction** — whatever the allocator charged is
    exactly the difference between the total and the rooms — so the card cannot
    print a breakdown that fails to reach its own total. That was the fault that
    got the old two-section card deleted, and a row saying "₹2,250/night" under
    a "₹5,700 total" would be the same bug again.

    Zero is a real answer and means the hotel prices this room's bed at nothing;
    the callers use it to decide whether there is a price worth printing.
    """
    rooms_total = sum(
        price_by_name.get(name, 0.0) * count for name, count in packing.counts
    )
    return round(packing.total_per_night - rooms_total, 2)


def _bed_note(base_seats: int | None, bed_charge: float) -> str:
    """
    The mattress line: the room sleeps this many, and a bed costs this much.

    the product's wording, asked for on 2026-09-24 and given here almost verbatim:
    *"Hey if you can adjust your 3 people in this room / bed for 2 thats good
    BUT If you need an extra bed / matress that can be added on too for blabla
    price."* The point of it is that the bed reads as something the guest may
    choose, not something bundled into a price they never agreed to.

    `base_seats` is the room's own `base_adult_occupancy` — the number the
    card's ROOMS list prints as "max N", and the number the guest has to fit
    into if they decline the mattress. It is None on a set of several room
    types, where there is no single capacity to quote, and the line then states
    the price without claiming a berth count.

    "Adjust" is on every note that can quote a capacity, including a party
    larger than the beds — 3 guests in a room that sleeps 2, which is the product's
    own example, gets it. The gate is his and it is not `base_adults`: it is the
    room's `max_adult_occupancy` covering the party, which is the hotel saying
    the room takes that many people. Every row on this card has already cleared
    it (`min_cost_packing` seats the party or the row is not drawn), so there is
    nothing left to test here.

    The stay total that used to close this line — `(₹5,700 total with it)` — is
    gone, on product instruction of 2026-09-24 that the rows carry a per-night
    rate and nothing else. It was redundant besides: the guest reaches the stay
    figure from this rate and the `(2N)` already in the header.
    """
    # A statement, not an invitation. This read "adjust, or add an extra bed for
    # Rs 600/night" until 2026-09-24, which is an offer to *buy* something the
    # row's own allocator option already contains — and guests answered it. One
    # typed "without extra bed" before ever seeing the card, `understand` set
    # `no_extra_bed`, and `quote_generator` waived the mattress that was seating
    # their third guest. The sentence now says what the Rs 600 is; the choice is
    # put to the guest once, after they pick a row, by the `need_extra_bed` turn.
    if base_seats:
        return (
            f"🛏️ Sleeps {base_seats} — ₹{bed_charge:,.0f}/night extra bed for the "
            "guest above that"
        )
    return f"🛏️ ₹{bed_charge:,.0f}/night extra bed"


def _seat_lookup(seats_by_name: dict[str, int] | None, name: str) -> int | None:
    """
    A room's `base_adult_occupancy`, however the two sides happen to be spelled.

    `seats_by_name` is keyed by the catalogue's name, upper-cased and stripped
    (`room_service.py:1968`), while `packing.counts` carries the allocator's own
    spelling of the same room. A bare `.get(name)` therefore misses whenever the
    two disagree on case, and the note silently loses its berth count -- which is
    how a capacity line quietly becomes a price line. Exact hit first, then the
    normalised one.
    """
    if not seats_by_name:
        return None
    if name in seats_by_name:
        return seats_by_name[name] or None
    return seats_by_name.get(str(name).strip().upper()) or None


def _no_extra_bed_note(guests: int) -> str:
    """
    "No extra bed for N people" -- the party fits, without a mattress.

    product wording of 2026-09-29, verbatim: *"if customer wants 3 people to
    adjust WITHOUT an extra bed thats fine, just MENTION in the QUOATATION and
    CONFRIMATION card, wherever this case is true that 'No extra bed for N
    people'."* Sample Hotel's Deluxe Room King Bed is the case he named:
    `base_adults` 2, `max_adults` 3, so three guests sleep there with no bed
    added and nothing extra charged.

    The card already implies this -- it prints a rate and no mattress line --
    but implying it is not stating it. A guest who asked for three people and
    sees no bed line cannot tell whether the bot understood, quietly dropped a
    charge, or is about to bill them at check-in. The line answers that.

    Only ever called when no mattress was charged AND the party outnumbers the
    room's own berths, so it cannot appear on a row that did buy a bed.

    Always plural. `_seat_lookup` maps a stored 0 to None, so a room that
    reports berths reports at least one -- and the caller only reaches this line
    when the party exceeds that, which makes 1 unreachable.
    """
    return f"🛏️ No extra bed for {guests} people"


def _packing_price(
    packing,
    price_by_name: dict[str, float],
    seats_by_name: dict[str, int] | None = None,
    guests: int = 0,
) -> str:
    """
    `₹2,250/night`, or `(₹3,250 + ₹2,250) = ₹5,500/night` for a mixed set.

    **Per night only.** This printed a stay total until 2026-09-24, when product
    asked for the rate and nothing else on the option rows; the header already
    says `1-3 Oct (2N)`, so the total was the nights multiplied out one line
    under a rate the guest can multiply themselves. Dropping it also removed the
    only reason this function needed the nights or the allocator's stay figure,
    so both parameters went with it — a total that is never printed cannot
    disagree with the rates above it.

    The per-room breakdown is the point of the card: `2 x Deluxe Family Room —
    ₹7,760/night` makes the guest take the figure on trust, and the earlier card
    did exactly that. Naming each room's own rate is what lets them check it,
    and a mixed set still shows the sum, so the arithmetic is visible even
    though the stay total is not.

    A set carrying a mattress prints two lines: the room, at what the room
    costs, and beneath it the bed at what the bed costs. Until 2026-09-24 it
    printed one, `₹2,250 + ₹600 extra bed = ₹2,850/night`, which tells the guest
    the bed is part of the set. It is not — see `_bed_note`.

    A bedded set's room line is `total_per_night - bed_charge`, never the
    option's own figure, because that figure *includes* the mattress and would
    show the bed in the room price and again on the note beneath it. On a set
    with no bed the option's own figure is printed unchanged.

    Returns a string that may contain one embedded newline, already indented for
    the second line, so the caller's own indent applies to the first and the two
    line up under the row number.
    """
    bed_charge = _bed_charge(packing, price_by_name)

    if packing.room_count <= 1:
        name = packing.counts[0][0]
        line = f"₹{price_by_name.get(name, 0.0):,.0f}/night"
        if packing.bed_count and bed_charge > 0:
            seats = _seat_lookup(seats_by_name, name)
            return f"{line}\n   {_bed_note(seats, bed_charge)}"
        # No mattress is charged on this row. If the party still outnumbers the
        # berths the room actually has, the guests are being fitted in without
        # one and the card says so. `guests` is 0 when the caller had no party
        # size to give, which reads as "unknown" and prints nothing -- a row
        # that quietly claims the wrong number is worse than a silent one.
        #
        # Single-room rows only. On a multi-room set there is no single room to
        # attribute the count to, and "No extra bed for 6 people" across two
        # rooms would be read as per-room.
        seats = _seat_lookup(seats_by_name, name)
        if seats and guests > seats:
            return f"{line}\n   {_no_extra_bed_note(guests)}"
        return line

    units: list[str] = []
    for name, count in packing.counts:
        units.extend([f"₹{price_by_name.get(name, 0.0):,.0f}"] * count)
    if packing.bed_count and bed_charge > 0:
        rooms_nightly = round(packing.total_per_night - bed_charge, 2)
        return (
            f"({' + '.join(units)}) = ₹{rooms_nightly:,.0f}/night\n"
            f"   {_bed_note(None, bed_charge)}"
        )
    return f"({' + '.join(units)}) = ₹{packing.total_per_night:,.0f}/night"


def _rooms_section(
    options: list[dict], rooms: list | None, state: dict, rows: list[dict] | None = None
) -> list[str]:
    """
    The `ROOMS` list — one line per room type, at the price the combinations use.

    This is the half of the card that was deleted and is now restored, and the
    reason it was deleted is the reason it is built this way: the old list was
    drawn from the catalogue's **static** rates while the combinations beneath
    it were priced from the **calendar**, so one card quoted Deluxe at Rs 2,550
    and charged Rs 3,000.

    Both halves are therefore sourced from `_card_candidates`, which reads its
    prices out of the allocator's own options. That is a second call to a pure
    function on the same inputs rather than a shared intermediate, so the two
    cannot disagree — the alternative was widening `card_rows`' return shape,
    which nine tests and two probes unpack.

    Rooms the packer refuses (no base occupancy, or no calendar price) are left
    off rather than shown at Rs 0. They cannot appear in a combination either,
    so listing them would advertise something the card cannot sell.

    The capacity printed is `base_adults` — the figure the packer fills to —
    and not the booking system's `max_occupancy`. product confirmed the
    distinction on 2026-09-19 ("Capacity shown: 4 / Capacity in system: 5"):
    this card adds a room rather than a bed, so a maximum the hotel will never
    actually sell to one party is a number the guest cannot use.

    Filtered to the guest's own AC class, and never wider. Until 2026-09-24 it
    was also widened by every room the combinations quoted, because
    `cheapest_and_comfy` made the comfortable row AC even for a guest who asked
    for non-AC, and a guest looking at "1 x Deluxe Family" with no Deluxe
    Family rate above it was the fault that got this list deleted the first
    time. The rows no longer cross classes, so the union has nothing left to
    add; it is kept as a guard so a row can never cite a rate this list did not
    print.

    Unstated means AC, the same rule the rows follow. It used to switch the
    filter off entirely, which put every room of both classes on the card.
    """
    candidates, _ = _card_candidates(options, rooms)
    if not candidates:
        return []

    preference = ac_preference_from_state(state.get("ac_preference"))
    # Unstated means AC — keep this in step with `room_packing._card_class_pool`.
    want_ac = preference != "NON_AC"

    quoted: set[str] = set()
    for row in rows or []:
        packing = row.get("packing")
        if packing is not None:
            quoted.update(name for name, _ in packing.counts)
            continue
        for r in row.get("option", {}).get("rooms") or []:
            name = str(r.get("room_name") or "").strip().upper()
            if name:
                quoted.add(name)

    # One line per *room type*, at its own rate and its own capacity. A room the
    # allocator also prices with a mattress yields a second candidate of the
    # same name, dearer by the mattress — right for the packer, wrong for this
    # list, which is a catalogue of what the hotel has, not of what one party
    # would be sold. Keeping the cheapest candidate per name keeps the one with
    # no mattress on it, since a mattress only ever adds cost.
    #
    # Either candidate states the same ceiling, because a mattress never
    # changes the seats (`_card_candidates`). This read `base_adults - beds`
    # until 2026-09-25, when `base_adults` still counted the mattress among the
    # room's seats and subtracting it recovered the real capacity. It no longer
    # does, so subtracting would now understate the room by one berth on any
    # card that priced a mattress.
    best_by_name: dict[str, Any] = {}
    for c in candidates:
        if not c.packable():
            continue
        if want_ac is not None and not (c.is_ac is want_ac or c.name in quoted):
            continue
        current = best_by_name.get(c.name)
        if current is None or (c.price, c.name) < (current.price, current.name):
            best_by_name[c.name] = c

    shown = sorted(best_by_name.values(), key=lambda c: (c.price, c.name))
    return [
        f"{_pretty_room(c.name)} — ₹{c.price:,.0f}/night · max {c.base_adults}"
        for c in shown
    ]


def _card_candidates(
    options: list[dict], rooms: list | None, allow_beds: bool = True
) -> tuple[list, dict[str, float]]:
    """
    One RoomCandidate per room type the allocator priced, plus its rate by name.

    The two halves come from two places on purpose. **Price** comes from the
    option, because that is the calendar-resolved rate for the guest's own
    dates — the catalogue's static price is not what they will be charged.
    **Base occupancy and AC class** come from the catalogue document, because
    the option carries neither.

    **The seating capacity** is the catalogue's `max_adults` — the hotel's own
    adult ceiling for the room — and **the mattress** is read off the option's
    `extra_adult_charge`, which is non-zero only where the guest actually asked
    for one. Both were `base_adults + (assigned_adults - base_adults)` until
    2026-09-25, which said a room seats only what the room rate covers plus one
    berth per mattress. A Deluxe King (base 2 / max 3) therefore drew as
    "Sleeps 2" without a bed, so a party of three who declined the mattress
    looked unseatable and was answered with two rooms. See `distribute_guests`.

    The mattress is deliberately not re-derived from the hotel's policy here:
    the allocator is the one that read `extra_bed_allowed`, and a second opinion
    could quote a mattress the final quote refuses.

    Two candidates can share a name — one Deluxe with a mattress and one without
    — so the de-duplication is keyed on the pair. Keyed on the name alone, as it
    was, the bedded one would silently lose to whichever came first.

    A room whose catalogue document cannot be resolved is skipped rather than
    defaulted. Guessing its capacity would let the packer quote a set the hotel
    cannot seat, and a missing room is a visible gap while a wrong capacity is
    an invisible one.
    """
    by_name: dict[str, Any] = {}
    for room in rooms or []:
        nm = str(getattr(room, "name", "") or "").strip().upper()
        if nm:
            by_name[nm] = room

    candidates: list[RoomCandidate] = []
    price_by_name: dict[str, float] = {}
    seen: set[tuple[str, int]] = set()
    for opt in options:
        for r in opt.get("rooms") or []:
            nm = str(r.get("room_name") or "").strip().upper()
            if not nm:
                continue
            doc = by_name.get(nm)
            if doc is None:
                continue
            occ = getattr(doc, "occupancy", None)
            base = int(getattr(occ, "base_adults", 0) or 0)
            # How many adults this room seats — the hotel's own ceiling.
            #
            # Until 2026-09-25 this was `base + beds`, where `beds` was derived
            # from `assigned_adults - base_adults`. That made the mattress the
            # source of the room's seating: a Deluxe King (base 2 / max 3)
            # advertised "Sleeps 2" on its bed-less row, so a party of three who
            # declined the mattress appeared not to fit and the packer answered
            # with two rooms. `max_adult_occupancy` is 3 — the room seats three.
            # The mattress seats nobody the room was not already seating.
            #
            # `max(base, …)` and not the ceiling alone: the schema defaults
            # `max_adults` to 3, so a room whose document states base 4 and no
            # ceiling would arrive here capped at 3 — a ceiling *below* the
            # floor it is supposed to raise. A ceiling under the base is not a
            # ceiling, so the base is the floor of this read.
            #
            # Shared with the allocator's per-room-split branch and with
            # `RoomType.adult_seats`, which is the same expression. It moved
            # there on 2026-09-25 because a third copy was about to be written,
            # and a seat count that drifts between two readers is exactly how
            # the ₹600 mattress defect happened.
            seats = doc.adult_seats()
            # The room's *second* ceiling — how many people the hotel will put
            # in it, adults and children together. `seats` above is how many
            # adults it sleeps, and the two are different numbers: a Deluxe Room
            # Four Bed seats 5 and holds 6. The packer needs both, because a
            # party of five adults and a free child is one room's worth of
            # people and two rooms' worth of berths. Read from the same
            # catalogue document as `seats`, for the same reason.
            heads = int(getattr(occ, "max_occupancy", 0) or 0)
            # The mattress is priced, never inferred. Reading it off occupancy
            # is what put a bed on the card for every third adult.
            #
            # `extra_bed_count` is the pillow itself; `extra_adult_charge` is
            # only money, and the two parted company when a CP / MAP guest
            # started being charged for an adult above base occupancy without
            # ever being offered a mattress. Reading the charge as a bed count
            # would lay a bed into the card for that guest and make the row
            # unskippable for one who refused.
            #
            # `None` is "nobody said" — a selection written before the field —
            # and keeps the old charge-based reading, which is what that
            # selection was built against.
            _bed_count = r.get("extra_bed_count")
            beds = (
                int(_bed_count)
                if _bed_count is not None
                else (1 if float(r.get("extra_adult_charge") or 0) > 0 else 0)
            )
            if beds and not allow_beds:
                # The guest refused the mattress. Skip the bedded candidate and
                # let the packer choose among the rooms' own seats — which now
                # include the third adult, so the same room still seats them.
                continue
            if (nm, beds) in seen:
                continue
            seen.add((nm, beds))
            room_rate = float(r.get("price_per_night") or 0)
            price_by_name[nm] = room_rate
            candidates.append(
                RoomCandidate(
                    name=nm,
                    base_adults=seats,
                    price=room_rate
                    + float(r.get("extra_adult_charge") or 0)
                    + float(r.get("extra_child_charge") or 0),
                    is_ac=room_matches_feature(doc, "AC"),
                    beds=beds,
                    max_occupancy=heads,
                )
            )
    return candidates, price_by_name


def _packing_key(packing) -> tuple[tuple[str, ...], float, int]:
    """
    A set's identity: the rooms in it, and what it costs.

    The same key `_match_option` matches on, so two packings that would draw the
    same line are two packings with the same key. `_render_rows` prints the
    number, the room list and the price and nothing else, so anything this key
    ignores is a difference the guest cannot see.

    The bed count is in the key because it *is* something the guest sees — the
    row reads "1 x Deluxe Room King Bed + 1 Extra Bed" — and a mattress always
    moves the price anyway, so this only makes explicit what the total already
    kept apart.
    """
    return (
        tuple(sorted(n for name, count in packing.counts for n in [name] * count)),
        round(float(packing.total_per_night), 2),
        packing.bed_count,
    )


def _match_option(options: list[dict], packing, tol: float = 1.0) -> int | None:
    """
    Index of the allocator option that *is* this set, or None.

    The card decides which sets to show; the allocator decides what a set is
    made of and what it costs. Matching them back up is what keeps selection and
    booking on the existing contract — a reply of "1" resolves to "option N",
    and `validator._resolve_room_selection` takes it from there. A set the
    packer chose that the allocator never offered has no index and is not drawn,
    which is the safe direction: a row the guest cannot select is worse than a
    row they never saw.
    """
    want = sorted(n for name, count in packing.counts for n in [name] * count)
    for idx, opt in enumerate(options):
        got = sorted(str(r.get("room_name") or "").strip().upper() for r in (opt.get("rooms") or []))
        if got != want:
            continue
        total = float(
            opt.get("total_price_per_night_inclusive") or opt.get("total_price_per_night") or 0
        )
        if abs(total - packing.total_per_night) < tol:
            return idx
    return None


def card_rows(
    options: list[dict] | None, rooms: list | None, state: dict | None = None
) -> list[dict]:
    """
    The priced rows the card prints: cheapest, comfy, then a real step up.

    Up to three, and often two. The third is `upgraded_row`'s answer and it is
    None whenever the hotel has nothing above Comfortable — a single-grade hotel
    would otherwise print the same set twice under two labels, which is the
    complaint that produced this card.

    The rows are numbered 1, 2, 3 and carry no tier names. The client asked
    for a third row by naming its tiers ("Economical, Comfort, Upgraded"), and
    both card templates product wrote drop the names and let the order say
    it — row 1 is the cheapest because the packer puts it first, and a label on
    the line is one more thing to read before the price. They were lettered
    A, B, C on that template until 2026-09-23, when product asked for numbers to
    match the "e.g. *Option 1*" prompts elsewhere on the card.

    Each row is `{"packing", "prices", "option_index", "option"}` — the option
    dict itself, not just its position, because the caller assembling the card
    hands back a list of its own and an index into a list nobody kept is a
    pointer to nothing. A row without an `option_index` is one the allocator
    cannot supply and is dropped — see `_match_option`. When neither survives,
    the caller falls back to the allocator's own first options so a card always
    renders.
    """
    state = state or {}
    options = [_normalize_option(o) for o in (options or [])]
    if not options:
        return []

    adults, children = _card_party(state)
    preference = ac_preference_from_state(state.get("ac_preference"))
    # The party the packer fills, and the party the room has to hold. Two
    # numbers, because a child the hotel does not charge for shares the parents'
    # bed: it needs no berth of its own, and it is still a person in the room.
    #
    # A paid child takes a berth like anyone else, so `paid_child_count` is the
    # only part of `children` that moves the first number. It is clamped to
    # `children` because it is written by `validator` from the ages it resolved
    # and a stale one must not be able to describe more children than the party
    # has.
    paid_children = int(state.get("paid_child_count") or 0)
    if not get_settings().FREE_CHILD_NEEDS_NO_BERTH:
        # The pre-2026-09-29 rule, kept reachable for a one-push rollback: every
        # child takes a berth, so the two parties are the same number.
        paid_children = children
    paid_children = max(0, min(paid_children, children))
    party = adults + paid_children
    head_party = adults + children

    # A guest who has refused the mattress is shown only sets that seat them
    # without one — that is what their answer to the bed question bought, and the
    # alternative (this row, Rs 600 cheaper) is not a set any room can offer.
    #
    # The fallback is not decoration. A hotel whose rooms only seat their parties
    # with a mattress would otherwise render an empty card, and a blank screen is
    # worse than the bedded row with the question already asked about it. Falling
    # back re-admits the bed rows and leaves the footnote to disclose the charge,
    # which is exactly the behaviour every guest had before this existed.
    refused_bed = bool(state.get("no_extra_bed"))
    candidates, price_by_name = _card_candidates(options, rooms, allow_beds=not refused_bed)
    cheapest, comfy = (
        cheapest_and_comfy(party, candidates, preference, head_party)
        if candidates else (None, None)
    )
    if refused_bed and cheapest is None and comfy is None:
        logger.info(
            "ROOM_CARD | no bed-less set seats %s — drawing the bedded rows instead",
            party,
        )
        candidates, price_by_name = _card_candidates(options, rooms, allow_beds=True)
    if not candidates:
        return []

    cheapest, comfy = cheapest_and_comfy(party, candidates, preference, head_party)
    upgraded = upgraded_row(party, candidates, comfy, preference, head_party)

    # ── "Show both": three rows, one class each ─────────────────────────────
    # A guest who tapped **Show both** used to be served the ordinary one-class
    # card, because `ac_preference_from_state` folds an explicit "both" in with
    # "never said". On 2026-10-28 a party of 8 adults and 4 children was offered
    # two Non-AC sets and nothing else; the AC rooms were in the catalogue and on
    # the ROOMS list above the rows, and not one of them could be chosen. Live
    # report: "the only and only 2 options for non ac is being sent".
    #
    # Gated, and the gate defaults ON because it cannot be flipped from the
    # account that deploys (see `settings.ROOM_CARD_SHOW_BOTH_ENABLED`). Turning
    # it off restores the one-class card exactly.
    #
    # An empty answer means one class cannot seat this party — keep the card we
    # already built rather than print a two-class card with a class missing.
    show_both = get_settings().ROOM_CARD_SHOW_BOTH_ENABLED and wants_both_classes(
        state.get("ac_preference")
    )
    if show_both:
        both = both_class_rows(party, candidates, head_party)
        if both:
            cheapest, comfy, upgraded = both
        else:
            show_both = False

    rows: list[dict] = []
    drawn: set[tuple] = set()
    for packing in (cheapest, comfy, upgraded):
        if packing is None:
            continue
        # Three tiers, and the packer is free to answer the same set for two of
        # them — it does exactly that whenever the candidate pool is thin. With
        # a single room type on offer, "cheapest for five" and "comfortable for
        # five" are the same two rooms, and the card drew them as A and B: two
        # letters, one price, one set, presented as a choice. The live
        # transcript's twin rows are this, not the fallback. A repeat is not a
        # tier — drop it and let `upgraded_row` speak, or let the card run one
        # row shorter.
        key = _packing_key(packing)
        if key in drawn:
            logger.info(
                "ROOM_CARD | tier repeats a set already drawn, dropped | set=%s",
                _packing_title(packing),
            )
            continue
        index = _match_option(options, packing)
        if index is None:
            logger.info(
                "ROOM_CARD | packing not offered by the allocator, dropped | set=%s",
                _packing_title(packing),
            )
            continue
        drawn.add(key)
        rows.append(
            {
                "packing": packing,
                "prices": price_by_name,
                "option_index": index,
                "option": options[index],
            }
        )
    if show_both:
        rows = _ensure_both_classes(rows, options, candidates)
    return rows


def render_room_card_text(rooms: list | None, options: list[dict] | None, state: dict | None = None) -> str:
    """
    The card: a ROOMS list, lettered combinations, and a build-your-own line.

    This is the two-section shape restored, and the earlier attempt at it is
    worth knowing about because two of its three faults are what the current
    implementation is built to avoid:

      * The list and the combinations were drawn from two different price books
        — the catalogue's static rates against the calendar-resolved ones — so
        one card quoted Deluxe at Rs 2,550 and charged Rs 3,000. Both sections
        are now priced from `_card_candidates`, i.e. from the allocator's own
        options.
      * The list was capped at four, so a party of three was offered "1 x Family
        Room" in the combinations while the Family Room's rate appeared nowhere
        on the card. The list is uncapped now, and it is filtered by the same AC
        class as the combinations, so the two halves describe one set of rooms.
      * It was, at six rooms and four combinations, "too big". That one is a
        real trade and it is why this card carries no tier names and no
        per-room split for single-room sets: the lines that earn their place are
        the ones the guest is choosing between.

    Capacity is printed as `base_adults`, the figure the packer fills to, not
    the booking system's `max_occupancy`. product settled this on 2026-09-19
    ("Capacity shown: 4 / Capacity in system: 5"): this card adds a room rather
    than a bed, so a maximum the hotel will never sell to one party is a number
    the guest cannot use. Confirmed with product on 2026-09-21.

    A row the allocator cannot supply is dropped rather than invented; if that
    leaves nothing, the allocator's own first options are shown so the guest
    still gets a card.
    """
    state = state or {}
    options = [_normalize_option(o) for o in (options or [])]
    return _render_rows(rows_for_card(options, rooms, state), options, state, rooms)


def rows_for_card(
    options: list[dict], rooms: list | None, state: dict | None = None
) -> list[dict]:
    """
    `card_rows`, backfilled up to ROOM_CARD_MAX_COMBOS (3) with distinct suitable options
    from the allocator, or fallback if nothing matched.

    Ensures that multi-person parties (e.g. 8 guests) get up to three closest, suitable
    room combinations (checking capacity and max occupancy), rather than collapsing to 1.
    If only 1 or 2 options are genuinely suitable, returns only those available options.
    """
    rows = list(card_rows(options, rooms, state) or [])
    if not options:
        return rows

    target_count = ROOM_CARD_MAX_COMBOS
    if len(rows) < target_count:
        def _set_key(opt: dict) -> tuple[str, ...]:
            return tuple(
                sorted(
                    str(r.get("room_name") or "").strip().upper()
                    for r in (opt.get("rooms") or [])
                )
            )

        seen = {_set_key(r["option"]) for r in rows if r.get("option")}
        for idx, raw in enumerate(options):
            opt = _normalize_option(raw)
            key = _set_key(opt)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append({"packing": None, "prices": {}, "option_index": idx, "option": opt})
            if len(rows) == target_count:
                break

    return rows


def _distinct_fallback_rows(options: list[dict], want: int = ROOM_CARD_MAX_COMBOS) -> list[dict]:
    """
    The allocator's own first options — never the same set twice.

    The fallback exists so a guest always gets a card, and it is only ever
    reached when the packer's sets matched nothing. That is exactly the state a
    one-room-type catalogue leaves the allocator in: when a room-type preference
    narrows the pool to a single room type, `allocate_rooms` can only emit one
    shape of set, and the first two entries of its pool are the SAME set at the
    SAME price — often the same `option_id`, because the engine numbers as it
    goes. The card then drew "A. 3 x Deluxe Room King Bed" and "B. 3 x Deluxe
    Room King Bed" as two different choices, which is the live complaint this
    whole card rewrite started from.

    Two identical rows are not one bad row out of two: they are a choice the
    guest cannot make, printed as if it were one. So the fallback walks the pool
    and keeps the first `want` DISTINCT sets, counted the way `_match_option`
    counts them — the sorted room names — and returns fewer rows, or none, when
    the pool has nothing distinct left to say. A short card is honest; a padded
    one is not.
    """
    seen: set[tuple[str, ...]] = set()
    rows: list[dict] = []
    for idx, raw in enumerate(options or []):
        opt = _normalize_option(raw)
        key = tuple(
            sorted(
                str(r.get("room_name") or "").strip().upper()
                for r in (opt.get("rooms") or [])
            )
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({"packing": None, "prices": {}, "option_index": idx, "option": opt})
        if len(rows) == want:
            break
    return rows


def _option_class(opt: dict, ac_by_name: dict[str, bool]) -> bool | None:
    """
    True for an all-AC set, False for an all-Non-AC one, None for mixed/unknown.

    The class is read off the option's own room names, through the same
    catalogue map `_card_candidates` built, so a row's class is a fact about
    what will be booked rather than about which bucket the packer drew it from.
    """
    flags = set()
    for room in opt.get("rooms") or []:
        nm = str(room.get("room_name") or "").strip().upper()
        if nm in ac_by_name:
            flags.add(ac_by_name[nm])
    if len(flags) != 1:
        return None
    return flags.pop()


def _ensure_both_classes(rows: list[dict], options: list[dict], candidates: list) -> list[dict]:
    """
    A "show both" card that came back one-class gets the missing class back.

    Why this exists, and why it is here rather than in the packer: on
    2026-10-28 a party of twelve tapped **Show both** and the card still showed
    Non-AC rows only, after the packer-side fix had shipped and been verified on
    a smaller party. The log line that explains it is

        ROOM_CARD | packing not offered by the allocator, dropped | set=2 x Deluxe Room Four Bed

    — and the allocator HAD offered that set, at position three of its own
    ranked list for that turn. `_match_option` compares the room names first
    (they matched) and then the nightly total with a Rs 1 tolerance (it did
    not). The packer's total is summed from `_card_candidates` prices; the
    option's is the allocator's inclusive one; the two can differ by hundreds
    of rupees for reasons that have nothing to do with whether the set exists.
    So the AC row was real and was thrown away by a price check.

    Rather than widen that tolerance — which risks drawing a row that resolves
    to a different price than the one printed — this fills the gap from the
    other side: pick the class the card is missing straight from the
    allocator's own options and draw it with `packing=None`, the same shape
    `_distinct_fallback_rows` uses. Those rows print the option's own price, so
    there is nothing left to mismatch.

    Never invents a class. If the allocator offered nothing of the missing
    class, the card stays one-class and `_missing_class_note` explains why.
    """
    ac_by_name = {
        str(getattr(c, "name", "") or "").strip().upper(): bool(getattr(c, "is_ac", False))
        for c in candidates
    }

    def _names(opt: dict) -> tuple[str, ...]:
        return tuple(
            sorted(str(r.get("room_name") or "").strip().upper()
                   for r in (opt.get("rooms") or []))
        )

    seen_sets = {_names(row["option"]) for row in rows if row.get("option")}
    have = {_option_class(row["option"], ac_by_name) for row in rows if row.get("option")}
    have.discard(None)

    added = False
    for want in (False, True):  # Non-AC first, matching the three-row design
        if want in have:
            continue
        for idx, opt in enumerate(options):
            if _option_class(opt, ac_by_name) is not want:
                continue
            if _names(opt) in seen_sets:
                continue
            rows.append(
                {"packing": None, "prices": {}, "option_index": idx, "option": opt}
            )
            seen_sets.add(_names(opt))
            have.add(want)
            added = True
            logger.info(
                "ROOM_CARD | show-both card was missing a class,"
                " drawn from the allocator's own pool | class=%s option_index=%d",
                "AC" if want else "NON_AC",
                idx,
            )
            break

    if not added:
        return rows
    # Stable: only the appended row moves, into its class slot.
    rows = sorted(
        rows,
        key=lambda row: 0 if _option_class(row["option"], ac_by_name) is False else 1,
    )
    # The card has three priced rows and the resolver reads `4` as build-your-own
    # (`ROOM_CARD_MAX_COMBOS`). Trimming from the end drops the AC step-up first,
    # which is the right row to lose: a class the guest asked for beats a nicer
    # set of the class they already had.
    return rows[:ROOM_CARD_MAX_COMBOS]


def _inventory_candidates(rooms: list | None) -> list[RoomCandidate]:
    if not rooms:
        return []
    candidates: list[RoomCandidate] = []
    seen: set[str] = set()
    for doc in rooms:
        nm = str(getattr(doc, "name", "") or "").strip().upper()
        if not nm or nm in seen:
            continue
        seen.add(nm)
        occ = getattr(doc, "occupancy", None)
        base = int(getattr(occ, "base_adults", 0) or 0)
        seats = doc.adult_seats() if hasattr(doc, "adult_seats") else max(base, int(getattr(occ, "max_adults", 0) or 0))
        heads = int(getattr(occ, "max_occupancy", 0) or 0)
        price = float(getattr(doc, "base_price", 0) or getattr(doc, "price", 0) or 1000)
        candidates.append(
            RoomCandidate(
                name=nm,
                base_adults=seats,
                price=price,
                is_ac=room_matches_feature(doc, "AC"),
                beds=0,
                max_occupancy=heads,
            )
        )
    return candidates


def _missing_class_note(
    options: list[dict], rooms: list | None, state: dict
) -> str | None:
    """
    The sentence a "show both" guest gets when only one class can seat them.

    Why it exists: on 2026-10-28 a party of twelve tapped **Show both**, no
    Non-AC room was free for their dates, and the card came back with AC-only
    rows and no explanation. To the guest — and to the staff member who reported
    it — that is indistinguishable from the bot ignoring the request. It cost a
    test round-trip and a wrong diagnosis before the logs showed the inventory
    was the reason, not the card.

    So the card says so. This asks the packer whether each class can seat the
    party (`class_can_seat`), which is a fact about the hotel, rather than
    looking at which rows were drawn, which a rendering bug could falsify.

    Silent in three cases, all deliberate:

      * the flag is off, so the card is the old one-class card;
      * the guest never asked for both — nothing was promised, so nothing is
        owed, and the commonest card in the bot stays the same length;
      * *neither* class can seat them. That is "no rooms", which the caller
        answers with a different card entirely; a note claiming AC exists while
        the card shows nothing would be worse than silence.

    No question is asked. "Do you want the AC ones instead?" would need a reply
    the bot can resolve, and the card already carries one — `change`.
    """
    if not get_settings().ROOM_CARD_SHOW_BOTH_ENABLED:
        return None
    if not wants_both_classes(state.get("ac_preference")):
        return None
    try:
        candidates, _ = _card_candidates(options, rooms)
    except Exception:  # a note is never worth failing a card over
        logger.warning("ROOM_CARD | class note skipped: candidates unavailable")
        return None
    if not candidates:
        return None

    adults, children = _card_party(state)
    paid_children = int(state.get("paid_child_count") or 0)
    if not get_settings().FREE_CHILD_NEEDS_NO_BERTH:
        paid_children = children
    paid_children = max(0, min(paid_children, children))
    party = adults + paid_children
    head_party = adults + children

    inv_cands = _inventory_candidates(rooms) if rooms else candidates
    ac_ok = class_can_seat(party, inv_cands, True, head_party)
    non_ac_ok = class_can_seat(party, inv_cands, False, head_party)
    if ac_ok and non_ac_ok:
        return None
    if not ac_ok and not non_ac_ok:
        return None

    missing, shown = ("AC", "Non-AC") if not ac_ok else ("Non-AC", "AC")
    return (
        f"_{missing} rooms are not available for these dates. "
        f"Every option above is {shown} — reply *change* to adjust._"
    )


def _render_rows(
    rows: list[dict], options: list[dict], state: dict, rooms: list | None = None
) -> str:
    """
    Draw the card: header, ROOMS, the numbered combinations, footer, reply line.

    Shared by the renderer and the card builder so the text and the `options`
    handed back in the card's `data` come from one computation — two calls to
    the packer would be two chances to disagree about what the guest was shown.

    `rooms` is the catalogue, and it is what the ROOMS section is built from.
    It is optional for the same reason it is optional on the card builder: a
    caller holding only options gets combinations and no list, rather than no
    card.
    """
    num_nights = state.get("num_nights")
    if not num_nights and state.get("check_in") and state.get("check_out"):
        from app.services.booking_engine import calculate_num_nights
        try:
            num_nights = calculate_num_nights(state.get("check_in", ""), state.get("check_out", ""))
        except Exception:
            num_nights = 1
    num_nights = num_nights or 1

    adults, children = _card_party(state)

    # Each room's own `base_adult_occupancy` — how many the room sleeps before a
    # mattress — for the note `_bed_note` writes. Read from the catalogue for the
    # same reason `_card_candidates` reads it there: the allocator's option
    # carries no capacity. Empty when the caller had no catalogue
    # (`response_generator` builds a card from options alone), which is why the
    # note has a form that states the bed's price without claiming a berth count.
    seats_by_name = {
        str(getattr(room, "name", "") or "").strip().upper(): int(
            getattr(getattr(room, "occupancy", None), "base_adults", 0) or 0
        )
        for room in (rooms or [])
    }

    lines: list[str] = []
    header = _room_card_header(state, adults, children, num_nights)
    if header:
        lines.append(header)
        lines.append("")

    # Section 1: the room types, at the rate the combinations were priced on.
    # Absent when the caller had no catalogue — `response_generator` builds a
    # card from options alone — and the card then carries combinations only,
    # which is what it did before this section existed.
    room_lines = _rooms_section(options, rooms, state, rows)
    if room_lines:
        lines.append("*ROOMS*")
        lines.extend(room_lines)
        lines.append("")

    # Section 2: the priced sets, numbered. Keycap numerals rather than "1."
    # for the reason `_card_numeral` gives: a line starting "1." is an
    # ordered-list marker to the Markdown renderer (chat_ui.py:264), which eats
    # the number and leaves the guest a card that says "Reply *1-4*" with no 1s
    # on it. Keycaps are not list syntax in either renderer, and the rest of the
    # bot already numbers with them (`_card_party`'s menus).
    #
    # These were letters (A, B, C) until 2026-09-23, on the client's template.
    # product asked for numbers: the card's own follow-up prompts already said
    # "e.g. *Option 1*", so the guest was being told two different alphabets on
    # one screen. `room_card_reply` accepts both, so a guest holding an old
    # lettered card in their scrollback still resolves.
    if rows:
        lines.append(f"*OR READY COMBINATIONS FOR {adults + children} GUESTS*")
        for idx, row in enumerate(rows, 1):
            packing = row["packing"]
            opt = row["option"]
            if packing is None:
                # The allocator's own option, when the packer found nothing it
                # will sell (see `rows_for_card`). No packing means no sum to
                # break down, so the option's own nightly figure is the price.
                nightly = float(
                    opt.get("total_price_per_night_inclusive")
                    or opt.get("total_price_per_night")
                    or 0
                )
                lines.append(f"{_card_numeral(idx)} {_option_title(opt)}")
                # Rate only, like every other option row — see `_packing_price`.
                lines.append(f"   ₹{nightly:,.0f}/night")
            else:
                # One decision, one place: the bed is named in the title only
                # when the note beneath cannot quote a price for it. Otherwise
                # the guest would read the same mattress twice.
                bed_charge = _bed_charge(packing, row["prices"])
                lines.append(
                    f"{_card_numeral(idx)} "
                    f"{_packing_title(packing, beds_in_title=bed_charge <= 0)}"
                )
                lines.append(
                    "   "
                    + _packing_price(
                        packing,
                        row["prices"],
                        seats_by_name,
                        # The party this set is for, so a row that seats them
                        # without a mattress can say so. `_card_party` was
                        # already read above for the header.
                        guests=adults + children,
                    )
                )
        lines.append(
            f"{_card_numeral(len(rows) + 1)} Tell me how you want to split — I'll quote it"
        )
        lines.append("")

    lines.append(f"_{_meal_footer([r['option'] for r in rows])}_")

    # Say why a "show both" card carries one class. Placed here rather than
    # under the ROOMS list because it is about the choice, not the catalogue.
    # Never gates a row: it explains the card, it does not change it.
    note = _missing_class_note(options, rooms, state)
    if note:
        lines.append("")
        lines.append(note)

    # No rows, no trailer. The old card invited "Reply *1* for a room" even with
    # nothing on it, which is an invitation to a reply that resolves to nothing.
    # The range ends on the build-your-own line, which is `len(rows)` past 1 and
    # which `room_card_reply._choice` resolves as `shown + 1` — so a two-row card
    # says 1-3 and a three-row card says 1-4, and neither names a number that
    # resolves to nothing.
    #
    # Plain digits rather than the keycaps used on the rows above: this is a
    # sentence, not a list item, so "1." cannot start it and the renderer leaves
    # it alone. `*1-3*` is not list syntax either.
    if rows:
        lines.append(f"Reply *1-{len(rows) + 1}*")

    return "\n".join(lines)


def format_stay_totals(
    options: list[dict] | None,
    num_nights: int,
    state: dict | None = None,
) -> str | None:
    """
    Each option's total for the whole stay, numbered and titled as the card above it.

    `_packing_price` prints per-night rates and nothing else — deliberately,
    since 2026-09-24 — so the card carries no figure that answers "what's the
    total?". A guest asking it has done nothing wrong and the card cannot
    answer, which is why the reply is this and not the card again: re-sending
    the card hands back the rates the guest is already reading. Three direct
    price questions in one live conversation got the same card back three times
    on 2026-09-26, and the guest's fourth message was the question again.

    The figures are the option's own `total_price_for_stay`, put there by
    `enrich_options_with_stay_totals` from the same `PricingEngine` the payment
    will be taken against — so this is the amount that will be charged, not a
    rate multiplied out by hand here.

    The rows come from `rows_for_card` with the arguments
    `build_room_options_card` passes it at the call sites that build from
    options alone (`rooms` omitted). That is the whole reason the numerals mean
    on this reply what they mean on the card: a row numbered 3 there is
    numbered 3 here. Passing the catalogue would break that — the card this
    answers was built without one.

    Returns None when nothing could be totalled — no rows, or the engine priced
    none of them — so the caller falls back rather than sending an empty
    answer. A total of 0 means "not priced", never "free", and is skipped.
    """
    if not num_nights:
        return None

    quoted: list[tuple[int, dict, float]] = []
    for idx, row in enumerate(
        rows_for_card([_normalize_option(o) for o in (options or [])], None, state or {}),
        1,
    ):
        opt = row["option"]
        total = float(opt.get("total_price_for_stay") or 0)
        if total > 0:
            quoted.append((idx, opt, total))

    if not quoted:
        return None

    nights = "1 night" if num_nights == 1 else f"{num_nights} nights"
    lines = [f"**Total for your stay — {nights}**", ""]
    for idx, opt, total in quoted:
        lines.append(f"{_card_numeral(idx)} {_option_title(opt)} — **₹{total:,.0f}**")
    lines.append("")
    lines.append(f"Each figure is the whole {nights}, not per night.")
    return "\n".join(lines)


def build_room_options_card(
    options: list[dict] | None,
    state: dict | None = None,
    rooms: list | None = None,
) -> dict:
    """
    Build the room_options card.

    `rooms` is optional and additive: `response_generator.py:330` calls this
    with options alone and keeps working, rendering the combinations without
    the room list. `room_searcher.search_rooms` has the catalogue in hand and
    passes it, which is what puts the ROOMS section on the card a guest
    actually sees.
    """
    state = state or {}
    options = [_normalize_option(o) for o in (options or [])]

    num_nights = state.get("num_nights")
    if not num_nights and state.get("check_in") and state.get("check_out"):
        from app.services.booking_engine import calculate_num_nights
        try:
            num_nights = calculate_num_nights(state.get("check_in", ""), state.get("check_out", ""))
        except Exception:
            num_nights = 1
    num_nights = num_nights or 1

    # The rows the card prints, computed once. `data["options"]` is then exactly
    # the card's own lines in the card's own order, because a reply of "1" is
    # resolved by position against that list a turn later
    # (`validator._resolve_room_selection`). Handing back the allocator's full
    # list — 80 options for a party of seven — would make "1" mean the
    # allocator's first pick rather than the one the guest is looking at, and
    # the two are not the same option.
    rows = rows_for_card(options, rooms, state)
    text = _render_rows(rows, options, state, rooms)
    card_options = [row["option"] for row in rows]

    from app.prompts.templates import format_assumption_line

    child_notes = state.get("child_age_notes") or []
    if child_notes:
        text += f"\n\n_{' '.join(child_notes)}_"

    assumption_str = format_assumption_line(state.get("assumptions"))
    if assumption_str:
        text += f"\n\n{assumption_str}"

    # The card numbers its combinations and then asks the guest to type the
    # number. On WhatsApp that is a keyboard exercise for a choice we already
    # know the shape of, so each priced row also becomes a button. The titles
    # are the card's own words — "Option 1" — which `room_card_reply` already
    # resolves by position, so a tap and a typed "1" take the same path.
    #
    # WhatsApp caps reply buttons at three, which is exactly the card's own cap
    # (`ROOM_CARD_MAX_COMBOS`), so no row is ever left without a button. The
    # build-your-own row is not buttoned: it asks the guest for a sentence, and
    # no button can carry one.
    #
    # `buttons` is additive on purpose. Nothing downstream is required to
    # understand it — the chat UI renders `data`, and the WhatsApp export in
    # `chat.py` only turns this into a tappable message when the card fits a
    # Meta body (see `to_interactive`); otherwise the card goes out as text
    # exactly as before.
    buttons = [
        {"id": f"room_option_{idx}", "title": f"Option {idx}"}
        for idx in range(1, min(len(rows), 3) + 1)
    ]

    return {
        "type": "room_options",
        "text": text,
        "data": {
            "options": card_options,
            "budget": state.get("budget"),
            "num_nights": num_nights,
        },
        "buttons": buttons or None,
    }


def format_room_options(options: list[AllocationOption]) -> str:
    """Format allocation options as clean, readable text."""
    if not options:
        return "Sorry, we couldn't find any room combinations matching your requirements."

    lines: list[str] = ["Here are the available options for your stay:\n"]

    for i, opt in enumerate(options, 1):
        from app.services.pricing_engine import PricingEngine
        price_pn = PricingEngine.estimate_inclusive_price(opt.total_price_per_night)
        extra_a = getattr(opt, "extra_adult_count", 0)
        
        room_names = []
        for r in opt.rooms:
            r_name = str(r.room_name).strip().upper()
            if extra_a > 0 and "+ EXTRA BED" not in r_name and "EXTRA BED" not in r_name:
                r_name = f"{r_name} + Extra Bed"
            room_names.append(r_name)
            
        if len(opt.rooms) == 1:
            room_str = room_names[0]
        else:
            counts: dict[str, int] = {}
            for name in room_names:
                counts[name] = counts.get(name, 0) + 1
            room_str = ", ".join(f"{cnt}x {n}" if cnt > 1 else n for n, cnt in counts.items())

        max_guests = max(opt.max_adults or 0, opt.base_adults + extra_a)

        lines.append(f"*Option {i}* — {room_str}")
        lines.append(f"👥 Up to {max_guests} Guests | ₹{price_pn:,.0f}/night\n")

    num_opts = len(options)
    opt_range = "1, 2 or 3" if num_opts == 3 else f"1 to {num_opts}" if num_opts > 1 else "1"
    lines.append(f"Reply with the option number ({opt_range}) or the room name.")
    return "\n".join(lines)


def _friendly_rate_plan_name(raw_name: str) -> str:
    """Convert rate plan codes to guest-friendly names."""
    name_upper = raw_name.strip().upper()
    mapping = {
        "EP": "Room Only",
        "CP": "Includes Breakfast",
        "MAP": "Includes Breakfast & Dinner (or Lunch)",
        "AP": "All Meals Included",
    }
    for code, friendly in mapping.items():
        if name_upper == code or name_upper.startswith(code + " ") or name_upper.startswith(code + "(") or name_upper.startswith(code + "_"):
            return friendly
    
    # Fallback cleanup for stuff like "Bed & Breakfast" to look nice
    return raw_name.strip().title()


# ── Extra-bed note (client requirement, 2026-09-19) ──────────────────────────
#
# "When suggesting an extra bed, mention 'You can add an extra bed.'"
#
# The client's own wording, so it is a constant rather than a sentence we
# phrase ourselves, and it stays English in Hinglish chats for the same reason.
# It rides under any option whose `extra_adult_count` is above zero — which is
# the only sense in which the bot "suggests an extra bed": the bed is already
# in the title and already priced, and the guest needs to be told it is an
# addition they are choosing rather than something quietly folded into the rate.
#
# It is deliberately *not* added to options that carry no bed. Printing "you can
# add an extra bed" under a room that cannot take one would be a promise the
# allocation engine will refuse to keep.
EXTRA_BED_NOTE = "🛏️ You can add an extra bed."


def extra_bed_note(extra_beds: int) -> str:
    """The extra-bed line for an option, or "" when it carries no extra bed."""
    try:
        return EXTRA_BED_NOTE if int(extra_beds or 0) > 0 else ""
    except (TypeError, ValueError):
        return ""
