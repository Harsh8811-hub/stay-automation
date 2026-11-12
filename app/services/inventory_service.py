"""
Inventory Service — date-based dynamic room availability.

Availability is sourced from the `daily_inventory` collection (per room-type,
per date `available` counts + `status`) and `daily_rates` stop-sell
restrictions, with a booking-derived fallback for dates that have no
`daily_inventory` coverage yet (the collection is only rolled forward a few
days at a time).

The service also exposes atomic `reserve_inventory` / `release_inventory`
helpers so the bot updates inventory directly when a hold is created, a
booking is confirmed, or a hold is cancelled/expired.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

from app.models.schemas import RoomType
from app.repositories import room_repository, booking_repository
from app.db.mongodb import (
    daily_inventory_collection,
    daily_rates_collection,
    room_types_collection,
)
from app.repositories.rate_threshold_repository import get_rate_threshold
from app.utils.rate_threshold import RateThreshold, clamp_nightly_rate

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# THE PRICE-RESOLUTION LADDER
#
# Where a nightly rate comes from, most specific first. This is the only
# statement of that order; every pricing path is expected to agree with it.
#
#   Rung 1  daily_rates, keyed (room_type_id, rate_plan_id, date)          LIVE
#           A rate set for this room, this plan, this exact night.
#
#   Rung 2  daily_rates, keyed (room_type_id, rate_type_id, date)          LIVE
#           A rate set for the meal type on this night. Accepted only when
#           the row names no plan or names ours — the sibling-plan guard in
#           `calendar_rate_for_night` below. Without that guard a sibling
#           plan's row silently overwrites our price.
#
#   Rung 3  rate_plans.base_price                                LIVE
#           The catalogue: the static fallback for any night that rungs 1
#           and 2 don't cover.
#
# `calendar_rate_for_night` walks rungs 1→2 and returns None when neither
# applies. Callers apply rung 3 themselves, because the catalogue value lives
# on the plan object rather than in the calendar — `_apply_calendar_rates` here
# and `calculate_price_breakdown` in booking_engine both do it.
#
# For a whole stay, the order of application is:
#
#   1. resolve each night          rungs 1 → 2 → 3
#   2. average the nights          `average_nightly`, rounded to 10
#   3. seasonal multiplier         LIVE — `get_active_season_modifier`
#   3b. rate threshold             LIVE — the hotel's own floor and ceiling
#   4. + extra adult/child         per the room's occupancy rules
#   5. GST                         inclusive extracts it, exclusive adds it
#   6. round to 10
#
# Step 3 is a *multiplier on a resolved rate*, not a rival rung — it scales
# whatever steps 1-2 produced, so it cannot sit in the ladder above without
# competing with the calendar.
#
# An earlier version of this comment said step 3 was reserved and that nothing
# seasonal was applied. That was wrong and had been for months: seasons really
# do move quotes (measured 2026-09-25 — a Diwali season at 1.4x, an off-peak
# night at 0.8x), and both `_apply_calendar_rates` below and
# `booking_engine.calculate_price_breakdown` apply them.
#
# Step 3b is the leash on step 3. The season multiplier is floored at 0.50 and
# has no ceiling, so it is the only unbounded modifier on a nightly rate in this
# codebase; the hotel's configured window bounds how far it may drag the price.
# Both producers apply it, against the same pre-season reference.
# ─────────────────────────────────────────────────────────────────────────────


def _nights(check_in: str | None, check_out: str | None) -> list[str]:
    """Return the stay-night dates (YYYY-MM-DD) in [check_in, check_out)."""
    if not check_in or not check_out:
        return []
    try:
        ci = datetime.strptime(str(check_in).split("T")[0].split(" ")[0], "%Y-%m-%d")
        co = datetime.strptime(str(check_out).split("T")[0].split(" ")[0], "%Y-%m-%d")
    except (TypeError, ValueError):
        return []
    if co <= ci:
        return []
    dates: list[str] = []
    cur = ci
    while cur < co:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def calendar_rate_for_night(
    rates_by_plan: dict,
    rates_by_type: dict,
    room_id: str,
    rate_plan_id: str,
    date: str,
    rate_type_ids: Iterable[str | None] = (),
) -> float | None:
    """
    One night's calendar rate for one rate plan, or None if the calendar has no
    row for it. **This is the only place that lookup is allowed to happen.**

    Why it is a function and not two copies
    ---------------------------------------
    This lookup existed twice: once here in `_apply_calendar_rates` (which
    prices the room card) and once, hand-copied, in
    `booking_engine.calculate_price_breakdown` (which prices the stay summary).
    Only this copy had the sibling-plan guard below, so for five of Sample Hotel's
    thirteen plans the card and the summary quoted different numbers for the
    same room on the same night — measured 2026-09-17, e.g. STANDARD NON AC ROOM
    at 1550 on the card and 2550 in the summary. Two copies of a rule is one
    copy too many; the second one drifted.

    The guard
    ---------
    `by_type` is keyed by rate TYPE, which every plan of the same meal plan
    shares, so it can hand back a *sibling* plan's row. A row that names a
    different plan is that plan's rate, not ours — taking it overwrites our
    price with theirs. (`daily_rates` carries RP_EA7A12 rows under six rooms at
    2550; that is how 2550 leaked onto five rooms whose own EP plans have no
    calendar rows.) A row naming no plan is a genuine type-level rate and is
    accepted.

    `rate_type_ids` is an ordered list of candidates to try, because not every
    caller knows the plan's declared type: `RoomType.rate_plans` carries
    `rate_type_id`, but `SelectedRoom` does not, so the booking engine infers a
    type from the plan's display name and passes it here. Declared first, then
    inferred — a declared type is a fact, an inferred one is a guess.
    """
    room_id = str(room_id)
    rate_plan_id = str(rate_plan_id)

    rate_doc = rates_by_plan.get((room_id, rate_plan_id), {}).get(date)
    if rate_doc:
        return _base_rate_or_none(rate_doc)

    for raw_type in rate_type_ids:
        if not raw_type:
            continue
        type_doc = rates_by_type.get((room_id, str(raw_type)), {}).get(date)
        if type_doc is None:
            continue
        doc_plan = str(type_doc.get("rate_plan_id") or "")
        if not doc_plan or doc_plan == rate_plan_id:
            return _base_rate_or_none(type_doc)

    return None


def _base_rate_or_none(rate_doc: dict) -> float | None:
    """`base_rate` if it is a usable positive number, else None."""
    raw = rate_doc.get("base_rate")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def average_nightly(plan_rates: list[float]) -> float:
    """The nightly figure a multi-night stay is quoted at: mean, rounded to 10."""
    return float(round((sum(plan_rates) / len(plan_rates)) / 10) * 10)


def _apply_calendar_rates(
    rooms: list[RoomType],
    nights: list[str],
    rates_by_plan: dict,
    rates_by_type: dict,
    season_multiplier: float = 1.0,
    threshold: RateThreshold | None = None,
) -> None:
    """
    Overlay `daily_rates.base_rate` onto each plan's `base_price`, in place.

    A night with no calendar row keeps the plan's static `base_price`, so a stay
    that straddles the calendar's coverage blends the two rather than failing.
    Applies `season_multiplier` (Step 3 of the price ladder) to `plan.base_price`
    while preserving `plan.raw_base_price` for downstream components.

    `threshold` is the hotel's own floor and ceiling on a nightly rate, measured
    against `raw_avg` — the calendar average BEFORE the season multiplier. That
    is the same reference `booking_engine.calculate_price_breakdown` uses, and it
    is the whole point: the bound is on how far the season may drag the price,
    not a second opinion on the calendar. It is a parameter rather than a lookup
    because this function is sync and has no `hotel_id`; the async caller already
    fetches the season multiplier the same way.
    """
    if not nights:
        return
    for room in rooms:
        room_id = str(room.room_id)
        for plan in room.rate_plans or []:
            plan_rates: list[float] = []
            for dt in nights:
                night = calendar_rate_for_night(
                    rates_by_plan,
                    rates_by_type,
                    room_id,
                    str(plan.rate_plan_id),
                    dt,
                    (plan.rate_type_id,),
                )
                plan_rates.append(night if night is not None else float(plan.base_price))

            if plan_rates:
                raw_avg = average_nightly(plan_rates)
                plan.raw_base_price = raw_avg
                if season_multiplier and season_multiplier != 1.0:
                    plan.base_price = float(round((raw_avg * season_multiplier) / 10) * 10)
                else:
                    plan.base_price = raw_avg

                plan.base_price, _ = clamp_nightly_rate(
                    raw_avg, plan.base_price, threshold
                )


async def _fetch_inventory(hotel_id: int | str, nights: list[str]) -> dict[str, dict[str, dict]]:
    """Map room_type_id -> {date: daily_inventory doc} for the requested nights."""
    if not nights:
        return {}
    variants = room_repository._get_hotel_variants(hotel_id)
    col = daily_inventory_collection()
    cursor = col.find({"hotel_id": {"$in": variants}, "date": {"$in": nights}})
    docs = await cursor.to_list(length=2000)
    by_room: dict[str, dict[str, dict]] = {}
    for d in docs:
        rt = d.get("room_type_id")
        dt = d.get("date")
        if rt and dt:
            by_room.setdefault(str(rt), {})[str(dt)] = d
    return by_room


async def _fetch_stop_sold(hotel_id: int | str, nights: list[str]) -> dict[str, set[str]]:
    """room_type_id -> {the nights it is stopped for}, out of `nights`.

    The row's `room_type_id` is the room the hotel closed, and that is the
    whole meaning of the row. `rate_plan_id` is **not read**, deliberately:
    it is a stale artifact the dashboard writes beside the room, and
    `daily_rates` carries stop-sell rows naming RP_EA7A12 under six rooms,
    five of which do not own that plan.

    Reading it was the 2026-09-25 defect. The reader treated a row naming a
    plan the room does not own as "somebody else's row" — a rule borrowed
    from `calendar_rate_for_night`, where it is correct, because there a row
    is a rate and a rate belongs to a plan. A stop-sell is not a rate: it is
    a statement about the room. So the rule hid exactly one of the six rooms
    (the only one that happened to own RP_EA7A12) and kept selling the other
    five. Measured on the clone before the fix, hotel 111111, 2026-09-11:
    five of six stopped rooms still on the card.

    `is_available: False` is the same statement in a different flag and is
    read identically — see `_stop_sold_nights`.
    """
    if not nights:
        return {}
    variants = room_repository._get_hotel_variants(hotel_id)
    col = daily_rates_collection()
    cursor = col.find(
        {
            "hotel_id": {"$in": variants},
            "date": {"$in": nights},
            "$or": [{"stop_sell": True}, {"is_available": False}],
        },
        {"room_type_id": 1, "date": 1},
    )
    docs = await cursor.to_list(length=1000)
    blocked: dict[str, set[str]] = {}
    for d in docs:
        rt = d.get("room_type_id")
        dt = d.get("date")
        if rt and dt:
            blocked.setdefault(str(rt), set()).add(str(dt))
    return blocked


def _stop_sold_nights(
    room_id: str,
    blocked: dict[str, set[str]],
    nights: Iterable[str],
) -> set[str]:
    """The nights this room is stop-sold for, out of the nights asked about.

    Any row naming this room stops it — no test against the room's rate plans.
    There is nothing to test: the row's `room_type_id` is the room the hotel
    closed, and the plan id beside it is not a relation. The plan check that
    used to live here (`p in own_plans`) came from the pricing path and let
    five of six stopped rooms keep selling; `_fetch_stop_sold` carries the
    measurement and the reasoning.

    `nights` is intersected rather than trusted. `_fetch_stop_sold` already
    scopes its query to these nights, so in the real path the intersection is
    a no-op — it is here so the function means what its first line says. The
    caller reads a non-empty return as "drop this room", so a map carrying a
    night outside the stay would otherwise hide a room the guest never asked
    about, which is the same class of bug as the one being fixed.
    """
    return set(blocked.get(room_id) or ()) & set(nights)


async def get_stop_sold_room_ids(
    hotel_id: int | str,
    check_in: str | None = None,
    check_out: str | None = None,
) -> set[str]:
    """
    Return the set of room_id strings that are stop-sold / closed by management
    for the requested dates (or today if dates not provided).

    Checks both:
    1. daily_rates: stop_sell == True or is_available == False
    2. daily_inventory: status == "closed"
    """
    nights = _nights(check_in, check_out)
    if not nights:
        nights = [datetime.utcnow().strftime("%Y-%m-%d")]

    base_rooms = await room_repository.get_all_rooms(hotel_id)
    if not base_rooms:
        return set()

    stop_sold = await _fetch_stop_sold(hotel_id, nights)
    inv_by_room = await _fetch_inventory(hotel_id, nights)

    stopped_room_ids: set[str] = set()
    for room in base_rooms:
        room_id = str(room.room_id)
        # 1. daily_rates stop-sell (per product ruling: any row naming this room stops it)
        if _stop_sold_nights(room_id, stop_sold, nights):
            stopped_room_ids.add(room_id)
            continue
        # 2. daily_inventory status == "closed"
        room_inv = inv_by_room.get(room_id, {})
        if any(str(room_inv[dt].get("status") or "open").lower() == "closed" for dt in nights if dt in room_inv):
            stopped_room_ids.add(room_id)

    return stopped_room_ids


async def fetch_daily_rates(hotel_id: int | str, nights: list[str]) -> dict[str, dict]:
    """
    Fetch calendar rates from daily_rates for the requested stay nights.
    Returns:
      {
        "by_plan": {(room_type_id, rate_plan_id): {date: doc}},
        "by_type": {(room_type_id, rate_type_id): {date: doc}},
      }
    """
    if not nights:
        return {"by_plan": {}, "by_type": {}}
    variants = room_repository._get_hotel_variants(hotel_id)
    col = daily_rates_collection()
    cursor = col.find(
        {
            "hotel_id": {"$in": variants},
            "date": {"$in": nights},
            "is_available": {"$ne": False},
            "stop_sell": {"$ne": True},
        },
        {
            "date": 1,
            "room_type_id": 1,
            "rate_plan_id": 1,
            "rate_type_id": 1,
            "base_rate": 1,
            "extra_adult_rate": 1,
            "extra_child_rate": 1,
        },
    )
    docs = await cursor.to_list(length=2000)
    by_plan: dict[tuple[str, str], dict[str, dict]] = {}
    by_type: dict[tuple[str, str], dict[str, dict]] = {}
    for d in docs:
        dt = d.get("date")
        rt = d.get("room_type_id")
        rp = d.get("rate_plan_id")
        r_type = d.get("rate_type_id")
        if not dt or not rt:
            continue
        str_rt = str(rt)
        str_dt = str(dt)
        if rp:
            by_plan.setdefault((str_rt, str(rp)), {})[str_dt] = d
        if r_type:
            by_type.setdefault((str_rt, str(r_type)), {})[str_dt] = d

    return {"by_plan": by_plan, "by_type": by_type}


async def get_available_rooms(hotel_id: int, check_in: str | None, check_out: str | None) -> list[RoomType]:
    """
    Fetch base rooms and subtract rooms already consumed for the requested range.

    - Honors `daily_inventory` per-night `available` counts and `status == "closed"`.
    - Honors `daily_rates.stop_sell` / `is_available == False`: a blocked row takes the whole
      room off the calendar for that night, not just the rate plan the row names.
    - Overlays dynamic calendar rates (`daily_rates.base_rate`) onto room rate plans for the requested stay dates.
    - Falls back to booking-overlap math for nights without `daily_inventory` coverage.
    """
    base_rooms = await room_repository.get_all_rooms(hotel_id)

    nights = _nights(check_in, check_out)
    if not nights:
        # Date-less price query ("deluxe room king bed price?"). There is no stay
        # to price, so price from today: `daily_rates` is the source of truth, and
        # `base_price` is only the fallback for a date it has no row for.
        #
        # Availability is deliberately NOT applied here. The guest gave us nothing
        # to check availability *for*, and tonight's daily_inventory or stop-sell
        # would hide rooms they never asked about. The catalogue already defers
        # availability to the dates prompt.
        today = [datetime.utcnow().strftime("%Y-%m-%d")]
        rates = await fetch_daily_rates(hotel_id, today)
        _apply_calendar_rates(
            base_rooms, today, rates.get("by_plan", {}), rates.get("by_type", {}),
            threshold=await get_rate_threshold(hotel_id),
        )
        return base_rooms

    # Booking-derived consumption (fallback + safety cross-check)
    overlapping_bookings = await booking_repository.get_overlapping_bookings(hotel_id, check_in, check_out)
    booked_counts: dict[str, int] = {}
    for booking in overlapping_bookings:
        for booked_room in booking.get("rooms", []):
            r_id = booked_room.get("room_id")
            if r_id:
                booked_counts[str(r_id)] = booked_counts.get(str(r_id), 0) + 1

    inv_by_room = await _fetch_inventory(hotel_id, nights)
    stop_sold = await _fetch_stop_sold(hotel_id, nights)
    daily_rates_data = await fetch_daily_rates(hotel_id, nights)
    rates_by_plan = daily_rates_data.get("by_plan", {})
    rates_by_type = daily_rates_data.get("by_type", {})

    from app.repositories.season_repository import get_active_season_modifier
    season_multiplier, season_name = await get_active_season_modifier(hotel_id, check_in, check_out)
    # The hotel's own bound on a nightly rate. Same reference as booking_engine:
    # the calendar average before the season multiplier, so the card and the stay
    # summary clamp against the same base.
    threshold = await get_rate_threshold(hotel_id)

    available_rooms: list[RoomType] = []
    for room in base_rooms:
        room_id = str(room.room_id)

        # 1) Stop-sell takes the whole room off the calendar, for any night it
        #    covers. The hotel's "stop sell" means "do not offer this room",
        #    so one blocked night drops the room rather than just that plan.
        if _stop_sold_nights(room_id, stop_sold, nights):
            continue

        # 2) Overlay dynamic calendar rates (daily_rates) onto rate plans for stay dates
        _apply_calendar_rates(
            [room], nights, rates_by_plan, rates_by_type, season_multiplier, threshold
        )

        # 3) daily_inventory coverage for the stay.
        room_inv = inv_by_room.get(room_id, {})
        covered = [dt for dt in nights if dt in room_inv]
        sold_out_night = any(
            str(room_inv[dt].get("status") or "open").lower() == "closed"
            or int(room_inv[dt].get("available", 0) or 0) <= 0
            for dt in covered
        )
        if sold_out_night:
            continue

        booking_net = room.number_of_rooms - booked_counts.get(room_id, 0)
        if len(covered) == len(nights):
            inventory_count = min(int(room_inv[dt].get("available", 0) or 0) for dt in nights)
            net_available = min(booking_net, inventory_count)
        else:
            # Partial/no inventory coverage — trust booking math for those nights.
            net_available = booking_net

        if net_available > 0:
            room.number_of_rooms = net_available
            available_rooms.append(room)

    return available_rooms


async def _room_capacity(hotel_id: int | str, room_type_id: str) -> int | None:
    """Physical room count from the `room_types` catalog (for seeding inventory rows)."""
    try:
        doc = await room_types_collection().find_one(
            {"hotel_id": str(hotel_id), "room_type_id": str(room_type_id)},
            {"total_rooms": 1, "number_of_rooms": 1},
        )
        if doc:
            total = int(doc.get("total_rooms") or doc.get("number_of_rooms") or 0)
            return total or None
    except Exception as e:
        logger.warning(f"Failed to read capacity for {room_type_id}: {e}")
    return None


async def reserve_inventory(
    hotel_id: int | str,
    room_type_id: str | None,
    check_in: str | None,
    check_out: str | None,
    qty: int = 1,
) -> bool:
    """
    Atomically reserve `qty` rooms for each night of the stay in `daily_inventory`.

    Decrements `available`, increments `booked`, and never goes below zero.
    Returns False if any night cannot be satisfied (insufficient availability).
    """
    nights = _nights(check_in, check_out)
    if not nights or qty <= 0 or not room_type_id:
        return True

    col = daily_inventory_collection()
    hid = str(hotel_id)
    now = datetime.utcnow().isoformat() + "Z"
    ok = True

    for dt in nights:
        key = {"hotel_id": hid, "room_type_id": str(room_type_id), "date": dt}
        existing = await col.find_one(key)

        if existing is None:
            total = await _room_capacity(hotel_id, str(room_type_id))
            if total is None or total < qty:
                logger.warning(
                    f"INVENTORY_RESERVE_SKIPPED | hotel={hotel_id} room={room_type_id} date={dt} "
                    f"(no capacity source or capacity<qty)"
                )
                ok = False
                continue
            try:
                await col.update_one(
                    key,
                    {
                        "$set": {
                            "total_rooms": total,
                            "available": total - qty,
                            "booked": qty,
                            "out_of_order": 0,
                            "status": "open" if (total - qty) > 0 else "closed",
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            except Exception as e:
                logger.warning(f"INVENTORY_RESERVE_FAIL | {key} error={e}")
                ok = False
            continue

        res = await col.update_one(
            {**key, "available": {"$gte": qty}},
            {"$inc": {"available": -qty, "booked": qty}, "$set": {"updated_at": now}},
        )
        if res.modified_count == 0:
            logger.warning(
                f"INVENTORY_RESERVE_SHORTFALL | hotel={hotel_id} room={room_type_id} date={dt} qty={qty}"
            )
            ok = False

    return ok


async def release_inventory(
    hotel_id: int | str,
    room_type_id: str | None,
    check_in: str | None,
    check_out: str | None,
    qty: int = 1,
) -> None:
    """Return `qty` rooms to `daily_inventory` for each night (cancel / hold expiry)."""
    nights = _nights(check_in, check_out)
    if not nights or qty <= 0 or not room_type_id:
        return

    col = daily_inventory_collection()
    hid = str(hotel_id)
    now = datetime.utcnow().isoformat() + "Z"
    for dt in nights:
        try:
            await col.update_one(
                {"hotel_id": hid, "room_type_id": str(room_type_id), "date": dt},
                {"$inc": {"available": qty, "booked": -qty}, "$set": {"updated_at": now}},
            )
        except Exception as e:
            logger.warning(
                f"INVENTORY_RELEASE_FAIL | hotel={hotel_id} room={room_type_id} date={dt} error={e}"
            )
