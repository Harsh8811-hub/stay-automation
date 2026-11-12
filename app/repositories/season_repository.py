"""
Season Repository — reads seasonal pricing rules from MongoDB 'seasons' collection.
Calculates percentage modifiers (increase/decrease) for booking check-in/check-out date ranges.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from app.db.mongodb import get_database

logger = logging.getLogger(__name__)


def _parse_date(d_val: Any) -> datetime | None:
    """Parse date from datetime, date, or date string (YYYY-MM-DD, ISO)."""
    if not d_val:
        return None
    if isinstance(d_val, datetime):
        return d_val
    if isinstance(d_val, date):
        return datetime.combine(d_val, datetime.min.time())
    clean = str(d_val).strip().split("T")[0].split(" ")[0]
    try:
        return datetime.strptime(clean, "%Y-%m-%d")
    except Exception:
        return None


def _dates_overlap(
    dt_in: datetime,
    dt_out: datetime | None,
    r_start: datetime,
    r_end: datetime,
) -> bool:
    """
    Check if a stay [dt_in, dt_out) overlaps with a seasonal date range [r_start, r_end].
    Seasonal ranges are inclusive of the end date night.
    Auto-corrects inverted start/end dates and handles recurring annual season ranges.
    """
    if r_start > r_end:
        return False

    # Multi-night stay [dt_in, dt_out)
    if dt_out and dt_out > dt_in:
        season_end_exclusive = r_end + timedelta(days=1)
        if max(dt_in, r_start) < min(dt_out, season_end_exclusive):
            return True
    elif r_start <= dt_in <= r_end:
        return True

    # If the range was configured for another year (e.g. 2024 annual recurring festival),
    # check if the stay overlaps with the same calendar days across start/end year shifts.
    shifts = {dt_in.year - r_start.year, dt_in.year - r_end.year}
    if dt_out and dt_out > dt_in:
        shifts.add(dt_out.year - r_start.year)
        shifts.add(dt_out.year - r_end.year)

    for year_shift in shifts:
        if year_shift == 0:
            continue
        try:
            try:
                cur_r_start = r_start.replace(year=r_start.year + year_shift)
            except ValueError:
                cur_r_start = r_start.replace(month=2, day=28, year=r_start.year + year_shift)
            try:
                cur_r_end = r_end.replace(year=r_end.year + year_shift)
            except ValueError:
                cur_r_end = r_end.replace(month=2, day=28, year=r_end.year + year_shift)

            if cur_r_start > cur_r_end:
                cur_r_start, cur_r_end = cur_r_end, cur_r_start
            if dt_out and dt_out > dt_in:
                cur_end_exclusive = cur_r_end + timedelta(days=1)
                if max(dt_in, cur_r_start) < min(dt_out, cur_end_exclusive):
                    return True
            elif cur_r_start <= dt_in <= cur_r_end:
                return True
        except (ValueError, OverflowError):
            pass

    return False


def _months_overlap(
    dt_in: datetime,
    dt_out: datetime | None,
    months: list[bool],
) -> bool:
    """
    Check if any night of stay [dt_in, dt_out) falls in a month enabled in months array (12 booleans).
    """
    if not months or len(months) != 12:
        return False

    if dt_out and dt_out > dt_in:
        cur = dt_in
        while cur < dt_out:
            m_idx = cur.month - 1
            if 0 <= m_idx < 12 and bool(months[m_idx]):
                return True
            cur += timedelta(days=1)
        return False

    m_idx = dt_in.month - 1
    return 0 <= m_idx < 12 and bool(months[m_idx])


async def get_active_season_modifier(
    hotel_id: int | str,
    check_in: str,
    check_out: str | None = None,
) -> tuple[float, str | None]:
    """
    Finds the active seasonal modifier for a given hotel and date range.
    Returns: (multiplier: float, season_name: str | None)

    Resolution rules:
    1. Exact date range match (ranges: [{start, end}]) takes precedence over recurring month array (months: [12 booleans]).
    2. Higher priority breaks ties.
    3. Multiplier: 1.0 + (pct / 100.0), floored at 0.50x.
    """
    dt_in = _parse_date(check_in)
    if not dt_in:
        return 1.0, None
    dt_out = _parse_date(check_out)

    str_hid = str(hotel_id)
    variants = [str_hid, int(str_hid)] if str_hid.isdigit() else [str_hid]

    try:
        db = get_database()
        cursor = db["seasons"].find({
            "hotel_id": {"$in": variants},
            "active": {"$ne": False},
        })
        seasons = await cursor.to_list(length=100)
    except Exception as e:
        logger.warning(f"Error fetching seasons for hotel {hotel_id}: {e}")
        return 1.0, None

    if not seasons:
        return 1.0, None

    matched_seasons = []

    for s in seasons:
        s_name = str(s.get("name") or "").strip().lower()
        if s_name in ("try", "test", "temp"):
            continue

        is_match = False
        match_type = 0  # 2 for exact date range, 1 for month match

        span_days = 9999

        # 1. Check explicit date ranges (highest specificity)
        ranges = s.get("ranges") or []
        for r in ranges:
            if isinstance(r, dict):
                r_start = _parse_date(r.get("start") or r.get("start_date"))
                r_end = _parse_date(r.get("end") or r.get("end_date"))
                if r_start and r_end and _dates_overlap(dt_in, dt_out, r_start, r_end):
                    is_match = True
                    match_type = 2
                    span_days = max(1, (r_end - r_start).days)
                    break

        # 2. Check month array (12 booleans)
        if not is_match:
            months = s.get("months") or []
            if _months_overlap(dt_in, dt_out, months):
                is_match = True
                match_type = 1
                span_days = 365

        if is_match:
            matched_seasons.append((match_type, int(s.get("priority") or 0), -span_days, s))

    if not matched_seasons:
        return 1.0, None

    # Sort by: (1) match_type descending (explicit range > month), (2) priority descending, (3) narrower range descending
    matched_seasons.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    _, _, _, best = matched_seasons[0]

    raw_modifier = float(best.get("modifier") or 0.0)
    mode = str(best.get("modifier_mode") or "percentage").lower()
    season_name = best.get("name") or "Seasonal Rate"

    if mode == "decrease" or raw_modifier < 0:
        pct = -abs(raw_modifier)
    else:
        pct = abs(raw_modifier)

    multiplier = 1.0 + (pct / 100.0)
    # Guard against invalid negative pricing (floor at 0.50x)
    multiplier = max(0.50, multiplier)

    logger.info(
        f"SEASON_APPLIED | hotel={hotel_id} season='{season_name}' modifier={pct:+.0f}% multiplier={multiplier:.2f} check_in={check_in} check_out={check_out}"
    )
    return multiplier, season_name


async def get_season_negotiation_limit(
    hotel_id: int | str,
    check_in: str | None = None,
    check_out: str | None = None,
) -> tuple[float, str | None]:
    """
    Finds the maximum negotiation discount percentage allowed for the given hotel and dates.
    - Peak season (modifier > 0): capped at 5.0%
    - Off season (modifier < 0): max(10.0, |modifier|)%
    - Regular / No season: 10.0%
    """
    str_hid = str(hotel_id)
    variants = [str_hid, int(str_hid)] if str_hid.isdigit() else [str_hid]

    try:
        db = get_database()
        cursor = db["seasons"].find({
            "hotel_id": {"$in": variants},
            "active": {"$ne": False},
        })
        seasons = await cursor.to_list(length=100)
    except Exception as e:
        logger.warning(f"Error fetching seasons for negotiation limit (hotel {hotel_id}): {e}")
        return 10.0, None

    if not seasons:
        return 10.0, None

    dt_in = _parse_date(check_in) if check_in else None
    dt_out = _parse_date(check_out) if check_out else None

    matched_seasons = []
    all_decrease_seasons = []

    for s in seasons:
        s_name = str(s.get("name") or "").strip().lower()
        if s_name in ("try", "test", "temp"):
            continue

        raw_mod = float(s.get("modifier") or 0.0)
        mode = str(s.get("modifier_mode") or "percentage").lower()
        if mode == "decrease" or raw_mod < 0:
            all_decrease_seasons.append((abs(raw_mod), s.get("name")))

        if not dt_in:
            continue

        is_match = False
        span_days = 9999

        # 1. Date range match
        ranges = s.get("ranges") or []
        for r in ranges:
            if isinstance(r, dict):
                r_start = _parse_date(r.get("start") or r.get("start_date"))
                r_end = _parse_date(r.get("end") or r.get("end_date"))
                if r_start and r_end and _dates_overlap(dt_in, dt_out, r_start, r_end):
                    is_match = True
                    match_type = 2
                    span_days = max(1, (r_end - r_start).days)
                    break

        # 2. Month match
        if not is_match:
            months = s.get("months") or []
            if _months_overlap(dt_in, dt_out, months):
                is_match = True
                match_type = 1
                span_days = 365

        if is_match:
            matched_seasons.append((match_type, int(s.get("priority") or 0), -span_days, s))

    if matched_seasons:
        matched_seasons.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        _, _, _, best = matched_seasons[0]
        raw_modifier = float(best.get("modifier") or 0.0)
        mode = str(best.get("modifier_mode") or "percentage").lower()
        season_name = best.get("name") or "Seasonal Rate"

        if mode == "decrease" or raw_modifier < 0:
            max_limit = max(10.0, abs(raw_modifier))
        elif raw_modifier >= 20.0:
            # High peak season (+20% or higher) is capped at 5%
            max_limit = 5.0
        else:
            # Direct negotiation allows 5% then 10% ladder (up to 10% direct privilege)
            max_limit = 10.0

        return max_limit, season_name

    # If no date provided or no match, check if any decrease season exists
    if not dt_in and all_decrease_seasons:
        all_decrease_seasons.sort(key=lambda x: x[0], reverse=True)
        max_limit, s_name = all_decrease_seasons[0]
        return max(10.0, max_limit), s_name

    return 10.0, None
