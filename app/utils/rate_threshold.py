"""Clamping a computed nightly rate into the hotel's own floor and ceiling.

The Control Centre's "Rate Threshold Protection" page stores two **factors** per
hotel, in `hotel_settings.default_rate_threshold`:

    {"min_threshold_factor": 0.73, "max_threshold_factor": 5.57,
     "enforcement": "warm", "sync_to_pms_ota": true}

They are multipliers on a reference rate, not rupee amounts. The reference this
bot uses is the calendar rate **before** the seasonal multiplier, because that
multiplier is the only unbounded modifier on a nightly rate in this codebase —
`season_repository.get_active_season_modifier` floors it at 0.50 and applies no
ceiling at all. Everything else that moves a nightly rate (extra adult, extra
child, child policy) is a configured rupee amount, and GST is a fixed rate.

So "rate threshold protection" is, concretely, a leash on the season.

Why the reference is the pre-season rate and not the catalogue price: a night
priced off-peak by the calendar is *correct*, and clamping it back toward the
catalogue would be the bug rather than the fix. The window is meant to bound how
far a modifier may drag the price, not to overrule the calendar.

Fail-closed
-----------
Every ambiguous input yields `None`, and `None` means "no clamp". A threshold we
cannot read confidently must never move a guest's price: a missing clamp leaves
a quote as it is today, while a wrongly-parsed one invents a number the hotel
never set.

Pure. No Mongo, no settings, no logging. The reader that touches the database is
`app/repositories/rate_threshold_repository.py`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateThreshold:
    """A hotel's floor and ceiling, as multipliers on a reference rate."""

    min_factor: float
    max_factor: float


def _as_factor(value: Any) -> float | None:
    """A usable positive multiplier, or None.

    `bool` is rejected explicitly: `isinstance(True, int)` is true, so a stored
    JSON `true` would otherwise read as the factor 1.0 and silently pass.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        factor = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(factor) or factor <= 0:
        return None
    return factor


def parse_rate_threshold(raw: Any) -> RateThreshold | None:
    """``RateThreshold`` from a stored `default_rate_threshold` dict, or None.

    None means "no clamp", and is returned for every shape the Control Centre
    might plausibly write that does not describe a usable window: a missing
    block, a missing or non-numeric factor, a zero or negative factor, an
    inverted window, or the whole block written as a JSON string.

    Note `enforcement` is deliberately not consulted. The one hotel that has
    this configured stores `"warm"` — not a value any code would recognise, and
    the frontend reads the same field back expecting `"warn"`. Keying the clamp
    on that field would mean a hotel that armed thresholds got no protection at
    all, which is the exact failure this feature exists to prevent. The presence
    of a usable window is the switch.
    """
    if not isinstance(raw, dict):
        return None

    min_factor = _as_factor(raw.get("min_threshold_factor"))
    max_factor = _as_factor(raw.get("max_threshold_factor"))
    if min_factor is None or max_factor is None:
        return None
    if max_factor < min_factor:
        # An inverted window describes no interval. Guessing which end the hotel
        # meant would be inventing policy.
        return None

    return RateThreshold(min_factor=min_factor, max_factor=max_factor)


def clamp_nightly_rate(
    base: float | None,
    rate: float | None,
    threshold: RateThreshold | None,
) -> tuple[float | None, bool]:
    """``(clamped_rate, was_clamped)`` for one room's nightly rate.

    ``base`` is the reference rate (pre-season), ``rate`` is what the modifiers
    produced. Returns ``rate`` unchanged whenever a clamp is not warranted — no
    threshold, no base to measure against, or a non-positive either side.

    Rounding
    --------
    Every nightly rate in this repo ends in 0, and the rest of the pricing
    pipeline enforces that (`booking_engine` rounds the final figure to the
    nearest Rs 10). Rounding the clamp's own output could push it back out
    through the bound it was just clamped to, so the rounding is done *inward*:
    the ceiling rounds down, the floor rounds up. Both stay inside the window.

    For a window narrower than one Rs 10 step there is no multiple of 10 inside
    it, and the exact bound is returned instead. That is a misconfiguration and
    the honest answer is to honour the hotel's number rather than round past it.
    """
    if threshold is None or base is None or rate is None:
        return rate, False
    if base <= 0 or rate <= 0:
        return rate, False

    low = base * threshold.min_factor
    high = base * threshold.max_factor

    if rate > high:
        return _round_inward(high, low, high, downward=True), True
    if rate < low:
        return _round_inward(low, low, high, downward=False), True
    return rate, False


def _round_inward(bound: float, low: float, high: float, *, downward: bool) -> float:
    """``bound`` rounded to a multiple of Rs 10 that is still inside [low, high]."""
    if downward:
        rounded = math.floor(bound / 10) * 10
    else:
        rounded = math.ceil(bound / 10) * 10

    if low <= rounded <= high:
        return float(rounded)
    # The window is narrower than one rounding step: no multiple of 10 fits.
    return float(bound)
