"""
What a hotel charges for an early arrival or a late departure.

The rule, in the hotel's own terms
----------------------------------
The Control Centre stores, per hotel, a set of time bands and a price for each:
"3 hours early costs 27% of a night, 6 hours costs 42%, 9 hours costs a flat
Rs 1502". ``hours`` counts outward from the hotel's **standard** check-in or
check-out time, which lives in ``hotel_policies`` — not in this document.

Three things make the number:

1. **One night's rate, charged once.** A band's percentage applies to a single
   night of the rooms booked — never to the whole stay, never per night, and
   never to the subtotal. ``one_night_rate`` is supplied by the caller as the
   sum of one night across every room in the booking, so a two-room booking
   prices the same rule once, on the money it actually protects.
2. **Past the last band, or past the hotel's own bound, it is a full day.**
   Arriving before ``earliest_time`` is not a bigger band, it is another night —
   "add a day", in the operator's words. That case is charged as one full
   night's rate.
3. **No time, no number.** If the guest never stated a time there is nothing to
   price, and this module returns nothing at all. It never asks.

The fee is not taxed again. The rate a percentage is taken of is the room's
*inclusive* price — the engine already backs GST out of it — so taking a
percentage and then taxing that percentage would charge GST twice on the same
money.

Everything here returns ``[]`` rather than raising. A quote is more important
than a fee: a Mongo hiccup, an unparseable policy time or a hotel with no
configuration must cost the guest nothing but the fee line.
"""

from __future__ import annotations

import logging
from datetime import time

from app.config.settings import get_settings
from app.repositories.early_late_checkout_repository import (
    EarlyLateCheckoutConfig,
    EarlyLateRule,
    get_early_late_checkout_config,
)
from app.utils.guest_time import parse_clock
from app.prompts.templates import format_fee_disclaimer

logger = logging.getLogger(__name__)

#: Below this many hours outside the standard time, the guest is not really
#: early or late — round the clock and it is noise, not a request.
_MIN_HOURS_OUTSIDE = 0.0

#: A "full extra day" is one more night at the room's own rate.
_FULL_DAY_FRACTION = 1.0

_ROUND_TO = 10.0


def _round_to_nearest(value: float, step: float = _ROUND_TO) -> float:
    """Round to the nearest ``step`` rupees, matching how the quote rounds."""
    return float(round(value / step) * step)


def hours_outside(actual_min: int, standard_min: int, *, direction: str) -> float:
    """Hours between a stated clock time and the hotel's standard one.

    Positive means the guest is genuinely early (arrival) or late (departure);
    zero or negative means they are not, and the caller charges nothing.

    The difference is deliberately **not** taken modulo 24. A guest who says
    "arriving at 11pm" when check-in is 1pm is *late*, not fifteen hours early,
    and wrapping the subtraction would read them as the latter and invent a
    charge. A raw difference gets that right by going negative.
    """
    if direction == "arrival":
        return (standard_min - actual_min) / 60.0
    return (actual_min - standard_min) / 60.0


def select_band(rules: list[EarlyLateRule], hours: float) -> EarlyLateRule | None:
    """The cheapest enabled band that covers ``hours``, or None if none does.

    The smallest band whose reach is at least ``hours``. A request under the
    smallest band gets the smallest band — the Control Centre describes them as
    "up to N hours", so a 30-minute-early guest pays the 3-hour band.
    """
    usable = [r for r in rules if r.enabled and r.value > 0]
    if not usable:
        return None
    covering = [r for r in usable if r.hours >= hours]
    if not covering:
        return None  # Beyond every band — the caller turns this into a full day.
    return min(covering, key=lambda r: r.hours)


def directional_fee(one_night_rate: float, rule: EarlyLateRule) -> float:
    """The money one band asks for, on one night's rate.

    A percentage is *derived*, so it rounds to the nearest ten rupees the way
    every other computed figure on the quote does. A flat amount is **not** —
    the operator typed it, and 111111's 9-hour band is ₹1502. Rounding a chosen
    number to ₹1500 would silently overrule whoever set it.
    """
    if rule.value_type == "%":
        return _round_to_nearest(one_night_rate * rule.value / 100.0)
    return float(rule.value)


def _format_hour_label(minutes: int) -> str:
    """``780`` -> ``"1:00 PM"`` — how a guest reads a clock."""
    hour24, minute = divmod(minutes, 60)
    suffix = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def _normalise_room_name(name: str | None) -> str:
    """Casefold and squash whitespace, so a category match survives typing."""
    return " ".join(str(name or "").split()).casefold()


def room_allows(cfg: EarlyLateCheckoutConfig, room_names: list[str], direction: str) -> bool:
    """Whether every booked room is eligible for this direction.

    An **empty** ``room_controls`` list means the hotel never configured
    per-room gating, and the feature is not restricted — reading that as "no
    room is allowed" would silently switch the feature off for every hotel that
    has not filled the page in. When the list *is* populated it is exhaustive:
    a room absent from it is not permitted.
    """
    if not cfg.room_controls:
        return True

    flag = "early_check_in" if direction == "arrival" else "late_check_out"
    allowed = {
        _normalise_room_name(control.room_category)
        for control in cfg.room_controls
        if getattr(control, flag, False)
    }
    if not allowed:
        return False
    return all(_normalise_room_name(name) in allowed for name in room_names)


def is_blacked_out(dates: list[str], date_str: str | None) -> bool:
    """Whether the given date is on the hotel's blackout list."""
    if not dates or not date_str:
        return False
    target = str(date_str)[:10]
    return any(str(d)[:10] == target for d in dates)


def _to_minutes(value: object) -> int | None:
    """Minutes since midnight for a policy time, whatever shape it arrives in.

    ``hotel_policies`` stores these as strings ("01:00 PM"), but the module-level
    fallbacks in ``stage_resolver`` are ``datetime.time`` objects — feeding one
    of those to :func:`parse_clock` yields None, which would silently price every
    band against midnight.
    """
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    return parse_clock(value)


def _standard_minutes(policies: dict | None, key: str, default: object) -> int:
    """The hotel's standard check-in/out time in minutes, with a safe fallback."""
    minutes = _to_minutes((policies or {}).get(key))
    if minutes is None:
        minutes = _to_minutes(default)
    return minutes if minutes is not None else 0


_KIND_LABEL = {
    "early_check_in": "Early check-in",
    "late_check_out": "Late check-out",
}


def _build_fee(
    *,
    kind: str,
    amount: float,
    label: str,
) -> dict:
    return {
        "kind": kind,
        "amount": float(amount),
        "label": label,
        "disclaimer": format_fee_disclaimer(kind),
    }


def _price_one_direction(
    *,
    kind: str,
    rule_set,
    requested: str | None,
    standard_min: int,
    date_str: str | None,
    blackout: list[str],
    direction: str,
    room_names: list[str],
    cfg: EarlyLateCheckoutConfig,
    one_night_rate: float,
) -> dict | None:
    """The fee dict for one direction, or None when nothing is charged."""
    if not requested or rule_set is None or not rule_set.rules:
        return None
    if is_blacked_out(blackout, date_str):
        return None
    if not room_allows(cfg, room_names, direction):
        return None

    actual_min = parse_clock(requested)
    if actual_min is None:
        return None

    hours = hours_outside(actual_min, standard_min, direction=direction)
    if hours <= _MIN_HOURS_OUTSIDE:
        return None  # They are not actually early/late — nothing to charge.

    boundary_min = parse_clock(rule_set.boundary)
    beyond_bound = boundary_min is not None and (
        actual_min < boundary_min if direction == "arrival" else actual_min > boundary_min
    )

    if beyond_bound:
        return _build_fee(
            kind=kind,
            amount=_round_to_nearest(one_night_rate * _FULL_DAY_FRACTION),
            label=f"{_KIND_LABEL.get(kind, kind)} (full extra day)",
        )

    rule = select_band(rule_set.rules, hours)
    if rule is None:
        # Beyond every band but inside the hotel's bound. Treat it as a day
        # rather than quoting nothing — the guest is asking for a lot of the
        # hotel's morning, and silence would read as "free".
        return _build_fee(
            kind=kind,
            amount=_round_to_nearest(one_night_rate * _FULL_DAY_FRACTION),
            label=f"{_KIND_LABEL.get(kind, kind)} (full extra day)",
        )

    amount = directional_fee(one_night_rate, rule)
    if amount <= 0:
        return None

    preposition = "before" if direction == "arrival" else "after"
    return _build_fee(
        kind=kind,
        amount=amount,
        label=(
            f"{_KIND_LABEL.get(kind, kind)} "
            f"({hours:g}h {preposition} {_format_hour_label(standard_min)})"
        ),
    )


async def resolve_early_late_fees(
    hotel_id: int | str | None,
    *,
    check_in_date: str | None,
    check_out_date: str | None,
    room_names: list[str],
    one_night_rate: float,
    requested_arrival_time: str | None,
    requested_departure_time: str | None,
) -> list[dict]:
    """Every fee this booking has earned, ready to render. Never raises.

    Returns ``[]`` unless a hotel is configured, the feature is on for it, the
    guest stated a time, and the rooms and dates are eligible.
    """
    if not requested_arrival_time and not requested_departure_time:
        return []

    try:
        if not get_settings().EARLY_LATE_CHECKOUT_FEE_ENABLED:
            return []

        cfg = await get_early_late_checkout_config(hotel_id)
        if cfg is None or not cfg.is_active:
            return []

        from app.repositories.hotel_repository import get_hotel_policies
        from app.services.stage_resolver import DEFAULT_CHECK_IN_TIME, DEFAULT_CHECK_OUT_TIME

        policies = await get_hotel_policies(hotel_id)
        check_in_min = _standard_minutes(policies, "check_in_time", DEFAULT_CHECK_IN_TIME)
        check_out_min = _standard_minutes(policies, "check_out_time", DEFAULT_CHECK_OUT_TIME)

        # A free child or an extra adult does not change what a night costs, so
        # the fee is priced off room rate alone.
        rate = max(0.0, float(one_night_rate or 0.0))
        if rate <= 0:
            return []

        common = dict(
            room_names=list(room_names or []),
            cfg=cfg,
            one_night_rate=rate,
        )
        early = _price_one_direction(
            kind="early_check_in",
            rule_set=cfg.early_check_in,
            requested=requested_arrival_time,
            standard_min=check_in_min,
            date_str=check_in_date,
            blackout=cfg.blackout_early,
            direction="arrival",
            **common,
        )
        late = _price_one_direction(
            kind="late_check_out",
            rule_set=cfg.late_check_out,
            requested=requested_departure_time,
            standard_min=check_out_min,
            date_str=check_out_date,
            blackout=cfg.blackout_late,
            direction="departure",
            **common,
        )

        fees = [f for f in (early, late) if f]
        if fees:
            logger.info(
                "EARLY_LATE_FEE | hotel=%s rooms=%s rate=%.0f fees=%s",
                hotel_id, room_names, rate,
                [(f["kind"], f["amount"]) for f in fees],
            )
        return fees

    except Exception as e:
        # A fee is never worth a failed quote.
        logger.warning(f"early_late_checkout: could not resolve fees for hotel {hotel_id}: {e}")
        return []
