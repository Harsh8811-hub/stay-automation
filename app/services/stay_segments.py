"""
A stay whose room count changes part-way through, cut into flat segments.

Why this exists
---------------
Every part of the bot below the conversation assumes ONE set of rooms held for
one date range: `allocate_rooms` returns a fixed list, `PricingEngine.get_price`
takes a single check-in/check-out for all of them, and the booking row has one
pair of dates. the product's request (2026-10-29) is the shape that has no home in
that model:

    "3 rooms 3 days, 2 rooms last day"
    "1 room before checkin"

He offered two ways to do it — step by step, or combined. There is a third that
costs much less: **cut the stay into segments whose room count never changes,
and let each segment be an ordinary booking.** Three rooms for two nights is one
booking; two rooms for the third night is another. Nothing below the
conversation has to learn a new shape.

What that costs, and why it is affordable
-----------------------------------------
The guest ends up holding two bookings and paying twice. product accepted that
on 2026-10-29. The things that would normally make it unacceptable do not:

  * GST does not move. `booking_engine` prices inclusively and picks the slab
    per room per night, not from a booking total, so cutting a stay in two
    leaves the guest's price and the tax rate alone.
  * Inventory subtraction is already date-scoped —
    `inventory_service` counts an existing booking only on the nights it
    overlaps — so segment B is priced against the rooms segment A did not take.
  * The state can already hold two bookings in one conversation; see
    `app/services/second_booking.py`.

This module is the arithmetic only
----------------------------------
Text in, plain dicts out: no model call, no database, no state, no I/O. The
conversation decides *what* the guest said; this decides whether that answer is
one a hotel could actually honour, and it says no rather than guessing. That
split is deliberate — a mis-parse here produces a confident, plausible, wrong
price, which is the worst failure this bot has.

The rule the whole module enforces is tiling: the segments must cover every
night of the requested stay exactly once, with no gap, no overlap, and no
segment asking for more rooms than were originally quoted.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

#: Ceiling on how many segments will be cut out of one request. A real stay has
#: two or three; a number that large means the declaration is malformed (or the
#: guest is describing a year of building work), and turning that into two
#: hundred bookings is not a service. Over this, the caller escalates.
MAX_SEGMENTS = 6

#: Ceiling on how many nights `enumerate_nights` will walk. The dates come from
#: a guest message, so a mistyped year ("2026 to 3026") would otherwise be a
#: 365,000-iteration loop inside a request handler. `quote-365-nights` is the
#: measured precedent for what an unbounded long-stay path costs this bot: it
#: times out the turn at 120s. Over the ceiling the range is treated as
#: unreadable, which escalates rather than quotes.
MAX_NIGHTS = 400


def _as_date(value) -> date | None:
    """`YYYY-MM-DD`, a `date`, or a `datetime` — anything else is None.

    The repo stores dates as strings and compares them as strings in places, so
    both shapes reach here. Returning None rather than raising keeps every
    caller's failure mode "we could not read this", which is the answer that
    escalates instead of quoting.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def enumerate_nights(check_in, check_out) -> list[str]:
    """Every night in the stay, check-in inclusive and check-out exclusive.

    "5 Nov to 7 Nov" is two nights: the 5th and the 6th. The guest leaves on the
    7th and does not sleep there. Getting this off by one is the single most
    likely error in the whole feature — it is the difference between a 9
    room-night stay and an 8 — so it lives in one function with a test rather
    than in each caller.
    """
    start, end = _as_date(check_in), _as_date(check_out)
    if not start or not end or end <= start:
        return []
    if (end - start).days > MAX_NIGHTS:
        return []
    nights: list[str] = []
    cursor = start
    while cursor < end:
        nights.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return nights


def night_after(night: str) -> str:
    """The date one night after `night`, as `YYYY-MM-DD`. '' when unreadable."""
    day = _as_date(night)
    return (day + timedelta(days=1)).isoformat() if day else ""


def validate_night_map(
    night_map: dict[str, int],
    check_in,
    check_out,
) -> list[str]:
    """
    Every reason this declaration cannot be honoured, as plain sentences.

    An empty list means it tiles. The caller must treat any other result as
    "ask, then escalate" — never as a rounding problem to be smoothed over. A
    gap night means the guest is homeless for one night of their own booking; an
    overlap means two bookings claim the same room on the same night, which the
    hotel discovers at check-in.

    The messages are written for a staff ticket, not for the guest.
    """
    problems: list[str] = []
    nights = enumerate_nights(check_in, check_out)
    if not nights:
        return [f"the stay dates are not readable ({check_in!r} to {check_out!r})"]

    if not night_map:
        return ["no nights were declared"]

    declared = set(night_map)
    expected = set(nights)

    missing = [n for n in nights if n not in declared]
    if missing:
        problems.append(
            "these nights are not covered by any segment: " + ", ".join(missing)
        )

    extra = [n for n in sorted(declared) if n not in expected]
    if extra:
        problems.append(
            "these nights fall outside the stay and cannot be booked: "
            + ", ".join(extra)
        )

    bad = sorted(n for n, c in night_map.items() if not isinstance(c, int) or c < 1)
    if bad:
        problems.append(
            "these nights have a room count that is not a positive whole number: "
            + ", ".join(bad)
        )

    if len(compress(night_map)) > MAX_SEGMENTS:
        problems.append(
            f"the request needs more than {MAX_SEGMENTS} separate bookings"
        )

    return problems


def compress(night_map: dict[str, int]) -> list[dict]:
    """
    Run-length encode a per-night room count into contiguous segments.

    {"2026-11-05": 3, "2026-11-06": 3, "2026-11-07": 2}
        -> [{05 to 07, 3 rooms}, {07 to 08, 2 rooms}]

    Only nights present in the map are returned, and only runs of equal count
    merge. The output is the input to booking: each segment is one date range
    plus one room count, which is exactly what the existing flow already knows
    how to price.

    Unreadable or non-positive counts break a run rather than being skipped into
    their neighbours — silently absorbing a bad night would change the guest's
    total without anyone seeing it.
    """
    if not night_map:
        return []

    segments: list[dict] = []
    current: dict | None = None

    for night in sorted(night_map):
        count = night_map[night]
        usable = isinstance(count, int) and count >= 1
        if (
            current is not None
            and usable
            and current["rooms"] == count
            and current["check_out"] == night
        ):
            current["check_out"] = night_after(night)
            continue

        current = None
        if not usable:
            continue
        current = {
            "check_in": night,
            "check_out": night_after(night),
            "rooms": count,
        }
        segments.append(current)

    return segments


def total_room_nights(night_map: dict[str, int]) -> int:
    """Room-nights the whole stay costs. The number the guest is really buying."""
    return sum(c for c in night_map.values() if isinstance(c, int) and c > 0)


def describe(segments: list[dict]) -> str:
    """
    One plain line per segment, for the guest to check against their own plan.

    Deliberately says "nights", not "days": the guest said "3 days", the hotel
    sells nights, and the two disagree by one every single time. Restating it in
    the hotel's unit is how a disagreement is caught before it becomes a
    booking.
    """
    lines = []
    for seg in segments:
        rooms = seg.get("rooms")
        word = "room" if rooms == 1 else "rooms"
        lines.append(f"• {seg.get('check_in')} to {seg.get('check_out')}: {rooms} {word}")
    return "\n".join(lines)


def night_map_from_segments(segments: list[dict]) -> dict[str, int]:
    """
    The inverse of `compress`, for validating a declaration that arrived as
    segments rather than as a per-night map.

    Overlapping segments are not merged — the later one wins the night, and the
    overlap is caught by `validate_night_map` only if the counts are then wrong.
    That is why callers should prefer validating the per-night map they built
    themselves. This exists so a model's segment list can be turned into
    something checkable.
    """
    night_map: dict[str, int] = {}
    for seg in segments or []:
        rooms = seg.get("rooms")
        for night in enumerate_nights(seg.get("check_in"), seg.get("check_out")):
            night_map[night] = rooms
    return night_map
