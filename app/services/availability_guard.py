"""
Availability pre-check for the payment and confirm paths (B2).

Why this module exists
----------------------
Two call sites — `payment/link.py` and `booking_confirmer.py` — asked "is the room we
quoted still free?" with the same three-line expression:

    avail_dict = {r.room_id: r.number_of_rooms for r in available_rooms}
    for r_id, count in requested_counts.items():
        if avail_dict.get(r_id, 0) < count:
            ... "someone else just booked the last available room" ...

That expression answers "sold out" for three different situations, only one of which is
actually sold out:

1. **A failed lookup.** `get_available_rooms` returns `[]` on an error path; every id then
   reads as 0 free, which is indistinguishable from genuine zero inventory. The guest was
   told their room was gone because a query returned nothing.
2. **A keying mismatch.** `RoomType.room_id` is a `str` (schemas.py:222) and the inventory
   service keys by `str(room.room_id)`, but the ids carried on `selected_rooms` come from
   `AllocationOption.model_dump()` and may be `int`. `avail_dict.get(7)` against a `"7"`
   key returns `None`, and `None < count` is true. The room was never looked up.
3. **A defaulted capacity.** `room_repository.py:205` defaults `number_of_rooms` to `1`
   when `total_rooms` is absent, so a two-room request of a real room type can never pass
   `1 < 2` regardless of what the hotel actually has.

Only after those three are ruled out does a shortfall mean what the copy claims.

What this module does NOT do
----------------------------
It does not relax the check. A genuine shortfall still blocks the booking — that guard is
correct and stays. What changes is that a genuine shortfall is now *escalated in the same
turn* instead of dropping the guest back to "search again" with their selection wiped, and
that the other two cases stop being reported to the guest as a race.

Everything here is pure except `raise_availability_ticket`, which makes the ticket. Its
imports are local to the function precisely so that importing this module — which the tests
do — still needs no Mongo, no Chroma and no Gemini key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Verdict statuses.
OK = "ok"
LOOKUP_FAILED = "lookup_failed"
ID_MISMATCH = "id_mismatch"
GENUINE_SHORT = "genuine_short"

# The one message a guest sees when we cannot confirm automatically. Shared by both call
# sites so the two paths cannot drift apart. Deliberately does not say "someone else booked
# it": we have not established that, and saying it about our own keying bug lost bookings.
BLOCKED_GUEST_COPY = (
    "I can't confirm that room automatically just now. I've asked our team to check it, "
    "and they'll confirm with you shortly — your selection is safe with me."
)


def _norm_room_id(value: Any) -> str:
    """
    Normalise a room id to the string form used across the codebase.

    `room_searcher.py:32` and `menu_handler.py:267` both spell this `str(v or "").strip()`.
    `str(None)` is `"None"`, so the `or ""` matters: without it a missing id becomes a
    lookup key that silently matches nothing.
    """
    if value is None:
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class AvailabilityVerdict:
    """The result of the pre-check, with enough detail to write an honest ticket."""

    status: str
    reason: str
    short: tuple[tuple[str, int, int], ...] = ()  # (room_id, needed, free)

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def is_genuine_shortfall(self) -> bool:
        return self.status == GENUINE_SHORT


def requested_counts(selected_rooms: Iterable[dict] | None) -> dict[str, int]:
    """How many of each room type the guest selected, keyed by normalised room id."""
    counts: dict[str, int] = {}
    for room in selected_rooms or []:
        if not isinstance(room, dict):
            continue
        room_id = _norm_room_id(room.get("room_id"))
        if room_id:
            counts[room_id] = counts.get(room_id, 0) + 1
    return counts


def available_counts(available_rooms: Iterable[Any] | None) -> dict[str, int]:
    """
    How many of each room type are free, keyed by normalised room id.

    Reads `room_id` and `number_of_rooms` off either a `RoomType` or a plain dict, and sums
    duplicates rather than letting the last one win.
    """
    counts: dict[str, int] = {}
    for room in available_rooms or []:
        if isinstance(room, dict):
            raw_id = room.get("room_id")
            raw_count = room.get("number_of_rooms")
        else:
            raw_id = getattr(room, "room_id", None)
            raw_count = getattr(room, "number_of_rooms", None)

        room_id = _norm_room_id(raw_id)
        if not room_id:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            count = 0
        counts[room_id] = counts.get(room_id, 0) + count
    return counts


def classify_availability(
    selected_rooms: Iterable[dict] | None,
    available_rooms: Iterable[Any] | None,
) -> AvailabilityVerdict:
    """
    Decide whether the selected rooms are still available.

    Order matters: the cheap "did the lookup even work" checks run before any shortfall is
    believed, because both of the first two statuses produce a shortfall reading that would
    otherwise be indistinguishable from genuine zero inventory.
    """
    needed = requested_counts(selected_rooms)
    if not needed:
        return AvailabilityVerdict(OK, "")

    free = available_counts(available_rooms)

    if not free:
        return AvailabilityVerdict(
            LOOKUP_FAILED,
            f"availability returned no rooms at all, but {len(needed)} room type(s) are "
            f"selected ({sorted(needed)}) — treating as a failed lookup, not sold out",
        )

    missing = sorted(room_id for room_id in needed if room_id not in free)
    if missing:
        return AvailabilityVerdict(
            ID_MISMATCH,
            f"selected room type(s) {missing} are absent from the availability answer, "
            f"which holds {sorted(free)[:8]} — a keying mismatch, not a race",
        )

    short = tuple(
        sorted(
            (room_id, needed[room_id], free[room_id])
            for room_id in needed
            if free[room_id] < needed[room_id]
        )
    )
    if short:
        return AvailabilityVerdict(
            GENUINE_SHORT,
            "; ".join(f"{room_id} needs {need}, {have} free" for room_id, need, have in short),
            short,
        )

    return AvailabilityVerdict(OK, "")


def escalation_updates(verdict: AvailabilityVerdict) -> dict[str, Any]:
    """
    State updates that raise a BOOKING_DISPUTE ticket in the same turn.

    `escalation_required` alone is not enough on these paths: the graph only routes to
    `handle_escalation` on the *next* turn's dispatch, so a guest who never replies would
    never get a ticket. Callers must therefore `await evaluate_escalation(...)` with these
    fields applied, rather than just setting the flags.

    `BOOKING_DISPUTE` is in `REQUIRES_INVESTIGATION` and `BOOKING_MISMATCH` is in
    `INVESTIGATION_REASON_CODES` (escalation_engine.py:41-52), so a ticket is created.
    `OTHER` is in `KNOWLEDGE_ANSWERABLE_CODES` and would silently create nothing.
    """
    return {
        "escalation_required": True,
        "escalation_issue_type": "BOOKING_DISPUTE",
        "escalation_reason_code": "BOOKING_MISMATCH",
        "escalation_reason": f"{verdict.status}: {verdict.reason}",
    }


async def raise_availability_ticket(state: dict, verdict: AvailabilityVerdict) -> dict:
    """
    Create the ticket, in this turn, for a booking we could not confirm.

    Two things this has to get right:

    1. **It must run now.** `routing.py:30-49` only reaches `handle_escalation` on the
       *next* turn's dispatch, so setting `escalation_required` and stopping would leave a
       guest who never replies with no ticket and no contact — exactly the reported bug.
       `BOOKING_MISMATCH` is in `INVESTIGATION_REASON_CODES` (escalation_engine.py:45), and
       step 4 of `evaluate_escalation` checks that *before* the knowledge-base lookup, so
       this costs one Gemini call for the staff summary and no Chroma query.

    2. **It must never raise.** A ticketing outage cannot be allowed to swallow the guest's
       reply. On failure the caller still sends `BLOCKED_GUEST_COPY`; what is lost is the
       ticket, which is logged with the full verdict so it can be found and re-raised.

    Returns the escalation result dict, or `{}` if the attempt failed.
    """
    payload = {**state, **escalation_updates(verdict)}
    try:
        from app.services.escalation_engine import evaluate_escalation

        return await evaluate_escalation(payload) or {}
    except Exception:  # noqa: BLE001 — a ticket failure must not lose the guest's reply
        logger.exception(
            "availability escalation failed (status=%s, hotel=%s, selection=%s) — "
            "the guest reply is still sent, but this needs a ticket raised by hand",
            verdict.status,
            state.get("hotel_id"),
            verdict.short or "n/a",
        )
        return {}
