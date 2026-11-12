"""
Legacy field names the StayInn dashboard's booking aggregate still sums.

Why this exists
---------------
The dashboard backend (`stayinn-backend-newdb`, branch `new-db`) reads the
*same* collection the bot writes — `WEB_MONGO_URI` points at the self-hosted
`stayinn_web`, and `dashboard.py` does `web_db["bookings"]`. Measured
2026-10-27: `GET /api/dashboard/bookings?hotelId=111111` returned

    {"totalBookings": 319, "totalGuests": 0, "totalAdults": 0,
     "totalRevenue": 0.0, "bookingStates": {}, "roomTypes": {}}

319 rows found, every number zero. The rows are there; the aggregation is
summing keys we do not write:

    {"$sum": {"$ifNull": ["$Guests", 0]}}
    {"$sum": {"$ifNull": ["$Adults", 0]}}
    {"$sum": {"$ifNull": ["$Children", 0]}}
    {"$sum": {"$ifNull": ["$Pricing.total", {"$ifNull": ["$pricing.total", 0]}]}}
    {"$push": "$BookingState"}
    {"$push": "$RoomType"}

Our documents carry `adults`, `children`, `num_rooms`, `total_price`,
`booking_status`, `rooms` — lowercase throughout, as `build_booking_document`
and `create_booking` write them.

Two ways to close the gap: rename the six keys in their aggregation, or write
both spellings from here. product chose the bot side on 2026-10-27, so this
module is the only place that knows the panel's field names. Nothing in the
bot reads these aliases — they exist solely so the panel's aggregate stops
reporting zero.

Every alias is derived from the canonical field at write time, never stored as
the source of truth, so the two can only disagree if someone hand-edits one.
The whole thing is gated by `PANEL_BOOKING_FIELD_ALIASES_ENABLED`; `False`
stops the writes immediately and the panel goes back to reporting zeros, which
is the pre-2026-10-27 behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Derived, never authored. Listed for readers and for the backfill tool.
PANEL_ALIAS_FIELDS = ("Adults", "Children", "Guests", "Pricing", "BookingState", "RoomType")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def room_type_label(rooms: Any) -> str:
    """
    One label per booking, because the panel's `$push: "$RoomType"` goes
    straight into a `Counter` — a list here would make every booking its own
    bucket.

    Repeated rooms collapse ("DELUXE ROOM KING BED", not "…, …"), so two rooms
    of the same type still land in one bucket; genuinely mixed room types are
    joined with " + " so the breakdown stays honest.
    """
    if not isinstance(rooms, list):
        return ""
    seen: list[str] = []
    for room in rooms:
        if not isinstance(room, dict):
            continue
        name = (room.get("room_name") or room.get("room_type") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return " + ".join(seen)


def party_counts(doc: dict[str, Any]) -> tuple[int, int]:
    """
    (adults, children) for the dashboard, with the room allocation as fallback.

    The top-level counts are absent on a large share of rows — measured
    2026-10-27 on hotel 111111: 169 rows carry `adults: None` and 53 have no
    `adults` key at all, because the payment hold is built from a session that
    never received them (`payment/link.py` → `state_to_session` →
    `build_booking_document`). Summing the canonical field alone therefore
    reports zero adults for roughly half the bookings, which is the same wrong
    answer the dashboard already gives.

    So: when the canonical count is missing, sum what each room was actually
    allocated (`assigned_adults` / `assigned_children` — the numbers the rate
    was priced on). A present value is never overridden, even a zero.
    """
    rooms = doc.get("rooms") if isinstance(doc.get("rooms"), list) else []
    room_adults = sum(_int(r.get("assigned_adults")) for r in rooms if isinstance(r, dict))
    room_children = sum(_int(r.get("assigned_children")) for r in rooms if isinstance(r, dict))

    adults = doc.get("adults")
    children = doc.get("children")
    return (
        _int(adults if adults is not None else room_adults),
        _int(children if children is not None else room_children),
    )


def aliases_for_document(doc: dict[str, Any]) -> dict[str, Any]:
    """The full alias set for a booking document, derived from its canonical fields."""
    adults, children = party_counts(doc)
    return {
        "Adults": adults,
        "Children": children,
        "Guests": adults + children,
        # Nested because that is the path their pipeline dereferences. Nothing
        # else writes `Pricing`, so a whole-object `$set` cannot clobber a
        # sibling key today; if one is ever added, this must become a dotted
        # path instead.
        "Pricing": {"total": _float(doc.get("total_price") or doc.get("amount"))},
        "BookingState": doc.get("booking_status") or "",
        "RoomType": room_type_label(doc.get("rooms")),
    }


def aliases_for_status(booking_status: str | None) -> dict[str, Any]:
    """
    The subset that changes with a status transition.

    Status is the one canonical field that moves without the document being
    rewritten, and a stale `BookingState` is worse than a zero: it would show a
    cancelled booking under "confirmed". Callers pass the status they are
    writing in the same update.
    """
    if not booking_status:
        return {}
    return {"BookingState": booking_status}


def _enabled() -> bool:
    try:
        return bool(get_settings().PANEL_BOOKING_FIELD_ALIASES_ENABLED)
    except Exception as e:  # noqa: BLE001 — a settings read must never block a write
        logger.warning("PANEL_ALIASES_SETTINGS_READ_FAILED | %s", e)
        return False


def apply_aliases(doc: dict[str, Any]) -> dict[str, Any]:
    """Merge the alias set into an outgoing document. Returns the same dict."""
    if _enabled():
        doc.update(aliases_for_document(doc))
    return doc


def status_aliases(booking_status: str | None) -> dict[str, Any]:
    """Status-only alias fields, for an update that changes `booking_status`."""
    return aliases_for_status(booking_status) if _enabled() else {}
