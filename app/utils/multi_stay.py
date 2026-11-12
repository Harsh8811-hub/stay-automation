"""
Collecting several bookings in one conversation, then quoting them together.

One message can name two stays — "2 deluxe 20-22 Dec, 1 family 23-25 Dec". The
queue has held both since 2026-11 (see `queue_manager`), but only the *first*
stay was ever collected: the guest picked a room, got a quote for stay one and
a payment link, and stay two was forgotten until after that payment cleared.

What this module does is move the selection, not the payment. When the guest
picks a room for a stay that is not the last in the queue, the choice is stashed
on that queue entry and the conversation moves on to the next stay's card. Once
the last stay is picked, the guest sees one quote that itemises every stay and a
grand total — then pays one link per booking, which is what happens today and
what the hotel's gateway expects.

Payment for booking *n* still needs `selected_rooms`/`total_price` in the flat
session state, so the combined quote rewinds `active_booking_index` to 0 and
loads stay one's stashed selection. The stash is the source of truth: every
function here reads the queue entry, never a copy.

Gate: `MULTI_STAY_COMBINED_ENABLED`, off by default. With it off nothing in
this module is called and the old one-stay-at-a-time behaviour is exact.
"""

from __future__ import annotations

import logging

from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

#: Session fields a selection is made of. Stashed together because a payment
#: link needs all of them and a half-restored selection prices a different room.
SELECTION_FIELDS = (
    "selected_rooms",
    "total_price",
    "price_breakdown",
    "selected_option_label",
    "check_in",
    "check_out",
    "adults",
    "children",
    "num_rooms",
    "room_type_preference",
)

#: Transient session fields a change of stay invalidates. Mirrors the reset in
#: `queue_manager.advance_booking_queue` on purpose: the same hand-off, one
#: terminal status earlier.
_COLLECTION_RESET = {
    "booking_stage": "collecting",
    "next_action": "search_rooms",
    "selected_rooms": [],
    "available_rooms": [],
    "allocation_options": [],
    "price_breakdown": None,
    "total_price": None,
    "payment_link": None,
    "payment_status": None,
    "booking_id": None,
    "room_type_not_available": False,
    "extra_bed_asked": False,
    "quote_photos_sent": False,
    # Per-stay asks. Stay one's early check-in, mattress, meal plan or coupon
    # must never be priced onto stay two, which is a different set of nights in
    # a different room. `room_requirements` especially: left in place, stay
    # one's "2 deluxe" allocates stay two's rooms as well, so a family room
    # booked for the second stay would be searched as two deluxes.
    "room_requirements": None,
    "ac_preference": None,
    "ac_qualified": False,
    "meal_plan_preference": None,
    "no_extra_bed": False,
    "extra_bed_preference": None,
    "requested_arrival_time": None,
    "requested_departure_time": None,
    "coupon_code": None,
    "applied_coupon": None,
    "discount_amount": 0.0,
    "price_override": None,
}


def enabled() -> bool:
    """Is the combined-quote flow switched on for this deploy?"""
    from app.config.settings import get_settings

    return bool(getattr(get_settings(), "MULTI_STAY_COMBINED_ENABLED", False))


def _queue(state: ConversationState) -> tuple[list[dict], int]:
    queue = list(state.get("booking_queue", []) or [])
    try:
        idx = int(state.get("active_booking_index", 0) or 0)
    except (TypeError, ValueError):
        idx = 0
    return queue, max(0, min(idx, len(queue) - 1)) if queue else 0


def is_collecting(state: ConversationState) -> bool:
    """True when more than one booking is queued and the gate is on."""
    if not enabled():
        return False
    if state.get("combined_quote_shown"):
        # Already quoted together. From here the queue behaves as it always
        # has: `active_booking_index` is 0 and a modification re-prices booking
        # one, not the whole queue.
        return False
    queue, _ = _queue(state)
    return len(queue) > 1


def entry_label(booking: dict, index: int) -> str:
    """How one queued booking is named in a combined quote.

    A group list names its guest; a multi-stay request names its dates. The two
    shapes share the queue, so the label follows the entry rather than the flow.
    """
    from app.utils.helpers import format_display_date

    name = booking.get("guest_name")
    if name:
        return f"Booking {index + 1} — {name}"
    check_in, check_out = booking.get("check_in"), booking.get("check_out")
    if check_in and check_out:
        return (
            f"Booking {index + 1} — {format_display_date(str(check_in))} to "
            f"{format_display_date(str(check_out))}"
        )
    return f"Booking {index + 1}"


def stash_selection(state: ConversationState, quote_text: str) -> tuple[list[dict], dict]:
    """Write the current stay's selection onto its queue entry.

    Returns the new queue and the new *current* entry. The entry keeps the
    rendered quote text as well as the raw fields: the combined card is built
    from it later, and re-pricing a past stay at that point would risk pricing
    it against today's dates and today's catalogue instead of the ones the
    guest agreed to.
    """
    queue, idx = _queue(state)
    if not queue:
        return queue, {}
    current = dict(queue[idx])
    current["selection"] = {
        field: state.get(field) for field in SELECTION_FIELDS
    }
    current["selection"]["quote_text"] = quote_text
    queue[idx] = current
    return queue, current


def advance_to_next_stay(state: ConversationState, queue: list[dict]) -> dict:
    """Load the next queued booking's details into the flat session state."""
    _, idx = _queue(state)
    next_idx = idx + 1
    if next_idx >= len(queue):
        return {}
    nxt = dict(queue[next_idx])
    nxt["queue_status"] = "processing"
    queue[next_idx] = nxt

    updates: dict = {"booking_queue": queue, "active_booking_index": next_idx}
    for field in (
        "adults",
        "children",
        "num_rooms",
        "check_in",
        "check_out",
        "room_type_preference",
        "guest_name",
        "guest_phone",
        "guest_email",
    ):
        updates[field] = nxt.get(field)
    updates.update(_COLLECTION_RESET)

    # Rebuild this stay's own room requirement from its own entry. The queue
    # entry carries a preferred room type and a count; the parser's shape is a
    # list of {room_type, count}, so the search for stay two is bounded by stay
    # two's rooms and not stay one's.
    rtype = nxt.get("room_type_preference")
    rooms = nxt.get("num_rooms")
    if rtype and rooms:
        updates["room_requirements"] = [{"room_type": rtype, "count": int(rooms)}]

    # Carry a per-stay plan/AC preference that the guest already stated.
    for field in ("meal_plan_preference", "ac_preference"):
        if nxt.get(field):
            updates[field] = nxt[field]

    updates["next_action"] = "search_rooms"
    return updates


def stash_and_advance(state: ConversationState, quote_text: str) -> dict | None:
    """Stash this stay, then hand the conversation to the next one.

    Returns state updates when there *is* a next stay, or None when this was the
    last one and the caller should build the combined quote instead.
    """
    queue, idx = _queue(state)
    if not queue or idx + 1 >= len(queue):
        return None

    queue, current = stash_selection(state, quote_text)
    if not current:
        return None
    advance = advance_to_next_stay({**state, "booking_queue": queue}, queue)
    if not advance:
        return None

    from app.prompts import templates

    nxt_label = entry_label(queue[advance["active_booking_index"]], advance["active_booking_index"])
    body = (
        f"{quote_text}\n\n"
        f"*{entry_label(current, idx)}* is noted at "
        f"**₹{float(current['selection'].get('total_price') or 0):,.0f}**.\n\n"
        f"Now let's choose the room for *{nxt_label}*."
    )
    advance["bot_response"] = templates.text_card(body)
    logger.info(
        "MULTI_STAY_STAGED: booking %s of %s selected, moving to %s",
        idx + 1,
        len(queue),
        advance["active_booking_index"] + 1,
    )
    return advance


def stash_and_combine(state: ConversationState, quote_text: str) -> dict:
    """Stash the last stay and return the combined quote as state updates."""
    queue, idx = _queue(state)
    queue, _ = stash_selection(state, quote_text)
    _cards, updates = combined_quote(queue, idx)
    return updates


def combined_quote(queue: list[dict], current_idx: int) -> tuple[list[dict], dict]:
    """Build the one quote that names every stay and the grand total.

    Returns the cards to send and the state updates that make the *first*
    booking payable: the flat `selected_rooms`/`total_price` pair a payment link
    is minted from is rewound to stay one, and each stay's own total stays on
    its queue entry for the link that follows its confirmation.
    """
    if not queue:
        return [], {}

    parts: list[str] = []
    grand_total = 0.0
    for i, booking in enumerate(queue):
        selection = booking.get("selection") or {}
        total = float(selection.get("total_price") or 0.0)
        grand_total += total
        text = selection.get("quote_text")
        if not text:
            # A stay that never got a selection is a stay the guest has not
            # agreed to; say so rather than printing a blank 0.
            text = "_No room chosen for this stay yet._"
        parts.append(f"*{entry_label(booking, i)}*\n{text}")

    first = (queue[0].get("selection") or {}) if queue else {}
    first_total = float(first.get("total_price") or 0.0)
    min_advance = round((first_total * 0.30) / 10) * 10
    n = len(queue)

    body = "\n\n".join(parts)
    body += (
        f"\n\n*Grand total: ₹{grand_total:,.0f}* across {n} bookings.\n"
        f"Each booking is paid separately. Type **'Confirm'** or **'Yes'** to get the "
        f"advance link for {entry_label(queue[0], 0)} "
        f"(30% = ₹{min_advance:,.0f}); the next stays follow one by one."
    )

    card = {
        "type": "quote",
        "text": body,
        "data": {
            "combined": True,
            "grand_total": grand_total,
            "stays": [
                {
                    "label": entry_label(b, i),
                    "total": float((b.get("selection") or {}).get("total_price") or 0.0),
                }
                for i, b in enumerate(queue)
            ],
        },
        "buttons": [
            {
                "id": "quote_confirm",
                "title": "Confirm Booking",
                "label": "Confirm Booking",
                "action": "postback",
                "value": "confirm booking",
            }
        ],
    }

    updates = {
        "booking_queue": queue,
        "active_booking_index": 0,
        "booking_stage": "quote_shown",
        "next_action": "quote_shown",
        "price_breakdown": first.get("price_breakdown"),
        "total_price": first.get("total_price"),
        "selected_rooms": first.get("selected_rooms") or [],
        "selected_option_label": first.get("selected_option_label"),
        "check_in": first.get("check_in") or queue[0].get("check_in"),
        "check_out": first.get("check_out") or queue[0].get("check_out"),
        "adults": first.get("adults") or queue[0].get("adults"),
        "children": first.get("children") or queue[0].get("children"),
        "bot_response": [card],
        "combined_quote_shown": True,
    }
    logger.info(
        "MULTI_STAY_COMBINED: %s bookings, grand_total=%s, quoted at index %s",
        n,
        grand_total,
        current_idx,
    )
    return updates["bot_response"], updates
