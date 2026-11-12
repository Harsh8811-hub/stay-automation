from datetime import datetime, timezone, timedelta
from app.workflows.state import ConversationState

import logging
logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_iso() -> str:
    """Current time in IST, ISO 8601 format."""
    return datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def _multi_stay_enabled() -> bool:
    """Is the combined-quote flow on? Read lazily so settings stay optional."""
    try:
        from app.config.settings import get_settings

        return bool(getattr(get_settings(), "MULTI_STAY_COMBINED_ENABLED", False))
    except Exception:  # settings must never break a queue advance
        return False


def queue_line(index: int, booking: dict) -> str | None:
    """One numbered line describing a queued booking, or None if it says nothing."""
    bits: list[str] = []
    check_in = booking.get("check_in")
    check_out = booking.get("check_out")
    if check_in and check_out:
        from app.utils.helpers import format_display_date

        bits.append(
            f"{format_display_date(str(check_in))} to {format_display_date(str(check_out))}"
        )
    rooms = booking.get("num_rooms")
    room_type = booking.get("room_type_preference")
    if rooms:
        bits.append(f"{int(rooms)} room{'s' if int(rooms) > 1 else ''}")
    if room_type:
        bits.append(str(room_type))
    adults = booking.get("adults")
    if adults:
        bits.append(f"{int(adults)} guests")
    if not bits:
        return None
    return f"*{index}.* " + " · ".join(bits)


def describe_queue(queue: list[dict]) -> str:
    """
    What the guest is told when one message contained several bookings.

    Two shapes share the queue. A group-booking list names a *guest* per entry,
    and "let's complete the booking for **Asha**" is the right sentence for it.
    A multi-stay request names no guests at all — "2 deluxe 20-22 Dec, 1 family
    23-25 Dec" is one guest with two stays — so that wording is replaced with
    the stays themselves, which is the only way the guest can check we read the
    dates right before we spend two turns quoting.

    Lives here rather than in a node because the card is drawn by
    `room_searcher` and the text reply by `response_generator`: one shape of
    sentence, two renderers, so it must not be written twice. Never raises on a
    half-filled entry — a line that renders nothing is left out.
    """
    lines = [line for i, b in enumerate(queue, 1) if (line := queue_line(i, b))]
    named = any(b.get("guest_name") for b in queue)

    if named:
        from app.utils.helpers import format_display_date

        first = queue[0]
        who = first.get("guest_name") or "the first guest"
        dates = (
            f"({format_display_date(str(first['check_in']))} to "
            f"{format_display_date(str(first['check_out']))})"
            if first.get("check_in") and first.get("check_out")
            else ""
        )
        return (
            f"I found **{len(queue)} booking requests** in your message.\n\n"
            f"Let's complete the booking for **{who}** {dates} first. Once that's "
            f"finished, I'll automatically continue with the next booking so you "
            f"won't need to send the details again."
        )

    body = "\n".join(lines) if lines else f"**{len(queue)} bookings**"
    nxt = 2 if len(queue) > 1 else 1
    return (
        f"I read **{len(queue)} bookings** in your message:\n{body}\n\n"
        f"Let's finish *booking 1* first — I'll carry on with booking {nxt} "
        f"straight after, so you won't need to repeat the details."
    )


def advance_booking_queue(state: ConversationState, terminal_status: str) -> dict:
    """
    Called when the active booking reaches a terminal state
    (confirmed, cancelled, failed, abandoned, skipped, expired).

    Stamps timestamps on the current booking, advances the index, and
    loads the next booking's fields into state.

    Per-booking audit trail fields:
        created_at   — set once at queue creation
        last_updated — updated on every status transition
        completed_at — set when a terminal status is reached
    """
    queue = list(state.get("booking_queue", []) or [])
    active_idx = state.get("active_booking_index", 0)
    now = _now_iso()

    completed = list(state.get("completed_bookings", []) or [])
    failed    = list(state.get("failed_bookings",    []) or [])

    # ── Stamp the current booking ──────────────────────────────
    if len(queue) > active_idx:
        current = dict(queue[active_idx])           # shallow copy — don't mutate in-place
        current["queue_status"]   = terminal_status
        current["terminal_status"] = terminal_status
        current["last_updated"]   = now
        current["completed_at"]   = now             # terminal = done, regardless of outcome
        queue[active_idx] = current

        from app.observability.metrics import metrics
        metrics.incr(f"queue.{terminal_status}")

        queue_id = state.get("queue_id", "UNKNOWN_QUEUE")
        logger.info(
            f"QUEUE_{terminal_status.upper()}: "
            f"Booking {active_idx + 1} of {len(queue)} in {queue_id} | "
            f"guest={current.get('guest_name')} | completed_at={now}"
        )

        if terminal_status == "confirmed":
            completed.append(current)
        else:
            failed.append(current)

    updates = {
        "completed_bookings": completed,
        "failed_bookings":    failed,
        "booking_queue":      queue,
    }

    # ── Advance to next booking ────────────────────────────────
    if active_idx + 1 < len(queue):
        next_idx     = active_idx + 1
        next_booking = dict(queue[next_idx])
        next_booking["queue_status"] = "processing"
        next_booking["last_updated"] = now
        # completed_at stays None until it reaches a terminal state
        queue[next_idx] = next_booking

        updates["active_booking_index"] = next_idx
        updates["booking_queue"]        = queue

        from app.observability.metrics import metrics
        metrics.incr("queue.advanced")
        queue_id = state.get("queue_id", "UNKNOWN_QUEUE")
        logger.info(
            f"QUEUE_ADVANCED: index {active_idx} → {next_idx} in {queue_id} | "
            f"next_guest={next_booking.get('guest_name')}"
        )

        # Load next booking's fields into active session
        fields_to_load = [
            "adults", "children", "num_rooms",
            "check_in", "check_out",
            "room_type_preference",
            "guest_name", "guest_phone", "guest_email",
        ]
        for f in fields_to_load:
            updates[f] = next_booking.get(f)

        # Reset session-level transient state
        updates.update({
            "booking_stage":        "collecting",
            "next_action":          "search",
            "selected_rooms":       [],
            "available_rooms":      [],
            "allocation_options":   [],
            "price_breakdown":      None,
            "total_price":          None,
            "payment_link":         None,
            "payment_status":       None,
            "booking_id":           None,
            "intent":               "BOOK_ROOM",
            "room_type_not_available": False,
        })

        # A stay that was already chosen in a combined quote carries its own
        # selection. Restoring it means booking two goes to its own payment
        # link without being re-shown a room card the guest has already
        # answered — one link per booking is the whole point of collecting
        # them together. Gated by the same flag that wrote the selection.
        selection = next_booking.get("selection") or {}
        already_chosen = False
        if selection and _multi_stay_enabled():
            for field in ("selected_rooms", "total_price", "price_breakdown", "selected_option_label"):
                if selection.get(field) is not None:
                    updates[field] = selection[field]
            updates.update({
                "booking_stage": "quote_shown",
                "next_action": "quote_shown",
                # Per-stay asks from the previous booking do not carry over to
                # this one. Same list `multi_stay._COLLECTION_RESET` clears.
                "room_requirements": None,
                "ac_preference": None,
                "ac_qualified": False,
                "meal_plan_preference": None,
                "no_extra_bed": False,
                "extra_bed_preference": None,
                "requested_arrival_time": None,
                "requested_departure_time": None,
                "price_override": None,
            })
            already_chosen = True

        # Build transition message for the guest
        from app.prompts import templates
        name  = next_booking.get("guest_name", "the next guest")
        ci    = next_booking.get("check_in")
        co    = next_booking.get("check_out")
        dates = f"({ci} to {co})" if ci else ""

        if terminal_status == "confirmed":
            status_msg = "The previous booking is confirmed."
        elif terminal_status == "cancelled":
            status_msg = "The previous booking was cancelled."
        else:
            status_msg = "The previous booking could not be completed."

        if already_chosen:
            from app.utils.multi_stay import entry_label

            amount = float(selection.get("total_price") or 0.0)
            msg = (
                f"{status_msg}\n\n"
                f"Next: *{entry_label(next_booking, next_idx)}* — you already chose "
                f"it at **₹{amount:,.0f}**. Type **'Confirm'** to get its own "
                f"payment link."
            )
        else:
            msg = f"{status_msg}\n\nNow, let's continue with the booking for **{name}** {dates}."
        updates["bot_response_prepend"] = templates.text_card(msg)
        updates["queue_advanced"]       = True

    else:
        # ── All bookings exhausted ─────────────────────────────
        queue_id = state.get("queue_id", "UNKNOWN_QUEUE")
        logger.info(f"QUEUE_FINISHED: All {len(queue)} bookings done in {queue_id}")
        updates.update({
            "booking_queue":       [],
            "active_booking_index": 0,
            "queue_id":            None,
        })

    return updates
