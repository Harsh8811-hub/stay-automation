"""Answer "what do your rooms cost?" for a guest who is already booked.

The booking flow already knows how to answer this. `handle_menu`'s
``SHOW_PRICES`` branch builds the list — room name, occupancy, the EP nightly
rate from the *calendar* (never `rate_plans.base_price`, which nothing keeps in
sync), a meal-plan note and the room photos — and the new-guest menu has used it
since the beginning. The problem was routing, not the builder: `routing.py`
sends every ``lifecycle_stage == "confirmed"`` turn to `stage_confirmed` and
returns, so a booked guest could never reach it, and the two places that came
close both refused instead (see the flag's comment in `settings.py`).

So this is `kb_answer.knowledge_base_answer`'s shape for a different builder:
call the node, keep the stage, never raise. It lives in its own module rather
than in `stage_confirmed.py` because `stage_in_house` will want the same call
the moment an in-house guest asks for next season's rates.

Two limits worth stating, because the list looks more authoritative than it is:

  * With no dates in the message the list is priced **from today** — that is
    `inventory_service.get_available_rooms`' documented date-less behaviour, and
    the reply says "standard pricing" rather than quoting the guest's stay.
  * Availability is deliberately not applied for the same reason: there is no
    stay to check it for. The reply asks for dates, which is the step that turns
    this into a real quote.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def room_price_answer(
    state: dict[str, Any],
    *,
    stage: str,
    log_tag: str,
) -> dict[str, Any] | None:
    """Try to answer this turn with the room price list.

    Returns a state dict ready to be returned by the caller (with
    `booking_stage` and `lifecycle_stage` forced back to `stage`), or `None` to
    say "not answered — use your own canned line".

    `None` is returned when the catalogue is empty, when the menu node produces
    no reply, or when it raises. All three are the same outcome to the caller on
    purpose: a pricing failure must cost the guest a fallback sentence, never a
    turn with no reply.
    """
    hotel_id = state.get("hotel_id")

    try:
        from app.workflows.nodes.menu_handler import handle_menu

        menu_state = await handle_menu({**state, "intent": "SHOW_PRICES"})
    except Exception as e:
        logger.warning("%s_PRICE_LIST_FAILED | hotel=%s error=%s", log_tag, hotel_id, e)
        return None

    if not menu_state.get("bot_response"):
        logger.info("%s_PRICE_LIST_EMPTY | hotel=%s", log_tag, hotel_id)
        return None

    logger.info(
        "%s_PRICE_LIST_ANSWERED | hotel=%s stage=%s msg='%s'",
        log_tag, hotel_id, stage, (state.get("user_message") or "")[:60],
    )
    # `handle_menu` spreads the incoming state, so the stage survives today —
    # but that is a property of a different module, and re-asserting it here is
    # what keeps a confirmed guest confirmed. Same reason as `kb_answer`.
    return {
        **menu_state,
        "booking_stage": stage,
        "lifecycle_stage": stage,
    }
