"""Collecting the name for the booking slip after payment, not before it.

See `NAME_AFTER_PAYMENT_ENABLED`. The pre-payment rule was "no name, no payment
link": the guest was stopped the moment they said confirm and asked whose name
the booking was in. That is friction at the point a guest has already decided to
pay, and it asks for a confirmation detail before there is anything to confirm.

Now the link goes out on "confirm", the booking row is created with whatever
name we hold (the WhatsApp profile name if it looks like one, otherwise
"Guest"), and the confirmation turn asks for the name to print on the slip.
This node is what reads the answer.

It is reached only when the message actually parses as a name — see the branch
in `routing.route_by_intent`. A guest who replies with a question instead is
answered normally; the pending flag simply waits for a name-shaped message, and
the booking keeps the name it was created with.
"""

from __future__ import annotations

import logging

from app.prompts.templates import text_card
from app.repositories.booking_repository import update_booking_guest_name
from app.utils.person_name import looks_like_a_name_answer
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


def _booking_id(state: ConversationState) -> str | None:
    return (
        state.get("booking_id")
        or state.get("active_booking_id")
        or (state.get("active_booking") or {}).get("booking_id")
    )


async def handle_name_after_payment(state: ConversationState) -> ConversationState:
    """Write the name the guest gave after paying onto their booking."""
    name = looks_like_a_name_answer(state.get("user_message") or "")
    booking_id = _booking_id(state)

    # `None` is "this was not a name answer", never "use the whole message" —
    # see the predicate's docstring. Routing only sends a name this way, so this
    # is a belt-and-braces branch rather than the normal path.
    if not name:
        logger.info(
            "NAME_AFTER_PAYMENT_UNPARSED | hotel=%s booking=%s — prompt left open",
            state.get("hotel_id"), booking_id,
        )
        return {
            **state,
            "bot_response": text_card(
                "Sorry, I couldn't catch the name. Could you write just the "
                "name as it appears on the ID?"
            ),
        }

    updated = await update_booking_guest_name(booking_id, name) if booking_id else False
    if not updated:
        # The row is missing or cancelled — say only what is true. The name is
        # still kept in state so the confirmation slip fields can use it.
        logger.warning(
            "NAME_AFTER_PAYMENT_NO_ROW | hotel=%s booking=%s name=%s",
            state.get("hotel_id"), booking_id, name,
        )

    ref = f" **{booking_id}**" if booking_id else ""
    return {
        **state,
        "guest_name": name,
        "name_after_payment_pending": False,
        "bot_response": text_card(
            f"Thank you! Your booking slip{ref} will carry the name **{name}**.\n\n"
            f"If that is not the name you want on it, just reply with the correct one."
        ),
    }
