"""Minting a booking id only when the guest has proof of payment.

Until 2026-10-21 `payment/link.py` minted a `BK-…` id and wrote a
`pending_payment` row on the turn the guest typed "confirm", then printed the
id in the payment message. A guest who never paid still held a booking number,
and staff saw a booking in the CRM that was only an unpain quote.

Under `BOOKING_ID_ON_PAYMENT_PROOF_ENABLED` that turn mints a `PAY-…` reference
instead and the real id is minted here — when a payment screenshot arrives, or
at confirmation for a gateway-only payment. The provisional row is renamed to
the new id, so the 30-minute inventory hold survives the rename and
`confirm_pending_booking` still finds the row.
"""

from __future__ import annotations

import logging
from typing import cast

from app.config.settings import get_settings
from app.repositories.booking_repository import reassign_booking_id
from app.services.booking_engine import generate_booking_id
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def mint_booking_id_on_payment_proof(
    state: ConversationState,
) -> tuple[ConversationState, str | None]:
    """
    Return `(state, booking_id)`, minting the id when the flag calls for it.

    Idempotent: a state that already carries a booking id is returned untouched,
    so a screenshot can be checked on more than one turn without minting twice.
    `None` means the flag is off and there is nothing to mint here — the caller's
    old path still applies.
    """
    existing = state.get("booking_id")
    if existing:
        return state, existing

    if not get_settings().BOOKING_ID_ON_PAYMENT_PROOF_ENABLED:
        return state, None

    booking_id = generate_booking_id()
    reference = state.get("payment_reference")
    adopted = await reassign_booking_id(reference, booking_id)
    if reference and not adopted:
        # The provisional row is missing (link creation failed, or the session
        # predates this flag). The id is still valid — `confirm_booking` creates
        # the confirmed row itself — but the inventory hold is gone, so say so.
        logger.warning(
            "BOOKING_ID_MINT | provisional hold %s not found; id %s has no held row",
            reference,
            booking_id,
        )

    return cast(ConversationState, {**state, "booking_id": booking_id}), booking_id
