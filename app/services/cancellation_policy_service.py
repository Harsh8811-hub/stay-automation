"""The hotel's cancellation and refund policy, in the guest's words.

One implementation, because there were about to be two. The logic here was
lifted out of `cancellation_handler.py`, which had been the only reader of
`payment_policies.cancellation_rules` — and which a confirmed guest can never
reach, because `routing.py` sends every confirmed guest to `stage_confirmed`
first. Item 4 (Product, 2026-10-10) needs the same text in that handler, and a
second copy of a refund table is exactly the kind of thing that drifts: one copy
gets a fix, the other keeps quoting the old percentages to a guest who is about
to lose money.

The rules are read from `payment_policies`, scoped to the hotel with a
`"default"` fallback document, and rendered as a bullet per band. A hotel with no
row — or a row with no `cancellation_rules` key — falls back to the same generic
line `cancellation_handler` has always shown, so an unconfigured hotel behaves
exactly as before.

`get_cancellation_policy_parts` is the exact-parity entry point (the two blocks
the handler prints under its own headings); `get_cancellation_policy_text` is the
combined block for callers that just want one string.
"""

from __future__ import annotations

import logging

from app.db.mongodb import payment_policies_collection

logger = logging.getLogger(__name__)

# Shown when the hotel has configured no bands. Wording unchanged from
# `cancellation_handler` — a guest told one thing yesterday must not be told
# something else today.
DEFAULT_CANCELLATION_POLICY = (
    "Cancellation policy: Free cancellation up to 72 hours before check-in. "
    "Cancellations within 24 hours will receive 25% refund."
)

# The refund half of the answer in every case: the bands say what is refundable,
# this says when the money arrives.
DEFAULT_REFUND_POLICY = (
    "Refund policy: Refunds for eligible cancellations are processed within "
    "5-7 business days back to the original payment method."
)


def _format_rule(hours_before_check_in: int, refund_percent: int) -> str:
    """One band, in the wording `cancellation_handler` already used."""
    if refund_percent == 100:
        return f"• Up to {hours_before_check_in} hours before check-in: Full Refund"
    if refund_percent == 0:
        return f"• Within {hours_before_check_in} hours of check-in: No Refund"
    return (
        f"• Up to {hours_before_check_in} hours before check-in: "
        f"{refund_percent}% Refund"
    )


async def get_cancellation_policy_parts(hotel_id: int | str) -> tuple[str, str]:
    """``(cancellation_block, refund_line)``. Never raises.

    A Mongo failure returns the generic default rather than an empty string: a
    guest asking to cancel must always be told *something* about the money, and
    silence reads as "no policy applies".
    """
    policy_doc = None
    try:
        policy_coll = payment_policies_collection()
        policy_doc = await policy_coll.find_one({"hotel_id": str(hotel_id)})
        if not policy_doc:
            policy_doc = await policy_coll.find_one({"hotel_id": "default"})
    except Exception as e:
        logger.warning(f"cancellation_policy: lookup failed for hotel {hotel_id}: {e}")

    if not policy_doc or "cancellation_rules" not in policy_doc:
        return DEFAULT_CANCELLATION_POLICY, DEFAULT_REFUND_POLICY

    try:
        # Furthest-out band first: the guest reads what they still get before
        # what they have already lost.
        ordered = sorted(
            policy_doc["cancellation_rules"],
            key=lambda r: r.get("hours_before_check_in", 0),
            reverse=True,
        )
        rule_texts = [
            _format_rule(r.get("hours_before_check_in", 0), r.get("refund_percent", 0))
            for r in ordered
        ]
    except Exception as e:
        logger.warning(
            f"cancellation_policy: could not parse rules for hotel {hotel_id}: {e}"
        )
        return DEFAULT_CANCELLATION_POLICY, DEFAULT_REFUND_POLICY

    return "Cancellation policy:\n" + "\n".join(rule_texts), DEFAULT_REFUND_POLICY


async def get_cancellation_policy_text(hotel_id: int | str) -> str:
    """The policy as one guest-facing block, for a handler with no headings."""
    cancel_policy, refund_policy = await get_cancellation_policy_parts(hotel_id)
    return f"{cancel_policy}\n\n{refund_policy}"
