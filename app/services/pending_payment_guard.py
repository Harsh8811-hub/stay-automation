"""
A payment is under verification: no reply may ask for money or hand over a link.

Measured 2026-10-27, guest 919999000111 (SANJU DEY, hotel 111111). He paid a
Rs.4,400 advance on 26 Oct and staff sent the voucher. Over the next day the bot
told him, twice and then again:

    To confirm your booking, please complete the *25% advance payment* ...
    Payment Link: https://razorpay.me/@hotelsahu
    (Your payment verification is currently in progress ...)

and, after his own arithmetic reached the extractor:

    the total due is *Rs.17,880* ... a small balance of *Rs.380* remaining
    You can complete the payment here: https://razorpay.me/@hotelsahu

The two halves are composed by different code paths: the ask comes from a
handler that knows nothing about payments under verification (or from a RAG
article reproduced verbatim), and the "verification in progress" line is glued
under it by `append_booking_resume`, which inspects the stage and never the body
it is appending to. He filed a complaint and cancelled.

This module is the single choke point. It does not try to fix each producer —
`save_state` is the last node before the reply is persisted and sent, so
enforcing there covers every producer at once, including FAQ answers retrieved
from the knowledge base.

What it enforces, and no more:

- a sentence that *asks for payment* is dropped;
- a payment link, VPA or QR is dropped, in a text sentence or as a whole card;
- if that leaves nothing, or if nothing in the reply says the payment is being
  verified, the pending line is stated.

It never adds a number, never changes the stage, and never calls the payment
gateway. A reply that mentions a price without demanding it is left alone — the
guest is allowed to ask what the stay costs while a payment sits with the front
desk.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.utils.sentences import split_sentences

logger = logging.getLogger(__name__)

PENDING_STAGES = frozenset({"payment_check", "payment_pending"})

# A link, a handle, or a QR. Any of these in a reply that should not be selling
# is an invitation to pay, whatever the sentence around it says.
_PAY_LINK = re.compile(
    r"(razorpay\.me|rzp\.io|razorpay\.com/[a-z0-9]|payments?\s+link|payment\s+link|"
    r"upi://|@ybl\b|@okhdfcbank\b|@oksbi\b|@paytm\b|@axl\b|scan\s+(?:the|this)?\s*qr)",
    re.IGNORECASE,
)

# Words that make a sentence a *demand* rather than a statement. "Your payment
# of Rs.4,400 is being verified" must survive; "please complete the payment"
# must not.
_DEMAND = re.compile(
    r"("
    r"advance\s+payment|"
    r"complete\s+(?:the\s+|your\s+)?payment|"
    r"make\s+(?:the\s+|your\s+)?payment|"
    r"proceed\s+with\s+(?:the\s+|your\s+)?payment|"
    r"payment\s+(?:karein|kar\s+dein|kar\s+lein|ke\s+liye)|"
    r"bhugtan|"
    r"pay\s+now|please\s+pay|kindly\s+pay|pay\s+(?:rs\.?|₹|inr)\s*[\d,]|"
    r"total\s+due|amount\s+due|amount\s+payable|payable\s+amount|"
    r"balance\s+(?:of|amount|due|remaining)|remaining\s+balance|"
    r"share\s+the\s+payment\s+link|"
    r"to\s+confirm\s+your\s+booking\s*,?\s*please|"
    r"click\s+here\s+to\s+pay"
    r")",
    re.IGNORECASE,
)

# "verification is in progress" — the honest half. If the reply already says it,
# we do not add a second copy.
_VERIFYING = re.compile(
    r"(verification\s+is\s+(?:currently\s+)?in\s+progress|"
    r"being\s+verified|verif(y|ication)[a-z]*\s+(?:kar|ho)\s*rah)",
    re.IGNORECASE,
)


def payment_under_verification(state: dict[str, Any]) -> bool:
    """True while a claimed payment is waiting on a human."""
    if state.get("payment_status") == "paid":
        return False
    stage = str(state.get("booking_stage") or "")
    lifecycle = str(state.get("lifecycle_stage") or "")
    if stage == "confirmed":
        return False
    return stage in PENDING_STAGES or lifecycle in PENDING_STAGES


# The half of `_DEMAND` that is an *instruction* rather than a label. A label
# ("Balance Due at Check-in: Rs 6,000") states a number; an instruction tells
# the guest to send money. Only the instruction may be removed from a reply to
# a guest whose stay is already confirmed — the number is what they need at the
# desk, and deleting it is how a part-paid confirmation slip came to show
# "Advance Paid: Rs 3,000" and never say what was still owed (the product's PAYMENT
# FIGURE report, 2026-11-06; eval case slip-partial-payment-is-split).
_DEMAND_INSTRUCTION = re.compile(
    r"("
    r"advance\s+payment|"
    r"complete\s+(?:the\s+|your\s+)?payment|"
    r"make\s+(?:the\s+|your\s+)?payment|"
    r"proceed\s+with\s+(?:the\s+|your\s+)?payment|"
    r"payment\s+(?:karein|kar\s+dein|kar\s+lein|ke\s+liye)|"
    r"bhugtan|"
    r"pay\s+now|please\s+pay|kindly\s+pay|pay\s+(?:rs\.?|₹|inr)\s*[\d,]|"
    r"share\s+the\s+payment\s+link|"
    r"to\s+confirm\s+your\s+booking\s*,?\s*please|"
    r"click\s+here\s+to\s+pay"
    r")",
    re.IGNORECASE,
)


def is_payment_ask(sentence: str, *, labels_are_facts: bool = False) -> bool:
    """A sentence that tells the guest to pay, or hands them a way to.

    `labels_are_facts` is for a stay that is already confirmed: there, naming a
    balance is disclosure, and only an instruction (or a link) is an ask.
    """
    if not sentence or not sentence.strip():
        return False
    if _PAY_LINK.search(sentence):
        return True
    if not _DEMAND.search(sentence):
        return False
    if labels_are_facts and not _DEMAND_INSTRUCTION.search(sentence):
        return False
    return True


def _is_link_card(card: Any) -> bool:
    """A card whose whole purpose is to carry a payment route."""
    if not isinstance(card, dict):
        return False
    if card.get("type") in ("payment", "payment_link", "qr", "upi"):
        return True
    for key in ("url", "link", "qr_code_url", "upi_uri"):
        value = card.get(key)
        if isinstance(value, str) and _PAY_LINK.search(value):
            return True
    return False


def enforce(cards: list[Any], state: dict[str, Any], pending_text: str) -> list[Any]:
    """
    Remove payment asks and links; state the pending status if it went missing.

    `pending_text` is the externalised "verification is in progress" line (it
    lives in `booking_resume`, already localised and reference-aware).
    """
    stage = str(state.get("booking_stage") or "")
    lifecycle = str(state.get("lifecycle_stage") or "")
    is_confirmed = (
        stage == "confirmed"
        or lifecycle == "confirmed"
        or bool(state.get("crm_confirmed_stay"))
        or state.get("payment_status") == "paid"
    )

    if not payment_under_verification(state) and not is_confirmed:
        return cards

    out: list[Any] = []
    removed = 0
    announces = False

    for card in cards:
        if _is_link_card(card):
            removed += 1
            continue
        if not (isinstance(card, dict) and card.get("type") == "text"):
            out.append(card)
            continue
        text = card.get("text")
        if not isinstance(text, str) or not text.strip():
            out.append(card)
            continue

        # A confirmed stay may state what is owed; it may not be sold to.
        labels_are_facts = is_confirmed and not payment_under_verification(state)
        kept: list[str] = []
        for sentence in split_sentences(text):
            if is_payment_ask(sentence, labels_are_facts=labels_are_facts):
                removed += 1
                continue
            if _VERIFYING.search(sentence):
                announces = True
            kept.append(sentence)
        cleaned = re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()
        if cleaned:
            out.append({**card, "text": cleaned})

    if removed:
        logger.warning(
            "PENDING_PAYMENT_ASKS_REMOVED | sess=%s removed=%d still_states_pending=%s",
            state.get("session_id"), removed, announces,
        )

    if not is_confirmed and not announces and pending_text:
        if out and isinstance(out[0], dict) and out[0].get("type") == "text":
            out[0] = {**out[0], "text": f"{out[0]['text'].rstrip()}\n\n{pending_text}"}
        else:
            out.insert(0, {"type": "text", "text": pending_text})

    return out
