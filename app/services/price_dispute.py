"""
A guest who has done the arithmetic differently, and keeps being told a number.

Measured 2026-10-27, guest 919999000111. Staff quoted Rs.17,880 on 25 Oct. The
guest then priced the same stay himself - 2 triple rooms x 4 nights at Rs.1,760,
1 double x 3 nights at Rs.1,480, plus 5% GST, less 10% - and got Rs.17,501. Over
the next two days he asked "please check" five times; every reply re-asserted
Rs.17,880 and none acknowledged that a second number existed. He cancelled.

The bot cannot settle a price. What it can do is stop pretending the question
was not asked: drop the sentences that re-assert our figure, say plainly that
two numbers exist, and put the difference in front of the front desk once.

Everything here is deterministic on purpose. Deciding "is this a dispute?" is
exactly the judgement a model gets wrong quietly - and a false positive replaces
a correct price with an apology.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.utils.sentences import split_sentences

logger = logging.getLogger(__name__)

# Words that mean "the guest is talking about a figure for the stay". Deliberately
# excludes bare "rate" and "per night": those are unit prices, and comparing a
# unit price with a total is how a correct quote gets "corrected".
_TOTAL_CUE = re.compile(
    r"\b(total|payable|bill|amount|grand\s*total|kul|kitna|final)\b",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d{1,2})?)(?![\d])")

# How far after a cue word a figure may sit and still belong to it.
_CUE_WINDOW = 40

# A guest's figure only counts as a rival total when it is in the same league as
# ours. "Total person - 7 adult 3 child" carries a cue and a number; treating 3
# as a disputed total would be nonsense.
_RIVAL_LOW = 0.5
_RIVAL_HIGH = 1.5


def stated_total(text: str) -> float | None:
    """
    The last figure the guest attached to a total-ish word, or None.

    "Last" because a guest who shows their working ends on their answer:
    `Total- 18520/- ... Total- 19446/- ... Total 17501` is a request to accept
    17,501, not 18,520.
    """
    if not text:
        return None

    found: list[float] = []
    squashed = text.replace("*", " ")
    for cue in _TOTAL_CUE.finditer(squashed):
        window = squashed[cue.end(): cue.end() + _CUE_WINDOW]
        match = _NUMBER.search(window)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        try:
            found.append(float(raw))
        except ValueError:
            continue
    return found[-1] if found else None


def rival_total(guest_message: str, stored_total: Any) -> float | None:
    """The guest's figure when it disagrees with ours, otherwise None."""
    try:
        stored = float(stored_total)
    except (TypeError, ValueError):
        return None
    if stored <= 0:
        return None

    guest = stated_total(guest_message)
    if guest is None or guest <= 0:
        return None
    if not (stored * _RIVAL_LOW <= guest <= stored * _RIVAL_HIGH):
        return None
    # A rupee of rounding is not a dispute; the guest's own arithmetic will
    # differ from ours by a few rupees routinely.
    if abs(guest - stored) <= 1.0:
        return None
    return guest


def _mentions_amount(sentence: str, amount: float) -> bool:
    """True if the sentence states `amount`, comma-formatted or not."""
    plain = f"{amount:.0f}"
    squashed = sentence.replace(",", "")
    return bool(re.search(rf"(?<!\d){re.escape(plain)}(?!\d)", squashed))


def scrub(cards: list[Any], stored_total: float) -> list[Any]:
    """
    Remove sentences that assert our price as if it were settled.

    A sentence that also says it is being checked is kept — that is the honest
    version of the same fact. Cards without text pass through untouched.
    """
    out: list[Any] = []
    for card in cards:
        if not (isinstance(card, dict) and card.get("type") == "text"):
            out.append(card)
            continue
        text = card.get("text")
        if not isinstance(text, str) or not text.strip():
            out.append(card)
            continue

        kept = []
        for sentence in split_sentences(text):
            if _mentions_amount(sentence, stored_total) and not re.search(
                r"\b(check|re-?check|verify|confirm(?:ing)?\s+the\s+(?:final\s+)?(?:amount|total))\b",
                sentence,
                re.IGNORECASE,
            ):
                continue
            kept.append(sentence)
        cleaned = re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()
        if cleaned:
            out.append({**card, "text": cleaned})
    return out


def notice(guest_total: float, stored_total: float, is_hi: bool) -> str:
    """The one sentence pair that states both numbers and who will settle it."""
    g = f"₹{guest_total:,.0f}"
    s = f"₹{stored_total:,.0f}"
    if is_hi:
        return (
            f"Aapne {g} calculate kiya hai aur hamare record mein {s} hai.\n\n"
            "Humne front desk se final amount re-check karne ko kaha hai — "
            "wo sahi total isi chat par confirm karenge. "
            "Usse pehle kripya koi aur payment na karein."
        )
    return (
        f"You have calculated {g}, and our records currently show {s}.\n\n"
        "I have asked the front desk to re-check the final amount — they will "
        "confirm the correct total on this chat. Please do not make any further "
        "payment until they do."
    )


async def ensure_dispute_task(
    hotel_id: int | str,
    session_id: str | None,
    phone: str | None,
    booking_id: str | None,
    guest_total: float,
    stored_total: float,
) -> str | None:
    """
    Raise one front-desk task for the discrepancy; never a second one.

    The dedupe is a database question, not a state flag: `save_state` is the last
    node and a flag kept only in the graph's return value would not survive to
    the next turn (LangGraph drops keys that are not in `ConversationState`).
    """
    from app.db.mongodb import hotel_tasks_collection

    variants = [str(hotel_id)]
    if str(hotel_id).isdigit():
        variants.append(int(hotel_id))
    identity = []
    if session_id:
        identity.append({"session_id": session_id})
    if phone:
        identity.append({"customer_phone": phone})
    if not identity:
        return None

    coll = hotel_tasks_collection()
    existing = await coll.find_one(
        {
            "hotel_id": {"$in": variants},
            # BILLING_INQUIRY is the closest declared subtype (it is what a
            # money question to the front desk files under); the metadata tag is
            # what keeps us from swallowing an ordinary invoice request by the
            # same guest.
            "subtype": "BILLING_INQUIRY",
            "metadata.source": "price_dispute_guard",
            "status": {"$in": ["OPEN", "ASSIGNED", "IN_PROGRESS"]},
            "$or": identity,
        }
    )
    if existing:
        return existing.get("task_id")

    from app.workflows.nodes.stage_handlers import create_stage_task

    task_id = await create_stage_task(
        hotel_id=hotel_id,
        session_id=session_id or "",
        title=f"Price Discrepancy (₹{guest_total:,.0f} vs ₹{stored_total:,.0f})",
        description=(
            f"Guest states the total is ₹{guest_total:,.0f}; our record says "
            f"₹{stored_total:,.0f}. Guest has asked more than once. "
            "Please confirm the correct total before asking for any payment."
        ),
        task_type="FRONT_DESK",
        subtype="BILLING_INQUIRY",
        department="FRONT_DESK",
        priority="HIGH",
        booking_id=booking_id,
        customer_phone=phone,
        metadata={
            "guest_total": guest_total,
            "stored_total": stored_total,
            "source": "price_dispute_guard",
        },
    )
    logger.warning(
        "PRICE_DISCREPANCY_TASK | hotel=%s phone=%s guest_total=%.0f stored_total=%.0f task=%s",
        hotel_id, phone or "NONE", guest_total, stored_total, task_id,
    )
    return task_id
