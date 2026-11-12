"""One way for a guest who has already booked to get an answer from the KB.

Product, 2026-10-11: *"making the bot in faq mode if booking confirmed"*, then
*"faq mode till check out"*. Two stages need this — the confirmed guest and the
in-house guest — and `checked_out` may follow. It lives here rather than copied
into each handler so the two cannot drift, and so the guard below is written
once.

Why it is needed at all: `routing.py:40` returns `stage_confirmed` for every
`lifecycle_stage == "confirmed"` turn *before* the `intent == "FAQ"` branch at
`routing.py:99`, and the in-house router does the same. So the knowledge base is
unreachable for a booked guest unless the stage handler calls it itself. Both
stages used to end in a fixed paragraph instead — so "what is the wifi password",
"do you allow pets" and "is there a restaurant" all got the same non-answer while
the hotel's own knowledge base sat unread.

Three things the call must NOT be allowed to do. Each returns `None`, and the
caller falls back to its own canned line:

  * Move the guest out of the stage. `handle_faq` spreads the incoming state, so
    it preserves the stage today — but that is a property of a different module,
    and re-asserting it here is what keeps the guest where the resolver put them.
  * Sell. Two ways this leaks, and both are guarded below.
  * Raise. `handle_faq` reaches Gemini, ChromaDB, Mongo and the persona
    rewriter; a knowledge-base failure must cost the guest a fallback sentence,
    never a turn with no reply.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── The ZERO SELLING guard ───────────────────────────────────────────────────
#
# `handle_faq`'s own guard is `next_action == "rooms_shown"` — a redirect to
# `handle_room_query`, which shows option cards. That was the leak this file was
# written for and it is checked below. It is NOT the only one, and that is the
# measured finding behind this regex:
#
#   Guest (confirmed): "show me available rooms"
#   Answer:            "Sure, here are the room types available at Sample Hotel:
#                       DELUXE ROOM KING BED … Rate Plans: CP (Breakfast…"
#
# That came back as `next_action == "faq_answered"` — a plain knowledge-base
# answer, waved through by the `rooms_shown` check — because the hotel's room
# list and rate plans are *in* the knowledge base, and Gemini will happily recite
# them. Measured 2026-10-11 against `stayinn_web_clone`; at HEAD the same
# message got the canned paragraph.
#
# So the question is filtered, not the answer. Filtering the answer is tempting
# and wrong: "does the deluxe room have AC?" is a factual question whose answer
# names a room type, and blocking on room names would send that guest to the
# front desk for a question the bot can answer. The set the bot must not serve is
# *inventory and prices* — and that is a property of the question.
_ROOM_WORD = r"(?:rooms?|kamr\w+|suite|deluxe|standard|super\s*deluxe|non[\s-]*ac)"
_PRICE_WORD = r"(?:prices?|pricing|rates?|tariff|costs?|charges?|kitna|kitne\s+ka|how\s+much)"

_SELLING_QUESTION_RE = re.compile(
    # a room word and a price word near each other, in either order
    rf"\b{_ROOM_WORD}\b[^.?!]{{0,32}}\b{_PRICE_WORD}\b"
    rf"|\b{_PRICE_WORD}\b[^.?!]{{0,32}}\b{_ROOM_WORD}\b"
    # asking what is available
    r"|\brooms?\s+available\b|\bavailable\s+rooms?\b"
    r"|\broom\s+(?:options?|list|types?|categories)\b"
    r"|\b(?:show|dikha\w*|list|send)\b[^.?!]{0,24}\b(?:rooms?|kamr\w+)\b"
    r"|\b(?:which|what|kaun\s*sa|kon\s*sa)\b[^.?!]{0,20}\b(?:rooms?|kamr\w+)\b",
    re.IGNORECASE,
)


def is_selling_question(text: str) -> bool:
    """Is this the guest asking what rooms exist, or what they cost?

    Deliberately narrow. A question that merely *mentions* a room — "does the
    deluxe room have AC", "is there hot water in the room" — is not one of these
    and is answered from the knowledge base.
    """
    return bool(_SELLING_QUESTION_RE.search(text or ""))


async def knowledge_base_answer(
    state: dict[str, Any],
    *,
    stage: str,
    log_tag: str,
) -> dict[str, Any] | None:
    """Try to answer this turn from the hotel's knowledge base.

    Returns a state dict ready to be returned by the caller (with `booking_stage`
    and `lifecycle_stage` forced back to `stage`), or `None` to say "not
    answered — use your own canned line".

    `None` is returned when the question is a selling one, when the FAQ node
    declines to answer, or when it raises. All three are the same outcome to the
    caller on purpose: the guest gets a sentence either way.
    """
    user_msg = state.get("user_message", "") or ""
    hotel_id = state.get("hotel_id")

    if is_selling_question(user_msg):
        logger.info(
            "%s_SELLING_GUARD | hotel=%s msg='%s'",
            log_tag, hotel_id, user_msg[:60],
        )
        return None

    try:
        from app.workflows.nodes.faq_handler import handle_faq

        faq_state = await handle_faq(state)
    except Exception as e:
        logger.warning("%s_KB_FAILED | hotel=%s error=%s", log_tag, hotel_id, e)
        return None

    if faq_state.get("next_action") != "faq_answered" or not faq_state.get("bot_response"):
        # `rooms_shown` and friends: the FAQ node wanted to show option cards,
        # which is the right answer in the booking flow and selling here.
        logger.info(
            "%s_KB_SKIPPED | hotel=%s reason='%s' msg='%s'",
            log_tag, hotel_id, faq_state.get("next_action"), user_msg[:60],
        )
        return None

    logger.info(
        "%s_KB_ANSWERED | hotel=%s stage=%s msg='%s'",
        log_tag, hotel_id, stage, user_msg[:60],
    )
    return {
        **faq_state,
        "booking_stage": stage,
        "lifecycle_stage": stage,
    }
