"""
Replies to the rooms card.

The card prints one lettered list — "A. {set}", "B. {set}", "C. {set}" when the
hotel has a third tier, then a trailing "D. Tell me how you want to split — I'll
quote it" — and this module is the only place that decides what a bare `B` means
on it. It runs from `apply_intent_overrides`, before the generic numeric handler,
because that handler would otherwise read `2` as "the second priced option" from
a list the card did not show.

Both vocabularies are answered: the letters are what the card prints, and the
digits are the same positions, kept because guests who were shown the older
numbered card still have it on screen. One list of choices is what removed the
old ambiguity — rooms 1-4 bearing numbers while the combinations bore letters,
so a number meant a room type and a letter meant a priced set.

Note the range is not fixed. A two-row card ends at C, and `Reply *A-C*` is what
it prints, because the build-your-own line is `rows + 1` — a fixed A-D would
name a letter that resolves to nothing.

Scoping is the whole safety argument. The resolver answers only while
`get_active_prompt(state) == "rooms_shown"`, which is the action `room_searcher`
sets when it emits the card. Outside that window a bare letter or number keeps
whatever meaning it already had — most importantly the AC menu, which is live
under `need_ac_preference` and whose `1`/`2`/`3` mean AC / Non-AC / Both.

Pure functions over `state`: no I/O, no model call, so the routing can be
tested directly.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: The one prompt action this module answers to. Kept as a constant rather than
#: a literal at each use so the scoping is greppable in one place.
ROOM_CARD_ACTION = "rooms_shown"

#: A reply is a choice, not a sentence. Anything longer is a guest explaining
#: themselves and belongs to the LLM path.
_MAX_REPLY_CHARS = 24

#: "room 2", "option 2", "no. 2", "R2" → the leading word is noise, the number
#: is the answer. The card prints "B." and "Reply *A-C*", and guests type all of
#: those and more.
#: The trailing `\b` is load-bearing. Without it the alternation matches the
#: shortest branch first and `rooms 2` strips only `room`, leaving `s 2` — which
#: is not a digit, so the reply silently fell through. The boundary also stops
#: `no` from eating the front of words like `none`.
_LEADING_KEYWORD = re.compile(
    r"^(?:rooms?|options?|opts?|choice|selection|number|no)\b[.:\-]?\s*",
    re.IGNORECASE,
)

#: Intents for which a stray digit must never be read as a card choice — the
#: same list the generic numeric handler refuses.
_NEVER_A_CHOICE = frozenset({
    "CANCEL_BOOKING", "PAYMENT_STATUS", "SHOW_AMENITIES", "CONFIRM_BOOKING",
    "COMPLAINT", "HUMAN_REQUEST", "FAQ",
})


_KEYCAP_DIGITS: dict[str, str] = {
    "1️⃣": "1", "1\u20e3": "1", "1\ufe0f\u20e3": "1", "①": "1", "❶": "1",
    "2️⃣": "2", "2\u20e3": "2", "2\ufe0f\u20e3": "2", "②": "2", "❷": "2",
    "3️⃣": "3", "3\u20e3": "3", "3\ufe0f\u20e3": "3", "③": "3", "❸": "3",
    "4️⃣": "4", "4\u20e3": "4", "4\ufe0f\u20e3": "4", "④": "4", "❹": "4",
    "5️⃣": "5", "5\u20e3": "5", "5\ufe0f\u20e3": "5", "⑤": "5", "❺": "5",
    "6️⃣": "6", "6\u20e3": "6", "6\ufe0f\u20e3": "6", "⑥": "6", "❻": "6",
    "7️⃣": "7", "7\u20e3": "7", "7\ufe0f\u20e3": "7", "⑦": "7", "❼": "7",
    "8️⃣": "8", "8\u20e3": "8", "8\ufe0f\u20e3": "8", "⑧": "8", "❽": "8",
    "9️⃣": "9", "9\u20e3": "9", "9\ufe0f\u20e3": "9", "⑨": "9", "❾": "9",
    "🔟": "10",
}

def _normalize_keycaps(text: str) -> str:
    """Normalize unicode keycaps and circled digits to standard ASCII strings."""
    if not text:
        return ""
    res = text
    for k, v in _KEYCAP_DIGITS.items():
        res = res.replace(k, v)
    return res


def _active_prompt(state: dict[str, Any]) -> str:
    """
    `next_action`, falling back to `last_prompt_action`.

    Imported lazily from `understand` rather than reimplemented, so there is one
    definition of "the active prompt" in the codebase. It has to be lazy:
    `understand` imports `intent_overrides` at module level, which imports this
    module, so a top-level import here would be a cycle.
    """
    from app.workflows.nodes.understand import get_active_prompt

    return get_active_prompt(state)


def _token(text: str) -> tuple[str, Any] | None:
    """
    Reduce a reply to ("num", 2) or ("letter", "B"), or None if it is not one.

    Deliberately narrow. A message that is not *just* a choice is not a choice —
    "2 rooms for 5 people" must reach the LLM, not be read as "room 2".
    """
    t = _normalize_keycaps((text or "").strip().rstrip(".!,")).strip()
    if not t or len(t) > _MAX_REPLY_CHARS:
        return None

    t = _LEADING_KEYWORD.sub("", t).strip()
    if not t or len(t) > 12:
        return None

    if t.isdigit():
        return ("num", int(t))

    if len(t) == 1 and t.isalpha():
        return ("letter", t.upper())

    return None


def _choice(state: dict[str, Any], number: int) -> tuple[str, dict[str, Any]] | None:
    """
    Line N of the card → that priced option; one line past them → build your own.

    The selection path deliberately reuses the existing option-selection
    contract instead of resolving the option itself: it sets
    `room_type_preference` to "option N", which is exactly what a guest typing
    "option 2" produces, and `validator._resolve_room_selection` then matches it
    against the live `allocation_options` — which the card sets to its own lines
    in its own order, so "option 2" is the row the guest is looking at.

    `shown` is capped at `ROOM_CARD_MAX_COMBOS` as a belt-and-braces bound: the
    card itself never draws more than three priced lines, but `allocation_options`
    is state, and state can be stale from a turn that ran an older card. A
    number past the cap resolves to nothing rather than to an option nobody saw.
    """
    options = state.get("allocation_options") or []
    if not options:
        return None

    from app.services.room_service import ROOM_CARD_MAX_COMBOS

    shown = min(len(options), ROOM_CARD_MAX_COMBOS)

    if 1 <= number <= shown:
        return "SELECT_ROOM", {
            "room_type_preference": f"option {number}",
            "selected_option_number": number,
        }

    if number == shown + 1:
        # Mirrors validator.py:468-477, which reaches the same state when
        # `_resolve_room_selection` recognises its own build-your-own marker.
        # Set outright here because the marker it tests for is derived from
        # option labels that the new card never writes.
        return "BOOK_ROOM", {
            "booking_stage": "collecting",
            "next_action": "need_custom_rooms",
            "room_type_preference": None,
            "room_type_not_available": False,
            "selected_rooms": [],
            "room_requirements": None,
        }

    return None


def resolve_room_card_reply(
    user_message: str, state: dict[str, Any], intent: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    """
    (intent, state_extras) for a reply to the room card, or None to fall through.

    `intent` is passed rather than read from state: it is this turn's
    classification, which the caller holds and which `state` may not yet carry.

    None is the common case and means "this was not a card reply" — the caller
    then behaves exactly as it did before this module existed.
    """
    try:
        if _active_prompt(state) != ROOM_CARD_ACTION:
            return None
        if str(intent or state.get("intent") or "").upper() in _NEVER_A_CHOICE:
            return None

        parsed = _token(user_message)
        if parsed is None:
            return None

        kind, value = parsed
        if kind == "num":
            return _choice(state, value)

        # Letters are the card's own vocabulary now; the digits below them are
        # the alias, kept for guests holding a numbered card in their
        # scrollback: 1→A, 2→B, 3→build your own. Both are the same positions,
        # so the alternative to aliasing is a guest replying "2" to a card that
        # is still on screen and getting nothing back.
        return _choice(state, ord(value) - ord("A") + 1)
    except Exception:
        # A reply parser must never be the reason a guest gets no answer. Any
        # surprise here degrades to the pre-existing behaviour.
        logger.exception("ROOM_CARD_REPLY | resolver failed | input=%r", user_message)
        return None
