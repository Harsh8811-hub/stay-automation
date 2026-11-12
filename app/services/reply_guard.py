"""
Do not ask a guest for something the bot already has.

Measured on a real conversation (guest 919051141676, 2026-10-18). Check-in had
already been recorded as 28/10, and the bot then asked for it four times in
three minutes:

    06:39:02  check-in date ab 28/10/2026 aur check-out 29/10/2026 hai  ✅
    06:39:15  check-out 31/10/2026 updated ✅ …
              Please share your check-in date and number of guests.
    06:39:25  number of adults updated to 10 ✅
              What is your check-in date?
    06:41:02  [a correct 28–31 Oct allocation card]
              What is your check-in date?            (same second)

The guest answered "Check in 28 october" and was asked for it again anyway.
Nothing was broken about the extraction; the reply was generated as though the
slot were empty.

The cost of a wrong question is not one extra turn. In the same conversation
"Non AC" arrived as its own fragment, so the guest saw a question they had
already answered and had to answer again — which is what makes a burst of
messages feel like the bot is not listening.

What this does
--------------
A deterministic post-filter on the reply. It reads the state that was just
saved and removes any sentence that asks for a field that is already filled.

It never empties a reply. If the only thing the bot had to say was one of these
questions, the turn goes out unchanged: the guest is waiting, and silence tells
them nothing, while a repeat at least tells them the bot is still there. The
loop this would otherwise break is already caught by the repeat counter and
escalation in `routing.py`.

What it deliberately does not do
--------------------------------
It does not touch non-text cards. A room card's example ("tell me — 3 rooms")
is content, not a request for a missing slot, and rewriting card bodies is how
a formatting change turns into a lost quote.

It does not judge whether a question was *useful*. Only whether its answer is
already known.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# ── What a sentence is asking for ────────────────────────────────────

_CHECK_IN = re.compile(r"check[\s\-_]?in|arrival|\baane\b|\bpahunch", re.IGNORECASE)
_CHECK_OUT = re.compile(r"check[\s\-_]?out|departure", re.IGNORECASE)
#: "dates" alone names both ends of the stay. Mapped to both rather than to
#: neither: a guest asked for "your dates" when one of them is known is still
#: being asked for the one that is known.
#:
#: "stay" is deliberately NOT a cue on its own. It appears in questions that
#: have nothing to do with dates — S57's "Kya aap is *confirmed stay* ke alawa
#: ek naya alag booking karna chahte hain?" was dropped by this guard for a
#: confirmed guest, so the guest read the "reply Yes/No" line with the question
#: it answers missing. Measured 2026-10-27 by
#: `test_s57_stage_2_ek_aur_room_chahiye`. A real date question says "dates".
_DATES = re.compile(r"\bdate[s]?\b|\btarikh\b", re.IGNORECASE)
_GUESTS = re.compile(
    r"\badults?\b|\bguests?\b|\bpersons?\b|\bpeople\b|\blog\b|\bkam log\b", re.IGNORECASE
)
_ROOMS = re.compile(r"\brooms?\b|\bkamre?\b|\bkamra\b", re.IGNORECASE)

#: A sentence must *ask* before it can be dropped. Without this a summary —
#: "Your check-in is 28 Oct" — reads as a request and gets deleted.
_ASK_CUE = re.compile(
    r"^(please|kindly|plz|share|tell|provide|send|confirm|let me know|"
    r"bata\w*|batayein|apni|apna|aap)\b",
    re.IGNORECASE,
)

#: Questions about *when they arrive*, not *which day*. The early/late
#: check-in flow asks these legitimately even when the date is known.
_NOT_A_SLOT_QUESTION = re.compile(r"\bearly\b|\blate\b|\btime\b", re.IGNORECASE)

#: Sentence-ish chunks, whitespace preserved so the reply can be rebuilt
#: byte-for-byte when nothing is removed.
_SENTENCE = re.compile(r"[^.!?\n]*[.!?]+[ \t]*|[^.!?\n]+|\n")


def _fields_asked(sentence: str) -> set[str]:
    asked: set[str] = set()
    if _CHECK_IN.search(sentence):
        asked.add("check_in")
    if _CHECK_OUT.search(sentence):
        asked.add("check_out")
    # "dates" on its own names both ends of the stay — but "check-in date"
    # names one, and expanding it to both would keep a question whose answer
    # is already known.
    if not asked and _DATES.search(sentence):
        asked.update(("check_in", "check_out"))
    if _GUESTS.search(sentence):
        asked.add("guests")
    if _ROOMS.search(sentence):
        asked.add("rooms")
    return asked


def _known_fields(state: Any) -> set[str]:
    """Which of the fields this module understands the state already holds."""
    get = state.get if hasattr(state, "get") else (lambda _k, _d=None: None)
    known: set[str] = set()
    if get("check_in"):
        known.add("check_in")
    if get("check_out"):
        known.add("check_out")
    if get("adults") or get("effective_adults") or get("children"):
        known.add("guests")
    if get("num_rooms") or get("selected_rooms"):
        known.add("rooms")
    return known


def _is_asking(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return False
    if _NOT_A_SLOT_QUESTION.search(text):
        return False
    return text.endswith("?") or bool(_ASK_CUE.match(text))


def _should_drop(sentence: str, known: set[str]) -> bool:
    if not _is_asking(sentence):
        return False
    asked = _fields_asked(sentence)
    # Every field it asks for is already answered. A sentence that also asks
    # for something still missing ("dates and number of guests") is kept: it
    # is doing real work, even if part of it is redundant.
    return bool(asked) and asked <= known


def _strip(text: str, known: set[str]) -> str:
    kept = [part for part in _SENTENCE.findall(text) if not _should_drop(part, known)]
    cleaned = "".join(kept)
    # Dropping a sentence leaves the blank line that separated it. Collapse
    # runs of blank lines, and never leave a trailing separator behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_questions_about_known_data(
    cards: Iterable[Any] | None, state: Any
) -> list[Any]:
    """Remove text-card sentences that ask for a field the state already has.

    Returns a new list. Cards are only replaced when their text actually
    changes, so an untouched reply is returned with its identity intact.
    """
    response = list(cards or [])
    known = _known_fields(state)
    if not known:
        return response

    cleaned: list[Any] = []
    for card in response:
        if not (isinstance(card, dict) and card.get("type") == "text"):
            cleaned.append(card)
            continue
        text = card.get("text")
        if not isinstance(text, str) or not text.strip():
            cleaned.append(card)
            continue

        stripped = _strip(text, known)
        if stripped == text.strip():
            cleaned.append(card)
            continue
        if not stripped:
            logger.info(
                "REPLY_GUARD_DROPPED | every sentence asked for a known field | known=%s",
                sorted(known),
            )
            continue

        logger.info(
            "REPLY_GUARD_STRIPPED | known=%s | before=%r | after=%r",
            sorted(known),
            text[:120],
            stripped[:120],
        )
        cleaned.append({**card, "text": stripped})

    if not cleaned and response:
        # Stripping every card would leave the guest with nothing at all, and
        # they are waiting on a reply. A redundant question is bad; silence
        # after the guest's message is worse, because there is no signal at
        # all that the bot is still there. So the guard edits replies and
        # never empties one — the loop this would have broken is already
        # caught by the repeat counter and escalation.
        logger.info(
            "REPLY_GUARD_KEPT | every card was a known-field question, but the "
            "reply would have been empty | known=%s",
            sorted(known),
        )
        return response

    return cleaned
