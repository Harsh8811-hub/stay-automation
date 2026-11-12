"""Did the guest ask for the extra bed, refuse it, or say nothing about it?

`quote_generator.py:59` charges or waives the mattress on the strength of two
session fields. Until now only one direction was ever written: `understand.py`
set `no_extra_bed = True` from a regex and **nothing ever set it back**. The flag
was a latch. A guest who said "no extra bed" once — and guests say it early,
before they have seen a price — could not buy a bed for the rest of the session.
Every later turn waived the charge, the hotel lost the money quietly, and no
error was raised anywhere. Same shape as the cache-key bug of 2026-09-24: all
the code involved does its job, and the field that decides the money is simply
never written.

The classifier is deliberately pure — no state, no database, no model call — so
a test can pin it. A money decision made by a regex with no test is how the
first bug survived twelve commits.

**`None` is a real answer and the common one.** It means the guest said nothing
about a bed on this turn, and the caller must change nothing: a "no" given three
turns ago is still a "no" after a message about dates.
"""

from __future__ import annotations

import re

REFUSE = "NO"
REQUEST = "YES"

# The original negative pattern, moved here verbatim from `understand.py:75` so
# that every phrase it already matched keeps matching.
_REFUSAL = re.compile(
    r'\b(?:no\s*extra\s*bed|without\s*extra\s*bed|no\s*bed\s*need|'
    r'extra\s*bed\s*(?:nahi|not|mat|no|nhi)|don\'?t\s*need\s*extra\s*bed|'
    r'adjust\s*without\s*extra\s*bed|bina\s*extra\s*bed|no\s*extra\s*beds)\b'
)

# Asking FOR the bed. Every alternative carries an affirmative word, because the
# bare phrase "extra bed" is a question as often as a request — "is there an
# extra bed?" would otherwise add Rs 600 to a bill nobody agreed to.
_REQUEST = re.compile(
    r'\b(?:add|with|need|want|include|take|book|prefer|get|require|yes|haan|chahiye|lagao)\b'
    r'(?:\W+\w+){0,2}?\W+extra\s*beds?\b'
    r'|\bextra\s*beds?\s*(?:please|chahiye|add|yes|kar\s*do|lagao|de\s*do)\b'
    r'|\b(?:one|1|ek)\s+extra\s*beds?\b'
)

# Refusals the pattern above does not spell out: "we don't want the extra bed",
# "no need of extra bed", "skip the extra bed". Checked BEFORE `_REQUEST` —
# "want extra bed" sits inside "don't want extra bed" and would otherwise read
# as a request for the very thing the guest is declining.
#
# The window is TWO words, and that bound is load-bearing rather than arbitrary.
# At three, "no ac room with extra bed" matched: the `no` belongs to the AC, and
# the three free words "ac", "room", "with" carried it across to the bed, so a
# guest asking FOR a bed was recorded as refusing one. Two is still enough for
# every real refusal ("no need of extra bed", "don't want an extra bed").
_NEGATED_REQUEST = re.compile(
    r'\b(?:no|not|never|without|don\'?t|doesn\'?t|didn\'?t|won\'?t|nahi|nhi|bina|mat|skip|avoid|refuse)\b'
    r'(?:\W+\w+){0,2}?\W+extra\s*beds?\b'
)


def classify_extra_bed(text: str | None) -> str | None:
    """Return "NO" if the guest refused the bed, "YES" if they asked for it.

    Returns `None` when the message says nothing either way — which is most
    messages, including every date, party size and menu choice.
    """
    if not text:
        return None
    cleaned = text.strip().lower()
    if cleaned in ("bed_yes", "with extra bed", "extra bed ke saath"):
        return REQUEST
    if cleaned in ("bed_no", "without extra bed", "bina extra bed"):
        return REFUSE
    if _REFUSAL.search(cleaned) or _NEGATED_REQUEST.search(cleaned):
        return REFUSE
    if _REQUEST.search(cleaned):
        return REQUEST
    return None


# ── The answer to the one-liner ───────────────────────────────────────────
#
# `classify_extra_bed` reads a sentence that mentions a bed. This reads the
# reply to "keep the extra bed? Reply *yes* or *no*", where the guest answers
# with the word itself and never says "bed" at all.
#
# Deliberately narrow: the message must be the answer and nothing else, plus
# filler. "no ac room" is three words beginning with "no", and reading that as
# a refusal of the mattress would re-price a guest who was asking about the AC.
# A reply this function cannot read is not a refusal — the caller keeps the row
# the guest picked, which is the only choice that cannot move them somewhere
# they did not ask to be.
_YES = frozenset({"yes", "y", "yeah", "yep", "ya", "yup", "sure", "keep", "haan", "han", "ha", "ji", "bed_yes"})
# "remove" and "drop" answer the question the card asks — "Keep it?" — as
# directly as "no" does, and a guest who types one of them and is not understood
# keeps a mattress they refused. Only reachable from the bed question, so the
# verb cannot be read as a refusal of anything else.
_NO = frozenset({"no", "nope", "nah", "nahi", "nhi", "nai", "na", "not", "remove", "drop", "bed_no"})
_FILLER = frozenset({
    "please", "pls", "thanks", "thank", "you", "ok", "okay", "sir", "maam",
    "madam", "it", "the", "bed", "continue", "go", "ahead",
    # "ji" is the Hinglish politeness particle and lands in the middle of the
    # reply far more often than at the start: "haan ji" and "nahi ji" were both
    # unreadable without it, and the second one keeps a bed the guest refused.
    "ji",
    # "remove the extra bed" is the same refusal as "no"; the noun between the
    # verb and the end of the sentence must not be what makes it unreadable.
    "extra",
})


def read_prompt_answer(text: str | None) -> str | None:
    """Read a bare yes/no reply. `None` means "not an answer", not "no"."""
    if not text:
        return None
    raw = text.strip().lower()
    if raw in ("bed_yes", "yes"):
        return REQUEST
    if raw in ("bed_no", "no"):
        return REFUSE
    words = re.findall(r"[a-z']+", raw)
    if not words or len(words) > 4:
        return None
    if not all(w in _FILLER for w in words[1:]):
        return None
    if words[0] in _NO:
        return REFUSE
    if words[0] in _YES:
        return REQUEST
    return None


def answer_bed_question(text: str | None) -> str | None:
    """The guest's answer to the post-selection bed question.

    A sentence that names the bed wins (that is `classify_extra_bed`, and it is
    what a guest typing "no extra bed" instead of "no" gets). Otherwise the bare
    token. `None` — neither — means the guest did not answer, and the caller
    keeps the row they picked.
    """
    return classify_extra_bed(text) or read_prompt_answer(text)


def apply_to_state(state: dict, text: str | None) -> str | None:
    """Classify `text` and write the answer into `state`; return the answer.

    A `None` answer writes nothing at all, which is what keeps an earlier "no"
    standing. The three fields are always written together so they cannot drift
    apart — `quote_generator.py:59` reads the first two, `no_extra_bed` is
    exactly "the preference is NO", and `wants_bed` is its mirror image, the one
    the allocator reads to price a mattress. Three spellings of one answer is
    two too many, and this is the only place they are all set at once.
    """
    answer = classify_extra_bed(text)
    if answer is None:
        return None
    state["extra_bed_preference"] = answer
    state["no_extra_bed"] = answer == REFUSE
    state["wants_bed"] = answer == REQUEST
    return answer
