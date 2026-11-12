"""One reading of what a guest means by AC.

Six places in this codebase decided, separately, whether a guest had asked for
AC or for Non-AC, and each spelled the question out as its own list of exact
strings. The lists drifted, so the same sentence resolved in one place and was
silently ignored in another — and the silent case was the expensive one. An
answer nobody could read fell through to `room_searcher`'s "default assumption
is AC", so a guest who wrote *non - ac* was shown air-conditioned rooms, and
was never told the question had gone unanswered.

Everything here runs on one normalised form of the text:

    "Non - A/C Room."   ->  "non ac room"
    "NON_AC"            ->  "non ac"
    "Air-Conditioned"   ->  "air conditioned"

Lowercased; `a/c` read as `ac`; every dash, underscore and run of punctuation
read as a space; runs of whitespace collapsed. After that the matcher has to
know four vocabularies instead of six spellings of the same four.

The hard case is that "without ac" contains the word AC, so any test of the
form "does this mention AC" answers yes to a guest who has just refused it.
Stripping the Non-AC phrases out before asking that question is what keeps the
two apart, and it happens in one place — inside the resolver, and inside
`mentions_both_classes`, which is the resolver's "they named both classes"
test published for callers that need to distinguish a guest comparing the two
from a guest asking for one.

This module reads text, and only text. `ac_preference_from_state` in
`room_packing` reads the *state enum*, where "NO_PREFERENCE" means something
the card treats differently from an unstated preference; it delegates here for
the words and keeps its own contract for the enum.
"""

from __future__ import annotations

import re

__all__ = [
    "normalise_ac_text",
    "ac_preference_from_text",
    "ac_preference_from_prompt_reply",
    "mentions_both_classes",
    "is_bare_ac_class",
    "is_plausible_ac_answer",
]


# ── normalisation ────────────────────────────────────────────────────────────

_A_SLASH_C_RE = re.compile(r"\ba\s*/\s*c\b")
#: hyphens, en/em dashes and underscores, with any space around them, become
#: one space — so "non-ac", "non - ac", "non_ac" and "non ac" are one string.
_DASH_RE = re.compile(r"\s*[-‐-―_]\s*")
_PUNCT_RE = re.compile(r"[.,!?;:'\"()\[\]{}*]+")
_WS_RE = re.compile(r"\s+")


def normalise_ac_text(value: str | None) -> str:
    """The form every matcher in this module reads. See the module docstring."""
    text = (value or "").strip().lower()
    from app.utils.room_card_reply import _normalize_keycaps
    text = _normalize_keycaps(text)
    text = _A_SLASH_C_RE.sub("ac", text)
    text = _DASH_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# ── vocabulary ───────────────────────────────────────────────────────────────

#: "non ac" also covers "nonac" — the space is optional. The Hinglish negative
#: is here because it is the one that fails loudest: "ac nahi chahiye" contains
#: the word AC, and read without this branch it is an AC request.
NON_AC_RE = re.compile(
    r"\b(?:"
    r"non\s*(?:ac|air\s*condition(?:ed|ing|er)?|cooling)|"
    r"(?:no|without|bina)\s+(?:ac|air\s*condition(?:ed|ing|er)?|cooling)|"
    r"(?:ac|air\s*condition(?:ed|ing|er)?|cooling)\s+(?:nahi|nahin|nhi|mat|na|not)|"
    r"mat\s+(?:lagao|karo|do)?\s*ac|"
    r"nahi\s+chahiye\s*ac"
    r")\b"
)
AC_WORD_RE = re.compile(r"\b(?:ac|air\s*condition(?:ed|ing|er)?)\b")
#: Bare "all" and "any" are NOT here, though every site's list had them. As
#: exact matches they were safe; as substrings they fire on a room type —
#: "All Seasons Suite" is not a guest asking to see both room classes. They
#: stay available as whole-message answers, in `_ORDINAL_THREE` below.
NO_PREFERENCE_RE = re.compile(
    r"\b(?:both|dono|donon|sab|no\s+preference|show\s+all|show\s*both)\b"
)

#: A short reply that names none of these is a plausible attempt at an answer;
#: anything that does name one is the guest moving on, and must be left alone.
_MOVED_ON_RE = re.compile(
    r"\b(?:"
    r"check\s*in|check\s*out|rooms?|kamre?|kamra|adults?|guests?|pax|people|persons?|"
    r"children|bachche?|kids?|budget|price|rate|charges?|cost|total|"
    r"cancel|book|booking|photo|photos|menu|food|breakfast|dinner|lunch|"
    r"amenit\w*|wifi|parking|address|location|where|airport|station|"
    r"hi+|hello+|hey+|namaste|thanks|thank\s*you|help|agent|human|manager"
    r")\b"
)


# ── the two questions ────────────────────────────────────────────────────────

def mentions_both_classes(value: str | None) -> bool:
    """Does the guest name Non-AC *and* AC, as two separate things?

    "non ac or ac" does; "without ac" does not — it only looks like it does if
    you ask "does this contain the word ac", which is the trap the strip below
    exists to avoid. `understand`'s comparison detector uses this to tell a
    guest weighing the two classes up from a guest asking for one of them.
    """
    text = normalise_ac_text(value)
    return bool(NON_AC_RE.search(text)) and bool(AC_WORD_RE.search(NON_AC_RE.sub(" ", text)))


def ac_preference_from_text(value: str | None) -> str | None:
    """What the guest means, as "AC" / "NON_AC" / "NO_PREFERENCE" — or None.

    None is a real answer: it means the text carried no AC signal at all, not
    that the guest wants AC. Callers decide what to do with that, and the one
    thing they must not do is pass it off as a preference.
    """
    text = normalise_ac_text(value)
    if not text:
        return None

    has_non_ac = bool(NON_AC_RE.search(text))
    # The AC question is asked of the text with the Non-AC phrases removed,
    # or "without ac" answers yes to both and the guest is shown both classes.
    has_ac = bool(AC_WORD_RE.search(NON_AC_RE.sub(" ", text)))
    wants_both = bool(NO_PREFERENCE_RE.search(text))

    if has_non_ac and (has_ac or wants_both):
        return "NO_PREFERENCE"
    if has_non_ac:
        return "NON_AC"
    if wants_both:
        return "NO_PREFERENCE"
    if has_ac:
        return "AC"
    return None


# ── the answer to the AC question specifically ──────────────────────────────

#: The question lists the classes in order, so a bare ordinal is an answer only
#: because of where it sat on the menu. That is presentation, not language, and
#: it lives here rather than in the matcher above: "2" means Non-AC under this
#: one prompt and something else entirely anywhere else in the flow.
_ORDINAL_ONE = frozenset({
    "1", "1a", "1b", "a", "option 1", "option1", "opt 1", "first", "1st", "pehla", "pahla",
    "option 1 please", "1 please", "1 wala", "1 wali", "ac_yes", "ac yes",
})
_ORDINAL_TWO = frozenset({
    "2", "2a", "2b", "b", "option 2", "option2", "opt 2", "second", "2nd", "doosra", "dusra",
    "option 2 please", "2 please", "2 wala", "2 wali", "ac_no", "ac no",
})
_ORDINAL_THREE = frozenset({
    "3", "3a", "3b", "c", "option 3", "option3", "opt 3", "third", "3rd", "teesra", "tisra",
    "all", "any", "option 3 please", "3 please", "3 wala", "3 wali", "ac_both", "ac both",
})

#: The words that make a string a CLASS and nothing else. Everything in
#: "non ac room", nothing in "AC Deluxe" — which is a class *and* a room name,
#: and has to be read as both.
_AC_CLASS_TOKENS = frozenset({
    "ac", "non", "nonac", "no", "without", "bina", "air", "conditioned",
    "conditioner", "conditioning", "cooling", "with", "wala", "room", "rooms", "only",
    "please", "pls", "both", "dono", "donon", "any", "all", "sab",
    "preference", "show", "1", "2", "3", "nahi", "nahin", "nhi", "mat",
    "option", "opt",
})


_COMPOUND_AC_OVERRIDE_RE = re.compile(
    r"\b\d+\s*(?:rooms?|kamre?|kamra|adults?|guests?|pax|people|persons?|beds?|nights?)\b"
    r"|\b(?:deluxe|premier|family|suite|standard|king|queen|single|twin|triple|tripple|four)\b"
    r"|\b(?:check\s*in|check\s*out|budget|discount|price|rate|booking|book)\b",
    re.IGNORECASE,
)


def is_bare_ac_class(value: str | None) -> bool:
    """Is this string the AC class and nothing else — "non ac room", not "AC Deluxe" or "2 rooms AC"?"""
    text = normalise_ac_text(value)
    if not text:
        return False
    if _COMPOUND_AC_OVERRIDE_RE.search(text):
        return False
    if text in _ORDINAL_ONE or text in _ORDINAL_TWO or text in _ORDINAL_THREE:
        return True
    return all(token in _AC_CLASS_TOKENS for token in text.split())


def ac_preference_from_prompt_reply(value: str | None) -> str | None:
    """A guest's answer to the AC question: by ordinal, or by words."""
    text = normalise_ac_text(value)
    if not text:
        return None
    if text in _ORDINAL_ONE:
        return "AC"
    if text in _ORDINAL_TWO:
        return "NON_AC"
    if text in _ORDINAL_THREE:
        return "NO_PREFERENCE"
    return ac_preference_from_text(text)


def is_plausible_ac_answer(value: str | None) -> bool:
    """Could this unreadable message have been an attempt at an answer?

    True only for a short message that names none of the things a guest moving
    on would name. It is deliberately strict: re-asking is the right response
    to a missed answer and the wrong one to a new request, and a guest who
    writes "3 rooms, 15-17 Oct" while the AC question is open must be let
    through, not held at the question.
    """
    text = normalise_ac_text(value)
    if not text or len(text.split()) > 4:
        return False
    return not _MOVED_ON_RE.search(text)
