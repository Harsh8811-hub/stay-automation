"""A guest naming a property that is not the one this number belongs to.

Measured 2026-11-05, Sample Hotel. A guest wrote:

    We want to Stay at Hotel Samman. 3 Rooms Required.
    Check In- 25-12-2026, Check out- 29-12-2026

The bot answered "Got it! I've updated your details… How many guests will
be staying?" and went on collecting a booking for *Sahu*. Nobody told the
guest they had the wrong number. That is the worst shape a defect can
take here: it is silent, and the guest only finds out at the front desk
on arrival.

`app/utils/competitor_mention.py` is the near neighbour and the wrong one
to reuse. It knows OTA names and price-comparison phrases, and its job is
to *block a discount*, not to tell a guest they dialled the wrong hotel.

Deliberately conservative. A false positive costs the guest a confusing
sentence; a false negative leaves the silent booking above. So the rule
is: a name only counts when the guest either capitalised it (a proper
noun) or wrote it straight after "stay at / book at / move to".
"""

from __future__ import annotations

import re
from typing import Any

# "Hotel", "Resort" … open a property name, but they are never part of it.
# Scoped `(?i:…)` rather than a flag on the whole pattern: the name branch
# below must stay case-*sensitive* to mean "proper noun", and a global flag
# would let `[A-Z]` match a lowercase word and call it a hotel.
_PROPERTY_WORD = r"(?i:hotel|resort|inn|palace|lodge|guest\s*house|homestay)"

# Up to three capitalised words: "Samman", "Samman Palace", "The Samman
# Palace". Three is enough for real names and short enough that a sentence
# trailing off after the name cannot be swallowed whole.
_CAPITALISED = re.compile(
    rf"\b{_PROPERTY_WORD}\s+"
    r"(?P<name>[A-Z][a-z'’]+(?:[ \t]+[A-Z][a-z'’]+){0,2})"
)

# The all-lowercase form, which WhatsApp guests type constantly. Only after
# a verb that means "I mean that property" — otherwise "hotel feels nice"
# would read "feels" as a hotel name.
_CUE = re.compile(
    r"\b(?:stay|staying|book|booking|reserve|reserving|shift|shifting|move|moving)"
    r"\s+(?:at|in|to)\s+(?:the\s+)?"
    rf"(?:{_PROPERTY_WORD}\s+)?"
    r"(?P<name>[a-z][a-z'’]+(?:[ \t]+[a-z][a-z'’]+){0,2})",
    re.IGNORECASE,
)

# Words that can follow a property word without being its name. Trimmed from
# the end of a candidate until a real name word is left. The capitalised
# branch needs this most: "Please check the Hotel Availability" is a proper
# noun by shape and nothing at all by meaning.
_STOP_WORDS = frozenset(
    {
        # Sentence glue that appears capitalised.
        "the", "and", "for", "in", "at", "on", "to", "with", "near", "from",
        "by", "of", "is", "are", "was", "we", "i", "it", "this", "that",
        "our", "my", "your", "has", "have", "please", "kindly", "a", "an",
        # Things a guest says about a hotel, not its name. Property words
        # ("hotel", "palace", "resort") are deliberately *absent* here: real
        # names end in them — "Samman Palace" — so they are only stripped from
        # the front, by `_GENERIC_PROPERTY_WORDS` below.
        "room", "rooms", "price", "prices", "rate", "rates", "rent",
        "availability", "booking", "bookings", "amenities", "location",
        "details", "staff", "reception", "policy", "photos", "images",
        "food", "parking", "pool", "wifi", "service", "services",
        "check", "checkin", "checkout", "today", "tomorrow",
        "good", "nice", "best", "cheap", "cheapest", "budget",
    }
)

_GENERIC_PROPERTY_WORDS = frozenset(
    {"hotel", "resort", "inn", "palace", "lodge", "guest", "house", "homestay", "the"}
)


def _tokens(value: Any) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(value or "").lower()) if t]


def _name_tokens(value: Any) -> list[str]:
    """Trim stop words off the end of a candidate, then return what is left."""
    words = str(value or "").split()
    while words and words[-1].lower() in _STOP_WORDS:
        words.pop()
    return _tokens(" ".join(words))


def _own_name_tokens(hotel_name: Any) -> list[str]:
    """The words that make our hotel *ours*: "Sample Hotel" -> ["sahu"].

    Empty when the config has no name ("our hotel", ""). An unnamed hotel
    cannot tell one property from another, and guessing would put a wrong
    hotel's name in every reply — so an empty result disables the check.
    """
    generic = _GENERIC_PROPERTY_WORDS | _STOP_WORDS
    return [t for t in _tokens(hotel_name) if t not in generic]


def _is_ours(candidate_tokens: list[str], own_tokens: list[str]) -> bool:
    if not candidate_tokens or not own_tokens:
        return not candidate_tokens
    return set(candidate_tokens) & set(own_tokens) == set(own_tokens)


def other_property_named(message: str, hotel_name: Any) -> str | None:
    """The name the guest used for a hotel that is not ours, or None.

    Returns the guest's own spelling so the reply can quote it back — a
    guest who typed "Samman" should see "Samman".
    """
    own_tokens = _own_name_tokens(hotel_name)
    if not own_tokens:
        return None

    text = message or ""
    for pattern in (_CAPITALISED, _CUE):
        for match in pattern.finditer(text):
            raw = match.group("name").strip(" .,!?")
            candidate = _name_tokens(raw)
            if not candidate or _is_ours(candidate, own_tokens):
                continue
            # The quoted form is the trimmed one, so "Hotel Samman rooms"
            # comes back as "Samman" rather than "Samman rooms".
            return " ".join(
                w for w in raw.split()[: len(candidate)]
            )
    return None
