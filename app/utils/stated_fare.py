"""The total a guest wrote in their own message — captured, never applied.

2026-10-29, hotel Sample Property. product forwarded a guest's message that was a
complete booking brief, pasted as text:

    Fare:- Rs. 2000/- x 3 Rooms x 3 Nights = Rs. 18000/-

The bot quoted the catalogue, and the Rs.18,000 the guest had just written was
never stored anywhere. The photographed-slip path already records an amount
(see `_record_booking_note` in `app/services/chat_runner.py`); the typed path
had no equivalent, because `understand`'s extractor reads dates, guests and
rooms, and `budget` is only filled when the guest uses the word "budget".

Same boundary as the slip: the number is RECORDED as a note and is not applied
to `total_price`. A sentence the guest typed is not authority to charge below
the catalogue — reception decides that.

Deliberately conservative. Returning None is always safe; returning a wrong
total puts a number into the session that staff may act on, so the reading is
only taken when the shape is unambiguous:

* two or more amounts, with the last one after an `=` or a "total"/"fare"
  word — an equation, where the final amount is the sum;
* exactly one amount, and it carries a total word near it.

"Rs 2000 per night" is a rate, not a total, and is left alone.
"""

from __future__ import annotations

import re

# `Rs.`, `Rs`, `INR`, `₹` followed by an amount. The group allows thousands
# separators and an optional paise part; "2000/-" is the common Indian
# spelling and the `/-` falls outside the group.
_AMOUNT = re.compile(r"(?:rs\.?|inr|₹)\s*([\d][\d,]*(?:\.\d{1,2})?)", re.IGNORECASE)

# Words that mark the amount as a whole-stay figure rather than a rate.
_TOTAL_WORD = re.compile(
    r"\b(?:fare|total|grand\s+total|amount|kul)\b", re.IGNORECASE
)

# How far before an amount a total word still describes it.
_TOTAL_WORD_WINDOW = 40

# "Rs 2000 per night" sits right next to the word "fare" and is still a rate.
# Checked in the few characters after the amount, where it always appears.
_RATE_AFTER = re.compile(r"^\s*(?:/-)?\s*(?:per\s+night|/\s*night|nightly|per\s+day)", re.IGNORECASE)


def _to_float(raw: str) -> float | None:
    try:
        value = float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _is_rate(text: str, match: re.Match) -> bool:
    """True when this amount is followed by "per night" and is therefore a
    rate, not a stay total."""
    return bool(_RATE_AFTER.match(text[match.end():match.end() + 24]))


def extract_stated_fare(text: str) -> float | None:
    """The whole-stay total the guest wrote, or None.

    Pure and total: any input type returns a value or None, never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    matches = list(_AMOUNT.finditer(text))
    if not matches:
        return None

    if len(matches) > 1:
        last = matches[-1]
        between = text[matches[-2].end():last.start()]
        # An equation names its own total: "2000 x 3 x 3 = 18000". Without the
        # `=` or a total word, several amounts are rates for different rooms
        # and there is no defensible way to pick one.
        if _is_rate(text, last):
            return None
        if "=" in between or _TOTAL_WORD.search(text):
            return _to_float(last.group(1))
        return None

    only = matches[0]
    if _is_rate(text, only):
        return None
    window = text[max(0, only.start() - _TOTAL_WORD_WINDOW):only.start()]
    if _TOTAL_WORD.search(window):
        return _to_float(only.group(1))
    return None
