"""
Numbers that belong to a date, and the one place that decides so.

Why this exists
---------------
The bot reads counts out of the guest's sentence — "6 people", "2 rooms". A
date carries numbers too, and a date's numbers sit in exactly the position a
count would. The live case, 2026-09-24:

    Check in: 12/11/26 check out 15/11/26 person 5+1

The headcount regex at ``understand.py:1734`` looks for digits immediately
before the word "person". The digits immediately before "person" are the last
two of the checkout date. It read ``26`` as the party size, forced children to
0, and told a guest who asked for six people that the bot had 26 of them —
after which no room could seat the party and the guest got "we don't have rooms
available".

Three separate places in this repo had already met this bug and each solved it
alone, or not at all:

  * ``understand.py:1734``       matched ``26 person`` — the live defect
  * ``intent_overrides.py:238``  "Guard against check-in/out date digits"
  * ``room_requirement_parse._is_part_of_a_date`` — whose own docstring
                                 describes ``cheack in 24/10/26 at 5pm``

One notice is a bug, three is a pattern. This module is the single
implementation; the others call into it.

How the defence is built
------------------------
Two independent layers, because they fail differently:

1. **The producer** (``understand.py``) blanks the dates out of the message
   with :func:`mask_dates` before running any count regex over it. The wrong
   value is never formed, so the guest is asked for a headcount rather than
   being told a wrong one.

2. **The contract** (``extraction_contract`` Rule 1) refuses a number that
   could only have come from a date, using :func:`number_only_in_dates`. This
   catches the paths layer 1 cannot reach — the LLM extractor reads the raw
   message, not the masked one, and the funnel in ``understand.py`` has
   seventeen producers, not one.

Either layer alone would have caught the live case. Neither alone closes the
class: layer 1 misses the LLM's own guesses, layer 2 only runs once a value has
already been formed and would leave the field null where a cleaner read was
available.
"""

from __future__ import annotations

import re

# English month names, matched on a three-letter prefix so "Sept"/"September"
# and "Nov"/"November" are both covered without listing every spelling.
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"

# A day-of-month, with or without an ordinal suffix: 5, 15th, 2nd.
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"

_COUNT_WORDS = r"(?:adult|child|kid|pax|person|people|guest|room|bed|bache|log)"

#: Every shape a date or a time takes in guest traffic.
#:
#: Deliberately does NOT include a bare ``dd/mm``. "2-3 rooms" and "5-7 people"
#: are count ranges, and a pattern that read ``dd/mm`` would blank them and
#: lose a real booking request. A day-without-year is rare in this traffic; a
#: room range is not. The trade is stated here so the next person does not
#: "fix" it by adding one back.
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 12/11/26 · 12-11-26 · 12.11.26 · 12/11/2026 · 2026-11-12
    re.compile(r"\b\d{1,4}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{2,4}\b"),
    # 15 Nov 26 · 15th November · 15 of Nov 2026 · 15-Nov
    re.compile(
        rf"\b{_DAY}\s*(?:of\s+)?[/.\-]?\s*{_MONTH}\.?(?:\s*,?\s*\d{{2,4}}\b(?!\s*{_COUNT_WORDS}))?",
        re.IGNORECASE,
    ),
    # Nov 15 · November 15 2026
    re.compile(
        rf"\b{_MONTH}\.?\s*{_DAY}(?!\s*{_COUNT_WORDS})(?:\s*,?\s*\d{{2,4}}\b(?!\s*{_COUNT_WORDS}))?",
        re.IGNORECASE,
    ),
    # 5 to 7 sep · 12-15 nov · 5th till 7th Nov
    re.compile(
        rf"\b{_DAY}\s*(?:to|till|until|-|–)\s*{_DAY}\s*[/.\-]?\s*{_MONTH}\.?",
        re.IGNORECASE,
    ),
    # 5pm · 5 pm · 5.30pm · 5:30 pm · 17:30 · 5 o'clock
    re.compile(r"\b\d{1,2}\s*(?::\d{2}|\.\d{2})?\s*(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}\s*o'?\s*clock\b", re.IGNORECASE),
)


def date_spans(text: str | None) -> list[tuple[int, int]]:
    """Every ``(start, end)`` span in ``text`` that holds a date or a time."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    for pattern in _DATE_PATTERNS:
        spans.extend(m.span() for m in pattern.finditer(text))
    if len(spans) < 2:
        return spans
    # Merged, so an overlapping pair ("15 Nov" inside a longer range) reads as
    # one span rather than two that each cover half of it.
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def mask_dates(text: str | None) -> str:
    """
    ``text`` with every date and time blanked to spaces.

    The result is the *same length* as the input, so a caller that has already
    located something keeps its offsets. Whitespace inside a span is preserved
    rather than overwritten, so line structure survives.
    """
    if not text:
        return text or ""
    chars = list(text)
    for start, end in date_spans(text):
        for index in range(start, min(end, len(chars))):
            if not chars[index].isspace():
                chars[index] = " "
    return "".join(chars)


def is_position_in_a_date(text: str | None, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` falls inside a date or time expression."""
    return any(
        span_start <= start and end <= span_end
        for span_start, span_end in date_spans(text)
    )


def number_only_in_dates(text: str | None, number: object) -> bool:
    """
    True when every occurrence of ``number`` in ``text`` is inside a date.

    This is the question the contract actually needs. Not "does a date contain
    this number" — a guest may write ``12/11/26 for 26 people`` and mean both —
    but "is a date the *only* place this number came from".

    So the number is looked for twice: once in the original message, and once
    in the message with its dates blanked. Present in the first and absent from
    the second means the date is the only source, and the value is not a count
    the guest gave. Present in both means the guest said it in their own right
    — "26 people on 15/11/26" — and it is left alone.

    Whole-token matching, so ``6`` is not "found" inside ``26``.
    """
    if not text or number is None:
        return False
    try:
        token = str(int(number))
    except (TypeError, ValueError):
        return False
    pattern = re.compile(rf"(?<!\d){re.escape(token)}(?!\d)")
    if not pattern.search(text):
        return False
    return not pattern.search(mask_dates(text))
