"""A typed guest list that puts its children under a heading.

2026-10-29, hotel Sample Property. The guest pasted a complete booking brief:

    List of Guests:-
    ...
    6) MAYABEN SHRIVAS
    Age:- 60 Year, Female

    Children:-

    7) RUDRA SHARMA
    Age:- 5 Year, Male
    8) TRISHIKA SHRIVAS
    Age:- 2.5 Year, Female

The model read `children = 1` and `child_ages = [5]`. It dropped the
two-and-a-half-year-old twice over: `child_ages` is list[int] so 2.5 cannot be
written in it, and the child it lost also left the count. The bot then quoted
"6 guests + 1 children" for a party of eight.

`validator` already knows how to assume an age it was not given. What it cannot
do is invent a child who was never counted, and it should not have to: the
guest wrote that child down, under a heading.

So this reads the heading. Deliberately narrow — it only looks inside a
"Children" section, so the adults' own ages above it can never be read as
child ages, and text with no such heading yields nothing at all.
"""

from __future__ import annotations

import re

# A line that starts a children section: "Children:-", "Child :", "Kids -",
# "Bachche:-". Anchored to the start of a line so "2 children, 5 and 8" in
# ordinary prose is not treated as a heading.
_CHILD_HEADER = re.compile(
    r"(?:^|\n)[ \t]*(?:child(?:ren)?|kids?|bachch\w*)[ \t]*[:\-–]+",
    re.IGNORECASE,
)

# The next section heading, if any: a short lettered label ending in ":-" and
# then nothing else on the line. The end-of-line requirement is what keeps a
# guest's own "Age:- 5 Year, Male" field from being read as a new section —
# which it was, on the first version of this, truncating the section to one
# name and losing every age in it.
_NEXT_HEADER = re.compile(
    r"\n[ \t]*[A-Za-z][A-Za-z /]{1,30}[ \t]*:-[ \t]*(?=\r?\n|$)"
)

# "Age:- 5 Year", "aged 8", "2.5 yrs", "3 saal".
_AGE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:saal|years?|yrs?)\b", re.IGNORECASE)

# Above this, the number is an adult named in the same section by mistake, or
# part of a stray phrase. The contract treats anyone over this as an adult.
_OLDEST_CHILD = 17


def child_section_text(text: str) -> str:
    """The text under the first children heading, or "" when there is none."""
    if not isinstance(text, str):
        return ""
    header = _CHILD_HEADER.search(text)
    if not header:
        return ""
    rest = text[header.end():]
    following = _NEXT_HEADER.search(rest)
    return rest[:following.start()] if following else rest


def child_section_ages(text: str) -> list[int]:
    """Ages named in a children section, floored, oldest-child filter applied.

    "2.5 Year" -> 2 rather than being dropped: a child is younger than the
    number says, and the younger reading is the one that keeps them in the
    hotel's free band rather than charging for them.
    """
    section = child_section_text(text)
    if not section:
        return []
    ages = [int(float(a)) for a in _AGE.findall(section)]
    return [age for age in ages if age <= _OLDEST_CHILD]
