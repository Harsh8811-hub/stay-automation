"""
The arrival or departure time a guest states, if they state one.

Why this exists
---------------
The hotel sells paid early check-in and late check-out, priced by *how many
hours* outside the standard times the guest wants. So the bot needs a clock
time, and it needs it only when the guest volunteers one — we never ask, because
turning a free-text chat into a form is worse than not quoting.

Where the tokens come from
--------------------------
Not from a new regex. :mod:`app.utils.date_masking` is already "the one place
that decides" what a date or a time is, and its ``_DATE_PATTERNS`` already
covers ``5pm``, ``5.30pm``, ``5:30 pm``, ``17:30`` and ``5 o'clock``. We ask
that module for :func:`~app.utils.date_masking.date_spans` and then keep the
spans that are genuinely clock-shaped. A token the repo does not already read as
a time is not a time here either.

What it deliberately does NOT read
----------------------------------
* A bare number is never a clock — ``"6 rooms"`` and ``"2 adults"`` must not
  price an early check-in.
* ``"5 baje"`` / ``"5 o'clock"`` with no am/pm is **skipped**, not guessed.
  Assuming a meridiem there would be right half the time and would put a wrong
  number on a guest's bill the other half. Skipping costs a quote; guessing
  costs money.
* ``"5:30"`` with no am/pm is skipped for the same reason. ``"17:30"`` is not
  ambiguous and is read.
"""

from __future__ import annotations

import logging
import re

from app.utils.date_masking import date_spans

logger = logging.getLogger(__name__)

# ── Reading a clock ──────────────────────────────────────────────────────────

_CLOCK_SHAPE = re.compile(
    r"^\s*(?P<hour>\d{1,2})"
    r"(?:\s*[.:]\s*(?P<minute>\d{2}))?"
    r"\s*(?P<meridiem>am|pm)?\s*$",
    re.IGNORECASE,
)

#: A span only counts as a clock if it carries a meridiem or a colon.
#:
#: No leading ``\b``: guests write ``7am`` as often as ``7 am``, and there is no
#: word boundary between the ``7`` and the ``a`` — ``\bam\b`` silently matches
#: neither. Only ever applied to a span :func:`date_spans` already called a
#: date or a time, so a stray "am" from prose cannot reach it.
_HAS_MERIDIEM = re.compile(r"(?:am|pm)\b", re.IGNORECASE)
_HAS_COLON = re.compile(r"\d{1,2}\s*[.:]\s*\d{2}")


def parse_clock(text: str | None) -> int | None:
    """Minutes since midnight for a written clock time, or None.

    Handles the shapes the Control Centre and ``hotel_policies`` use
    (``"01:00 PM"``, ``"11:00 AM"``, ``"04:00 am"``) and the shapes a guest uses
    (``"7pm"``, ``"7 pm"``, ``"7.30pm"``). A bare ``"6"`` is not a time and
    returns None; a bare ``"18:00"`` is unambiguously 24-hour and is read.
    """
    if not text:
        return None
    match = _CLOCK_SHAPE.match(str(text).strip())
    if not match:
        return None

    try:
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
    except (TypeError, ValueError):
        return None

    if minute > 59:
        return None

    meridiem = (match.group("meridiem") or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12
        if meridiem == "pm":
            hour += 12
    else:
        # No am/pm: only a colon form is unambiguous enough to read.
        if not match.group("minute"):
            return None
        if not 0 <= hour <= 23:
            return None

    return hour * 60 + minute


def _format_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ── Deciding which spans are clocks, and which way they point ────────────────

#: "arrive by 5pm" — the guest is telling us when they turn up.
ARRIVAL_CUES = (
    "arriv", "reach", "reach ", "will be there", "get there", "turn up",
    "check in", "check-in", "checkin", "aana", "aaunga", "pahunch", "pohch",
)

#: "checkout at 11am" — the guest is telling us when they leave.
#:
#: Deliberately excludes "train" and "flight": a guest may just as easily be
#: arriving on one as leaving on one, and a wrong direction here is a wrong
#: charge. When both a departure and an arrival cue are present the closer one
#: wins, so "check out" still claims its own clock.
DEPARTURE_CUES = (
    "check out", "check-out", "checkout", "depart", "leaving", "leave",
    "vacating", "vacate", "nikal", "nikalna", "chale jayenge",
)

#: How far a cue may sit from its clock and still claim it.
_CUE_WINDOW = 40


def _nearest_cue_gap(text: str, start: int, end: int, cues: tuple[str, ...]) -> int | None:
    """Character gap between the clock and the closest of ``cues``, or None.

    Distance is measured edge-to-edge, so "checkout 11am" scores 1 (the cue ends
    immediately before the clock) rather than counting the whole word.
    """
    best: int | None = None
    for cue in cues:
        index = text.find(cue)
        while index != -1:
            cue_end = index + len(cue)
            if cue_end <= start:
                gap = start - cue_end
            elif index >= end:
                gap = index - end
            else:
                gap = 0  # Overlapping — the cue is the clock's own text.
            if gap <= _CUE_WINDOW and (best is None or gap < best):
                best = gap
            index = text.find(cue, index + 1)
    return best


def _direction_for(text: str, start: int, end: int) -> str:
    """``"departure"`` or ``"arrival"`` for the clock at ``text[start:end]``.

    Whichever cue sits *closest* to the clock wins, so a neighbouring clause
    cannot steal it: in "checkout 11am and arrive 5pm" the 11am is 1 character
    from "checkout" and the 5pm is 0 from "arrive". Departure wins an exact tie —
    "check out" is the more specific phrase and contains "check".
    """
    lowered = text.lower()
    arrival_gap = _nearest_cue_gap(lowered, start, end, ARRIVAL_CUES)
    departure_gap = _nearest_cue_gap(lowered, start, end, DEPARTURE_CUES)

    if departure_gap is not None and (arrival_gap is None or departure_gap <= arrival_gap):
        return "departure"
    if arrival_gap is not None:
        return "arrival"
    # An unqualified clock is far more often "I'll arrive at 7" than a departure.
    return "arrival"


#: Words that carry no meaning once the clock is removed — what is left of
#: "at around 10 in the morning" after "10 am" is taken out.
_CLOCK_FILLER = frozenset(
    {
        "at", "by", "on", "in", "the", "a", "an", "around", "about", "approx",
        "approximately", "near", "sharp", "ist", "am", "pm", "morning", "evening",
        "afternoon", "night", "noon", "midnight", "midday", "o", "clock", "baje",
        "bajey", "only", "approx.", "hrs", "hour", "hours", "o'clock",
    }
)


def is_only_a_clock(text: str | None) -> bool:
    """True when the message is a clock and essentially nothing else.

    The bot asks the guest for their arrival time; the answer is "10AM". That
    sentence has no verb, no verb cue, and — crucially — no early/late keyword,
    because the bot's own question supplied the context. This is how a caller
    can tell it apart from prose that merely mentions a time.

        "10AM", "10 am", "at 10 in the morning"   -> True
        "we will reach at 10 am"                  -> False
        "the train is at 6pm"                     -> False

    Pure and total. A message with no clock at all is False, so this can never
    be the only guard on a branch.
    """
    if not text:
        return False
    found = False
    remaining = text
    # Rightmost span first: every removal then happens to the right of the
    # offsets still to be used, so none of them shift.
    for start, end in sorted(date_spans(text), reverse=True):
        token = text[start:end]
        if not (_HAS_MERIDIEM.search(token) or _HAS_COLON.search(token)):
            continue
        if parse_clock(token) is None:
            continue
        found = True
        remaining = remaining[:start] + " " + remaining[end:]

    if not found:
        return False
    words = re.findall(r"[a-z']+", remaining.lower())
    return all(w in _CLOCK_FILLER for w in words)


def extract_requested_times(text: str | None) -> dict[str, str | None]:
    """``{"arrival": "HH:MM"|None, "departure": "HH:MM"|None}`` from one message.

    Pure and total — never raises, and returns both keys always. When a guest
    states two clocks for the same direction the first one wins, so the reading
    is stable rather than dependent on dict ordering.
    """
    result: dict[str, str | None] = {"arrival": None, "departure": None}
    if not text:
        return result

    for start, end in date_spans(text):
        token = text[start:end]
        if not (_HAS_MERIDIEM.search(token) or _HAS_COLON.search(token)):
            continue  # A calendar date, or an am/pm-less "5 baje" — not readable.

        minutes = parse_clock(token)
        if minutes is None:
            logger.debug("guest_time: clock-shaped span %r did not parse", token)
            continue

        direction = _direction_for(text, start, end)
        if result[direction] is None:
            result[direction] = _format_clock(minutes)

    return result
