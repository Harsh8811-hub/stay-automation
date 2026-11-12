"""Never trade a good stay for an impossible one.

2026-10-28, hotel Sample Property, guest 917376603689. The bot had already told the
guest "check-in 03/11/2026, check-out 05/11/2026", rendered a room card for
"3-5 Nov (2N)", and then — on the very next turn, where the guest only typed
`2` (the number of guests) — the state became:

    check_in  = None
    check_out = 2024-11-05

which `validate_proposed_date_range` correctly rejected. From then on the guest
was trapped: every tap ("Option 1") re-answered "Hmm, ye dates sahi nahi lag
rahi hain", and a card that offered rooms sat one message away from a claim
that the stay was impossible.

The bad pair's origin is upstream of here (nothing in this checkout reproduces
it from the guest's own words — the same three turns return 2026-11-03 /
2026-11-05 consistently). So this does not chase the origin. It holds a line
the origin cannot cross: **an extraction may not replace a valid future pair
with an invalid one when the guest did not say a date in that message.**

Two things keep it narrow:

* Only when the guest's own message names no date. A guest correcting their
  stay must still be able to say a past or reversed pair and be answered.
* Only when the pair already in state is itself valid and in the future. With
  no good pair to fall back to, the bad values pass through and the guest is
  asked properly — which is the current, correct behaviour.
"""

from __future__ import annotations

import logging
import re

from app.utils.helpers import is_future_date, validate_proposed_date_range

logger = logging.getLogger(__name__)

# A message that talks about dates. Deliberately broad: the point is to notice
# that the guest is discussing a stay window, not to parse it. "2" does not
# match; "3rd Nov", "05/11", "next friday", "kal", "3-5 Nov" all do.
_NAMES_A_DATE = re.compile(
    r"(\d{1,2}\s*(?:st|nd|rd|th)?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"|\d{1,2}\s*[/-]\s*\d{1,2}"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:today|tomorrow|tonight|aaj|kal|parso|next\s+\w+day)\b)",
    re.IGNORECASE,
)


def names_a_date(text: str | None) -> bool:
    return bool(text and _NAMES_A_DATE.search(text))


def _pair_ok(ci: str | None, co: str | None) -> bool:
    """Is this a stay the hotel could actually take?

    `validate_proposed_date_range` alone is not enough: it only checks the
    check-out's *format*, so `(None, 2024-11-05)` — the pair that trapped guest
    917376603689 — comes back from it as valid. A check-out in the past is not
    a stay, whatever the check-in says.
    """
    if co is not None and not is_future_date(co):
        return False
    return validate_proposed_date_range(ci, co)[0]


def drop_impossible_dates(
    updates: dict,
    state: dict,
    user_message: str | None,
) -> dict:
    """Remove a newly-extracted date pair that is invalid when the guest did
    not ask for a date change and the state already holds a good pair.

    Mutates and returns `updates`. Pure otherwise: no I/O, never raises.
    """
    if not isinstance(updates, dict):
        return updates

    ci = updates.get("check_in", state.get("check_in"))
    co = updates.get("check_out", state.get("check_out"))
    if ci is None and co is None:
        return updates

    ok = _pair_ok(ci, co)
    if ok:
        return updates

    # `validate_proposed_date_range` says "valid" for an empty range, which is
    # right for its own job and useless here: a state with no dates at all is
    # not something to fall back to. Require an actual pair.
    previous_ci, previous_co = state.get("check_in"), state.get("check_out")
    previous_ok = bool(previous_ci or previous_co) and _pair_ok(previous_ci, previous_co)
    if not previous_ok:
        # Nothing good to keep. Let the validator ask for the dates properly.
        return updates

    if names_a_date(user_message):
        # The guest is talking about a stay window; a bad reading is theirs to
        # correct, and swallowing it would ignore an actual request.
        return updates

    if "check_in" in updates or "check_out" in updates:
        logger.warning(
            "IMPOSSIBLE_DATES_DROPPED | incoming=(%s, %s) keeping=(%s, %s) message=%r",
            ci, co, state.get("check_in"), state.get("check_out"),
            (user_message or "")[:80],
        )
    updates.pop("check_in", None)
    updates.pop("check_out", None)
    return updates
