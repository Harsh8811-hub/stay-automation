"""
The extraction contract — the one place that decides whether a value the LLM
or the guest produced is fit to enter conversation state.

Why this exists
---------------
``understand.py`` merges roughly seventeen hand-rolled extraction branches at
a single line::

    return cast(ConversationState, {**state, **updates})

Nothing validated ``updates`` on the way out. So free text and unbounded
numbers were promoted into *typed* state fields with no check, and then
rendered back to the guest verbatim. Three production defects of 2026-09-17
were all that one mechanism:

  ``adults = 2026``           ``int | None`` accepts any int, so the year in
                              "21.10.2026" became a guest count while the
                              guest's actual "Person : 4" was ignored.
  ``room_type_preference =    ``str | None`` accepts any string, so the whole
  "one room with attached     guest sentence became a room type and was
   bathroom"``                 echoed back as "we don't have **one room with
                              attached bathroom** rooms available".
  ``feature_preferences =     An unmatchable feature emptied the catalogue —
  ["attached bathroom"]``     defect #10, already patched once at the
                              *consumption* site in ``allocation_engine.py``,
                              which is exactly why it came back at a new
                              producer site.

Every one of those was fixed individually at its own call site. That is the
whack-a-mole this module ends. There is one funnel, so there is one check.

The 2026-09-24 defect is the same lesson one layer down. ``adults = 26`` came
from the year in ``15/11/26``, which the year check below could not see — it
only recognises four-digit years, and ``26`` is a perfectly plausible
headcount. The fix is Rule 1's third check, ``number_only_in_dates``; it is
here rather than at ``understand.py:1734`` for the reason this module exists,
and it is written against the guest's message rather than the value so that it
holds for the LLM's guesses as much as for the regexes'.

Relationship to ``session_contract``
------------------------------------
``app/models/session_contract.py`` already does this for the *persistence*
direction (state → MongoDB), and derives its field list from
``BookingSession.model_fields`` so the contract cannot drift from the model.
This is the same contract for the other direction — LLM text → state —
deriving from ``ExtractedBookingData.model_fields`` for the same reason.

How to add a field
------------------
Adding a field to ``ExtractedBookingData`` fails
``tests/test_extraction_contract.py`` until it is classified here: either give
it a rule, or list it in ``PASS`` with a written reason. If you cannot write
that reason, it needs a rule.

Policy on a rejected value
--------------------------
Rejected means rejected: the value is set to ``None``, never repaired. A
recovery heuristic would be one more unvalidated guess about the same input,
which is how the original bug got in. ``validator.py`` already treats a null
field as missing, so the bot re-asks and the guest corrects in one turn.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from app.models.schemas import ExtractedBookingData
from app.services.allocation_engine import is_hard_inventory_feature
from app.utils.date_masking import number_only_in_dates
from app.utils.helpers import is_valid_date, normalize_date

logger = logging.getLogger(__name__)

# ── What this contract governs ─────────────────────────────────────────────
#
# Two sets, because the funnel at the end of ``understand.py`` carries more
# than the model's own fields.
#
# ``DECLARED``: whatever ``ExtractedBookingData`` declares, derived from the
# model so the contract cannot drift from it — the same trick
# ``session_contract.py`` uses on ``BookingSession``.
DECLARED: frozenset[str] = frozenset(ExtractedBookingData.model_fields)

# ``DERIVED``: state fields that ``understand.py`` computes itself and writes
# into the same ``updates`` dict, so they pass through the same funnel. They
# are not extractable, but three of them are *copies of an extracted value* —
# ``understand.py:833`` and ``:1533`` do ``updates["effective_adults"] =
# updates["adults"]``. Without a rule, a rejected ``adults`` re-enters as
# ``effective_adults`` and the rejection is theatre. Bounded for that reason.
DERIVED: frozenset[str] = frozenset({"effective_adults", "effective_children"})

# Everything the funnel can carry, and therefore everything that must be
# classified. The coverage test asserts on this union, not on DECLARED alone.
GOVERNED: frozenset[str] = DECLARED | DERIVED

# ── Bounds ─────────────────────────────────────────────────────────────────
#
# A self-serve booking is a handful of people; ``MAX_SELF_SERVE_ROOMS`` in
# allocation_engine is 4. Group requests take the lead-capture path instead.
_COUNT_MIN = 1
_COUNT_MAX = 300

# A four-digit number in this range is a year, never a headcount. This is the
# exact shape of the 2026 defect, stated as a principle rather than as a magic
# threshold on one field.
_YEAR_MIN = 1900
_YEAR_MAX = 2100

# A room type is a short noun phrase — "DELUXE ROOM FOUR BED" is four words.
# Anything longer is a sentence, which is what the guest actually wrote.
_MAX_NOUN_WORDS = 4


# ── Rule 1: integer counts that must be plausible human quantities ─────────
#
# ``effective_*`` are derived from their unsuffixed twins inside understand.py,
# so they need the same bound — otherwise a rejected ``adults`` could re-enter
# through the ``effective_adults`` door.
BOUNDED_INT: dict[str, tuple[int, int]] = {
    "adults": (_COUNT_MIN, _COUNT_MAX),
    "effective_adults": (_COUNT_MIN, _COUNT_MAX),
    "children": (0, _COUNT_MAX),
    "effective_children": (0, _COUNT_MAX),
    "num_rooms": (_COUNT_MIN, _COUNT_MAX),
    "ambiguous_guests": (_COUNT_MIN, 10),
}


def _looks_like_a_year(value: int) -> bool:
    """True for a value in the calendar-year range — a date, not a headcount."""
    return _YEAR_MIN <= value <= _YEAR_MAX


def _reject_reason_for_count(
    number: int, low: int, high: int, guest_text: str
) -> str | None:
    """
    Why this integer cannot be a count, or None if it can be.

    The year check below only knows four-digit years, which is why it could not
    stop the 2026-09-24 defect: the year in ``15/11/26`` is ``26``, a plausible
    headcount that is not recognisable as a year at all. ``number_only_in_dates``
    answers the question that actually matters — is a date the only place this
    number appears in what the guest wrote — and covers two-digit years, times,
    and every other date shape.

    Checked against the guest's own message rather than against the value, so
    it holds for a value from any of the funnel's producers, the LLM included.
    """
    if _looks_like_a_year(number):
        return f"{number} is a calendar year, not a headcount"
    if number_only_in_dates(guest_text, number):
        return f"{number} appears only inside a date in the guest's message"
    if not (low <= number <= high):
        return f"{number} outside plausible range {low}..{high}"
    return None


# ── Rule 2: free text that will be quoted back to the guest as a noun ──────
#
# Only the fields that get interpolated into a guest-facing sentence need this.
# "one room with attached bathroom" is a request *describing* a room, and the
# guest's words must never be echoed as though they named one.
NOUN_PHRASE: frozenset[str] = frozenset({"room_type_preference"})

# Function words that mark a describing sentence rather than a name. Real
# catalogue entries ("STANDARD NON AC ROOM", "DELUXE ROOM FOUR BED") contain
# none of these.
_SENTENCE_MARKERS_RE = re.compile(
    r"\b(with|without|having|attached|and|plus|for|me|us|need|want|should)\b",
    re.IGNORECASE,
)


def _reject_reason_for_noun(value: Any) -> str | None:
    """Why this value cannot be a room type, or None if it is acceptable."""
    if not isinstance(value, str):
        return "not a string"
    text = value.strip()
    if not text:
        return "empty"
    if is_hard_inventory_feature(text):
        # "AC" / "Non-AC". The repo already treats these as inventory
        # attributes, not room names (`_HARD_INVENTORY_FEATURES`,
        # app/services/allocation_engine.py:46): they have their own field
        # (`ac_preference`) and their own hard filter. Landing in
        # `room_type_preference` instead made the bot announce "we don't have
        # **ac** rooms available" while listing AC rooms — a false claim about
        # an amenity, phrased as a claim about a room type.
        return (
            f"{text!r} is an amenity (AC / Non-AC), not a room type — "
            "it belongs in ac_preference"
        )
    if len(text.split()) > _MAX_NOUN_WORDS:
        return f"more than {_MAX_NOUN_WORDS} words — a sentence, not a room name"
    marker = _SENTENCE_MARKERS_RE.search(text)
    if marker:
        return f"describing word {marker.group(0)!r} — a request, not a room name"
    return None


# ── Rule 2b: a stay date must be a date ────────────────────────────────────
#
# Measured live on 2026-10-18, guest 918918122852. A staff summary in the
# conversation history read "standard check-in at 1:00 PM and check-out at
# 11:00 AM"; the model copied those clock times into the guest's stay dates.
# The session then held ``check_in="1:00 PM"``, every night count fell back to
# one, and the guest was quoted "1 night" for a 22-24 Oct stay and told
# "Stay Dates: 1:00 PM se 11:00 AM".
#
# The old PASS entry below said validity was enforced "downstream, by
# validator.validate" — but ``normalize_date`` is written to return unparseable
# input *verbatim* (so ``validate_proposed_date_range`` can echo it back in an
# error message), and the only gate that rejected non-dates was a branch that
# did not run on this path. A value that is not a date is not a date in any
# branch, so the check belongs here, on the one funnel every path crosses.
DATE_FIELDS: frozenset[str] = frozenset({"check_in", "check_out"})


def _reject_reason_for_date(value: Any) -> str | None:
    """
    Why this value cannot be a stay date, or None if it can.

    Accepts the ISO form and any form ``normalize_date`` converts to it, so a
    direct caller passing DD/MM/YYYY is not rejected for a formatting quirk the
    normaliser already handles. A clock time is not converted by either, which
    is exactly the point.
    """
    if not isinstance(value, str):
        return f"not a string ({value!r})"
    raw = value.strip()
    if not raw:
        return "empty"
    if is_valid_date(raw):
        return None
    try:
        if is_valid_date(normalize_date(raw) or ""):
            return None
    except Exception:  # noqa: BLE001 — a normaliser crash must not drop the turn
        pass
    return f"not a date ({value!r})"


# ── Rule 3: fields deliberately left alone, each with a reason ─────────────
#
# Every ``ExtractedBookingData`` field must appear either in a rule above or
# here. The coverage test fails on anything unclassified, so this dict is the
# place a new field lands when it genuinely needs no check.
PASS: dict[str, str] = {
    # ── validated downstream, by a node that already owns the check ──
    "child_ages": "length is reconciled against `children` by validator.validate",
    # ── classification, not data: a wrong value routes oddly but cannot be
    #    rendered as a guest-facing noun or a filter ──
    "ac_preference": "controlled vocabulary set by understand; drives a search filter, never echoed",
    "occupancy_preference": "controlled vocabulary set by understand",
    "occupancy_mode": "controlled vocabulary set by understand",
    "recommendation_preference": "free tag consumed by the ranking code, never echoed",
    "display_preference": "free tag consumed by the ranking code, never echoed",
    "room_count_preference": "free tag consumed by the ranking code, never echoed",
    "comfort_preference": "free tag consumed by the ranking code, never echoed",
    "photo_tag_preference": "matched against photo tags in this turn, never echoed",
    "coupon_code": "validated against the coupon catalogue by quote_generator",
    "extra_services": "validated against the service catalogue by the pricing path",
    "meal_plan_preference": "scored against real rate-plan names by get_preferred_rate_plan",
    "room_requirements": "each item resolved against the catalogue by allocate_rooms_by_requirements",
    # ── free text that is stored, but never interpolated into a sentence as
    #    though the guest had named something. `guest_name` has its own
    #    non-name filter at understand.py:1638-1648. ──
    "guest_name": "filtered by the non-name/inquiry guard at understand.py:1638-1648",
    "guest_email": "contact detail, rendered only into the booking record",
    "guest_phone": "contact detail, rendered only into the booking record",
    "special_requests": "echoed into the booking record, never into a bot sentence",
    "budget": "a number the guest volunteered; only ever compared, never echoed",
    "total_price": "recomputed by PricingEngine on every quote",
    "pricing_quote": "recomputed by PricingEngine on every quote",
    "payment_details": "validated by the payment handler before use",
    "partial_payment_amount": "validated by the payment handler before use",
    "active_persona": "matched against configured personas",
    "composite_template_shown": "a bool flag, no guest-facing surface",
    # ── handled by their own rule, listed here so coverage is explicit ──
    "adults": "BOUNDED_INT",
    "children": "BOUNDED_INT",
    "num_rooms": "BOUNDED_INT",
    "ambiguous_guests": "BOUNDED_INT",
    "room_type_preference": "NOUN_PHRASE",
    "feature_preferences": (
        "each entry is resolved against the live catalogue by "
        "allocation_engine.resolve_honourable_features, which drops the ones "
        "no room can satisfy — the catalogue is not available at ingest"
    ),
}


def names_a_catalogue_room(
    preference: Any, catalogue_names: Iterable[str]
) -> bool:
    """
    True when the guest's preference actually names a room the hotel sells.

    The rules above reject a preference that *cannot* be a room name, using no
    I/O. This answers the weaker question a renderer needs: is it one? It lives
    here so both halves of "may this be shown to the guest as a room type" have
    one home, even though only this half needs the catalogue passed in.

    Used where a preference is interpolated into a guest-facing sentence —
    ``response_generator`` scrolling "we don't have **{pref}** rooms available".
    A preference that names nothing falls back to generic wording rather than
    being quoted back.

    Matching is per whole word, not by substring. Substring matching made this
    useless for the very case it exists for: ``"ac"`` is a substring of
    ``"STANDARD NON AC ROOM"``, so a bare amenity token was "found" in the
    catalogue and echoed back as "we don't have **ac** rooms available".
    Whole-word matching stops ``"ac"``/``"room"``/``"bed"`` from matching
    everything, and the amenity check below stops AC from matching at all.
    """
    if not isinstance(preference, str):
        return False
    text = preference.strip()
    if not text:
        return False
    # AC / Non-AC name an amenity, not a room — and "STANDARD NON AC ROOM"
    # contains the standalone word "ac", so word matching alone would still
    # find it. Same reasoning as _reject_reason_for_noun.
    if is_hard_inventory_feature(text):
        return False
    words = re.findall(r"\w+", text.lower())
    if not words:
        return False
    for name in catalogue_names:
        name_words = set(re.findall(r"\w+", (name or "").lower()))
        if name_words and all(w in name_words for w in words):
            return True
    return False


def validate_updates(
    state: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Return ``updates`` with every rule violation set to ``None``, plus a map of
    field → reason for what was rejected.

    Pure: no database, no catalogue, no network. It decides only what a value
    *cannot* be, which is why it can run on the one funnel line in
    ``understand.py`` without I/O.

    The caller logs the rejections. A rejected value is the only signal that
    the model produced something unusable, and until now that was invisible.
    """
    clean = dict(updates or {})
    rejected: dict[str, str] = {}

    # The guest's own words, for the date check in Rule 1. A caller that passes
    # a state without it gets "" and the check simply does not fire, so this
    # stays safe for the tests and tools that call validate_updates directly.
    guest_text = str((state or {}).get("user_message") or "")

    # A count the guest stated on an earlier turn is not evidence about *this*
    # turn's message, so the "only inside a date" rule must not be applied to
    # one. `finalise` validates the merged state, which means a stored
    # `adults: 8` is re-judged against "8th Nov to 11th Nov" and thrown away --
    # the guest's party is noted, dropped, asked for again, and dropped again
    # (2026-11-06, 3 rooms / 8 adults / 8-11 Nov: "EXTRACTION_REJECTED |
    # field=adults | reason=8 appears only inside a date" on the date turn, then
    # "How many guests will be staying?" and a card reading "2 Adults").
    #
    # Only the date rule is skipped. A carried value still has to be an integer
    # in range; it was checked against its own message when it was first stored.
    prior = state or {}

    for field, (low, high) in BOUNDED_INT.items():
        if field not in clean:
            continue
        value = clean[field]
        if value is None:
            continue
        reason: str | None = None
        try:
            number = int(value)
        except (TypeError, ValueError):
            reason = f"not an integer ({value!r})"
        else:
            try:
                carried = prior.get(field) is not None and int(prior[field]) == number
            except (TypeError, ValueError):
                carried = False
            if carried:
                clean[field] = number
                continue
            reason = _reject_reason_for_count(number, low, high, guest_text)
            if reason is None:
                clean[field] = number
        if reason:
            clean[field] = None
            rejected[field] = reason

    for field in NOUN_PHRASE:
        if field not in clean or clean[field] is None:
            continue
        reason = _reject_reason_for_noun(clean[field])
        if reason:
            clean[field] = None
            rejected[field] = reason

    for field in DATE_FIELDS:
        if field not in clean or clean[field] is None:
            continue
        reason = _reject_reason_for_date(clean[field])
        if reason:
            clean[field] = None
            rejected[field] = reason

    for field, reason in rejected.items():
        logger.warning(
            "EXTRACTION_REJECTED | field=%s | reason=%s | session_id=%s",
            field,
            reason,
            (state or {}).get("session_id", ""),
        )

    return clean, rejected
