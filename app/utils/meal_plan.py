"""Which meal plan a rate plan is, and whether it may seat a guest for free.

Two questions, and the second is the reason this module exists.

**What plan is this?** The catalogue answers it properly: `rate_types` holds
`RT_EP` / `RT_CP` / `RT_MAP` / `RT_AP`, `rate_plans` carries `rate_type_id`, and
`room_repository` already copies that onto every `RatePlan`
(`room_repository.py:204`). Until now **nothing read it** — five separate files
instead substring-matched the plan's *name* (`"cp" in rp.name.lower()`:
`booking_engine.py:106`, `amenity_service.py:200`, `quote_generator.py:164`,
`menu_handler.py:231`, `room_query_handler.py:493`). That works until a hotel
names a plan "Continental Breakfast Included", or "Room Only — CP Rate".

`classify_rate_plan` reads `rate_type_id` first and only falls back to the name
for plans synthesized without one — the hardcoded fallback plan at
`room_repository.py:154-161` builds `RatePlan(rate_plan_id=f"{room_id}_EP",
name="Room Only (EP)")` and sets no `rate_type_id`.

**Who may sit above the base occupancy for free?** Product, 2026-10-06:

    "On EP Plan we can allow extra person to stay in room without any extra bed
     so for example in Deluxe Room which has base_occupancy = 2 and max_adults =
     3, EP can have NO EXTRA BED (no extra charges) but that SHOULD NOT be
     allowed on MAP and CP"

So the *permission* is a property of the plan, while the *price* is a property
of the room type — "Extra person rate is for each room type and not for each
plan". The two are deliberately kept apart here: this module only says whether a
charge is owed, never how much.
"""

from __future__ import annotations

import re

# The four plans the catalogue knows. AP (all meals) is included because the
# same reasoning applies to it as to CP and MAP — a guest above base occupancy
# is eating more meals than the room was sold for.
EP = "EP"
CP = "CP"
MAP = "MAP"
AP = "AP"
UNKNOWN = "UNKNOWN"

#: Plans on which an adult above the base occupancy is charged the extra person
#: rate whether or not they were given a mattress. EP is deliberately absent:
#: on EP the extra guest may stay with no bed and no charge
#: (`stayinn-extra-person-plan-rule`).
PLANS_THAT_CHARGE_ABOVE_BASE = (CP, MAP, AP)

# `RT_EP` -> `EP`. Also tolerates a bare `EP` and lower case.
_RATE_TYPE_RE = re.compile(r"^(?:RT[_-])?([A-Z]+)$", re.IGNORECASE)

# Name fallbacks, most specific first. Every pattern is anchored on a word
# boundary: the old `"cp" in name` read "Room Only — CP Rate" correctly but
# would equally have matched any word containing those two letters.
_NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (AP, re.compile(r"\ball[ -]?(?:meals|inclusive)\b|\bfull board\b|\bamerican plan\b|\bap\b")),
    (
        MAP,
        re.compile(
            r"\bmap\b|\bhalf board\b|\bmodified american\b"
            r"|\bbreakfast\b[\s\S]*\b(?:dinner|lunch)\b"
            r"|\b(?:dinner|lunch)\b[\s\S]*\bbreakfast\b"
        ),
    ),
    (CP, re.compile(r"\bcp\b|\bbreakfast\b|\bcontinental\b|\bbed and breakfast\b|\bb&b\b")),
    (EP, re.compile(r"\bep\b|\broom only\b|\beuropean plan\b|\bno meals?\b")),
)


def classify_rate_plan(
    rate_type_id: str | None, plan_name: str | None = None
) -> str:
    """`EP` / `CP` / `MAP` / `AP`, or `UNKNOWN` when the plan does not say.

    `UNKNOWN` is not a failure to be papered over — every caller must treat it
    as "this plan gives me no permission", i.e. behave exactly as the bot did
    before the rule existed. A plan the catalogue cannot name must never start
    charging guests.
    """
    raw = (rate_type_id or "").strip()
    if raw:
        match = _RATE_TYPE_RE.match(raw)
        if match:
            token = match.group(1).upper()
            if token in (EP, CP, MAP, AP):
                return token

    name = (plan_name or "").lower()
    if not name:
        return UNKNOWN
    for plan_class, pattern in _NAME_PATTERNS:
        if pattern.search(name):
            return plan_class
    return UNKNOWN


def plan_charges_above_base(plan_class: str) -> bool:
    """Does this plan owe for an adult it does not seat?

    False for EP — deliberately — and false for `UNKNOWN`, which is the load-
    bearing half: a plan the catalogue cannot name must never start billing
    guests, so an unclassifiable plan keeps the bot's old bed-only behaviour.
    """
    return plan_class in PLANS_THAT_CHARGE_ABOVE_BASE


def charges_an_extra_person(plan_class: str, adults_above_base: int) -> bool:
    """Is an extra person owed on this room, mattress or no mattress?

    True only where the guest has genuinely put more adults in the room than the
    hotel sells it for *and* the plan is one that does not include them (CP, MAP
    or AP). On EP — and on any plan we cannot classify — this is False, which
    leaves the caller's existing bed-only rule in charge of the money.

    `adults_above_base` is a count of people, not a price: what they are charged
    is the room type's own extra-adult rate, decided by the caller.
    """
    if adults_above_base <= 0:
        return False
    return plan_charges_above_base(plan_class)
