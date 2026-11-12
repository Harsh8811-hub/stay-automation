"""
Multi-category room requirements, read out of a raw guest message.

"10 Double, 3 Triple" is a room request. "Cheack in 24/10/26 at 5pm" is not —
but on 2026-09-21 the second one was read as **26 rooms of whichever room sorts
first**, and the guest was quoted Rs 58,000/night for a party of twelve. The two
rooms that note was read into were literally:

    {"room_type": "at 5pm  \\ncheack out 27",       "count": 26}
    {"room_type": "kashi classic room in 4th floor", "count": 3}

Two things made that possible, and both are closed here.

  * The room-noun test was a bare substring, `any(k in rtype for k in
    ROOM_KEYWORDS)`. `"ac"` is one of those keywords and `"cheack"` contains
    `"ac"` — so the tail of a date line passed as a room name. The noun now has
    to appear as a **word**: `\\bac\\b` does not match inside `cheack`.
  * The count scan took the first digit run anywhere in the fragment. In
    `24/10/26` the `26` is a year, and the greedy tail after it became the
    "room type". A count now has to sit next to a room noun, and a number with
    a date delimiter on either side is a date, not a count.

A fragment this module cannot read yields nothing, and nothing means "leave the
caller's own extraction alone" — so a message we fail to parse is not turned
into a room request by omission. The model's `room_requirements` still stands
when this returns [].

Pure: a string in, plain dicts out. No model call, no I/O, no state, so every
guard below is tested directly.
"""

from __future__ import annotations

import re

from app.utils.date_masking import is_position_in_a_date

#: A room noun has to appear as a *word* in the fragment before the fragment
#: counts as a room request — see the module docstring for why substrings were
#: not enough.
ROOM_KEYWORDS = (
    "room", "rooms", "deluxe", "standard", "premier", "family", "triple",
    "tripple", "double", "single", "suite", "bed", "ac", "non ac", "non-ac",
    "kamra", "kamre", "four bed",
)

WORD_NUMS = {
    "one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}

_ROOM_NOUN_RE = re.compile(
    r'\b(?:' + "|".join(re.escape(k) for k in ROOM_KEYWORDS) + r')\b'
)

#: Words for the mattress in a room rather than for a room.
_BEDDING_WORDS = ("bed", "beds", "gadda", "gaddi", "mattress", "cot")

#: A room type made of nothing but bedding words and counts — "bed", "2 bed" —
#: describes a bed, not a room, so it must not become a `room_type` of its own
#: and must not be counted as a room.
#:
#: Note what this deliberately does NOT match: "four bed" and "4 bed room"
#: carry a *bed count* word, and the hotel's own catalogue has a room named
#: "DELUXE ROOM FOUR BED" that a guest asking for a "4 bed" is asking for
#: (`allocation_engine.normalize_bed_counts`, `validator.py:802`,
#: `tests/test_feature_filter_dead_end.py`). Only a fragment with no word left
#: once the bedding is removed is rejected — removing "bed" from "four bed"
#: leaves "four", and from "bed room" leaves "room", so both survive.
_BEDDING_ONLY_RE = re.compile(
    r'^(?:[\s\-]*(?:' + "|".join(_BEDDING_WORDS) + r'|\d+)[\s\-]*)+$'
)


def is_bedding_only(room_type: str) -> bool:
    """
    Whether a room type names only bedding, and therefore no room.

    The 2026-09-26 defect: a guest wrote "2 room chahiye, 1- 2 bed, 2- 4 bed" —
    two rooms, and how the beds in each are arranged. The numbered list was read
    as two *requests*, `parse_room_requirements` returned `("bed", 2)` and
    `("bed", 4)`, `len(req_items) >= 2` passed, and `gemini_service` replaced the
    guest's real "2 rooms" with a sum of the bed counts.

    A bare bed word is not a room noun on its own, so this is the second half of
    the noun check in `parse_room_requirements` — and the same rule the sums in
    `gemini_service` use, so a bedding line that reaches them from the model
    rather than from this scanner is excluded there too.
    """
    return bool(_BEDDING_ONLY_RE.match((room_type or "").strip().lower()))

#: `24/10/26`, `24-10-26`, `24.10.26`. A count digit sitting against one of
#: these is part of a date.
_DATE_DELIMS = "/-.\\:"

_WORD_ALT = "|".join(WORD_NUMS)

#: The count, then a short run of word characters: "3 kashi classic room".
#: `\s+` is load-bearing and must stay a `+` — with `\s*` a count glued to a
#: word ("4yrs", "3pm") would match as a count.
_COUNT_RE = re.compile(
    r'\b(\d+|' + _WORD_ALT + r')\s+(?:x\s+)?([a-z0-9\s\-]+)'
)

#: A fragment led by one of these is a *guest* count, not a room count —
#: "2 people", "3 nights". Unchanged from the block this module was extracted
#: from; it is the reason `2 persons` never became two rooms.
#:
#: "head"/"heads" joined the list on 2026-11-07, after guest 917003620122 wrote
#: "4 heads one tripple bed" — four people, one triple-bed room. "heads" was
#: not a guest word, so the count stayed attached to the room and the guest was
#: quoted **four triple-bed rooms**, each filled to three adults with a mattress
#: (Rs 12,600/night for four people). It is trade vocabulary — a B2B agent
#: quoting head counts — so it belongs here beside "pax", which is the same
#: word in a different spelling.
_GUEST_LED_RE = (
    "adult", "guest", "person", "people", "child", "kid", "log", "pax",
    "night", "raat", "head",
)

_SPLIT_RE = re.compile(r'(?:\band\b|\baur\b|\+|\&|,)')
_EXTRA_BED_RE = re.compile(r'\b(?:extra\s*bed|extra\s*gadda|with\s*extra)\b')
_NOISE_RE = re.compile(
    r'\b(?:i\s+want|need|give\s+me|with\s+extra\s*bed|with\s+extra|extra\s*bed'
    r'|please|book|chahiye)\b'
)
_TRAILING_ASK_RE = re.compile(
    r'\b(?:\d+\s*nights?|total\s*(?:price|amount|cost|charges?)|after\s*discount|discount\s*ke\s*baad|after\s*offers?|kindly\s*(?:mention|tell|send|bataiye)|price\s*batao|rate\s*batao).*$',
    re.IGNORECASE,
)


def _is_part_of_a_date(part: str, match: re.Match) -> bool:
    """
    Whether the matched number is a piece of a date rather than a count.

    `24/10/26` itself will not match — `_COUNT_RE` needs whitespace after the
    digits and there is a `/` there. What this catches is a number that ends a
    date and is followed by a space, which is exactly `26 at 5pm ...` in
    `cheack in 24/10/26 at 5pm`.

    The emptiness check on each side is required, not defensive: `"" in "/-."`
    is True in Python, so an unguarded `in` would report every number at the
    start or end of a fragment as a date.

    The delimiter test above only fires when the separator is still attached.
    `date_masking.is_position_in_a_date` is asked as well, so a number inside a
    date that is written with a space — `15 Nov 26`, `5 to 7 sep` — is caught
    too. This module carried the repo's only date guard for a while; it is now
    one caller of the shared one rather than a second implementation of it.
    """
    before = part[match.start(1) - 1] if match.start(1) > 0 else ""
    after = part[match.end(1):match.end(1) + 1]
    if (bool(before) and before in _DATE_DELIMS) or (
        bool(after) and after in _DATE_DELIMS
    ):
        return True
    return is_position_in_a_date(part, match.start(1), match.end(1))


def _clean_room_type(rtype: str) -> str:
    cleaned = _TRAILING_ASK_RE.sub("", rtype)
    return _NOISE_RE.sub("", cleaned).strip()


def parse_room_requirements(user_message: str) -> list[dict]:
    """
    The `(room_type, count)` pairs this message actually supports, in order.

    At most one requirement per `+`/`and`/`,`-separated fragment, matching the
    scan this replaced. Returns [] when the message names no room quantities.
    """
    if not user_message:
        return []

    items: list[dict] = []

    for part in _SPLIT_RE.split(user_message.lower().strip()):
        part = part.strip()
        if not part:
            continue

        with_extra_bed = bool(_EXTRA_BED_RE.search(part))

        # `search` in a loop rather than `finditer`, because `_COUNT_RE`'s tail
        # is greedy: a rejected match has usually swallowed the real request
        # behind it. In "24/10/26 2 deluxe rooms" the date match takes "2 deluxe
        # rooms" whole, and `finditer` would resume after it and never look
        # again. Resuming just past the digits puts the swallowed text back in
        # front of the scanner.
        pos = 0
        while pos < len(part):
            match = _COUNT_RE.search(part, pos)
            if match is None:
                break
            pos = max(match.end(1), pos + 1)

            if _is_part_of_a_date(part, match):
                continue

            raw = match.group(1).lower()
            count = WORD_NUMS.get(raw, int(raw) if raw.isdigit() else 1)
            rtype = match.group(2).strip()

            if rtype.startswith(_GUEST_LED_RE):
                # "2 people", "3 nights" — re-read the fragment for a count
                # that leads a room name instead, or move on.
                inner = _COUNT_RE.search(rtype)
                if not inner:
                    continue
                inner_raw = inner.group(1).lower()
                count = WORD_NUMS.get(
                    inner_raw, int(inner_raw) if inner_raw.isdigit() else 1
                )
                rtype = inner.group(2).strip()

            rtype = _clean_room_type(rtype)

            if not rtype or count < 1:
                continue
            if not _ROOM_NOUN_RE.search(rtype):
                # No room word anywhere in the fragment: not a room request.
                continue
            if is_bedding_only(rtype):
                # "bed", "2 bed" — a room word is present but it is the only
                # thing there, and it names a mattress. See `is_bedding_only`.
                continue

            items.append({
                "room_type": rtype,
                "count": count,
                "is_foc": (
                    "foc" in rtype or "free" in rtype or "complimentary" in rtype
                ),
                "with_extra_bed": with_extra_bed,
            })
            break

    return items


#: The most rooms any single request may ask for when the party size is not
#: known. A real group booking is nowhere near this; the only things above it
#: are misparses, and the one that reached production asked for 29.
MAX_ROOMS_WHEN_PARTY_UNKNOWN = 20


def validate_room_requirements(
    items: list[dict],
    party_size: int = 0,
) -> tuple[list[dict], str | None]:
    """
    Bound a parsed request against the party, and say why when it cannot be
    honoured.

    `parse_room_requirements` rejects a *date* being read as a count. This
    rejects a count that is simply not a number of rooms a party could want —
    the second half of the same hole. The 29-room quote needed both to be
    wrong at once, and only the first is closed in the reader: "5 deluxe" for a
    party of two parses perfectly and is still not a booking.

    The rule is one line: **a room sleeps at least one guest, so a request for
    more rooms than there are guests leaves a room with nobody in it.** FOC
    rooms are exempt — a complimentary room for a driver is not a room the
    party sleeps in, so it is not counted against the party.

    All-or-nothing on purpose. The count is what is wrong, not the room type,
    so quietly dropping the excess would quote a booking nobody asked for —
    worse than asking again. On a failure the caller gets `[]` and a sentence
    to put in front of the guest.

    Called from `understand`'s `need_custom_rooms` branch (`understand.py:985`),
    which is where a party size the extraction can be trusted to have is finally
    in hand. This docstring said "Nothing calls this yet" until 2026-09-25, long
    after that stopped being true: the parse fix and the allocator guard closed
    the *date* half of the hole while the note stayed behind.

    Returns `(items, None)` when the request stands, `([], reason)` when it
    does not, and `([], None)` when there was nothing to validate.
    """
    if not items:
        return [], None

    sleeping_rooms = sum(
        int(item.get("count") or 0) for item in items if not item.get("is_foc")
    )

    if party_size > 0 and sleeping_rooms > party_size:
        return [], (
            f"That is {sleeping_rooms} rooms for {party_size} "
            f"{'guest' if party_size == 1 else 'guests'} — more rooms than "
            f"people. Could you check the room count?"
        )

    if party_size <= 0 and sleeping_rooms > MAX_ROOMS_WHEN_PARTY_UNKNOWN:
        return [], (
            f"That is {sleeping_rooms} rooms. Could you confirm how many "
            f"guests are staying?"
        )

    return items, None


# ── the per-room split ───────────────────────────────────────────────────────
#
# "Room 1 2 people, room 2 4 people, room 3 2 people tonight ac"
#
# `parse_room_requirements` reads *how many of a room type*, and nothing else.
# It has no way to say "one room for two, one for four, one for two" — the idea
# is not in its vocabulary, and the pairs it returns cannot carry it. On
# 2026-09-25 that message was answered as "seat 8 AC guests as cheaply as
# possible", which is two rooms at Rs 5,500 rather than the three at Rs 7,750
# the guest had just described. The guest lost the split, not the arithmetic.
#
# So this reader exists alongside it, with a different output: each line is one
# room (or several identical rooms) and carries the number of **adult seats**
# that room has to provide. `room_type` stays empty — the guest named an
# occupancy, not a category, and the allocator picks the cheapest room that
# seats each line.
#
# It returns None, never [], when there is no split — the same distinction the
# module docstring draws: nothing means leave the caller's own extraction alone.

def parse_split_occupancies(user_message: str) -> list[dict]:
    """Parse per-room occupancy splits from a guest message.

    E.g.:
      "For 4 person and another one for 2 person" ->
        [{room_type: "", count: 1, guests: 4}, {room_type: "", count: 1, guests: 2}]
      "2 people 1 room, 3 people 1 room, 4 people 1 room all ac" ->
        3 lines with guests 2, 3, 4.

    Reconciles with explicit total rooms (e.g. 'No room-2' or '2 rooms') so that
    a request for 2 rooms split across 2 requirements never allocates 3+ rooms.
    """
    if not user_message or not user_message.strip():
        return []

    text = user_message.strip()
    m_tr = re.search(
        r'\b(?:no\.?\s*(?:of\s*)?rooms?|number\s*of\s*rooms?|total\s*rooms?)\s*[:-]?\s*(\d+)\b|\b(\d+)\s*rooms?\b',
        text,
        re.I,
    )
    total_rooms = int(m_tr.group(1) or m_tr.group(2)) if m_tr else None

    from app.utils.ac_text import ac_preference_from_text

    clauses = re.split(r'[\n,;]|\band\b|\baur\b|\+', text)
    items: list[dict] = []

    for clause in clauses:
        cl = clause.strip()
        if not cl:
            continue
        m_pax = re.search(
            r'\b(\d+|one|two|three|four|five|six)\s*(?:persons?|people|adults?|guests?|pax|members?|log)\b',
            cl,
            re.I,
        )
        if not m_pax:
            m_pax = re.search(
                r'\b(\d+|one|two|three|four|five|six)\s*(?:in|me|mein)\s*(?:another|one|each|ek|\d+)?\s*rooms?\b',
                cl,
                re.I,
            )
        if m_pax:
            raw_p = m_pax.group(1).lower()
            word_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
            pax = word_map.get(raw_p, int(raw_p) if raw_p.isdigit() else 1)

            m_rc = re.search(r'\b(\d+)\s*rooms?\b', cl, re.I)
            if m_rc:
                rc = int(m_rc.group(1))
            else:
                rc = 1

            ac_class = ac_preference_from_text(cl)
            items.append({
                "room_type": "",
                "count": rc,
                "guests": pax,
                "is_foc": False,
                "with_extra_bed": False,
                "ac": ac_class if ac_class != "NO_PREFERENCE" else None,
            })

    if len(items) >= 2:
        if total_rooms and len(items) == total_rooms:
            for it in items:
                it["count"] = 1
        elif total_rooms and sum(it["count"] for it in items) > total_rooms:
            for it in items:
                it["count"] = 1
        return items

    return []


def is_an_explicit_split(room_requirements) -> bool:
    """
    Does this requirement list come from a guest-stated per-room split?

    Read off the lines themselves rather than carried in a state key — a new
    key has to be declared in `schemas.py` and classified in
    `extraction_contract`, and `tests/test_state_contract.py` fails until it is.
    The `guests` key is the marker.

    It has *two* writers, not one: `parse_split_occupancies` below, and the
    model's room-request extractor (see 5c-ii in `gemini_service`), which fills
    it in by dividing a stated party across the rooms. So "has `guests`" is not
    the same question as "did the guest state a per-room figure" — this answers
    only the former. `split_party_from_message` answers the latter, and it is
    what the party count must be taken from.
    """
    for item in room_requirements or []:
        if isinstance(item, dict):
            if item.get("guests"):
                return True
        elif getattr(item, "guests", None):
            return True
    return False


def split_party_from_message(user_message: str) -> int:
    """The head count the guest themselves spread across rooms, or 0 if none.

    `guests` on a room-requirement line is a per-room *ceiling*, and its two
    writers mean different things by it:

      * `parse_split_occupancies` writes it only when the message carries two
        or more clauses that each name a head count — "2 people 1 room, 3
        people 1 room, 4 people 1 room". That is the guest's own arithmetic,
        and its total is the party.
      * the model's extractor writes it whenever it sees a party that needs
        more than one room, and it *derives* the figure by dividing the stated
        total and rounding up. "3 rooms from 25 Dec to 29 Dec for 8 adults"
        comes back as one line of count 3 and `guests: 3` — nine seats for
        eight people.

    Summing `guests` is therefore only authoritative when the guest said it,
    and `parse_split_occupancies` is what tells the two apart. Measured against
    the shapes that must *not* fire — "3 rooms for 8 adults", "3 rooms, 8
    adults and 2 children", "2 rooms for 4 people", "2 adults and 1 child" —
    it returns nothing for every one of them, because a single occupancy
    clause is a total, not a split.

    0 therefore means "the guest did not spread a party across rooms", not
    "no party".
    """
    items = parse_split_occupancies(user_message or "")
    return sum(int(i.get("count") or 1) * int(i.get("guests") or 0) for i in items)


def reconcile_split_with_party(room_requirements, party_total: int):
    """Re-cut a split's lines so their seats add up to `party_total`.

    The extractor's per-room figure is a rounded-up quotient, so the lines can
    claim more people than the guest brought: three rooms of three for "8
    adults". Written through unchanged, the search prices nine and the guest is
    quoted for a party one room-seat larger than the one they described.

    The fix is not to drop the split — the per-room seating is what lets the
    allocator put three people in one room and two in another — but to
    redistribute the stated total across the rooms it names. 8 over 3 rooms
    becomes 3, 3, 2.

    Returns the request unchanged when there is nothing to reconcile: no
    occupancy on any line, or more rooms than people (a misparse that
    `validate_room_requirements` refuses; papering over it would hide it).
    """
    items = [
        i.model_dump() if hasattr(i, "model_dump") else dict(i)
        for i in (room_requirements or [])
        if i is not None
    ]
    if not items or int(party_total or 0) <= 0:
        return list(room_requirements or [])
    if not any(int(i.get("guests") or 0) for i in items):
        return list(room_requirements or [])

    rooms = sum(int(i.get("count") or 1) for i in items)
    if rooms <= 0:
        return list(room_requirements or [])
    # Nothing to reconcile: the lines already say what the guest said. Left
    # exactly as they are, so a correct split keeps the extractor's per-room
    # figures instead of being redistributed into an equally-sized set.
    if sum(int(i.get("count") or 1) * int(i.get("guests") or 0) for i in items) == int(party_total):
        return list(room_requirements or [])
    base, remainder = divmod(int(party_total), rooms)
    if base < 1:
        return list(room_requirements or [])

    out: list[dict] = []
    for item in items:
        for _ in range(int(item.get("count") or 1)):
            line = dict(item)
            line["count"] = 1
            line["guests"] = base + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            out.append(line)
    return out
