"""
Room packing — how many of which room a party actually needs.

Why this module exists
----------------------
The rooms card has to answer one question for every party size: *which set of
rooms seats these people, and how much does each set cost?* That is an
optimisation, not a lookup, and getting it wrong is a money bug in both
directions — a greedy pick over-charges a guest who could have been seated
cheaper, and an unchecked pick sells rooms the hotel cannot supply.

Two rules define a valid set:

  1. **The cheapest set of rooms that holds the party, each filled to its own
     real ceiling.** This rule has been rewritten twice and the two earlier
     versions are worth knowing, because both were arguing about a berth that
     does not exist.

     The client's original, 2026-09-19: *"Extra beds are not used on this card
     at all — a party that does not fit the base occupancy of the rooms it was
     given is given more rooms instead."* Reversed on 2026-09-23 by Product, on
     the argument that the guest is buying a place to sleep and the card should
     quote the cheapest one actually for sale — at Sample Hotel a party of three
     saw a Deluxe Family Room at Rs 3,875 while one Deluxe King Bed with a
     mattress came to Rs 3,150, the exact room the guest had asked about, Rs 725
     cheaper and invisible.

     The reversal was right and its reasoning was wrong. The third guest was
     never on the mattress; the Deluxe King's own `max_adult_occupancy` is 3, so
     the room seats three on its own beds and the Rs 600 bought comfort, not a
     berth (see `allocation_engine.distribute_guests`, rewritten 2026-09-25).
     The card was under-quoting the room, not over-quoting it — the same room
     should have been on the card at its own rate all along, and is now.

     So the rule this module actually implements is the plain one: every room
     is packed to the ceiling the hotel states for it, a mattress is a priced
     add-on that seats nobody, and the cheapest set wins. A party of two is
     never upsold a bed, because a bedded candidate seats the same two people
     for more money and the search never reaches for it.

  2. **An option never mixes air-conditioning classes.** A party split across
     an AC and a non-AC room is not one product, and quoting it as one number
     hides that from the guest. AC is a filter on the candidate set, applied
     before anything is packed.

Why a knapsack and not a greedy pick
------------------------------------
The obvious implementation — repeatedly take the cheapest room that fits the
remaining guests — is wrong whenever a larger room is cheaper per seat than a
smaller one. If a 3-seat room costs Rs 1,850 and a 2-seat room costs Rs 1,550,
greedy fills 5 guests as 3+2 = Rs 3,400; but if a 4-seat room ever came in
under Rs 3,100 the greedy answer would miss it. `min_cost_packing` solves the
minimum over all combinations exactly, by dynamic programming over the party
size. The search space is tiny (party size x room types), so exactness is free.

The packing is unbounded: the same room type may be taken any number of times,
which is what "3 x Deluxe" means.

Grades
------
`room_grade` reads a quality word out of the room's own name:

    STANDARD  <  DELUXE  <  PREMIER  <  SUITE  <  EXECUTIVE

This is a string convention, not a field, and it is the weakest part of the
module — see the note on `room_grade`. Price is deliberately *not* used to
derive a grade: a bigger room in a cheaper grade costs more than a smaller room
in a better one, so ranking by price ranks by size and calls a Standard Triple
Bed a step up from a Standard Room. Price is used only to break ties *within* a
grade.

Pure module: no I/O, no database, no imports from the service layer. Everything
here is a function of its arguments, so the whole decision surface is testable
without a hotel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.utils.ac_text import ac_preference_from_text

# Ordered worst to best. The client's catalogue uses the first three; SUITE and
# EXECUTIVE are included because a hotel that names rooms "Suite" should not
# have them land at STANDARD alongside the budget rooms.
ROOM_GRADES: tuple[str, ...] = ("STANDARD", "DELUXE", "PREMIER", "SUITE", "EXECUTIVE")

# The two AC classes a room can belong to, as `room_matches_feature` words them.
AC_CLASS = "AC"
NON_AC_CLASS = "Non-AC"


def room_grade(name: str | None) -> int:
    """
    The quality index of a room, from its name. Unknown names are STANDARD.

    This is a string convention and it is the honest weak point of this module.
    A hotel whose rooms are named "Kodai Hut" and "Kodai Cottage" gets both at
    index 0, so every room ties on grade and the packer falls back to price
    within the class — which still produces a correct *cheapest*, but a "comfy"
    that means "the bigger one" rather than "the better one".

    The fix is a grade field on the room document rather than a name match.
    Until that exists, this is the documented behaviour, and it degrades to
    something sane rather than raising.

    Matching is on the whole uppercase name, so "PREMIER" is found in
    "PREMIER ROOM KING BED". Highest match wins, so a hypothetical
    "DELUXE SUITE" grades as SUITE.
    """
    upper = (name or "").upper()
    best = 0
    for index, grade in enumerate(ROOM_GRADES):
        if grade in upper and index > best:
            best = index
    return best


@dataclass(frozen=True)
class RoomCandidate:
    """
    One room type offered as a packing option.

    `price` is per night, inclusive of taxes, matching what the card prints.
    `base_adults` is the boundary the packer fills to — and it is the room's
    **ceiling**, `max_adult_occupancy`, not the occupancy the room rate covers.
    A Deluxe King (base 2 / max 3) arrives here as `base_adults=3` whether or
    not anyone bought a mattress, because the hotel seats three either way.
    `is_ac` must come from `room_matches_feature(room, "AC")` at the caller, not
    from string matching here — the packer must not re-derive a fact the
    inventory layer already answers, or the two can disagree.

    `beds` is how many extra mattresses this room is sold with, and it is
    **already inside `price`** — a mattress is never sold without being paid
    for. It is **not** the difference between `base_adults` and the rate's
    occupancy; that difference is a normal guest. A Deluxe King with a mattress
    bought arrives as `base_adults=3, beds=1, price=<room rate + mattress
    charge>`; the same room with none arrives as `base_adults=3, beds=0,
    price=<room rate>`. Two candidates that seat the same three people, and the
    cheaper one wins — which is why a mattress only ever appears in a packed set
    when the guest asked for it.

    The packer deliberately does not add beds itself: the allocator already
    decided which rooms may have one, against the hotel's own
    `extra_bed_allowed` / `max_extra_beds`. Re-deriving that here would give the
    card a second opinion about a hotel's bed policy, and the two could disagree
    — the card quoting a mattress the final quote refuses, which is the
    strongest argument for keeping this a plain number the caller supplies.

    `beds=0` is every room that is not sold with one, and is what every caller
    meant before 2026-09-23.
    """

    name: str
    base_adults: int
    price: float
    is_ac: bool
    beds: int = 0
    #: The catalogue's `max_occupancy` — how many people the hotel will put in
    #: the room, adults and children together. A different number from
    #: `base_adults`, which is the *adult berth* ceiling. Defaults to 0, which
    #: `head_capacity` reads as "unstated"; see there.
    max_occupancy: int = 0
    grade: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grade", room_grade(self.name))

    @property
    def ac_class(self) -> str:
        return AC_CLASS if self.is_ac else NON_AC_CLASS

    @property
    def head_capacity(self) -> int:
        """
        How many people the hotel will put in this room: its `max_occupancy`.

        The second of a room's two ceilings, and not the same number as
        `base_adults`. `base_adults` is how many *adults* it sleeps; this is how
        many *people* it holds. A Deluxe Room Four Bed seats 5 adults and holds
        6 — the sixth is a child sharing a bed, who needs a head's worth of room
        and no berth of their own. A set is only sellable when it clears both.

        `max(base_adults, …)` and not the ceiling alone: the schema defaults
        `max_occupancy` to a value a document may never state, and a room that
        arrived here holding nobody would drop out of every packing. The head
        ceiling is never below the berth ceiling by definition — the builder
        computes `max(max_adults, base_adults + max_children)` — so falling back
        to the berths is the safe read. It degrades a candidate with no stated
        `max_occupancy` to the pre-head behaviour instead of excluding it.
        """
        return max(self.base_adults, self.max_occupancy)

    def packable(self) -> bool:
        """
        Whether this room may be packed at all.

        A room with no base capacity cannot seat anyone. A room with no price
        is excluded deliberately: `get_available_rooms` returns
        `price_per_night` as 0 for every room, so a zero price means "we failed
        to price this", not "this room is free" — and a free room would win
        every cheapest comparison and be quoted to a guest at Rs 0.
        """
        return self.base_adults > 0 and self.price > 0


@dataclass(frozen=True)
class Packing:
    """
    A set of rooms that seats a party, and what it costs per night.

    `counts` is ordered by room name so the card prints the same combination
    the same way every time; a set built from a dict would otherwise depend on
    insertion order and make the tests flaky.

    `beds` is the extra mattresses in the set, keyed by the room name they are
    in, and it is carried separately rather than folded into `counts` because
    the two mean different things to a guest: `counts` is how many rooms,
    `beds` is how many of those rooms have a mattress in them. A set of two
    Deluxes where only one takes a bed is `counts=(('DELUXE', 2),)` and
    `beds=(('DELUXE', 1),)`, and the card prints it as two rooms plus one extra
    bed — which is what the guest is being sold and what the hotel has to make
    up. Folding it into the name would print a room type that does not exist.

    Keyed by name rather than by position because a bed is a fact about a room
    type, not about the second room on the list. It defaults to `()` so every
    construction that predates beds — and every test that builds one — is still
    exactly right.
    """

    counts: tuple[tuple[str, int], ...]
    total_per_night: float
    beds: tuple[tuple[str, int], ...] = ()

    @property
    def room_count(self) -> int:
        return sum(count for _, count in self.counts)

    @property
    def bed_count(self) -> int:
        return sum(count for _, count in self.beds)

    def describe(self) -> str:
        """`2 x Deluxe Room King Bed + 1 x Premier Room King Bed`."""
        text = " + ".join(f"{count} x {name}" for name, count in self.counts)
        if self.bed_count:
            text += f" + {self.bed_count} extra bed" + ("s" if self.bed_count > 1 else "")
        return text

    def as_dict(self) -> dict:
        return {
            "rooms": [{"name": name, "count": count} for name, count in self.counts],
            "beds": [{"name": name, "count": count} for name, count in self.beds],
            "total_per_night": self.total_per_night,
            "room_count": self.room_count,
        }


def min_cost_packing(
    party: int,
    candidates: list[RoomCandidate],
    head_party: int | None = None,
) -> Packing | None:
    """
    The cheapest set of rooms that seats `party`, or None if they cannot be seated.

    Exact, by dynamic programming over **two** capacities. `best[s][h]` is the
    cheapest set that fills `s` berths and holds `h` heads, both capped at what
    the party needs, where taking a room with `b` berths and `o` heads moves to
    `(min(party, s + b), min(head_party, h + o))` — so a room that over-seats
    the remainder still counts, and the answer is "at least `party` berths and
    at least `head_party` heads", not "exactly".

    Two dimensions because a room has two ceilings and they are different
    numbers. `base_adults` is how many *adults* the room sleeps; `head_capacity`
    is how many *people* the hotel will put in it. A Deluxe Room Four Bed seats
    5 adults and holds 6 people — the sixth is a child sharing a bed, who needs
    a head's worth of room and no berth of their own. On 2026-09-29 that room
    cost Rs 3,400 and the card quoted Rs 3,600 for two King rooms, because this
    search knew only about berths and so could not see that one room already
    held everyone.

    `head_party` defaults to `party`. The two are the same number whenever the
    party holds no free child — a paid child takes a berth exactly as an adult
    does, so only a child the hotel does not charge for is absent from `party` —
    which is every call site that predates this parameter. A caller that does
    not pass it keeps its exact old answer.

    Ties are broken toward fewer rooms, then toward the alphabetically earlier
    room, so the result is a deterministic function of the input. Two packings
    at the same price are not the same product to a guest — fewer rooms means
    less to walk between — so the tie-break is a real preference, not just
    test hygiene.

    Beds need no search of their own: they change a candidate's price and never
    its seats, so "1 x Deluxe, 3 seats, Rs 3,550" and "1 x Deluxe, 3 seats,
    Rs 4,150 with a mattress" are two ordinary candidates and the search takes
    the cheaper one unless the guest asked for the mattress. The bed is threaded
    through the search only so the finished `Packing` can say which room has it.
    """
    party = int(party or 0)
    if party <= 0:
        return None
    # Never below the berths: everyone who needs a bed is also a person in the
    # room, so a head count under the seat count is a caller error rather than a
    # smaller party. Clamping here keeps one dimension from being smaller than
    # the other, which the grid below assumes when it caps both at the goal.
    heads = max(int(head_party) if head_party is not None else party, party)

    usable = sorted(
        (c for c in candidates if c.packable()),
        key=lambda c: (c.grade, c.price, c.name),
    )
    if not usable:
        return None

    # best[s][h] = (cost, room_count, names, beds) for a set filling `s` berths
    # and holding `h` heads; None = unreachable. `beds` holds one (name,
    # mattresses) pair per room taken, in the same order as `names`, so both can
    # be summed back into the Packing. It is the *fourth* element deliberately:
    # the tie-break above stays cost, then fewer rooms, then the room names,
    # exactly as it was before beds existed, and a bed separates only the sets
    # those three could not.
    #
    # Filled forward rather than by the old pull-style recurrence. Every room
    # adds at least one berth — a candidate with none is not `packable` — so
    # every transition strictly increases `s + h`, and the grid is a DAG: one
    # ascending pass settles each state before anything reads it. The one
    # transition that does not move is a set already at `(party, heads)`, and it
    # only ever re-offers itself at a higher cost, so it is discarded by the
    # comparison rather than looping.
    best: list[
        list[tuple[float, int, tuple[str, ...], tuple[tuple[str, int], ...]] | None]
    ] = [[None] * (heads + 1) for _ in range(party + 1)]
    best[0][0] = (0.0, 0, (), ())

    for berths in range(party + 1):
        for held in range(heads + 1):
            state = best[berths][held]
            if state is None:
                continue
            for candidate in usable:
                next_berths = min(party, berths + candidate.base_adults)
                next_held = min(heads, held + candidate.head_capacity)
                reach = (
                    state[0] + candidate.price,
                    state[1] + 1,
                    state[2] + (candidate.name,),
                    state[3] + ((candidate.name, candidate.beds),),
                )
                current = best[next_berths][next_held]
                if current is None or reach < current:
                    best[next_berths][next_held] = reach

    final = best[party][heads]
    if final is None:
        return None

    counts = Counter(final[2])
    ordered = tuple(sorted(counts.items()))
    bed_totals: Counter[str] = Counter()
    for name, beds_in_room in final[3]:
        if beds_in_room:
            bed_totals[name] += beds_in_room
    return Packing(
        counts=ordered,
        total_per_night=round(final[0], 2),
        beds=tuple(sorted(bed_totals.items())),
    )


def _pool(candidates: list[RoomCandidate], want_ac: bool) -> list[RoomCandidate]:
    return [c for c in candidates if c.is_ac is want_ac and c.packable()]


def _card_class_pool(
    candidates: list[RoomCandidate], ac_preference: str | None
) -> list[RoomCandidate]:
    """
    The one AC class a card is allowed to draw on.

    Unstated means AC. A guest who never mentioned air conditioning sees the AC
    range — the default the rest of the bot already applies when the question
    is not answered — and never the cheaper class, because a card that opens on
    non-AC and climbs into AC is two different products on one ladder.

    Non-AC is shown only to a guest who asked for it by name.

    A hotel with nothing in the guest's class still gets a card — the pool
    falls back to the other class rather than coming back empty, which would
    leave a non-AC-only hotel with no rows at all. That fallback is the one
    place a card crosses classes, and it is unavoidable: the alternative is
    telling the guest nothing.
    """
    preference = (ac_preference or "").strip().upper()
    want_ac = preference != "NON_AC"
    return _pool(candidates, want_ac) or _pool(candidates, not want_ac)


def cheapest_and_comfy(
    party: int,
    candidates: list[RoomCandidate],
    ac_preference: str | None = None,
    head_party: int | None = None,
) -> tuple[Packing | None, Packing | None]:
    """
    The first two priced rows of the rooms card, both in one AC class.

    Option 1 — cheapest, inside the guest's class:

        "AC"      -> cheapest AC packing
        "NON_AC"  -> cheapest non-AC packing
        unstated  -> cheapest AC packing

    Option 2 — comfy: the cheapest packing in the *same* class that ranks
    strictly above every room type option 1 used, so it is a real step up. When
    nothing ranks above, this returns option 1 unchanged and `card_rows` drops
    the repeat rather than printing one set under two labels.

    Both rows share a class because a card is one product. Until 2026-09-24
    option 2 was *always* AC whatever the guest had asked for: a guest who said
    "non-AC" was quoted AC rooms in rows 2 and 3, and the ROOMS list above them
    followed the rows, so two thirds of their card was rooms they had ruled
    out. That rule was added on 2026-09-19 to stop rows 1 and 2 coming out
    identical on a hotel whose non-AC range has a single grade — but showing
    fewer rows answers that case, and `card_rows` has dropped repeats since
    2026-09-23, so the cross-class pair is no longer what prevents it.

    Returns (cheapest, comfy). Either may be None if nothing is packable.

    `head_party` is passed straight to the packer — see `min_cost_packing`. It
    is a fifth parameter rather than a reworked second, because every existing
    call site wants the default and a card that forgot to pass it would still
    draw today's card rather than nothing.
    """
    pool = _card_class_pool(candidates, ac_preference)

    cheapest = min_cost_packing(party, pool, head_party)
    if cheapest is None:
        return None, None

    # Step up by (grade, price) rank, and never invent a row above the top of
    # the ladder — repeating option 1 here is what lets `card_rows` dedupe it.
    ranked = sorted(pool, key=lambda c: (c.grade, c.price, c.name))
    top_rank = max(
        (index for index, c in enumerate(ranked) if c.name in dict(cheapest.counts)),
        default=-1,
    )
    above = ranked[top_rank + 1:]
    comfy = min_cost_packing(party, above, head_party) if above else None
    if comfy is None:
        comfy = cheapest

    return cheapest, comfy


def upgraded_row(
    party: int,
    candidates: list[RoomCandidate],
    comfort: Packing | None,
    ac_preference: str | None = None,
    head_party: int | None = None,
) -> Packing | None:
    """
    The third row — one step above `comfort` on the same ladder, or None.

    Deliberately a separate function rather than a third return value from
    `cheapest_and_comfy`. That function's two-row contract is documented and
    nine call sites unpack it as a pair; widening it would have meant editing
    all of them to keep a card decision from leaking into the packer's API.
    This is additive: the first two rows are computed exactly as before, and
    this one is derived from the second.

    It steps the ladder the same way `cheapest_and_comfy` steps from row 1 to
    row 2 — rank the guest's own class pool by (grade, price, name), find the
    highest-ranked room type `comfort` used, and pack from strictly above it.
    So "Upgraded" is a real step in quality, not merely a bigger bill.

    The class is the guest's, not AC's. Deriving the third row from the AC pool
    regardless would put an AC room under a non-AC card for the same reason the
    second row used to — see `cheapest_and_comfy`.

    None is a real answer, and the card prints two rows when it gets one. A
    hotel with a single grade has nothing above `comfort`, and inventing a
    third row there would print the same set twice under two labels — which is
    the complaint that produced the two-row card in the first place.
    """
    if comfort is None:
        return None

    pool = _card_class_pool(candidates, ac_preference)
    if not pool:
        return None

    ranked = sorted(pool, key=lambda c: (c.grade, c.price, c.name))
    top_rank = max(
        (index for index, c in enumerate(ranked) if c.name in dict(comfort.counts)),
        default=-1,
    )
    above = ranked[top_rank + 1:]
    if not above:
        return None

    upgraded = min_cost_packing(party, above, head_party)
    # Beds are part of a set's identity: "1 x Deluxe + 1 bed" and "1 x Deluxe"
    # are two different products even though `counts` is the same for both, and
    # the card draws them as two different lines.
    if upgraded is None or (upgraded.counts, upgraded.beds) == (comfort.counts, comfort.beds):
        return None
    if upgraded.total_per_night <= comfort.total_per_night:
        return None
    return upgraded


def wants_both_classes(value: str | None) -> bool:
    """
    Did the guest *explicitly* ask to see both AC classes?

    This is a narrower question than `ac_preference_from_state`, and the
    difference is the whole point. That reader answers "which class should the
    card draw?", and it folds an explicit "show both" in with "never said" so
    the card has one class to draw. Correct for one card, wrong for this: a
    guest who tapped **Show both** has stated a preference, and a guest who
    never mentioned air conditioning has not. Only the explicit spellings count
    here — `SHOW_BOTH` / `BOTH`, which `understand` and the button registry both
    write, and `NO_PREFERENCE`, which the shared AC resolver produces from
    "dono dikhaiye" and its cousins.

    Unstated text returns False deliberately. Widening the card to two classes
    for every guest who never asked would double the length of the commonest
    card in the bot, and nobody complained about that one.
    """
    return (value or "").strip().upper() in ("SHOW_BOTH", "BOTH", "NO_PREFERENCE")


def both_class_rows(
    party: int,
    candidates: list[RoomCandidate],
    head_party: int | None = None,
) -> list[Packing | None]:
    """
    The three rows a "show both" card carries: Non-AC, AC, then a step up.

    The guest asked to see both classes, and until now they could not: the card
    drew from a single class (`_card_class_pool`), and — worse — the allocator
    cut its option list to the ten cheapest combinations before the card was
    built, so on any large party the AC sets were ranked out entirely and never
    reached this function's input at all. The allocation side of that fix is in
    `allocation_engine._keep_both_classes`; this is the drawing side.

    Row 1 is the cheapest set in the guest's cheaper class and row 2 the cheapest
    in the other one, in the order the guest is most likely to want to read
    them: price first. Row 3 steps up from row 2 (AC in practice, and in the
    client's own example) using the same rule as every other third row — a
    higher grade, not merely a bigger bill.

    Returns `[row1, row2, row3]` where row 3 may be None, or an **empty list**
    when either class cannot seat the party. Empty is the important answer: the
    caller then keeps the ordinary one-class card rather than printing a
    two-class card with a class missing from it. A hotel with no non-AC rooms,
    or none free on the guest's dates, is the common case and must not lose
    rows to a feature aimed at a case it does not have.

    This is additive on purpose, like `upgraded_row` before it. The two-row
    contract of `cheapest_and_comfy` is unpacked as a pair at nine call sites;
    widening it would have meant editing all of them to add a row the caller
    may not want.
    """
    non_ac = min_cost_packing(party, _pool(candidates, False), head_party)
    ac = min_cost_packing(party, _pool(candidates, True), head_party)
    if non_ac is None or ac is None:
        return []
    return [non_ac, ac, upgraded_row(party, candidates, ac, "AC", head_party)]


def class_can_seat(
    party: int,
    candidates: list[RoomCandidate],
    want_ac: bool,
    head_party: int | None = None,
) -> bool:
    """
    Can any set of rooms in this one AC class seat the party?

    The card needs this as a *question about the inventory*, not about the rows
    it happens to be showing: a guest who asked to see both classes and got one
    is owed the reason, and "the rows contain no Non-AC set" is not the reason —
    a broken draw would say the same thing. This asks the packer directly, so
    the sentence the guest reads describes the hotel rather than the renderer.
    """
    return min_cost_packing(party, _pool(candidates, want_ac), head_party) is not None


def ac_preference_from_state(value: str | None) -> str | None:
    """
    Normalise the several spellings of the AC preference that reach the card.

    `ac_preference` on state holds "AC" / "NON_AC" / "NO_PREFERENCE" (see
    `understand`), but the same intent also arrives via the S3 qualifier and
    via free text, so the values are mixed-case and occasionally "BOTH" or
    "SHOW_BOTH". Anything unrecognised is treated as unstated rather than
    guessed at, because guessing wrong here shows a guest a room class they
    explicitly did not ask for.

    The spellings are the shared reader's job (`app.utils.ac_text`), which is
    what keeps "NON_AC" here and "non - ac" in the parser from being two
    different answers. What stays local is the contract: an unstated
    preference IS a real case for this card, so anything the reader resolves to
    "they want to see both" comes back as None rather than as a third value.
    """
    resolved = ac_preference_from_text(value)
    return resolved if resolved in ("AC", "NON_AC") else None
