"""
Hybrid Room Alias Resolution Engine.

Performance-first pipeline (never calls Gemini when deterministic is confident):

  1. Exact match (name / room_type)
  2. Hotel-specific alias map (MongoDB / HotelConfig)
  3. Automatic normalization
  4. Fuzzy match (RapidFuzz)
  5. Gemini fallback — once, only if still ambiguous
  6. Guest clarification — never guess

This is the single source of truth for room-name resolution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

logger = logging.getLogger(__name__)

DEFAULT_FUZZY_THRESHOLD = 90.0

# Relative selection words — not room names; skip name resolution
_RELATIVE_PREFERENCES = frozenset({
    "cheapest", "best", "most affordable", "most expensive",
    "lowest", "highest", "cheaper", "expensive", "affordable",
})

_STOP_TOKENS = frozenset({
    "room", "rooms", "type", "types", "suite", "suites",
    "accommodation", "accommodations", "the", "a", "an",
    "category", "class",
})

# Built-in abbreviation expansions (hotel aliases always win over these)
_ABBREVIATIONS = {
    "dlx": "deluxe",
    "delux": "deluxe",
    "std": "standard",
    "exec": "executive",
    "exe": "executive",
    "prem": "premium",
    "premuim": "premium",
    "fam": "family",
    "honeymoon": "honeymoon",
    "king": "king",
    "queen": "queen",
    "twin": "twin",
    "dbl": "double",
    "sgl": "single",
}

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RoomRef:
    """Normalized catalog entry used internally by the resolver."""
    room_id: str
    name: str
    room_type: str = ""
    description: str = ""
    raw: Any = None  # original object (RoomType / dict / etc.)


@dataclass
class RoomResolveResult:
    """Outcome of resolving a guest room reference."""
    status: Literal["resolved", "ambiguous", "unresolved"]
    matched_rooms: list[Any] = field(default_factory=list)
    canonical_name: str | None = None
    confidence: float = 0.0
    method: str = "none"  # exact|alias|normalized|fuzzy|gemini|none

    @property
    def room(self) -> Any | None:
        return self.matched_rooms[0] if len(self.matched_rooms) == 1 else None

    @property
    def candidate_names(self) -> list[str]:
        names: list[str] = []
        for r in self.matched_rooms:
            name = _room_attr(r, "name") or _room_attr(r, "room_name")
            if name and name not in names:
                names.append(name)
        return names

    def clarification_message(self) -> str:
        names = self.candidate_names
        if len(names) >= 2:
            shown = names[:4]
            if len(shown) == 2:
                return f"Did you mean **{shown[0]}** or **{shown[1]}**?"
            joined = ", ".join(f"**{n}**" for n in shown[:-1])
            return f"Did you mean {joined}, or **{shown[-1]}**?"
        if len(names) == 1:
            return f"Did you mean **{names[0]}**?"
        return "Which room were you referring to? Please pick from the available options."


def normalize_room_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces, drop stop words, expand abbreviations."""
    if not text:
        return ""
    s = text.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    tokens = []
    for tok in s.split():
        tok = _ABBREVIATIONS.get(tok, tok)
        if tok in _STOP_TOKENS:
            continue
        tokens.append(tok)
    return " ".join(tokens)


def is_relative_preference(query: str) -> bool:
    """True for 'cheapest' / 'best' style preferences (not room names)."""
    n = normalize_room_text(query)
    return n in _RELATIVE_PREFERENCES or query.strip().lower() in _RELATIVE_PREFERENCES


#: A trailing clause that describes an ATTRIBUTE of a room instead of naming it
#: — "delux family with extra bed?", "deluxe family room extra bed".
#:
#: The guest is naming one room and asking a question about it. Handing the
#: whole string to the matcher is what made the live case fail twice over: the
#: words as the guest typed them ("delux family with extra bed?") fell through
#: every step to `unresolved`, and the same words without the "?" landed on a
#: 90.0 fuzzy near-tie that returned three rooms led by DELUXE ROOM KING BED,
#: with DELUXE FAMILY ROOM — the room being asked about — second. The two
#: answers differ only because the "?" pushed a raw-string WRatio a hair under
#: the threshold, so the guest's own punctuation decided which wrong answer
#: they got.
#:
#: Only a TRAILING clause, and only one naming an extra bed. "extra bed" is the
#: attribute no room in this catalogue carries in its name, which is what makes
#: it safe to drop; a wider tail would eventually eat "STANDARD NON AC ROOM".
_EXTRA_BED_CLAUSE = re.compile(
    r"[\s,+\-/]*(?:(?:with|having|need(?:s|ed)?|want(?:s|ed)?|add|plus|and)\s+)*"
    r"(?:an?\s+)?(?:\d{1,2}\s+)?extra\s+beds?\b.*$",
    re.IGNORECASE,
)


def room_name_core(query: str) -> str:
    """
    `query` with a trailing "with extra bed"-style clause removed.

    Empty when that clause was all there was — "extra bed for 3" names no room,
    so a caller must read "" as "this module has no opinion", never as a query
    to match. Unchanged when the clause is not there at all.
    """
    if not isinstance(query, str) or not query.strip():
        return ""
    return _EXTRA_BED_CLAUSE.sub("", query).strip()


def _room_attr(obj: Any, key: str, default: str = "") -> str:
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    return str(val) if val is not None else default


def _to_refs(rooms: Sequence[Any]) -> list[RoomRef]:
    refs: list[RoomRef] = []
    seen: set[str] = set()
    for r in rooms:
        room_id = _room_attr(r, "room_id") or _room_attr(r, "id")
        name = _room_attr(r, "name") or _room_attr(r, "room_name")
        if not name:
            continue
        key = room_id or name.lower()
        if key in seen:
            continue
        seen.add(key)
        refs.append(RoomRef(
            room_id=room_id or name,
            name=name,
            room_type=_room_attr(r, "room_type"),
            description=_room_attr(r, "description"),
            raw=r,
        ))
    return refs


def _alias_lookup(query: str, aliases: dict[str, str] | None) -> str | None:
    """Hotel aliases override everything. Keys matched case-insensitively / normalized."""
    if not aliases or not query:
        return None
    q_raw = query.strip().lower()
    q_norm = normalize_room_text(query)

    # Build lookup once
    for key, canonical in aliases.items():
        if not key or not canonical:
            continue
        k_raw = str(key).strip().lower()
        k_norm = normalize_room_text(str(key))
        if q_raw == k_raw or (q_norm and q_norm == k_norm):
            return str(canonical).strip()
    return None


def _exact_matches(query: str, refs: list[RoomRef]) -> list[RoomRef]:
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for ref in refs:
        if ref.name.strip().lower() == q or (ref.room_type and ref.room_type.strip().lower() == q):
            hits.append(ref)
    return hits


def _normalized_matches(query: str, refs: list[RoomRef]) -> list[RoomRef]:
    qn = normalize_room_text(query)
    if not qn:
        return []
    hits = []
    for ref in refs:
        name_n = normalize_room_text(ref.name)
        type_n = normalize_room_text(ref.room_type)
        # Strict normalized equality only — loose containment belongs to fuzzy
        if qn == name_n or (type_n and qn == type_n):
            hits.append(ref)
    return hits


def _fuzzy_matches(
    query: str,
    refs: list[RoomRef],
    threshold: float,
) -> tuple[list[RoomRef], float]:
    fuzz = None
    try:
        from rapidfuzz import fuzz
    except ImportError:
        import difflib

    q_raw = query.strip().lower()
    q_norm = normalize_room_text(query)
    if not q_raw and not q_norm:
        return [], 0.0

    q_tokens = (q_norm or q_raw).split()
    multi_token = len(q_tokens) >= 2

    scored: list[tuple[float, RoomRef]] = []
    for ref in refs:
        targets = [
            ref.name.lower(),
            normalize_room_text(ref.name),
            ref.room_type.lower() if ref.room_type else "",
            normalize_room_text(ref.room_type),
        ]
        best = 0.0
        for t in targets:
            if not t:
                continue
            if fuzz is not None:
                wr = float(fuzz.WRatio(q_raw, t))
                sort_r = float(fuzz.token_sort_ratio(q_norm or q_raw, t))
                if multi_token:
                    best = max(best, wr, sort_r, (wr + sort_r) / 2.0)
                else:
                    set_r = float(fuzz.token_set_ratio(q_norm or q_raw, t))
                    best = max(best, wr, sort_r, set_r)
            else:
                import difflib
                # difflib fallback (0 to 100)
                seq_r = difflib.SequenceMatcher(None, q_raw, t).ratio() * 100.0
                norm_r = difflib.SequenceMatcher(None, q_norm or q_raw, t).ratio() * 100.0
                # Token overlap bonus
                t_tokens = t.split()
                overlap = set(q_tokens).intersection(set(t_tokens))
                token_r = (len(overlap) / max(len(q_tokens), 1)) * 85.0 if overlap else 0.0
                best = max(best, seq_r, norm_r, token_r)

        eff_threshold = threshold if fuzz is not None else max(threshold - 15.0, 70.0)
        if best >= eff_threshold:
            scored.append((best, ref))

    if not scored:
        return [], 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][0]
    # Keep near-ties within 3 points of the top score
    hits = [ref for score, ref in scored if score >= top - 3.0]
    return hits, top


def _result_from_refs(
    refs: list[RoomRef],
    *,
    method: str,
    confidence: float,
    prefer_single: bool,
) -> RoomResolveResult:
    rooms = [r.raw for r in refs]
    if not refs:
        return RoomResolveResult(status="unresolved", method=method, confidence=0.0)

    if prefer_single:
        if len(refs) == 1:
            return RoomResolveResult(
                status="resolved",
                matched_rooms=rooms,
                canonical_name=refs[0].name,
                confidence=confidence,
                method=method,
            )
        return RoomResolveResult(
            status="ambiguous",
            matched_rooms=rooms,
            confidence=confidence,
            method=method,
        )

    # Filter mode — any non-empty set is a successful match set
    return RoomResolveResult(
        status="resolved",
        matched_rooms=rooms,
        canonical_name=refs[0].name if len(refs) == 1 else None,
        confidence=confidence,
        method=method,
    )


def _match_by_name(
    query: str,
    refs: list[RoomRef],
    alias_target: str | None,
    fuzzy_threshold: float,
    prefer_single: bool,
) -> RoomResolveResult | None:
    """
    Steps 3–4 of the deterministic pipeline — normalized, then fuzzy — or None
    when neither matched.

    Extracted so the same two steps can run on a room's NAME and, when that
    fails, on the query as the guest typed it. `alias_target` is passed rather
    than looked up because the alias step belongs to the caller: it has already
    had its turn on the full query by the time this runs on the core.
    """
    search = alias_target or query

    hits = _normalized_matches(search, refs)
    if not hits and alias_target:
        hits = _normalized_matches(query, refs)
    if hits:
        # Deduplicate by room_id
        uniq: dict[str, RoomRef] = {}
        for h in hits:
            uniq[h.room_id] = h
        return _result_from_refs(
            list(uniq.values()),
            method="alias" if alias_target else "normalized",
            confidence=95.0,
            prefer_single=prefer_single,
        )

    hits, score = _fuzzy_matches(search, refs, fuzzy_threshold)
    if not hits and alias_target:
        hits, score = _fuzzy_matches(query, refs, fuzzy_threshold)
    if hits:
        return _result_from_refs(
            hits, method="fuzzy", confidence=score, prefer_single=prefer_single
        )

    return None


def resolve_room_reference_sync(
    query: str,
    rooms: Sequence[Any],
    *,
    aliases: dict[str, str] | None = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    prefer_single: bool = True,
) -> RoomResolveResult:
    """
    Deterministic resolution only (steps 1–4). No Gemini.

    prefer_single=True  → multiple hits = ambiguous (selection flows)
    prefer_single=False → multiple hits = resolved set (search filter)
    """
    if not query or not str(query).strip():
        return RoomResolveResult(status="unresolved", method="none")

    if is_relative_preference(query):
        return RoomResolveResult(status="unresolved", method="relative")

    refs = _to_refs(rooms)
    if not refs:
        return RoomResolveResult(status="unresolved", method="none")

    # Step 2 first on the raw query — hotel aliases override automatic rules
    alias_target = _alias_lookup(query, aliases)
    search_query = alias_target or query

    # Step 1 — Exact
    hits = _exact_matches(search_query, refs)
    if hits:
        method = "alias" if alias_target else "exact"
        return _result_from_refs(hits, method=method, confidence=100.0, prefer_single=prefer_single)

    # Also try exact on original query if alias was used but missed
    if alias_target:
        hits = _exact_matches(query, refs)
        if hits:
            return _result_from_refs(hits, method="exact", confidence=100.0, prefer_single=prefer_single)

    # Step 2b — read the room's NAME out of a request that also asks about a bed.
    #
    # "delux family with extra bed?" names DELUXE FAMILY ROOM and asks a question
    # about a bed. Every step above and below this one matches on the whole
    # string, which is why the name never surfaced. Unreachable when the hotel
    # really does sell a room by the full name — step 1's exact match would have
    # returned before this.
    core = room_name_core(search_query)
    if core and core != search_query.strip():
        matched = _match_by_name(
            core,
            refs,
            _alias_lookup(core, aliases),
            fuzzy_threshold,
            prefer_single,
        )
        if matched is not None:
            logger.info(
                "ROOM_RESOLVE | extra-bed clause read as an attribute, not a name"
                " | %r -> %r | method=%s",
                search_query,
                core,
                matched.method,
            )
            return matched

    # Steps 3–4 — Normalized, then fuzzy, on the query as the guest wrote it.
    matched = _match_by_name(
        query, refs, alias_target, fuzzy_threshold, prefer_single
    )
    if matched is not None:
        return matched

    return RoomResolveResult(status="unresolved", method="none", confidence=0.0)


async def resolve_room_reference(
    query: str,
    rooms: Sequence[Any],
    *,
    aliases: dict[str, str] | None = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    guest_message: str | None = None,
    allow_gemini: bool = True,
    prefer_single: bool = True,
) -> RoomResolveResult:
    """
    Full pipeline including optional Gemini fallback (step 5).

    Gemini is called at most once, only when steps 1–4 fail or stay ambiguous,
    and only with room names + guest message (no booking/PII/config).
    """
    result = resolve_room_reference_sync(
        query,
        rooms,
        aliases=aliases,
        fuzzy_threshold=fuzzy_threshold,
        prefer_single=prefer_single,
    )

    if result.status == "resolved":
        return result

    if not allow_gemini:
        return result

    refs = _to_refs(rooms)
    if not refs:
        return result

    # Step 5 — Gemini (only if still unresolved or ambiguous)
    try:
        from app.services.gemini_service import resolve_room_name_with_gemini

        names = [r.name for r in refs]
        desc_map = {r.name: r.description for r in refs if r.description}
        message = (guest_message or query or "").strip()
        if not message:
            return result

        gemini_name = await resolve_room_name_with_gemini(
            guest_message=message,
            room_names=names,
            descriptions=desc_map or None,
        )
        if not gemini_name or gemini_name.upper() == "AMBIGUOUS":
            # Keep prior ambiguous candidates if we had them; else unresolved
            if result.status == "ambiguous":
                return result
            return RoomResolveResult(
                status="ambiguous" if len(refs) > 1 else "unresolved",
                matched_rooms=[r.raw for r in refs[:4]] if len(refs) > 1 else [],
                method="gemini",
                confidence=0.0,
            )

        # Map Gemini answer back to catalog — never invent
        mapped = _exact_matches(gemini_name, refs) or _normalized_matches(gemini_name, refs)
        if not mapped:
            # Fuzzy map Gemini output to catalog
            mapped, score = _fuzzy_matches(gemini_name, refs, threshold=85.0)
            if mapped and len(mapped) == 1:
                return RoomResolveResult(
                    status="resolved",
                    matched_rooms=[mapped[0].raw],
                    canonical_name=mapped[0].name,
                    confidence=score,
                    method="gemini",
                )
            return RoomResolveResult(
                status="ambiguous",
                matched_rooms=[r.raw for r in (mapped or refs[:3])],
                method="gemini",
                confidence=0.0,
            )

        uniq: dict[str, RoomRef] = {r.room_id: r for r in mapped}
        mapped_list = list(uniq.values())
        if len(mapped_list) == 1:
            return RoomResolveResult(
                status="resolved",
                matched_rooms=[mapped_list[0].raw],
                canonical_name=mapped_list[0].name,
                confidence=90.0,
                method="gemini",
            )
        return RoomResolveResult(
            status="ambiguous",
            matched_rooms=[r.raw for r in mapped_list],
            method="gemini",
            confidence=0.0,
        )
    except Exception as e:
        logger.warning(f"Gemini room resolve skipped: {e}")
        return result


def filter_rooms_by_reference(
    query: str,
    rooms: Sequence[Any],
    *,
    aliases: dict[str, str] | None = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[Any]:
    """
    Deterministic multi-room filter for search / pricing.
    Never calls Gemini. Empty list = no confident matches (caller may fall back).
    """
    result = resolve_room_reference_sync(
        query,
        rooms,
        aliases=aliases,
        fuzzy_threshold=fuzzy_threshold,
        prefer_single=False,
    )
    if result.status in ("resolved", "ambiguous") and result.matched_rooms:
        return result.matched_rooms
    return []


# ── Bed terms ────────────────────────────────────────────────────────────────
#
# "double bed" names a BED, not a room. Nothing in the resolution pipeline above
# looked at the bed, so on 2026-09-21 a guest asking for a non-AC double bed for
# three people matched nothing at every stage and was told "I'm sorry, we don't
# have rooms available that match your requirements" — while that same message
# listed "STANDARD NON AC ROOM", whose own description reads "Normal non ac room
# features double bed room with attach bathroom only". The room was in stock at
# base 2 / max 3, the hotel's own prose said "double bed", and nothing read it.
#
# Two tiers, and the ORDER is the whole design:
#
#   1. The hotel's own words — name, description, aliases. Literal, with no
#      synonym expansion, so a hit here is evidence rather than inference.
#   2. Only when tier 1 finds nothing, the guest's word expanded through
#      `_BED_SYNONYMS` and matched against the `bed_config` types.
#
# Tier 2 must never run once tier 1 has answered. At Sample Hotel all six rooms
# carry a `King Bed`, so letting "double" reach "king" in tier 1 would make
# "double bed" match the entire catalogue — including the triple bed that guest
# had just refused two messages earlier. Tier 1 finds the two rooms whose prose
# says "double bed" and stops.

#: A guest's bed word → the words the hotel may have used for the same bed.
#:
#: `double` is the entry that does the work. No hotel in this catalogue has a
#: bed type called "Double Bed" — every `bed_config.type` is King Bed, Queen Bed
#: or Single Bed — so without this expansion a guest asking for a double bed can
#: never reach the king-bedded room they mean. Add nothing here that a hotel in
#: this catalogue has not actually written.
_BED_SYNONYMS: dict[str, frozenset[str]] = {
    "double": frozenset({"double", "king", "queen"}),
    "king": frozenset({"king"}),
    "queen": frozenset({"queen"}),
    "single": frozenset({"single"}),
    "twin": frozenset({"twin", "single"}),
    "three": frozenset({"three", "triple", "tripple"}),
    "triple": frozenset({"three", "triple", "tripple"}),
    "tripple": frozenset({"three", "triple", "tripple"}),
    "3": frozenset({"three", "triple", "tripple"}),
    "four": frozenset({"four", "4"}),
    "4": frozenset({"four", "4"}),
}

#: A bed word, but only where it is bound to "bed" or "room" — and a bare number
#: only where it is bound to "bed". That last restriction is load-bearing: "3
#: rooms" is a room COUNT, and reading it as a triple bed would turn an ordinary
#: booking into a bed request and quote the wrong room. The catalogue's own
#: spelling is "TRIPPLE", which is why both spellings are listed.
_BED_TERM = re.compile(
    r"\b(double|single|twin|king|queen|triple|tripple|three|four)\s*(?:bed|room)s?\b"
    r"|\b(\d{1,2})\s*beds?\b",
    re.IGNORECASE,
)


def bed_terms(text: Any) -> set[str]:
    """
    The bed words a phrase names — `"non ac double bed"` → `{"double"}`.

    Empty for anything that names no bed at all: `"3 rooms"`, `"deluxe room"`,
    `"2 adults"`. Callers use that emptiness to decide whether this module has an
    opinion, so it must not guess.
    """
    if not isinstance(text, str) or not text:
        return set()
    found: set[str] = set()
    for match in _BED_TERM.finditer(text):
        word = (match.group(1) or match.group(2) or "").lower()
        if word:
            found.add(word)
    return found


def _bed_field(room: Any, name: str, default: Any) -> Any:
    """
    One field off a room that may be a model or a plain mapping.

    Deliberately NOT the module's own `_room_attr`, which coerces to `str` — that
    is right for a room name and wrong for `aliases` and `beds`, which are lists
    this matcher has to walk. Two helpers, two contracts, and neither shadows the
    other: defining a second `_room_attr` here silently replaced the resolver's
    own and broke `_to_refs` with a missing-argument `TypeError`.
    """
    if isinstance(room, dict):
        return room.get(name, default)
    return getattr(room, name, default)


def _word_set(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def rooms_matching_bed_terms(query: Any, rooms: Sequence[Any] | None) -> list[Any]:
    """
    Catalogue rooms whose own record says they have the bed the guest asked for.

    Strictly additive: it answers only "does any room here evidence this bed",
    and returns `[]` when the guest named no bed. A caller should resolve the
    room by NAME first and reach for this only when that found nothing, because a
    name match is the stronger signal — this one is a fallback, not a peer.

    Order is the catalogue's own, so a caller may take `[0]` and get a stable
    answer. Ties are not broken by price here; pricing is the allocator's job.

    Multi-term queries are OR-ed ("king and queen bed" matches a room with
    either), because a guest listing beds is describing what would do, not
    demanding both in one room.
    """
    terms = bed_terms(query)
    if not terms:
        return []

    room_list = [r for r in (rooms or []) if r is not None]

    # Tier 1 — the hotel's own words, literally.
    literal: list[Any] = []
    for room in room_list:
        aliases = _bed_field(room, "aliases", None) or []
        prose = _word_set(
            f"{_bed_field(room, 'name', '')} "
            f"{_bed_field(room, 'description', '')} "
            f"{' '.join(str(a) for a in aliases)}"
        )
        if terms & prose:
            literal.append(room)
    if literal:
        return literal

    # Tier 2 — inferred, and only because tier 1 found nothing.
    wanted: set[str] = set()
    for term in terms:
        wanted |= set(_BED_SYNONYMS.get(term, frozenset({term})))

    inferred: list[Any] = []
    for room in room_list:
        beds = _bed_field(room, "beds", None) or []
        bed_types = _word_set(" ".join(
            str(b.get("type") or "") for b in beds if isinstance(b, dict)
        ))
        if wanted & (bed_types | _word_set(_bed_field(room, "name", ""))):
            inferred.append(room)

    # Nearest-match: if triple room requested and none exists in catalog, match double/king with extra bed
    if not literal and not inferred and (terms & {"three", "triple", "tripple", "3"}):
        for room in room_list:
            occ = getattr(room, "occupancy", None)
            max_a = getattr(occ, "max_adults", 2) if occ else 2
            extra_allowed = getattr(occ, "extra_bed_allowed", True) if occ else True
            if max_a >= 3 or extra_allowed:
                inferred.append(room)

    return inferred


# ── Nearest miss ─────────────────────────────────────────────────────────────

#: Guest words for a bed → the label the guest reads back. Only the words
#: `bed_terms` can return need an entry; anything missing falls back to itself.
_BED_LABELS: dict[str, str] = {
    "double": "double bed",
    "single": "single bed",
    "twin": "twin bed",
    "king": "king bed",
    "queen": "queen bed",
    "triple": "triple bed",
    "tripple": "triple bed",
    "three": "three-bed",
    "four": "four-bed",
}


def _ac_class_label(room: Any) -> str:
    """"Non-AC" or "AC" for one room, using the shared feature matcher."""
    from app.services.allocation_engine import room_matches_feature
    return "Non-AC" if room_matches_feature(room, "Non-AC") else "AC"


#: Longest word first, so "tripple bed" reads as a triple and not as a single.
_BED_WORD_ORDER = (
    "king", "queen", "double", "twin", "tripple", "triple", "four", "three", "single",
)


def room_bed_label(room: Any) -> str | None:
    """The bed one room offers, in the guest's own vocabulary — or None.

    The NAME is read before the description, because the name is what the guest
    is choosing from: "DELUXE ROOM FOUR BED" is a four-bed room even though its
    description also mentions the king bed in it. Then `bed_config`, for a room
    that names no bed at all.
    """
    name_words = _word_set(_bed_field(room, "name", ""))
    desc_words = _word_set(_bed_field(room, "description", ""))
    for words in (name_words, desc_words):
        for word in _BED_WORD_ORDER:
            if word in words:
                return _BED_LABELS.get(word, word)
    for bed in (_bed_field(room, "beds", None) or []):
        if isinstance(bed, dict) and bed.get("type"):
            return str(bed["type"]).lower()
    return None


def nearest_room_alternatives(
    query: Any,
    rooms: Sequence[Any] | None,
    feature_preferences: Sequence[str] | None = None,
) -> dict:
    """What the catalogue has that sits closest to a request it cannot satisfy.

    A guest asks for something the hotel does not sell — "non-AC king bed" at a
    hotel whose king rooms are all AC — and the answer they got was a flat "we
    don't have rooms available that match your requirements" followed by the
    whole catalogue, which tells them nothing about *which* part of their
    request was the problem. (Sample Hotel, 2026-09-21.)

    So this splits the request into the two independent things a guest can name
    about a room — the bed and the AC class — and reports which rooms satisfy
    one of them. It never claims a room satisfies both, because by the time it
    is called nothing does.

    Returns a dict with the names the guest's words map to, and the rooms that
    meet one half of the request:

        {"bed": "king bed" | None, "ac_class": "Non-AC" | None,
         "want_bed": "king" | None, "want_ac": "NON_AC" | None,
         "same_bed": [...], "same_class": [...]}

    The caller decides the wording; this returns structure, not prose.
    """
    from app.utils.ac_text import ac_preference_from_text

    room_list = [r for r in (rooms or []) if r is not None]
    if not room_list:
        return {}

    terms = bed_terms(query)
    want_bed = next(iter(sorted(terms)), None)
    bed_label = _BED_LABELS.get(want_bed or "", want_bed) if want_bed else None

    # The class can arrive two ways: spelled in the guest's words ("non ac king
    # bed"), or already extracted into the feature list. The words win, because
    # they are what the guest is looking at on their own screen.
    want_ac = ac_preference_from_text(query) if isinstance(query, str) else None
    if want_ac in (None, "NO_PREFERENCE"):
        for pref in (feature_preferences or []):
            extracted = ac_preference_from_text(str(pref))
            if extracted in ("AC", "NON_AC"):
                want_ac = extracted
                break
    if want_ac == "NO_PREFERENCE":
        want_ac = None

    if not want_bed and not want_ac:
        return {}

    # A gap needs BOTH halves to have been asked for. With only one named, any
    # room meeting it IS the answer, not a near miss — and saying otherwise is
    # how the caller would be told "we don't have a double bed room" while
    # standing in front of one the guest never asked to be AC.
    if not (want_bed and want_ac):
        return {}

    same_bed = rooms_matching_bed_terms(query, room_list) if want_bed else []
    same_class = (
        [r for r in room_list if _ac_class_label(r) == ("Non-AC" if want_ac == "NON_AC" else "AC")]
        if want_ac else []
    )

    # A room that meets both halves is not a near miss, it is a match — and the
    # callers only reach here when nothing matched. Dropping them keeps a stale
    # or over-broad tier-1 prose hit from being offered as "close" when it is
    # actually the answer.
    both = {id(r) for r in same_bed} & {id(r) for r in same_class}
    if both:
        same_bed = [r for r in same_bed if id(r) not in both]
        same_class = [r for r in same_class if id(r) not in both]

    return {
        "bed": bed_label,
        "ac_class": None if not want_ac else ("Non-AC" if want_ac == "NON_AC" else "AC"),
        "want_bed": want_bed,
        "want_ac": want_ac,
        "same_bed": same_bed,
        "same_class": same_class,
    }


def aliases_from_hotel_config(hotel_config: dict | Any | None) -> dict[str, str]:
    """Extract room_aliases from a HotelConfig model or serialized dict."""
    if not hotel_config:
        return {}
    if isinstance(hotel_config, dict):
        aliases = hotel_config.get("room_aliases") or {}
        if not aliases:
            biz = hotel_config.get("business_config") or {}
            if isinstance(biz, dict):
                aliases = biz.get("room_aliases") or {}
        return {str(k): str(v) for k, v in aliases.items()} if isinstance(aliases, dict) else {}
    aliases = getattr(hotel_config, "room_aliases", None) or {}
    return {str(k): str(v) for k, v in aliases.items()} if isinstance(aliases, dict) else {}
