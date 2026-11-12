"""
Read a room-count-change out of a guest message, and refuse to guess.

Product, 2026-10-29: "3 rooms 3 days, 2 rooms last day", "1 room before checkin".
`app/services/stay_segments.py` does the arithmetic for that shape; this decides
whether the guest actually said it.

Why the model, and why only sometimes
-------------------------------------
The phrasings are Hinglish and unstructured — "pehle 2 din 3 kamre, aakhri raat
2", "one extra room the night before" — and a regex that reads them is a regex
that will also read "2 rooms for 3 people" as a split. `understand` already
stamps structured fields from the model (`task_classification`,
`room_requirements`), so this follows the house pattern.

What it does *not* follow is calling the model on every turn. There is a cheap
gate in front: `looks_like_a_split` returns False for almost every message, and
only a True message costs a call. Sending every message to a second model would
double the bot's latency and cost to answer a question nobody asked.

Failure is always "no split"
----------------------------
Every path here — flag off, gate miss, model down, model unsure, model returns
something that does not tile — returns a result whose `.status` is not "ok".
The caller then behaves exactly as it does today. There is no path where a bad
extraction produces a quote, which is the point: a wrong split is a confident,
plausible, wrong price, and that is the worst thing this bot can send.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from app.services import stay_segments

logger = logging.getLogger(__name__)

#: The deployment switch. Read through `os.getenv` rather than `get_settings()`
#: because `settings.py` is contended in this working tree (another session has
#: uncommitted changes there) and because `TRANSLATE_ROOM_CARDS` already set the
#: precedent for a feature flag that lives outside settings. Default off: this
#: changes what a guest is quoted and how many times they pay.
FLAG = "STAY_SEGMENTS_ENABLED"

#: A room count nobody means. Guards against a model reading "2" out of a date
#: or a phone number and quoting twenty rooms.
MAX_ROOMS_PER_NIGHT = 20

#: How many bookings one message may produce. the two product examples are two and
#: two. Three leaves room for the pre-check-in night plus a split; beyond that
#: the request has stopped being something a front desk wants, and the caller
#: escalates instead. (`stay_segments.MAX_SEGMENTS` is the outer ceiling.)
MAX_BOOKINGS = 3

#: Gate. A message has to mention a room quantity AND something that varies it
#: over time. "2 rooms for 3 nights" matches the first half and not the second,
#: so it does not cost a call. Kept deliberately loose — a false positive costs
#: one model call and an "invalid" result, a false negative costs the feature.
_SPLIT_CUE = (
    r"(?:last|final|first|pehli|pehla|aakhri|akhri|only|sirf)\s+"
    r"(?:night|day|raat|din)"
    r"|before\s+(?:the\s+)?check[\s-]?in"
    r"|check[\s-]?in\s+se\s+pehle"
    r"|night\s+before"
    r"|early\s+arrival"
)


def enabled() -> bool:
    """True when the feature is switched on for this deployment."""
    return os.getenv(FLAG, "0").strip().lower() not in ("0", "false", "off", "no", "")


@dataclass
class SplitResult:
    """What the extractor decided, and why.

    `status` is the whole contract:
      off            - flag down; caller behaves as before
      not_a_split    - the gate did not fire; no model call was made
      no_answer      - the model was unavailable or returned nothing usable
      not_confident  - the model saw something but would not commit
      invalid        - a declaration that does not tile the stay, or needs more
                       bookings than MAX_BOOKINGS
      ok             - segments is set and prices exactly the stay asked for

    `problems` is written for a staff ticket: if the guest has to be handed to a
    person, the person should see why the bot would not proceed.
    """

    status: str
    segments: list[dict] = field(default_factory=list)
    night_map: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    extended_check_in: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.segments)


def looks_like_a_split(message: str | None) -> bool:
    """Cheap gate: is this worth a model call at all?

    Requires both halves — a variation cue and a room quantity — because either
    alone is common. "2 rooms" is every group booking; "last night" is every
    review. Together they are rare.
    """
    import re

    text = (message or "").strip()
    if not text:
        return False
    if not re.search(_SPLIT_CUE, text, re.IGNORECASE):
        return False
    return bool(re.search(r"\b(?:\d+|one|two|three|four|five|six)\s*"
                          r"(?:room|rooms|kamra|kamre)\b", text, re.IGNORECASE))


_PROMPT = """You extract a room plan from one hotel guest message.

The stay being discussed:
  check_in:  {check_in}
  check_out: {check_out}
  rooms:     {rooms}

Return the nights on which the guest wants a DIFFERENT number of rooms than
that, as a JSON object:

{{"confident": true,
  "segments": [{{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD", "rooms": 2}}]}}

Rules:
- "to" is the CHECK-OUT date and is NOT a night. 1 Nov to 3 Nov is two nights.
- Segments must be adjacent and must cover the whole stay with no gaps and no
  overlaps.
- A room wanted for the night BEFORE check-in is a segment starting one day
  earlier.
- If the guest says the same room count throughout, return segments: [].
- If you are not sure what they mean, return {{"confident": false}} rather than
  a guess. A wrong plan is worse than no plan.

Guest message: {message}

JSON only."""


def _parse(date_text) -> date | None:
    try:
        return datetime.strptime(str(date_text)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def nights_from_segments(segments) -> tuple[dict[str, int], list[str]]:
    """
    Turn the model's segment list into a per-night room count.

    Overlaps are reported, not merged. `stay_segments.night_map_from_segments`
    lets the later segment win a contested night, which is fine for arithmetic
    and wrong for a declaration: two bookings claiming one night is the thing
    the hotel discovers at check-in, so it has to surface here.
    """
    night_map: dict[str, int] = {}
    problems: list[str] = []

    if not isinstance(segments, list):
        return {}, ["segments was not a list"]

    for raw in segments:
        if not isinstance(raw, dict):
            problems.append(f"segment is not an object: {raw!r}")
            continue

        start = _parse(raw.get("from"))
        end = _parse(raw.get("to"))
        rooms = raw.get("rooms")

        if not start or not end:
            problems.append(f"unreadable dates in segment: {raw!r}")
            continue
        if end <= start:
            problems.append(f"segment ends on or before it starts: {raw!r}")
            continue
        if not isinstance(rooms, int) or rooms < 1 or rooms > MAX_ROOMS_PER_NIGHT:
            problems.append(f"implausible room count {rooms!r} in {raw!r}")
            continue

        cursor = start
        while cursor < end:
            key = cursor.isoformat()
            if key in night_map:
                problems.append(f"two segments both claim {key}")
            night_map[key] = rooms
            cursor += timedelta(days=1)

    return night_map, problems


def validate(
    night_map: dict[str, int],
    check_in,
    check_out,
) -> tuple[list[str], list[dict], str | None]:
    """
    Is this declaration something the hotel could honour?

    Returns (problems, segments, extended_check_in). Empty problems means yes.

    Tiling is checked against the *guest's stated stay*, never against the range
    the segments happen to span — otherwise any segment list trivially tiles
    itself and the check means nothing. One extension is allowed, forward of
    check-in only: that is the product's "1 room before checkin", which is a real
    night the guest sleeps in the building and must be paid for.
    """
    base_in = stay_segments._as_date(check_in)
    base_out = stay_segments._as_date(check_out)
    if not base_in or not base_out or base_out <= base_in:
        return [f"the stay dates are not readable ({check_in!r} to {check_out!r})"], [], None

    required = set(stay_segments.enumerate_nights(check_in, check_out))
    declared = set(night_map)

    # The single permitted extension: one night before check-in. It has to be
    # resolved BEFORE the tiling check, because that check treats any night
    # outside the stated stay as unbookable — which is right for every other
    # kind of extra night and wrong for this one.
    extended = None
    extra = sorted(declared - required)
    if extra:
        allowed = {(base_in - timedelta(days=1)).isoformat()}
        outside = sorted(set(extra) - allowed)
        if outside:
            return ["rooms are requested outside the stay: " + ", ".join(outside)], [], None
        extended = extra[0]

    # Tiling is now checked against the stay as extended, never against the
    # range the segments happen to span — a segment list trivially tiles itself,
    # so that check would mean nothing.
    problems: list[str] = list(stay_segments.validate_night_map(
        night_map, extended or check_in, check_out
    ))
    if problems:
        return problems, [], None

    segments = stay_segments.compress(night_map)
    if len(segments) > MAX_BOOKINGS:
        return [f"this would need {len(segments)} separate bookings"], [], None

    return [], segments, extended


async def _default_generate(prompt: str) -> str:
    """One JSON call through the same circuit breaker `understand` uses."""
    from app.services.gemini_service import _generate_content_async, _get_model

    import google.generativeai as genai

    response = await _generate_content_async(
        _get_model(),
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
            max_output_tokens=256,
        ),
    )
    return getattr(response, "text", "") or ""


async def extract_stay_split(
    message: str | None,
    *,
    check_in,
    check_out,
    rooms: int,
    generate=None,
) -> SplitResult:
    """
    Read a room plan out of `message`. Never raises; never guesses.

    `generate` is injectable so the tests exercise every branch without a model
    call. Production passes nothing and gets `_default_generate`.
    """
    if not enabled():
        return SplitResult(status="off")

    if not looks_like_a_split(message):
        return SplitResult(status="not_a_split")

    if not (check_in and check_out) or not stay_segments.enumerate_nights(check_in, check_out):
        return SplitResult(status="invalid", problems=["the stay dates are not usable"])

    prompt = _PROMPT.format(
        check_in=check_in, check_out=check_out, rooms=rooms, message=(message or "")[:600]
    )

    try:
        raw = await (generate or _default_generate)(prompt)
    except Exception as exc:  # a model outage must not cost the turn
        logger.warning("STAY_SPLIT_EXTRACT_FAILED | %s", exc)
        return SplitResult(status="no_answer", problems=[f"model call failed: {exc}"])

    try:
        payload = json.loads((raw or "").strip() or "{}")
    except (ValueError, TypeError):
        logger.warning("STAY_SPLIT_EXTRACT_NOT_JSON | %r", (raw or "")[:200])
        return SplitResult(status="no_answer", problems=["the model did not return JSON"])

    if not isinstance(payload, dict):
        return SplitResult(status="no_answer", problems=["the model returned a non-object"])

    if not payload.get("confident", False):
        return SplitResult(status="not_confident", problems=["the model would not commit"])

    segments_raw = payload.get("segments") or []
    if not segments_raw:
        # A confident "the count never changes" is a real answer, and the
        # ordinary flow already handles it.
        return SplitResult(status="not_a_split")

    night_map, problems = nights_from_segments(segments_raw)
    if problems:
        return SplitResult(status="invalid", problems=problems, night_map=night_map)

    problems, segments, extended = validate(night_map, check_in, check_out)
    if problems:
        return SplitResult(status="invalid", problems=problems, night_map=night_map)

    return SplitResult(
        status="ok",
        segments=segments,
        night_map=night_map,
        extended_check_in=extended,
    )
