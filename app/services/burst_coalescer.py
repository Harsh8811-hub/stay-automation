"""
Merge a guest's rapid-fire WhatsApp messages into a single turn.

Guests do not type one message; they type one *thought* in pieces. Measured on
real conversations (2026-10-18, hotel 111111): "Jo client ko deti hu" /
"Aapko" / "From Bengali" arrived 9 seconds apart and produced three replies,
all asking for the same dates. "Check in 28 october" / "Check out 31 october"
produced a correct room card and, in the same second, a text asking the guest
for their check-in date. And "Non AC" / "Room" produced "we have no Non-AC
rooms" immediately followed by "here are our available rooms".

The pieces are also harder for the extractor than the whole: "28 october"
alone has no direction, so it was read as both check-in and check-out.

This module holds the pieces briefly and lets the newest message run the turn
over all of them. The mechanism is a per-conversation buffer in MongoDB, not
an in-process dict, because consecutive WhatsApp messages are forwarded as
separate HTTP requests and Cloud Run may land them on different instances.

Contract: every caller is a winner or a loser. A loser must send nothing —
`response: []`, the same "say nothing" signal the takeover pause path uses.
A winner runs one turn over `Burst.text`.

Failure policy: if the buffer cannot be read or written, the caller is always
treated as a winner with its own message. Deferring is an improvement, not a
correctness requirement, so a Mongo blip must never lose a guest's message.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db.mongodb import get_database

logger = logging.getLogger(__name__)

COLLECTION = "message_bursts"

#: Messages older than this relative to the newest one are never merged in.
#: They can only be there because a previous winner crashed before clearing
#: the buffer; re-answering them would feed stale text to the extractor.
#:
#: Also the ceiling on how long a burst may be held: a guest who keeps typing
#: must still get an answer. Each fresh message resets the debounce, so this is
#: what stops a long stream from starving the reply forever.
MAX_BURST_SPAN_SECONDS = 30.0

#: A message this short is mid-thought, not a finished one. Measured against
#: real bursts: "Non AC" (6), "Room" (4), "28 october" (10), "Check in" (8),
#: "10 person with kids" (19), "Check out 31 october" (20) are all fragments;
#: "I will confirm in the evening today" (34, 7 words) is not.
FRAGMENT_MAX_CHARS = 30
FRAGMENT_MAX_WORDS = 5

_indexes_ready = False


@dataclass(frozen=True)
class Burst:
    """The outcome of joining a burst.

    ``is_winner`` False means another message in this burst superseded us and
    the caller must not run the graph. ``text`` is only meaningful when True.
    """

    is_winner: bool
    text: str


async def _ensure_indexes(collection: Any) -> None:
    """TTL index so a crashed winner's buffer cannot outlive the burst."""
    global _indexes_ready
    if _indexes_ready:
        return
    try:
        await collection.create_index("updated_at", expireAfterSeconds=120)
    except Exception:
        # An in-memory fake, or a database that denies index creation. Neither
        # is a reason to stop merging.
        pass
    _indexes_ready = True


def looks_like_a_fragment(message: str) -> bool:
    """Is this message a piece of a thought rather than a whole one?

    Deliberately crude and deterministic: a language model call here would add
    cost and latency to every single turn, and the decision only chooses
    between two waits. Short messages are the ones guests split up.
    """
    text = (message or "").strip()
    if not text:
        return False
    return len(text) <= FRAGMENT_MAX_CHARS or len(text.split()) <= FRAGMENT_MAX_WORDS


def _as_utc(value: Any) -> datetime | None:
    """Motor returns naive UTC; the cutoffs here are aware. Normalise."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _merge(messages: list[dict], fallback: str, now: datetime) -> str:
    cutoff = now - timedelta(seconds=MAX_BURST_SPAN_SECONDS)
    parts: list[str] = []
    for message in messages:
        at = _as_utc(message.get("at"))
        text = (message.get("text") or "").strip()
        if not text:
            continue
        if at is not None and at < cutoff:
            continue
        parts.append(text)
    # One space, not a newline or a separator: these are continuations of one
    # sentence ("28 october" + "Check in"), and the extractor reads prose.
    return " ".join(parts) or fallback


async def join_burst(
    *,
    hotel_id: str | int | None,
    session_key: str,
    message: str,
    window_seconds: float,
    collection: Any = None,
) -> Burst:
    """Buffer this message, wait out the window, and elect one winner.

    Safe to call concurrently for the same ``session_key``: exactly one caller
    can be the winner, because the winner is whoever still owns the buffer's
    ``token`` when it wakes up.
    """
    text = (message or "").strip()
    window = max(0.0, float(window_seconds or 0.0))
    if not text or not session_key or window == 0.0:
        return Burst(True, text)

    if collection is None:
        collection = get_database()[COLLECTION]
    await _ensure_indexes(collection)

    key = f"{hotel_id}:{session_key}"
    token = uuid.uuid4().hex
    now = datetime.now(timezone.utc)

    try:
        buffered = await collection.find_one_and_update(
            {"_id": key},
            {
                "$push": {"messages": {"text": text, "token": token, "at": now}},
                "$set": {"token": token, "updated_at": now},
            },
            upsert=True,
            return_document=True,  # ReturnDocument.AFTER
        )
    except Exception:
        logger.warning("BURST_BUFFER_FAILED | key=%s — answering immediately", key, exc_info=True)
        return Burst(True, text)

    # Do not hold a burst longer than its own span. A guest typing steadily
    # resets the window on every message, so without this the reply could be
    # deferred indefinitely; here it is capped at MAX_BURST_SPAN_SECONDS from
    # the first message in the burst.
    oldest = min(
        (t for t in (_as_utc(m.get("at")) for m in (buffered or {}).get("messages") or []) if t),
        default=now,
    )
    remaining = MAX_BURST_SPAN_SECONDS - (now - oldest).total_seconds()
    await asyncio.sleep(max(0.0, min(window, remaining)))

    try:
        document = await collection.find_one({"_id": key}) or {}
    except Exception:
        logger.warning("BURST_READ_FAILED | key=%s — answering immediately", key, exc_info=True)
        return Burst(True, text)

    if document.get("token") != token:
        # A newer message took over. It will answer for all of us.
        logger.info("BURST_SUPERSEDED | key=%s", key)
        return Burst(False, text)

    merged = _merge(document.get("messages") or [], text, now)
    try:
        # Scoped to our token: if a new message arrived between the read and
        # here, its buffer survives and it will answer on its own.
        await collection.delete_one({"_id": key, "token": token})
    except Exception:
        logger.warning("BURST_CLEAR_FAILED | key=%s", key, exc_info=True)

    if merged != text:
        logger.info("BURST_MERGED | key=%s parts=%s chars=%s", key, len(document.get("messages") or []), len(merged))
    return Burst(True, merged)
