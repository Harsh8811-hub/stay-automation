"""
Inbound webhook idempotency (B3a).

Why this exists
---------------
Meta retries a webhook when it does not get a 200 back fast enough. One turn of this bot can
take 52 seconds (`enrich_options_with_stay_totals` prices options sequentially,
room_service.py:120-151), so the retry is routine, not exceptional — and `whatsapp.py` had no
guard against it. The same `wamid` was processed again, the guest got the same combo and the
same photos 2-4× at near-identical timestamps, and the loop never advanced.

That is the shared root of two of the three reported issues. Issue 3 (the same reply up to 19×)
is the visible one; Issue 1 (the combo and photos re-sent every turn) is the same mechanism
wearing the guest's patience thinner.

The design, and why it is claim-first
-------------------------------------
The duplicate arrives *while the first attempt is still running*. That rules out the obvious
"mark it processed when we finish" — by then the second copy has already been let through.
So the claim happens before the work: `insert_one` with the wamid as `_id`, and whoever loses
the race gets a `DuplicateKeyError` and stops.

The cost of that choice, stated plainly: **a hard crash mid-turn loses that guest message.** We
claimed it, so the retry is refused, and nothing replays it. That is a real regression in
durability traded against a certain regression in correctness. I took it because the duplicate
is happening thousands of times and the crash is not, but it is the reason this is claim-first
rather than claim-after, and it is worth knowing before this ships.

What this deliberately does NOT do
----------------------------------
Nothing here is on the `chat_runner` / Streamlit path. There is no wamid there to key on, and
the demo UI has no retrying intermediary. This guard is only meaningful where Meta retries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError

from app.db.mongodb import processed_messages_collection

logger = logging.getLogger(__name__)

# Meta retries a failed delivery for days. Anything shorter than that would let a late retry
# through and re-open the exact bug this closes, so the TTL is set past the retry window
# rather than to whatever keeps the collection smallest.
RETENTION_SECONDS = 7 * 24 * 60 * 60  # 7 days

# Why a claim was or was not granted. Kept as a field rather than collapsed into the bool so
# the caller can log the difference — "duplicate" is the metric we want to see fall.
NEW = "new"
DUPLICATE = "duplicate"
NO_ID = "no_id"
STORE_UNAVAILABLE = "store_unavailable"


@dataclass(frozen=True)
class MessageClaim:
    """The result of trying to claim an inbound message id."""

    reason: str
    wamid: str = ""

    @property
    def should_process(self) -> bool:
        """
        Whether the caller should run the turn.

        Only an explicit `DUPLICATE` stops the message. Every other outcome — including a
        Mongo outage — lets it through, because dropping a guest's message is worse than
        processing it twice.
        """
        return self.reason != DUPLICATE


async def ensure_indexes() -> None:
    """Create the TTL index. Idempotent; safe to call on every startup."""
    coll = processed_messages_collection()
    try:
        await coll.create_index(
            "received_at",
            expireAfterSeconds=RETENTION_SECONDS,
            name="received_at_ttl",
        )
    except Exception as e:  # noqa: BLE001 — a background-init step must not block startup
        logger.warning(f"Failed to create processed_messages TTL index: {e}")


async def claim_message(
    wamid: str | None,
    *,
    hotel_id: int | None = None,
    phone: str | None = None,
    channel: str = "whatsapp",
) -> MessageClaim:
    """
    Try to claim an inbound message id. Returns whether this copy is ours to process.

    `_id` is the wamid, so the unique index is Mongo's own `_id` index — no custom index, and
    no partial filter to get wrong. Keyed on the wamid alone rather than (hotel, wamid)
    because a wamid is globally unique to Meta, and a cross-tenant replay should be refused
    too, not just a same-tenant one.
    """
    if not wamid:
        # No id to key on. Processing is the only safe answer: we cannot tell a retry from a
        # first attempt, and refusing would drop real messages.
        logger.warning("Inbound message has no id — cannot deduplicate, processing")
        return MessageClaim(NO_ID)

    try:
        await processed_messages_collection().insert_one({
            "_id": wamid,
            "hotel_id": hotel_id,
            "phone": phone,
            "channel": channel,
            "received_at": datetime.now(timezone.utc),
        })
        return MessageClaim(NEW, wamid)
    except DuplicateKeyError:
        return MessageClaim(DUPLICATE, wamid)
    except Exception as e:  # noqa: BLE001 — never drop a guest message over infrastructure
        logger.warning(f"Idempotency store unavailable, processing anyway ({wamid}): {e}")
        return MessageClaim(STORE_UNAVAILABLE, wamid)
