"""
Tell the guest when staff resolve their escalation — whichever door staff used.

Why this exists (2026-11-10, product): "Escalations — task and inquiry updates."
The bot already messages the guest when a ticket is resolved *through the API*:
`PATCH /api/escalations/{id}/status` and `POST /api/escalations/{id}/resolve`
both call `_notify_escalation_resolved`. But the CRM panel is the place staff
actually work, and there is no evidence it calls those routes — the same
measurement that made `task_notification_service` necessary showed the panel
writing `hotel_tasks` straight into Mongo.

So a ticket resolved in the panel is silent: the guest raised a complaint, staff
fixed it, and nobody told them. This sweep closes that, by reading the same
collection the panel writes.

Two things to know before changing this file:

1. **The ticket document is not ours alone.** The panel may rewrite it, so the
   "already told this guest" marker lives in its own collection
   (`ticket_notifications`, unique on `ticket_id`).
2. **Cloud Run scales to zero.** The loop only ticks while an instance is warm,
   so it looks back over a window rather than at "now".

Nothing historical fires on the first deploy: a ticket is only eligible if it
was resolved after `ESCALATION_NOTIFICATION_MIN_RESOLVED_AT`. That floor is a
fixed date, not "now", so the open and already-resolved tickets from before the
feature existed are never messaged.

A resolution reached through the API is recorded in the ledger by the route
itself (`record_ticket_notified`), so the sweep does not message that guest a
second time.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

from app.config.settings import get_settings
from app.db.mongodb import get_database, support_tickets_collection

logger = logging.getLogger(__name__)

SWEEP_BATCH = 50
MAX_ATTEMPTS = 3
STALE_CLAIM_MINUTES = 10
SWEEP_INTERVAL_SECONDS = 120

#: Statuses that mean "staff finished this". RESOLVED sends the resolution text;
#: CLOSED is the tidy-up afterwards and must not produce a second message.
_DONE_STATUSES = ("RESOLVED",)

#: How the query matches the status. It is a regex on purpose: the same
#: collection holds the bot's uppercase `RESOLVED` and the panel's lowercase
#: `resolved` (measured 2026-11-11: all 5 resolved rows in production are
#: lowercase, so an exact `"RESOLVED"` match returned nothing and the sweep was
#: silently dead). A status value has no business being case-sensitive.
_STATUS_MATCH = {"$regex": r"^resolved$", "$options": "i"}


# ═══════════════════════════════════════════════════════════════════
# TIME HELPERS
# ═══════════════════════════════════════════════════════════════════

def _as_utc(value) -> datetime | None:
    """Read a timestamp the panel or the bot wrote, in either shape.

    The same collection holds `datetime` objects (bot writes) and ISO strings
    (panel writes). Comparing them directly raises TypeError.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _resolved_at(ticket: dict) -> datetime | None:
    """When this ticket became RESOLVED, from whichever field recorded it.

    Three sources, because two of them are absent on real rows:

    1. `resolved_at` — what `update_ticket_status` writes. **Measured
       2026-11-11: zero of 290 production tickets have it.**
    2. the newest `STATUS_CHANGED` event — the panel *may* append one. Its 106
       existing events all say "expired", never "resolved".
    3. `updated_at` — the only timestamp the five genuinely resolved tickets
       actually carry. It is approximate: any later note bumps it. That is
       acceptable because the send happens once per ticket and is bounded by
       `ESCALATION_NOTIFICATION_MIN_RESOLVED_AT`, so the worst case is one late
       announcement for a ticket somebody edited — never a replayed backlog.
    """
    stamped = _as_utc(ticket.get("resolved_at"))
    if stamped is not None:
        return stamped

    newest: datetime | None = None
    for event in ticket.get("events") or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("event_type") or "").upper() != "STATUS_CHANGED":
            continue
        metadata = event.get("metadata")
        new_status = ""
        if isinstance(metadata, dict):
            new_status = str(metadata.get("new_status") or "").upper()
        if not new_status:
            description = str(event.get("description") or "").upper()
            if "RESOLVED" in description:
                new_status = "RESOLVED"
        if new_status not in _DONE_STATUSES:
            continue
        when = _as_utc(event.get("timestamp"))
        if when is not None and (newest is None or when > newest):
            newest = when
    if newest is not None:
        return newest
    return _as_utc(ticket.get("updated_at"))


def _resolution_floor() -> datetime:
    """Nothing resolved before this instant is ever announced."""
    raw = getattr(get_settings(), "ESCALATION_NOTIFICATION_MIN_RESOLVED_AT", None)
    return _as_utc(raw) or datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════
# WHAT TO SAY
# ═══════════════════════════════════════════════════════════════════

def resolution_message(ticket: dict, resolution: str | None = None) -> str:
    """The guest-facing words for a resolved ticket.

    The resolution text is staff-written for the guest's benefit ("room shifted
    to the first floor"), so it is quoted. The ticket's `summary` is not: that
    is our internal note-taking and reading it back looks careless.
    """
    text = (resolution or ticket.get("resolution") or "").strip()
    if text:
        return (
            f"Your support request has been resolved by our team: {text} 🙏 "
            "Please let us know if you need anything else!"
        )
    return (
        "Your support inquiry has been resolved by our front desk team. "
        "Please let us know if you need any further assistance! 🙏"
    )


def _send(phone: str, hotel_id, message: str) -> bool:
    from app.services.whatsapp_service import whatsapp_service

    try:
        ok = bool(whatsapp_service.send_message(
            to_number=phone,
            message_body=message,
            hotel_id=hotel_id,
        ))
    except Exception as exc:
        logger.error("ESCALATION_NOTIFY | send failed to %s: %s", phone, exc)
        return False
    if not ok:
        logger.warning("ESCALATION_NOTIFY | provider rejected the message to %s", phone)
    return ok


# ═══════════════════════════════════════════════════════════════════
# THE DEDUPE LEDGER
# ═══════════════════════════════════════════════════════════════════

def _ledger():
    return get_database()["ticket_notifications"]


async def ensure_indexes() -> None:
    await _ledger().create_index("ticket_id", unique=True)
    logger.info("ticket_notifications indexes ensured")


async def _claim(ticket_id: str) -> bool:
    """Take the right to notify this ticket. False if someone already has it.

    The claim is written *before* the send so two warm instances cannot both
    message the same guest. A failed send is retried, but only MAX_ATTEMPTS
    times, so a number the provider keeps rejecting is not poked forever.
    """
    now = datetime.now(timezone.utc)
    # Order matters. The retryable-row update goes first because it is the only
    # atomic step here; the insert is a fallback for "no row yet" and is what
    # the unique index protects. Measured 2026-11-11: in a database where
    # `ensure_indexes` has not run, `insert_one` never raises, so the old
    # insert-first order minted a fresh `attempts: 1` row on every sweep and the
    # retry counter never moved.
    try:
        claimed = await _ledger().find_one_and_update(
            {
                "ticket_id": ticket_id,
                "$or": [
                    {"status": "failed", "attempts": {"$lt": MAX_ATTEMPTS}},
                    {"status": "sending",
                     "claimed_at": {"$lt": now - timedelta(minutes=STALE_CLAIM_MINUTES)}},
                ],
            },
            {"$inc": {"attempts": 1},
             "$set": {"status": "sending", "claimed_at": now}},
        )
    except Exception as exc:
        logger.warning("ESCALATION_NOTIFY | ledger claim failed for %s: %s", ticket_id, exc)
        return False
    if claimed is not None:
        return True

    # No retryable row. Either the ticket has no row (claim it) or its row is
    # already `sent`/exhausted (leave it alone). The existence check keeps the
    # second case correct even where the unique index is missing.
    try:
        if await _ledger().find_one({"ticket_id": ticket_id}, {"_id": 1}):
            return False
    except Exception as exc:
        logger.warning("ESCALATION_NOTIFY | ledger read failed for %s: %s", ticket_id, exc)
        return False

    try:
        await _ledger().insert_one({
            "ticket_id": ticket_id,
            "status": "sending",
            "attempts": 1,
            "claimed_at": now,
        })
        return True
    except DuplicateKeyError:
        # Another instance won the insert between our check and ours.
        return False
    except Exception as exc:
        logger.warning("ESCALATION_NOTIFY | ledger write failed for %s: %s", ticket_id, exc)
        return False


async def record_ticket_notified(ticket_id: str | None, reason: str = "sent_by_staff_route") -> None:
    """Write the dedupe marker after the API route has already messaged the guest.

    Without this, a ticket resolved through `PATCH /api/escalations/{id}/status`
    would be messaged a second time by the sweep minutes later.
    """
    if not ticket_id:
        return
    now = datetime.now(timezone.utc)
    try:
        await _ledger().update_one(
            {"ticket_id": ticket_id},
            {"$setOnInsert": {"claimed_at": now, "attempts": 1},
             "$set": {"status": "sent", "settled_at": now, "reason": reason}},
            upsert=True,
        )
    except Exception as exc:
        logger.warning("ESCALATION_NOTIFY | could not record %s as notified: %s", ticket_id, exc)


async def _settle(ticket_id: str, sent: bool, reason: str | None = None) -> None:
    update: dict = {"status": "sent" if sent else "failed",
                    "settled_at": datetime.now(timezone.utc)}
    if reason:
        update["reason"] = reason
    await _ledger().update_one({"ticket_id": ticket_id}, {"$set": update})
    if not sent:
        logger.info("ESCALATION_NOTIFY_FAILED | ticket=%s | %s", ticket_id, reason)


# ═══════════════════════════════════════════════════════════════════
# THE SWEEP
# ═══════════════════════════════════════════════════════════════════

def _eligible(ticket: dict, floor: datetime, now: datetime) -> bool:
    if str(ticket.get("status") or "").upper() not in _DONE_STATUSES:
        return False
    resolved_at = _resolved_at(ticket)
    if resolved_at is None:
        return False
    return floor <= resolved_at <= now


async def sweep_resolved_tickets() -> int:
    """Announce every ticket staff resolved that the guest has not heard about.

    Returns the number of guests actually messaged.
    """
    settings = get_settings()
    if not getattr(settings, "ESCALATION_NOTIFICATION_SWEEP_ENABLED", False):
        return 0

    now = datetime.now(timezone.utc)
    floor = _resolution_floor()
    notified = 0

    coll = support_tickets_collection()
    # Only "sent" is final. A failed ticket must come back round, and `_claim`
    # decides whether it has retries left.
    already = set(await _ledger().distinct("ticket_id", {"status": "sent"}))
    # Newest first, so the batch never fills up with old tickets that sit before
    # the floor and would otherwise starve today's.
    cursor = coll.find(
        {
            "status": _STATUS_MATCH,
            "ticket_id": {"$nin": list(already)},
        }
    ).sort("_id", -1).limit(SWEEP_BATCH)

    async for ticket in cursor:
        ticket_id = ticket.get("ticket_id")
        if not ticket_id or not _eligible(ticket, floor, now):
            continue
        # The claim comes first: `_settle` only updates an existing ledger row,
        # so settling a ticket that was never claimed wrote nothing, and the
        # ticket came back — and re-logged — on every sweep, forever.
        if not await _claim(ticket_id):
            continue
        phone = ticket.get("customer_phone") or ticket.get("customer_identifier")
        if not phone:
            logger.info("ESCALATION_NOTIFY_SKIPPED | ticket=%s | no phone", ticket_id)
            await _settle(ticket_id, False, "no phone")
            continue
        try:
            sent = _send(phone, ticket.get("hotel_id", 111111), resolution_message(ticket))
        except Exception as exc:
            logger.error("ESCALATION_NOTIFY | ticket=%s raised: %s", ticket_id, exc, exc_info=True)
            await _settle(ticket_id, False, f"exception: {exc}")
            continue
        await _settle(ticket_id, sent, None if sent else "nothing reached the guest")
        if sent:
            notified += 1
            logger.info(
                "ESCALATION_NOTIFY_SENT | ticket=%s | hotel=%s | phone=…%s",
                ticket_id, ticket.get("hotel_id"), str(phone)[-4:],
            )

    return notified


async def start_escalation_notification_background_task():
    """Background loop: every two minutes, tell guests their ticket is resolved."""
    logger.info("Starting escalation-resolution notification sweep (2 min interval)...")
    while True:
        try:
            await sweep_resolved_tickets()
        except Exception as e:
            logger.error(f"Error in escalation notification sweep: {e}")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
