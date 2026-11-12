"""
Tell the guest when staff finish a task — whichever door staff came through.

Why this exists (2026-11-09, product): "Task escalation changes — message
automatic update should be sent to the customer, like pending verification when
accepted in the CRM."

The bot already had the message. `PATCH /api/tasks/{id}/status` and
`POST /api/tasks/{id}/confirm-payment` both write the task, confirm the booking
and send WhatsApp. But the CRM panel never calls those routes. Measured in
production Cloud Logging: exactly **one** request to `/api/tasks/*` since
2026-10-01 (2026-10-16, a manual confirm-payment). The panel writes
`STATUS_CHANGED` straight into `stayinn_web.hotel_tasks` with `actor=staff` —
we can see those rows: TASK-20261107-000006 completed by `staff` on
2026-11-08, TASK-20261105-000001/2 on 2026-11-05, and so on.

So a task completed in the panel is a silent event: the guest paid, staff
verified it, and nobody tells the guest. That is the gap this closes.

The mechanism is a sweep, not a webhook, because there is nothing to hook: the
panel's write is a Mongo `find_one_and_update`, and the bot reads the same
database. Two consequences worth knowing before changing this file:

1. **The task document is not ours alone.** The panel may rewrite it. So the
   "already told this guest" marker lives in a separate collection
   (`task_notifications`, unique on `task_id`), not as a field on the task.
2. **Cloud Run scales to zero.** The loop only ticks while an instance is warm.
   A completion that lands while the service is idle is picked up on the next
   warm tick, which is why the sweep looks back a day rather than a minute.

Nothing historical fires on the first deploy: a task is only eligible if its
last `STATUS_CHANGED` is after `TASK_NOTIFICATION_MIN_COMPLETED_AT`. That floor
is a date, not "now", precisely so the 300-odd already-completed rows are never
messaged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

from app.config.settings import get_settings
from app.db.mongodb import get_database, hotel_tasks_collection

logger = logging.getLogger(__name__)

SWEEP_BATCH = 50
MAX_ATTEMPTS = 3
STALE_CLAIM_MINUTES = 10


# ═══════════════════════════════════════════════════════════════════
# TIME HELPERS
# ═══════════════════════════════════════════════════════════════════

def _as_utc(value) -> datetime | None:
    """Read a timestamp the panel or the bot wrote, in either shape.

    The same collection holds `datetime` objects (bot writes, bson) and ISO
    strings (panel writes, e.g. "2026-11-09T08:06:41"). Comparing them directly
    raises TypeError, so everything is normalised here.
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


def _last_status_change(task: dict) -> tuple[datetime | None, str | None]:
    """Newest `STATUS_CHANGED` event: when it happened, and to what status."""
    newest_at: datetime | None = None
    newest_status: str | None = None
    for event in task.get("events") or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("event_type") or "").upper() != "STATUS_CHANGED":
            continue
        when = _as_utc(event.get("timestamp"))
        if when is None:
            continue
        if newest_at is None or when > newest_at:
            newest_at = when
            newest_status = str(event.get("status") or "").upper() or None
    return newest_at, newest_status


def _completion_floor() -> datetime:
    """Nothing completed before this instant is ever announced."""
    raw = getattr(get_settings(), "TASK_NOTIFICATION_MIN_COMPLETED_AT", None)
    return _as_utc(raw) or datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════
# WHAT TO SAY
# ═══════════════════════════════════════════════════════════════════

def _is_payment_task(task: dict) -> bool:
    subtype = str(task.get("subtype") or "").upper()
    task_type = str(task.get("task_type") or "").lower()
    return (
        subtype in ("PAYMENT_CONFIRMATION", "PAYMENT_VERIFICATION")
        or task_type in ("payment_check", "payment_verification", "payment_proof")
    )


def _is_id_task(task: dict) -> bool:
    subtype = str(task.get("subtype") or "").upper()
    task_type = str(task.get("task_type") or "").lower()
    return (
        subtype in ("ID_VERIFICATION", "ID_PROOF")
        or task_type in ("id_verification", "id_proof")
        or "id" in task_type
    )


def _generic_message(task: dict) -> str:
    """Words for a task that is neither payment nor ID.

    Deliberately does **not** quote `summary`: the summary is written for staff
    ("Guest Claims Existing Booking (919883230282)"), and pasting it into a
    WhatsApp message shows the guest our internal note-taking.
    """
    task_type = str(task.get("task_type") or "").lower()
    summary = str(task.get("summary") or "").lower()
    if "early" in task_type or "early" in summary:
        return "Your early check-in request has been approved and completed. We look forward to welcoming you! 🙏"
    if "late" in task_type or "late" in summary:
        return "Your late check-out request has been approved and completed. Enjoy your stay! 🙏"
    return "Your request has been completed by our team. Please let us know if you need anything else! 🙏"


# ═══════════════════════════════════════════════════════════════════
# THE ACT
# ═══════════════════════════════════════════════════════════════════

#: A reference that names no booking. `BK-PENDING` is the old placeholder;
#: `None`/`null` arrive as strings from writers that stringified a missing id.
_NOT_A_BOOKING = {"", "NONE", "NULL", "BK-PENDING", "NAN"}


def _payment_reference(task: dict) -> str | None:
    """The `PAY-…` hold reference, wherever the writer happened to put it.

    A payment task raised before the guest produced proof of payment has no
    `BK-…` id to carry — the real id is only minted once a screenshot arrives.
    What it can carry is the provisional reference, which is the id of the same
    booking row. New tasks put it in `metadata`; older rows may carry it in an
    event's metadata, so both are read.
    """
    direct = task.get("payment_reference")
    if direct:
        return str(direct)
    metadata = task.get("metadata")
    if isinstance(metadata, dict) and metadata.get("payment_reference"):
        return str(metadata["payment_reference"])
    for event in task.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_metadata = event.get("metadata")
        if isinstance(event_metadata, dict) and event_metadata.get("payment_reference"):
            return str(event_metadata["payment_reference"])
    return None


async def resolve_booking_id(task: dict, customer_phone: str | None) -> str | None:
    """The booking this task is about — or None when that cannot be known.

    Never guesses between two live bookings. Measured 2026-11-10: nine of the
    eleven completed payment tasks carried no booking id at all, so an earlier
    version fell back to "the guest's newest confirmed booking". For a guest
    with an old confirmed stay and a new payment being verified, that mails the
    old stay's confirmation slip — wrong dates, wrong amount — to someone who
    just paid for something else. A guest must never receive a slip for a
    booking they did not just pay for, so ambiguity returns None and the caller
    says less rather than saying something false.

    Resolution order, each step refusing to proceed on doubt:
      1. the id the task states, followed through a `PAY-…` rename;
      2. the provisional payment reference the task recorded;
      3. exactly one live booking for this phone — zero or two both give None.
    """
    from app.repositories import booking_repository

    stated = str(task.get("booking_id") or "").strip()
    if stated and stated.upper() not in _NOT_A_BOOKING:
        row = await booking_repository.get_booking(stated)
        if row:
            return row.get("booking_id") or stated
        # Stamped before `reassign_booking_id` renamed the hold to its BK-… id.
        adopted = await booking_repository.get_booking_by_payment_reference(stated)
        if adopted:
            return adopted.get("booking_id")
        return stated

    for reference in filter(None, (_payment_reference(task),)):
        adopted = await booking_repository.get_booking_by_payment_reference(reference)
        if adopted:
            return adopted.get("booking_id")
        row = await booking_repository.get_booking(reference)
        if row:
            return row.get("booking_id") or reference

    if not customer_phone:
        return None

    recent = await booking_repository.get_recent_bookings(
        hotel_id=task.get("hotel_id", 111111),
        customer_phone=customer_phone,
        limit=2,
        booking_status=["confirmed", "pending_payment"],
    )
    if len(recent) == 1:
        return recent[0].get("booking_id")
    if len(recent) > 1:
        logger.warning(
            "TASK_NOTIFY_AMBIGUOUS_BOOKING | task=%s | %d live bookings for …%s — naming none",
            task.get("task_id"), len(recent), str(customer_phone)[-4:],
        )
    return None


async def dispatch_task_completed_notification(
    task: dict,
    amount_override: float | None = None,
) -> bool:
    """Send the guest the one message this completed task deserves.

    Used by both the panel sweep and the staff routes, so a completion means
    the same thing regardless of which side recorded it. `amount_override` is
    how `confirm-payment` says what actually arrived; the sweep has only what
    the task recorded.

    Returns True only if something actually reached WhatsApp. A task with no
    phone, or a send the provider rejected, returns False and the caller may
    retry — nothing here writes the dedupe marker.
    """
    customer_phone = task.get("customer_phone") or task.get("customer_identifier")
    hotel_id = task.get("hotel_id", 111111)
    if not customer_phone:
        logger.info("TASK_NOTIFY_SKIPPED | task=%s | no phone", task.get("task_id"))
        return False

    booking_id = await resolve_booking_id(task, customer_phone)

    if _is_payment_task(task):
        if booking_id:
            from app.repositories import booking_repository
            from app.services.confirmation_service import send_booking_confirmation_notification

            amount_paid = (
                amount_override if amount_override is not None
                else _verified_payment_amount(task)
            )
            if amount_paid is None and get_settings().STAFF_CONFIRM_ASSERTS_FULL_PAYMENT:
                booking = await booking_repository.get_booking(booking_id)
                amount_paid = (
                    float(booking.get("total_price") or booking.get("amount") or 0.0)
                    if booking else None
                )
            await booking_repository.set_booking_confirmed(
                booking_id=booking_id,
                confirmation_source="staff_confirmed",
                amount_paid=amount_paid,
            )
            try:
                slip_kwargs: dict = {"booking_id": booking_id, "force_resend": True}
                if amount_paid is not None:
                    slip_kwargs["amount_paid"] = amount_paid
                sent = await send_booking_confirmation_notification(**slip_kwargs)
                if sent:
                    return True
            except Exception as notify_err:
                logger.error(
                    "TASK_NOTIFY | confirmation slip failed for %s: %s",
                    booking_id, notify_err,
                )

        # No booking row, or a booking we could not name without guessing: the
        # guest still hears the one fact that is true — the payment was
        # verified — without being told a booking is confirmed when nothing on
        # our side can show which one.
        return _send(
            customer_phone, hotel_id,
            "Your payment has been received and verified by our team. 🙏 "
            "Our front desk is completing your booking and will share the details shortly.",
        )

    if _is_id_task(task):
        return _send(customer_phone, hotel_id, "Your ID verification has been completed successfully. 🙏")

    return _send(customer_phone, hotel_id, _generic_message(task))


def _verified_payment_amount(task: dict, override: float | None = None) -> float | None:
    """The amount this task recorded when the guest's receipt came in.

    A completed task carries no amount of its own; the proof arrived earlier as
    `events[].metadata["amount"]`. Reading it here is what stops the slip from
    printing the whole total as the balance (2026-10-26: 11 of 11 confirmed
    PAY-* rows carried `amount_due == total_price`).

    An explicit amount from the caller wins; otherwise the newest event that
    recorded a positive amount is used.
    """
    if override is not None:
        return override
    for event in reversed(task.get("events") or []):
        if not isinstance(event, dict):
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        try:
            amount = float(metadata.get("amount"))
        except (TypeError, ValueError):
            continue
        if amount > 0:
            return amount
    return None


def _send(phone: str, hotel_id, message: str) -> bool:
    from app.services.whatsapp_service import whatsapp_service

    try:
        ok = bool(whatsapp_service.send_message(
            to_number=phone,
            message_body=message,
            hotel_id=hotel_id,
        ))
    except Exception as exc:
        logger.error("TASK_NOTIFY | send failed to %s: %s", phone, exc)
        return False
    if not ok:
        logger.warning("TASK_NOTIFY | provider rejected the message to %s", phone)
    return ok


# ═══════════════════════════════════════════════════════════════════
# THE DEDUPE LEDGER
# ═══════════════════════════════════════════════════════════════════

def _ledger():
    return get_database()["task_notifications"]


async def ensure_indexes() -> None:
    await _ledger().create_index("task_id", unique=True)
    logger.info("task_notifications indexes ensured")


async def _claim(task_id: str) -> bool:
    """Take the right to notify this task. False if someone already has it.

    The claim is written *before* the send so two warm instances cannot both
    message the same guest. A failed send is retried, but only MAX_ATTEMPTS
    times — a phone the provider keeps rejecting must not be poked every two
    minutes forever. A claim left behind by a crashed instance expires.
    """
    now = datetime.now(timezone.utc)
    try:
        await _ledger().insert_one({
            "task_id": task_id,
            "status": "sending",
            "attempts": 1,
            "claimed_at": now,
        })
        return True
    except DuplicateKeyError:
        pass
    except Exception as exc:
        logger.warning("TASK_NOTIFY | ledger write failed for %s: %s", task_id, exc)
        return False

    claimed = await _ledger().find_one_and_update(
        {
            "task_id": task_id,
            "$or": [
                {"status": "failed", "attempts": {"$lt": MAX_ATTEMPTS}},
                {"status": "sending",
                 "claimed_at": {"$lt": now - timedelta(minutes=STALE_CLAIM_MINUTES)}},
            ],
        },
        {"$inc": {"attempts": 1},
         "$set": {"status": "sending", "claimed_at": now}},
    )
    return claimed is not None


async def record_task_notified(task_id: str | None, reason: str = "sent_by_staff_route") -> None:
    """Write the dedupe marker after a staff route has already messaged the guest.

    Without this, a task completed through `PATCH /api/tasks/{id}/status` would
    be messaged a second time by the sweep minutes later.
    """
    if not task_id:
        return
    now = datetime.now(timezone.utc)
    try:
        await _ledger().update_one(
            {"task_id": task_id},
            {"$setOnInsert": {"claimed_at": now, "attempts": 1},
             "$set": {"status": "sent", "settled_at": now, "reason": reason}},
            upsert=True,
        )
    except Exception as exc:
        logger.warning("TASK_NOTIFY | could not record %s as notified: %s", task_id, exc)


async def _settle(task_id: str, sent: bool, reason: str | None = None) -> None:
    update: dict = {"status": "sent" if sent else "failed",
                    "settled_at": datetime.now(timezone.utc)}
    if reason:
        update["reason"] = reason
    await _ledger().update_one({"task_id": task_id}, {"$set": update})
    if not sent:
        logger.info("TASK_NOTIFY_FAILED | task=%s | %s", task_id, reason)


# ═══════════════════════════════════════════════════════════════════
# THE SWEEP
# ═══════════════════════════════════════════════════════════════════

def _eligible(task: dict, floor: datetime, now: datetime) -> bool:
    if str(task.get("status") or "").upper() != "COMPLETED":
        return False
    completed_at, status = _last_status_change(task)
    if completed_at is None:
        return False
    if status and status != "COMPLETED":
        return False
    return floor <= completed_at <= now


async def sweep_completed_tasks() -> int:
    """Announce every task that staff completed and the guest has not heard about.

    Returns the number of guests actually messaged.
    """
    settings = get_settings()
    if not getattr(settings, "TASK_NOTIFICATION_SWEEP_ENABLED", False):
        return 0

    now = datetime.now(timezone.utc)
    floor = _completion_floor()
    notified = 0

    coll = hotel_tasks_collection()
    # Only "sent" is final. A failed task must come back round, and `_claim`
    # decides whether it has retries left — excluding every ledger row here
    # would silently drop failures after the first attempt.
    already = set(await _ledger().distinct("task_id", {"status": "sent"}))
    # Newest first: the batch must never fill up with old completions that sit
    # before the floor and would otherwise starve today's tasks forever.
    cursor = coll.find(
        {
            "status": {"$in": ["COMPLETED", "completed"]},
            "task_id": {"$nin": list(already)},
        }
    ).sort("_id", -1).limit(SWEEP_BATCH)

    async for task in cursor:
        task_id = task.get("task_id")
        if not task_id or not _eligible(task, floor, now):
            continue
        if not await _claim(task_id):
            continue
        try:
            sent = await dispatch_task_completed_notification(task)
        except Exception as exc:
            logger.error("TASK_NOTIFY | task=%s raised: %s", task_id, exc, exc_info=True)
            await _settle(task_id, False, f"exception: {exc}")
            continue
        await _settle(task_id, sent, None if sent else "nothing reached the guest")
        if sent:
            notified += 1
            logger.info(
                "TASK_NOTIFY_SENT | task=%s | hotel=%s | phone=…%s | subtype=%s",
                task_id, task.get("hotel_id"),
                str(task.get("customer_phone") or "")[-4:], task.get("subtype"),
            )

    return notified


TASK_NOTIFICATION_INTERVAL_SECONDS = 120


async def start_task_notification_background_task():
    """Background loop: every two minutes, tell guests their task is done."""
    logger.info("Starting task-completion notification sweep (2 min interval)...")
    while True:
        try:
            await sweep_completed_tasks()
        except Exception as e:
            logger.error(f"Error in task notification sweep: {e}")
        await asyncio.sleep(TASK_NOTIFICATION_INTERVAL_SECONDS)
