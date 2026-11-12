"""
When does a payment under verification stop being the guest's problem?

Measured 2026-10-27, guest 919999000111: paid at 06:32 on 26 Oct, still reading
"PENDING VERIFICATION" 28 hours later, cancelled and complained. Nothing in the
bot measured the wait, so nothing could ever raise it.

The decision is kept pure here — two datetime arguments and a number — so the
clock edge cases (missing value, naive vs aware, exactly at the threshold) are
testable without a database or a conversation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def as_utc(value: Any) -> datetime | None:
    """
    A stored datetime as an aware UTC datetime, or None if it is not one.

    Rows are written from several paths and read back through pymongo with no
    timezone attached (BSON dates come back naive), while `datetime.now(utc)` is
    aware. Comparing the two raises `TypeError`, which — inside a handler that
    swallows exceptions — would silently disable escalation forever.
    """
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def escalation_due(claimed_at: Any, now: datetime, hours: float) -> bool:
    """True when a payment claim has been waiting at least `hours`."""
    started = as_utc(claimed_at)
    if started is None:
        return False
    current = as_utc(now) or datetime.now(timezone.utc)
    # A negative age means the stored value is in the future — a clock skew or a
    # row written by another host. Never escalate on a value we cannot explain.
    age_hours = (current - started).total_seconds() / 3600.0
    if age_hours < 0:
        return False
    return age_hours >= hours


def hours_waiting(claimed_at: Any, now: datetime) -> float:
    """The age in hours, clamped at 0. Used for the note staff read."""
    started = as_utc(claimed_at)
    current = as_utc(now) or datetime.now(timezone.utc)
    if started is None:
        return 0.0
    return max(0.0, (current - started).total_seconds() / 3600.0)
