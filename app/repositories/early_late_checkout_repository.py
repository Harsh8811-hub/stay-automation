"""
Early / late checkout Repository — reads the paid early check-in and late
check-out fee bands from the MongoDB ``early_late_checkout`` collection.

One document per hotel, keyed by ``hotel_id``, written by the Control Centre
(``updated_by: "admin-ui"``) and never by the bot. The bands are guest-facing
money: a percentage or a flat amount, chosen by how many hours the guest's
stated arrival/departure time falls outside the hotel's standard check-in /
check-out time.

**Read only the ``early_late_checkout`` collection on the production database.**
Four stale look-alike collections exist elsewhere (``StayInn.earlyCheckinCheckout``
in camelCase, and ``early_late_checkout`` under ``bot_migration`` and
``temp_profile``); they are snapshots taken at different times and they disagree
with each other and with production.

**A hotel with no row is not an error, but it is also not a fee.** The Control
Centre backend synthesises a default (``04:00 am / 3h:25% / 6h:40% / 9h:₹1500``)
for a hotel that has never been configured, purely so its UI has something to
show. We do not adopt that default: :data:`BACKEND_DEFAULT` records what it is,
so the two systems cannot silently disagree about the shape, but **no code path
may charge or bound a fee from it**. With no row there is nothing to read
``is_active`` from, so the feature is off — the same as a row with
``is_active: false``. The caller treats ``None`` and ``is_active is False``
identically.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from app.db.mongodb import get_database

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300.0

# hotel_id (str) -> (raw doc or None, monotonic expiry)
_CONFIG_CACHE: dict[str, tuple[dict | None, float]] = {}

# What the Control Centre backend returns for a hotel with no configured row.
# Recorded for documentation and tests only — never returned by the reader.
BACKEND_DEFAULT: dict[str, Any] = {
    "early_check_in": {
        "earliest_time": "04:00 am",
        "rules": [
            {"hours": 3, "enabled": True, "type": "%", "value": 25},
            {"hours": 6, "enabled": True, "type": "%", "value": 40},
            {"hours": 9, "enabled": True, "type": "₹", "value": 1500},
        ],
    },
    "late_check_out": {
        "latest_time": "08:00 pm",
        "rules": [
            {"hours": 3, "enabled": True, "type": "%", "value": 25},
            {"hours": 6, "enabled": True, "type": "%", "value": 40},
            {"hours": 9, "enabled": True, "type": "₹", "value": 1500},
        ],
    },
    "blackout_dates": {"early_check_in": [], "late_check_out": []},
    "room_controls": [],
    "is_active": True,
}


class EarlyLateRule(BaseModel):
    """One time band: 'this many hours outside standard, for this much money'.

    ``hours`` counts hours *before* standard check-in (early) or *after*
    standard check-out (late). ``value_type`` is ``"%"`` (a percentage of one
    night's room rate) or ``"₹"`` (a flat amount). It is named ``value_type``
    rather than ``type`` to avoid shadowing the builtin in this module; the
    stored key is still ``type``.
    """

    hours: float
    enabled: bool = True
    value_type: str = "₹"
    value: float = 0.0


class EarlyLateRuleSet(BaseModel):
    """The bands for one direction, plus the outer bound on the clock.

    ``boundary`` is ``earliest_time`` for early check-in and ``latest_time`` for
    late check-out — the furthest the hotel will go. A request beyond it is not
    a band, it is a full extra day.
    """

    boundary: str | None = None
    rules: list[EarlyLateRule] = Field(default_factory=list)


class RoomControl(BaseModel):
    """Whether a room category is eligible, per direction.

    ``room_category`` is matched against the booked room's name. For hotel
    111111 these match ``room_types.name`` exactly, uppercase; matching is still
    normalised because other hotels are not so tidy.
    """

    room_category: str
    early_check_in: bool = False
    late_check_out: bool = False


class EarlyLateCheckoutConfig(BaseModel):
    """A parsed ``early_late_checkout`` row.

    Existence of this object means a row was found. Whether the feature is on is
    always ``is_active`` — check it at the call site, never assume.
    """

    hotel_id: str
    is_active: bool = False
    early_check_in: EarlyLateRuleSet | None = None
    late_check_out: EarlyLateRuleSet | None = None
    blackout_early: list[str] = Field(default_factory=list)
    blackout_late: list[str] = Field(default_factory=list)
    room_controls: list[RoomControl] = Field(default_factory=list)


def _tenant_variants(hotel_id: int | str) -> list[Any]:
    """The hotel_id spellings to match — the collection stores it as a string."""
    str_hid = str(hotel_id)
    if str_hid.isdigit():
        return [str_hid, int(str_hid)]
    return [str_hid]


def _as_float(value: Any) -> float | None:
    """Coerce to float, or None. Booleans are rejected — True is not the number 1."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    """Coerce to bool, or None when the value says nothing at all."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _normalise_value_type(raw: Any) -> str:
    """Map whatever the UI wrote to ``"%"`` or ``"₹"``. Unknown defaults to flat."""
    text = str(raw or "").strip().lower()
    if text.startswith("%") or "percent" in text or "pct" in text:
        return "%"
    return "₹"


def _parse_rule_set(raw: Any, *, boundary_key: str) -> EarlyLateRuleSet | None:
    """Parse one direction's block. Returns None if it is not a dict at all.

    Rules that carry no usable number are dropped rather than defaulted — a
    malformed band must not become a free one, or a ₹0 one.
    """
    if not isinstance(raw, dict):
        return None

    boundary = raw.get(boundary_key)
    boundary = str(boundary).strip() if boundary else None

    rules: list[EarlyLateRule] = []
    for item in raw.get("rules") or []:
        if not isinstance(item, dict):
            continue
        hours = _as_float(item.get("hours"))
        value = _as_float(item.get("value"))
        if hours is None or value is None or hours <= 0:
            logger.warning(
                "early_late_checkout: dropping malformed rule %r (hours=%r value=%r)",
                item, item.get("hours"), item.get("value"),
            )
            continue
        rules.append(
            EarlyLateRule(
                hours=hours,
                enabled=_as_bool(item.get("enabled")) is not False,
                value_type=_normalise_value_type(item.get("type")),
                value=value,
            )
        )

    return EarlyLateRuleSet(boundary=boundary, rules=rules)


def _parse_room_controls(raw: Any) -> list[RoomControl]:
    controls: list[RoomControl] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        category = item.get("room_category") or item.get("roomCategory")
        if not category:
            continue
        controls.append(
            RoomControl(
                room_category=str(category),
                early_check_in=_as_bool(item.get("early_check_in")) is True,
                late_check_out=_as_bool(item.get("late_check_out")) is True,
            )
        )
    return controls


def _parse_blackout_dates(raw: Any) -> tuple[list[str], list[str]]:
    if not isinstance(raw, dict):
        return [], []
    early = [str(d)[:10] for d in (raw.get("early_check_in") or []) if d]
    late = [str(d)[:10] for d in (raw.get("late_check_out") or []) if d]
    return early, late


def parse_early_late_checkout(doc: dict | None) -> EarlyLateCheckoutConfig | None:
    """Parse a raw ``early_late_checkout`` row. Pure; never raises.

    ``None`` in, ``None`` out — "no row" and "row that failed to parse" are the
    same thing to every caller (feature off), but they are logged differently.
    """
    if not isinstance(doc, dict):
        return None

    try:
        blackout_early, blackout_late = _parse_blackout_dates(doc.get("blackout_dates"))
        return EarlyLateCheckoutConfig(
            hotel_id=str(doc.get("hotel_id") or ""),
            # Fail closed: an absent is_active is off, not on.
            is_active=_as_bool(doc.get("is_active")) is True,
            early_check_in=_parse_rule_set(doc.get("early_check_in"), boundary_key="earliest_time"),
            late_check_out=_parse_rule_set(doc.get("late_check_out"), boundary_key="latest_time"),
            blackout_early=blackout_early,
            blackout_late=blackout_late,
            room_controls=_parse_room_controls(doc.get("room_controls")),
        )
    except Exception as e:
        logger.warning(f"early_late_checkout: could not parse row for hotel {doc.get('hotel_id')}: {e}")
        return None


async def get_early_late_checkout_doc(hotel_id: int | str) -> dict | None:
    """Fetch the raw row for a hotel, or None. Cached, and never raises.

    The ``None`` result is cached too: an unconfigured hotel is the common case
    and must not hammer Mongo on every quote.
    """
    cache_key = str(hotel_id)
    now = time.monotonic()
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None and now < cached[1]:
        return cached[0]

    doc: dict | None = None
    try:
        db = get_database()
        doc = await db["early_late_checkout"].find_one({"hotel_id": {"$in": _tenant_variants(hotel_id)}})
    except Exception as e:
        logger.warning(f"Error fetching early_late_checkout for hotel {hotel_id}: {e}")
        return None

    _CONFIG_CACHE[cache_key] = (doc, now + _CACHE_TTL_SECONDS)
    return doc


async def get_early_late_checkout_config(hotel_id: int | str) -> EarlyLateCheckoutConfig | None:
    """Parsed config for a hotel, or None when there is no usable row."""
    return parse_early_late_checkout(await get_early_late_checkout_doc(hotel_id))


def format_early_late_checkout_policy(cfg: EarlyLateCheckoutConfig | None) -> str:
    """Guest-facing policy text, deliberately carrying **no numbers**.

    Used when a guest asks about early check-in or late check-out but has not
    stated a time, so there is nothing to price. Quoting a band they did not ask
    about would invent a charge.
    """
    if cfg is None or not cfg.is_active:
        return ""

    parts: list[str] = []
    if cfg.early_check_in and cfg.early_check_in.rules:
        bound = f" (from {cfg.early_check_in.boundary})" if cfg.early_check_in.boundary else ""
        parts.append(f"Early check-in{bound} is available at an extra charge")
    if cfg.late_check_out and cfg.late_check_out.rules:
        bound = f" (until {cfg.late_check_out.boundary})" if cfg.late_check_out.boundary else ""
        parts.append(f"late check-out{bound} is available at an extra charge")

    if not parts:
        return ""

    return (
        " · ".join(parts)
        + ". Share your arrival or departure time and I'll confirm the exact amount — "
        "both are subject to availability."
    )


def invalidate_cache(hotel_id: int | str | None = None) -> None:
    """Drop the cached row for one hotel, or all of them."""
    if hotel_id is None:
        _CONFIG_CACHE.clear()
        return
    _CONFIG_CACHE.pop(str(hotel_id), None)
