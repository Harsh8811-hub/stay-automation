"""
Conversation override resolver — deterministic priority layer.

All pricing / payment / amenity / inventory / booking services should call
these helpers before falling back to hotel master config.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.models.conversation_overrides import ConversationOverrides

logger = logging.getLogger(__name__)


def parse_overrides(raw: Any) -> ConversationOverrides:
    if isinstance(raw, ConversationOverrides):
        return raw
    if isinstance(raw, dict):
        try:
            return ConversationOverrides(**raw)
        except Exception:
            logger.warning("Failed to parse conversation_overrides; using empty", exc_info=True)
    return ConversationOverrides()


def overrides_from_state(state: dict[str, Any] | None) -> ConversationOverrides:
    if not state:
        return ConversationOverrides()
    return parse_overrides(state.get("conversation_overrides"))


def resolve(state: dict[str, Any] | None, domain: str, key: str, fallback: Any = None) -> Any:
    """Return override value if present, else fallback."""
    cov = overrides_from_state(state)
    if cov.has(domain, key):
        return cov.get(domain, key)
    return fallback


def resolve_payable_override(state: dict[str, Any] | None) -> float | None:
    """
    The staff-negotiated payable total, or None if staff set none.

    Deliberately has no fallback. "Staff negotiated a price" and "a price exists"
    are different questions, and conflating them is what let a calculated total be
    written back into `price_override` and then read as an override — which made
    quote_generator skip the pricing engine on every re-quote.
    """
    cov = overrides_from_state(state)
    if cov.has("pricing", "payable_total"):
        try:
            return float(cov.get("pricing", "payable_total"))
        except (TypeError, ValueError):
            pass
    legacy = (state or {}).get("price_override")
    if legacy is not None:
        try:
            return float(legacy)
        except (TypeError, ValueError):
            pass
    return None


def resolve_effective_total(state: dict[str, Any] | None, calculated: float | None) -> float | None:
    """
    The number to actually charge: the staff override if one exists, else the
    calculated total.

    Use this at payment and collection sites, which want a number. Use
    resolve_payable_override where the question is whether staff intervened.
    """
    override = resolve_payable_override(state)
    return override if override is not None else calculated


def resolve_amount_paid(state: dict[str, Any] | None, fallback: float = 0.0) -> float:
    cov = overrides_from_state(state)
    if cov.has("payment", "amount_paid_confirmed"):
        try:
            return float(cov.get("payment", "amount_paid_confirmed"))
        except (TypeError, ValueError):
            pass
    return float(fallback or 0.0)


def resolve_amount_due(
    state: dict[str, Any] | None,
    *,
    total: float | None = None,
    amount_paid: float | None = None,
    fallback: float | None = None,
) -> float:
    cov = overrides_from_state(state)
    if cov.has("payment", "amount_due_override"):
        try:
            return max(0.0, float(cov.get("payment", "amount_due_override")))
        except (TypeError, ValueError):
            pass
    if total is not None and amount_paid is not None:
        return max(0.0, float(total) - float(amount_paid))
    if fallback is not None:
        return float(fallback)
    return 0.0


def skip_payment_ask(state: dict[str, Any] | None) -> bool:
    cov = overrides_from_state(state)
    if cov.get("payment", "skip_payment_ask") is True:
        return True
    status = str(cov.get("payment", "payment_status") or "").lower()
    if status in ("paid", "waived", "comped"):
        return True
    paid = resolve_amount_paid(state, (state or {}).get("amount_paid") or 0.0)
    total = resolve_effective_total(state, (state or {}).get("total_price"))
    if total is not None and paid >= float(total) and float(total) > 0:
        return True
    return False


def resolve_amenity_override(state: dict[str, Any] | None, amenity_name: str) -> dict[str, Any] | None:
    """Lookup amenity override by slug / fuzzy key."""
    if not amenity_name:
        return None
    cov = overrides_from_state(state)
    amenities = cov.amenities or {}
    if not amenities:
        return None

    slug = _slug(amenity_name)
    if slug in amenities and isinstance(amenities[slug], dict):
        return amenities[slug]
    # direct key
    if amenity_name in amenities and isinstance(amenities[amenity_name], dict):
        return amenities[amenity_name]
    # fuzzy contains
    for key, val in amenities.items():
        if not isinstance(val, dict):
            continue
        if slug in _slug(str(key)) or _slug(str(key)) in slug:
            return val
        if slug in _slug(str(val.get("name") or "")):
            return val
    return None


def clear_overrides_in_state(state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    """Return state updates that clear decision overrides (for lifecycle)."""
    cov = overrides_from_state(state)
    cov.clear_decision_overrides(reason=reason)
    # Also clear legacy flat fields that mirrored staff pricing
    return {
        "conversation_overrides": cov.model_dump(mode="json"),
        "price_override": None,
    }


def apply_booking_field_overrides(state: dict[str, Any]) -> dict[str, Any]:
    """
    Soft-apply booking-domain overrides onto state fields used by the graph.
    Does not invent new booking state — only overlays when staff set a value.
    """
    cov = overrides_from_state(state)
    booking = cov.booking or {}
    updates: dict[str, Any] = {}
    mapping = {
        "check_in": "check_in",
        "check_out": "check_out",
        "guest_name": "guest_name",
        "num_rooms": "num_rooms",
        "adults": "adults",
        "children": "children",
        "room_type": "room_type_preference",
    }
    for src, dest in mapping.items():
        if src in booking and booking[src] is not None:
            updates[dest] = booking[src]
    return updates


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())
