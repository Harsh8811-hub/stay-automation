"""
Conversation-level overrides — session-scoped staff decisions.

Staff decisions are the highest-priority truth for the active conversation only.
They never mutate hotel master data (rates, amenities DB, inventory master).
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from pydantic import BaseModel, Field


# Canonical keys per domain (deterministic services read these)
PRICING_KEYS = frozenset({
    "payable_total",       # float — final amount for this conversation
    "discount_amount",     # float
    "discount_percent",    # float
    "rate_note",           # str
    "override_price",      # float — staff command override price
    "override_set_at",     # str — ISO timestamp when staff set the price
    "override_offered",    # bool — whether the 60s delayed offer was presented
})
PAYMENT_KEYS = frozenset({
    "amount_paid_confirmed",  # float — staff-confirmed amount already paid
    "amount_due_override",    # float
    "payment_status",         # str e.g. paid / partial / pending
    "skip_payment_ask",       # bool
})
AMENITY_KEYS = frozenset({
    # dynamic amenity slug → {available: bool, note: str, ...}
})
INVENTORY_KEYS = frozenset({
    "availability_confirmed",  # bool
    "room_type",               # str
    "rooms_available_note",    # str
})
BOOKING_KEYS = frozenset({
    "late_checkout",
    "early_checkin",
    "breakfast_included",
    "airport_pickup",
    "upgrade",
    "special_notes",
    "guest_name",
    "check_in",
    "check_out",
    "num_rooms",
    "adults",
    "children",
})


class OverrideAuditEntry(BaseModel):
    """One staff-decision change event (replace keeps history)."""

    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "staff_takeover"  # staff_takeover | staff_update | system_clear
    reason: str = ""
    staff_id: int | str | None = None
    intervention_id: str | None = None
    summary: str | None = None
    changes: dict[str, Any] = Field(default_factory=dict)  # domain → {key: {old, new}}
    cleared_domains: list[str] = Field(default_factory=list)


class HumanTakeoverState(BaseModel):
    """AI pause/resume while hotel staff handle the guest."""

    active: bool = False
    staff_id: int | str | None = None
    reason: str = "STAFF_MESSAGE"
    activated_at: datetime | None = None
    last_staff_message_at: datetime | None = None
    expires_at: datetime | None = None
    intervention_id: str | None = None
    seen_message_ids: list[str] = Field(default_factory=list)


class ConversationOverrides(BaseModel):
    """
    Highest-priority layer above deterministic engines for THIS conversation.

    Domain dicts hold only explicit staff decisions. Empty domain ⇒ use normal logic.
    """

    human_takeover: HumanTakeoverState = Field(default_factory=HumanTakeoverState)
    pricing: dict[str, Any] = Field(default_factory=dict)
    payment: dict[str, Any] = Field(default_factory=dict)
    amenities: dict[str, Any] = Field(default_factory=dict)
    inventory: dict[str, Any] = Field(default_factory=dict)
    booking: dict[str, Any] = Field(default_factory=dict)
    pending_decisions: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    audit: list[OverrideAuditEntry] = Field(default_factory=list)
    updated_at: datetime | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def add_note(self, text: str) -> None:
        """Append a staff note, keeping only the most recent 30."""
        self.notes.append(text)
        if len(self.notes) > 30:
            self.notes = self.notes[-30:]

    def domain(self, name: str) -> dict[str, Any]:
        return getattr(self, name, None) or {}

    def get(self, domain_name: str, key: str, default: Any = None) -> Any:
        dom = self.domain(domain_name)
        if key not in dom:
            return default
        return dom[key]

    def has(self, domain_name: str, key: str) -> bool:
        return key in self.domain(domain_name)

    def apply_decisions(
        self,
        decisions: dict[str, dict[str, Any]],
        *,
        source: str = "staff_takeover",
        reason: str = "",
        staff_id: int | str | None = None,
        intervention_id: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """
        Merge staff decisions into domain dicts. Newer values replace older.
        Returns the change set recorded in the audit trail.
        """
        change_set: dict[str, Any] = {}
        for domain_name in ("pricing", "payment", "amenities", "inventory", "booking"):
            incoming = decisions.get(domain_name) or {}
            if not isinstance(incoming, dict) or not incoming:
                continue
            current = dict(self.domain(domain_name))
            domain_changes: dict[str, Any] = {}
            for key, new_val in incoming.items():
                if new_val is None:
                    continue
                old_val = current.get(key)
                if old_val == new_val:
                    continue
                domain_changes[key] = {"old": old_val, "new": new_val}
                current[key] = new_val
            if domain_changes:
                setattr(self, domain_name, current)
                change_set[domain_name] = domain_changes

        if summary:
            self.add_note(summary)

        if change_set or summary:
            self.audit.append(
                OverrideAuditEntry(
                    source=source,
                    reason=reason,
                    staff_id=staff_id,
                    intervention_id=intervention_id,
                    summary=summary,
                    changes=change_set,
                )
            )
            if len(self.audit) > 50:
                self.audit = self.audit[-50:]
            self.touch()

        return change_set

    def clear_decision_overrides(self, *, reason: str = "lifecycle_clear") -> None:
        """
        Wipe staff decision domains so they cannot leak into a future booking.
        Preserves human_takeover.seen_message_ids (idempotency against old CRM msgs).
        """
        cleared = []
        for domain_name in ("pricing", "payment", "amenities", "inventory", "booking"):
            if self.domain(domain_name):
                cleared.append(domain_name)
            setattr(self, domain_name, {})

        had_pending = bool(self.pending_decisions)
        self.pending_decisions = {}
        if had_pending:
            cleared.append("pending_decisions")

        if cleared or self.notes:
            self.audit.append(
                OverrideAuditEntry(
                    source="system_clear",
                    reason=reason,
                    cleared_domains=cleared,
                    summary=f"Cleared conversation decision overrides ({reason})",
                )
            )
            if len(self.audit) > 50:
                self.audit = self.audit[-50:]

        self.notes = []
        self.touch()

    def prompt_summary(self) -> str:
        """Compact text for Gemini so the AI does not contradict staff."""
        parts: list[str] = []
        if self.pricing:
            parts.append(f"Pricing overrides: {self.pricing}")
        if self.payment:
            parts.append(f"Payment overrides: {self.payment}")
        if self.amenities:
            parts.append(f"Amenity overrides: {self.amenities}")
        if self.inventory:
            parts.append(f"Inventory overrides: {self.inventory}")
        if self.booking:
            parts.append(f"Booking overrides: {self.booking}")
        if not parts:
            return ""
        return (
            "STAFF CONVERSATION TRUTH (do not contradict; highest priority):\n"
            + "\n".join(f"- {p}" for p in parts)
        )
