"""
Lifecycle Manager Service — governs safe, auditable booking state transitions.

Every destructive action (reset, cancel, stay extension, early checkout) is:
  1. Intercepted before execution.
  2. Summarised for the guest.
  3. Confirmed by the guest before applying.
  4. Audited via ActivityLog.

This module is purely deterministic Python — no AI calls.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses (no side effects, safe to pass around)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StartFreshWarning:
    """Everything the guest will lose if they confirm start-fresh."""
    has_dates: bool = False
    check_in: str | None = None
    check_out: str | None = None
    has_rooms: bool = False
    room_summary: str = ""
    has_pricing: bool = False
    total_price: float | None = None
    has_pending_modification: bool = False
    has_payment_link: bool = False
    items_at_risk: list[str] = field(default_factory=list)


@dataclass
class StayExtensionResult:
    """Result of a stay-extension check for a checked-in guest."""
    valid: bool
    original_check_out: str
    new_check_out: str
    extra_nights: int
    additional_amount: float
    currency: str = "INR"
    room_still_available: bool = True
    failure_reason: str | None = None


@dataclass
class EarlyCheckoutResult:
    """Result of an early-checkout check for a checked-in guest."""
    valid: bool
    original_check_out: str
    new_check_out: str
    nights_removed: int
    revised_total: float
    currency: str = "INR"
    refund_note: str = (
        "Refund eligibility depends on the hotel's cancellation policy. "
        "Our team will confirm the exact amount within 24 hours."
    )
    failure_reason: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# 1. Start-Fresh Warning Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_start_fresh_warning(state: dict[str, Any]) -> StartFreshWarning:
    """
    Inspect current session state and produce a structured summary of
    everything that will be discarded if the guest confirms start-fresh.
    """
    warn = StartFreshWarning()

    check_in = state.get("check_in")
    check_out = state.get("check_out")
    if check_in or check_out:
        from app.utils.helpers import check_in_before_check_out, is_valid_date
        valid_dates = True
        if check_in and check_out:
            if not is_valid_date(check_in) or not is_valid_date(check_out) or not check_in_before_check_out(check_in, check_out):
                valid_dates = False
        elif check_in and not is_valid_date(check_in):
            valid_dates = False
        elif check_out and not is_valid_date(check_out):
            valid_dates = False

        if valid_dates:
            warn.has_dates = True
            warn.check_in = check_in
            warn.check_out = check_out
            date_str = ""
            from app.utils.helpers import format_display_date
            ci_disp = format_display_date(check_in) or check_in
            co_disp = format_display_date(check_out) or check_out
            if check_in and check_out:
                date_str = f"Check-in: {ci_disp}, Check-out: {co_disp}"
            elif check_in:
                date_str = f"Check-in: {ci_disp}"
            else:
                date_str = f"Check-out: {co_disp}"
            warn.items_at_risk.append(f"• Dates — {date_str}")


    selected_rooms: list[dict] = state.get("selected_rooms") or []
    if selected_rooms:
        warn.has_rooms = True
        names = list({r.get("room_name", "Room") for r in selected_rooms})
        warn.room_summary = ", ".join(names)
        warn.items_at_risk.append(f"• Selected room(s) — {warn.room_summary}")
    elif state.get("allocation_options"):
        warn.items_at_risk.append("• Room options shown (no selection yet)")

    adults = state.get("adults")
    children = state.get("children")
    if adults is not None:
        guests_str = f"{adults} adult{'s' if adults != 1 else ''}"
        if children:
            guests_str += f", {children} child{'ren' if children != 1 else ''}"
        warn.items_at_risk.append(f"• Guest details — {guests_str}")

    total_price = state.get("total_price")
    if total_price:
        warn.has_pricing = True
        warn.total_price = total_price
        warn.items_at_risk.append(f"• Pricing — ₹{total_price:,.0f} quoted")

    mod_meta = state.get("modification_metadata")
    if mod_meta or state.get("requested_values"):
        warn.has_pending_modification = True
        warn.items_at_risk.append("• Pending modification (not yet applied)")

    payment_link = state.get("payment_link")
    if payment_link:
        warn.has_payment_link = True
        warn.items_at_risk.append("• Active payment link (will be invalidated)")

    return warn


def format_start_fresh_warning(warn: StartFreshWarning) -> str:
    """Convert StartFreshWarning into a guest-facing message string."""
    if not warn.items_at_risk:
        # Nothing to lose — no need for confirmation
        return ""

    lines = [
        "You have a booking in progress.\n",
        "Starting a new booking will clear your current session progress:\n",
    ]
    lines.extend(warn.items_at_risk)
    lines.append("\nDo you want to start a new booking? This will clear your current session.\n")
    lines.append("• **Yes, start fresh**")
    lines.append("• **No, continue current booking**")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Stay Extension Detector
# ──────────────────────────────────────────────────────────────────────────────

def detect_stay_extension(
    state: dict[str, Any],
    new_check_out: str,
) -> StayExtensionResult:
    """
    For a checked-in guest requesting a later checkout.
    Calculates extra nights + additional cost from current price_per_night.
    Does NOT call external APIs — caller is responsible for availability recheck.
    """
    original_check_out = state.get("check_out", "")
    try:
        orig_dt = date.fromisoformat(original_check_out)
        new_dt = date.fromisoformat(new_check_out)
    except (ValueError, TypeError) as e:
        return StayExtensionResult(
            valid=False,
            original_check_out=original_check_out,
            new_check_out=new_check_out,
            extra_nights=0,
            additional_amount=0.0,
            failure_reason=f"Invalid date format: {e}",
        )

    if new_dt <= orig_dt:
        return StayExtensionResult(
            valid=False,
            original_check_out=original_check_out,
            new_check_out=new_check_out,
            extra_nights=0,
            additional_amount=0.0,
            failure_reason="New check-out must be later than the current check-out date.",
        )

    extra_nights = (new_dt - orig_dt).days

    # Derive nightly rate from selected rooms
    selected_rooms: list[dict] = state.get("selected_rooms") or []
    nightly_rate = sum(
        float(r.get("price_per_night", 0)) for r in selected_rooms
    )
    if nightly_rate == 0 and state.get("total_price") and state.get("check_in"):
        # Fallback: derive from total_price / original nights
        try:
            check_in_dt = date.fromisoformat(state["check_in"])
            original_nights = (orig_dt - check_in_dt).days
            if original_nights > 0:
                nightly_rate = float(state["total_price"]) / original_nights
        except Exception:
            pass

    additional_amount = round(nightly_rate * extra_nights, 2)

    currency = "INR"
    if selected_rooms:
        currency = selected_rooms[0].get("currency", "INR")

    return StayExtensionResult(
        valid=True,
        original_check_out=original_check_out,
        new_check_out=new_check_out,
        extra_nights=extra_nights,
        additional_amount=additional_amount,
        currency=currency,
    )


def format_stay_extension_summary(result: StayExtensionResult, state: dict[str, Any]) -> str:
    """Format a stay extension summary for the guest."""
    check_in = state.get("check_in", "—")
    selected_rooms: list[dict] = state.get("selected_rooms") or []
    room_names = ", ".join({r.get("room_name", "Room") for r in selected_rooms}) or "Current Room"
    adults = state.get("adults", "—")
    children = state.get("children")
    guests_str = f"{adults} adult{'s' if adults != 1 else ''}"
    if children:
        guests_str += f", {children} child{'ren' if children != 1 else ''}"

    symbol = result.currency if result.currency == "INR" else result.currency
    currency_symbol = "₹" if symbol == "INR" else symbol

    from app.utils.helpers import format_display_date
    ci_disp = format_display_date(check_in) or check_in
    co_orig_disp = format_display_date(result.original_check_out) or result.original_check_out
    co_new_disp = format_display_date(result.new_check_out) or result.new_check_out

    lines = [
        "📋 **Stay Extension Summary**\n",
        "**Current Booking**",
        f"• Check-in: {ci_disp}",
        f"• Check-out: {co_orig_disp}",
        f"• Room: {room_names}",
        f"• Guests: {guests_str}\n",
        "**Requested Change**",
        f"• Check-out → **{co_new_disp}**",
        f"• Additional nights: **{result.extra_nights}**\n",
        f"• Additional amount: **{currency_symbol}{result.additional_amount:,.0f}**\n",
        "Your current room will be held subject to availability.\n",
        "Would you like to proceed?",
        "• **Yes, extend my stay**",
        "• **No, keep current checkout**",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Early Checkout Detector
# ──────────────────────────────────────────────────────────────────────────────

def detect_early_checkout(
    state: dict[str, Any],
    new_check_out: str,
) -> EarlyCheckoutResult:
    """
    For a checked-in guest requesting an earlier checkout.
    Calculates revised stay total. Never promises a refund.
    """
    original_check_out = state.get("check_out", "")
    try:
        orig_dt = date.fromisoformat(original_check_out)
        new_dt = date.fromisoformat(new_check_out)
    except (ValueError, TypeError) as e:
        return EarlyCheckoutResult(
            valid=False,
            original_check_out=original_check_out,
            new_check_out=new_check_out,
            nights_removed=0,
            revised_total=0.0,
            failure_reason=f"Invalid date format: {e}",
        )

    if new_dt >= orig_dt:
        return EarlyCheckoutResult(
            valid=False,
            original_check_out=original_check_out,
            new_check_out=new_check_out,
            nights_removed=0,
            revised_total=0.0,
            failure_reason="New check-out must be earlier than the current check-out date.",
        )

    check_in = state.get("check_in", "")
    try:
        check_in_dt = date.fromisoformat(check_in)
    except (ValueError, TypeError):
        return EarlyCheckoutResult(
            valid=False,
            original_check_out=original_check_out,
            new_check_out=new_check_out,
            nights_removed=0,
            revised_total=0.0,
            failure_reason="Could not determine check-in date from session.",
        )

    nights_removed = (orig_dt - new_dt).days
    new_total_nights = (new_dt - check_in_dt).days

    # Derive per-night rate
    selected_rooms: list[dict] = state.get("selected_rooms") or []
    nightly_rate = sum(float(r.get("price_per_night", 0)) for r in selected_rooms)
    if nightly_rate == 0 and state.get("total_price"):
        orig_nights = (orig_dt - check_in_dt).days
        if orig_nights > 0:
            nightly_rate = float(state["total_price"]) / orig_nights

    revised_total = round(nightly_rate * max(new_total_nights, 0), 2)
    currency = selected_rooms[0].get("currency", "INR") if selected_rooms else "INR"

    return EarlyCheckoutResult(
        valid=True,
        original_check_out=original_check_out,
        new_check_out=new_check_out,
        nights_removed=nights_removed,
        revised_total=revised_total,
        currency=currency,
    )


def format_early_checkout_summary(result: EarlyCheckoutResult, state: dict[str, Any]) -> str:
    """Format early checkout summary for the guest. Never promises a refund."""
    check_in = state.get("check_in", "—")
    symbol = "₹" if result.currency == "INR" else result.currency
    original_total = state.get("total_price")

    from app.utils.helpers import format_display_date
    ci_disp = format_display_date(check_in) or check_in
    co_orig_disp = format_display_date(result.original_check_out) or result.original_check_out
    co_new_disp = format_display_date(result.new_check_out) or result.new_check_out

    lines = [
        "📋 **Early Checkout Summary**\n",
        "**Revised Stay**",
        f"• Check-in: {ci_disp}",
        f"• Original Check-out: {co_orig_disp}",
        f"• New Check-out: **{co_new_disp}**",
        f"• Nights removed: **{result.nights_removed}**\n",
        f"• Revised Total: **{symbol}{result.revised_total:,.0f}**",
    ]
    if original_total:
        diff = float(original_total) - result.revised_total
        if diff > 0:
            lines.append(f"• Original Total: {symbol}{float(original_total):,.0f}")
            lines.append(f"• Difference: {symbol}{diff:,.0f}")

    lines.append(f"\n{result.refund_note}\n")
    lines.append("Would you like to proceed with early checkout?")
    lines.append("• **Yes, check out earlier**")
    lines.append("• **No, keep original checkout**")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Human Takeover Check
# ──────────────────────────────────────────────────────────────────────────────

async def check_human_takeover(state: dict[str, Any]) -> bool:
    """
    Return True if Human Takeover Mode is active for this session.

    Prefers session-persisted conversation_overrides (SSOT). Falls back to
    the human_interventions audit collection for legacy in-flight takeovers.
    Lifecycle operations must be blocked when human takeover is active.
    """
    try:
        from datetime import datetime, UTC
        from app.models.conversation_overrides import ConversationOverrides

        # 1) Session override SSOT (from graph state if hydrated)
        cov_raw = state.get("conversation_overrides")
        if cov_raw:
            cov = (
                cov_raw
                if isinstance(cov_raw, ConversationOverrides)
                else ConversationOverrides(**cov_raw)
            )
            ht = cov.human_takeover
            if ht.active and ht.expires_at:
                expires = ht.expires_at
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
                if datetime.now(UTC) < expires:
                    return True

        # 2) Load session if we have an id
        session_id = state.get("session_id")
        hotel_id = state.get("hotel_id")
        if session_id:
            from app.repositories import session_repository
            from app.services.takeover_engine import is_takeover_active

            session = await session_repository.get_session(session_id)
            if session and is_takeover_active(session):
                return True

        # 3) Legacy audit collection fallback
        if hotel_id and session_id:
            from app.db.mongodb import human_interventions_collection
            coll = human_interventions_collection()
            doc = await coll.find_one({
                "hotel_id": hotel_id,
                "session_id": session_id,
                "status": "ACTIVE",
            })
            if doc:
                expires = doc.get("expires_at")
                if expires is not None:
                    if getattr(expires, "tzinfo", None) is None:
                        expires = expires.replace(tzinfo=UTC)
                    return datetime.now(UTC) < expires
                return True
        return False
    except Exception:
        logger.warning("LIFECYCLE_TAKEOVER_CHECK_FAILED", exc_info=True)
        return False  # Fail open — don't block guest unnecessarily


# ──────────────────────────────────────────────────────────────────────────────
# 5. Payment Link Invalidation
# ──────────────────────────────────────────────────────────────────────────────

def invalidate_payment_links(state: dict[str, Any]) -> dict[str, Any]:
    """
    Return state updates that safely clear all payment link fields.
    Must be called whenever a lifecycle action changes the booking price.
    Does NOT cancel the Razorpay link via API — that is handled by payment_service.
    """
    return {
        "payment_link": None,
        "payment_link_id": None,
        "payment_link_amount": None,
        "payment_status": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Audit Logger
# ──────────────────────────────────────────────────────────────────────────────

async def log_lifecycle_event(
    event_type: str,
    state: dict[str, Any],
    *,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Fire-and-forget lifecycle audit log entry.
    Swallows all errors to never interrupt the guest-facing flow.
    """
    try:
        from app.repositories.activity_repository import log_activity
        from app.models.activity_schemas import ActivityLog

        hotel_id = state.get("hotel_id", 0)
        booking_id = state.get("booking_id")
        session_id = state.get("session_id", "")
        queue_id = state.get("queue_id")

        if not description:
            description = f"Lifecycle event: {event_type}"

        activity = ActivityLog(
            hotel_id=hotel_id,
            activity_type=event_type,
            reference_id=booking_id or session_id or "unknown",
            booking_id=booking_id,
            session_id=session_id,
            queue_id=queue_id,
            customer_phone=state.get("customer_identifier") or state.get("guest_phone"),
            customer_name=state.get("guest_name"),
            description=description,
            metadata=metadata or {},
        )
        await log_activity(activity)
    except Exception:
        logger.warning("LIFECYCLE_AUDIT_LOG_FAILED event=%s", event_type, exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Booking Change Management — Deterministic Diff & Confirmation
# ──────────────────────────────────────────────────────────────────────────────

_FIELD_LABELS: dict[str, str] = {
    "check_in": "Check-in",
    "check_out": "Check-out",
    "adults": "Adults",
    "children": "Children",
    "num_rooms": "Rooms",
    "room_type_preference": "Room Type",
    "meal_plan_preference": "Meal Plan",
    "special_requests": "Special Requests",
}

# Fields whose change always requires explicit guest confirmation
_CONFIRMATION_REQUIRED_FIELDS = frozenset({
    "check_in",
    "check_out",
    "adults",
    "children",
    "num_rooms",
    "room_type_preference",
    "meal_plan_preference",
})


def _fmt_date(raw: Any) -> str:
    """Format ISO date string as '24 Jul' for guest-facing display."""
    if not raw:
        return str(raw)
    try:
        d = date.fromisoformat(str(raw))
        return d.strftime("%-d %b") if hasattr(d, "strftime") else str(raw)
    except Exception:
        return str(raw)


def _fmt_value(field: str, value: Any) -> str:
    """Produce a human-readable representation of a field value."""
    if value is None:
        return "—"
    if field in ("check_in", "check_out"):
        return _fmt_date(value)
    if field == "meal_plan_preference":
        # Convert internal codes to friendly names
        mapping = {
            "breakfast": "Breakfast Included",
            "breakfast_dinner": "Breakfast & Dinner",
            "all_inclusive": "All Inclusive",
            "room_only": "Room Only",
            "ep": "Room Only",
            "cp": "Breakfast Included",
            "map": "Breakfast & Dinner",
            "ap": "All Inclusive",
        }
        return mapping.get(str(value).lower().strip(), str(value).title())
    if field in ("adults", "children", "num_rooms"):
        return str(value)
    return str(value).title()


def build_change_summary(
    prev_state: dict[str, Any],
    req_vals: dict[str, Any],
    new_price: float | None,
    old_price: float | None,
    new_rooms: list[dict] | None = None,
) -> str:
    """
    Build a deterministic, human-readable booking change summary.

    Only mentions fields that actually changed.
    Always shows a transparent price comparison when pricing changes.
    """
    lines: list[str] = ["**Booking Changes Prepared**", ""]

    # 1. Field-by-field diff (only truly changed fields)
    for field in _FIELD_LABELS:
        new_val = req_vals.get(field)
        if new_val is None:
            continue
        old_val = prev_state.get(field)
        if old_val == new_val:
            continue
        old_str = _fmt_value(field, old_val)
        new_str = _fmt_value(field, new_val)
        lines.append(f"• **{_FIELD_LABELS[field]}:** {old_str} → {new_str}")

    # 2. Room change (if room was swapped)
    if new_rooms:
        old_rooms: list[dict] = prev_state.get("selected_rooms") or []
        old_room_names = {r.get("room_name", "") for r in old_rooms}
        new_room_names = {r.get("room_name", "") for r in new_rooms}
        if old_room_names and new_room_names and old_room_names != new_room_names:
            old_str = ", ".join(sorted(old_room_names)) or "—"
            new_str = ", ".join(sorted(new_room_names))
            lines.append(f"• **Room:** {old_str} → {new_str}")

    lines.append("")

    # 3. Transparent price comparison
    if old_price is not None and new_price is not None:
        lines.append(f"**Previous Total:** ₹{old_price:,.0f}")
        lines.append(f"**Updated Total:** ₹{new_price:,.0f}")
        delta = new_price - old_price
        if delta > 0.01:
            lines.append(f"**Additional Amount: ₹{delta:,.0f}**")
        elif delta < -0.01:
            lines.append(f"**Price Reduction: ₹{abs(delta):,.0f}**")
        else:
            lines.append("**Price remains the same.**")
    elif new_price is not None:
        lines.append(f"**Updated Total:** ₹{new_price:,.0f}")

    lines.extend([
        "",
        "Would you like to apply these changes?",
        "",
        "• **Yes, apply changes**",
        "• **No, keep current booking**",
    ])
    return "\n".join(lines)


def requires_confirmation(
    req_vals: dict[str, Any],
    prev_state: dict[str, Any],
    old_price: float | None,
    new_price: float | None,
) -> bool:
    """
    Determine whether a booking modification requires explicit guest confirmation.

    Minor changes (special_requests, guest_name, etc.) auto-apply.
    Major changes (dates, guests, room, price delta) require a confirmation turn.
    """
    # Any date, guest count, room, or meal plan change → confirm
    for field in _CONFIRMATION_REQUIRED_FIELDS:
        if field in req_vals:
            new_val = req_vals[field]
            old_val = prev_state.get(field)
            if old_val != new_val:
                # Room type only forces confirmation when a room is already selected
                if field == "room_type_preference" and not prev_state.get("selected_rooms"):
                    continue
                return True

    # Any price delta → confirm
    if old_price is not None and new_price is not None:
        if abs(new_price - old_price) > 0.01:
            return True

    return False


def build_final_modification_confirmation(shadow_state: dict[str, Any]) -> str:
    """
    Generate the final receipt-style confirmation message shown to the guest
    after a modification has been successfully applied.
    """
    lines = ["✅ **Your booking has been updated successfully.**", ""]

    check_in = shadow_state.get("check_in")
    check_out = shadow_state.get("check_out")
    if check_in:
        lines.append(f"✔ Check-in: {_fmt_date(check_in)}")
    if check_out:
        lines.append(f"✔ Check-out: {_fmt_date(check_out)}")

    selected_rooms: list[dict] = shadow_state.get("selected_rooms") or []
    if selected_rooms:
        room_names = ", ".join(r.get("room_name", "Room") for r in selected_rooms)
        lines.append(f"✔ Room: {room_names}")

    adults = shadow_state.get("adults")
    children = shadow_state.get("children")
    if adults is not None:
        guest_str = f"{adults} Adult{'s' if adults != 1 else ''}"
        if children:
            guest_str += f", {children} Child{'ren' if children != 1 else ''}"
        lines.append(f"✔ Guests: {guest_str}")

    meal = shadow_state.get("meal_plan_preference")
    if meal:
        lines.append(f"✔ Meal Plan: {_fmt_value('meal_plan_preference', meal)}")

    total = shadow_state.get("total_price")
    if total is not None:
        lines.append(f"✔ Updated Total: ₹{total:,.0f}")

    return "\n".join(lines)


async def log_modification_audit(
    state: dict[str, Any],
    shadow_state: dict[str, Any],
    req_vals: dict[str, Any],
    old_price: float | None,
    new_price: float | None,
) -> None:
    """
    Emit an immutable audit trail entry for a confirmed booking modification.

    Stored in activity_logs collection via existing ActivityLog pipeline.
    Internal modification_id is NOT exposed to the guest.
    """
    mod_meta: dict = state.get("modification_metadata") or {}
    modification_id = mod_meta.get("modification_id", "")
    booking_id = state.get("booking_id") or shadow_state.get("booking_id")

    changed_fields = list(req_vals.keys())
    previous_values = {f: state.get(f) for f in changed_fields}
    updated_values = dict(req_vals)
    pricing_delta = round((new_price or 0) - (old_price or 0), 2)

    from datetime import timezone
    performed_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "modification_id": modification_id,
        "booking_id": booking_id or "UNKNOWN",
        "modified_fields": changed_fields,
        "previous_values": previous_values,
        "updated_values": updated_values,
        "pricing_delta": pricing_delta,
        "performed_at": performed_at,
        "performed_by": "guest",
    }

    await log_lifecycle_event(
        "MODIFICATION_CONFIRMED",
        shadow_state,
        description=(
            f"Booking modification applied. Fields: {', '.join(changed_fields)}. "
            f"Price delta: ₹{pricing_delta:+,.0f}"
        ),
        metadata=metadata,
    )
