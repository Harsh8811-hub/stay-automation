"""
Lifecycle Handler Node — safe, auditable booking state transitions.

Handles:
  1. Start-Fresh confirmation flow (2-turn: warn → confirm/abort)
  2. Stay Extension summary + confirmation (for checked-in guests)
  3. Early Checkout summary + confirmation (for checked-in guests)
  4. Human-takeover blocking of lifecycle actions

This node is deterministic Python — no AI calls.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import cast

from app.prompts.templates import text_card
from app.services.lifecycle_manager import (
    build_final_modification_confirmation,
    build_start_fresh_warning,
    check_human_takeover,
    detect_early_checkout,
    detect_stay_extension,
    format_early_checkout_summary,
    format_start_fresh_warning,
    format_stay_extension_summary,
    invalidate_payment_links,
    log_lifecycle_event,
    log_modification_audit,
)
from app.utils.booking_resume import clear_in_progress_booking_fields
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Confirmation pattern helpers
# ──────────────────────────────────────────────────────────────────────────────

_CONFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|proceed|confirm|go\s+ahead|do\s+it|extend|checkout\s+early|start\s*fresh|fresh|1|haan|ha|theek\s*hai|sahi\s*hai)\b",
    re.IGNORECASE,
)
_ABORT = re.compile(
    r"\b(no|nope|nah|cancel|abort|stop|keep|don'?t|never\s*mind|back|2|continue|mat\s*karo|nahi)\b",
    re.IGNORECASE,
)


def _check_user_response(message: str) -> str:
    """Return 'confirm', 'abort', or 'ambiguous'."""
    msg = (message or "").strip().lower()
    # Check abort first to prevent "don't" matching "do it"
    if _ABORT.search(msg):
        return "abort"
    if _CONFIRM.search(msg):
        return "confirm"
    return "ambiguous"


# ──────────────────────────────────────────────────────────────────────────────
# Main handler
# ──────────────────────────────────────────────────────────────────────────────

async def handle_lifecycle(state: ConversationState) -> ConversationState:
    """
    Lifecycle Handler graph node.

    Turn 1: Detect what lifecycle action the guest wants and display a
            confirmation prompt (never mutate state destructively yet).
    Turn 2: Guest confirms or aborts. Apply or discard accordingly.
    """
    user_message = state.get("user_message", "")
    pending_action = state.get("lifecycle_action_pending")
    hotel_id = state.get("hotel_id", 0)

    # ── Check human takeover first — block lifecycle ops if active ────────────
    under_takeover = await check_human_takeover(state)
    if under_takeover:
        await log_lifecycle_event(
            "LIFECYCLE_BLOCKED",
            state,
            description="Lifecycle action blocked: human takeover active",
        )
        msg = (
            "A staff member is currently assisting you. "
            "Lifecycle changes will resume once the session is handed back to me.\n\n"
            "If you need immediate help, please let the staff member know directly."
        )
        return cast(ConversationState, {
            **state,
            "bot_response": text_card(msg),
            "next_action": "lifecycle_blocked_takeover",
        })

    # ── Phase 2: Pending action — waiting for guest confirmation ─────────────
    if pending_action:
        response = _check_user_response(user_message)
        action_data: dict = state.get("lifecycle_action_data") or {}

        # ── Booking Modification Confirmation ─────────────────────────────────
        if pending_action == "booking_modification":
            if response == "confirm":
                candidate: dict = state.get("modification_candidate_state") or {}
                if not candidate:
                    # Defensive: candidate state missing — abort gracefully
                    logger.warning("booking_modification confirmed but candidate_state is empty; aborting.")
                    return cast(ConversationState, {
                        **state,
                        "lifecycle_action_pending": None,
                        "lifecycle_action_data": None,
                        "modification_candidate_state": None,
                        "bot_response": text_card(
                            "Something went wrong retrieving your modification. "
                            "Your original booking is unchanged. Please try again."
                        ),
                        "next_action": "modification_error",
                    })

                req_vals: dict = action_data.get("req_vals") or {}
                old_price: float | None = action_data.get("old_price")
                new_price: float | None = action_data.get("new_price")
                generated_at_str = action_data.get("generated_at")

                # ── Candidate Expiry Check (15 minutes) ──
                if generated_at_str:
                    try:
                        generated_at = datetime.fromisoformat(generated_at_str)
                        if (datetime.now(timezone.utc) - generated_at).total_seconds() > 900:  # 15 mins
                            logger.info("Modification candidate expired. generated_at=%s", generated_at_str)
                            return cast(ConversationState, {
                                **state,
                                "lifecycle_action_pending": None,
                                "lifecycle_action_data": None,
                                "modification_candidate_state": None,
                                "requested_values": None,
                                "modified_fields": None,
                                "booking_change_summary": None,
                                "modification_metadata": None,
                                "bot_response": text_card(
                                    "The proposed booking changes have expired. "
                                    "Please tell me again what you'd like to change so I can check current availability."
                                ),
                                "next_action": "modification_expired",
                            })
                    except Exception:
                        logger.warning("Failed to parse generated_at=%s", generated_at_str, exc_info=True)

                # Atomically apply the pre-validated candidate state
                # (no second availability call needed — already done in modification_handler)
                payment_clears = invalidate_payment_links(candidate)
                for k, v in payment_clears.items():
                    candidate[k] = v

                # Clear all modification/pending state
                candidate["requested_values"] = None
                candidate["modified_fields"] = None
                candidate["booking_change_summary"] = None
                candidate["modification_metadata"] = None
                candidate["modification_candidate_state"] = None
                candidate["lifecycle_action_pending"] = None
                candidate["lifecycle_action_data"] = None

                # Revert to quote-ready stage so payment flow works correctly
                old_booking_stage = state.get("booking_stage", "")
                if old_booking_stage in ("awaiting_payment", "confirmed"):
                    candidate["booking_stage"] = "quote_shown"
                else:
                    candidate["booking_stage"] = candidate.get("booking_stage", "quote_shown")

                # Emit immutable audit entry (fire-and-forget)
                try:
                    await log_modification_audit(state, candidate, req_vals, old_price, new_price)
                except Exception:
                    logger.warning("Modification audit log failed (non-fatal)", exc_info=True)

                await log_lifecycle_event(
                    "MODIFICATION_CONFIRMED",
                    candidate,
                    description=(
                        f"Guest confirmed modification. Fields: {', '.join(req_vals.keys())}. "
                        f"Price: ₹{old_price or 0:,.0f} → ₹{new_price or 0:,.0f}"
                    ),
                    metadata=action_data,
                )

                confirmation_text = build_final_modification_confirmation(candidate)

                # If price increased, add payment nudge
                if new_price and old_price and new_price > old_price + 0.01:
                    delta = new_price - old_price
                    confirmation_text += (
                        f"\n\n**Additional amount due: ₹{delta:,.0f}**\n"
                        "A new payment link will be generated for the difference."
                    )

                candidate["bot_response"] = text_card(confirmation_text)
                candidate["next_action"] = "modification_applied"
                candidate["intent"] = "BOOK_ROOM"

                return cast(ConversationState, candidate)

            elif response == "abort":
                await log_lifecycle_event(
                    "MODIFICATION_DISCARDED",
                    state,
                    description="Guest declined modification. Original booking preserved.",
                    metadata=action_data,
                )
                return cast(ConversationState, {
                    **state,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    "modification_candidate_state": None,
                    "requested_values": None,
                    "modified_fields": None,
                    "booking_change_summary": None,
                    "modification_metadata": None,
                    "bot_response": text_card(
                        "No problem! Your original booking is unchanged.\n\n"
                        + _get_current_booking_summary(state)
                    ),
                    "next_action": "modification_aborted",
                })
            else:
                # Ambiguous — re-prompt with the original summary
                return cast(ConversationState, {
                    **state,
                    "bot_response": text_card(
                        "Please reply with **Yes, apply changes** to confirm the modification, "
                        "or **No, keep current booking** to discard it."
                    ),
                    "next_action": "lifecycle_awaiting_confirmation",
                })

        if pending_action == "start_fresh":
            if response == "confirm":
                cleared = clear_in_progress_booking_fields(state)
                await log_lifecycle_event(
                    "BOOKING_RESET",
                    state,
                    description="Guest confirmed start-fresh. All booking progress cleared.",
                )
                # The reset wiped the guest's name and language along with the
                # draft, so greet them the way a brand-new guest is greeted.
                # Asking for dates here — as this used to — would be the bot
                # acting as though it still knew them, and would skip the two
                # questions a real new guest is asked.
                from app.prompts.templates import greeting_response
                return cast(ConversationState, {
                    **state,
                    **cleared,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    "bot_response": greeting_response(
                        hotel_name=state.get("hotel_name") or "our hotel"
                    ),
                    "next_action": "booking_reset",
                })
            elif response == "abort":
                await log_lifecycle_event(
                    "BOOKING_RESET_ABORTED",
                    state,
                    description="Guest declined start-fresh. Booking preserved.",
                )
                return cast(ConversationState, {
                    **state,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    "bot_response": text_card(
                        "✅ No problem! Your current booking is still active.\n\n"
                        + _get_current_booking_summary(state)
                    ),
                    "next_action": "start_fresh_aborted",
                })
            else:
                # Ambiguous — re-prompt
                return cast(ConversationState, {
                    **state,
                    "bot_response": text_card(
                        "Please reply with **Yes, start fresh** to clear your booking, "
                        "or **No, continue current booking** to keep it."
                    ),
                    "next_action": "lifecycle_awaiting_confirmation",
                })

        elif pending_action == "stay_extension":
            if response == "confirm":
                new_check_out = action_data.get("new_check_out", "")
                extra_nights = action_data.get("extra_nights", 0)
                additional_amount = action_data.get("additional_amount", 0.0)
                # Apply the extension
                payment_invalidations = invalidate_payment_links(state)
                await log_lifecycle_event(
                    "STAY_EXTENDED",
                    state,
                    description=f"Stay extended to {new_check_out} (+{extra_nights} nights, +₹{additional_amount:,.0f})",
                    metadata=action_data,
                )
                return cast(ConversationState, {
                    **state,
                    **payment_invalidations,
                    "check_out": new_check_out,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    # Trigger re-quote
                    "total_price": None,
                    "price_breakdown": None,
                    "selected_rooms": [],
                    "allocation_options": [],
                    "booking_stage": "collecting",
                    "force_new_search": True,
                    "bot_response": text_card(
                        f"✅ Your checkout has been extended to **{new_check_out}**.\n\n"
                        "Let me recalculate your updated booking now..."
                    ),
                    "next_action": "stay_extended",
                    "intent": "BOOK_ROOM",
                })
            elif response == "abort":
                return cast(ConversationState, {
                    **state,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    "bot_response": text_card(
                        "No problem! Your checkout date remains as "
                        f"**{state.get('check_out', 'original')}**."
                    ),
                    "next_action": "stay_extension_aborted",
                })
            else:
                return cast(ConversationState, {
                    **state,
                    "bot_response": text_card(
                        "Please reply with **Yes, extend my stay** to confirm, "
                        "or **No, keep current checkout** to cancel."
                    ),
                    "next_action": "lifecycle_awaiting_confirmation",
                })

        elif pending_action == "early_checkout":
            if response == "confirm":
                new_check_out = action_data.get("new_check_out", "")
                payment_invalidations = invalidate_payment_links(state)
                await log_lifecycle_event(
                    "EARLY_CHECKOUT_REQUESTED",
                    state,
                    description=f"Early checkout confirmed to {new_check_out}",
                    metadata=action_data,
                )
                return cast(ConversationState, {
                    **state,
                    **payment_invalidations,
                    "check_out": new_check_out,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    "total_price": None,
                    "price_breakdown": None,
                    "selected_rooms": [],
                    "allocation_options": [],
                    "booking_stage": "collecting",
                    "force_new_search": True,
                    "bot_response": text_card(
                        f"✅ Your checkout has been updated to **{new_check_out}**.\n\n"
                        "Our team will process the revised booking. "
                        "Refund eligibility (if any) will be confirmed within 24 hours."
                    ),
                    "next_action": "early_checkout_confirmed",
                    "intent": "BOOK_ROOM",
                })
            elif response == "abort":
                return cast(ConversationState, {
                    **state,
                    "lifecycle_action_pending": None,
                    "lifecycle_action_data": None,
                    "bot_response": text_card(
                        "Your checkout date remains as "
                        f"**{state.get('check_out', 'original')}**. No changes made."
                    ),
                    "next_action": "early_checkout_aborted",
                })
            else:
                return cast(ConversationState, {
                    **state,
                    "bot_response": text_card(
                        "Please reply with **Yes, check out earlier** to confirm, "
                        "or **No, keep original checkout** to cancel."
                    ),
                    "next_action": "lifecycle_awaiting_confirmation",
                })

    # ── Phase 1: Detect and present the lifecycle action ─────────────────────

    # ── Start Fresh ──────────────────────────────────────────────────────────
    warn = build_start_fresh_warning(state)
    if warn.items_at_risk:
        msg_text = format_start_fresh_warning(warn)
        await log_lifecycle_event(
            "BOOKING_CANCEL_REQUESTED",
            state,
            description="Guest requested start-fresh. Confirmation warning shown.",
            metadata={"items_at_risk": warn.items_at_risk},
        )
        return cast(ConversationState, {
            **state,
            "lifecycle_action_pending": "start_fresh",
            "lifecycle_action_data": {
                "has_dates": warn.has_dates,
                "has_rooms": warn.has_rooms,
                "has_pricing": warn.has_pricing,
                "has_payment_link": warn.has_payment_link,
                "has_pending_modification": warn.has_pending_modification,
            },
            "bot_response": text_card(msg_text),
            "next_action": "lifecycle_awaiting_confirmation",
        })

    # No draft to protect — just clear and greet
    cleared = clear_in_progress_booking_fields(state)
    await log_lifecycle_event("BOOKING_RESET", state, description="Start-fresh: no active draft to protect.")
    from app.prompts.templates import greeting_response
    return cast(ConversationState, {
        **state,
        **cleared,
        "bot_response": greeting_response(
            hotel_name=state.get("hotel_name") or "our hotel"
        ),
        "next_action": "booking_reset",
    })


def _get_current_booking_summary(state: dict) -> str:
    """Short summary of current booking to reassure guest after aborting reset."""
    lines = []
    check_in = state.get("check_in")
    check_out = state.get("check_out")
    from app.utils.helpers import format_display_date
    ci_disp = format_display_date(check_in) or check_in
    co_disp = format_display_date(check_out) or check_out
    if check_in and check_out:
        lines.append(f"**Dates:** {ci_disp} → {co_disp}")
    elif check_in:
        lines.append(f"**Check-in:** {ci_disp}")
    adults = state.get("adults")
    if adults is not None:
        lines.append(f"**Guests:** {adults} adult{'s' if adults != 1 else ''}")
    selected_rooms = state.get("selected_rooms") or []
    if selected_rooms:
        room_names = ", ".join({r.get("room_name", "Room") for r in selected_rooms})
        lines.append(f"**Rooms:** {room_names}")
    total = state.get("total_price")
    if total:
        lines.append(f"**Quote:** ₹{total:,.0f}")
    if not lines:
        return "Your booking progress has been preserved."
    return "Here's what you had so far:\n" + "\n".join(f"• {l}" for l in lines)
