"""
Context Loader Node — loads session AND hotel configuration into graph state.

This is the FIRST node in every graph execution.
It hydrates the state from:
1. Persisted session (MongoDB)
2. Hotel configuration (MongoDB → cached)

The hotel config is split into:
- AI config (for Gemini prompts)
- Business config (for Python feature enforcement)
"""

import logging

from app.config.settings import get_settings
from app.models.schemas import BookingSession
from app.repositories import session_repository
# Deprecated: customization_repository removed
from app.services.hotel_config_service import get_hotel_config
from app.services.second_booking import second_booking_gate_enabled
from app.utils.helpers import generate_session_id
from app.workflows.session_mapper import session_to_state
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


def _second_booking_gate_enabled(state: dict) -> bool:
    """Thin alias so this module reads in its own terms; the logic is shared.

    `stage_confirmed` asks the same function before it writes the latch, so the
    read and write sides cannot disagree about whether the feature is on.
    """
    return second_booking_gate_enabled(state)


async def _apply_pending_staff_context(
    session: BookingSession,
    hotel_id: int | str,
    customer_identifier: str,
) -> None:
    """
    Fold staff context queued while the guest had no live session into the session.

    The pending document is single-shot: it is deleted whether or not it was still
    inside its 24h window, so an expired note can neither be applied nor resurface.
    """
    from datetime import datetime, timezone
    from app.db.mongodb import pending_context_collection
    from app.models.schemas import MessageRecord
    from app.models.conversation_overrides import ConversationOverrides
    from app.utils.phone import phone_lookup_variants

    pending_col = pending_context_collection()
    p_variants = phone_lookup_variants(customer_identifier) or [customer_identifier]
    # The writer may have stored the hotel id as either str or int, same as here.
    h_variants = [hotel_id, str(hotel_id), int(hotel_id)] if str(hotel_id).isdigit() else [hotel_id]
    pending = await pending_col.find_one({
        "hotel_id": {"$in": h_variants},
        "phone": {"$in": p_variants}
    })
    if not pending:
        return

    exp = pending.get("expires_at")
    is_valid = True
    if exp:
        exp_aware = exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)
        is_valid = exp_aware > datetime.now(timezone.utc)

    summary = pending.get("staff_summary")
    # Shape-guarded: the document is deleted only on the success path, so a
    # malformed one would raise on every guest turn and never be cleaned up.
    decisions = pending.get("staff_decisions")
    if not isinstance(decisions, dict):
        decisions = None
    if is_valid and summary:
        from app.services.takeover_engine import apply_staff_decisions

        session.messages.append(MessageRecord(
            role="system",
            content=f"SYSTEM NOTE: Human agent interacted with the guest. Summary: {summary}",
            sender_type="system",
        ))
        if not getattr(session, "conversation_overrides", None):
            session.conversation_overrides = ConversationOverrides()
        if decisions:
            # Written while the guest had no session, applied at the same strength
            # as the live-session path. The summary rides along into the audit
            # trail and `add_note`, so it is not added again below.
            await apply_staff_decisions(
                session,
                decisions,
                summary=summary,
                source="staff_update",
                reason="PENDING_CONTEXT",
            )
        else:
            session.conversation_overrides.add_note(f"Staff Summary: {summary}")
        await session_repository.upsert_session(session)
        logger.info(
            f"PENDING_CONTEXT_APPLIED | hotel={hotel_id} | phone={customer_identifier} | "
            f"decisions={sorted(decisions or {})}"
        )

    await pending_col.delete_one({"_id": pending["_id"]})


async def load_context(state: ConversationState) -> ConversationState:
    """
    Load or create a session, then populate the graph state.

    Also loads hotel configuration (cached) and injects it into state
    so downstream nodes can:
    - Use AI config for prompt construction
    - Use business config for feature gate enforcement
    """
    session_id = state.get("session_id", "")
    hotel_id = state.get("hotel_id", 0)

    # Normalize phone so WhatsApp / web resume share the same identifier
    from app.utils.phone import digits_only
    raw_customer = state.get("customer_identifier", "") or ""
    customer_identifier = digits_only(raw_customer) or raw_customer

    session: BookingSession | None = None

    # Try to load existing session (strictly scoped to hotel_id)
    if session_id:
        session = await session_repository.get_session(session_id, hotel_id=hotel_id)
        if session and hotel_id and str(session.hotel_id) != str(hotel_id):
            logger.warning(
                "CROSS_TENANT_LOAD_CONTEXT_DISCARD | session=%s sess_hotel=%s state_hotel=%s",
                session_id, session.hotel_id, hotel_id,
            )
            session = None

    # If no session found, try by customer identifier
    if not session and customer_identifier:
        session = await session_repository.find_active_session(
            hotel_id, customer_identifier
        )

    # Create new session if none found
    if not session:
        session_id = session_id or generate_session_id()
        session = BookingSession(
            session_id=session_id,
            hotel_id=hotel_id,
            customer_identifier=customer_identifier,
        )
        logger.info(f"Created new session: {session_id}")
    elif customer_identifier and not session.customer_identifier:
        session.customer_identifier = customer_identifier

    # ── Check for pending staff context ──────────────────────────
    if customer_identifier:
        try:
            await _apply_pending_staff_context(session, hotel_id, customer_identifier)
        except Exception as e:
            # Advisory only: a pending-context lookup must never cost the guest a turn.
            logger.warning(f"Failed to check/apply pending context for {customer_identifier}: {e}", exc_info=True)

    # ── Check returning guest by phone (Spec §2.2 / §9 / Ticket H3) ──
    prior_guest_name = getattr(session, "prior_guest_name", None) or state.get("prior_guest_name")
    if not prior_guest_name and customer_identifier:
        from app.repositories.booking_repository import find_prior_confirmed_by_phone
        prior_doc = await find_prior_confirmed_by_phone(hotel_id, customer_identifier)
        if prior_doc and prior_doc.get("guest_name"):
            prior_guest_name = prior_doc["guest_name"]

    # ── Load hotel configuration (cached) ────────────────────────
    settings = get_settings()
    request_hotel_id = state.get("hotel_id")
    if request_hotel_id and str(request_hotel_id).strip().isdigit():
        resolved_hotel_id = str(request_hotel_id)
    elif session and session.hotel_id:
        resolved_hotel_id = str(session.hotel_id)
    else:
        resolved_hotel_id = str(settings.HOTEL_ID)

    if session:
        session.hotel_id = resolved_hotel_id

    hotel_config = await get_hotel_config(resolved_hotel_id)
    hotel_name = hotel_config.hotel_name or settings.HOTEL_NAME

    # ── Legacy Customisations (Removed) ──────────────────────────

    # Hydrate graph state from session + config
    from app.services.override_resolver import (
        overrides_from_state,
        apply_booking_field_overrides,
        resolve_payable_override,
        resolve_amount_paid,
        resolve_amount_due,
    )

    # Conversation fields come from the contract, which is what makes this the
    # load counterpart of state_to_session rather than a list that drifts from
    # it. A field written to Mongo but omitted here used to read as its default
    # every turn — silently, because nothing errored.
    base_state = {
        **state,
        **session_to_state(session, state),
        # ── request-scoped: not part of the stored conversation ──
        "session_id": session.session_id,
        "hotel_id": resolved_hotel_id,
        "customer_identifier": customer_identifier or session.customer_identifier or "",
        "hotel_config": hotel_config.model_dump(),
        "hotel_name": hotel_name,
        # The same window the understand prompt uses, from the same setting —
        # these were two literals that had to agree and did not.
        "messages_history": session.get_recent_messages(n=settings.HISTORY_WINDOW_MESSAGES),
        "prior_guest_name": prior_guest_name,
        # A copy of the repo-managed updated_at, which response_generator reads
        # to decide whether the session moved on. Not a stored conversation
        # field, so it is set here rather than in the contract.
        "session_updated_at": getattr(session, "updated_at", None),
        # ── deliberately reset on every turn ──
        # missing_fields is recomputed from scratch by validator, so an empty
        # list here is the intent, not a fallback.
        "missing_fields": [],
        "rag_context": None,
        "bot_response": [],
        "next_action": None,
    }

    # Soft-apply staff booking-field overrides onto working state
    base_state.update(apply_booking_field_overrides(base_state))
    # Only a genuine staff override may set price_override. This used to write the
    # *calculated* total into price_override (via resolve_payable_total's fallback),
    # which made "an override exists" true on every subsequent turn and sent
    # quote_generator down its override branch forever after the first quote.
    override = resolve_payable_override(base_state)
    if override is not None:
        base_state["total_price"] = override
        base_state["price_override"] = override
    paid = resolve_amount_paid(base_state, base_state.get("amount_paid") or 0.0)
    base_state["amount_paid"] = paid
    base_state["amount_due"] = resolve_amount_due(
        base_state,
        total=base_state.get("total_price"),
        amount_paid=paid,
        fallback=base_state.get("amount_due"),
    )

    # ── Resolve deterministic booking lifecycle stage (Sir's Spec) ─────────
    from app.services.stage_resolver import (
        BookingStage,
        resolve_guest_stage,
        second_booking_latch_expired,
    )
    guest_phone = customer_identifier or session.customer_identifier or ""
    simulated_now = state.get("simulated_now")
    explicit_stage = state.get("explicit_booking_stage")
    session_stage = (
        session.booking_stage.value
        if session and getattr(session, "booking_stage", None)
        else None
    )
    # Both dismissals are stored latches, loaded above with the rest of the session,
    # and read here rather than below because they *feed* the resolution rather than
    # record it: they are how a "start fresh" survives the turn that asked for it.
    # `dismissed_booking_id` covers a booking the bot itself created, which resolves
    # the stage on every turn whatever the session says; `dismissed_crm_check_out`
    # covers a stay staff confirmed in the CRM, which has no row of ours.
    # A guest who said yes to S57 is building a second booking. The latch is
    # passed rather than acted on here so the resolver stays the one place that
    # decides a stage — but it is cleared right here the moment the session comes
    # back `confirmed`, because that is the new booking completing and the two
    # stays no longer need separating. Nothing else has to remember to turn it
    # off, which is what keeps a stuck latch from hiding a paid booking.
    separate_booking_active = bool(base_state.get("separate_booking_active")) and (
        session_stage or ""
    ) != BookingStage.CONFIRMED.value
    # The latch hides the guest's own booking id from every stage path, so it
    # must not outlive the draft it was set for. `SECOND_BOOKING_LATCH_HOURS`
    # bounds it; a guest who said "yes, new booking" and then went quiet comes
    # back to their confirmed stay instead of a permanent lead.
    if separate_booking_active and second_booking_latch_expired(
        base_state.get("separate_booking_started_at"), simulated_now
    ):
        logger.info(
            "SECOND_BOOKING_LATCH_EXPIRED | session=%s started_at=%s — the guest "
            "did not finish the second booking; their standing stay owns the "
            "stage again",
            getattr(session, "session_id", None),
            base_state.get("separate_booking_started_at"),
        )
        separate_booking_active = False
        base_state["separate_booking_active"] = False
        base_state["separate_booking_started_at"] = None
    if base_state.get("separate_booking_active") and not separate_booking_active:
        logger.info(
            "SECOND_BOOKING_LATCH_CLEARED | session=%s — the new booking is "
            "confirmed, so the two stays are no longer separated",
            getattr(session, "session_id", None),
        )
        base_state["separate_booking_active"] = False
        base_state["separate_booking_started_at"] = None
    stage_ctx = await resolve_guest_stage(
        hotel_id=resolved_hotel_id,
        customer_phone=guest_phone,
        now_dt=simulated_now,
        explicit_stage=explicit_stage,
        session_stage=session_stage,
        dismissed_crm_check_out=base_state.get("crm_confirmation_dismissed_check_out"),
        dismissed_booking_id=base_state.get("dismissed_booking_id"),
        # Broader than the id above: every row this phone had when it last said
        # "/start" stops naming it. The id-shaped latch is kept for the one stay
        # the guest dismissed *without* a full reset (the checked-out flow).
        guest_reset_at=base_state.get("guest_reset_at"),
        # Both gates, decided here so the resolver never needs a hotel's config.
        separate_booking_active=(
            separate_booking_active and _second_booking_gate_enabled(base_state)
        ),
    )

    # The stay a second-booking guest still holds. Recomputed every load, like
    # `active_booking` beside it, and never written to the session.
    base_state["standing_booking"] = stage_ctx.standing_booking
    if (
        separate_booking_active
        and _second_booking_gate_enabled(base_state)
        and stage_ctx.stage_value == BookingStage.LEAD.value
    ):
        # The draft is the guest's live context now. Leaving the old booking id
        # on the state would make `guest_is_confirmed` answer "you already have a
        # booking" to every room question the new draft asks — the guard exists
        # to stop selling to booked guests, and this guest is deliberately buying.
        base_state["booking_id"] = None

    base_state["lifecycle_stage"] = stage_ctx.stage_value
    # Captured here, while it is still the resolver's verdict. Everything below
    # this line — and every node after this one — is free to rewrite
    # `lifecycle_stage`; `stage_lock` compares against this copy to tell a
    # deliberate stage change from a node losing the guest's confirmation.
    base_state["resolved_booking_stage"] = stage_ctx.stage_value
    base_state["stage_allowed_tools"] = stage_ctx.allowed_tools
    base_state["active_booking"] = stage_ctx.active_booking
    base_state["active_booking_id"] = stage_ctx.active_booking_id
    base_state["fail_safe_triggered"] = stage_ctx.fail_safe_triggered
    # Recomputed every turn from the CRM, so it is deliberately NOT persisted on the
    # session: a stay whose check-out passes mid-conversation must stop reading as
    # confirmed on the very next message.
    base_state["crm_confirmed_stay"] = stage_ctx.crm_confirmed
    base_state["crm_confirmed_room_type"] = stage_ctx.crm_room_type
    # The whole stay the CRM recorded, not just the room name — so a confirmed
    # guest asking "how many guests is my booking for?" is answered from the
    # record. Empty for a guest whose stay the bot owns a row for: that row is
    # the better source, and it is loaded just below.
    base_state["crm_confirmed_details"] = stage_ctx.crm_details or None
    # The stay's own check-out, distinct from `check_out`, which the block below
    # fills from the same context but only when the guest has not typed a draft
    # date of their own. understand's fresh-start path needs the former: it has to
    # name the stay it is dismissing, and the draft date would name the wrong one.
    base_state["crm_confirmed_check_out"] = stage_ctx.check_out_date if stage_ctx.crm_confirmed else None

    # If the guest has an active confirmed, in-house, checked-out, or payment_check booking,
    # the booking stage MUST NOT be overwritten to 'collecting' or 'new'.
    if stage_ctx.stage_value in ("confirmed", "in_house", "checked_out", "payment_check"):
        base_state["booking_stage"] = stage_ctx.stage_value
        if stage_ctx.guest_name and not base_state.get("guest_name"):
            base_state["guest_name"] = stage_ctx.guest_name
        if stage_ctx.active_booking_id:
            base_state["booking_id"] = stage_ctx.active_booking_id
        if stage_ctx.stage_value != "checked_out":
            if stage_ctx.check_in_date and not base_state.get("check_in"):
                base_state["check_in"] = stage_ctx.check_in_date
            if stage_ctx.check_out_date and not base_state.get("check_out"):
                base_state["check_out"] = stage_ctx.check_out_date
        if stage_ctx.room_number:
            base_state["room_number"] = stage_ctx.room_number
        if stage_ctx.amount_paid > 0:
            base_state["amount_paid"] = stage_ctx.amount_paid
        if stage_ctx.total_amount > 0:
            base_state["total_price"] = stage_ctx.total_amount

    # Sync verified booking details from actual DB record so bot never makes assumptions or hallucinates
    ab = stage_ctx.active_booking
    if ab:
        if ab.get("rooms"):
            base_state["selected_rooms"] = ab["rooms"]
            base_state["booked_rooms"] = ab["rooms"]
            base_state["num_rooms"] = len(ab["rooms"])
        elif ab.get("num_rooms"):
            base_state["num_rooms"] = ab["num_rooms"]
        if ab.get("adults") is not None:
            base_state["adults"] = ab["adults"]
        if ab.get("children") is not None:
            base_state["children"] = ab["children"]
        if ab.get("room_type"):
            base_state["room_type"] = ab["room_type"]
        base_state["room_type_preference"] = None

    # Check for active escalation (tickets or urgent escalated operational tasks)
    active_escalation = None
    if guest_phone:
        try:
            from app.repositories.ticket_repository import find_open_ticket_by_customer
            open_ticket = await find_open_ticket_by_customer(resolved_hotel_id, guest_phone)
            if open_ticket:
                active_escalation = {
                    "type": "ticket",
                    "id": open_ticket.get("ticket_id"),
                    "subject": open_ticket.get("subject"),
                    "category": open_ticket.get("category"),
                    "status": open_ticket.get("status"),
                }
            if not active_escalation:
                from app.db.mongodb import hotel_tasks_collection
                t_coll = hotel_tasks_collection()
                open_task = await t_coll.find_one(
                    {
                        "customer_phone": guest_phone,
                        "status": {"$in": ["OPEN", "ASSIGNED", "IN_PROGRESS"]},
                        "$or": [
                            {"priority": "URGENT"},
                            {"escalated_at": {"$ne": None}},
                            {"status": "ESCALATED"},
                        ],
                    },
                    sort=[("updated_at", -1)],
                )
                if open_task:
                    active_escalation = {
                        "type": "task",
                        "id": open_task.get("task_id"),
                        "subject": open_task.get("summary") or open_task.get("customer_request"),
                        "category": open_task.get("task_type"),
                        "status": open_task.get("status"),
                    }
        except Exception as e:
            logger.warning(f"Error checking active escalation for {guest_phone}: {e}")

    if active_escalation:
        base_state["has_active_escalation"] = True
        base_state["active_escalation"] = active_escalation
        base_state["escalated_task_id"] = active_escalation["id"]
        base_state["escalation_subject"] = active_escalation.get("subject") or "Guest Support Request"

    return base_state

