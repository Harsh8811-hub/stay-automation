"""
The conversation field contract — the one place that decides what a
conversation carries and where it lives.

Why this exists
---------------
Three things used to decide that independently, and none of them agreed:

  1. ``ConversationState`` (app/workflows/state.py) — the LangGraph schema
  2. ``BookingSession``   (app/models/schemas.py) — the MongoDB document
  3. four hand-written field lists in the save and load paths:

       session_mapper.state_to_session          86 explicit kwargs (save)
       session_mapper.session_fields_for_state  82 keys, dead code (load)
       context_loader._load_base_state          ~100 keys (the real load)
       booking_resume.clear_in_progress_booking_fields   38 keys (reset)

Nothing checked them against each other, so every disagreement was silent.

Why "missing from the mapper" is not "left alone"
-------------------------------------------------
``session_repository.upsert_session`` replaces the whole ``workflow_state``
field with ``session.model_dump(mode="json")``, and ``state_to_session``
builds a *fresh* ``BookingSession`` from graph state every turn. So a field
the mapper does not copy does not keep its stored value — it is rebuilt as
its pydantic default and written back. Silently, every turn.

That is the mechanism behind the bugs this contract fixes:

  ``composite_template_shown``  passed to BookingSession(...) but not a field
                                of it; pydantic's default ``extra="ignore"``
                                discarded it. Always False.
  ``applied_coupon``            never copied in either direction. Always None.
  ``upsell_shown``              same.
  ``declined_upsells``          same — a "don't re-offer this" list.
  ``last_upsell_offered``       same.
  ``soft_hold_expires_at``      on both classes, copied by neither. Always None.
  ``active_personality_mode``   same.
  ``price_override``            read into state by context_loader, never
                                copied back out.

How it works now
----------------
``DECLARED`` is derived from ``BookingSession.model_fields`` rather than
typed out, so adding a field to the model is enough to make it round-trip.
The only fields that do not round-trip are named explicitly below, with a
reason. ``tests/test_state_contract.py`` fails if any field is unclassified,
if any declared field is unmapped, or if a round-trip loses a value.

Adding a field is therefore a two-line change with a test behind it:
add it to ``BookingSession``, and either let it round-trip or add it to
``EPHEMERAL`` with a reason.
"""

from __future__ import annotations

from app.models.schemas import BookingSession

# ── Derived: whatever the model declares is what a session stores ─────────
DECLARED: frozenset[str] = frozenset(BookingSession.model_fields)

# ── Owned by the repository, never written from graph state ───────────────
#
# These are real columns on the document, but their value does not come from
# the conversation, so round-tripping them through graph state would mean
# writing a stale copy back over the authoritative one.
REPO_MANAGED: frozenset[str] = frozenset({
    "created_at",   # set once, by $setOnInsert
    "updated_at",   # stamped by upsert_session on every save
    "resolved",     # derived from booking_stage by upsert_session
    "messages",     # owned by save_state, which appends this turn's two rows

    # Written by `takeover_resume_service` after it sends a hand-back, and read
    # by the *next* context refresh to refuse answering the same unanswered guest
    # turn twice. It is deliberately outside graph state: the only thing that
    # clears it is a real guest turn, and a guest turn is exactly when it should
    # expire — a new message is a new run of unanswered turns to answer.
    "takeover_resume_fingerprint",
    "takeover_resume_at",
})

# ── Written to the document, never read back as current-turn state ────────
#
# These are real columns, and the save path fills them, but hydrating them
# into the next turn's state would be wrong: they describe the message being
# handled, not the conversation.
SAVE_ONLY: dict[str, str] = {
    "intent": (
        "this turn's classification. Routing branches on it, so loading last "
        "turn's value would let a stale intent route a message that this "
        "turn's understand did not classify"
    ),
}

# ── The fields that round-trip through graph state ────────────────────────
#
# state_to_session copies every one of these out of state, and the load path
# copies every one of these back in. Both are generated from this set.
PERSISTED: frozenset[str] = DECLARED - REPO_MANAGED - frozenset(SAVE_ONLY)

# ── Mapped by hand because a plain copy would be wrong ────────────────────
#
# Each of these has a handler in BOTH directions. A test asserts that, so a
# name can never be listed here without a handler.
SPECIAL: dict[str, str] = {
    "booking_stage": (
        "str on the wire, BookingStage enum in the model; the save side also "
        "carries a substring ladder for legacy stage strings"
    ),
    "special_requests": "list[str] in state, a single joined str on the model",
    "room_option_offset": "coerced with int() on save; state may hold a str",
    "repeat_response_count": "coerced with int() on save; state may hold a str",
    "last_prompt_action": (
        "falls back to next_action on save, because a node that sets "
        "next_action should also move the prompt pointer"
    ),
    # Nested pydantic models: state holds dicts, the model holds objects.
    "selected_rooms": "list[dict] in state, list[SelectedRoom] on the model",
    "price_breakdown": "dict in state, PriceBreakdown on the model",
    "payment_details": "dict in state, PaymentDetails on the model",
    "conversation_overrides": (
        "dict in state, ConversationOverrides on the model; save_state also "
        "preserves the stored value when this turn's state omits it"
    ),
}

# ── Turn-scoped: declared in the graph, never persisted ───────────────────
#
# These flow between nodes within one turn (and must be declared in
# ConversationState or LangGraph drops them at a node boundary — see the
# show_more_combinations comment in state.py), but they describe the turn
# being handled, not the conversation. Persisting one would be a bug.
#
# Every entry states why it is turn-scoped. If you cannot write that reason,
# it belongs on BookingSession instead.
EPHEMERAL: dict[str, str] = {
    # ── this turn's input and output ──
    "user_message": "the inbound text, consumed by save_state into messages",
    "bot_response": "this turn's reply cards, consumed by save_state",
    "messages_history": (
        "a projection of the stored messages, rebuilt from the session on "
        "every load; storing it would duplicate the authoritative copy"
    ),
    # ── hotel config, reloaded per request ──
    "hotel_config": "serialized HotelConfig, reloaded by context_loader",
    "hotel_name": "read from hotel_config on every load",
    "hotel_rules": "read from hotel_config on every load",
    "persona_prompt": "read from hotel_config on every load",
    "persona_changed": "a signal that this turn changed the persona",
    # ── Gemini's per-turn understanding ──
    # (`intent` is not listed here — BookingSession has a column for it, so it
    # is SAVE_ONLY rather than ephemeral. See that set above.)
    "faq_question": "this turn's extracted FAQ question",
    "pricing_query_room_type": "this turn's parsed pricing query",
    "pricing_query_nights": "this turn's parsed pricing query",
    "pricing_query_num_rooms": "this turn's parsed pricing query",
    "extraction_result": "this turn's raw extraction, consumed in-turn",
    "payment_turn_verdict": (
        "this turn's CONFIRM/CHANGE/UNCLEAR read of the guest's message, "
        "consumed by routing and validator in the same turn; re-derived on "
        "every turn it is needed, so persisting it would only risk a stale "
        "verdict outliving the message that produced it"
    ),
    "llm_unavailable": (
        "turn-scoped flag indicating the LLM provider (Gemini) was unreachable "
        "or returned a 429/5xx/timeout on this turn"
    ),
    "wants_photos": (
        "cleared by validator before search_rooms runs; the durable signal "
        "is quote_presented, which is on the model"
    ),
    "early_late_requote": (
        "this turn's read that the guest named a clock for an early check-in / "
        "late check-out against an already-shown quote, consumed by validator "
        "in the same turn to re-render it; persisting it would re-render the "
        "quote on every later turn, including the one where the guest confirms "
        "and should be reaching payment instead"
    ),
    # ── hotel task classification (per turn) ──
    "task_is_request": "this turn's classification",
    "task_type": "this turn's classification",
    "task_subtype": "this turn's classification",
    "task_urgency": "this turn's classification",
    "task_confidence": "this turn's classification",
    "task_summary": "this turn's classification",
    # ── per-turn handler payloads ──
    "room_service_extraction": "handed to room_service_handler within the turn",
    "amenity_query": "handed to amenity_handler within the turn",
    "rag_context": "hard-coded to None by context_loader, filled in-turn",
    "assumptions": "produced by room_searcher, read by quote_generator, same turn",
    "booking_status": (
        "read from state but never written to it — the real value lives on "
        "the booking document"
    ),
    "selected_option_label": "set from the option tap, consumed by the quote",
    "custom_room_problem": (
        "written by understand and read by generate_response, which runs "
        "later in the same turn (understand -> intent_router -> "
        "generate_response); it explains one rejected answer, and the next "
        "turn either replaces it or clears it on a request that stands"
    ),
    "quote_offset": "read by room_service but never written to state",
    "num_nights": "derived from check_in/check_out by calculate_num_nights",
    "customer_name": "an alias for guest_name, mapped by understand",
    "meal_plan": "read by price_match_service but never written to state",
    "session_updated_at": "a copy of the repo-managed updated_at",
    "prior_guest_name": (
        "loaded by context_loader but never populated, so always None — kept "
        "declared because understand reads it as a name fallback"
    ),
    # ── escalation: the classification, not the outcome ──
    # The durable counterpart is escalation_failure_count, which is on the model.
    "escalation_required": "this turn's classification",
    "escalation_issue_type": "this turn's classification",
    "escalation_reason_code": "this turn's classification",
    "escalation_urgency": "this turn's classification",
    "escalation_sentiment": "this turn's classification",
    "escalation_reason": "this turn's classification",
    "escalation_confidence": "this turn's classification",
    # ── recommendation preferences: this turn's extraction ──
    # Set by understand from the Gemini extraction and read later in the same
    # turn by recommendation_engine. A guest repeating "cheapest" re-sets it.
    "recommendation_preference": "this turn's extraction, read in-turn",
    "display_preference": "this turn's extraction, read in-turn",
    "room_count_preference": "this turn's extraction, read in-turn",
    "comfort_preference": "this turn's extraction, read in-turn",
    # ── in-turn control signals ──
    "missing_fields": (
        "recomputed from scratch by validator every turn and deliberately "
        "reset to [] by context_loader"
    ),
    "force_new_search": (
        "set and cleared within one turn by validator — it is a within-turn "
        "quote trap, not a durable flag"
    ),
    "show_more_combinations": (
        "set by intent_overrides and read by room_searcher in the same turn; "
        "the durable paging state is room_option_offset/room_options_key"
    ),
    "abandoned_draft": (
        "the clear reaches state as a merged dict within the same turn"
    ),
    "room_match_candidates": (
        "vestigial — every writer sets it to None, only response_generator "
        "reads it"
    ),
    "room_selection_resolved": "set and read by validator within one turn",
    "multi_booking_detected": (
        "set by understand to force response_generator to speak, same turn"
    ),
    "group_lead_submitted": (
        "set by understand from this turn's lead-creation outcome, read by "
        "response_generator in the same turn"
    ),
    "bot_response_prepend": (
        "cards queue_manager prepends to this turn's reply, consumed by "
        "response_generator"
    ),
    "reset_booking": (
        "this turn's semantic start-fresh signal from understand, read by "
        "response_generator to clear the draft"
    ),
    # ── consumed in-turn by a single consumer ──
    "price_match_data": (
        "set by understand, read and nulled by price_match_handler in the "
        "same run"
    ),
    "partial_payment_amount": (
        "set by understand, read and nulled by payment/link in the same run"
    ),
    "coupon_code": (
        "set by understand, then consumed (set back to None) by "
        "quote_generator in the same turn"
    ),
    "check_in_date": "an alias price_match_handler reads in-turn",
    "check_out_date": "an alias price_match_handler reads in-turn",
    "claimed_existing_booking": "in-turn flag set when guest claims existing booking",
    "other_hotel_mentioned": (
        "the wrong-number property named this turn, set by the `other_hotel` "
        "fast path in understand and quoted in the same turn's reply. It is a "
        "fact about one message, not about the guest — persisting it would let "
        "a stale name outlive the turn that heard it"
    ),
    "customer_phone": "an alias understand/validator map onto guest phone in-turn",
    "stop_sold_room_name": "set when room is stop-sold for escalation in-turn",
    "opted_out": "set when guest opts out via STOP/UNSUBSCRIBE in-turn",
    # ── booking-stage lifecycle: resolved fresh every turn ──
    # context_loader is the entry node and calls resolve_guest_stage, which
    # looks the guest's booking up by phone, on every single turn. Storing the
    # result would only let it go stale against the booking document, which is
    # the thing being edited by staff and by the payment webhook.
    "lifecycle_stage": (
        "recomputed by context_loader from the booking record on every load; "
        "the durable counterpart is booking_stage"
    ),
    "stage_allowed_tools": "derived from lifecycle_stage, in the same node",
    "resolved_booking_stage": (
        "context_loader's copy of resolve_guest_stage's verdict, captured before "
        "understand can rewrite lifecycle_stage. It exists precisely to be "
        "compared within one turn — `app/services/stage_lock.py` reads it in "
        "feature_gate to tell a deliberate stage change from a node losing the "
        "guest's confirmation. A stored copy would defeat that: the next load "
        "recomputes it, and a stale one would let a demotion through"
    ),
    "active_booking": (
        "the booking record resolve_guest_stage just fetched; re-fetched every "
        "turn, so a stored copy could only disagree with the live document"
    ),
    "active_booking_id": "an alias for the active booking, derived with active_booking",
    "standing_booking": (
        "the confirmed stay a second-booking guest still holds, fetched by the "
        "same resolve_guest_stage call as active_booking and re-derived every "
        "turn; the durable fact is separate_booking_active, which is what makes "
        "the resolver look for it at all"
    ),
    "fail_safe_triggered": (
        "reports whether this turn's stage resolution fell back — a fact "
        "about this lookup, not about the conversation"
    ),
    "is_separate_booking_inquiry": (
        "turn-scoped signal for the same turn's reply. Note it currently has "
        "no reader; the durable latch is separate_booking_clarification_asked"
    ),
    "crm_confirmed_stay": (
        "written by context_loader from resolve_guest_stage's stage context on "
        "every load, exactly like lifecycle_stage, and read the same turn by "
        "the confirmed-stage handler and guest_is_confirmed; the durable record "
        "is the CRM's own CONFIRMED_BOOKING, so a stored copy could only go "
        "stale against a booking staff cancel or amend"
    ),
    "crm_confirmed_room_type": (
        "the room name on that same CRM confirmation, written beside "
        "crm_confirmed_stay in context_loader and read with it in-turn; it "
        "means nothing on its own, so it lives and dies with that flag"
    ),
    "crm_confirmed_details": (
        "the rest of the stay the CRM recorded — the room, the dates, party "
        "size, rooms, nights and rate plan — written beside crm_confirmed_stay "
        "and read in-turn by the confirmed-stage handler, so a guest is not "
        "asked to repeat details we already hold. No money field is carried. "
        "Ephemeral for the same reason crm_confirmed_stay is: staff can amend "
        "the stay at any time, and a stored copy would go stale silently"
    ),
    "crm_confirmed_check_out": (
        "the check-out date on that same CRM confirmation, written beside the "
        "two above from the same stage context. Ephemeral for the reason "
        "crm_confirmed_stay gives — re-resolved from the CRM every turn, so a "
        "stored copy could only go stale against a stay staff amend. Distinct "
        "from the stored `check_out`, which is the guest's typed draft whenever "
        "one exists; the two disagree exactly when a guest with a confirmed "
        "stay is drafting a second one. The durable counterpart is "
        "crm_confirmation_dismissed_check_out, which is a latch and does persist"
    ),
    # ── test-only overrides ──
    "simulated_now": (
        "test-only override of the clock, read by context_loader; production "
        "never sets it, and persisting a frozen time would be a bug"
    ),
    "explicit_booking_stage": (
        "test-only override that short-circuits resolve_guest_stage; the real "
        "stage is always derived from the booking document"
    ),
}


def state_field_names() -> frozenset[str]:
    """
    Every field ConversationState is allowed to declare.

    The contract test asserts ConversationState's annotations are exactly
    this set, so a field cannot be added to the graph without landing in
    PERSISTED (round-trips), SAVE_ONLY or EPHEMERAL (each with a reason).
    """
    return PERSISTED | frozenset(SAVE_ONLY) | frozenset(EPHEMERAL)


def persisted_without_special() -> frozenset[str]:
    """
    The PERSISTED fields mapped by a plain copy.

    This is the set the generated loops walk; SPECIAL is the set they skip.
    Together they must cover PERSISTED exactly, which the contract test checks.
    """
    return PERSISTED - frozenset(SPECIAL)
