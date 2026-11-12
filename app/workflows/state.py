"""
LangGraph State Definition.

This TypedDict defines everything that flows through the graph.
Each node can read/write specific fields.
"""

from __future__ import annotations

from datetime import datetime

from typing import Any, TypedDict, NotRequired


class ConversationState(TypedDict, total=False):
    """
    Full state that flows through the LangGraph.

    Fields are grouped by responsibility:
    - Session identity
    - Current turn input
    - Gemini understanding output
    - Accumulated booking data
    - Room search results
    - Pricing
    - Payment / Booking
    - Response output
    """

    # ── Session ──────────────────────────────────────────────────
    session_id: str
    hotel_id: int
    customer_identifier: str
    preferred_language: str | None      # Conversational style: "Hinglish", "Formal Hindi", "Kannada"
    conversation_language: str | None   # Base language family: "Hindi", "English", "Telugu", "Kannada"
    conversation_overrides: NotRequired[dict | None]  # Human takeover + future domain overrides
    completed_bookings: NotRequired[list[dict]]  # Multi-booking archive
    booking_queue: NotRequired[list[dict]]
    queue_id: NotRequired[str | None]
    # These three were persisted by the session mapper but not declared here, so
    # they were dropped at every node boundary: the multi-booking queue index
    # could never survive the turn that set it. See app/models/session_contract.py.
    active_booking_index: NotRequired[int]
    failed_bookings: NotRequired[list[dict]]
    combined_quote_shown: NotRequired[bool]
    last_prompt_action: NotRequired[str | None]

    # ── Hotel Configuration (loaded once per request) ────────────
    hotel_config: NotRequired[dict | None]       # Serialized HotelConfig (for state transfer)
    hotel_name: str

    # ── Current turn ─────────────────────────────────────────────
    user_message: str
    messages_history: list[dict]          # Last N messages

    # ── Gemini understanding ─────────────────────────────────────
    intent: str
    faq_question: str | None
    pricing_query_room_type: str | None
    pricing_query_nights: int | None
    pricing_query_num_rooms: int | None

    # ── Accumulated booking data ─────────────────────────────────
    adults: int | None
    children: int | None
    child_ages: list[int]
    child_age_ask_count: NotRequired[int]
    child_age_assumed: NotRequired[bool]
    effective_adults: NotRequired[int | None]
    effective_children: NotRequired[int | None]
    #: Of `effective_children`, how many the HOTEL's own policy makes
    #: chargeable — the band strictly between its free_till and paid_till.
    #: Set by the validator, read by the pricing paths. Absent or 0 means the
    #: hotel has no child policy of its own, and no child charge is added.
    paid_child_count: NotRequired[int]
    child_age_notes: NotRequired[list[str] | None]
    ac_qualified: NotRequired[bool]
    composite_template_shown: NotRequired[bool]
    claimed_existing_booking: NotRequired[bool]
    ambiguous_guests: int | None
    budget: float | None
    num_rooms: int | None
    room_requirements: list[dict] | None
    check_in: str | None
    check_out: str | None
    #: The arrival/departure time the guest volunteered, "HH:MM" 24-hour. Priced
    #: by the early/late checkout fee bands; never asked for. Declared here as
    #: well as on BookingSession because LangGraph drops any key that is not in
    #: this TypedDict — an undeclared field reads back as absent every turn.
    requested_arrival_time: NotRequired[str | None]
    requested_departure_time: NotRequired[str | None]
    #: This turn, the guest named a clock for an early check-in / late check-out
    #: while a quote was already on screen. Set by `apply_intent_overrides` and
    #: read by `validator` in the same turn to re-render the quote instead of
    #: searching again. Deliberately turn-scoped — see the EPHEMERAL note in
    #: `session_contract.py`. Declared here or LangGraph drops it.
    early_late_requote: NotRequired[bool]
    #: True only for the one turn after the bot asked the guest for their
    #: arrival/departure time, so a bare "10AM" answer is understood as the
    #: reply to that question rather than as a new search. Persisted (it has to
    #: cross a turn boundary), which is why it is cleared on every turn that is
    #: not the answer to a fresh question. Declared here or LangGraph drops it.
    awaiting_early_late_time: NotRequired[bool]
    awaiting: NotRequired[str | None]
    llm_unavailable: NotRequired[bool]
    context_refresh_status: NotRequired[str | None]
    room_type_preference: str | None
    #: Why the guest's custom room request could not be honoured, printed above
    #: the prompt when it is asked a second time. Cleared on a request that
    #: stands, so it is never shown for the current question.
    custom_room_problem: NotRequired[str | None]
    selected_rooms: list[dict]
    guest_name: str | None
    guest_email: str | None
    guest_phone: str | None
    special_requests: str | None
    # `customer_name` and `customer_phone` are aliases understand/validator map onto guest info;
    # keeping both declared stops the alias from being dropped mid-turn.
    customer_name: NotRequired[str | None]
    customer_phone: NotRequired[str | None]
    claimed_existing_booking: NotRequired[bool]
    #: The property a guest asked for when it was not ours. Set by the
    #: `other_hotel` fast path in understand. Declared here because LangGraph
    #: drops any key that is not a field of this TypedDict, and this one is the
    #: only record that a turn was a wrong-number turn rather than a sale.
    other_hotel_mentioned: NotRequired[str | None]
    stop_sold_room_name: NotRequired[str | None]
    opted_out: NotRequired[bool]
    prior_guest_name: NotRequired[str | None]
    stop_sold_room_name: NotRequired[str | None]
    simulated_now: NotRequired[Any]
    explicit_booking_stage: NotRequired[str | None]
    checkin_welcome_sent: NotRequired[bool]
    payment_proof_submitted: NotRequired[bool]
    # Staff-set booking extras. Until they were declared and mapped they were
    # reset to their defaults on every save (see session_contract.py).
    special_notes: NotRequired[str | None]
    included_breakfast: NotRequired[bool]
    late_checkout: NotRequired[bool]
    airport_pickup: NotRequired[bool]
    id_docs: NotRequired[list[dict]]
    feature_preferences: list[str] | None
    ac_preference: str | None                   # "AC", "NON_AC", "NO_PREFERENCE"
    ac_qualified: NotRequired[bool]             # True once guest has answered AC qualifier
    occupancy_preference: str | None            # "BASE", "MAX", "NO_PREFERENCE"
    occupancy_mode: str | None                  # "MODE_BASE", "MODE_MAX", "MODE_CUSTOM"
    grouping_mode: str | None                   # "SEGMENTED", etc. (§5.5)
    composition_asked: NotRequired[bool]        # True if 5+ pax fit question was prompted
    composition_options_count: NotRequired[int] # Number of viable arrangements found
    # The group handoff card has already been sent this conversation. Persisted
    # (see BookingSession) so that it is sent once, not on every turn.
    group_handoff_sent: NotRequired[bool]
    subgroups: NotRequired[list[dict] | None]   # Parsed family/group divisions
    gst_slab_applied: float | None
    soft_hold_expires_at: str | None
    meal_plan_preference: str | None
    recommendation_preference: str | None
    display_preference: str | None
    room_count_preference: str | None
    comfort_preference: str | None
    photo_tag_preference: str | None
    # ── Coupons, upsells, add-ons ────────────────────────────────
    # Each of these is read on a turn AFTER the one that set it. They were
    # declared on BookingSession but absent here and absent from the mapper, so
    # they were destroyed twice over: dropped at the node boundary and reset on
    # every save. `declined_upsells` is a "don't re-offer this" list — it could
    # never survive to be read.
    applied_coupon: NotRequired[str | None]
    upsell_shown: NotRequired[bool]
    last_upsell_offered: NotRequired[str | None]
    declined_upsells: NotRequired[list[str]]
    extra_services: NotRequired[list[str]]
    pending_extra_services: NotRequired[list[dict]]
    booking_resume_nudge_count: NotRequired[int]
    # `coupon_code` is different: this turn's extraction, consumed by
    # quote_generator in the same turn (set back to None).
    coupon_code: NotRequired[str | None]
    room_number: str | None
    extra_bed_preference: str | None
    no_extra_bed: NotRequired[bool]
    extra_bed_asked: NotRequired[bool]
    #: Asked the mattress question a second time because the first answer was
    #: unreadable. Bounds the re-ask at one — see `BookingSession.extra_bed_reasked`.
    extra_bed_reasked: NotRequired[bool]
    #: The guest asked for a mattress in so many words — either answering the
    #: post-selection question with a yes, or saying "with extra bed" outright
    #: (`extra_bed_intent` reads both). Fed to the allocator as `wants_bed`.
    #:
    #: Distinct from `extra_bed_preference`, which records the answer to the
    #: question even when it was unreadable; this one is only ever set when the
    #: guest is actually buying the mattress, because it is what prices it.
    #:
    #: Persisted (`BookingSession.wants_bed`), not turn-scoped: a modification
    #: three turns later re-prices the same set, and a dropped flag would
    #: silently refund the mattress the guest already agreed to.
    wants_bed: NotRequired[bool]
    id_verified: NotRequired[bool]
    id_type: str | None
    id_masked_number: str | None
    id_card_details: dict | None
    id_received_date: str | None
    verified_ids: NotRequired[list[dict]]
    # Candidates when room alias resolver needs guest clarification
    room_match_candidates: list[str] | None

    # ── Customizations ───────────────────────────────────────────
    active_personality_mode: str | None
    personality_mode_switched_at: NotRequired[str | None]
    persona_prompt: str | None
    persona_changed: NotRequired[bool]
    hotel_rules: list[str] | None

    # ── Workflow state ───────────────────────────────────────────
    booking_stage: str
    lifecycle_stage: NotRequired[str | None]
    # The stage `stage_resolver` returned at the top of this turn, captured by
    # context_loader before `understand` runs. `lifecycle_stage` is mutable — any
    # node may overwrite it — so it cannot be used to tell whether the stage was
    # moved backwards. This can. Declared here for the same LangGraph reason as
    # `crm_confirmed_stay`: a key missing from this schema is dropped at the node
    # boundary and the reader silently sees None. See `app/services/stage_lock.py`.
    resolved_booking_stage: NotRequired[str | None]
    stage_allowed_tools: NotRequired[list[str]]
    active_booking: NotRequired[dict | None]
    active_booking_id: NotRequired[str | None]
    fail_safe_triggered: NotRequired[bool]
    # Set by context_loader when the CRM's staff confirmation (not a booking row of
    # ours) put the guest in the confirmed stage. Declared here or LangGraph drops it
    # at the node boundary and `understand`'s ghost guard reads it back as None — the
    # exact failure mode this key exists to prevent.
    crm_confirmed_stay: NotRequired[bool]
    # The CRM's own name for a CRM-confirmed guest's room, e.g. "DELUXE DOUBLE KING
    # SIZE BED". Such a guest has no booking row of ours, so `stage_confirmed` would
    # otherwise fall back to a hardcoded "Deluxe Room" and show a guest who booked
    # something else the wrong room. Declared here for the same LangGraph reason as
    # `crm_confirmed_stay` above.
    crm_confirmed_room_type: NotRequired[str | None]
    # The rest of the stay the CRM recorded for that same guest — the room, the
    # dates, guests, rooms, nights and the rate plan it read off the conversation.
    # No money field is carried: the CRM's `bookingAmount` is what was discussed,
    # never what was received, so it must not reach a slip. Turn-scoped and never
    # persisted, exactly like the two keys above: it is re-read from the CRM every
    # turn so a stay that ends mid-conversation stops applying at once.
    # Declared here, or LangGraph drops it at the node boundary and `stage_confirmed`
    # silently gets None — the failure that makes the bot re-ask a guest for details
    # it is already holding.
    crm_confirmed_details: NotRequired[dict | None]
    # The check-out date of that same CRM confirmation, written beside the two
    # above. Turn-scoped like them — but its one reader is `understand`'s fresh-start
    # fast path, which needs to know *which* stay it is dismissing in order to
    # remember that on the session. Reading `check_out` instead would be wrong: it
    # holds the guest's typed draft date whenever one exists.
    crm_confirmed_check_out: NotRequired[str | None]
    # The check-out date of the CRM confirmation this guest asked to dismiss with an
    # explicit "start fresh". Persisted, so the dismissal outlives the turn that set
    # it — without it the next turn re-resolves the CRM confirmation and the reset
    # lasts exactly one message. Holding the *date* rather than a bare bool is what
    # stops it being permanent: a later confirmation for a different stay compares
    # unequal and applies again, so a guest who books again is not left blind.
    crm_confirmation_dismissed_check_out: NotRequired[str | None]
    # The booking row this guest dismissed with an explicit "start fresh", persisted
    # beside the latch above. That one covers a CRM-confirmed stay, which has no row
    # of ours; this one covers a guest who booked through the bot, whose row resolves
    # the stage on every turn no matter what the session holds. Declared here for the
    # same LangGraph reason as the rest of this block.
    dismissed_booking_id: NotRequired[str | None]
    # When this phone last asked to start over, ISO-8601. Unlike `dismissed_booking_id`
    # it names no row, so it drops *every* booking created before it — which is what
    # "treat this number as new" actually means. Persisted for the same LangGraph
    # reason as the rest of this block.
    guest_reset_at: NotRequired[str | None]
    separate_booking_clarification_asked: NotRequired[bool]
    # The guest confirmed S57's question: a new booking is being built while the
    # confirmed stay still stands. Persisted (see the BookingSession column), read
    # by `context_loader` every turn and passed to `resolve_guest_stage`, which
    # then resolves the guest as a lead and carries the old stay as
    # `standing_booking` instead of letting its row own the stage.
    separate_booking_active: NotRequired[bool]
    # When that latch was set, ISO-8601 UTC, so `context_loader` can expire it.
    # Persisted beside the flag it bounds (see the BookingSession column).
    separate_booking_started_at: NotRequired[str | None]
    # The confirmed stay a `separate_booking_active` guest still holds, carried
    # beside the draft being built. Recomputed from the booking rows on every load,
    # so it is not persisted — a stored copy could only disagree with the live
    # document, the same reasoning as `active_booking`.
    standing_booking: NotRequired[dict | None]
    is_separate_booking_inquiry: NotRequired[bool]
    feedback_recorded: NotRequired[bool]
    # Read by the stage handlers but declared nowhere until now. An undeclared
    # key is dropped by LangGraph at the node boundary and reads back as None,
    # so the in-stay welcome and the payment claim were both re-sent/forgotten
    # every turn. The first three have a BookingSession column (they persist);
    # `simulated_now` and `explicit_booking_stage` are test-only overrides of
    # the clock and the stage, and are deliberately turn-scoped.
    checkin_welcome_sent: NotRequired[bool]
    payment_proof_submitted: NotRequired[bool]
    simulated_now: NotRequired[datetime | None]
    explicit_booking_stage: NotRequired[str | None]
    missing_fields: list[str]
    updated_fields: list[str]
    modified_fields: NotRequired[list[str] | None]
    previous_values: NotRequired[dict | None]
    requested_values: NotRequired[dict | None]
    booking_change_summary: NotRequired[str | None]
    modification_metadata: NotRequired[dict | None]
    modification_candidate_state: NotRequired[dict | None]  # Pre-validated shadow state for deferred apply
    room_type_not_available: bool
    force_new_search: bool  # Clear quote trap; search for new dates/availability
    abandoned_draft: bool  # User declined continue → session cleared this turn

    # ── Room search ──────────────────────────────────────────────
    available_rooms: list[dict]
    allocation_options: list[dict]
    # T2/T3 paging. allocation_options is always the page currently rendered; these three let a
    # later turn re-render or advance that page without re-running the search engine.
    room_options_all: NotRequired[list[dict]]  # full ranked catalogue, capped at MAX_OPTIONS_PERSISTED
    room_option_offset: NotRequired[int]       # offset of the page currently displayed
    room_options_key: NotRequired[str]         # quote fingerprint that room_options_all belongs to
    # "MORE" was tapped this turn — advance the page rather than re-running the search.
    # This MUST be declared here even though it is set and consumed within a single turn.
    # intent_overrides sets it, understand merges it into its own return value, and
    # room_searcher reads it — three different nodes. A key absent from this class is
    # dropped as the update crosses a node boundary, silently: no error, no log line, the
    # consumer just reads None. Undeclared, this made the paging guard in room_searcher.py
    # unreachable, so MORE re-ran the whole search and re-rendered page 1 forever.
    show_more_combinations: NotRequired[bool]
    # B1. True once this quote's photos have gone out. Keyed against room_options_key, so a
    # genuinely new quote (dates or pax changed) shows photos again while a re-render of the same
    # quote does not. This cannot be an in-turn signal: `wants_photos` is cleared by validator
    # before search_rooms runs, and never reaches the node that attaches the cards.
    quote_presented: NotRequired[bool]
    room_previews_sent: NotRequired[bool]
    quote_photos_sent: NotRequired[bool]

    # ── Pricing ──────────────────────────────────────────────────
    price_breakdown: dict | None
    total_price: float | None
    base_total_price: NotRequired[float | None]
    discount_amount: NotRequired[float]
    discount_reason: NotRequired[str | None]
    negotiated_discount_pct: NotRequired[float | None]
    # A staff price override. context_loader reads it off the session and
    # resolve_payable_override may set it; until it was declared and mapped, a
    # value set here was read in-turn and then thrown away on save.
    price_override: NotRequired[float | None]

    # ── Payment & Booking ────────────────────────────────────────
    payment_link: str | None
    payment_link_id: str | None  # Razorpay plink_... for verify-payment
    # The Razorpay locked-amount UPI QR behind the current hold. `qr_order_id`
    # (ord_...) is what `/verifyqr_guest` takes and what the qr webhook's notes
    # echo back; `qr_id` (qr_...) is Razorpay's own handle for the same code.
    # Declared here because an undeclared key is dropped at the node boundary —
    # write them in `payment/link.py`, read them in `payment/status.py`, and the
    # turn in between loses nothing.
    qr_order_id: NotRequired[str | None]
    qr_id: NotRequired[str | None]
    qr_link: NotRequired[str | None]
    payment_status: str | None
    booking_id: str | None
    # The `PAY-…` key behind the current gateway order and inventory hold, when
    # the booking id itself has not been minted yet. See
    # `BOOKING_ID_ON_PAYMENT_PROOF_ENABLED`.
    payment_reference: NotRequired[str | None]
    # Set on the confirmation turn when we had no name to print. The next message
    # is read as the name for the booking slip — see NAME_AFTER_PAYMENT_ENABLED
    # and `name_after_payment.py`. Only a message that parses as a name is
    # claimed by it, so a guest who asks something else is answered normally.
    name_after_payment_pending: NotRequired[bool | None]
    payment_details: NotRequired[dict | None]
    payment_missing_fields: NotRequired[list[str] | None]
    payment_followup_index: NotRequired[int]
    price_match_data: NotRequired[dict | None]
    partial_payment_amount: float | None   # Amount the user wants to pay now
    amount_paid: float                     # Total amount already paid
    amount_due: float                      # Remaining balance
    payment_link_amount: float | None      # Amount on the current payment link

    # ── RAG ──────────────────────────────────────────────────────
    rag_context: str | None

    # ── Escalation ───────────────────────────────────────────────
    escalation_required: bool
    escalation_issue_type: str
    escalation_reason_code: str
    escalation_urgency: str
    escalation_sentiment: str
    escalation_reason: str
    escalation_confidence: float
    escalation_failure_count: int
    # B3b. escalation_failure_count only counts UNKNOWN intents (understand.py:1021), so it
    # never saw the reported loops — every turn in them was classified correctly. These two
    # count repetition instead, which is what actually happened: the same reply, again.
    last_response_hash: NotRequired[str]   # fingerprint of the reply sent last turn
    repeat_response_count: NotRequired[int]  # consecutive identical replies; routing escalates at 3

    # ── Hotel Task Classification ────────────────────────────────
    task_is_request: bool
    task_type: str
    task_subtype: str
    task_urgency: str
    task_confidence: float
    task_summary: str

    # ── Room Service ─────────────────────────────────────────────
    room_service_extraction: dict | None

    # ── Amenity Query ────────────────────────────────────────────
    amenity_query: dict | None

    # ── Lifecycle ────────────────────────────────────────────────
    # lifecycle_action_pending tracks a destructive action requiring guest confirmation.
    # Valid values: "start_fresh", "stay_extension", "early_checkout"
    lifecycle_action_pending: NotRequired[str | None]
    lifecycle_action_data: NotRequired[dict | None]  # Extra context (new_check_out, extra_nights, etc.)

    # ── Output ───────────────────────────────────────────────────
    bot_response: list[dict]
    next_action: str                     # Used for conditional routing

    # ── In-turn keys that were read before they were declared ─────
    # Every one of these was already being read (and mostly written) by nodes
    # while absent from this class, so LangGraph dropped them crossing a node
    # boundary and the consumer silently read None. They are listed here rather
    # than left implicit; EPHEMERAL in app/models/session_contract.py says why
    # each does not persist.
    assumptions: NotRequired[list[str]]
    booking_status: NotRequired[str]
    bot_response_prepend: NotRequired[list[dict]]
    check_in_date: NotRequired[str | None]
    check_out_date: NotRequired[str | None]
    extraction_result: NotRequired[dict | None]
    payment_turn_verdict: NotRequired[str | None]
    group_lead_submitted: NotRequired[bool]
    meal_plan: NotRequired[str | None]
    multi_booking_detected: NotRequired[bool]
    num_nights: NotRequired[int]
    quote_offset: NotRequired[int]
    reset_booking: NotRequired[bool]
    room_selection_resolved: NotRequired[bool]
    selected_option_label: NotRequired[str | None]
    session_updated_at: NotRequired[Any]
    wants_photos: NotRequired[bool]
