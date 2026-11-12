"""LangGraph conditional routing helpers."""

from app.services.confirmed_guard import guest_should_not_be_sold_to
from app.services.repetition_guard import REPEAT_ESCALATION_THRESHOLD
from app.utils.intent_overrides import resolve_negotiation_flag
from app.utils.person_name import looks_like_a_name_answer
from app.workflows.state import ConversationState

#: Intents that end in a room card, a rate, or a booking flow. A guest we already
#: know is confirmed must not reach any of them, whatever this turn's resolved
#: stage said — see `app/services/confirmed_guard.py`. Listed rather than derived
#: so a new selling intent has to be added here deliberately.
_SELL_CAPABLE_INTENTS = frozenset({
    "SELECTED_ROOM_DETAILS",
    "BOOKING_SUMMARY",
    "ROOM_FEATURE_LOOKUP",
    "SHOW_PRICES",
    "BOOK_ROOM",
    "CHECK_AVAILABILITY",
    "SELECT_ROOM",
    "CONFIRM_BOOKING",
})

#: Intents representing hotel information, amenities, policies, and social courtesies.
#: For guests awaiting payment or ID verification, these must be answered directly
#: from the grounded FAQ rather than trapped in a stage template.
_DIGRESSION_FAQ_AND_SOCIAL_INTENTS = frozenset({
    "FAQ",
    "DIGRESSION_FAQ",
    "AMENITY_QUERY",
    "SHOW_AMENITIES",
    "SHOW_LOCATION",
    "SHOW_PRICES",
    "SHOW_CHECKIN_POLICY",
    "ROOM_FEATURE_LOOKUP",
    "GREETING",
    "SOCIAL",
    "SMALL_TALK",
    "THANK_YOU",
})

#: Intents that have their own destination and must not be answered with the
#: "front desk is verifying your payment" paragraph, even while a payment is
#: pending. Each one asserts something the payment-check handler cannot honour:
#: that the guest wants out (cancel / reset), that something is wrong
#: (complaint), that they want a person (human request), or that the booking
#: itself should change (modification).
#:
#: Deliberately narrow. Selling intents (BOOK_ROOM, CHECK_AVAILABILITY,
#: SELECT_ROOM) are NOT here: a guest still holding an unverified booking being
#: walked into a fresh room flow is the double-sale product reported, so those
#: stay in payment_check.
_PAYMENT_CHECK_EXEMPT_INTENTS = frozenset({
    "CANCEL_BOOKING",
    "MODIFY_BOOKING",
    "COMPLAINT",
    "HUMAN_REQUEST",
    "LIFECYCLE_RESET",
})


def _payment_check_routing_enabled() -> bool:
    """Read lazily: `routing` is imported at graph build time, before settings load."""
    from app.config.settings import get_settings

    return bool(getattr(get_settings(), "PAYMENT_CHECK_INTENT_ROUTING_ENABLED", True))


def _show_prices_party_routing_enabled() -> bool:
    """Same lazy read, same reason. See the flag's comment in settings.py."""
    from app.config.settings import get_settings

    return bool(getattr(get_settings(), "SHOW_PRICES_PARTY_ROUTING_ENABLED", True))


def _prices_with_a_party(state: ConversationState) -> bool:
    """Is this a "how much?" turn that already knows who is staying?

    Read from the session, not from this turn's `updated_fields` — that list is
    emptied for every non-booking intent (understand.py:1559), so on a
    `SHOW_PRICES` turn it is always empty and a guard written against it would
    never fire. The state still carries what the guest gave on an earlier turn,
    which is exactly the information the menu handler throws away.

    Four conditions, all deliberate:

    * Both dates must be known. Without them the booking flow can only ask for
      them, which is a worse reply than the catalogue.
    * A party must be known. Same reason: without one the best the flow can do
      is ask "how many guests?".
    * No card may already be on screen. Options or selected rooms mean the
      guest has something to reply to, and re-running the search would replace
      it underneath them.
    * It is `SHOW_PRICES` only. Every other intent that lands in `menu` —
      location, amenities, existing booking, check-in policy — has nothing to
      do with a dated quote.

    A guest who is already confirmed never reaches here; the lifecycle and
    `guest_should_not_be_sold_to` checks above run first.
    """
    if not _show_prices_party_routing_enabled():
        return False

    if state.get("allocation_options") or state.get("selected_rooms"):
        return False

    if not (state.get("check_in") and state.get("check_out")):
        return False

    return bool(state.get("adults") or state.get("children"))


def route_after_gate(state: ConversationState) -> str:
    """Route after feature gate: blocked → save, allowed → intent router."""
    if state.get("next_action") == "blocked":
        return "blocked"
    return "allowed"


def route_by_intent(state: ConversationState) -> str:
    """Route to the appropriate handler based on detected intent."""
    if state.get("llm_unavailable"):
        return "unknown"

    intent = state.get("intent", "UNKNOWN")
    booking_stage = state.get("booking_stage", "new")

    escalation_type = state.get("escalation_issue_type", "NORMAL")
    if escalation_type in ("MEDICAL", "SAFETY", "SECURITY", "LEGAL"):
        return "escalation"

    # The confirmation turn asked for the name to print on the slip; this is the
    # answer. Claimed only when the message actually parses as a name, so a
    # guest who replies with a question instead is answered normally and is
    # never trapped in the prompt. Escalation still outranks it.
    if (
        state.get("name_after_payment_pending")
        and not state.get("escalation_required")
        and looks_like_a_name_answer(state.get("user_message") or "")
    ):
        return "name_after_payment"

    # ── Canonical Stage Enforcements (Sir's Spec) ─────────────────────────
    # Booking stage controls the bot's behavior, NOT the guest's latest message.
    lifecycle_stage = state.get("lifecycle_stage") or state.get("booking_stage")
    user_msg_clean = (state.get("user_message") or "").strip().lower()

    # Detect payment proof / payment claims in Stage 1 to transition to Stage 1b
    is_payment_claim = (
        state.get("payment_proof_submitted")
        or any(p in user_msg_clean for p in ("maine payment kar diya", "payment kar diya", "payment done", "already paid", "paid rs", "transfer kar diya", "upi kar diya"))
    )
    if is_payment_claim and (lifecycle_stage in ("lead", "payment_check", "payment_pending", "new", "collecting", "rooms_shown", "quote_shown", "awaiting_payment") or booking_stage in ("payment_check", "payment_pending", "awaiting_payment")):
        return "stage_payment_check"

    is_pending_verification = (
        lifecycle_stage in ("payment_check", "payment_pending", "id_verification")
        or booking_stage in ("payment_check", "payment_pending", "id_verification")
    )
    if is_pending_verification:
        # P0 Fix: If intent is in {DIGRESSION_FAQ, SOCIAL} and not a payment claim,
        # intent router runs BEFORE stage templates so the guest gets a grounded FAQ answer
        # plus a one-line stage reminder instead of the payment-verification template.
        #
        # The second half of the same idea (Prit, 2026-10-27): an intent with its
        # own destination — cancel, modify, complain, ask for a person, reset —
        # used to be answered with the payment paragraph anyway, because only the
        # FAQ/social list was excused. The FAQ answer was fixed and the register
        # of "requests that are not about the payment" was never widened.
        exempt = _DIGRESSION_FAQ_AND_SOCIAL_INTENTS
        if _payment_check_routing_enabled():
            exempt = exempt | _PAYMENT_CHECK_EXEMPT_INTENTS
        if not is_payment_claim and intent in exempt:
            pass  # Fall through to intent routing below
        else:
            return "stage_payment_check"

    if lifecycle_stage == "confirmed":
        return "stage_confirmed"

    if lifecycle_stage == "in_house":
        return "stage_in_house"

    if lifecycle_stage == "checked_out":
        # A checked-out guest who answers with a new stay's dates is booking
        # again, not talking about the stay that just ended. This branch used to
        # send every checked-out turn to the handler, whose enquiry route
        # answers "kis date par check-in karna chahenge?" — because the latch
        # that dismisses the old booking row only takes effect on the *next*
        # turn, and the handler was written to trade one wasted question for
        # that simplicity. When the guest has already supplied the dates there
        # is nothing to ask: measured 2026-10-28, "I want a room from 5 Nov to
        # 7 Nov for 2 adults" parsed both dates, and the reply asked for them
        # again. See DEFECTS item 2.
        #
        # Gated on the dates having *changed this turn*, not merely being
        # present: `context_loader` copies the ended stay's check-in/check-out
        # into the state, so a booking intent without new dates would otherwise
        # quote the old stay back to the guest. A guest who re-types the same
        # dates as their last stay falls through to the handler and is asked —
        # a worse reply, but a safe one.
        if (
            intent in (_SELL_CAPABLE_INTENTS | {"SHOW_PRICES_FOR_DATES"})
            and {"check_in", "check_out"} & set(state.get("updated_fields") or ())
        ):
            return "booking_flow"
        return "stage_checked_out"

    # If there's a pending lifecycle action awaiting confirmation, always
    # route back to lifecycle_handler regardless of intent.
    if state.get("lifecycle_action_pending"):
        return "lifecycle"

    if state.get("next_action") in ("lead_created", "need_custom_rooms", "opted_out"):
        return "greeting"

    if booking_stage == "cancellation_pending" and intent != "FAQ":
        return "cancel_booking"

    if intent == "HUMAN_REQUEST":
        return "escalation"

    if intent == "ESCALATE":
        return "escalation"

    if state.get("escalation_required"):
        return "escalation"

    if state.get("escalation_failure_count", 0) >= 3:
        return "escalation"

    # A pick is progress, not a loop. The repeat guard exists to break a guest
    # out of a card they keep being sent back, but it runs before intent
    # dispatch, so three identical cards followed by a perfectly good
    # "Option 1" went to escalation instead of being resolved. That is the
    # whole of guest 919088198242's conversation (2026-10-19 11:33): the fast
    # path recognised the pick, then the escalation handler claimed the turn
    # and opened SUP-20261019-000002.
    #
    # Only a selection with live options on screen is excused, and only the
    # selection intent: a repeated BOOK_ROOM with no card in front of the guest
    # is still someone who needs a human, and that is the case the guard was
    # written for. If the pick is valid the guest moves on and the reply
    # changes, so the counter drops to 1 at save_state. If the validator cannot
    # use it, the same card goes out again and the counter rises to 4 there —
    # the next turn still escalates. The loop stays bounded, not unguarded.
    if intent == "SELECT_ROOM" and state.get("allocation_options"):
        state = {**state, "repeat_response_count": 0}

    if state.get("repeat_response_count", 0) >= REPEAT_ESCALATION_THRESHOLD:
        return "escalation"

    if intent in ("GREETING", "SOCIAL", "SMALL_TALK"):
        return "greeting"

    if intent == "QUEUE_MODIFICATION":
        return "greeting" # We just route it to generate_response which is what "greeting" routes to in graph.py

    if intent == "SHOW_CHANGE_MENU":
        return "greeting"

    if intent == "COMPLAINT":
        return "escalation"

    if intent == "HOTEL_TASK":
        return "hotel_task"

    if intent == "ROOM_SERVICE_ORDER":
        return "room_service"

    if intent == "AMENITY_QUERY":
        return "amenity_query"

    if intent == "THANK_YOU":
        return "thank_you"

    if intent in ("FAQ", "DIGRESSION_FAQ"):
        return "faq"

    # Defence in depth for the zero-selling rule. The lifecycle check above is the
    # primary gate and is what normally catches a confirmed guest; this catches the
    # turns where stage resolution did not say "confirmed" even though we already
    # know the guest is — a CRM lookup that timed out, a session stage written
    # before the confirmation landed, a revision older than the CRM fix. Without
    # it those turns fall through the ladder below and quote a stay the guest
    # already holds. See `app/services/confirmed_guard.py` for the measurement.
    if intent in _SELL_CAPABLE_INTENTS and guest_should_not_be_sold_to(state):
        return "stage_confirmed"

    if intent in ("SELECTED_ROOM_DETAILS", "BOOKING_SUMMARY", "ROOM_FEATURE_LOOKUP"):
        return "room_query"

    # A "send me the room rent" that arrives with its own dates and party is a
    # quote request, and `menu` cannot answer it: the menu handler renders the
    # catalogue, whose cards describe a single room's capacity ("2 Adults, 2
    # Children"), never the party that is actually asking. Measured 2026-10-28
    # on guest 917980021370 — 9 adults and 1 child over 25-27 Dec, answered
    # three times with "2 Adults, 2 Children" room cards. See
    # `SHOW_PRICES_PARTY_ROUTING_ENABLED` in settings.py.
    if intent == "SHOW_PRICES" and _prices_with_a_party(state):
        return "booking_flow"

    if intent in ("SHOW_PRICES", "SHOW_LOCATION", "SHOW_AMENITIES", "CHECK_EXISTING_BOOKING", "SOMETHING_ELSE", "SHOW_CHECKIN_POLICY"):
        return "menu"

    if intent == "PAYMENT_STATUS":
        return "payment"

    if intent in ("BOOK_ROOM", "CHECK_AVAILABILITY", "SELECT_ROOM", "BOOKING_DATES_PROVIDED"):
        return "booking_flow"
        
    if intent == "MODIFY_BOOKING":
        return "handle_modification"

    if intent == "CANCEL_BOOKING":
        return "cancel_booking"

    if intent == "CONFIRM_BOOKING":
        # Shared with `validator`, through one function so the two cannot drift.
        # That function prefers this turn's LLM verdict (CONFIRM/CHANGE) and
        # falls back to the word-bounded regex when the model abstained.
        is_negotiation_or_change = resolve_negotiation_flag(
            state, state.get("user_message") or ""
        )
        if is_negotiation_or_change:
            return "booking_flow"
        if (booking_stage in ("quote_shown", "awaiting_payment") or state.get("total_price") or state.get("selected_rooms")) and state.get("guest_name"):
            return "payment"
        return "booking_flow"

    if intent == "PRICE_MATCH":
        # A guest who supplies their dates *while* asking for a discount is
        # asking to be priced, and the answer to that is a room card, not a
        # policy paragraph about screenshots. The handler can only ever reply
        # from what it already holds — it has no route to a room search — so
        # sending it a turn whose dates were parsed this very message leaves
        # the dates unused and the guest asked for them again. That is
        # DEFECTS item 2: measured 2026-10-28 on `discount-request`, state
        # held check_in/check_out/adults and `missing_fields` was empty,
        # while the reply said "let me know your travel dates".
        #
        # Deliberately narrow, in the same shape as the `checked_out` branch
        # above: only turns that *change* the dates are diverted, and never
        # when a screenshot came with the message — a price-match submission
        # must still reach `validate_price_match` or matching stops working
        # altogether. A bare "can you give me a discount?" or a repeat of the
        # same dates still goes to the handler.
        if (
            not state.get("price_match_data")
            and not state.get("total_price")
            and {"check_in", "check_out"} & set(state.get("updated_fields") or ())
        ):
            return "booking_flow"
        return "price_match"

    if intent == "LIFECYCLE_RESET":
        return "lifecycle"

    if intent == "SILENT_DOCUMENT_UPLOAD":
        return "silent_upload"

    if intent == "BANK_DETAILS":
        return "bank_details"

    return "unknown"


def route_by_validation(state: ConversationState) -> str:
    """Route based on validation results."""
    next_action = state.get("next_action", "")

    if next_action in ("need_dates", "need_guests", "invalid_dates", "invalid_room_selection"):
        return "need_info"

    if next_action == "search_rooms":
        return "search"

    if next_action == "show_quote":
        return "quote"

    if next_action == "ready_for_payment":
        return "payment"

    return "need_info"


def route_after_search(state: ConversationState) -> str:
    """Route after room search."""
    next_action = state.get("next_action", "")

    if next_action == "no_rooms":
        return "no_rooms"

    if next_action == "rooms_shown":
        return "done"

    if next_action == "show_quote":
        return "quote"

    return "done"


def route_after_quote(state: ConversationState) -> str:
    """Route after a quote.

    One case only: a multi-stay request that has just priced its first stay
    asks for the *next* stay's room card instead of ending the turn. That node
    sets `next_action = "search_rooms"`; every other quote ends the turn as it
    always has. The gate is the state value, not a time or a feature flag, so a
    mis-set flag cannot send an ordinary quote into a search.
    """
    if state.get("next_action") == "search_rooms":
        return "search"
    return "done"


def route_after_payment(state: ConversationState) -> str:
    """Route after payment handling."""
    next_action = state.get("next_action", "")

    if next_action == "payment_verified":
        return "confirm"

    return "done"
