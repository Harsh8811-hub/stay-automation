"""The button-tap ladder, out of `understand`.

Extracted from `app/workflows/nodes/understand.py` on 2026-10-28 (DEFECTS.md
#25). The statements are the original lines, in the same order, with the same
names and the same comments; the only edits are on the eleven `return`
statements, which now hand back `_meal_plan_from_button` as well as the reply.
Every returned dict is proved AST-identical to the one that was there before.

Why this one needed different handling. It is the only moved block that writes
outward in two directions:

  * `state["meal_plan_preference"] = plan_code` mutates the dict it was handed.
    That survives the move untouched, because it is the same object the caller
    holds.
  * `_meal_plan_from_button = True` sets a *local* of `understand` that is read
    ~2,700 lines later, where the model's extraction is folded into the turn.
    A local cannot cross a function boundary, so it is returned. The alternative
    — recomputing the condition at the call site — would make two places decide
    what "tapped a meal button" means, which is the drift this ladder has
    already been bitten by once.

The contract:
  * return `(reply, meal_plan_from_button)`; `reply` is the update dict to hand
    back to the graph, or `None` for "not a routed tap, carry on with the node";
  * no `await`.

The section comment below is the original one and still explains the ladder;
`resolve_button` is called here now, so the caller no longer needs it.
"""

from __future__ import annotations

import logging
from typing import Any

from app.utils.button_registry import MEAL_PLAN_BY_ACTION, TAP_ROUTED_ACTIONS, resolve_button

# Same logger object as the node, by name: importing it would be a cycle.
logger = logging.getLogger("app.workflows.nodes.understand")


def handle_tap_routed(
    state: dict[str, Any],
    user_msg_clean: str,
    active_prompt_action: str | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Answer a routed button tap. `(None, False)` means "not a routed tap"."""
    # ── ZERO-LLM FAST PATH: INTERACTIVE BUTTON TAPS ──
    #
    # A tap arrives as the button's *title*. The CRM forwards it as plain text and drops
    # `button_reply.id` (`combot_crm_v2/webhook_receiver/views.py:115-128`), so matching on
    # the title is the only thing that works on the live channel, and matching on the id is
    # the free upgrade if that ever changes. `resolve_button` tries both.
    #
    # WHICH titles mean what is NOT decided here. It is declared once in
    # `app.utils.button_registry`, which is also what `button_card` validates against — so a
    # template cannot send a button whose title this ladder would fail to recognise. That
    # pairing is the point: before the registry those two halves were maintained separately,
    # and four buttons had already drifted into doing nothing when tapped.
    # Set by the meal arm below, read ~2,700 lines down where the LLM's extraction is folded
    # into the turn. A tap on a meal button is a definite answer — it is the button we sent —
    # so the extractor must not be allowed to overwrite it. See the comment at the read site.
    _meal_plan_from_button = False

    _tapped = resolve_button(user_msg_clean)
    if _tapped is not None and _tapped.action in TAP_ROUTED_ACTIONS:
        action = _tapped.action
        logger.info(
            "UNDERSTAND_FAST_PATH | Button %s (id=%s) -> %s",
            user_msg_clean, _tapped.id, action,
        )

        if action == "SHOW_ROOMS":
            has_dates = bool(state.get("check_in") and state.get("check_out"))
            intent = "CHECK_AVAILABILITY" if has_dates else "SHOW_PRICES"
            return ({
                **state,
                "intent": intent,
                "next_action": "menu_handled" if intent == "SHOW_PRICES" else state.get("next_action"),
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "BOOK_NOW":
            # Straight into the booking flow. `next_action` is left empty on
            # purpose: the validator is what decides which slot to ask for first,
            # and pre-setting one here would skip its ordering.
            return ({
                **state,
                "intent": "BOOK_ROOM",
                "next_action": None,
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "CHECK_EXISTING_BOOKING":
            return ({
                **state,
                "intent": "CHECK_EXISTING_BOOKING",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "HUMAN_REQUEST":
            return ({
                **state,
                "intent": "HUMAN_REQUEST",
                "escalation_required": True,
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "PAYMENT_PAID":
            return ({
                **state,
                "intent": "PAYMENT_STATUS",
                "payment_proof_submitted": True,
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "CONFIRM_BOOKING":
            return ({
                **state,
                "intent": "CONFIRM_BOOKING",
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "CANCEL_BOOKING":
            return ({
                **state,
                "intent": "CANCEL_BOOKING",
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "CANCEL_ABORT":
            from app.prompts import templates
            return ({
                **state,
                "intent": "USER_ACKNOWLEDGMENT",
                # A second-booking guest has no `booking_id` — `context_loader`
                # drops it while the latch is on — so the old rule wrote "new"
                # and quietly moved their draft back to the start. Keep the
                # stage the draft is actually on; not "confirmed", which would
                # clear the latch from an unrelated acknowledgement.
                "booking_stage": (
                    (state.get("booking_stage") or "lead")
                    if state.get("separate_booking_active")
                    else ("confirmed" if state.get("booking_id") else "new")
                ),
                "bot_response": templates.text_card("Your booking remains active! Let me know if you need anything else."),
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "NUDGE_CONTINUE":
            from app.prompts import templates
            return ({
                **state,
                "intent": "USER_ACKNOWLEDGMENT",
                "bot_response": templates.text_card("Great! Please share your details or questions and we'll continue."),
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        if action == "NUDGE_CANCEL":
            from app.prompts import templates
            return ({
                **state,
                "intent": "USER_ACKNOWLEDGMENT",
                "bot_response": templates.text_card("No problem! Whenever you're ready, feel free to message us again. Have a great day!"),
                "extracted_data": None,
                "extraction_result": None,
            }, _meal_plan_from_button)

        plan_code = MEAL_PLAN_BY_ACTION.get(action)
        if plan_code is not None:
            state["meal_plan_preference"] = plan_code
            _meal_plan_from_button = True
            logger.info("UNDERSTAND_FAST_PATH | Meal button %s -> %s", user_msg_clean, plan_code)

            # The tap answered the meal prompt, so there is nothing left to interpret and
            # the turn is complete: re-price with the new plan. This arm used to fall
            # through for every case, on the argument that the AC qualifier further down
            # still had to run on the same message. It cannot: `active_prompt_action` holds
            # one value, and while the meal prompt is the active one the AC block (guarded
            # on `need_ac_preference`) never fires. What the fall-through did do is run the
            # message on into the room-selection ladder, where `validator.py:423` could not
            # read "Breakfast (CP)" as a room and returned `invalid_room_selection`,
            # overwriting the answer and re-quoting the room card. Measured 2026-10-15 —
            # tapping a meal plan stored `meal_plan_preference='CP'` and then asked the
            # guest for a room.
            #
            # Only taken when the meal prompt is genuinely the active one. A meal button
            # arriving against some other prompt still falls through exactly as before.
            if active_prompt_action == "need_meal_plan_preference":
                return ({
                    **state,
                    "meal_plan_preference": plan_code,
                    "intent": "BOOK_ROOM",
                    "next_action": "search",
                    "force_new_search": True,
                    "selected_rooms": [],
                    "allocation_options": [],
                    "price_breakdown": None,
                    "total_price": None,
                    "booking_stage": "collecting",
                    "updated_fields": ["meal_plan_preference"],
                }, _meal_plan_from_button)

    return (None, _meal_plan_from_button)
