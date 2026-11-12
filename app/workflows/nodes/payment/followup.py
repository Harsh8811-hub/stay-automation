"""Ask for missing payment-receipt fields one at a time."""

from typing import cast

from app.prompts.templates import text_card
from app.workflows.nodes.payment.constants import SUCCESS_STATUSES, get_followup_question
from app.workflows.state import ConversationState


def try_payment_followup(
    state: ConversationState,
    payment_details: dict,
) -> tuple[ConversationState, ConversationState | None]:
    """
    Auto-fill / ask missing receipt fields.

    Returns (updated_state, early_return).
    If early_return is not None, the caller should return it immediately.
    """
    payment_missing = list(state.get("payment_missing_fields") or [])
    followup_idx = state.get("payment_followup_index", 0)

    if not payment_missing:
        return state, None

    status = payment_details.get("status", "").upper()
    if status not in SUCCESS_STATUSES:
        return state, None

    # Auto-fill fields from session context
    session_fill = {
        "check_in": state.get("check_in"),
        "check_out": state.get("check_out"),
        "num_adults": state.get("adults"),
        "num_children": state.get("children"),
        "num_rooms": state.get("num_rooms"),
        "customer_name": state.get("guest_name"),
        "customer_phone": state.get("guest_phone") or state.get("customer_identifier"),
        "customer_email": state.get("guest_email"),
        "room_type": state.get("room_type_preference"),
    }
    for field, session_val in session_fill.items():
        if field in payment_missing and session_val is not None and session_val != "":
            payment_details[field] = session_val
            payment_missing.remove(field)

    from app.services.image_vision import CRITICAL_FIELDS, IMPORTANT_FIELDS
    askable = [f for f in payment_missing if f in CRITICAL_FIELDS or f in IMPORTANT_FIELDS]

    if askable:
        next_field = askable[0]
        question = get_followup_question(next_field)

        from app.prompts.templates import payment_receipt_summary
        if followup_idx == 0:
            bot_msg = payment_receipt_summary(payment_details) + text_card(
                f"I noticed a few details are missing from the receipt. Let me ask about them:\n\n{question}"
            )
        else:
            bot_msg = text_card(question)

        return state, cast(ConversationState, {
            **state,
            "payment_details": payment_details,
            "payment_missing_fields": payment_missing,
            "payment_followup_index": followup_idx + 1,
            "next_action": "payment_followup",
            "bot_response": bot_msg,
        })

    # All critical/important fields collected — clear and continue
    updated = cast(ConversationState, {
        **state,
        "payment_missing_fields": None,
        "payment_details": payment_details,
    })
    return updated, None
