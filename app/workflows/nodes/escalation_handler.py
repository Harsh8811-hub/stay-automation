"""
Escalation Handler Node — LangGraph node for the escalation pipeline.

This node orchestrates the full escalation flow:
1. Receives state with Gemini's escalation classification
2. Calls the escalation engine for the decision
3. Returns appropriate bot response cards

Handles:
- Complaints (with knowledge verification first)
- Explicit human requests
- Safety/medical/security/legal emergencies
- Payment disputes requiring investigation
- Booking disputes requiring staff intervention
- Out-of-scope queries
- Repeated failure escalation
"""

import logging

from app.prompts.templates import text_card
from app.services.escalation_engine import evaluate_escalation
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def handle_escalation(state: ConversationState) -> ConversationState:
    """
    Main escalation handler node.
    
    Routes through the escalation engine which:
    - Checks if knowledge can answer the query first
    - Only creates tickets when human intervention is truly needed
    - Handles duplicate ticket prevention
    - Retrieves contact info dynamically
    """
    intent = state.get("intent", "UNKNOWN")
    issue_type = state.get("escalation_issue_type", "NORMAL")
    sentiment = state.get("escalation_sentiment", "NEUTRAL")
    user_message = state.get("user_message", "")
    
    logger.info(
        f"Escalation handler: intent={intent}, issue_type={issue_type}, "
        f"sentiment={sentiment}, message='{user_message[:80]}...'"
    )

    if state.get("escalation_reason_code") == "PHONE_CALL_HANDOFF" or state.get("escalation_reason") == "guest_called_staff":
        bot_response = state.get("bot_response") or text_card(
            "Understood! 🙏 We have noted that you recently called our team. Our staff will assist you directly. Bot replies are paused so our team can handle your request."
        )
        return {
            **state,
            "bot_response": bot_response,
            "next_action": "escalation_handled",
            "escalation_required": True,
        }
    result = await evaluate_escalation(dict(state))
    
    action = result.get("action", "no_escalation")
    
    if action == "no_escalation":
        # Shouldn't normally reach here, but handle gracefully
        # Generate empathetic response for complaints
        if intent == "COMPLAINT":
            from app.services.gemini_service import generate_natural_response
            from app.services.ticket_service import get_formatted_contact_info
            
            bot_text = await generate_natural_response(
                context=(
                    "The user is complaining or expressing dissatisfaction. "
                    "Apologize warmly, acknowledge their concern, and guide them "
                    "toward resolution. Do NOT offer refunds or specific compensation."
                ),
                action_result="Generated empathetic response.",
                user_message=user_message,
                hotel_id=state.get("hotel_id", 0),
                session_id=state.get("session_id", ""),
            )
            
            contact_info = await get_formatted_contact_info(state.get("hotel_id", 0))
            if contact_info and contact_info != "Please contact our front desk for assistance.":
                bot_text += f"\n\nIf you'd like to speak with our team directly:\n\n{contact_info}"
            
            bot_response = text_card(bot_text)
        else:
            from app.prompts.journey_templates import get_template
            bot_response = get_template("STAY_NEG")
    else:
        # Ticket was created, updated, or self-served
        response_text = result.get("response_text", "")
        if response_text:
            bot_response = text_card(response_text)
        else:
            from app.prompts.journey_templates import get_template
            bot_response = get_template("STAY_NEG")
    
    # Reset failure count on successful handling
    # (only if we actually handled it, not on "no_escalation" fallback)
    new_failure_count = state.get("escalation_failure_count", 0)
    if action != "no_escalation":
        new_failure_count = 0

    # Reset the repetition counter unconditionally, not on the same condition as above.
    # Routing reached this node *because* the guest was in a loop, and this node is the remedy;
    # leaving the count at 3 would re-escalate on every subsequent turn for the rest of the
    # session. If the handoff produces nothing useful the count simply rebuilds over the next
    # three turns, which is a bounded retry rather than a trap.
    repeat_count = 0

    # Preserve booking stage context for active bookings
    booking_stage = state.get("booking_stage", "new")

    return {
        **state,
        "bot_response": bot_response,
        "next_action": "escalation_handled",
        "escalation_failure_count": new_failure_count,
        "repeat_response_count": repeat_count,
    }
