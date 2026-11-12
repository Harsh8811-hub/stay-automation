"""
Task Handler Node — LangGraph node for hotel operational tasks.

This node orchestrates the full task flow:
1. Receives state with Gemini's task classification
2. Calls the task engine for processing
3. Returns appropriate bot response cards

Completely separate from escalation_handler (support tickets).
Support Tickets → customer support & human escalation.
Hotel Tasks → operational work performed by hotel departments.
"""

import logging

from app.prompts.templates import text_card
from app.services.task_engine import process_task_request
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def handle_task(state: ConversationState) -> ConversationState:
    """
    Main task handler node.

    Routes through the task engine which:
    - Detects emergencies (→ CRITICAL support ticket)
    - Verifies booking existence and status
    - Detects duplicate tasks (→ append to existing)
    - Routes to the correct department
    - Resolves priority
    - Creates task in MongoDB
    """
    user_message = state.get("user_message", "")
    task_type = state.get("task_type", "OTHER")
    task_subtype = state.get("task_subtype", "OTHER")

    logger.info(
        f"Task handler: type={task_type}, subtype={task_subtype}, "
        f"message='{user_message[:80]}...'"
    )

    # Run the task processing engine
    result = await process_task_request(dict(state))

    action = result.get("action", "")
    response_text = result.get("response_text", "")

    if not response_text:
        response_text = (
            "I'm here to help with your request! "
            "Could you please provide more details? 😊"
        )

    bot_response = text_card(response_text)

    updates: dict = {
        **state,
        "bot_response": bot_response,
        "next_action": "task_handled",
    }
    if action in ("task_created", "task_updated") and state.get("booking_stage") in ("new", "collecting", None):
        updates["booking_stage"] = "in_house"
        updates["lifecycle_stage"] = "in_house"

    if result.get("task_id"):
        updates["last_task_id"] = result["task_id"]

    if task_subtype == "PAYMENT_CONFIRMATION" and action != "booking_invalid":
        updates["payment_status"] = "review_needed"
        updates["next_action"] = "payment_pending"

    return updates
