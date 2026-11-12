"""
Amenity Query Handler Node — deterministic business logic + Gemini for natural text.

Flow:
  Guest → Gemini (intent + extraction) → AMENITY_QUERY → amenity_service.py (MongoDB)
  → Structured Result → Gemini (answer_amenity_query) → Guest

All schedules, pricing, availability, and access decisions come from MongoDB
through deterministic Python logic. Gemini only converts the structured result
into natural conversational text.
"""

import logging

from app.workflows.state import ConversationState
from app.services.amenity_service import handle_amenity_query
from app.services.gemini_service import answer_amenity_query
from app.prompts.templates import text_card

logger = logging.getLogger(__name__)


async def handle_amenity_node(state: ConversationState) -> ConversationState:
    """
    LangGraph node for AMENITY_QUERY intent.

    1. Extract amenity_name + query_type from Gemini's understanding output.
    2. Call amenity_service (Python business logic over MongoDB) for structured data.
    3. Pass structured data + user question to Gemini for natural text conversion.
    4. Return bot_response.
    """
    hotel_id = state.get("hotel_id", 0)
    user_message = state.get("user_message", "")
    amenity_query = state.get("amenity_query") or {}

    amenity_name = amenity_query.get("amenity_name")
    query_types = amenity_query.get("query_types") or ["general"]

    if not amenity_name:
        return {
            **state,
            "bot_response": text_card(
                "I'd be happy to help with information about our facilities! "
                "Could you please let me know which amenity you're asking about? "
                "For example, the pool, gym, spa, restaurant, etc."
            ),
            "next_action": "amenity_need_name",
        }

    # ── Step 1: Deterministic business logic (Python + MongoDB) ──
    structured_result = await handle_amenity_query(hotel_id, amenity_name, query_types)

    # Staff conversation truth wins over master amenity config for THIS chat only
    from app.services.override_resolver import resolve_amenity_override
    amenity_ov = resolve_amenity_override(state, amenity_name)
    if amenity_ov is not None:
        name = amenity_ov.get("name") or amenity_name
        if amenity_ov.get("available") is True:
            note = amenity_ov.get("note") or "Confirmed available by our team for your stay."
            structured_result = (
                f"Amenity: {name}\n"
                f"Status: AVAILABLE (staff-confirmed for this conversation)\n"
                f"Note: {note}\n"
                f"(Staff decision overrides hotel default config for this guest only.)\n"
                f"---\nHotel config reference:\n{structured_result}"
            )
        elif amenity_ov.get("available") is False:
            note = amenity_ov.get("note") or "Confirmed unavailable by our team."
            structured_result = (
                f"Amenity: {name}\n"
                f"Status: UNAVAILABLE (staff-confirmed for this conversation)\n"
                f"Note: {note}"
            )

    logger.info(
        f"Amenity query: hotel={hotel_id}, amenity={amenity_name}, "
        f"types={query_types}, result_len={len(structured_result)}"
    )

    # ── Step 2: Extract AI config for response generation ──
    ai_config = None
    config_dict = state.get("hotel_config")
    if config_dict and isinstance(config_dict, dict):
        ai_raw = config_dict.get("ai_config")
        if ai_raw:
            from app.config.hotel_config import AIPromptConfig
            ai_config = AIPromptConfig(**ai_raw)

    hotel_name = state.get("hotel_name", "our hotel")

    # ── Step 3: Gemini converts structured data → natural text ──
    if amenity_name == "all":
        answer = structured_result
    else:
        answer = await answer_amenity_query(
            user_question=user_message,
            structured_data=structured_result,
            hotel_name=hotel_name,
            ai_config=ai_config,
        )

    # Append booking resume note if applicable
    from app.utils.booking_resume import append_booking_resume
    answer = append_booking_resume(answer, state)

    return {
        **state,
        "bot_response": text_card(answer),
        "next_action": "amenity_answered",
    }
