import logging
from typing import cast

from app.workflows.state import ConversationState
from app.services.room_service_cart import process_cart_action

logger = logging.getLogger(__name__)

async def handle_room_service(state: ConversationState) -> ConversationState:
    """
    Handle room service cart operations based on Gemini's extraction.
    """
    extraction = state.get("room_service_extraction")
    if not extraction or not extraction.get("action"):
        # Fallback if no specific action extracted
        return cast(ConversationState, {
            **state,
            "bot_response": [{"type": "text", "text": "I can help you with room service! What would you like to order?"}]
        })
        
    action = extraction.get("action")
    items = extraction.get("items", [])
    notes = extraction.get("notes")
    
    # Extract session details
    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")
    booking_id = state.get("booking_id")
    customer_phone = state.get("guest_phone")
    
    # We might not have room number directly in state unless it's an existing booking, 
    # but let's pass what we can. Ideally it's fetched from booking if booking_id exists.
    room_number = None 
    
    try:
        result = await process_cart_action(
            hotel_id=hotel_id,
            session_id=session_id,
            action=action,
            items=items,
            notes=notes,
            booking_id=booking_id,
            room_number=room_number,
            customer_phone=customer_phone
        )
        
        bot_msg = result.get("message", "Cart updated.")
        
        return cast(ConversationState, {
            **state,
            "bot_response": [{"type": "text", "text": bot_msg}],
            "room_service_extraction": None # Clear it after processing
        })
        
    except Exception as e:
        logger.error(f"Error processing room service action: {e}", exc_info=True)
        return cast(ConversationState, {
            **state,
            "bot_response": [{"type": "text", "text": "Sorry, there was an issue processing your room service request. Please try again."}]
        })
