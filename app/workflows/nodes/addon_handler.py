"""
Addon Handler Node — handles post-booking addon upselling & management.
"""

import logging
import asyncio
from app.workflows.state import ConversationState
from app.prompts.journey_templates import get_template
from app.prompts.templates import text_card
from app.services.extra_services import extra_service_manager

logger = logging.getLogger(__name__)

async def handle_addon(state: ConversationState) -> ConversationState:
    """
    Triggered when a booking is confirmed or during post-booking add-on requests.
    Dynamically loads hotel upsell items, verifies availability/lead times, and renders interactive add-on cards.
    """
    hotel_id = state.get("hotel_id", 0)
    user_msg = state.get("user_message", "")
    booking_stage = state.get("booking_stage", "confirmed")

    # Fetch post-booking available add-ons for this hotel
    services = await extra_service_manager.get_services(hotel_id=hotel_id, stage="post_booking", active_only=True)

    # If user message contains a direct selection or request, process via extra_service_manager
    if user_msg and any(s.name.lower() in user_msg.lower() or s.service_id in user_msg.lower() for s in services):
        handler_result = extra_service_manager.handle_message(
            user_message=user_msg,
            session_state={
                "pending_extra_services": state.get("pending_extra_services", []),
                "extra_services": state.get("extra_services", [])
            },
            payment_details=state.get("payment_details", {}),
            booking_stage=booking_stage
        )

        if handler_result.get("handled"):
            session_updates = handler_result.get("session_updates", {})
            payment_updates = handler_result.get("payment_updates", {})
            resp_text = handler_result.get("response", [{}])[0].get("text", "")

            # Record event
            asyncio.create_task(extra_service_manager.record_upsell_event(
                hotel_code=hotel_id,
                session_id=state.get("session_id", ""),
                upsell_item=user_msg,
                response="SELECTED",
                booking_id=state.get("booking_id")
            ))

            return {
                **state,
                "pending_extra_services": session_updates.get("pending_extra_services", state.get("pending_extra_services")),
                "bot_response": text_card(resp_text),
                "next_action": "addon_selected",
            }

    # Render post-booking addon catalog UI card if available
    if services:
        options = []
        buttons = []
        for srv in services[:4]:
            price_val = srv.default_price or 0.0
            options.append({
                "service_id": srv.service_id,
                "name": srv.name,
                "description": srv.description,
                "category": srv.category,
                "price": price_val,
                "custom_pricing": srv.custom_pricing,
                "image_url": srv.image_url
            })

            label_str = f"Add {srv.name} (₹{price_val:,.0f})" if price_val > 0 and not srv.custom_pricing else f"Add {srv.name}"
            buttons.append({
                "label": label_str,
                "action": "postback",
                "value": f"add addon: {srv.name}"
            })

        addon_card = {
            "type": "addon_catalog",
            "text": "✨ **Enhance Your Upcoming Stay**: Select an add-on service to customize your experience:",
            "data": options,
            "buttons": buttons
        }

        # Log impression
        asyncio.create_task(extra_service_manager.record_upsell_event(
            hotel_code=hotel_id,
            session_id=state.get("session_id", ""),
            upsell_item="POST_BOOKING_CATALOG",
            response="SHOWN",
            booking_id=state.get("booking_id")
        ))

        return {
            **state,
            "bot_response": [addon_card, text_card("Let us know if you'd like to include any of these services before your arrival!")[0]],
            "next_action": "addon_offered",
        }

    # Fallback to standard journey template
    return {
        **state,
        "bot_response": get_template("CFM_ADDON"),
        "next_action": "addon_offered",
    }

