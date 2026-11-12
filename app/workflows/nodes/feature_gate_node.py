"""
Feature Gate Node — Python-side business rule enforcement.

Handles lead-only mode, manual inquiry (directBooking=false), and limit checks.
"""

import logging

from app.config.hotel_config import HotelConfig
from app.prompts.templates import button_card, text_card
from app.services.feature_gate import FeatureGate
from app.services.stage_lock import apply_stage_lock
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def _response_with_hotel_contact(base_message: str, hotel_id: int) -> str:
    """
    Append live Hotel-Contact details so blocked/front-desk redirects
    are actionable (phone / email / WhatsApp from DB).
    """
    text = (base_message or "").rstrip()
    try:
        from app.services.ticket_service import get_formatted_contact_info

        contact = await get_formatted_contact_info(hotel_id)
    except Exception as e:
        logger.warning(f"Failed to load hotel contact for hotel_id={hotel_id}: {e}")
        return text

    fallback = "Please contact our front desk for assistance."
    if not contact or contact.strip() == fallback:
        return text

    # Avoid duplicating if the base message already includes the same numbers
    if contact.strip() in text:
        return text

    return (
        f"{text}\n\n"
        f"You can also contact our hotel directly:\n"
        f"{contact}"
    )


def _blocked_button_card(msg: str, state: ConversationState) -> list[dict]:
    is_hi = str(state.get("preferred_language") or state.get("conversation_language") or "").lower() in ("hinglish", "hindi", "hi")
    buttons = [
        {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
    ]
    return button_card(msg, buttons)


async def feature_gate(state: ConversationState) -> ConversationState:
    """
    Enforce business rules BEFORE routing to any handler.

    Handles three booking modes:
    1. leadOnlyMode=true → collect guest info, create lead (no rooms/quotes/payment)
    2. directBooking=false → allow browsing/quoting, block at confirmation (manual inquiry)
    3. directBooking=true → pass through unchanged

    Also enforces guest/room/same-day limits and feature-specific gates.
    """
    # Hold a settled stage before anything reads it.
    #
    # `understand` has several exits that rebuild the stage from scratch — the
    # fresh-start path and two ghost-booking guards that write `lead` when they
    # do not recognise the guest's booking id. Those exits never reach the
    # lifecycle invariant at the foot of `understand`, so until 2026-10-27 a
    # confirmed guest could be put back to `lead` mid-conversation and sold a
    # second booking (conversation 919739843511). This is the one node every
    # turn passes through between `understand` and the router, which is why the
    # repair lives here rather than in each of those branches.
    #
    # It can only hold a settled stage, never advance one, and a demotion the
    # guest asked for (`/start`, declining the draft) is released. See
    # `app/services/stage_lock.py`.
    state = apply_stage_lock(state)

    # Canonical Lifecycle Architecture (Sir's Spec):
    # Post-lead lifecycle stages (confirmed, in_house, checked_out, payment_check)
    # have their own dedicated stage handlers and must not be blocked by generic lead-flow gates.
    lifecycle_stage = state.get("lifecycle_stage") or state.get("booking_stage")
    if lifecycle_stage in ("payment_check", "confirmed", "in_house", "checked_out"):
        if state.get("next_action") == "blocked":
            return {**state, "next_action": None}
        return state

    config_dict = state.get("hotel_config")
    if not config_dict:
        return state

    config = HotelConfig(**config_dict)
    gate = FeatureGate(config, session_id=state.get("session_id"))
    intent = state.get("intent", "UNKNOWN")

    booking_intents = ("BOOK_ROOM", "SELECT_ROOM", "CHECK_AVAILABILITY", "CONFIRM_BOOKING")

    browsing_result = gate.check_booking_browsing()
    if not browsing_result.allowed and intent in booking_intents:
        logger.info(
            f"BOOKING_LEAD_MODE_ACTIVE | hotel_id={config.hotel_id} | "
            f"session_id={state.get('session_id')} | feature=leadOnlyMode | decision=LEAD_COLLECTION"
        )
        return await _handle_lead_collection(state, config)

    if intent == "CONFIRM_BOOKING" and not gate.is_lead_only_mode():
        confirm_result = gate.check_booking_confirmation()
        if not confirm_result.allowed:
            logger.info(
                f"BOOKING_BLOCKED_DIRECT_BOOKING_DISABLED | hotel_id={config.hotel_id} | "
                f"session_id={state.get('session_id')} | feature=directBooking | decision=MANUAL_INQUIRY"
            )
            return await _handle_manual_inquiry(state, config)

    result = gate.check_intent(intent)
    if not result.allowed:
        logger.info(f"Feature gate BLOCKED: intent={intent}, hotel_id={config.hotel_id}")
        msg = await _response_with_hotel_contact(
            result.response or "",
            state.get("hotel_id") or config.hotel_id,
        )
        return {
            **state,
            "next_action": "blocked",
            "bot_response": _blocked_button_card(msg, state),
        }

    if intent in booking_intents:
        adults = state.get("adults") or 0
        children = state.get("children") or 0
        num_rooms = state.get("num_rooms") or 1

        if adults > 0:
            guest_result = gate.validate_guest_count(adults, children)
            if not guest_result.allowed:
                msg = await _response_with_hotel_contact(
                    guest_result.response or "",
                    state.get("hotel_id") or config.hotel_id,
                )
                return {
                    **state,
                    "next_action": "blocked",
                    "bot_response": _blocked_button_card(msg, state),
                }

        if num_rooms > 1:
            room_result = gate.validate_room_count(num_rooms)
            if not room_result.allowed:
                msg = await _response_with_hotel_contact(
                    room_result.response or "",
                    state.get("hotel_id") or config.hotel_id,
                )
                return {
                    **state,
                    "next_action": "blocked",
                    "bot_response": _blocked_button_card(msg, state),
                }

        check_in = state.get("check_in")
        if check_in:
            from app.utils.helpers import today_str
            is_same_day = check_in == today_str()
            same_day_result = gate.validate_same_day_booking(is_same_day)
            if not same_day_result.allowed:
                msg = await _response_with_hotel_contact(
                    same_day_result.response or "",
                    state.get("hotel_id") or config.hotel_id,
                )
                return {
                    **state,
                    "next_action": "blocked",
                    "bot_response": _blocked_button_card(msg, state),
                }

    return state


async def _handle_lead_collection(state: ConversationState, config: HotelConfig) -> ConversationState:
    """Lead Only Mode: collect info and create a lead."""
    from app.repositories.lead_repository import (
        create_lead, generate_lead_id, find_existing_lead,
    )

    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")

    existing = await find_existing_lead(hotel_id, session_id)
    if existing:
        return {
            **state,
            "next_action": "blocked",
            "bot_response": text_card(
                "Thank you for your enquiry.\n\n"
                "Our team will contact you shortly with the best available rates "
                "and booking confirmation.\n\n"
                "Is there anything else I can help you with?"
            ),
        }

    missing_fields = []
    if not state.get("check_in"):
        missing_fields.append("check_in")
    if not state.get("check_out"):
        missing_fields.append("check_out")
    if state.get("adults") is None:
        missing_fields.append("adults")

    if missing_fields:
        if "check_in" in missing_fields or "check_out" in missing_fields:
            msg = (
                "I'd love to help you find the perfect room! Could you please share your "
                "**check-in** and **check-out** dates?"
            )
        else:
            msg = "How many **guests** will be staying? (adults and children)"

        return {
            **state,
            "next_action": "blocked",
            "booking_stage": "collecting",
            "bot_response": text_card(msg),
        }

    lead_id = await generate_lead_id(hotel_id)

    messages = state.get("messages_history", [])
    conversation_ref = [
        {"role": m["role"], "content": m["content"][:200]}
        for m in messages[-5:]
    ]

    lead_data = {
        "lead_id": lead_id,
        "hotel_id": hotel_id,
        "session_id": session_id,
        "customer_phone": state.get("customer_identifier"),
        "guest_name": state.get("guest_name"),
        "check_in": state.get("check_in"),
        "check_out": state.get("check_out"),
        "adults": state.get("adults"),
        "children": state.get("children") or 0,
        "num_rooms": state.get("num_rooms") or 1,
        "preferred_room": state.get("room_type_preference"),
        "special_requests": state.get("special_requests"),
        "lead_type": "LEAD_ONLY",
        "conversation_reference": conversation_ref,
    }

    await create_lead(lead_data)
    await _notify_hotel_new_lead(state, config, lead_id, lead_data)

    return {
        **state,
        "next_action": "blocked",
        "booking_stage": "lead_created",
        "bot_response": text_card(
            "Thank you for your enquiry.\n\n"
            "Our team will contact you shortly with the best available rates "
            "and booking confirmation.\n\n"
            "Is there anything else I can help you with?"
        ),
    }


async def _handle_manual_inquiry(state: ConversationState, config: HotelConfig) -> ConversationState:
    """directBooking=false: create a manual booking inquiry after quote confirmation."""
    from app.repositories.lead_repository import (
        create_lead, generate_lead_id, find_existing_lead,
    )

    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")

    existing = await find_existing_lead(hotel_id, session_id)
    if existing:
        return {
            **state,
            "next_action": "blocked",
            "bot_response": text_card(
                "Our team will contact you shortly to complete your booking "
                "and share the confirmation details.\n\n"
                "Is there anything else I can help you with?"
            ),
        }

    lead_id = await generate_lead_id(hotel_id)

    messages = state.get("messages_history", [])
    conversation_ref = [
        {"role": m["role"], "content": m["content"][:200]}
        for m in messages[-5:]
    ]

    lead_data = {
        "lead_id": lead_id,
        "hotel_id": hotel_id,
        "session_id": session_id,
        "customer_phone": state.get("customer_identifier"),
        "guest_name": state.get("guest_name"),
        "check_in": state.get("check_in"),
        "check_out": state.get("check_out"),
        "adults": state.get("adults"),
        "children": state.get("children") or 0,
        "num_rooms": state.get("num_rooms") or 1,
        "preferred_room": state.get("room_type_preference"),
        "special_requests": state.get("special_requests"),
        "selected_rooms": state.get("selected_rooms", []),
        "total_price": state.get("total_price"),
        "price_breakdown": state.get("price_breakdown"),
        "lead_type": "BOOKING_INQUIRY",
        "conversation_reference": conversation_ref,
    }

    await create_lead(lead_data)
    await _notify_hotel_new_lead(state, config, lead_id, lead_data)

    return {
        **state,
        "next_action": "blocked",
        "booking_stage": "inquiry_created",
        "bot_response": text_card(
            "Our team will contact you shortly to complete your booking "
            "and share the confirmation details.\n\n"
            "Is there anything else I can help you with?"
        ),
    }


async def _notify_hotel_new_lead(
    state: ConversationState,
    config: HotelConfig,
    lead_id: str,
    lead_data: dict,
) -> None:
    """Notify hotel staff about a new lead via CRM (never blocks lead creation)."""
    try:
        from app.services.crm_service import crm_service
        from app.config.settings import get_settings

        hotel_id = state.get("hotel_id", 0)
        guest_name = lead_data.get("guest_name") or "Guest"
        phone = lead_data.get("customer_phone") or "Unknown"
        check_in = lead_data.get("check_in") or "TBD"
        check_out = lead_data.get("check_out") or "TBD"
        adults = lead_data.get("adults") or 0
        room_pref = lead_data.get("preferred_room") or "Not specified"
        lead_type = lead_data.get("lead_type", "INQUIRY")

        settings = get_settings()
        notify_number = getattr(settings, "HOTEL_NOTIFICATION_PHONE", None)

        if notify_number:
            notification_text = (
                f"New Booking Lead [{lead_id}]\n\n"
                f"Type: {lead_type}\n"
                f"Guest: {guest_name}\n"
                f"Phone: {phone}\n"
                f"Dates: {check_in} to {check_out}\n"
                f"Guests: {adults} adults\n"
                f"Room: {room_pref}\n\n"
                f"Please follow up with the guest."
            )

            payload = {
                "hotelID": str(hotel_id),
                "phone_number": notify_number,
                "message_type": "text",
                "text": notification_text,
            }
            await crm_service._send_request(payload)
            logger.info(f"Hotel notified about lead {lead_id} via CRM")

    except Exception as e:
        logger.warning(f"Failed to notify hotel about lead {lead_id}: {e}")
