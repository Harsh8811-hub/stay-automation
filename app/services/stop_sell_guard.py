"""
Stop-Sell Guard — Enforces stop-sell rules and escalates insistent guest requests.

Why this module exists:
1. When hotel staff enables "Stop Sell" in Rate Inventory or Control Center
   (`daily_rates.stop_sell == True`, `daily_rates.is_available == False`, or
   `daily_inventory.status == "closed"`), the room must NOT be booked by the bot.
2. If a customer exclusively insists on that room (e.g. "I only want Deluxe Room",
   "Mujhe sirf deluxe room hi chahiye", "Can you please book Deluxe room only?",
   or rejects alternatives after being offered them):
   - The bot must ESCALATE to hotel staff immediately.
   - The bot must reassure the guest: "A hotel representative will contact you shortly".
   - A support ticket is created in `support_tickets` with reason `STOP_SOLD_ROOM_REQUEST`.
   - 3 AI Smart Replies are synced to the Control Center so staff can take 1-click action.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.models.schemas import RoomType
from app.prompts.templates import text_card
from app.repositories import room_repository
from app.services import inventory_service

logger = logging.getLogger(__name__)

# Patterns indicating strict customer exclusivity or insistence on a single room category
_EXCLUSIVE_ROOM_PATTERNS = re.compile(
    r"\b("
    r"only\b|"
    r"strictly\b|"
    r"exclusively\b|"
    r"alone\b|"
    r"just\s+(?:\w+\s+){0,2}(?:this|the|that)\b|"
    r"just\s+want\b|"
    r"no\s+other\b|"
    r"don'?t\s+want\s+(?:other|any\s+other|another)\b|"
    r"won'?t\s+take\s+(?:other|any\s+other|another)\b|"
    r"only\s+want\b|"
    r"want\s+only\b|"
    r"book\s+(?:\w+\s+){0,3}only\b|"
    # Hindi / Hinglish exclusivity patterns
    r"sirf\b|"
    r"bas\b|"
    r"khali\b|"
    r"yahi\b|"
    r"yehi\b|"
    r"dusra\s+nahi\b|"
    r"aur\s+koi\s+nahi\b|"
    r"koi\s+aur\s+nahi\b|"
    r"yahi\s+chahiye\b|"
    r"hi\s+chahiye\b|"
    r"hi\s+book\b|"
    r"hi\s+dena\b"
    r")",
    re.IGNORECASE,
)

_REJECTION_PATTERNS = re.compile(
    r"\b(?:no|nahi|not|dont\s+want|don't\s+want|na|never|neither|cancel|other\s+nahi)\b",
    re.IGNORECASE,
)


def is_room_exclusively_requested(user_message: str, state: dict[str, Any]) -> bool:
    """
    Detect whether the user is exclusively demanding a particular room type.
    
    Returns True if:
    1. The message contains explicit exclusivity keywords ('only', 'sirf', 'yahi chahiye', etc.).
    2. The previous turn showed alternatives because the preferred room was unavailable
       (`room_type_not_available` is True), and the user now rejects the alternatives
       or re-asserts their original preferred room.
    """
    msg = (user_message or "").strip()
    if not msg:
        return False

    # 1. Explicit exclusivity markers in message
    if bool(_EXCLUSIVE_ROOM_PATTERNS.search(msg)):
        return True

    # 2. Multi-turn insistence: previous turn showed alternatives because room was unavailable
    if state.get("room_type_not_available"):
        # Did the guest reject the offered alternatives?
        if bool(_REJECTION_PATTERNS.search(msg)):
            return True

        # Or did the guest repeat / reiterate their original room preference?
        pref = (state.get("room_type_preference") or "").strip().lower()
        if pref and len(pref) >= 3 and pref in msg.lower():
            return True

    return False


def _match_room_in_catalog(
    room_query: str,
    catalog: list[RoomType],
) -> RoomType | None:
    """Find a catalog room matching the user's room query or preference."""
    if not room_query or not catalog:
        return None

    cleaned_query = room_query.strip().lower()

    # 1. Direct name match
    for room in catalog:
        r_name = room.name.strip().lower()
        if cleaned_query == r_name or r_name in cleaned_query or cleaned_query in r_name:
            return room

    # 2. Match significant words (e.g. "deluxe", "suite", "executive", "family")
    query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", cleaned_query))
    query_tokens.discard("room")
    query_tokens.discard("rooms")
    query_tokens.discard("only")
    query_tokens.discard("sirf")
    query_tokens.discard("want")
    query_tokens.discard("book")

    best_room = None
    best_score = 0
    for room in catalog:
        r_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", room.name.lower()))
        common = query_tokens & r_tokens
        if common and len(common) > best_score:
            best_score = len(common)
            best_room = room

    return best_room


async def check_and_handle_stop_sell_escalation(
    state: dict[str, Any],
    preferred_room: str | None = None,
    check_in: str | None = None,
    check_out: str | None = None,
) -> dict[str, Any] | None:
    """
    Check if the customer exclusively wants a room that is currently Stop-Sold.
    If so, raises a ticket, syncs Control Center smart replies, and returns the
    escalated state with guest reassurance copy.
    
    Returns None if:
    - The customer is not exclusively insisting on this room.
    - The requested room is available or not stop-sold.
    """
    user_msg = state.get("user_message", "")
    if not is_room_exclusively_requested(user_msg, state):
        return None

    hotel_id = state.get("hotel_id", 0)
    if not hotel_id:
        return None

    ci = check_in or state.get("check_in")
    co = check_out or state.get("check_out")

    # Fetch stop-sold room IDs for these dates
    stopped_room_ids = await inventory_service.get_stop_sold_room_ids(hotel_id, ci, co)
    if not stopped_room_ids:
        return None

    # Resolve which room the guest is referring to
    base_rooms = await room_repository.get_all_rooms(hotel_id)
    if not base_rooms:
        return None

    room_pref = preferred_room or state.get("room_type_preference") or user_msg
    matched_room = _match_room_in_catalog(room_pref, base_rooms)
    if not matched_room:
        matched_room = _match_room_in_catalog(user_msg, base_rooms)

    if not matched_room:
        return None

    room_id = str(matched_room.room_id)
    if room_id not in stopped_room_ids:
        return None

    # Room is STOP-SOLD and guest exclusively insists on it -> ESCALATE!
    room_name = matched_room.name
    logger.info(
        "STOP_SELL_ESCALATION_TRIGGERED | hotel_id=%s | room_id=%s | room_name='%s' | user_msg='%s'",
        hotel_id,
        room_id,
        room_name,
        user_msg,
    )

    return await raise_stop_sell_escalation(
        state=state,
        room_id=room_id,
        room_name=room_name,
        check_in=ci,
        check_out=co,
    )


async def raise_stop_sell_escalation(
    state: dict[str, Any],
    room_id: str,
    room_name: str,
    check_in: str | None = None,
    check_out: str | None = None,
) -> dict[str, Any]:
    """
    Create support ticket, sync smart replies for Control Center,
    and return conversation state updates with reassurance copy.
    """
    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")
    user_message = state.get("user_message", "")
    customer_phone = state.get("guest_phone") or state.get("customer_identifier") or ""
    customer_name = state.get("guest_name") or ""

    dates_str = f"from {check_in} to {check_out}" if check_in and check_out else "for the selected dates"

    summary = (
        f"Guest insists exclusively on booking '{room_name}' {dates_str}. "
        f"This room category is currently marked STOP-SELL in rate inventory. "
        f"Staff action required: Contact guest at {customer_phone} to check if room can be released or accommodated manually."
    )

    ticket_id = ""
    try:
        from app.services.ticket_service import create_or_update_ticket, get_formatted_contact_info
        from app.services.escalation_engine import build_booking_state_snapshot

        booking_snapshot = build_booking_state_snapshot(state)
        ticket_id, _ = await create_or_update_ticket(
            hotel_id=int(hotel_id) if str(hotel_id).isdigit() else 0,
            session_id=session_id,
            user_message=user_message,
            issue_type="BOOKING_DISPUTE",
            reason_code="STOP_SOLD_ROOM_REQUEST",
            priority="HIGH",
            summary=summary,
            booking_state=booking_snapshot,
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=state.get("guest_email"),
            booking_id=state.get("booking_id"),
        )
        contact_info = await get_formatted_contact_info(hotel_id)
    except Exception as e:
        logger.warning(f"Could not create ticket for stop-sell escalation: {e}")
        contact_info = ""

    # Construct the clear, reassuring guest response
    response_lines = [
        f"I understand you specifically want to book our **{room_name}**.",
        f"Currently, online booking for this room is closed by our hotel management {dates_str}. "
        f"I have forwarded your request directly to our front desk team, and a hotel representative will contact you shortly to check if special arrangements can be made for you.",
    ]
    if ticket_id:
        response_lines.append(f"🎫 **Ticket ID:** {ticket_id}")
    if contact_info and contact_info != "Please contact our front desk for assistance.":
        response_lines.append(f"If you need immediate assistance, you can also reach our front desk directly at:\n\n{contact_info}")

    bot_response = text_card("\n\n".join(response_lines))

    return {
        "bot_response": bot_response,
        "escalation_required": True,
        "escalation_issue_type": "BOOKING_DISPUTE",
        "escalation_reason_code": "STOP_SOLD_ROOM_REQUEST",
        "escalation_reason": f"Customer exclusively requested stop-sold room '{room_name}'",
        "stop_sold_room_name": room_name,
        "stop_sold_room_id": room_id,
        "ticket_id": ticket_id,
        "next_action": "escalation_handled",
        "booking_stage": "escalated",
        "room_type_preference": room_name,
    }
