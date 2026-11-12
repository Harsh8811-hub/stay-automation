"""
Ticket Service — high-level ticket operations.

Handles:
- Create-or-update logic with duplicate prevention
- Gemini-powered staff summary generation
- Hotel contact info retrieval
- Booking state snapshot for ticket context
"""

import logging
from datetime import datetime

from pymongo.errors import DuplicateKeyError

from app.models.escalation_schemas import (
    SupportTicket,
    TicketEvent,
    TicketMessage,
    StaffNote,
)
from app.repositories import ticket_repository
from app.repositories.hotel_repository import get_hotel_contact_info

logger = logging.getLogger(__name__)

# Ticket ids are counted and then inserted, so two escalations landing in the same second can
# compute the same sequence and the second one loses. Retrying is what makes the assignment
# correct; 5 attempts is far more than the two-or-three-way collision this can realistically
# see, and it keeps the loop bounded if something is wrong for a reason retrying cannot fix.
_TICKET_ID_MAX_ATTEMPTS = 5

# The single sentence returned when a hotel has no contact details on file. Named rather than
# inlined because escalation_handler.py detects "no contact details" by comparing against this
# exact string — a second hardcoded copy of the sentence would silently stop matching the moment
# the copy changed, and the guest would get a contact block containing only the fallback.
NO_CONTACT_FALLBACK = "Please contact our front desk for assistance."


# ═══════════════════════════════════════════════════════════════════
# ESCALATION SMART REPLIES HELPERS
# ═══════════════════════════════════════════════════════════════════

def generate_escalation_smart_replies(
    issue_type: str,
    reason_code: str,
    user_message: str,
    summary: str = "",
    customer_name: str | None = None,
) -> list[str]:
    """
    Generate 3 high-probability suggested replies for hotel staff
    based on the escalation classification.
    """
    name_prefix = f"Hello {customer_name}! " if customer_name else "Hello! "
    itype = (issue_type or "OTHER").upper()
    rcode = (reason_code or "OTHER").upper()

    if rcode == "STOP_SOLD_ROOM_REQUEST" or "STOP_SELL" in rcode:
        import re
        room_label = "this room"
        room_name_direct = "This room"
        m = re.search(r"booking ['\"]([^'\"]+)['\"]", summary)
        if m:
            clean_name = m.group(1).strip()
            room_label = f"the {clean_name}"
            room_name_direct = clean_name
        return [
            f"{name_prefix}I am checking with our front desk team to see if we can open and confirm {room_label} for you. Please allow me a moment.",
            f"{name_prefix}{room_name_direct} is currently blocked in our system, but I can offer an upgrade to an alternative room or arrange it offline for you.",
            f"{name_prefix}Our reservation manager will call you directly in a few minutes to assist with booking {room_label}.",
        ]
    elif itype == "PAYMENT_PROBLEM" or "PAYMENT" in rcode:
        return [
            f"{name_prefix}We have verified your payment! Your booking has been confirmed and we're sending your confirmation voucher shortly.",
            f"{name_prefix}Could you please share your UTR number or a screenshot of the payment receipt so our accounts team can verify it?",
            f"{name_prefix}We are currently checking the transaction with our accounts team and will update you in 5 to 10 minutes.",
        ]
    elif itype == "BOOKING_DISPUTE" or "BOOKING" in rcode:
        return [
            f"{name_prefix}Our front desk manager is looking into your reservation right now to correct this for you.",
            f"{name_prefix}We have updated your room details as requested. Thank you for your patience!",
            f"{name_prefix}Could you please share your booking ID or check-in dates so I can locate your reservation immediately?",
        ]
    elif itype in ("COMPLAINT", "EMOTIONAL_CUSTOMER") or "COMPLAINT" in rcode:
        return [
            f"{name_prefix}We sincerely apologize for the inconvenience. Our manager is looking into this immediately to resolve it for you.",
            f"{name_prefix}Thank you for bringing this to our attention. We are sending a staff member to assist you right away.",
            f"{name_prefix}We truly regret your experience. Would you like our duty manager to call you directly to resolve this?",
        ]
    elif itype in ("SAFETY", "MEDICAL", "SECURITY", "LEGAL"):
        return [
            f"{name_prefix}Our duty manager and emergency team have been alerted and are assisting you immediately.",
            f"{name_prefix}Please stay where you are, our hotel security and staff are on their way.",
            f"{name_prefix}Please call reception immediately at our front desk emergency number or dial 0 from your room.",
        ]
    elif itype == "EXPLICIT_HUMAN_REQUEST":
        return [
            f"{name_prefix}This is the hotel front desk manager. How can I assist you today?",
            f"{name_prefix}I am reviewing your chat with our bot and am here to assist you directly.",
            f"{name_prefix}Welcome! Please let me know what you need help with and I'll take care of it right away.",
        ]
    else:
        return [
            f"{name_prefix}I'm reviewing your request with our front desk team and will assist you shortly.",
            f"{name_prefix}Thank you for reaching out. Let me look into this for you right away.",
            f"{name_prefix}Please let us know if you need any further assistance with your stay.",
        ]


async def _sync_smart_reply_collection(
    hotel_id: int | str,
    customer_phone: str,
    smart_replies: list[str],
    ticket_id: str,
    timestamp: datetime,
) -> None:
    """Sync top escalation smart replies to the smart_replies collection for the Control Center."""
    if not customer_phone or not smart_replies:
        return
    try:
        from app.db.mongodb import get_database
        from app.utils.phone import digits_only
        digits = digits_only(customer_phone) or customer_phone.strip()
        hid = int(hotel_id) if str(hotel_id).isdigit() else hotel_id

        coll = get_database()["smart_replies"]
        await coll.update_one(
            {"phone": digits, "hotel_id": hid},
            {
                "$set": {
                    "suggested_reply": smart_replies[0],
                    "suggested_replies": smart_replies,
                    "last_updated": timestamp,
                    "source": "escalation_ticket",
                    "ticket_id": ticket_id,
                }
            },
            upsert=True,
        )
        logger.info(f"SMART_REPLY_SYNC | phone={digits} hotel_id={hid} ticket_id={ticket_id}")
    except Exception as e:
        logger.warning(f"Could not sync smart reply: {e}")


# ═══════════════════════════════════════════════════════════════════
# CREATE OR UPDATE TICKET
# ═══════════════════════════════════════════════════════════════════

async def create_or_update_ticket(
    hotel_id: int,
    session_id: str,
    user_message: str,
    issue_type: str,
    reason_code: str,
    priority: str,
    summary: str,
    booking_state: dict,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    customer_email: str | None = None,
    booking_id: str | None = None,
) -> tuple[str, bool]:
    """
    Create a new ticket or update an existing one.
    
    Returns:
        (ticket_id, is_new) — ticket_id and whether it was newly created.
    """
    now = datetime.utcnow()
    smart_replies = generate_escalation_smart_replies(
        issue_type=issue_type,
        reason_code=reason_code,
        user_message=user_message,
        summary=summary,
        customer_name=customer_name,
    )

    # 1. Check for existing open ticket (by session first, then by customer)
    existing = await ticket_repository.find_open_ticket_by_session(hotel_id, session_id)
    if not existing and customer_phone:
        existing = await ticket_repository.find_open_ticket_by_customer(hotel_id, customer_phone)
    
    if existing:
        # Append to existing ticket
        ticket_id = existing["ticket_id"]
        await ticket_repository.append_customer_message(ticket_id, user_message)
        await ticket_repository.update_smart_replies(ticket_id, smart_replies)
        
        # Add system note about the update
        await ticket_repository.add_staff_note(
            ticket_id,
            f"Customer sent additional message. Issue type: {issue_type}, Reason: {reason_code}",
            author="system",
        )

        # Sync top reply to smart_replies collection for instant Control Center availability
        if customer_phone:
            await _sync_smart_reply_collection(hotel_id, customer_phone, smart_replies, ticket_id, now)
        
        logger.info(f"📎 Appended to existing ticket: {ticket_id}")
        return ticket_id, False
    
    # 2. Create new ticket
    ticket_id = await ticket_repository.generate_ticket_id(hotel_id)
    
    ticket = SupportTicket(
        ticket_id=ticket_id,
        hotel_id=hotel_id,
        session_id=session_id,
        booking_id=booking_id,
        customer_name=customer_name,
        customer_phone=customer_phone,
        customer_email=customer_email,
        issue_type=issue_type,
        reason_code=reason_code,
        priority=priority,
        status="OPEN",
        summary=summary,
        last_user_message=user_message,
        booking_state=booking_state,
        smart_replies=smart_replies,
        conversation=[
            TicketMessage(role="customer", content=user_message, timestamp=now),
        ],
        staff_notes=[
            StaffNote(
                author="system",
                content=f"Ticket auto-created by AI. Classification: {issue_type} / {reason_code}. Priority: {priority}.",
                timestamp=now,
            ),
        ],
        events=[
            TicketEvent(
                event_type="CREATED",
                description=f"Ticket created via AI. Issue: {issue_type} ({reason_code}). Priority: {priority}.",
                timestamp=now,
                actor="system",
                metadata={
                    "issue_type": issue_type,
                    "reason_code": reason_code,
                    "priority": priority,
                    "session_id": session_id,
                },
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    
    # generate_ticket_id counts and then we insert, so two escalations landing in the same
    # second can compute the same sequence and the second insert loses. The unique index on
    # ticket_id is what decides; the loser recomputes and retries rather than failing the
    # guest's turn with a crash. A losing attempt is safe to repeat because create_ticket
    # writes its activity log *after* the insert, so nothing has been recorded yet.
    for attempt in range(_TICKET_ID_MAX_ATTEMPTS):
        try:
            await ticket_repository.create_ticket(ticket)
            break
        except DuplicateKeyError:
            if attempt == _TICKET_ID_MAX_ATTEMPTS - 1:
                # Out of attempts. Re-raise rather than invent an id or silently drop the
                # ticket — a failed turn the caller can see beats a guest who thinks they
                # were heard when no ticket exists.
                raise
            ticket.ticket_id = await ticket_repository.generate_ticket_id(hotel_id)
            logger.warning(
                f"Ticket id already taken, retrying as {ticket.ticket_id} "
                f"(attempt {attempt + 2}/{_TICKET_ID_MAX_ATTEMPTS})"
            )

    # Sync top reply to smart_replies collection for instant Control Center availability
    if customer_phone:
        await _sync_smart_reply_collection(hotel_id, customer_phone, smart_replies, ticket.ticket_id, now)

    logger.info(f"🎫 New ticket created: {ticket.ticket_id}")
    return ticket.ticket_id, True


# ═══════════════════════════════════════════════════════════════════
# STAFF SUMMARY GENERATION (via Gemini)
# ═══════════════════════════════════════════════════════════════════

async def generate_staff_summary(
    user_message: str,
    issue_type: str,
    reason_code: str,
    booking_state: dict,
    conversation_history: list[dict],
) -> str:
    """
    Generate a concise support summary for hotel staff using Gemini.
    This is the ONLY additional Gemini call in the escalation pipeline.
    """
    try:
        from app.services.llm import compat as genai
        from app.config.settings import get_settings
        
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Build context
        history_text = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in conversation_history[-10:]
        ) if conversation_history else "No previous messages."
        
        booking_ctx = ", ".join(
            f"{k}: {v}" for k, v in booking_state.items()
            if v is not None and v != "" and v != [] and v != {}
        )
        
        prompt = f"""Generate a concise support ticket summary for hotel staff.

ISSUE TYPE: {issue_type}
REASON CODE: {reason_code}
BOOKING STATE: {booking_ctx or 'No active booking'}

CONVERSATION:
{history_text}

LATEST MESSAGE: {user_message}

Write a brief, structured summary (3-5 lines max) covering:
- Customer name (if known)
- Booking reference (if applicable)
- Issue description
- Customer emotion/urgency
- Recommended department (Front Desk, Finance, Security, Management, etc.)

Keep it factual. No pleasantries. Staff-only format."""

        from app.services.llm_errors import LLMUnavailableError
        from app.services.gemini_service import _generate_content_async

        response = await _generate_content_async(
            model,
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        return response.text.strip()
    except LLMUnavailableError as e:
        logger.warning(f"Staff summary generation fallback (LLM unavailable): {e}")
        customer_msgs = [
            str(m.get("content", "")).strip() for m in conversation_history
            if m.get("role") in ("user", "guest", "customer") and str(m.get("content", "")).strip()
        ]
        if user_message and (not customer_msgs or customer_msgs[-1] != user_message.strip()):
            customer_msgs.append(user_message.strip())
        last_3 = customer_msgs[-3:] if customer_msgs else ([user_message.strip()] if user_message else ["No customer messages."])
        return "\n".join(last_3)
    except Exception as e:
        logger.error(f"Staff summary generation failed: {e}")
        return f"Issue: {issue_type} ({reason_code}). Last message: {user_message[:200]}"


# ═══════════════════════════════════════════════════════════════════
# CONTACT INFO RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

async def get_formatted_contact_info(hotel_id: int) -> str:
    """
    Format the hotel's contact details for a guest-facing handoff.

    Reads through `get_hotel_contact_info`, which owns the key spelling. The previous version
    probed the contact record directly for camelCase names — customerCareNumber, hotelPhone,
    hotelEmail — while the record is snake_case, so every lookup missed and the guest was handed
    the no-contact fallback even when the number was on file. Both spellings are handled now, in
    the one place that decides what a contact record means.

    Returns NO_CONTACT_FALLBACK when the hotel has no contact details on file at all; callers
    compare against that constant rather than re-typing the sentence.
    """
    info = await get_hotel_contact_info(hotel_id)

    if not info:
        return NO_CONTACT_FALLBACK

    lines = []

    if info.get("phone"):
        lines.append(f"📞 {info['phone']}")
    if info.get("alt_phone"):
        lines.append(f"📞 {info['alt_phone']}")
    if info.get("whatsapp"):
        lines.append(f"💬 WhatsApp: {info['whatsapp']}")
    if info.get("email"):
        lines.append(f"📧 {info['email']}")
    if info.get("website"):
        lines.append(f"🌐 {info['website']}")
    if info.get("contact_name"):
        lines.append(f"👤 {info['contact_name']}")

    if not lines:
        return NO_CONTACT_FALLBACK

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# BOOKING STATE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════

def build_booking_state_snapshot(state: dict) -> dict:
    """
    Build a compact snapshot of the current booking state for ticket context.
    Only includes relevant, non-null fields.
    """
    fields = [
        "booking_stage", "adults", "children", "num_rooms",
        "check_in", "check_out", "room_type_preference",
        "guest_name", "guest_phone", "guest_email",
        "total_price", "booking_id", "payment_status",
        "amount_paid", "amount_due",
    ]
    snapshot = {}
    for f in fields:
        val = state.get(f)
        if val is not None and val != "" and val != []:
            snapshot[f] = val
    
    # Include selected room names
    selected = state.get("selected_rooms", [])
    if selected:
        snapshot["selected_rooms"] = [
            r.get("room_name", "Unknown") for r in selected
            if isinstance(r, dict)
        ]
    
    return snapshot
