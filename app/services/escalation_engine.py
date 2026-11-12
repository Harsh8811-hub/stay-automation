"""
Escalation Decision Engine — Python makes ALL final decisions.

Pipeline:
1. Receive Gemini's classification (escalation_required, issue_type, reason_code, etc.)
2. Apply business rules (always-escalate categories, explicit requests, etc.)
3. Verify against knowledge base (MongoDB + ChromaDB) before creating tickets
4. Decide: self-serve OR escalate
5. If escalating: create/update ticket via ticket_service

CRITICAL DESIGN PRINCIPLE:
- Tickets are ONLY created when backend investigation or human intervention is required.
- "How do I pay?" → No ticket (answerable by FAQ/policy)
- "Payment deducted but booking missing" → Ticket (needs backend investigation)
- "What time is check-in?" → No ticket (answerable by FAQ/policy)
- "My room doesn't match what I booked" → Ticket (needs staff verification)
"""

import logging
from typing import Any

from app.rag.chroma_service import query_knowledge
from app.repositories.room_repository import get_all_rooms
from app.repositories.hotel_repository import get_hotel_contact
from app.services.hotel_identity import resolve_hotel_identity
from app.services.repetition_guard import REPEAT_ESCALATION_THRESHOLD
from app.services.ticket_service import (
    create_or_update_ticket,
    generate_staff_summary,
    get_formatted_contact_info,
    build_booking_state_snapshot,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ALWAYS-ESCALATE CATEGORIES (no knowledge check needed)
# ═══════════════════════════════════════════════════════════════════

ALWAYS_ESCALATE = {"MEDICAL", "SAFETY", "SECURITY", "LEGAL"}

# Categories where the guest may be in danger right now, and the reply must
# send them to a human immediately rather than to the ticket queue. LEGAL is
# deliberately absent: it is always-escalated, but a guest raising a legal
# matter does not need to be told to ring reception this second.
EMERGENCY_TYPES = {"MEDICAL", "SAFETY", "SECURITY"}


def is_emergency(issue_type: str | None, priority: str | None) -> bool:
    """
    True when the guest needs a human now, not a ticket in a queue.

    Both readings are required to agree on the obvious cases and either one is
    enough otherwise: the classifier marks these types CRITICAL by rule, and a
    CRITICAL priority from any other path is still a person in trouble.
    """
    return (issue_type or "").upper() in EMERGENCY_TYPES or (priority or "").upper() == "CRITICAL"

# Categories that ALWAYS require backend investigation
REQUIRES_INVESTIGATION = {"PAYMENT_PROBLEM", "BOOKING_DISPUTE"}

# Reason codes that indicate backend investigation is needed
INVESTIGATION_REASON_CODES = {
    "PAYMENT_DUPLICATE", "PAYMENT_PENDING", "PAYMENT_FAILED",
    "BOOKING_NOT_FOUND", "BOOKING_MISMATCH",
    "SECURITY_CONCERN", "FRAUD_REPORT", "IDENTITY_MISMATCH",
    "MEDICAL_EMERGENCY", "SAFETY_CONCERN", "LEGAL_MATTER",
    "STOP_SOLD_ROOM_REQUEST",
}

# Reason codes that are answerable by knowledge base (no ticket needed)
KNOWLEDGE_ANSWERABLE_CODES = {
    "REFUND_REQUEST", "CANCELLATION_REQUEST", "GENERAL_COMPLAINT",
    "SERVICE_COMPLAINT", "BOOKING_MODIFICATION", "OTHER",
}

# Priority mapping
PRIORITY_MAP = {
    "MEDICAL": "CRITICAL",
    "SAFETY": "CRITICAL",
    "SECURITY": "CRITICAL",
    "LEGAL": "CRITICAL",
    "PAYMENT_PROBLEM": "HIGH",
    "BOOKING_DISPUTE": "HIGH",
    "COMPLAINT": "NORMAL",
    "EMOTIONAL_CUSTOMER": "NORMAL",
    "OUT_OF_SCOPE": "LOW",
    "EXPLICIT_HUMAN_REQUEST": "NORMAL",
    "TECHNICAL_FAILURE": "HIGH",
    "UNKNOWN": "LOW",
    "NORMAL": "LOW",
}


# ═══════════════════════════════════════════════════════════════════
# MAIN DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════

async def evaluate_escalation(state: dict) -> dict:
    """
    Main escalation decision function.
    
    Returns a dict with:
    - action: "self_serve" | "create_ticket" | "update_ticket"
    - response_text: The bot response text
    - ticket_id: If a ticket was created/updated
    - contact_info: Formatted contact details
    """
    issue_type = state.get("escalation_issue_type", "NORMAL")
    reason_code = state.get("escalation_reason_code", "OTHER")
    escalation_required = state.get("escalation_required", False)
    user_message = state.get("user_message", "")
    hotel_id = state.get("hotel_id", 0)
    intent = state.get("intent", "UNKNOWN")
    
    # Resolve priority from classification or category defaults
    gemini_urgency = state.get("escalation_urgency", "LOW")
    default_priority = PRIORITY_MAP.get(issue_type, "NORMAL")
    priority = _resolve_priority(gemini_urgency, default_priority, issue_type)
    
    logger.info(
        f"Escalation evaluation: issue={issue_type}, reason={reason_code}, "
        f"required={escalation_required}, priority={priority}, intent={intent}"
    )
    
    failure_count = state.get("escalation_failure_count", 0)
    llm_unavailable = bool(state.get("llm_unavailable"))

    # M2 explicit guard: Never escalate or create a ticket because LLM is down
    if llm_unavailable:
        logger.info(
            "ESCALATION_DECISION reason=infra_suppressed counter=%s llm_unavailable=True",
            failure_count,
        )
        return {"action": "no_escalation"}

    # 0. Always-escalate high urgency / angry complaints
    gemini_sentiment = (state.get("escalation_sentiment") or "").upper()
    is_angry_or_high_complaint = (
        (issue_type in ("COMPLAINT", "EMOTIONAL_CUSTOMER") or escalation_required)
        and (gemini_sentiment == "ANGRY" or gemini_urgency in ("HIGH", "CRITICAL") or priority in ("HIGH", "CRITICAL"))
    )
    if is_angry_or_high_complaint:
        logger.info("High urgency / angry complaint detected — escalating directly to staff")
        logger.info(
            "ESCALATION_DECISION reason=negative_sentiment counter=%s llm_unavailable=False",
            failure_count,
        )
        return await _create_ticket_response(state, priority)

    # 1. Always-escalate categories: MEDICAL, SAFETY, SECURITY, LEGAL
    if issue_type in ALWAYS_ESCALATE:
        logger.info(f"Always-escalate category: {issue_type}")
        logger.info(
            "ESCALATION_DECISION reason=issue_type counter=%s llm_unavailable=False",
            failure_count,
        )
        return await _create_ticket_response(state, priority)
    
    # 2. Explicit human request (user said "talk to a person", etc.)
    if intent == "HUMAN_REQUEST" or issue_type == "EXPLICIT_HUMAN_REQUEST":
        logger.info("Explicit human request detected")
        logger.info(
            "ESCALATION_DECISION reason=explicit_request counter=%s llm_unavailable=False",
            failure_count,
        )
        return await _create_ticket_response(state, priority)
    
    # 3. Repeated failure threshold (3+ consecutive misunderstandings)
    if failure_count >= 3:
        logger.info(f"Failure threshold reached: {failure_count} consecutive failures")
        logger.info(
            "ESCALATION_DECISION reason=genuine_unknown counter=%s llm_unavailable=False",
            failure_count,
        )
        return await _create_ticket_response(state, "NORMAL")

    # 3b. Repeat-reply threshold (B3b) — consecutive identical bot replies
    repeat_count = state.get("repeat_response_count", 0)
    if repeat_count >= REPEAT_ESCALATION_THRESHOLD:
        logger.info(f"Repeat-reply threshold reached: {repeat_count} identical replies in a row")
        logger.info(
            "ESCALATION_DECISION reason=genuine_unknown counter=%s llm_unavailable=False",
            failure_count,
        )
        return await _create_ticket_response(state, "NORMAL")
    
    # 4. Investigation-required reason codes (payment deducted, booking missing, etc.)
    if reason_code in INVESTIGATION_REASON_CODES:
        logger.info(f"Backend investigation required: {reason_code}")
        logger.info(
            "ESCALATION_DECISION reason=issue_type counter=%s llm_unavailable=False",
            failure_count,
        )
        return await _create_ticket_response(state, priority)
    
    # 5. Out of scope (jobs, partnerships, vendor, media, franchise)
    if issue_type == "OUT_OF_SCOPE":
        logger.info("Out of scope query — providing contact info without support ticket")
        return await _self_serve_with_contact(state)
    
    # 6. Knowledge verification — can we answer this from our data?
    if escalation_required or issue_type in ("COMPLAINT", "EMOTIONAL_CUSTOMER"):
        knowledge_result = await _verify_knowledge(hotel_id, user_message, issue_type)
        
        if knowledge_result:
            # We CAN answer — self-serve with knowledge + contact info
            logger.info("Knowledge found — self-serving instead of escalating")
            return await _self_serve_response(state, knowledge_result)
        elif issue_type in REQUIRES_INVESTIGATION or priority in ("HIGH", "CRITICAL") or gemini_urgency in ("HIGH", "CRITICAL") or (state.get("escalation_sentiment") or "").upper() == "ANGRY":
            # No knowledge AND requires investigation / high priority / angry sentiment — escalate
            logger.info(f"Escalating high-urgency/complaint issue: {issue_type} with priority {priority}")
            logger.info(
                "ESCALATION_DECISION reason=negative_sentiment counter=%s llm_unavailable=False",
                failure_count,
            )
            return await _create_ticket_response(state, priority)
        else:
            # No knowledge but not critical — offer contact info, no ticket yet
            logger.info("No knowledge found but not critical — offering contact info")
            return await _self_serve_with_contact(state)
    
    # 7. Default: no escalation needed
    logger.info("No escalation required — normal flow")
    logger.info(
        "ESCALATION_DECISION reason=none counter=%s llm_unavailable=False",
        failure_count,
    )
    return {"action": "no_escalation"}


# ═══════════════════════════════════════════════════════════════════
# KNOWLEDGE VERIFICATION
# ═══════════════════════════════════════════════════════════════════

async def _verify_knowledge(hotel_id: int, user_message: str, issue_type: str) -> str | None:
    """
    Search all available knowledge sources to see if we can answer the query.
    
    Searches:
    1. ChromaDB RAG (policies, FAQs)
    2. Room information from room_types/rate_plans
    3. Hotel amenities
    
    Returns the combined knowledge text if found, None if not.
    """
    knowledge_parts = []
    
    # 1. ChromaDB RAG search
    rag_result = query_knowledge(
        query=user_message,
        hotel_id=hotel_id,
        n_results=3,
    )
    if rag_result and len(rag_result.strip()) > 20:
        knowledge_parts.append(rag_result)
    
    # 2. Room information (for booking/room complaints)
    if issue_type in ("COMPLAINT", "BOOKING_DISPUTE", "EMOTIONAL_CUSTOMER", "NORMAL"):
        try:
            rooms = await get_all_rooms(hotel_id)
            if rooms:
                from app.services.pricing_engine import PricingEngine
                room_info = "Available Room Types:\n"
                for r in rooms:
                    base_price = min([rp.base_price for rp in r.rate_plans]) if r.rate_plans else None
                    inc_cost = PricingEngine.estimate_inclusive_price(base_price) if base_price else None
                    cost = f"₹{inc_cost:,.0f}/night (incl. taxes)" if inc_cost else "N/A"
                    room_info += f"- {r.name} ({r.room_type}): Capacity {r.occupancy.base_adults} adults. {cost}\n"
                knowledge_parts.append(room_info)
        except Exception as e:
            logger.debug(f"Room info fetch failed during knowledge verification: {e}")
    
    if knowledge_parts:
        return "\n\n---\n\n".join(knowledge_parts)
    
    return None


# ═══════════════════════════════════════════════════════════════════
# RESPONSE BUILDERS
# ═══════════════════════════════════════════════════════════════════

async def _create_ticket_response(state: dict, priority: str) -> dict:
    """Create a new ticket (or update existing) and build the response."""
    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")
    user_message = state.get("user_message", "")
    issue_type = state.get("escalation_issue_type", "UNKNOWN")
    reason_code = state.get("escalation_reason_code", "OTHER")

    # Check if support ticket creation is enabled for this hotel
    from app.services.hotel_config_service import get_hotel_config
    config = await get_hotel_config(hotel_id)
    support_ticket_enabled = getattr(config.business_config.features, "support_ticket_enabled", False)
    human_escalation_enabled = getattr(config.business_config.features, "human_escalation_enabled", False)
    # If support tickets are disabled, but it's an emergency, high/critical priority, or angry sentiment, bypass the gate and notify staff / create ticket
    gemini_urgency = state.get("escalation_urgency", "LOW")
    gemini_sentiment = (state.get("escalation_sentiment") or "").upper()
    is_high_urgency = priority in ("HIGH", "CRITICAL") or gemini_urgency in ("HIGH", "CRITICAL") or gemini_sentiment == "ANGRY"
    if not support_ticket_enabled and not human_escalation_enabled and issue_type not in ALWAYS_ESCALATE and not is_high_urgency:
        return await _self_serve_with_contact(state)
    
    # Build booking state snapshot
    booking_snapshot = build_booking_state_snapshot(state)
    
    # Generate staff summary via Gemini
    conversation_history = state.get("messages_history", [])
    summary = await generate_staff_summary(
        user_message=user_message,
        issue_type=issue_type,
        reason_code=reason_code,
        booking_state=booking_snapshot,
        conversation_history=conversation_history,
    )
    
    # Create or update ticket
    ticket_id, is_new = await create_or_update_ticket(
        hotel_id=hotel_id,
        session_id=session_id,
        user_message=user_message,
        issue_type=issue_type,
        reason_code=reason_code,
        priority=priority,
        summary=summary,
        booking_state=booking_snapshot,
        customer_name=state.get("guest_name"),
        customer_phone=state.get("guest_phone") or state.get("customer_identifier"),
        customer_email=state.get("guest_email"),
        booking_id=state.get("booking_id"),
    )
    
    # Get contact info
    contact_info = await get_formatted_contact_info(hotel_id)
    
    if reason_code == "STOP_SOLD_ROOM_REQUEST":
        room_name = state.get("stop_sold_room_name") or state.get("room_type_preference") or "requested room"
        check_in = state.get("check_in")
        check_out = state.get("check_out")
        dates_str = f" from {check_in} to {check_out}" if check_in and check_out else ""
        response_text = (
            f"I understand you specifically want to book our **{room_name}**.\n\n"
            f"Currently, online booking for this room is closed by our management{dates_str}. "
            f"I have forwarded your request directly to our hotel team, and a representative will contact you shortly to check if special arrangements can be made for you.\n\n"
            f"🎫 **Ticket ID:** {ticket_id}"
        )
        if contact_info and contact_info != "Please contact our front desk for assistance.":
            response_text += f"\n\nIf you need immediate assistance, you can also reach our front desk directly at:\n\n{contact_info}"
    elif is_emergency(issue_type, priority):
        # An emergency is not a ticket. Until 2026-10-28 this branch did not
        # exist, so "there is a fire on the second floor" — classified SAFETY at
        # 0.99, priority CRITICAL, logged as such — was answered with the same
        # "our hotel staff will review your request as soon as possible" as a
        # complaint about the breakfast. The escalation was correct; the guest
        # was told to wait for a review. What they need is the phone number, in
        # the first line, and the ticket is the record that it happened.
        identity = await resolve_hotel_identity(hotel_id, state)
        if identity.phone:
            call_now = f"**Please call our reception immediately on {identity.phone}.**"
        else:
            call_now = "**Please alert our reception staff immediately.**"
        ticket_line = (
            f"I've also opened support ticket **{ticket_id}** so our team has this in writing."
            if is_new
            else f"Support ticket **{ticket_id}** is already open and your message is attached to it."
        )
        response_text = (
            f"⚠️ **This sounds like an emergency.**\n\n"
            f"{call_now}\n\n"
            f"{ticket_line}"
        )
    elif is_new:
        sentiment = (state.get("escalation_sentiment") or "NEUTRAL").upper()
        if sentiment in ("NEGATIVE", "ANGRY") or issue_type in (
            "COMPLAINT", "EMOTIONAL_CUSTOMER", "PAYMENT_PROBLEM", "BOOKING_DISPUTE"
        ):
            lead = "I understand your concern. I've created a support ticket for you."
        else:
            lead = "I've connected you with our team and created a support ticket for you."
        response_text = (
            f"{lead}\n\n"
            f"🎫 **Ticket ID:** {ticket_id}\n\n"
            f"Our hotel staff will review your request as soon as possible.\n\n"
            f"If you'd like to contact us directly:\n\n{contact_info}"
        )
    else:
        response_text = (
            f"Your existing support ticket (**{ticket_id}**) is currently being handled by our team.\n\n"
            f"I've attached your latest message to the ticket for our staff to review.\n\n"
            f"If you'd like to contact us directly:\n\n{contact_info}"
        )
    
    return {
        "action": "create_ticket" if is_new else "update_ticket",
        "response_text": response_text,
        "ticket_id": ticket_id,
        "contact_info": contact_info,
    }


async def _self_serve_response(state: dict, knowledge: str) -> dict:
    """Answer using available knowledge + offer contact info."""
    hotel_id = state.get("hotel_id", 0)
    user_message = state.get("user_message", "")
    
    # Use Gemini FAQ answering with the retrieved knowledge
    from app.services.gemini_service import answer_faq
    answer = await answer_faq(
        question=user_message,
        rag_context=knowledge,
        hotel_id=hotel_id,
        session_id=state.get("session_id", ""),
    )
    
    # Get contact info
    contact_info = await get_formatted_contact_info(hotel_id)
    
    response_text = (
        f"{answer}\n\n"
        f"If your concern is still unresolved after reviewing this information, "
        f"our hotel staff will be happy to assist you.\n\n"
        f"{contact_info}"
    )
    
    return {
        "action": "self_serve",
        "response_text": response_text,
        "contact_info": contact_info,
    }


async def _self_serve_with_contact(state: dict) -> dict:
    """No knowledge found but not critical enough for a ticket — offer empathy + contact."""
    hotel_id = state.get("hotel_id", 0)
    user_message = state.get("user_message", "")
    sentiment = state.get("escalation_sentiment", "NEUTRAL")
    
    # Generate empathetic response via Gemini
    from app.services.gemini_service import generate_natural_response
    
    empathy_context = "The user has a concern or complaint."
    if sentiment in ("ANGRY", "DISTRESSED"):
        empathy_context = "The user is emotionally upset. Acknowledge their feelings warmly and professionally."
    
    bot_text = await generate_natural_response(
        context=empathy_context,
        action_result="We could not find a specific answer in our knowledge base. Offer contact info for direct assistance.",
        user_message=user_message,
        hotel_id=hotel_id,
        session_id=state.get("session_id", ""),
    )
    
    contact_info = await get_formatted_contact_info(hotel_id)
    
    response_text = (
        f"{bot_text}\n\n"
        f"If you'd like to speak with our team directly:\n\n{contact_info}\n\n"
        f"Or just say **\"connect me to a person\"** and I'll create a support ticket for you right away."
    )
    
    return {
        "action": "self_serve",
        "response_text": response_text,
        "contact_info": contact_info,
    }


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _resolve_priority(gemini_urgency: str, default_priority: str, issue_type: str) -> str:
    """
    Resolve final priority. Use the higher of Gemini's assessment and category default.
    Always-escalate categories override to CRITICAL.
    """
    priority_rank = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "CRITICAL": 3}
    
    if issue_type in ALWAYS_ESCALATE:
        return "CRITICAL"
    
    gemini_rank = priority_rank.get(gemini_urgency, 0)
    default_rank = priority_rank.get(default_priority, 1)
    
    if gemini_rank >= default_rank:
        return gemini_urgency
    return default_priority
