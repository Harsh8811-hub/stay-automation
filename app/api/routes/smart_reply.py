import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm import compat as genai

from app.db.mongodb import get_database
from app.repositories import session_repository
from app.services.gemini_service import _get_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/smart-reply", tags=["smart-reply"])

def smart_replies_collection():
    return get_database()["smart_replies"]

class SmartReplyResponse(BaseModel):
    phone: str
    suggested_reply: str
    message_count: int
    last_updated: str
    suggested_replies: list[str] = []
    ticket_id: str | None = None
    issue_type: str | None = None

async def generate_reply(history: list, ticket_context: str | None = None) -> str:
    """Uses Gemini to generate a suggested reply based on the chat history and any ticket context."""
    prompt = "You are a helpful hotel staff agent. Suggest a short, polite, and directly helpful reply to the guest based on this conversation history:\n\n"
    if ticket_context:
        prompt += f"ESCALATION CONTEXT: {ticket_context}\n\n"
    for m in history:
        sender = m.get("role", "unknown")
        content = m.get("content", "")
        prompt += f"[{sender.upper()}] {content}\n"
    prompt += "\nSuggested reply:"
    
    model = _get_model()
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Failed to generate smart reply: {e}")
        return "Please review the guest's message to reply."

@router.get("/{phone}", response_model=SmartReplyResponse)
async def get_smart_reply(phone: str, hotel_id: int):
    """
    Retrieves the latest smart reply for the given phone number.
    If the chat has moved forward (i.e. new messages were added), 
    it generates a new smart reply and caches it.
    Integrates with active escalation support tickets for contextual quick replies.
    """
    from app.repositories import ticket_repository

    # 1. Fetch active session
    session = await session_repository.find_active_session(hotel_id, phone)
    if not session:
        raise HTTPException(status_code=404, detail="No active session found")
        
    messages = session.messages
    current_count = len(messages)
    
    if current_count == 0:
        return SmartReplyResponse(
            phone=phone,
            suggested_reply="",
            message_count=0,
            last_updated=datetime.now(timezone.utc).isoformat()
        )

    # 2. Check for open support ticket for contextual escalation replies
    open_ticket = await ticket_repository.find_open_ticket_by_customer(hotel_id, phone)
    ticket_id = open_ticket.get("ticket_id") if open_ticket else None
    issue_type = open_ticket.get("issue_type") if open_ticket else None
    ticket_replies = open_ticket.get("smart_replies", []) if open_ticket else []
        
    # 3. Check if we already have a generated reply for this exact message count
    coll = smart_replies_collection()
    cached = await coll.find_one({"phone": phone, "hotel_id": hotel_id})
    
    if cached and cached.get("message_count") == current_count:
        cached_replies = cached.get("suggested_replies") or ticket_replies or [cached["suggested_reply"]]
        return SmartReplyResponse(
            phone=phone,
            suggested_reply=cached["suggested_reply"],
            message_count=current_count,
            last_updated=cached["last_updated"].isoformat(),
            suggested_replies=cached_replies,
            ticket_id=ticket_id or cached.get("ticket_id"),
            issue_type=issue_type or cached.get("issue_type"),
        )
        
    # 4. If open ticket has dedicated smart replies, use those
    if ticket_replies:
        top_reply = ticket_replies[0]
        now = datetime.now(timezone.utc)
        await coll.update_one(
            {"phone": phone, "hotel_id": hotel_id},
            {"$set": {
                "suggested_reply": top_reply,
                "suggested_replies": ticket_replies,
                "message_count": current_count,
                "last_updated": now,
                "ticket_id": ticket_id,
                "issue_type": issue_type,
            }},
            upsert=True
        )
        return SmartReplyResponse(
            phone=phone,
            suggested_reply=top_reply,
            message_count=current_count,
            last_updated=now.isoformat(),
            suggested_replies=ticket_replies,
            ticket_id=ticket_id,
            issue_type=issue_type,
        )

    # 5. Otherwise generate a new one via Gemini since the chat moved forward
    # Get last 10 messages for context
    recent_history = [m.model_dump() for m in messages[-10:]]
    ticket_context = None
    if open_ticket:
        ticket_context = (
            f"Active Ticket: {ticket_id}, Issue: {issue_type}, "
            f"Summary: {open_ticket.get('summary', '')}"
        )
    new_reply = await generate_reply(recent_history, ticket_context=ticket_context)
    replies_list = [new_reply]
    
    # 6. Store the new reply in DB
    now = datetime.now(timezone.utc)
    await coll.update_one(
        {"phone": phone, "hotel_id": hotel_id},
        {"$set": {
            "suggested_reply": new_reply,
            "suggested_replies": replies_list,
            "message_count": current_count,
            "last_updated": now,
            "ticket_id": ticket_id,
            "issue_type": issue_type,
        }},
        upsert=True
    )
    
    return SmartReplyResponse(
        phone=phone,
        suggested_reply=new_reply,
        message_count=current_count,
        last_updated=now.isoformat(),
        suggested_replies=replies_list,
        ticket_id=ticket_id,
        issue_type=issue_type,
    )
