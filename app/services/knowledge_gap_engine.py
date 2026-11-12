"""
Knowledge Gap Engine — identifies and tracks unanswered questions for continuous learning.
Targets canonical `knowledge_gaps` collection.
"""
import logging
from datetime import datetime, timezone
from app.services.llm import compat as genai
from app.config.settings import get_settings
from app.db.mongodb import knowledge_requests_collection

logger = logging.getLogger(__name__)

async def verify_and_log_knowledge_gap(
    hotel_id: int | str, 
    session_id: str, 
    question: str, 
    rag_context: str, 
    best_score: float
) -> bool:
    """
    Lightweight verification of retrieval confidence.
    Returns True if this is a genuine knowledge gap (NOT_SUPPORTED).
    Returns False if Gemini can answer it (SUPPORTED).
    """
    settings = get_settings()
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    prompt = f"""Can this question about the hotel or room booking be reasonably answered or addressed using the supplied context (which includes hotel information, available room types, rates, amenities, and policies)?
If the user is asking about room rates, availability, offers, room types, amenities, general booking inquiries, room recommendations (e.g. "which room is best", "kaun sa room accha rahega", "suggest a room"), or comparing room options, and room/hotel information is present in the context, reply SUPPORTED.
Reply ONLY with:
SUPPORTED
or
NOT_SUPPORTED

Question: {question}

Context:
{rag_context}
"""
    try:
        response = await model.generate_content_async(prompt)
        text = response.text.strip().upper()
    except Exception as e:
        logger.error(f"Knowledge gap verification failed: {e}")
        from app.observability.metrics import metrics
        metrics.incr("knowledge_gap_total", outcome="verify_error", hotel_id=str(hotel_id))
        return False  # default to False to avoid creating false tickets on API error
    
    if "NOT_SUPPORTED" in text:
        await _log_knowledge_gap(hotel_id, session_id, question, best_score)
        from app.observability.metrics import metrics
        metrics.incr("knowledge_gap_total", outcome="logged", hotel_id=str(hotel_id))
        return True

    from app.observability.metrics import metrics
    metrics.incr("knowledge_gap_total", outcome="supported", hotel_id=str(hotel_id))
    return False

async def _log_knowledge_gap(hotel_id: int | str, session_id: str, question: str, best_score: float):
    """Upserts a knowledge gap request into the MongoDB knowledge_gaps collection."""
    try:
        col = knowledge_requests_collection()
    except RuntimeError:
        logger.warning("MongoDB not connected. Skipping knowledge gap logging.")
        return

    str_hid = str(hotel_id)
    normalized_question = question.strip().lower()
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))
    
    # Duplicate prevention
    existing = await col.find_one({
        "hotel_id": {"$in": variants},
        "status": {"$in": ["Pending", "PENDING", "OPEN"]},
        "normalized_question": normalized_question
    })
    
    now = datetime.now(timezone.utc)
    
    if existing:
        await col.update_one(
            {"_id": existing["_id"]},
            {
                "$inc": {"times_asked": 1},
                "$set": {"last_asked": now, "updated_at": now}
            }
        )
        logger.info(f"KNOWLEDGE_REQUEST_DUPLICATE | hotel_id={hotel_id} | question='{question}'")
    else:
        today = now.strftime("%Y%m%d")
        count = await col.count_documents({
            "hotel_id": {"$in": variants},
            "gap_id": {"$regex": f"^GAP-{today}-"}
        })
        sequence = str(count + 1).zfill(6)
        gap_id = f"GAP-{today}-{sequence}"

        doc = {
            "gap_id": gap_id,
            "hotel_id": str_hid,
            "session_id": session_id,
            "guest_question": question,
            "normalized_question": normalized_question,
            "best_similarity_score": best_score,
            "status": "Pending",
            "times_asked": 1,
            "staff_answer": None,
            "metadata": {},
            "first_asked": now,
            "last_asked": now,
            "created_at": now,
            "updated_at": now,
        }
        await col.insert_one(doc)
        logger.info(f"KNOWLEDGE_REQUEST_CREATED | gap_id={gap_id} | hotel_id={hotel_id} | question='{question}'")
