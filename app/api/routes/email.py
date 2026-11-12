"""
Email API Routes — endpoint for guest email ingestion and context-aware reply generation.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.models.email_schemas import EmailReplyRequest, EmailReplyResponse
from app.services.email_response_service import generate_email_reply, retrieve_email_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/email", tags=["Email"])


@router.post(
    "/generate-reply",
    response_model=EmailReplyResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest guest email and generate context-aware reply",
    description=(
        "Ingests an incoming guest email, automatically handles all context retrieval "
        "(hotel profile, policies, ChromaDB FAQs, verified bookings, live inventory rates), "
        "and returns a professionally formatted email response."
    ),
)
async def generate_reply_endpoint(request: EmailReplyRequest) -> EmailReplyResponse:
    """Ingest guest email and return structured response."""
    body = (request.body or "").strip()
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email body cannot be empty.",
        )

    from_email = (request.from_email or "").strip()
    if not from_email or "@" not in from_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid from_email is required.",
        )

    try:
        response = await generate_email_reply(request)
        return response
    except Exception as e:
        logger.error(f"Error in generate_reply_endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate email reply: {str(e)}",
        )


@router.post(
    "/context",
    summary="Preview retrieved context for an incoming email",
    description="Inspect what context (policies, RAG, bookings, live inventory) would be retrieved for an email.",
)
async def preview_email_context_endpoint(request: EmailReplyRequest) -> dict:
    """Preview context retrieval without calling the LLM."""
    body = (request.body or "").strip()
    from_email = (request.from_email or "").strip()

    context, sources = await retrieve_email_context(
        hotel_id=request.hotel_id,
        from_email=from_email,
        subject=request.subject or "",
        body=body,
        explicit_booking_id=request.booking_id,
    )
    return {
        "hotel_id": request.hotel_id,
        "from_email": from_email,
        "context_sources_used": sources,
        "retrieved_context": context,
    }
