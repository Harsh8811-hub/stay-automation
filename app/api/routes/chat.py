"""
Chat API Routes — the main entry point for all chat interactions.
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ChatRequest, ChatResponse
from app.observability.failure_store import record_failure
from app.observability.tracing import bind_context, get_trace_id, mask_phone, reset_context
from app.repositories import session_repository
from app.services.burst_coalescer import join_burst, looks_like_a_fragment
from app.services.chat_runner import ensure_bot_ready, run_chat_turn, run_media_turn
from app.utils.helpers import generate_session_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])


from typing import Any
from app.api.routes.whatsapp import format_cards_for_whatsapp
from app.config.settings import get_settings
from app.utils.whatsapp_interactive import AUTHORED_TYPES, to_interactive

def _collapse_for_crm_whatsapp(cards: list[Any]) -> list[Any]:
    """
    Flattens complex cards (room_options, quote, etc) into text for WhatsApp CRM consumers,
    but preserves media cards (image, video) and interactive cards (reply buttons, lists)
    so the CRM can send them natively.
    """
    # `interactive` is in the pass-through list for the same reason the media types are:
    # the CRM forwards the card's `data` verbatim to Meta, and cannot do that with text we
    # have already flattened. Dropping it from this tuple is silent — the buttons would be
    # rendered as a numbered list and never sent — so any new native type belongs here.
    _PASSTHROUGH = ("image", "video", "audio", "document", "media_delivery", "interactive")

    new_cards = []
    text_buffer = []

    def flush_text():
        if text_buffer:
            combined = "\n\n".join(text_buffer).strip()
            if combined:
                new_cards.append({"type": "text", "text": combined})
            text_buffer.clear()

    for card in cards or []:
        if not isinstance(card, dict):
            continue
        ctype = card.get("type")
        
        # A prompt that authored buttons becomes a Meta payload here, or it does not
        # happen at all — below this point every unknown type is flattened to text, and
        # `format_cards_for_whatsapp` has no branch for the authored types. That flattening
        # is what made the main menu disappear on 2026-10-14, so it is done deliberately
        # and the gate is checked deliberately.
        # `room_options` is not an authored type — it is a data card the chat UI
        # renders itself — but since 2026-10-18 it carries reply buttons for its
        # numbered combinations. It goes through the same builder, which returns
        # None for a card too long for a Meta body and lets it fall back to text.
        _has_authored_buttons = ctype in AUTHORED_TYPES or (
            ctype == "room_options" and card.get("buttons")
        )
        if _has_authored_buttons and get_settings().WHATSAPP_BUTTONS_ENABLED:
            payload = to_interactive(card)
            if payload is not None:
                flush_text()
                new_cards.append(payload)
                continue
            logger.warning(
                "Authored %s card could not be built into a Meta payload; sending it as text",
                ctype,
            )

        # Keep media and interactive cards exactly as they are so CRM can render/send them
        if ctype in _PASSTHROUGH:
            flush_text()
            new_cards.append(card)
        else:
            # For complex UI cards and normal text cards, use the robust formatting logic
            formatted = format_cards_for_whatsapp([card])
            if formatted:
                text_buffer.append(formatted)

    flush_text()
    return new_cards

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, channel: str | None = None):
    tokens = bind_context(
        session_id=request.session_id or "-",
        hotel_id=request.hotel_id,
        channel=channel or "api",
    )
    try:
        message = (request.message or "").strip()
        if not message:
            logger.warning("Empty message received in chat API.")
            return ChatResponse(
                session_id=request.session_id or generate_session_id(),
                response=[{"type": "text", "text": "I didn't quite catch that. Could you please send some text?"}],
                booking_stage="new",
            )

        logger.info(
            "CHAT_TURN_START | phone=%s msg_len=%s",
            mask_phone(request.customer_phone),
            len(message),
        )

        # ── Coalesce a burst of guest messages ───────────────────────────
        # Guests send one thought as several messages. Buffer this one for the
        # configured window. If a newer message arrives while we wait, answer
        # nothing — the newer turn runs once, over all of the fragments joined.
        # An empty response is the same "say nothing" the takeover path uses;
        # the CRM simply dispatches no message for it.
        await ensure_bot_ready()
        resolved_hotel_id = request.hotel_id or get_settings().HOTEL_ID
        burst_session_id = request.session_id or generate_session_id(
            str(resolved_hotel_id), request.customer_phone or ""
        )
        burst = await join_burst(
            hotel_id=resolved_hotel_id,
            session_key=burst_session_id,
            message=message,
            window_seconds=(
                get_settings().MESSAGE_DEBOUNCE_FRAGMENT_SECONDS
                if looks_like_a_fragment(message)
                else get_settings().MESSAGE_DEBOUNCE_SECONDS
            ),
        )
        if not burst.is_winner:
            stage = "new"
            try:
                # str, not the raw id: sessions are stored under a string
                # hotel_id and the lookup is strictly scoped, so an int here
                # silently finds nothing and reports the stage as "new".
                session = await session_repository.get_session(
                    burst_session_id, hotel_id=str(resolved_hotel_id)
                )
                if session is not None:
                    stage = session.booking_stage or "new"
            except Exception:
                logger.warning(
                    "BURST_STAGE_LOOKUP_FAILED | session=%s", burst_session_id, exc_info=True
                )
            logger.info("CHAT_TURN_SUPERSEDED | session=%s", burst_session_id)
            return ChatResponse(
                session_id=burst_session_id,
                response=[],
                booking_stage=stage,
            )

        result = await run_chat_turn(
            message=burst.text,
            session_id=request.session_id,
            hotel_id=request.hotel_id,
            customer_phone=request.customer_phone or "",
        )
        logger.info(
            "CHAT_TURN_OK | session=%s stage=%s cards=%s",
            result.get("session_id"),
            result.get("booking_stage"),
            [c.get("type") for c in (result.get("response") or []) if isinstance(c, dict)],
        )
        
        from app.services.hotel_media_service import hotel_media_service
        if "response" in result and isinstance(result["response"], list):
            resolved_channel = (channel or "api").lower()
            
            # 1. First, collapse complex UI cards to Text if the CRM requested WhatsApp channel formatting
            if resolved_channel in ("whatsapp", "api"):
                result["response"] = _collapse_for_crm_whatsapp(result["response"])
                
            # 2. Resolve media delivery cards into actual image URLs
            result["response"] = await hotel_media_service.resolve_media_cards_for_api(
                request.hotel_id, result["response"]
            )
            
            # The CRM has a bug where it crashes on images with an empty caption ("") 
            # when it attempts to send them as WhatsApp messages. We inject a single space.
            if resolved_channel in ("whatsapp", "api"):
                for c in result["response"]:
                    if c.get("type") == "image" and not c.get("caption"):
                        c["caption"] = " "

        return ChatResponse(**result)
    except ValueError as e:
        from pydantic import ValidationError
        if isinstance(e, ValidationError):
            record_failure(
                component="chat.response_validation",
                error=e,
                hotel_id=request.hotel_id,
                session_id=request.session_id,
                phone=request.customer_phone,
                details={"result_keys": list(result.keys()) if "result" in locals() else None},
                exc=e,
            )
            logger.error(
                "ChatResponse validation failed after graph success: %s | result_keys=%s",
                e,
                list(result.keys()) if "result" in locals() else None,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Response formatting error. trace_id={get_trace_id()}",
            ) from e
        logger.error("ValueError in /api/chat turn (surfacing as 400): %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        record_failure(
            component="chat.graph",
            error=e,
            hotel_id=request.hotel_id,
            session_id=request.session_id,
            phone=request.customer_phone,
            exc=e,
        )
        logger.error("Graph execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error. trace_id={get_trace_id()}",
        ) from e
    finally:
        reset_context(tokens)

class ChatDebugResponse(ChatResponse):
    debug_log: list[dict]

@router.post("/chat/debug", response_model=ChatDebugResponse)
async def chat_debug(request: ChatRequest, channel: str | None = None):
    """Exact replica of /chat but returns all captured debug logs."""
    from app.observability.debug_logger import enable_debug_logging, disable_debug_logging, get_debug_logs
    
    # Enable debug logging for this request's context
    debug_token = enable_debug_logging()
    
    tokens = bind_context(
        session_id=request.session_id or "-",
        hotel_id=request.hotel_id,
        channel=channel or "api",
    )
    try:
        message = (request.message or "").strip()
        if not message:
            return ChatDebugResponse(
                session_id=request.session_id or generate_session_id(),
                response=[{"type": "text", "text": "I didn't quite catch that. Could you please send some text?"}],
                booking_stage="new",
                debug_log=get_debug_logs(),
            )

        result = await run_chat_turn(
            message=request.message,
            session_id=request.session_id,
            hotel_id=request.hotel_id,
            customer_phone=request.customer_phone or "",
        )
        
        from app.services.hotel_media_service import hotel_media_service
        if "response" in result and isinstance(result["response"], list):
            resolved_channel = (channel or "api").lower()
            if resolved_channel in ("whatsapp", "api"):
                result["response"] = _collapse_for_crm_whatsapp(result["response"])
            result["response"] = await hotel_media_service.resolve_media_cards_for_api(
                request.hotel_id, result["response"]
            )
            if resolved_channel in ("whatsapp", "api"):
                for c in result["response"]:
                    if c.get("type") == "image" and not c.get("caption"):
                        c["caption"] = " "

        return ChatDebugResponse(**result, debug_log=get_debug_logs())
    except Exception as e:
        logger.error("Debug Graph execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error. trace_id={get_trace_id()}",
        ) from e
    finally:
        reset_context(tokens)
        disable_debug_logging(debug_token)



@router.post("/chat/media", response_model=ChatResponse)
async def chat_media(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    message: str | None = Form(None),
    hotel_id: int | None = Form(None),
    customer_phone: str | None = Form(None),
    channel: str | None = Form(None),
    mime_type: str | None = Form(None),
):
    tokens = bind_context(session_id=session_id or "-", hotel_id=hotel_id, channel=channel or "api")
    try:
        file_bytes = await file.read()
        effective_mime = mime_type or file.content_type or "application/octet-stream"
        logger.info(
            "CHAT_MEDIA_START | phone=%s file=%s mime=%s declared_mime=%s bytes=%s",
            mask_phone(customer_phone),
            file.filename,
            file.content_type,
            mime_type,
            len(file_bytes),
        )
        result = await run_media_turn(
            file_bytes=file_bytes,
            filename=file.filename or "upload",
            content_type=effective_mime,
            session_id=session_id,
            message=message,
            hotel_id=hotel_id,
            customer_phone=customer_phone or "",
        )
        logger.info(
            "CHAT_MEDIA_OK | stage=%s cards=%s",
            result.get("booking_stage"),
            [c.get("type") for c in (result.get("response") or []) if isinstance(c, dict)],
        )
        
        from app.services.hotel_media_service import hotel_media_service
        if "response" in result and isinstance(result["response"], list) and hotel_id:
            resolved_channel = (channel or "api").lower()
            if resolved_channel in ("whatsapp", "api"):
                result["response"] = _collapse_for_crm_whatsapp(result["response"])
                
            result["response"] = await hotel_media_service.resolve_media_cards_for_api(
                hotel_id, result["response"]
            )
            
            if resolved_channel in ("whatsapp", "api"):
                for c in result["response"]:
                    if c.get("type") == "image" and not c.get("caption"):
                        c["caption"] = " "
            
        return ChatResponse(**result)
    except Exception as e:
        record_failure(
            component="chat.media",
            error=e,
            hotel_id=hotel_id,
            session_id=session_id,
            phone=customer_phone,
            details={"filename": file.filename, "content_type": file.content_type},
            exc=e,
        )
        logger.error("Media chat failed: %s", e, exc_info=True)
        return ChatResponse(
            session_id=session_id or generate_session_id(),
            response=[{"type": "text", "text": "Sorry, I couldn't process this file due to a system error."}],
            booking_stage="new",
        )
    finally:
        reset_context(tokens)
