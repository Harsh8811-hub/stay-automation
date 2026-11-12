"""
Document, Event Flyer & Map OCR Reader Service.

Multimodal extraction for non-ID documents:
- Booking request forms / letters → slot-fills reservation details
- Maps / Location screenshots → extracts destination and directions
- Transport tickets → extracts arrival time & flight/train details
- Event invitations / flyers → extracts event details (logged for staff follow-up)
"""

import json
import logging
import re
from typing import Any

from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError
from app.config.settings import get_settings
from app.services.image_vision import _detect_mime_type, _extract_json_from_text

logger = logging.getLogger(__name__)

DOCUMENT_OCR_PROMPT = """You are an intelligent document and image comprehension system for a luxury hotel assistant.
Analyze the provided document, flyer, map, or ticket.

Extract all relevant information into structured JSON:

1. `document_category`: EXACTLY one of:
   - "booking_request" (letter, inquiry, corporate booking request, requisition)
   - "map" (Google Maps screenshot, route map, location pinpoint)
   - "transport" (flight ticket, train ticket, boarding pass, cab receipt)
   - "event" (wedding card, conference schedule, party invitation, ceremony flyer)
   - "complaint" (written feedback, issue letter)
   - "general" (other document or text)

2. `full_text`: Summary or OCR transcription of key text.

3. `extracted_fields`:
   - For booking_request: check_in (YYYY-MM-DD), check_out (YYYY-MM-DD), num_rooms (int), adults (int), children (int), room_type (str), guest_name (str), guest_phone (str), guest_email (str), special_requests (str)
   - For map: destination (str), origin (str), distance_or_time (str)
   - For transport: transport_type (flight/train/bus/cab), identifier (e.g. 6E-204), arrival_time (str), arrival_date (str), origin (str), destination (str)
   - For event: event_name (str), event_date (str), event_time (str), venue_location (str), expected_attendees (int or str), organizer (str)

4. `quality`: "good", "fair", "poor", "unreadable"

OUTPUT FORMAT: Valid JSON only:
```json
{
  "document_category": "booking_request",
  "full_text": "Please reserve 2 Deluxe rooms from 2026-09-01 to 2026-09-03 for Mr. Sharma...",
  "extracted_fields": {
    "check_in": "2026-09-01",
    "check_out": "2026-09-03",
    "num_rooms": 2,
    "adults": 4,
    "room_type": "Deluxe",
    "guest_name": "Sharma"
  },
  "quality": "good"
}
```
"""


async def read_document_media(
    file_bytes: bytes,
    filename: str = "",
    declared_mime: str = ""
) -> dict[str, Any]:
    """
    Analyzes document media bytes and extracts structured entities.
    """
    mime_type = declared_mime or _detect_mime_type(file_bytes)
    size_bytes = len(file_bytes)
    logger.info("DOCUMENT_OCR_START | mime=%s | size_bytes=%s | filename=%s", mime_type, size_bytes, filename)

    fallback: dict[str, Any] = {
        "document_category": "general",
        "full_text": "",
        "extracted_fields": {},
        "quality": "poor",
    }

    try:
        settings = get_settings()
        api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
        if api_key:
            genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.0-flash")

        image_part = {
            "mime_type": mime_type if mime_type.startswith(("image/", "application/pdf")) else "image/jpeg",
            "data": file_bytes,
        }

        response = await model.generate_content_async(
            [image_part, DOCUMENT_OCR_PROMPT],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        if not response or not response.text:
            logger.warning("DOCUMENT_OCR_EMPTY_RESPONSE | mime=%s", mime_type)
            return fallback

        parsed = _extract_json_from_text(response.text)
        if not parsed or not isinstance(parsed, dict):
            logger.warning("DOCUMENT_OCR_PARSE_FAIL | raw=%s", response.text[:100])
            return fallback

        category = str(parsed.get("document_category") or "general").lower().strip()
        fields = parsed.get("extracted_fields") or {}

        # Log specialized events for operational follow-up
        if category == "event":
            logger.info(
                "EVENT_DOCUMENT_DETECTED | event_name=%s | date=%s | time=%s | venue=%s",
                fields.get("event_name"),
                fields.get("event_date"),
                fields.get("event_time"),
                fields.get("venue_location"),
            )
        elif category == "map":
            logger.info(
                "MAP_DOCUMENT_DETECTED | destination=%s | distance_or_time=%s",
                fields.get("destination"),
                fields.get("distance_or_time"),
            )

        logger.info(
            "DOCUMENT_OCR_RESULT | category=%s | fields_count=%s | quality=%s",
            category,
            len(fields),
            parsed.get("quality"),
        )

        return {
            "document_category": category,
            "full_text": parsed.get("full_text", ""),
            "extracted_fields": fields,
            "quality": parsed.get("quality", "good"),
        }

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.error("DOCUMENT_OCR_ERROR | err=%s", e, exc_info=True)
        return fallback
