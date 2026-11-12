"""
Booking Note & Handwritten Slip Extractor for StayInn Vision Pipeline.
Extracts stay dates, guest counts, room count, and room type from written notes.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError

from app.config.settings import get_settings
from app.services.vision.classify import _detect_mime_type
from app.services.vision.schemas import BookingNoteExtracted

logger = logging.getLogger(__name__)


async def extract_booking_note(file_bytes: bytes, user_message: str | None = None) -> BookingNoteExtracted:
    """
    Extracts reservation details from a handwritten slip, paper note, or text image using Gemini Vision.
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    prompt = """You are an expert OCR, document analysis, and lead qualification engine for hotel reservations.
Read this handwritten note, paper slip, diary entry, or written booking inquiry carefully.
Understand the handwritten text (English, Hindi, Hinglish, or bilingual hotel reception slips).
Extract all booking and customer information into a structured JSON format to convert it into a qualified lead.

FIELDS TO EXTRACT:
1. "guest_name": Full name of the guest or customer written on the note (e.g. "Rohan Sharma", "Dr. A. K. Verma", "Ramesh ji"), or null if absent.
2. "phone_number": 10-digit mobile or contact phone number written on the note (e.g. "9876543210", "+91 98123 45678"), or null if absent.
3. "check_in": Check-in date in YYYY-MM-DD format (e.g. "15/10/26" -> "2026-10-15", "tomorrow" relative to stay). Assume 2026 if year is 26.
4. "check_out": Check-out date in YYYY-MM-DD format (e.g. "17/10/26" -> "2026-10-17"). Assume 2026 if year is 26.
5. "adults": Number of adult guests / total persons (e.g. "2 x 3 = 6 Person" -> 6, "2 pax" -> 2).
6. "children": Number of children if mentioned (or null).
7. "num_rooms": Number of rooms requested (e.g. "2 NOS", "2 Room", "1 Kamra" -> 2).
8. "room_type_preference": Room type or bed preference mentioned (e.g. "Triple bed", "Deluxe", "Non-AC", "Family Suite").
9. "meal_plan_preference": Meal plan requested if mentioned (e.g. "EP", "CP", "MAP", "AP", "with breakfast", "food included").
10. "special_requests": Specific remarks or requests (e.g. "early check-in 10am", "late check-out", "ground floor", "car parking", "extra mattress").
11. "budget": Total budget or negotiated/quoted room rate if written (e.g. "2500 per night" -> 2500.0, "total 5000" -> 5000.0).
12. "confidence": Extraction confidence between 0.0 and 1.0.
13. "raw_notes": Exact, complete transcription of what is written on the paper slip / note.

Respond ONLY with valid JSON in this exact format:
{
  "guest_name": string or null,
  "phone_number": string or null,
  "check_in": "YYYY-MM-DD" or null,
  "check_out": "YYYY-MM-DD" or null,
  "adults": int or null,
  "children": int or null,
  "num_rooms": int or null,
  "room_type_preference": string or null,
  "meal_plan_preference": string or null,
  "special_requests": string or null,
  "budget": float or null,
  "confidence": float,
  "raw_notes": string
}"""

    if user_message:
        prompt += f"\n\nAdditional user text context: {user_message}"

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        import asyncio
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                [
                    prompt,
                    {"mime_type": mime_type, "data": file_bytes},
                ],
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            ),
        )

        data = json.loads(response.text.strip())
        extracted = BookingNoteExtracted(**data)
        logger.info(
            "BOOKING_NOTE_EXTRACT_SUCCESS | name=%s | phone=%s | in=%s | out=%s | adults=%s | rooms=%s | type=%s",
            extracted.guest_name,
            extracted.phone_number,
            extracted.check_in,
            extracted.check_out,
            extracted.adults,
            extracted.num_rooms,
            extracted.room_type_preference,
        )
        return extracted

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.warning(f"Booking note extraction failed with Gemini: {e}")
        return BookingNoteExtracted(confidence=0.0, raw_notes=None)
