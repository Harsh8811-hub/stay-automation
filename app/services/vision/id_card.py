"""
Government Identity Document Extractor for StayInn Vision Pipeline.
Extracts guest name, document type, masked ID number, DOB, and address
using Gemini Multimodal with structured Pydantic schema output.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError

from app.config.settings import get_settings
from app.services.vision.classify import _detect_mime_type
from app.services.vision.schemas import IDCardExtracted

logger = logging.getLogger(__name__)


def _mask_id_number(raw_num: str | None, id_type: str) -> str | None:
    """Mask document number for guest privacy while preserving the last 4 digits."""
    if not raw_num:
        return None
    clean = re.sub(r'[^a-zA-Z0-9]', '', raw_num)
    if len(clean) <= 4:
        return raw_num
    
    last4 = clean[-4:]
    if id_type == "aadhaar" or len(clean) == 12:
        return f"XXXX XXXX {last4}"
    elif id_type == "pan_card" or len(clean) == 10:
        return f"XXXXXX{last4}"
    elif id_type == "passport":
        return f"XXXX{last4}"
    else:
        return f"XXXX-XXXX-{last4}"


async def extract_id_card(file_bytes: bytes, user_message: str | None = None) -> IDCardExtracted:
    """
    Extracts structured identification data from a government ID card screenshot/photo.
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    prompt = """You are an expert identity document verification engine for a hotel check-in system.
Analyze this uploaded Government ID card (e.g. Indian Aadhaar Card, Passport, Driver's License, Voter ID, PAN Card, National Identity Card).

Extract the following fields accurately:
1. "id_type": "aadhaar" | "passport" | "driving_license" | "voter_id" | "pan_card" | "national_id" | "other_id" | "unknown"
2. "full_name": Full legal name of the cardholder as printed on the document (in English / Latin characters if transliterated or native script).
3. "document_number_masked": The ID number with only the last 4 digits exposed, preceding digits masked with 'X' (e.g. "XXXX XXXX 2163" for Aadhaar).
4. "dob": Date of birth if visible (e.g. "30/05/1995" or "1995-05-30").
5. "gender": "MALE" | "FEMALE" | "OTHER" if visible.
6. "address": Residential address, town/city, state, or pincode if visible on the document.
7. "is_valid_id": true if this is an authentic, readable government-issued ID card.
8. "confidence": Confidence score between 0.0 and 1.0.

Respond strictly in JSON format:
{
  "id_type": string,
  "full_name": string | null,
  "document_number_masked": string | null,
  "dob": string | null,
  "gender": string | null,
  "address": string | null,
  "is_valid_id": boolean,
  "confidence": float
}"""

    if user_message:
        prompt += f"\n\nAdditional user message context: {user_message}"

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                [
                    {"mime_type": mime_type, "data": file_bytes},
                    prompt
                ],
                generation_config={"response_mime_type": "application/json"}
            )
        )

        raw_text = response.text.strip()
        data = json.loads(raw_text)

        # Enforce masked number safety
        raw_doc_num = data.get("document_number_masked") or data.get("document_number")
        id_type = str(data.get("id_type", "unknown")).lower()
        masked_num = _mask_id_number(raw_doc_num, id_type)

        extracted = IDCardExtracted(
            id_type=id_type if id_type in ["aadhaar", "passport", "driving_license", "voter_id", "pan_card", "national_id", "other_id"] else "other_id",
            full_name=data.get("full_name") or None,
            document_number_masked=masked_num,
            dob=data.get("dob") or None,
            gender=data.get("gender") or None,
            address=data.get("address") or None,
            is_valid_id=bool(data.get("is_valid_id", True)),
            confidence=float(data.get("confidence", 0.9)),
        )

        logger.info(
            "ID_CARD_EXTRACTED | type=%s | name=%s | num=%s | conf=%.2f",
            extracted.id_type,
            extracted.full_name,
            extracted.document_number_masked,
            extracted.confidence,
        )
        return extracted

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.error(f"ID card extraction failed: {e}", exc_info=True)
        return IDCardExtracted(
            id_type="unknown",
            is_valid_id=False,
            confidence=0.0,
        )
