"""
ID Verification Engine — DPDP Compliant.

Processes government ID documents (Aadhaar, Passport, Driving License, Voter ID, PAN).
CRITICAL PRIVACY RULE:
- NEVER logs or stores full ID numbers.
- ONLY stores the last 4 digits (e.g., "5678").
- Masks all sensitive numbers before database persistence.
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

ID_VERIFICATION_PROMPT = """You are a strict compliance and identity document verification system for a hotel check-in desk.
Analyze the provided government ID document image or PDF page.

CRITICAL DPDP COMPLIANCE INSTRUCTIONS:
1. Extract the document type (one of: "aadhaar", "passport", "driving_license", "voter_id", "pan", "other").
2. Extract the guest's full name as written on the ID.
3. For the ID number: DO NOT RETURN THE FULL NUMBER. Return ONLY the LAST 4 DIGITS of the ID number (e.g., if number is 1234 5678 9012, return "9012").
4. Extract Date of Birth (DOB) if present.
5. Assess readability (is the text clear and legible?).
6. Identify any image quality issues (e.g., "blurry", "glare", "cropped", "low_resolution", "dark", or null if fine).

OUTPUT FORMAT: Return a valid JSON object ONLY:
```json
{
  "id_type": "aadhaar",
  "name": "Amit Sharma",
  "id_number_last4": "9012",
  "dob": "1990-05-12",
  "readable": true,
  "quality_issue": null,
  "verified": true
}
```
"""


def _mask_id_number(raw_id: str | None) -> str | None:
    """Strict Python-level guard ensuring only the last 4 digits are retained."""
    if not raw_id:
        return None
    # Extract only digits and alphanumeric chars
    clean = re.sub(r"[^a-zA-Z0-9]", "", str(raw_id).strip())
    if len(clean) >= 4:
        return clean[-4:]
    return clean


async def verify_id_document(
    file_bytes: bytes,
    filename: str = "",
    declared_mime: str = ""
) -> dict[str, Any]:
    """
    Analyzes an ID document image/PDF and returns DPDP-compliant verification metadata.
    """
    mime_type = declared_mime or _detect_mime_type(file_bytes)
    size_bytes = len(file_bytes)
    logger.info("ID_VERIFY_START | mime=%s | size_bytes=%s | filename=%s", mime_type, size_bytes, filename)

    fallback_result: dict[str, Any] = {
        "id_type": "other",
        "name": None,
        "id_number_last4": None,
        "dob": None,
        "readable": False,
        "quality_issue": "processing_error",
        "verified": False,
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
            [image_part, ID_VERIFICATION_PROMPT],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        if not response or not response.text:
            logger.warning("ID_VERIFY_EMPTY_RESPONSE | mime=%s", mime_type)
            return fallback_result

        parsed = _extract_json_from_text(response.text)
        if not parsed or not isinstance(parsed, dict):
            logger.warning("ID_VERIFY_PARSE_FAIL | raw=%s", response.text[:100])
            return fallback_result

        # Apply strict DPDP masking on id_number_last4
        raw_last4 = parsed.get("id_number_last4")
        masked_last4 = _mask_id_number(raw_last4)

        id_type = str(parsed.get("id_type") or "other").lower().strip()
        readable = bool(parsed.get("readable", True))
        verified = bool(parsed.get("verified", readable and masked_last4 is not None))

        result = {
            "id_type": id_type,
            "name": parsed.get("name"),
            "id_number_last4": masked_last4,
            "dob": parsed.get("dob"),
            "readable": readable,
            "quality_issue": parsed.get("quality_issue"),
            "verified": verified,
        }

        logger.info(
            "ID_VERIFY_RESULT | id_type=%s | last4=%s | readable=%s | verified=%s",
            id_type,
            masked_last4,
            readable,
            verified,
        )
        return result

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.error("ID_VERIFY_ERROR | err=%s", e, exc_info=True)
        return fallback_result
