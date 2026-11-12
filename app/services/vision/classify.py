"""
Fast Multimodal Image Classifier for StayInn Vision Pipeline.
Uses Gemini multimodal models with structured Pydantic schema output.
"""
from __future__ import annotations

import logging
from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError

from app.config.settings import get_settings
from app.services.vision.schemas import ImageClassification, ImageType

logger = logging.getLogger(__name__)


def _detect_mime_type(file_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif file_bytes.startswith(b'\xff\xd8'):
        return "image/jpeg"
    elif file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[8:12]:
        return "image/webp"
    return "image/jpeg"


async def classify_image(file_bytes: bytes, user_message: str | None = None) -> ImageClassification:
    """
    Classifies an image into one of 4 canonical types:
    - 'payment_proof': UPI payment screenshot, bank transfer receipt, or transaction confirmation
    - 'room_reference': Guest room interior, bed setup, bathroom, room photo
    - 'ota_price': Screenshot of an OTA / hotel booking platform (MMT, Goibibo, Agoda, Booking.com) showing rates/dates
    - 'unknown': Selfies, memes, non-hotel documents, random objects, or corrupted images
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    prompt = """You are an expert visual classifier for a hotel AI assistant.
Analyze this uploaded image and classify it into EXACTLY ONE of the following 6 categories:

1. "payment_proof":
   - A digital payment receipt or transaction confirmation screen (e.g. Google Pay, PhonePe, Paytm, BHIM, UPI, net banking, debit/credit card receipt).
   - Shows transaction amount, UTR / reference number, payment status, or bank transfer details.

2. "id_card":
   - An official Government Identity Document (e.g. Indian Aadhaar Card, Passport, Driver's License, Voter ID, PAN Card, National Identity Card).
   - Shows guest name, photo, government emblem/logo (e.g. Government of India, UIDAI, Ashoka lion emblem), ID number, address, or DOB.
   - Provided by a guest for identity verification or check-in compliance.

3. "booking_note":
   - A handwritten note on paper, handwritten slip, written reservation requirements, WhatsApp message/chat screenshot of booking inquiry, or paper diary note.
   - Contains dates (check-in/check-out), guest counts (adults/children/persons), room count, or room types (e.g. "Check in - 5/11/26", "Tripple bed - 2 NOS", "6 Person").

4. "room_reference":
   - A photograph of a hotel room interior, bedroom, bed setup (e.g. 1 king bed, twin beds, 3 beds), bathroom, living area, balcony, or guest room amenities.
   - Used by a guest to ask if the hotel has similar rooms or to show what style/beds they prefer.

5. "ota_price":
   - A screenshot from an Online Travel Agency (OTA) or hotel booking website/app (e.g. MakeMyTrip, Goibibo, Booking.com, Agoda, Expedia, TripAdvisor).
   - Shows hotel room rates, dates, prices, room category names, or price comparisons.

6. "unknown":
   - Anything else: selfies, food, outdoor landscapes, memes, unrelated receipts, or blank/corrupted images.

Provide a confidence score (0.0 to 1.0) and brief visual reasoning."""

    if user_message:
        prompt += f"\n\nContext from user's accompanying message: {user_message}"

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        prompt += "\n\nRespond ONLY with a JSON object:\n{\n  \"image_type\": \"payment_proof\" | \"id_card\" | \"booking_note\" | \"room_reference\" | \"ota_price\" | \"unknown\",\n  \"confidence\": float (0.0 to 1.0),\n  \"reasoning\": string\n}"

        response = await model.generate_content_async(
            [
                prompt,
                {"mime_type": mime_type, "data": file_bytes},
            ],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        import json
        raw_text = response.text.strip()
        data = json.loads(raw_text)
        classification = ImageClassification(**data)
        logger.info(
            "IMAGE_CLASSIFICATION_SUCCESS | type=%s | confidence=%.2f | reason=%s",
            classification.image_type,
            classification.confidence,
            classification.reasoning,
        )
        return classification

    except LLMUnavailableError:
        # The provider is down, not the image. The heuristic fallback below would
        # answer "unknown", and the guest would be told their screenshot "doesn't
        # appear to be a payment receipt" — a lie about their own photo. Let the
        # pipeline say the model is unavailable instead.
        raise
    except Exception as e:
        logger.warning(f"Image classification failed with structured schema: {e}. Falling back to heuristic.")
        # Fallback if API fails or schema isn't accepted
        return ImageClassification(
            image_type="unknown",
            confidence=0.0,
            reasoning=f"Classification error: {str(e)}",
        )
