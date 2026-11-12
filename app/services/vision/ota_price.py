"""
OTA / Competitor Price Screenshot Extractor for StayInn Vision Pipeline.
Extracts pricing, platform, tax breakdowns, room categories, and dates from OTA screenshots.
"""
from __future__ import annotations

import logging
from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError

from app.config.settings import get_settings
from app.services.vision.classify import _detect_mime_type
from app.services.vision.schemas import OTAPriceExtracted

logger = logging.getLogger(__name__)


async def extract_ota_price(
    file_bytes: bytes,
    user_message: str | None = None,
) -> OTAPriceExtracted:
    """
    Extracts structured competitor rates and booking parameters from an OTA screenshot.
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    prompt = """You are a hotel rate comparison and OTA screenshot parser.

TASK:
Analyze the uploaded screenshot from a hotel booking app/website (e.g. MakeMyTrip, Goibibo, Booking.com, Agoda, Expedia) and extract all rate details with high precision.

REQUIRED FIELDS:
1. booking_platform: The OTA platform name (e.g. "MakeMyTrip", "Goibibo", "Booking.com", "Agoda", "Expedia", "EaseMyTrip").
2. hotel_name: The name of the property/hotel shown on screen.
3. room_type: The specific room category name (e.g. "Deluxe Room King Bed", "Standard Room", "Executive Suite").
4. price: The total price number shown (e.g. 2300.0 or 4600.0). If multiple prices exist, extract the FINAL payable / bold total amount.
5. currency: Currency symbol/code (e.g. "INR" or "₹").
6. tax_included: Boolean. True if the price explicitly states "+ taxes included", False if it says "+ ₹XYZ taxes & fees", or null if not stated.
7. tax_amount: If taxes/fees are stated separately, extract the tax amount.
8. check_in: Check-in date shown in screenshot (e.g. "2026-09-10" or "10 Sep").
9. check_out: Check-out date shown in screenshot.
10. nights: Number of nights of stay if visible.
11. adults: Number of adult guests.
12. children: Number of children.
13. meal_plan: Meal plan included (e.g. "Room Only", "Free Breakfast / CP", "All Meals").
14. cancellation_policy: e.g. "Free Cancellation" or "Non-Refundable".
15. screenshot_timestamp: Time / battery bar / date visible on customer's phone status bar.
16. confidence: Float (0.0 to 1.0) indicating clarity of the screenshot.
17. missing_fields: List of any required comparison fields that could not be determined."""

    if user_message:
        prompt += f"\n\nCustomer message: {user_message}"

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        prompt += "\n\nRespond ONLY with a valid JSON object matching all requested fields."
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
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            ),
        )

        import json
        raw_text = response.text.strip()
        data = json.loads(raw_text)
        result = OTAPriceExtracted(**data)

        logger.info(
            "OTA_EXTRACTION_SUCCESS | platform=%s | hotel=%s | price=%s | tax_inc=%s",
            result.booking_platform,
            result.hotel_name,
            result.price,
            result.tax_included,
        )
        return result

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.error(f"OTA price extraction failed: {e}", exc_info=True)
        return OTAPriceExtracted(
            booking_platform=None,
            hotel_name=None,
            room_type=None,
            price=None,
            confidence=0.0,
            missing_fields=[f"Extraction failed: {str(e)}"],
        )
