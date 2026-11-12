"""
Room Reference Image Analyzer for StayInn Vision Pipeline.
Extracts visible room attributes: bed counts, bed arrangements, amenities, and room layout.
"""
from __future__ import annotations

import logging
from typing import Any
from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError

from app.config.settings import get_settings
from app.services.vision.classify import _detect_mime_type
from app.services.vision.schemas import RoomReferenceExtracted

logger = logging.getLogger(__name__)


async def analyze_room_reference(
    file_bytes: bytes,
    user_message: str | None = None,
    candidate_rooms: list[dict[str, Any]] | None = None,
) -> RoomReferenceExtracted:
    """
    Analyzes a hotel room reference photo to identify bed types, count, visible amenities, and aesthetic.
    When candidate_rooms is provided, grounds multimodal visual matching against the hotel's real room catalog.
    Never hallucinates amenities not clearly visible in the image.
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    import json

    catalog_context = ""
    catalog_instruction = ""
    if candidate_rooms:
        catalog_json = json.dumps(candidate_rooms, indent=2)
        catalog_context = f"\n\nHERE IS OUR HOTEL'S ACTUAL ROOM CATALOG:\n{catalog_json}\n"
        catalog_instruction = """
8. matched_room_id: The exact "room_id" of the single best matching room from our catalog that has the highest visual, aesthetic, and functional similarity.
9. matched_room_name: The exact "name" of that best matching room from our catalog.
10. similarity_score: Integer (0 to 100) representing how closely our catalog room matches the guest's photo.
11. match_explanation: A polite, conversational 2-sentence explanation directly to the guest explaining why our matched room is the closest fit (citing bed size, decor/tier, or key visible features)."""

    prompt = f"""You are an expert hotel room matching specialist analyzing a room photo shared by a prospective guest.

TASK:
Identify and extract the room's visual configuration, bed setup, and visible amenities.{catalog_context}
RULES:
1. bed_count: Count how many distinct beds or mattresses are visible (e.g. 1 for King/Queen, 2 for Twin, 3 for Triple/Family, 4 for Four-bed).
2. bed_type: Classify the bed arrangement:
   - "king": 1 large double/king mattress
   - "queen": 1 standard double/queen mattress
   - "twin": 2 separate single/twin beds side-by-side
   - "single": 1 single bed
   - "triple": 3 beds or 1 double + 1 single bed
   - "family": 4 beds or 2 large double beds
   - "unknown": if beds are not clearly visible
3. room_style: Describe the style/ambiance (e.g. "Modern Deluxe with wooden accents", "Spacious Premier King with warm ambient lighting", "Classic Heritage room").
4. visible_amenities: List ONLY the amenities you can clearly see (e.g. "AC Unit", "Flat-screen TV", "Balcony", "Sofa seating", "Attached Bathroom", "Bathtub", "Study Table", "Electric Kettle", "Wardrobe"). DO NOT guess or list things not visible.
5. room_category_inferred: Suggest what category this resembles (e.g. "Deluxe Room", "Premier King Room", "Family Suite", "Standard Room").
6. features_description: A concise 1-3 sentence summary of what this room looks like.
7. confidence: Float (0.0 to 1.0) indicating clarity of the photo.{catalog_instruction}"""

    if user_message:
        prompt += f"\n\nGuest's comment: {user_message}"

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

        raw_text = response.text.strip()
        data = json.loads(raw_text)
        result = RoomReferenceExtracted(**data)

        logger.info(
            "ROOM_REFERENCE_SUCCESS | beds=%s | bed_type=%s | matched_id=%s | matched_name=%s | similarity=%s",
            result.bed_count,
            result.bed_type,
            result.matched_room_id,
            result.matched_room_name,
            result.similarity_score,
        )
        return result

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.error(f"Room reference analysis failed: {e}", exc_info=True)
        return RoomReferenceExtracted(
            bed_count=None,
            bed_type="unknown",
            room_style="Unclear room photo",
            visible_amenities=[],
            features_description=f"Analysis failed: {str(e)}",
            confidence=0.0,
        )
