"""
Top-level Orchestrator for the StayInn Vision Pipeline.
Coordinates:
1. Classification (classify.py)
2. Extraction (payment_proof.py, room_reference.py, ota_price.py)
3. Grounded Business Validation (validator.py)
4. Structured Output Construction (schemas.py)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.config.settings import get_settings
from app.services.llm_errors import LLMUnavailableError
from app.services.vision.booking_note import extract_booking_note
from app.services.vision.classify import classify_image
from app.services.vision.id_card import extract_id_card
from app.services.vision.ota_price import extract_ota_price
from app.services.vision.payment_proof import extract_payment_proof
from app.services.vision.room_reference import analyze_room_reference
from app.services.vision.schemas import (
    ImageClassification,
    ValidationResult,
    VisionAction,
    VisionResult,
)
from app.services.vision.validator import (
    validate_booking_note,
    validate_id_card,
    validate_ota_price_match,
    validate_payment,
    validate_room_reference,
)

logger = logging.getLogger(__name__)


async def process_image_pipeline(
    file_bytes: bytes,
    filename: str | None = None,
    session_state: dict[str, Any] | None = None,
    hotel_id: str | int | None = None,
    user_message: str | None = None,
) -> VisionResult:
    """Public entry point. Adds one thing to the pipeline below: an honest
    answer when the model provider is unreachable.

    Every extraction step degrades on its own errors (a blurry photo, a prompt
    the model mangles) and those degradations are meaningful. A provider outage
    is not: the extractors return zero-confidence objects, the validators read
    them as bad uploads, and the guest is told their payment receipt "doesn't
    appear to be a payment receipt". That is a wrong answer about *their*
    document when the fault is ours, so it is caught here instead.
    """
    try:
        return await _run_image_pipeline(
            file_bytes=file_bytes,
            filename=filename,
            session_state=session_state,
            hotel_id=hotel_id,
            user_message=user_message,
        )
    except LLMUnavailableError as exc:
        logger.error("VISION_PIPELINE_LLM_UNAVAILABLE | file=%s | %s", filename, exc)
        # The hotel's own number, not "type agent": during an outage the next
        # text turn is answered by the same outage, so an instruction the model
        # has to classify is an instruction that cannot be followed.
        from app.services.ticket_service import get_formatted_contact_info

        contact = await get_formatted_contact_info(hotel_id or get_settings().HOTEL_ID)
        return VisionResult(
            image_type="unknown",
            confidence=0.0,
            extracted={},
            validation=ValidationResult(
                status="rejected",
                reason=f"Model provider unavailable: {exc}",
            ),
            action=VisionAction(
                type="no_action",
                message=(
                    "Sorry — I'm having trouble reading images at the moment, and that's on my side, "
                    "not your upload. Please send it again in a few minutes.\n\n"
                    f"If it's urgent: {contact}"
                ),
            ),
        )


async def _run_image_pipeline(
    file_bytes: bytes,
    filename: str | None = None,
    session_state: dict[str, Any] | None = None,
    hotel_id: str | int | None = None,
    user_message: str | None = None,
) -> VisionResult:
    """
    End-to-end multimodal image processing pipeline.
    Returns a fully validated, schema-compliant VisionResult object.
    """
    settings = get_settings()
    hid = hotel_id or settings.HOTEL_ID
    state = session_state or {}

    logger.info("VISION_PIPELINE_START | file=%s | hotel_id=%s", filename, hid)

    # STEP 1: Fast Multimodal Classification
    classification: ImageClassification = await classify_image(file_bytes, user_message=user_message)
    img_type = classification.image_type
    conf = classification.confidence

    logger.info("VISION_PIPELINE_CLASSIFIED | type=%s | confidence=%.2f", img_type, conf)

    # STEP 2 & 3: Extraction and Grounded Business Validation
    if img_type == "payment_proof":
        extracted_payment = await extract_payment_proof(file_bytes, user_message=user_message)
        val_res, action = await validate_payment(extracted_payment, state, hid)
        return VisionResult(
            image_type="payment_proof",
            confidence=extracted_payment.confidence or conf,
            extracted=extracted_payment.model_dump(),
            validation=val_res,
            action=action,
        )

    elif img_type == "id_card":
        extracted_id = await extract_id_card(file_bytes, user_message=user_message)
        val_res, action = await validate_id_card(extracted_id, hid, state)
        return VisionResult(
            image_type="id_card",
            confidence=extracted_id.confidence or conf,
            extracted=extracted_id.model_dump(),
            validation=val_res,
            action=action,
        )

    elif img_type == "room_reference":
        from app.repositories import room_repository
        all_rooms = await room_repository.get_all_rooms(hid)
        catalog_candidates = []
        for r in all_rooms:
            catalog_candidates.append({
                "room_id": r.room_id,
                "name": r.name,
                "bed_type": getattr(r, "bed_type", None),
                "description": (r.description or "").strip()[:400],
                "amenities": [k for k, v in r.amenities.items() if v] if isinstance(r.amenities, dict) else (r.amenities or []),
                "max_occupancy": getattr(r.occupancy, "max_occupancy", 2) if hasattr(r, "occupancy") else 2,
                "base_price": r.rate_plans[0].base_price if r.rate_plans else getattr(r, "base_price", 0.0),
            })
        extracted_room = await analyze_room_reference(
            file_bytes,
            user_message=user_message,
            candidate_rooms=catalog_candidates,
        )
        val_res, action = await validate_room_reference(
            extracted_room,
            hid,
            state,
            all_rooms=all_rooms,
        )
        return VisionResult(
            image_type="room_reference",
            confidence=extracted_room.confidence or conf,
            extracted=extracted_room.model_dump(),
            validation=val_res,
            action=action,
        )

    elif img_type == "ota_price":
        extracted_ota = await extract_ota_price(file_bytes, user_message=user_message)
        val_res, action = await validate_ota_price_match(extracted_ota, hid, state)
        return VisionResult(
            image_type="ota_price",
            confidence=extracted_ota.confidence or conf,
            extracted=extracted_ota.model_dump(),
            validation=val_res,
            action=action,
        )

    elif img_type == "booking_note":
        extracted_note = await extract_booking_note(file_bytes, user_message=user_message)
        val_res, action = await validate_booking_note(extracted_note, hid, state)
        return VisionResult(
            image_type="booking_note",
            confidence=extracted_note.confidence or conf,
            extracted=extracted_note.model_dump(),
            validation=val_res,
            action=action,
        )

    else:
        # Unknown / General / Irrelevant upload
        return VisionResult(
            image_type="unknown",
            confidence=conf,
            extracted={},
            validation=ValidationResult(
                status="rejected",
                reason="Uploaded image is not a payment proof, ID document, room reference, or hotel rate screenshot.",
            ),
            action=VisionAction(
                type="no_action",
                message=(
                    "I received your image, but it doesn't appear to be a payment receipt, ID document, room photo, or hotel booking screenshot. "
                    "How can I assist you with your hotel reservation today? 🏨"
                ),
            ),
        )


def format_vision_analysis_block(vision_res: VisionResult) -> str:
    """
    Serializes a VisionResult into the structured [IMAGE_ANALYSIS] block
    for backward compatibility with existing LangGraph nodes.
    """
    img_type = vision_res.image_type
    extracted = vision_res.extracted

    if img_type == "payment_proof":
        details_str = json.dumps(extracted, indent=2, default=str)
        return (
            f"[IMAGE_ANALYSIS]\n"
            f"Type: Payment Screenshot\n"
            f"Details: {details_str}\n"
            f"[/IMAGE_ANALYSIS]"
        )

    elif img_type == "ota_price":
        details_str = json.dumps(extracted, indent=2, default=str)
        return (
            f"[IMAGE_ANALYSIS]\n"
            f"Type: Competitor Price Match\n"
            f"Details: {details_str}\n"
            f"[/IMAGE_ANALYSIS]"
        )

    elif img_type == "room_reference":
        desc = extracted.get("features_description") or "Guest room reference photo"
        return (
            f"[IMAGE_ANALYSIS]\n"
            f"Type: Room Image\n"
            f"Details: {desc}\n"
            f"[/IMAGE_ANALYSIS]"
        )

    else:
        return (
            f"[IMAGE_ANALYSIS]\n"
            f"Type: General\n"
            f"Details: General or unrecognized image\n"
            f"[/IMAGE_ANALYSIS]"
        )
