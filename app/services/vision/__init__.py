"""
Vision and Multimodal Services Package for StayInn AI.
"""
from app.services.vision.schemas import (
    VisionResult,
    ImageClassification,
    PaymentProofExtracted,
    RoomReferenceExtracted,
    OTAPriceExtracted,
    ValidationResult,
    VisionAction,
)
from app.services.vision.classify import classify_image
from app.services.vision.payment_proof import extract_payment_proof
from app.services.vision.room_reference import analyze_room_reference
from app.services.vision.ota_price import extract_ota_price
from app.services.vision.validator import (
    validate_payment,
    validate_room_reference,
    validate_ota_price_match,
)
from app.services.vision.image_handler import (
    process_image_pipeline,
    format_vision_analysis_block,
)

__all__ = [
    "VisionResult",
    "ImageClassification",
    "PaymentProofExtracted",
    "RoomReferenceExtracted",
    "OTAPriceExtracted",
    "ValidationResult",
    "VisionAction",
    "classify_image",
    "extract_payment_proof",
    "analyze_room_reference",
    "extract_ota_price",
    "validate_payment",
    "validate_room_reference",
    "validate_ota_price_match",
    "process_image_pipeline",
    "format_vision_analysis_block",
]
