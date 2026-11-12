"""
Pydantic Schemas for StayInn Vision Pipeline.

Defines structured models for:
- Classification
- Payment Proof extraction
- Room Reference analysis
- OTA / Competitor Price Match extraction
- Grounded Business Validation
- Vision Actions and Final Result
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ─── CLASSIFICATION ─────────────────────────────────────────────────────────

ImageType = Literal["payment_proof", "room_reference", "ota_price", "id_card", "booking_note", "unknown"]


class ImageClassification(BaseModel):
    image_type: ImageType = Field(
        description="Classified category: 'payment_proof', 'room_reference', 'ota_price', 'id_card', 'booking_note', or 'unknown'"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this classification between 0.0 and 1.0"
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation of the visual evidence leading to this classification"
    )


# ─── EXTRACTED DATA SCHEMAS ──────────────────────────────────────────────────

class BookingNoteExtracted(BaseModel):
    guest_name: str | None = Field(default=None, description="Guest name written on the note/slip")
    phone_number: str | None = Field(default=None, description="Contact/phone number written on the note")
    check_in: str | None = Field(default=None, description="Check-in date in YYYY-MM-DD format")
    check_out: str | None = Field(default=None, description="Check-out date in YYYY-MM-DD format")
    adults: int | None = Field(default=None, description="Number of adults/guests")
    children: int | None = Field(default=None, description="Number of children")
    num_rooms: int | None = Field(default=None, description="Number of rooms requested")
    room_type_preference: str | None = Field(default=None, description="Preferred room type or bed configuration mentioned (e.g. Triple Bed, Deluxe, Non-AC)")
    meal_plan_preference: str | None = Field(default=None, description="Meal plan requested (e.g. EP, CP, MAP, AP, with breakfast)")
    special_requests: str | None = Field(default=None, description="Special requests or remarks (e.g. early check-in, ground floor, parking, cab)")
    budget: float | None = Field(default=None, description="Budget or agreed rate mentioned on note")
    lead_id: str | None = Field(default=None, description="Generated CRM Lead ID if converted to lead")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_notes: str | None = Field(default=None, description="Raw transcription of the handwritten or written note")

class IDCardExtracted(BaseModel):
    id_type: Literal[
        "aadhaar",
        "passport",
        "driving_license",
        "voter_id",
        "pan_card",
        "national_id",
        "other_id",
        "unknown",
    ] = Field(
        default="unknown",
        description="Type of government identity card (aadhaar, passport, driving_license, voter_id, pan_card, national_id)"
    )
    full_name: str | None = Field(default=None, description="Full name of the guest as shown on the identity card")
    document_number_masked: str | None = Field(
        default=None,
        description="Masked document number with only last 4 digits visible for privacy (e.g. XXXX XXXX 2163)"
    )
    dob: str | None = Field(default=None, description="Date of birth if visible (DD/MM/YYYY)")
    gender: str | None = Field(default=None, description="Gender (MALE, FEMALE, OTHER)")
    address: str | None = Field(default=None, description="Address or city/state if visible")
    is_valid_id: bool = Field(default=True, description="True if it appears to be a legitimate government ID document")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class PaymentProofExtracted(BaseModel):
    amount: float | None = Field(default=None, description="Payment amount in INR")
    transaction_id: str | None = Field(default=None, description="UTR, UPI Ref number, or transaction ID")
    status: Literal["SUCCESS", "PENDING", "FAILED", "INVALID"] = Field(
        default="SUCCESS",
        description="Visible payment status"
    )
    sender_name: str | None = Field(default=None, description="Payer name if visible")
    sender_upi: str | None = Field(default=None, description="Sender UPI ID / VPA if visible")
    receiver_name: str | None = Field(default=None, description="Receiver name or merchant name if visible")
    receiver_upi: str | None = Field(default=None, description="Receiver UPI ID if visible")
    payment_app: str | None = Field(default=None, description="Payment application (e.g. GPay, PhonePe, Paytm, BHIM, Bank)")
    timestamp: str | None = Field(default=None, description="Date/time of transaction as shown in screenshot")
    booking_id: str | None = Field(default=None, description="Booking ID if explicitly mentioned")
    customer_phone: str | None = Field(default=None, description="Phone number if mentioned")
    notes: list[str] = Field(default_factory=list, description="Any notable observations or visual flags")
    is_suspicious: bool = Field(default=False, description="True if signs of tampering, fake mockup, or altered text")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RoomReferenceExtracted(BaseModel):
    bed_count: int | None = Field(default=None, description="Estimated number of beds visible in the room")
    bed_type: Literal["king", "queen", "twin", "single", "triple", "family", "unknown"] = Field(
        default="unknown",
        description="Type of bed arrangement visible"
    )
    room_style: str | None = Field(default=None, description="Style or vibe (e.g. modern deluxe, traditional, budget, family suite)")
    visible_amenities: list[str] = Field(
        default_factory=list,
        description="Amenities clearly visible (e.g. AC unit, TV, bathtub, sofa, balcony, study desk)"
    )
    room_category_inferred: str | None = Field(
        default=None,
        description="Inferred category (e.g. Deluxe Room, Standard Room, Family Room, Premier Room)"
    )
    features_description: str = Field(
        default="",
        description="Detailed objective description of the room's interior, arrangement, and condition"
    )
    matched_room_id: str | None = Field(
        default=None,
        description="The room_id from candidate_rooms that best matches this photo"
    )
    matched_room_name: str | None = Field(
        default=None,
        description="The room_type/name of the matched candidate room"
    )
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Visual and functional similarity score (0 to 100) with matched room"
    )
    match_explanation: str | None = Field(
        default=None,
        description="Detailed conversational explanation why this candidate room was selected as the closest match"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OTAPriceExtracted(BaseModel):
    booking_platform: str | None = Field(
        default=None,
        description="Platform name (e.g. MakeMyTrip, Goibibo, Booking.com, Agoda, Expedia)"
    )
    hotel_name: str | None = Field(default=None, description="Hotel name shown in screenshot")
    room_type: str | None = Field(default=None, description="Room category name shown in screenshot")
    price: float | None = Field(default=None, description="Total displayed price amount")
    currency: str = Field(default="INR", description="Currency code (e.g. INR)")
    tax_included: bool | None = Field(
        default=True,
        description="True if price includes taxes/fees, False if excluded, null if unclear"
    )
    tax_amount: float | None = Field(default=None, description="Explicit tax amount if visible")
    check_in: str | None = Field(default=None, description="Check-in date if visible (YYYY-MM-DD or readable format)")
    check_out: str | None = Field(default=None, description="Check-out date if visible")
    nights: int | None = Field(default=None, description="Number of nights")
    adults: int | None = Field(default=None, description="Number of adults")
    children: int | None = Field(default=None, description="Number of children")
    meal_plan: str | None = Field(default=None, description="Meal plan (e.g. Room Only, Free Breakfast)")
    cancellation_policy: str | None = Field(default=None, description="Cancellation policy (e.g. Free Cancellation, Non-refundable)")
    screenshot_timestamp: str | None = Field(default=None, description="Timestamp or battery bar time on screenshot")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list, description="Critical comparison fields that are missing/unclear")


# ─── VALIDATION AND ACTION SCHEMAS ──────────────────────────────────────────

ValidationStatus = Literal[
    "valid",
    "needs_clarification",
    "duplicate_utr",
    "amount_mismatch",
    "unsupported_platform",
    "below_floor",
    "policy_exceeded",
    "escalate",
    "rejected",
    # A receipt arrived with no balance on record to compare it against
    # (expected_amount <= 0), so there is nothing to call a mismatch. Distinct
    # from "escalate": the action taken is a front-desk task, and the booking is
    # provisionally confirmed, not blocked.
    "needs_staff_review",
]

ActionType = Literal[
    "confirm_payment",
    "ask_clarification",
    "suggest_room",
    "search_rooms",
    "negotiate_price",
    "verify_id_card",
    "create_lead",
    "escalate",
    "no_action",
]


class ValidationResult(BaseModel):
    status: ValidationStatus = Field(default="valid")
    reason: str = Field(default="", description="Detailed reason for validation status")
    discrepancies: list[str] = Field(default_factory=list, description="List of specific mismatches or issues")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context from validation")


class VisionAction(BaseModel):
    type: ActionType = Field(default="no_action")
    message: str = Field(default="", description="Customer-facing response message")
    db_updates: dict[str, Any] = Field(
        default_factory=dict,
        description="State or database changes to commit (e.g. booking_stage, payment_status, total_price)"
    )
    escalation_reason: str | None = Field(default=None, description="Staff note if action is escalate")


class VisionResult(BaseModel):
    image_type: ImageType = Field(default="unknown")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extracted: dict[str, Any] = Field(default_factory=dict)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    action: VisionAction = Field(default_factory=VisionAction)
