"""
Google Vision API and Gemini integration for multimodal image analysis.

Supports:
- Structured payment receipt parsing (returns dict with extracted fields + missing fields).
- General image analysis (room photos, complaints, documents, etc.).
"""
import json
import logging
import re

from app.services.llm import compat as genai

from app.config.settings import get_settings
from app.models.schemas import PaymentDetails

logger = logging.getLogger(__name__)

# All PaymentDetails field names for tracking missing fields
ALL_PAYMENT_FIELDS = list(PaymentDetails.model_fields.keys())

# Fields that are critical and should be asked about if missing
CRITICAL_FIELDS = ["customer_name", "customer_phone", "num_adults", "num_children"]

# Fields that should be asked about if not already in the session context
IMPORTANT_FIELDS = ["check_in", "check_out", "num_rooms", "room_type", "meal_plan"]

# Fields that are optional and can be skipped
OPTIONAL_FIELDS = ["customer_email", "hotel_name", "booking_id", "notes",
                   "room_number", "num_nights", "receiver_name", "sender_name",
                   "payment_app", "timestamp"]


def _detect_mime_type(file_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif file_bytes.startswith(b'\xff\xd8'):
        return "image/jpeg"
    elif file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[8:12]:
        return "image/webp"
    return "image/jpeg"


def _extract_json_from_text(text: str) -> dict | None:
    """Extract a JSON object from LLM response text, handling markdown fences."""
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _compute_missing_fields(payment_data: dict) -> list[str]:
    """Determine which PaymentDetails fields are missing from the extracted data."""
    missing = []
    for field in ALL_PAYMENT_FIELDS:
        value = payment_data.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(field)
    return missing


def _compute_price_match_missing_fields(pm_data: dict) -> list[str]:
    """Determine which required Price Match fields are missing."""
    required = ["hotel_name", "price", "check_in"]
    missing = []
    for field in required:
        val = pm_data.get(field)
        if val is None or (isinstance(val, str) and val.strip() == "") or (isinstance(val, (int, float)) and val <= 0):
            missing.append(field)
    return missing


def parse_payment_image(file_bytes: bytes, user_message: str | None = None) -> dict:
    """
    Analyze an image and return a structured result.

    Returns a dict with:
    - is_payment_receipt: bool
    - payment_data: dict (PaymentDetails fields, null for missing)
    - missing_fields: list[str] (field names not found in the image)
    - image_type: str ("Payment Screenshot", "Room Image", "Complaint", "Document", "General")
    - image_description: str (for non-payment images)
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    try:
        logger.info("Using Gemini model for multimodal image analysis.")

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        prompt = """Analyze this image carefully and determine its type.

STEP 1 — CLASSIFICATION & VALIDATION:
Determine if the image is a valid hotel booking/payment document.
Acceptable booking documents: Hotel booking screenshots, Payment screenshots, Booking confirmation pages, Reservation PDFs, Hotel invoices, Booking platform screenshots, Hotel payment receipts.
Reject irrelevant uploads: Anime, movies, selfies, memes, CVs, bank statements (without hotel context), Aadhaar/PAN, products, social media, random PDFs, blank files.

Classify the image into one of these types:
- "Payment Screenshot" (Valid hotel payment receipt or bank transfer screenshot)
- "Competitor Price Match" (Any hotel booking screenshot, itinerary, or price quote from another website or platform, even if the price is not lower)
- "Room Image" (hotel room, accommodation)
- "Complaint" (dirty room, broken item, damage photo)
- "Document" (ID card, passport, Aadhaar, other document)
- "General" (anything else — selfie, landscape, meme, random image, rejected irrelevant uploads)

STEP 2 — IF PAYMENT SCREENSHOT:
Extract ALL of the following fields. Use null for any field NOT visible in the image:
- amount (float): Payment amount
- transaction_id (string): Transaction ID, UTR number, or reference number
- status (string): One of "SUCCESS", "PENDING", "FAILED", "INVALID"
- sender_name (string): Who sent the money
- receiver_name (string): Who received the money
- payment_app (string): App used (GPay, PhonePe, Paytm, bank name, etc.)
- timestamp (string): Date and time of transaction
- room_type (string): Room type if mentioned
- room_number (string): Room number if mentioned
- num_rooms (int): Number of rooms if mentioned
- num_adults (int): Number of adults if mentioned
- num_children (int): Number of children if mentioned
- num_nights (int): Number of nights if mentioned
- check_in (string): Check-in date if mentioned
- check_out (string): Check-out date if mentioned
- meal_plan (string): Meal plan if mentioned
- customer_name (string): Customer/guest name
- customer_phone (string): Customer phone number
- customer_email (string): Customer email
- booking_id (string): Booking or confirmation ID
- hotel_name (string): Hotel name if mentioned
- notes (list of strings, max 5): Any noteworthy observations not captured above

FAKE / INVALID SCREENSHOT DETECTION:
Check for visual editing artifacts, missing critical fields, mockup elements, or suspicious formatting. If suspicious, set status to "INVALID" and explain in notes.

STEP 3 — IF COMPETITOR PRICE MATCH AND is_hotel_booking_document IS TRUE:
Extract ALL of the following fields. Use null for any field NOT visible. DO NOT guess or calculate.
- hotel_name (string): Name of the hotel
- booking_platform (string): Platform name (e.g. Booking.com, Agoda, MakeMyTrip)
- room_type (string): Type of room
- room_view (string): Room view (e.g. Sea View, City View)
- price (float): Total price shown
- currency (string): Currency code (e.g. INR, USD)
- adults (int): Number of adults
- children (int): Number of children
- check_in (string): Check-in date
- check_out (string): Check-out date
- nights (int): Number of nights
- meal_plan (string): Meal plan included (e.g. Breakfast Included, Room Only)
- tax_included (bool): True if price includes taxes, False if excluded, null if unclear
- cancellation_policy (string): E.g. Refundable, Non-refundable
- screenshot_timestamp (string): Any visible timestamp/time on device
- confidence (float): Your confidence in the extraction (0.0 to 1.0)
- missing_fields (list of strings): List of required fields above that are missing or unclear in the image. DO NOT GUESS.

STEP 4 — IF NOT PAYMENT OR PRICE MATCH:
Provide a brief description of what the image shows.

IF ROOM IMAGE:
Analyze the room and describe its features (bed type, spaciousness, luxury level, view, bathroom, amenities).

IF COMPLAINT:
Describe the exact issue visible in the image.

IF ID OR DOCUMENT (Government ID card, Aadhaar, Passport, Driving License, Voter ID, PAN):
Identify the document type (e.g. Aadhaar Card, Passport), extract cardholder legal name, and mask all but the last 4 digits of document numbers (e.g. XXXX XXXX 1234). DO NOT dump raw unmasked numbers or verbose OCR text. State whether it is a valid government ID.

RESPONDS WITH ONLY THIS JSON (no markdown, no explanation):
{
  "is_hotel_booking_document": true/false,
  "document_type": "BOOKING_SCREENSHOT", "PAYMENT_SCREENSHOT", or "UNKNOWN",
  "document_confidence": 0.0 to 1.0,
  "document_reason": "Reason if rejected/unknown, else null",
  "image_type": "<classification>",
  "is_payment_receipt": true/false,
  "is_price_match": true/false,
  "payment_data": {
    "amount": null,
    "transaction_id": null,
    "status": null,
    "sender_name": null,
    "receiver_name": null,
    "payment_app": null,
    "timestamp": null,
    "room_type": null,
    "room_number": null,
    "num_rooms": null,
    "num_adults": null,
    "num_children": null,
    "num_nights": null,
    "check_in": null,
    "check_out": null,
    "meal_plan": null,
    "customer_name": null,
    "customer_phone": null,
    "customer_email": null,
    "booking_id": null,
    "hotel_name": null,
    "notes": null
  },
  "price_match_data": {
    "hotel_name": null,
    "booking_platform": null,
    "room_type": null,
    "room_view": null,
    "price": null,
    "currency": null,
    "adults": null,
    "children": null,
    "check_in": null,
    "check_out": null,
    "nights": null,
    "meal_plan": null,
    "tax_included": null,
    "cancellation_policy": null,
    "screenshot_timestamp": null,
    "confidence": null,
    "missing_fields": []
  },
  "image_description": "<description for non-payment/price match images, empty string otherwise>"
}"""

        if user_message:
            prompt += f"\n\nContext from User Message: {user_message}"

        response = model.generate_content([
            prompt,
            {
                "mime_type": mime_type,
                "data": file_bytes,
            },
        ])

        raw_text = response.text.strip()
        logger.info("Successfully analyzed image using Gemini.")

        parsed = _extract_json_from_text(raw_text)

        if parsed and isinstance(parsed, dict):
            is_payment = parsed.get("is_payment_receipt", False)
            is_price_match = parsed.get("is_price_match", False)
            image_type = parsed.get("image_type", "General")
            payment_data = parsed.get("payment_data", {}) or {}
            price_match_data = parsed.get("price_match_data", {}) or {}
            image_description = parsed.get("image_description", "")
            
            is_hotel_doc = parsed.get("is_hotel_booking_document", False)
            doc_type = parsed.get("document_type", "UNKNOWN")
            doc_reason = parsed.get("document_reason", "")

            missing_fields = []
            if is_payment:
                missing_fields = _compute_missing_fields(payment_data)
                
            if is_price_match or image_type == "Competitor Price Match":
                pm_missing = price_match_data.get("missing_fields") or []
                computed = _compute_price_match_missing_fields(price_match_data)
                price_match_data["missing_fields"] = list(set(pm_missing + computed))

            return {
                "is_hotel_booking_document": is_hotel_doc,
                "document_type": doc_type,
                "document_reason": doc_reason,
                "is_payment_receipt": is_payment,
                "is_price_match": is_price_match,
                "payment_data": payment_data,
                "price_match_data": price_match_data,
                "missing_fields": missing_fields,
                "image_type": image_type,
                "image_description": image_description,
            }

        logger.warning("Could not parse structured JSON from Gemini image response. Falling back to raw text.")
        return {
            "is_hotel_booking_document": False,
            "document_type": "UNKNOWN",
            "document_reason": "Failed to parse API response",
            "is_payment_receipt": False,
            "is_price_match": False,
            "payment_data": {},
            "price_match_data": {},
            "missing_fields": [],
            "image_type": "General",
            "image_description": raw_text,
        }

    except Exception as e:
        logger.error(f"Failed to process image with Gemini: {e}", exc_info=True)
        raise ValueError("Failed to analyze the image. Please ensure the image is clear.")


def process_image(file_bytes: bytes, user_message: str | None = None) -> str:
    """
    Return image analysis as a formatted [IMAGE_ANALYSIS] block for the chat pipeline.
    """
    result = parse_payment_image(file_bytes, user_message)

    if result["is_payment_receipt"]:
        payment_json = json.dumps(result["payment_data"], indent=2, default=str)
        return (
            f"[IMAGE_ANALYSIS]\n"
            f"Type: Payment Screenshot\n"
            f"Details: {payment_json}\n"
            f"[/IMAGE_ANALYSIS]"
        )

    if result.get("is_price_match") or result.get("image_type") == "Competitor Price Match":
        # Handle rejection explicitly
        if not result.get("is_hotel_booking_document"):
            reason = result.get("document_reason", "Uploaded image is not related to a hotel booking.")
            return (
                f"[IMAGE_ANALYSIS]\n"
                f"Type: Invalid Price Match\n"
                f"Details: {reason}\n"
                f"[/IMAGE_ANALYSIS]"
            )
            
        pm_json = json.dumps(result["price_match_data"], indent=2, default=str)
        return (
            f"[IMAGE_ANALYSIS]\n"
            f"Type: Competitor Price Match\n"
            f"Details: {pm_json}\n"
            f"[/IMAGE_ANALYSIS]"
        )

    image_type = result.get("image_type", "General")
    description = result.get("image_description", "No description available.")

    from app.utils.privacy_scrubber import scrub_id_numbers
    clean_description = scrub_id_numbers(description)
    if any(k in image_type.lower() for k in ("id", "aadhaar", "passport", "document")):
        if any(k in image_type.lower() for k in ("aadhaar", "passport", "identity", "id_card", "id card")):
            image_type = "ID Card"

    return (
        f"[IMAGE_ANALYSIS]\n"
        f"Type: {image_type}\n"
        f"Details: {clean_description}\n"
        f"[/IMAGE_ANALYSIS]"
    )
