"""
Payment Proof / UPI Screenshot Extractor for StayInn Vision Pipeline.
Extracts structured payment data and flags tampered or fake receipts.
"""
from __future__ import annotations

import logging
from app.services.llm import compat as genai
from app.services.llm_errors import LLMUnavailableError

from app.config.settings import get_settings
from app.services.vision.classify import _detect_mime_type
from app.services.vision.schemas import PaymentProofExtracted

logger = logging.getLogger(__name__)


async def extract_payment_proof(
    file_bytes: bytes,
    user_message: str | None = None,
) -> PaymentProofExtracted:
    """
    Extracts structured payment fields from a payment screenshot.
    Performs forensic inspection for signs of tampering, mockup generators, or invalid statuses.
    """
    settings = get_settings()
    mime_type = _detect_mime_type(file_bytes)

    prompt = """You are an expert financial OCR and fraud detection system analyzing a digital payment receipt or UPI screenshot.

TASK:
Extract all visible transaction details with 100% precision. DO NOT guess or hallucinate any numbers or names.

SUPPORTED PAYMENT SYSTEMS:
- UPI apps: Google Pay, PhonePe, Paytm, BHIM, Cred, Amazon Pay, WhatsApp Pay.
- Indian Banking apps: ICICI Bank (iMobile / Pockets), HDFC Bank (PayZapp / MobileBanking), SBI (YONO / UPI), Axis Mobile, Kotak, Bank of Baroda, PNB, etc.
- Merchant Aggregators / QR receipts: Pine Labs (e.g. handles ending in @pinelabs), BharatPe, Paytm QR.

CRITICAL APP LAYOUT NOTES:
- ICICI Bank / iMobile: Typically features a red/maroon header or card banner with "i ICICI Bank" logo, a white card container, a green circular checkmark badge (✓), "Paid to [Merchant/Hotel]", "From [Sender Name / UPI]", transaction reference ("Tr. ID" or "Ref No"), and a UPI logo at the bottom. This is an authentic bank transfer layout.
- Third-Party Payers: Guests often have friends, family members, or colleagues pay on their behalf (e.g. "From ATMA PRAKASH GUPTA"). Extract the sender name accurately. A difference between the sender name and the booking guest name is completely NORMAL in hotel hospitality and must NOT be marked suspicious or invalid.
- WhatsApp & Mobile Compression: Normal JPEG compression, WhatsApp compression noise, pixelation from camera photos of screens (Moiré), or slight blur does NOT indicate fraud.
- ONLY set is_suspicious=true if there is blatant forgery (such as cut-and-paste digits of mismatched font/resolution, or unmistakable fake receipt generator watermarks).

REQUIRED FIELDS:
1. amount: The exact total payment amount as a floating point number (e.g. 2000.0, 10200.0). Look for amounts prefixed with ₹, Rs., INR, or numbers in bold near "Paid", "Debited", or "Total". If unreadable or missing, return null.
2. transaction_id: The complete UTR number, UPI transaction ID, bank reference number, or Tr. ID (e.g. "110747975140", "482910394829", "T2609230844273571778160").
3. status:
   - "SUCCESS": Completed, Paid, Successful, Green checkmark badge, Money transferred.
   - "PENDING": Processing, In Progress, Awaiting confirmation.
   - "FAILED": Failed, Declined, Expired, Payment Cancelled.
   - "INVALID": Non-payment document (e.g. a selfie, meme, landscape), or confirmed fake template mockup.
4. sender_name: Name of the person who sent the money (e.g. "ATMA PRAKASH GUPTA").
5. sender_upi: Sender UPI VPA / ID (e.g. "9452712335.etb@icici", "guest@okhdfcbank") if visible.
6. receiver_name: Name of the merchant, payee, or hotel receiving the funds (e.g. "Sample Hotel", "Sample Property").
7. receiver_upi: Receiver UPI VPA / QR ID if visible (e.g. "pimslubc.ssq4249847@pinelabs").
8. payment_app: Application used (e.g. "ICICI Bank", "PhonePe", "Google Pay", "Paytm", "BHIM", "Pine Labs", "HDFC Bank", "SBI YONO").
9. timestamp: Exact date and time shown on the receipt (e.g. "12:28AM, 24 Sep '26 IST", "28 Aug 2026, 03:45 PM").
10. is_suspicious: Set to true ONLY if there are clear signs of image editing or forgery. Defaults to false.
11. notes: List of bullet points describing visual details (e.g. "ICICI Bank green tick verified", "Tr. ID 110747975140 visible", "Pine Labs merchant QR handle").
12. confidence: Float between 0.0 and 1.0 reflecting how clear and legible the image is."""

    if user_message:
        prompt += f"\n\nAccompanying message from customer: {user_message}"

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
        result = PaymentProofExtracted(**data)

        logger.info(
            "PAYMENT_EXTRACTION_SUCCESS | amount=%s | utr=%s | status=%s | suspicious=%s",
            result.amount,
            result.transaction_id,
            result.status,
            result.is_suspicious,
        )
        return result

    except LLMUnavailableError:
        # The provider is down. The degraded object below would be read by
        # the validator as a bad upload and the guest told their own
        # document is invalid — so let the pipeline report the outage.
        raise
    except Exception as e:
        logger.error(f"Payment proof extraction failed: {e}", exc_info=True)
        return PaymentProofExtracted(
            amount=None,
            transaction_id=None,
            status="PENDING",
            is_suspicious=False,
            notes=[f"OCR processing failed: {str(e)}"],
            confidence=0.0,
        )
