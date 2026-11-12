"""
Email Response Generation Service — handles context retrieval, RAG, booking lookup,
live inventory, and professional hospitality email generation.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.db.mongodb import bookings_collection, guests_collection
from app.models.email_schemas import EmailMessage, EmailReplyRequest, EmailReplyResponse
from app.rag.chroma_service import query_knowledge
from app.repositories import booking_repository
from app.repositories.hotel_repository import (
    get_formatted_hotel_policies,
    get_hotel_contact_info,
    get_hotel_details,
    get_standard_stay_times,
)
from app.services.date_extractor import extract_stay_dates
from app.services.gemini_service import _generate_content_async, _get_model
from app.services.inventory_service import get_available_rooms
from app.services.llm import compat as genai

logger = logging.getLogger(__name__)

_BOOKING_REF_RE = re.compile(r"\b(?:BK|SAHU|PAY)-[A-Za-z0-9-]+\b", re.IGNORECASE)


def _extract_booking_id_from_text(text: str) -> str | None:
    """Extract standard booking references (BK-..., SAHU-..., PAY-...) from text."""
    if not text:
        return None
    matches = _BOOKING_REF_RE.findall(text)
    if matches:
        return matches[0].upper()
    return None


def _format_booking_details(booking: dict[str, Any]) -> str:
    """Format a verified booking record into clear facts for the LLM prompt."""
    bid = booking.get("booking_id") or "N/A"
    status = booking.get("booking_status") or "confirmed"
    cin = booking.get("check_in") or "N/A"
    cout = booking.get("check_out") or "N/A"
    adults = booking.get("adults", 0)
    children = booking.get("children", 0)
    total_price = booking.get("total_price")
    amount_paid = booking.get("amount_paid", 0.0)
    payment_status = booking.get("payment_status", "unknown")

    # Rooms
    room_names = []
    rooms = booking.get("rooms") or []
    if isinstance(rooms, list):
        for r in rooms:
            if isinstance(r, dict):
                rname = r.get("room_name") or r.get("room_type")
                if rname:
                    room_names.append(str(rname))
            elif hasattr(r, "room_name"):
                room_names.append(str(r.room_name))
    room_str = ", ".join(room_names) if room_names else booking.get("room_type") or "Room"

    lines = [
        f"• Booking Reference: {bid}",
        f"• Reservation Status: {status.upper()}",
        f"• Check-in Date: {cin}",
        f"• Check-out Date: {cout}",
        f"• Guests: {adults} Adult(s)" + (f", {children} Child(ren)" if children else ""),
        f"• Room(s) Booked: {room_str}",
        f"• Total Amount: ₹{total_price:,.2f}" if total_price is not None else "• Total Amount: N/A",
        f"• Payment Status: {payment_status.upper()} (Amount Paid: ₹{amount_paid:,.2f})",
    ]
    return "\n".join(lines)


async def _find_guest_bookings_by_email(hotel_id: int | str, email: str) -> list[dict[str, Any]]:
    """Look up existing reservations associated with the sender's email."""
    if not email or not email.strip():
        return []
    email_clean = email.strip().lower()
    coll = bookings_collection()
    h_variants = [hotel_id, str(hotel_id), int(hotel_id)] if str(hotel_id).isdigit() else [hotel_id]

    query = {
        "hotel_id": {"$in": h_variants},
        "$or": [
            {"customer_email": {"$regex": f"^{re.escape(email_clean)}$", "$options": "i"}},
            {"guest_email": {"$regex": f"^{re.escape(email_clean)}$", "$options": "i"}},
        ],
    }
    cursor = coll.find(query).sort("created_at", -1).limit(3)
    results = await cursor.to_list(length=3)
    if not results:
        # Check via guest_id in guests collection
        g_coll = guests_collection()
        guest = await g_coll.find_one({"email": {"$regex": f"^{re.escape(email_clean)}$", "$options": "i"}})
        if guest and guest.get("guest_id"):
            cursor = coll.find({
                "hotel_id": {"$in": h_variants},
                "guest_id": guest["guest_id"],
            }).sort("created_at", -1).limit(3)
            results = await cursor.to_list(length=3)

    return results


async def retrieve_email_context(
    hotel_id: int | str,
    from_email: str,
    subject: str,
    body: str,
    explicit_booking_id: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Orchestrate all context and retrieval tasks for an incoming email:
    1. Hotel Profile, Timings & Policies.
    2. ChromaDB RAG for FAQs, amenities, rules, local attractions.
    3. Verified guest booking lookup by booking ID or email.
    4. Date & live inventory pricing lookup if dates are mentioned.
    """
    sources_used: list[str] = []
    combined_query = f"{subject} {body}".strip()

    # 1. Hotel Profile & Policies
    hotel_details = await get_hotel_details(hotel_id) or {}
    contact_info = await get_hotel_contact_info(hotel_id) or {}
    hotel_name = hotel_details.get("name") or "our hotel"
    hotel_address = hotel_details.get("address") or ""
    contact_phone = contact_info.get("phone") or ""
    contact_email = contact_info.get("email") or ""
    policies_text = await get_formatted_hotel_policies(hotel_id) or ""
    check_in_time, check_out_time = await get_standard_stay_times(hotel_id)

    hotel_context_parts = [
        f"Hotel Name: {hotel_name}",
        f"Address: {hotel_address}" if hotel_address else "",
        f"Standard Check-in Time: {check_in_time}",
        f"Standard Check-out Time: {check_out_time}",
        f"Front Desk Phone: {contact_phone}" if contact_phone else "",
        f"Front Desk Email: {contact_email}" if contact_email else "",
    ]
    if policies_text:
        hotel_context_parts.append(f"\nHotel Policies:\n{policies_text}")
    hotel_context = "\n".join(p for p in hotel_context_parts if p)
    sources_used.append("Hotel Profile & Policies")

    # 2. ChromaDB Knowledge Base (RAG)
    rag_text = ""
    try:
        rag_res = query_knowledge(query=combined_query, hotel_id=hotel_id, n_results=4)
        if isinstance(rag_res, dict):
            rag_text = rag_res.get("text", "")
        else:
            rag_text = str(rag_res or "")
        if rag_text.strip():
            sources_used.append("Hotel Knowledge Base (ChromaDB FAQs & Amenities)")
    except Exception as e:
        logger.warning(f"RAG retrieval skipped or failed: {e}")

    # 3. Reservation / Booking Retrieval
    booking_context = "No existing booking found for this guest."
    matched_booking_ref = None

    target_bid = explicit_booking_id or _extract_booking_id_from_text(combined_query)
    found_booking = None
    if target_bid:
        found_booking = await booking_repository.get_booking(target_bid)
        if found_booking:
            matched_booking_ref = target_bid

    if not found_booking and from_email:
        email_bookings = await _find_guest_bookings_by_email(hotel_id, from_email)
        if email_bookings:
            found_booking = email_bookings[0]
            matched_booking_ref = found_booking.get("booking_id")

    if found_booking:
        booking_context = _format_booking_details(found_booking)
        sources_used.append(f"Verified Reservation Record ({matched_booking_ref})")

    # 4. Date & Live Inventory Retrieval
    inventory_context = "No specific stay dates requested in email."
    today = date.today()
    stay_dates = extract_stay_dates(combined_query, today=today)
    if stay_dates.found and stay_dates.check_in:
        cin_str = stay_dates.check_in.strftime("%Y-%m-%d")
        cout_str = (
            stay_dates.check_out.strftime("%Y-%m-%d")
            if stay_dates.check_out
            else (stay_dates.check_in + timedelta(days=1)).strftime("%Y-%m-%d")
        )
        try:
            available_rooms = await get_available_rooms(hotel_id, cin_str, cout_str)
            if available_rooms:
                room_lines = []
                for room in available_rooms:
                    rates = getattr(room, "rate_plans", []) or []
                    base_rate = rates[0].base_price if rates else getattr(room, "base_price", None)
                    price_str = f"₹{base_rate:,.2f}/night (incl. taxes)" if base_rate is not None else "Rates on request"
                    max_guests = getattr(room.occupancy, "max_adults", 2)
                    room_lines.append(f"• {room.name}: {price_str} (accommodates up to {max_guests} guests)")
                inventory_context = (
                    f"Live Availability for {cin_str} to {cout_str}:\n"
                    + "\n".join(room_lines)
                    + "\n_Note: Displayed rates are inclusive of all taxes._"
                )
                sources_used.append(f"Live Inventory & Rates ({cin_str} to {cout_str})")
            else:
                inventory_context = f"All rooms are currently sold out for {cin_str} to {cout_str}."
                sources_used.append(f"Live Inventory Check ({cin_str} to {cout_str} - Sold Out)")
        except Exception as e:
            logger.warning(f"Inventory retrieval failed: {e}")

    context_bundle = {
        "hotel_name": hotel_name,
        "hotel_context": hotel_context,
        "contact_phone": contact_phone,
        "contact_email": contact_email,
        "rag_context": rag_text or "No specific FAQ chunks matched.",
        "booking_context": booking_context,
        "matched_booking_ref": matched_booking_ref,
        "inventory_context": inventory_context,
    }
    return context_bundle, sources_used


def _classify_email_intent(subject: str, body: str, has_booking: bool, has_dates: bool) -> str:
    """Classify the primary intent of the guest's email."""
    text = f"{subject} {body}".lower()
    if any(k in text for k in ("cancel", "cancellation", "refund", "modify", "change date", "postpone")):
        return "CANCELLATION_OR_MODIFICATION"
    if has_booking or any(k in text for k in ("reservation", "booking id", "check-in late", "late arrival", "confirm booking", "receipt", "paid")):
        return "EXISTING_RESERVATION_QUERY"
    if has_dates or any(k in text for k in ("availability", "available room", "price", "rates", "how much", "cost", "quote", "book a room")):
        return "RESERVATION_INQUIRY"
    if any(k in text for k in ("complaint", "issue", "dirty", "bad", "problem", "disappointed", "manager")):
        return "COMPLAINT_OR_FEEDBACK"
    return "HOTEL_FAQ_INQUIRY"


def _format_thread_history(history: list[EmailMessage]) -> str:
    """Format prior email turns in the thread."""
    if not history:
        return "No previous emails in this thread."
    lines = []
    for m in history:
        sender_label = "Guest" if m.sender.lower() == "guest" else "Hotel Reservations"
        ts_label = f" ({m.timestamp})" if m.timestamp else ""
        lines.append(f"[{sender_label}{ts_label}]:\n{m.body.strip()}\n")
    return "\n".join(lines)


async def generate_email_reply(request: EmailReplyRequest) -> EmailReplyResponse:
    """
    Main entry point: ingests guest email, retrieves all required context,
    and returns a structured, professional email response.
    """
    from_email = (request.from_email or "").strip()
    subject = (request.subject or "").strip()
    body = (request.body or "").strip()
    guest_name = request.guest_name

    # Retrieve all required context
    context, sources_used = await retrieve_email_context(
        hotel_id=request.hotel_id,
        from_email=from_email,
        subject=subject,
        body=body,
        explicit_booking_id=request.booking_id,
    )

    hotel_name = context["hotel_name"]
    has_booking = bool(context["matched_booking_ref"])
    has_dates = "Live Availability for" in context["inventory_context"]
    detected_intent = _classify_email_intent(subject, body, has_booking, has_dates)

    # Format reply subject
    if subject:
        clean_subj = subject
        if not clean_subj.lower().startswith("re:"):
            reply_subject = f"Re: {clean_subj}"
        else:
            reply_subject = clean_subj
    else:
        reply_subject = f"Regarding your inquiry - {hotel_name}"

    # Build thread history
    thread_context = _format_thread_history(request.thread_history)

    # Resolve salutation name
    salutation_name = guest_name.strip() if guest_name and guest_name.strip() else None

    # Contact line for signature
    contact_parts = []
    if context.get("contact_phone"):
        contact_parts.append(f"Tel: {context['contact_phone']}")
    if context.get("contact_email"):
        contact_parts.append(f"Email: {context['contact_email']}")
    contact_line = " | ".join(contact_parts)

    prompt = f"""You are the Reservations & Guest Relations Manager at {hotel_name}.
Your job is to write a warm, polished, comprehensive, and professional email response to a guest.

=== HOTEL PROFILE & POLICIES ===
{context['hotel_context']}

=== KNOWLEDGE BASE INFORMATION (FAQs & AMENITIES) ===
{context['rag_context']}

=== VERIFIED GUEST RESERVATION DETAILS ===
{context['booking_context']}

=== LIVE ROOM AVAILABILITY & RATES ===
{context['inventory_context']}

=== EMAIL THREAD HISTORY ===
{thread_context}

=== CURRENT INCOMING GUEST EMAIL ===
From: {from_email}
Guest Name: {salutation_name or 'Unknown'}
Subject: {subject}
Message Body:
{body}

=== INSTRUCTIONS & EMAIL WRITING GUIDELINES ===
1. SALUTATION:
   - Begin with "Dear {salutation_name}," if guest name is known, otherwise "Dear Guest,".
2. TONE & STRUCTURE:
   - Warm, welcoming, respectful, and elegant hospitality standards.
   - Address every question, request, or concern raised in the guest's email completely.
   - Use structured, easy-to-read paragraphs. Use neat bullet points if listing amenities, policies, or room options.
3. STRICT TRUTH ON RESERVATIONS & RATES:
   - Strictly use the verified reservation facts and hotel policies provided above. NEVER invent or hallucinate booking details, dates, or prices.
   - If quoting room rates, clearly state that prices include all applicable taxes (GST).
   - If the guest has a verified booking, reference their booking ID and confirm their dates/details with confidence.
4. NO CHATBOT OR WHATSAPP ARTIFACTS:
   - This is a formal/semi-formal EMAIL.
   - DO NOT include WhatsApp instructions like "Reply 1-2 to select", "Click Option 1 below", or WhatsApp button references.
5. SIGN-OFF:
   - Conclude with a warm closing (e.g. "We look forward to welcoming you to {hotel_name}.").
   - End with:
     Warm regards,
     Reservations & Guest Support Team
     {hotel_name}
     {contact_line}
6. LANGUAGE:
   - Reply in professional English if the email is in English. If the guest wrote in Hindi or Hinglish, reply appropriately.

Write the complete email body:"""

    model = _get_model()
    try:
        response = await _generate_content_async(
            model,
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=4096,
            ),
        )
        reply_body = response.text.strip()
    except Exception as e:
        logger.error(f"Email generation failed via LLM: {e}")
        # Graceful fallback response
        salutation = f"Dear {salutation_name}," if salutation_name else "Dear Guest,"
        reply_body = (
            f"{salutation}\n\n"
            f"Thank you for contacting {hotel_name}.\n\n"
            f"We have received your message regarding \"{subject or 'your stay'}\". "
            f"Our reservations team is currently reviewing your request, and we will follow up with you promptly with complete details.\n\n"
            f"If your request is urgent, please feel free to reach our front desk directly at {context.get('contact_phone') or 'our contact number'}.\n\n"
            f"Warm regards,\n"
            f"Reservations & Guest Support Team\n"
            f"{hotel_name}\n"
            f"{contact_line}"
        )

    return EmailReplyResponse(
        success=True,
        subject=reply_subject,
        reply_body=reply_body,
        guest_name=salutation_name,
        detected_intent=detected_intent,
        booking_reference=context.get("matched_booking_ref"),
        context_sources_used=sources_used,
    )
