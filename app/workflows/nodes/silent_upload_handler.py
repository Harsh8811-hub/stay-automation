import logging
import re
from typing import cast
from app.workflows.state import ConversationState
from app.prompts.templates import text_card

logger = logging.getLogger(__name__)

async def handle_silent_upload(state: ConversationState) -> ConversationState:
    """
    Handle images/documents uploaded without any accompanying text.
    Do not assume intent. Present a summary and ask how to proceed.
    """
    user_msg = state.get("user_message", "")
    price_match_data = state.get("price_match_data")
    payment_details = state.get("payment_details")
    
    # Extract image type and description from user message text
    image_type = "Document"
    type_match = re.search(r"Type:\s*(.*?)\n", user_msg)
    if type_match:
        image_type = type_match.group(1).strip()
        
    description = ""
    desc_match = re.search(r"Details:\s*(.*?)\n\[/IMAGE_ANALYSIS\]", user_msg, re.DOTALL)
    if desc_match and not price_match_data and not payment_details:
        description = desc_match.group(1).strip()

    updated_state = {**state, "price_match_data": None, "payment_details": None}

    if image_type == "Invalid Price Match" and price_match_data and price_match_data.get("invalid"):
        bot_response = text_card(
            "The uploaded file doesn't appear to be related to a hotel booking or payment.\n\n"
            "If you intended to share a hotel booking, reservation, or payment confirmation, please upload a relevant image or PDF."
        )
        return cast(ConversationState, {**updated_state, "next_action": "need_info", "bot_response": bot_response})

    if image_type == "Competitor Price Match" and price_match_data:
        # Check missing fields
        missing_fields = price_match_data.get("missing_fields") or []
        
        hotel_name = price_match_data.get("hotel_name", "Unknown Hotel")
        room_type = price_match_data.get("room_type", "Unknown Room")
        check_in = price_match_data.get("check_in", "Unknown")
        check_out = price_match_data.get("check_out", "Unknown")
        adults = price_match_data.get("adults", "Unknown")
        price = price_match_data.get("price")
        price_str = f"₹{price:,.0f}" if isinstance(price, (int, float)) else "Unknown"

        if missing_fields:
            missing_str = " and ".join(f.replace("_", " ").title() for f in missing_fields)
            summary = (
                f"I found some booking information, but I couldn't identify the {missing_str} clearly.\n\n"
                "If you'd like me to help with this booking, please upload a clearer screenshot."
            )
            return cast(ConversationState, {**updated_state, "next_action": "need_info", "bot_response": text_card(summary)})

        from app.utils.helpers import format_display_date
        ci_disp = format_display_date(check_in) or check_in
        co_disp = format_display_date(check_out) or check_out

        summary = (
            "I found the following booking details:\n"
            f"• Hotel: {hotel_name}\n"
            f"• Room: {room_type}\n"
            f"• Check-in: {ci_disp}\n"
            f"• Check-out: {co_disp}\n"
            f"• Guests: {adults} Adults\n"
            f"• Price: {price_str}\n\n"
            "How can I assist you with this booking?\n"
            "For example, I can:\n"
            "• Check availability at our hotel\n"
            "• Generate our quotation\n"
            "• Compare prices under our Price Match policy\n"
            "• Answer questions about this booking"
        )
        return cast(ConversationState, {**updated_state, "next_action": "need_info", "bot_response": text_card(summary)})

    if image_type == "Payment Screenshot" and payment_details:
        amount = payment_details.get("amount")
        amount_str = f"₹{amount:,.0f}" if isinstance(amount, (int, float)) else "Unknown"
        txn_id = payment_details.get("transaction_id", "Unknown")
        status = payment_details.get("status", "Unknown")
        date = payment_details.get("timestamp", "Unknown")
        
        summary = (
            "I found the following payment details:\n"
            f"• Amount: {amount_str}\n"
            f"• Transaction ID: {txn_id}\n"
            f"• Status: {status}\n"
            f"• Date: {date}\n\n"
            "How can I assist you with this payment receipt?\n"
            "For example, I can check its status against your booking or answer any questions."
        )
        return cast(ConversationState, {**updated_state, "next_action": "need_info", "bot_response": text_card(summary)})

    from app.utils.privacy_scrubber import scrub_id_numbers, contains_id_number

    # Check if this is an ID Card / Identity Document
    is_id_card = (
        image_type.lower() in (
            "id_card", "id card", "identity document", "government id",
            "aadhaar", "aadhar", "passport", "driving license", "driving_license",
            "voter id", "voter_id", "pan card", "pan_card", "national id", "national_id"
        )
        or bool(re.search(
            r'\b(aadhaar|aadhar|passport|driving\s*licen[sc]e|voter\s*id|pan\s*card|identity\s*card|unique\s*identification\s*authority|govt\.?\s*of\s*india|government\s*of\s*india)\b',
            user_msg,
            re.IGNORECASE
        ))
        or contains_id_number(user_msg)
    )

    if is_id_card:
        # 1. Determine ID label & type
        id_lower = user_msg.lower()
        if "aadhaar" in id_lower or "aadhar" in id_lower or re.search(r'\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b', user_msg):
            doc_label = "Aadhaar Card"
            id_type = "aadhaar"
        elif "passport" in id_lower:
            doc_label = "Passport"
            id_type = "passport"
        elif "driving" in id_lower:
            doc_label = "Driving License"
            id_type = "driving_license"
        elif "voter" in id_lower:
            doc_label = "Voter ID"
            id_type = "voter_id"
        elif "pan" in id_lower:
            doc_label = "PAN Card"
            id_type = "pan_card"
        else:
            doc_label = "Government ID"
            id_type = "government_id"

        # 2. Extract cardholder name
        name = state.get("guest_name")
        if not name or name.strip().lower() in ("guest", "unknown"):
            name_m = re.search(r'(?:Name|Cardholder|Holder|To)[\s,:\-]+([A-Z][a-zA-Z\s]{2,25})', user_msg, re.IGNORECASE)
            if name_m:
                candidate = name_m.group(1).strip()
                candidate = re.split(r'[\r\n]|DOB|Date|Gender|Male|Female|Address', candidate, flags=re.IGNORECASE)[0].strip()
                if candidate and len(candidate) >= 3:
                    name = candidate
            else:
                name_m2 = re.search(r'([A-Z][a-zA-Z\s]{2,25})\s*\n\s*(?:DOB|Date of Birth|Year of Birth|Gender)', user_msg, re.IGNORECASE)
                if name_m2:
                    candidate = name_m2.group(1).strip()
                    candidate = re.split(r'[\r\n]|DOB|Date|Gender|Male|Female|Address', candidate, flags=re.IGNORECASE)[0].strip()
                    if candidate and len(candidate) >= 3:
                        name = candidate
            if not name or name.strip().lower() in ("guest", "unknown"):
                name = "Guest"

        # 3. Mask ID number for DPDP compliance (preserve only last 4 digits)
        masked_num = ""
        aadh_m = re.search(r'\b[2-9]\d{3}[-\s]?\d{4}[-\s]?(\d{4})\b', user_msg)
        if aadh_m:
            masked_num = f"XXXX XXXX {aadh_m.group(1)}"
        else:
            pass_m = re.search(r'\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}([0-9]{1,4})\b', user_msg)
            if pass_m:
                masked_num = f"XXXX{pass_m.group(1)}"
            else:
                generic_m = re.search(r'(?:XXXX\s*XXXX\s*\d{4}|XXXX\d{4}|[X\*\-]{4,}\s*(\d{4}))', user_msg)
                if generic_m:
                    masked_num = generic_m.group(0).strip()

        # 4. Extract all ID cards (supporting multi-card uploads & sequential uploads)
        new_cards: list[dict] = []
        aadh_matches = list(re.finditer(r'\b([2-9]\d{3}[-\s]?\d{4}[-\s]?(\d{4}))\b', user_msg))
        masked_aadh = list(re.finditer(r'(?:XXXX\s*XXXX\s*(\d{4}))', user_msg))
        pass_matches = list(re.finditer(r'\b([A-PR-WYa-pr-wy][1-9]\d\s?\d{4}([0-9]{1,4}))\b', user_msg))

        def _find_name_near(start_pos: int, end_pos: int) -> str:
            # 1. Search in the text preceding this ID number (closest to it)
            win_start = max(0, start_pos - 350)
            window_before = user_msg[win_start:start_pos]
            matches_before = list(re.finditer(r'(?:Name|Cardholder|Holder|To)[\s,:\-]+([A-Z][a-zA-Z\s]{2,25})', window_before, re.IGNORECASE))
            if matches_before:
                c = matches_before[-1].group(1).strip()
                c = re.split(r'[\r\n]|DOB|Date|Gender|Male|Female|Address', c, flags=re.IGNORECASE)[0].strip()
                if c and len(c) >= 3:
                    return c

            m_before2 = list(re.finditer(r'([A-Z][a-zA-Z\s]{2,25})\s*\n\s*(?:DOB|Date of Birth|Year of Birth|Gender)', window_before, re.IGNORECASE))
            if m_before2:
                c = m_before2[-1].group(1).strip()
                c = re.split(r'[\r\n]|DOB|Date|Gender|Male|Female|Address', c, flags=re.IGNORECASE)[0].strip()
                if c and len(c) >= 3:
                    return c

            # 2. Search in the text following this ID number
            win_end = min(len(user_msg), end_pos + 350)
            window_after = user_msg[end_pos:win_end]
            m_after = re.search(r'(?:Name|Cardholder|Holder|To)[\s,:\-]+([A-Z][a-zA-Z\s]{2,25})', window_after, re.IGNORECASE)
            if m_after:
                c = m_after.group(1).strip()
                c = re.split(r'[\r\n]|DOB|Date|Gender|Male|Female|Address', c, flags=re.IGNORECASE)[0].strip()
                if c and len(c) >= 3:
                    return c

            return ""

        if aadh_matches:
            for m in aadh_matches:
                last4 = m.group(2)
                c_masked = f"XXXX XXXX {last4}"
                c_name = _find_name_near(m.start(), m.end()) or (name if len(aadh_matches) == 1 else "Guest")
                new_cards.append({
                    "guest_name": c_name,
                    "id_type": "aadhaar",
                    "doc_label": "Aadhaar Card",
                    "id_masked_number": c_masked,
                })
        elif masked_aadh:
            for m in masked_aadh:
                last4 = m.group(1)
                c_masked = f"XXXX XXXX {last4}"
                c_name = _find_name_near(m.start(), m.end()) or (name if len(masked_aadh) == 1 else "Guest")
                new_cards.append({
                    "guest_name": c_name,
                    "id_type": "aadhaar",
                    "doc_label": "Aadhaar Card",
                    "id_masked_number": c_masked,
                })
        elif pass_matches:
            for m in pass_matches:
                last4 = m.group(2)
                c_masked = f"XXXX{last4}"
                c_name = _find_name_near(m.start(), m.end()) or (name if len(pass_matches) == 1 else "Guest")
                new_cards.append({
                    "guest_name": c_name,
                    "id_type": "passport",
                    "doc_label": "Passport",
                    "id_masked_number": c_masked,
                })
        else:
            new_cards.append({
                "guest_name": name,
                "id_type": id_type,
                "doc_label": doc_label,
                "id_masked_number": masked_num,
            })

        # Merge into verified_ids list from session state
        existing_ids = list(updated_state.get("verified_ids") or [])
        for card in new_cards:
            is_dup = any(
                (card["id_masked_number"] and e.get("id_masked_number") == card["id_masked_number"])
                or (card["guest_name"] != "Guest" and e.get("guest_name", "").strip().lower() == card["guest_name"].strip().lower() and card["id_masked_number"] == e.get("id_masked_number"))
                for e in existing_ids
            )
            if not is_dup:
                existing_ids.append(card)

        total_count = len(existing_ids)
        cards_bullets = "\n".join([
            f"• **Guest {i+1}:** {c['guest_name']} — {c['doc_label']}" + (f" ({c['id_masked_number']})" if c.get('id_masked_number') else "")
            for i, c in enumerate(existing_ids)
        ])

        # 5. Grounded response based on session stage
        stage = updated_state.get("booking_stage") or "new"
        has_rooms = bool(updated_state.get("selected_rooms"))
        booking_id = updated_state.get("booking_id")
        hotel_name = updated_state.get("hotel_name") or (updated_state.get("hotel_config") or {}).get("hotel_name") or "Sample Hotel"

        id_updates: dict = {
            "id_verified": True,
            "id_type": existing_ids[-1]["id_type"] if existing_ids else id_type,
            "id_masked_number": existing_ids[-1]["id_masked_number"] if existing_ids else masked_num,
            "verified_ids": existing_ids,
        }
        if existing_ids and existing_ids[0]["guest_name"] != "Guest":
            id_updates["guest_name"] = existing_ids[0]["guest_name"]
        elif name and name != "Guest":
            id_updates["guest_name"] = name

        doc_with_num = f"{doc_label} ({masked_num})" if masked_num else doc_label

        if stage in ("quote_shown", "rooms_shown") or (stage == "collecting" and has_rooms):
            id_updates["booking_stage"] = "quote_shown"
            id_updates["next_action"] = "confirm_with_name"
            name_part = f" for {name}" if name and name != "Guest" else ""
            doc_part = f" ({doc_with_num})" if doc_with_num else ""
            reply = f"Your ID verification has been completed successfully{name_part}{doc_part}. Reply *confirm* to proceed with your booking! 🏨"
            return cast(ConversationState, {
                **updated_state,
                **id_updates,
                "bot_response": text_card(reply),
            })

        elif stage == "awaiting_payment":
            reply = "Your ID verification has been completed successfully. Please complete the advance payment to finalize your booking! 💳"
            return cast(ConversationState, {
                **updated_state,
                **id_updates,
                "bot_response": text_card(reply),
            })

        elif stage in ("confirmed", "paid"):
            reply = "Your ID verification has been completed successfully. We look forward to welcoming you! 🏨"
            return cast(ConversationState, {
                **updated_state,
                **id_updates,
                "bot_response": text_card(reply),
            })

        else:
            import datetime
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            id_updates["id_received_date"] = now_str
            id_updates["next_action"] = "ask_booking_or_checkin"

            name_part = f" for {name}" if name and name != "Guest" else ""
            doc_part = f" ({doc_with_num})" if doc_with_num else ""
            reply = (
                f"Your ID verification has been completed successfully{name_part}{doc_part}. "
                f"Are you checking in with your Booking ID or looking for a new booking? 🏨"
            )
            return cast(ConversationState, {
                **updated_state,
                **id_updates,
                "bot_response": text_card(reply),
            })

    # Fallback for General / Room Image / Non-ID Document
    clean_desc = scrub_id_numbers(description) if description else ""
    if clean_desc and len(clean_desc) > 300:
        clean_desc = clean_desc[:300].rstrip() + "..."

    bot_text = f"I received the {image_type.lower()} you uploaded."
    if clean_desc:
        bot_text += f" It looks like: {clean_desc}"
    bot_text += "\n\nHow would you like to proceed? (e.g. Do you want to book a room, report an issue, or ask a question?)"
    
    return cast(ConversationState, {
        **updated_state,
        "next_action": "need_info",
        "bot_response": text_card(bot_text)
    })
