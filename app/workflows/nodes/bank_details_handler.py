"""
Bank Details Handler — formats and sends hotel bank account details on explicit guest request.
"""
from typing import cast

from app.prompts.templates import text_card
from app.repositories.bank_repository import get_hotel_bank_details
from app.workflows.state import ConversationState


async def handle_bank_details(state: ConversationState) -> ConversationState:
    """
    Deliver the hotel's verified bank account details (Account Name, Bank Name,
    Account Number, IFSC, Branch) when explicitly requested by a guest.
    """
    hotel_id = state.get("hotel_id") or 0
    bank_info = await get_hotel_bank_details(hotel_id)

    booking_id = state.get("booking_id")
    total_price = float(state.get("total_price") or 0.0)
    amount_due = float(state.get("amount_due") or total_price)

    booking_lines = []
    if booking_id and str(booking_id).lower() != "none":
        booking_lines.append(f"📌 *Booking Reference:* `{booking_id}`")
    if amount_due > 0:
        booking_lines.append(f"💰 *Payable Amount:* ₹{amount_due:,.0f}")
    elif total_price > 0:
        booking_lines.append(f"💰 *Total Amount:* ₹{total_price:,.0f}")

    booking_clause = ("\n" + "\n".join(booking_lines) + "\n") if booking_lines else ""

    if bank_info["is_configured"]:
        card_text = (
            f"🏦 *Hotel Bank Account Details for Transfer*\n"
            f"{booking_clause}\n"
            f"• *Account Name:* {bank_info['account_name']}\n"
            f"• *Bank Name:* {bank_info['bank_name']}\n"
            f"• *Account Number:* `{bank_info['account_number']}`\n"
            f"• *IFSC Code:* `{bank_info['ifsc_code']}`\n"
        )
        if bank_info.get("branch"):
            card_text += f"• *Branch:* {bank_info['branch']}\n"
        if bank_info.get("swift_code"):
            card_text += f"• *SWIFT Code:* `{bank_info['swift_code']}`\n"

        card_text += (
            f"\n📲 *Next Step:* Once you complete the NEFT/RTGS/IMPS transfer, please share a screenshot of the payment receipt or UTR reference number here. "
            f"We will verify it and confirm your reservation immediately!\n\n"
            f"_(Note: You can also pay instantly via UPI or Payment Link if you prefer immediate confirmation.)_"
        )
    else:
        # Bank details not configured in DB for this property
        phone_contact = bank_info.get("contact_phone") or "our front desk"
        hotel_name = bank_info.get("hotel_name") or "the hotel"
        card_text = (
            f"🏦 *Bank Transfer Information*\n"
            f"{booking_clause}\n"
            f"Direct NEFT/RTGS transfer details for {hotel_name} are handled directly by our front desk / accounts team.\n\n"
            f"Please contact our front desk at **{phone_contact}** for bank account transfer assistance.\n\n"
            f"_(Alternatively, you can pay securely using our online Payment Link or UPI QR code for instant confirmation!)_"
        )

    return cast(ConversationState, {
        **state,
        "bot_response": text_card(card_text),
        "next_action": "bank_details_sent",
    })
