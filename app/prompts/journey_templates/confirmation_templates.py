def booking_confirmed(booking_id: str, check_in: str, check_out: str, **kwargs) -> list[dict]:
    guest_name = kwargs.get("guest_name") or kwargs.get("customer_name")
    hotel_name = kwargs.get("hotel_name")
    rooms_text = kwargs.get("rooms_text") or kwargs.get("room_summary")
    payment_text = kwargs.get("payment_text")
    timings_text = kwargs.get("timings_text")
    address = kwargs.get("address")
    phone = kwargs.get("phone")

    msg = f"🎉 *Booking Confirmed!*\n\n"
    if guest_name:
        msg += f"Dear *{guest_name}*,\n"
    if hotel_name:
        msg += f"Thank you for choosing *{hotel_name}*! Your reservation has been successfully confirmed.\n\n"
    
    msg += (
        f"📋 *Reservation Details:*\n"
        f"• *Booking ID:* `{booking_id}`\n"
        f"• *Dates:* {check_in} → {check_out}\n"
    )
    if rooms_text:
        msg += f"• *Rooms:* {rooms_text}\n"
    if payment_text:
        msg += f"\n💳 *Payment Summary:*\n{payment_text}\n"
    if timings_text:
        msg += f"\n⏰ *Timings:*\n{timings_text}\n"
    if address:
        msg += f"\n📍 *Location:* {address}\n"
    if phone:
        msg += f"📞 *Front Desk / Support:* {phone}\n"

    msg += "\nWe look forward to welcoming you! If you need anything else, feel free to ask. 🌟"
    return [{"type": "text", "text": msg}]


def cfm_addon(**kwargs) -> list[dict]:
    return [{"type": "text", "text": "Would you like to make your stay even better? 🌟\n\n"
                                     "We offer:\n"
                                     "• Airport Transfer\n"
                                     "• Breakfast Buffet\n"
                                     "• Local Sightseeing Tour\n\n"
                                     "Just let me know if you'd like to add any of these to your booking!"}]

