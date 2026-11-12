def lq_quote(rooms_text: str, **kwargs) -> list[dict]:
    return [{"type": "text", "text": "Here are our available rooms and standard pricing:\n\n"
                                     f"{rooms_text}\n\n"
                                     "To check live availability for these rooms, just let me know your **check-in and check-out dates**! 📅\n\n"
                                     "Would you like to start a booking?"}]

def lq_urgency(rooms_left: int = 2, **kwargs) -> list[dict]:
    return [{"type": "text", "text": f"⚠️ Just a quick heads up - we only have {rooms_left} rooms left for these dates. They are booking up fast!"}]

def lq_objection(direct_rate: float, ota_rate: float, **kwargs) -> list[dict]:
    return [{"type": "text", "text": f"I understand you found a price of ₹{ota_rate:,.0f} online. Our direct rate is ₹{direct_rate:,.0f}.\n"
                                     "However, booking directly with us gives you free early check-in (subject to availability) and complimentary breakfast! 🥐\n"
                                     "Please share a screenshot of the lower price so our manager can see if we can match it."}]
