"""Payment-flow constants and small helpers."""

import re

MIN_ADVANCE_PERCENT = 0.30  # 30% minimum advance

_PAYMENT_DONE_PHRASES = (
    "paid", "payment done", "payment completed", "made the payment",
    "money sent", "sent payment", "transferred", "receipt",
)
_GENERIC_DONE_WORDS = ("done", "completed", "success", "sent")

_FOLLOWUP_QUESTIONS = {
    "customer_name": "Could you please share the **guest name** for this booking? 🙂",
    "customer_phone": "What is the **phone number** we can reach you at? 📱",
    # Nothing on our side can send email — the booking flow has no outbound
    # mail path. The old wording ("for the booking confirmation") promised one
    # anyway, so a guest who gave an address waited for a mail that never came.
    # The address is still worth having on the record; the confirmation itself
    # is sent on WhatsApp.
    "customer_email": "Could you share your **email address** so we can keep it on your booking? 📧",
    "num_adults": "How many **adults** will be staying? 👥",
    "num_children": "Will there be any **children** staying? If yes, how many? 👶",
    "check_in": "What is your **check-in date**? 📅",
    "check_out": "And your **check-out date**? 📅",
    "num_rooms": "How many **rooms** do you need? 🏨",
    "room_type": "Do you have a **room type** preference (e.g., Deluxe, Standard, Suite)? 🛏️",
    "meal_plan": "Would you like to add a **meal plan** (e.g., breakfast included, full board)? 🍽️",
}

SUCCESS_STATUSES = ("SUCCESS", "COMPLETED", "DONE", "PAID", "SUCCESSFUL")


def get_followup_question(field: str) -> str:
    """Get a user-friendly question for a missing payment detail field."""
    return _FOLLOWUP_QUESTIONS.get(
        field,
        f"Could you please provide your **{field.replace('_', ' ')}**?"
    )


def user_claims_paid(user_msg: str, intent: str) -> bool:
    """Detect whether the user is claiming payment is already done."""
    msg = (user_msg or "").lower()
    if any(p in msg for p in _PAYMENT_DONE_PHRASES):
        return True
    if intent in ("CONFIRM_BOOKING", "PAYMENT_STATUS"):
        return any(re.search(rf"\b{word}\b", msg) for word in _GENERIC_DONE_WORDS)
    return False
