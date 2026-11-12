from typing import Any
from app.utils.template_renderers import (
    button_card,
    list_card,
    render_currency,
    render_date_single,
    render_dates_line,
    render_example_date,
    render_guests_line,
    render_label,
    render_nights_phrase,
    text_card,
)


def image_card(url: str, caption: str = "") -> list[dict]:
    return [{"type": "image", "url": url, "caption": caption}]


def document_card(url: str, filename: str = "Document", caption: str = "") -> list[dict]:
    return [{"type": "document", "url": url, "filename": filename, "caption": caption}]


from app.prompts.journey_templates import get_template


def format_assumption_line(assumptions: list[str] | None = None) -> str:
    """Mandatory assumption line on every S4 and S6 message for defaulted slots."""
    if not assumptions:
        assumptions = ["AC", "Room Only"]
    return f"_Assuming: {', '.join(assumptions)}. Reply *change* to adjust._"


def format_fee_disclaimer(kind: str) -> str:
    """The line that must accompany any early check-in / late check-out fee."""
    what = {
        "early_check_in": "Early check-in",
        "late_check_out": "Late check-out",
    }.get(kind, "This request")

    return (
        f"_{what} is subject to availability and must be confirmed by the front "
        f"desk. The charge is taken once, at booking, and is refundable if the "
        f"hotel cannot provide the time._"
    )


def greeting_response(
    hotel_name: str = "our hotel",
    resume_line: str | None = None,
    prior_guest_name: str | None = None,
    lang: str = "English",
) -> list[dict]:
    """S0 Entry Message — fires on first inbound of a new session (Catalog v2.0 §1.1)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if prior_guest_name:
        welcome_line = (
            f"🏨 Namaste {prior_guest_name}! 🙏 {hotel_name} mein aapka swagat hai.\nMain aapka 24/7 virtual assistant hoon. Aaj main aapki kaise help kar sakta hoon?"
            if is_hi
            else f"🏨 Namaste {prior_guest_name}! 🙏 Welcome back to {hotel_name}.\nI'm your 24/7 virtual assistant. How can I help you today?"
        )
    else:
        welcome_line = (
            f"🏨 Namaste! 🙏 {hotel_name} mein aapka swagat hai.\nMain aapka 24/7 virtual assistant hoon. Aaj main aapki kaise help kar sakta hoon?"
            if is_hi
            else f"🏨 Namaste! 🙏 Welcome to {hotel_name}.\nI'm your 24/7 virtual assistant. How can I help you today?"
        )

    if resume_line:
        welcome_line += f"\n\n{resume_line}"

    # Three is the channel's cap, and the set is chosen for the three things a
    # stranger actually wants: to book, to browse rates, or to reach a person.
    # "Check booking" moved out of the buttons and stays reachable by typing it
    # — the registry still resolves the phrase.
    buttons = [
        {"id": "menu_book", "title": "Room shuru karein" if is_hi else "Book a room"},
        {"id": "menu_rooms", "title": "Kamre aur rates" if is_hi else "Room types & rates"},
        {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
    ]
    footer = "Ya seedha apna sawal likhein" if is_hi else "Or type your query directly"
    return button_card(welcome_line, buttons, footer=footer)


def greeting_returning(
    hotel_name: str = "our hotel",
    guest_name: str = "",
    last_stay_nights: int | None = None,
    lang: str = "English",
) -> list[dict]:
    """S0.1 Returning Guest Greeting (Catalog v2.0 §1.2)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    name_part = f", {guest_name}" if guest_name else ""
    text = (
        f"🏨 Welcome back{name_part}! 🙏 {hotel_name} mein aapka fir se swagat hai.\nAaj main aapki kaise help kar sakta hoon?"
        if is_hi
        else f"🏨 Welcome back{name_part}! 🙏 Wonderful to host you again at {hotel_name}.\nHow can I assist you today?"
    )
    buttons = [
        {"id": "menu_book", "title": "Room shuru karein" if is_hi else "Book a room"},
        {"id": "menu_rooms", "title": "Kamre aur rates" if is_hi else "Room types & rates"},
        {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
    ]
    footer = "Ya seedha apna sawal likhein" if is_hi else "Or type your query directly"
    return button_card(text, buttons, footer=footer)


def main_menu(hotel_name: str = "our hotel", lang: str = "English") -> list[dict]:
    """M1 Main Menu (Catalog v2.0 §1.3)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        f"🏨 {hotel_name} mein main aapki in cheezon mein madad kar sakta hoon:"
        if is_hi
        else f"🏨 Here is what I can help you with at {hotel_name}:"
    )
    buttons = [
        {"id": "menu_book", "title": "Room shuru karein" if is_hi else "Book a room"},
        {"id": "menu_rooms", "title": "Kamre aur rates" if is_hi else "Room types & rates"},
        {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
    ]
    footer = "Ya seedha apna sawal likhein" if is_hi else "Or type your query directly"
    return button_card(text, buttons, footer=footer)


def change_options_menu(lang: str = "English") -> list[dict]:
    """
    M3 Change Menu — fires when guest replies 'change'.

    Four options, one past WhatsApp's three-button cap, so this is a list rather than a
    button group. The options are not written into the body: the rows are the options, and
    a second copy in prose is a second thing to keep in step — see
    `ask_meal_plan_preference` for the wording drift that produced, where the body said
    "With Breakfast (CP)" while the button said "Breakfast (CP)".

    The row titles read as plainly as the reader accepts them. "Stay Dates", not
    "📅 Stay Dates": the title is the token the router matches, so decorating it decorates
    the thing a guest has to send back.
    """
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "✏️ *Aap kya update karna chahte hain?*\n\n"
        "👉 Neeche se chuniye, ya seedha bata dijiye kya change karna hai."
        if is_hi
        else "✏️ *What would you like to update?*\n\n"
        "👉 Pick one below, or just tell me what you'd like to change."
    )
    rows = [
        {"id": "change_dates", "title": "Stay Dates"},
        {"id": "change_guests", "title": "Guests / Room count"},
        {"id": "change_ac", "title": "AC / Non-AC"},
        {"id": "change_meal", "title": "Meal Plan"},
    ]
    return list_card(text, "Change" if not is_hi else "Badlein", [
        {"title": "What to change" if not is_hi else "Kya badalna hai", "rows": rows},
    ])


def ask_change_dates(lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if is_hi:
        return text_card("📅 Nayi check-in aur check-out dates bata dijiye (jaise *25-27 Aug*).")
    return text_card("📅 Please share your new check-in and check-out dates (e.g. *25-27 Aug*).")


def ask_change_guests(lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if is_hi:
        return text_card("👥 Kitne adults, bachche (umar ke saath), aur kitne rooms chahiye bata dijiye (jaise *4 adults, 1 room*).")
    return text_card("👥 Please share your updated guest count (adults, children with ages) and how many rooms you need (e.g. *4 adults, 1 room*).")


def ask_meal_plan_preference(lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    # The three plans are NOT listed in the body. They are the three buttons below, and
    # writing them twice was how the wording drifted: the body said "With Breakfast (CP)"
    # while the button said "Breakfast (CP)". Only the button's words are what comes back
    # from a tap (`app.utils.button_registry` matches on them), so a second copy in prose
    # is a second thing to keep in step and no way to notice when it isn't.
    #
    # Nothing is lost when the buttons cannot be sent: `format_cards_for_whatsapp` renders
    # the card's own labels as a numbered list on that path, so the guest still gets the
    # same three options from the same declaration.
    text = (
        "🍽️ *Kaunsa meal plan prefer karenge?*"
        if is_hi
        else "🍽️ *Which meal plan would you prefer?*"
    )
    buttons = [
        {"id": "meal_ep", "title": "Room Only (EP)"},
        {"id": "meal_cp", "title": "Breakfast (CP)"},
        {"id": "meal_map", "title": "MAP (Bkfast+Din)"},
    ]
    return button_card(text, buttons)


def ask_name_and_language(hotel_name: str = "our hotel", lang: str = "English") -> list[dict]:
    """Legacy alias redirecting to S0 greeting."""
    return greeting_response(hotel_name=hotel_name, lang=lang)


def ask_guest_name(hotel_name: str = "our hotel", lang: str = "English") -> list[dict]:
    return greeting_response(hotel_name=hotel_name, lang=lang)


def ask_preferred_language(guest_name: str = "") -> list[dict]:
    return greeting_response()


def ask_composite_slots(lang: str = "English") -> list[dict]:
    """
    S2 Composite Ask Template (fires once, asks everything in one turn).
    Keep the example line — it materially raises single-message completion.
    """
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if is_hi:
        return text_card(
            "🏨 *Bilkul - main aapke liye availability check kar leta hoon.*\n\n"
            "Kripya niche di gayi details share karein:\n"
            "📅 *Check-in aur check-out dates*\n"
            "👥 *Kitne adults aur bachche (umar ke saath)*\n"
            "🛏️ *Kitne rooms chahiye*\n\n"
            "_Example: 22-23 Aug, 3 adults + 1 child (age 8), 1 room_"
        )
    return text_card(
        "🏨 *Perfect - let me check availability for you.*\n\n"
        "Please share the following details:\n"
        "📅 *Check-in and check-out dates*\n"
        "👥 *How many adults and children (with ages)*\n"
        "🛏️ *How many rooms you need*\n\n"
        "_Example: 22-23 Aug, 3 adults + 1 child (age 8), 1 room_"
    )


def ask_missing_slot_one_liner(
    missing_fields: list[str],
    check_in: str | None = None,
    check_out: str | None = None,
    adults: int | None = None,
    children: int | None = None,
    lang: str = "English",
) -> list[dict]:
    """
    Concise human-like one-liner prompt when individual fields are missing after
    the composite template was already shown once (Spec §Sir rule: template once, then one-liners).
    """
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    missing = set(missing_fields or [])

    # Missing check-in and check-out
    if "check_in" in missing and "check_out" in missing:
        if "adults" in missing:
            text = "Stay dates aur kitne log stay karenge bata dijiye." if is_hi else "Please share your stay dates and number of guests."
        else:
            text = "Aapki check-in aur check-out dates kya hain?" if is_hi else "What are your check-in and check-out dates?"
    # Missing only check-out
    elif "check_out" in missing and "check_in" not in missing:
        if "adults" in missing:
            text = "Check-out date aur kitne log stay karenge bata dijiye." if is_hi else "Please share your check-out date and number of guests."
        else:
            text = "Check-out date kya hogi?" if is_hi else "What is your check-out date?"
    # Missing only check-in
    elif "check_in" in missing and "check_out" not in missing:
        if "adults" in missing:
            text = "Check-in date aur kitne log stay karenge bata dijiye." if is_hi else "Please share your check-in date and number of guests."
        else:
            text = "Check-in date kya hogi?" if is_hi else "What is your check-in date?"
    # Missing only adults / guests
    elif "adults" in missing:
        text = "Kitne log stay karenge? (Adults aur bachche bata dijiye)" if is_hi else "How many guests will be staying (adults and children)?"
    else:
        text = "Dates aur kitne log stay karenge bata dijiye." if is_hi else "Please share your stay dates and number of guests."

    return text_card(text)


def ask_child_ages_followup(
    lang: str = "Hinglish",
    known_ages: list[int] | None = None,
    total_children: int = 0,
    free_limit: int = 10,
    adult_limit: int = 11,
) -> list[dict]:
    """
    Child age follow-up when children > 0 and ages needed or partially given.

    `free_limit` / `adult_limit` are the HOTEL's bands, not ours. Both sentences
    below used to hardcode 10, so a hotel whose children stay free to 7 and
    count as adults at 13 told its guests "10 and under stay free, older than 10
    is an adult" — wrong in both directions at once. The bands are read from
    `business_config.limits` in the validator and passed in by the caller; the
    defaults here match `hotel_config.py`'s so a caller that forgets them still
    says something the schema permits.
    """
    is_english = str(lang).lower() in ("english", "en", "casual english")
    known = known_ages or []
    missing_count = max(0, total_children - len(known))

    if known and missing_count > 0:
        last_age = known[-1]
        # Which of the three bands this child landed in — the same boundaries
        # the validator uses (age <= free_limit free, age >= adult_limit adult,
        # the gap between them paid). The acknowledgement named only two.
        if is_english:
            if last_age >= adult_limit:
                age_ack = f"Got it, counting the {last_age}-year-old as an adult."
            elif last_age <= free_limit:
                age_ack = f"Got it, the {last_age}-year-old stays free (no extra bed)."
            else:
                age_ack = f"Got it, the {last_age}-year-old is chargeable under our child policy."

            if missing_count == 1:
                ask_msg = f"{age_ack} What is the age of the other child?"
            else:
                ask_msg = f"{age_ack} What are the ages of the remaining {missing_count} children?"
            return text_card(ask_msg)
        else:
            if last_age >= adult_limit:
                age_ack = f"Theek hai, {last_age} saal wale ko adult count kar raha hoon."
            elif last_age <= free_limit:
                age_ack = f"Theek hai, {last_age} saal ka bachcha free rahega (bed nahi lagega)."
            else:
                age_ack = f"Theek hai, {last_age} saal ke bachche ka charge hotel policy ke hisaab se lagega."

            if missing_count == 1:
                ask_msg = f"{age_ack} Doosre bachche ki umar kya hai?"
            else:
                ask_msg = f"{age_ack} Baki {missing_count} bachchon ki umar bata dijiye."
            return text_card(ask_msg)

    if is_english:
        return text_card(
            f"Please share the children's ages — children {free_limit} and under stay free, "
            f"and children {adult_limit} and over are counted as adults.\n\n"
            "_Example: 4 years, 8 years_"
        )
    return text_card(
        f"Bachchon ki umar bata dijiye — {free_limit} saal ya usse chhota bachcha free hai, "
        f"aur {adult_limit} saal ya usse zyada bachcha adult count hota hai.\n\n"
        "_Example: 4 years, 8 years_"
    )


def ask_ac_preference(dates_str: str = "", pax_str: str = "", rooms_str: str = "", lang: str = "Hinglish") -> list[dict]:
    """S3 AC Qualifier Template — asked only when mixed inventory AND both AC/Non-AC free."""
    prefix = f"📋 {dates_str}, {pax_str}, {rooms_str}." if dates_str else ""
    is_english = str(lang).lower() in ("english", "en", "casual english")
    text = (
        f"{prefix}\n\nWould you prefer an AC or Non-AC room?".strip()
        if is_english
        else f"{prefix}\n\nAapko AC room chahiye ya Non-AC?".strip()
    )
    buttons = [
        {"id": "ac_yes", "title": "AC"},
        {"id": "ac_no", "title": "Non-AC"},
        {"id": "ac_both", "title": "Show both" if is_english else "Dono dikhaiye"},
    ]
    return button_card(text, buttons)


EXTRA_BED_QUESTION = (
    "If you want to book without extra bed say 'No', otherwise to continue with "
    "extra beds say 'Yes'"
)


def ask_extra_bed(lang: str = "English") -> list[dict]:
    """The post-selection mattress question with interactive buttons."""
    return button_card(
        EXTRA_BED_QUESTION,
        [
            {"id": "bed_yes", "title": "Yes"},
            {"id": "bed_no", "title": "No"},
        ],
    )


def ask_composition_preference(
    dates_str: str = "",
    nights_str: str = "",
    pax_str: str = "",
    lang: str = "Hinglish",
) -> list[dict]:
    """
    §5.5 Composition Turn (Fit Question for 5+ Pax).
    Fires before S4 room options when 5+ guests and >3 arrangements are viable.

    Two options, not the four this printed until 2026-10-15. "Split by family" and "You
    decide" were taken off the menu on 2026-09-19 and the reader stopped answering them the
    same day — with a comment saying so — but the body went on printing all four numbers.
    A guest reading it and tapping "3" was answering a question the bot had already
    stopped asking, and fell through to the model.

    So the body now describes only what the two rows offer, and the split question is still
    reachable by asking for it in words, which is what the reader's free-text arm has always
    accepted (`understand.py`, `need_composition_preference`).
    """
    header_parts = [p for p in [dates_str, nights_str, pax_str] if p]
    header = f"*{' · '.join(header_parts)}*\n\n" if header_parts else ""

    is_english = str(lang).lower() in ("english", "en", "casual english")

    if is_english:
        body = (
            f"{header}"
            "A few ways to arrange this. What suits you?\n\n"
            "Or just tell me — \"3 rooms\" or \"budget 6000\"."
        )
        rows = [
            {"id": "compose_max", "title": "Best price"},
            {"id": "compose_base", "title": "Most comfortable"},
        ]
    else:
        body = (
            f"{header}"
            "Itne logon ke liye do tarah se ho sakta hai.\n"
            "Aapko kaisa chahiye?\n\n"
            "Ya seedha likh dijiye — jaise \"3 room chahiye\" "
            "ya \"budget 6000 tak\"."
        )
        rows = [
            {"id": "compose_max", "title": "Kam kharcha"},
            {"id": "compose_base", "title": "Aaram se"},
        ]
    return list_card(body, "Choose" if is_english else "Chuniye", [
        {"title": "How to arrange" if is_english else "Kaise arrange karein", "rows": rows},
    ])


def ask_family_subgroups_followup(lang: str = "Hinglish") -> list[dict]:
    """
    Subgroup inquiry when guest selects Option 3 (Family alag / Split by family).
    """
    is_english = str(lang).lower() in ("english", "en", "casual english")
    if is_english:
        body = (
            "Got it. How would you like to split the groups?\n\n"
            "_Example: \"parents separate, we 4 together\"\n"
            "or \"2 couples + 3 adults\"_"
        )
    else:
        body = (
            "Theek hai. Kitne alag-alag group hain?\n\n"
            "_Jaise: \"mummy-papa alag, hum 4 saath\"\n"
            "ya \"2 couple + 3 bade\"_"
        )
    return text_card(body)


def ask_split_preference(
    dates_str: str = "",
    nights_str: str = "",
    pax_str: str = "",
    rooms_a: int = 3,
    base_a: int = 6,
    extra_a: int = 1,
    price_a: float = 0.0,
    rooms_b: int = 4,
    price_b: float = 0.0,
    lang: str = "Hinglish",
) -> list[dict]:
    """
    §5.2 SPLIT Turn (S3_FIT - When requested rooms require extra beds vs dedicated rooms).
    """
    header_parts = [p for p in [dates_str, nights_str, pax_str] if p]
    header = f"*{' | '.join(header_parts)}*\n\n" if header_parts else ""
    is_english = str(lang).lower() in ("english", "en", "casual english")

    extra_txt_en = f"{extra_a} extra bed" if extra_a == 1 else f"{extra_a} extra beds"
    extra_txt_hi = f"{extra_a} extra bed" if extra_a == 1 else f"{extra_a} extra beds"

    if is_english:
        body = (
            f"{header}"
            "Quick check -\n\n"
            f"{rooms_a} rooms fits {base_a} on base beds + {extra_txt_en}.\n"
            "A bit snug.\n"
            f"{rooms_b} rooms gives everyone their own proper bed.\n\n"
            f"*A - {rooms_a} rooms* ({base_a} + {extra_txt_en})  Rs {price_a:,.0f}/night\n"
            f"*B - {rooms_b} rooms* (dedicated beds)  Rs {price_b:,.0f}/night\n\n"
            "A or B?"
        )
    else:
        body = (
            f"{header}"
            "Ek baat clear kar lein -\n\n"
            f"{rooms_a} rooms mein {base_a} log base + {extra_txt_hi} lagega.\n"
            "Thoda tight rahega.\n"
            f"{rooms_b} rooms lein toh sabko proper bed mil jayega.\n\n"
            f"*A - {rooms_a} rooms* ({base_a} + {extra_txt_hi})  Rs {price_a:,.0f}/night\n"
            f"*B - {rooms_b} rooms* (sabko bed)  Rs {price_b:,.0f}/night\n\n"
            "A ya B?"
        )
    return text_card(body)


def ask_custom_distribution(
    distribution_proposal: str = "3 rooms: 3 + 2 + 2",
    lang: str = "Hinglish",
) -> list[dict]:
    """
    §5.3 Custom Distribution Proposal (S3_FIT).
    """
    is_english = str(lang).lower() in ("english", "en", "casual english")
    if is_english:
        body = (
            f"{distribution_proposal}\n"
            "Let me know if you want a different split (e.g. 2+2+3)."
        )
    else:
        body = (
            f"{distribution_proposal}\n"
            "Alag chahiye toh batayiye (jaise 2+2+3)."
        )
    return text_card(body)


def ask_occupancy_preference() -> list[dict]:
    """Legacy alias — V3 uses three parallel modes at S4."""
    return text_card(
        "How would you prefer your room arrangements?\n\n"
        "1️⃣ Option 1 - Comfortable (no extra bed)\n"
        "2️⃣ Option 2 - Best value (fewest rooms with extra beds)\n"
        "3️⃣ Option 3 - Build your own\n\n"
        "👉 _Please reply with 1, 2, or 3._"
    )


def ask_guest_name_for_payment(prefill_name: str | None = None) -> list[dict]:
    """
    S7 Guest Details Capture — fires only after guest confirms quote.
    """
    if prefill_name:
        return text_card(
            f"Booking in the name of **{prefill_name}**?\n\n"
            f"Reply *yes* or write a new name."
        )
    return text_card(
        "Just one detail needed -\n"
        "What is the guest's full name? (as on ID for check-in)"
    )


def build_your_own_prompt(
    room_types: list[str] | None = None,
    problem: str | None = None,
) -> list[dict]:
    """
    S5_CUSTOM Build-your-own path prompt matching exact spec in Image 3.

    `problem` is why a previous answer could not be used, and replaces the
    opening line when the prompt is asked a second time. It is the only thing
    that tells the guest their last reply was read and could not be honoured,
    rather than ignored — without it the prompt repeats verbatim and looks like
    the bot did not hear them. None (the first ask) keeps the wording above.
    """
    type_str = " / ".join(room_types[:3]) if room_types else "Deluxe / Premier / Family"
    return text_card(
        f"{problem or 'Sure, please tell me:'}\n\n"
        f"  * How many rooms\n"
        f"  * Which type ({type_str})\n\n"
        f"_Example: 2 Deluxe + 1 Family_\n\n"
        f"I will check the availability and confirm."
    )


def ask_split_one_liner(room_types: list[str] | None = None) -> list[dict]:
    """The reply to the card's own "Tell me how you want to split" line.

    Deliberately one line. The guest has already chosen this path — they do not
    need the question restated, only the format. `build_your_own_prompt` says
    the same thing over five lines and is kept for the re-ask, where there is a
    reason to explain that the last answer could not be used.
    """
    # The caller derives these from `room_name` by deleting the word "ROOM",
    # which leaves "DELUXE  KING BED" — all caps and a double space. Harmless
    # inside the five-line prompt, ugly as an entire one-line message.
    names = [" ".join(n.split()).title() for n in (room_types or []) if n]
    if len(names) >= 2:
        example = f"2 {names[0]} + 1 {names[1]}"
    else:
        example = "2 Deluxe + 1 Family"
    return text_card(
        f"Sure — tell me how to split it (e.g. *{example}*) and I'll quote it. ✅"
    )


def show_menu(hotel_name: str = "our hotel") -> list[dict]:
    return greeting_response(hotel_name=hotel_name)


def ask_dates(missing_check_in: bool = True, missing_check_out: bool = True, lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if missing_check_in and missing_check_out:
        return text_card("Aapki check-in aur check-out dates kya hain?" if is_hi else "What are your check-in and check-out dates?")
    elif missing_check_out:
        return text_card("Check-out date kya hogi?" if is_hi else "What is your check-out date?")
    else:
        return text_card("Check-in date kya hogi?" if is_hi else "What is your check-in date?")


def ask_guests(lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    return text_card("Kitne log stay karenge? (Adults aur bachche bata dijiye)" if is_hi else "How many guests will be staying (adults and children)?")


def ask_guest_type() -> list[dict]:
    return ask_child_ages_followup()


def ask_num_rooms(lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    return text_card("Aapko kitne rooms chahiye?" if is_hi else "How many rooms do you need?")


def ask_room_selection() -> list[dict]:
    return text_card("Which room option would you like? You can tell me the room name or option number.")


def invalid_dates() -> list[dict]:
    return text_card(
        "Hmm, those dates don't look right. 🤔\n\n"
        "Please make sure:\n"
        "• Check-in is today or a future date\n"
        "• Check-out is after check-in\n\n"
        "Could you share the dates again?"
    )


def no_rooms_available() -> list[dict]:
    # A dead end reads as a dead bot. The three suggestions above are all things
    # the guest has to type; the fourth — reach a person — is now a tap.
    return button_card(
        "I'm sorry, we don't have rooms available that match your requirements right now. 😔\n\n"
        "Would you like to:\n"
        "• Try different dates?\n"
        "• Adjust the number of guests?\n"
        "• Check a different room type?",
        [{"id": "menu_reception", "title": "Talk to reception"}],
    )


def booking_confirmed(booking_id: str, check_in: str, check_out: str, **kwargs) -> list[dict]:
    return get_template("BOOKING_CONFIRMED", booking_id=booking_id, check_in=check_in, check_out=check_out, **kwargs)


def payment_link_message(
    payment_link: str,
    total: float,
    vpa: str = None,
    qr_code_url: str = None,
    booking_id: str = None,
    expiry_mins: int = 15,
    lang: str = "English",
) -> list[dict]:
    """A Razorpay payment link, optionally with the hotel's own QR beside it.

    Catalog v2.0 §4.1's isolation rule is about *substitution*: a gateway link
    must never be replaced by a UPI VPA the guest was not offered. Since
    `02aaf0a` (2026-10-26) a hotel with a **configured** VPA may show both — the
    link stays the primary payment method and the QR is an alternative beside
    it. `qr_code_url` is only ever passed for a stored VPA; see
    `PAYMENT_UPI_REQUIRES_CONFIGURED_VPA` in `app/config/settings.py` for why
    nothing is derived from the hotel's name.
    """
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    amt_str = render_currency(total)
    bid_line = f"Booking ID: *{booking_id}*\n" if booking_id else ""
    text = (
        f"💳 {bid_line}Apni booking confirm karne ke liye secure payment link:\n\n"
        f"🔗 {payment_link}\n\n"
        f"💰 Amount: ₹{amt_str}\n"
        f"⏱️ Yeh link {expiry_mins} minute ke liye valid hai.\n\n"
        f"Payment complete hote hi booking turant confirm ho jayegi! 🎉"
        if is_hi
        else f"💳 {bid_line}Here is your secure payment link to confirm your booking:\n\n"
        f"🔗 {payment_link}\n\n"
        f"💰 Amount: ₹{amt_str}\n"
        f"⏱️ Link valid for {expiry_mins} minutes.\n\n"
        f"Once paid, your booking is instantly confirmed! 🎉"
    )
    cards = button_card(text, [{"id": "payment_paid", "title": "I've paid"}])
    if qr_code_url:
        caption = f"Scan to pay via any UPI App (VPA: {vpa})" if vpa else "Scan to pay via any UPI App"
        cards.append({
            "type": "image",
            "url": qr_code_url,
            "image_url": qr_code_url,
            "media_url": qr_code_url,
            "caption": caption,
        })
    return cards


def payment_upi_direct(
    vpa: str,
    hotel_name: str = "our hotel",
    total: float | None = None,
    booking_id: str | None = None,
    lang: str = "English",
) -> list[dict]:
    """Strictly for direct UPI inquiries; clearly explains manual verification (Catalog v2.0 §4.2)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    amt_line = f"\n💰 Amount: ₹{render_currency(total)}" if total else ""
    bid_line = f"Booking ID: *{booking_id}*\n" if booking_id else ""
    text = (
        f"🏦 {bid_line}*Direct UPI Payment Details:*\n\n"
        f"• UPI ID / VPA: *{vpa}*\n"
        f"• Account: *{hotel_name}*{amt_line}\n\n"
        f"⚠️ *Zaroori:* Payment ke baad kripya screenshot ya UTR reference number yahan share karein "
        f"taaki reception verify karke booking confirm kar sake."
        if is_hi
        else f"🏦 {bid_line}*Direct UPI Payment Details:*\n\n"
        f"• UPI ID / VPA: *{vpa}*\n"
        f"• Account: *{hotel_name}*{amt_line}\n\n"
        f"⚠️ *Important:* After paying, please share your payment screenshot or UTR reference number here "
        f"so our front desk can verify and confirm your booking."
    )
    return button_card(text, [{"id": "payment_paid", "title": "I've paid"}])


def payment_pending(
    link: str = None,
    prepend_text: str = None,
    payment_link: str = None,
    total: float = None,
    lang: str = "English",
) -> list[dict]:
    """Payment pending reminder (Catalog v2.0 §4.3)."""
    effective_link = payment_link or link
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    link_str = f"\n🔗 {effective_link}" if effective_link else ""
    amt_str = f" (₹{render_currency(total)})" if total else ""
    text = (
        f"⏳ Aapka payment{amt_str} abhi pending hai.{link_str}\n\n"
        f"Apna room confirm karne ke liye kripya payment complete karein. "
        f"Agar aap payment kar chuke hain, toh screenshot yahan share karein."
        if is_hi
        else f"⏳ Your payment{amt_str} is still pending.{link_str}\n\n"
        f"To secure your room, please complete the payment. "
        f"If you've already paid, please share the screenshot here."
    )
    if prepend_text:
        text = f"{prepend_text}\n\n{text}"
    buttons = [
        {"id": "payment_paid", "title": "I've paid"},
        {"id": "menu_reception", "title": "Talk to reception"},
    ]
    return button_card(text, buttons)


def payment_failed(link: str = None, prepend_text: str = None, **kwargs) -> list[dict]:
    return payment_pending(link=link, prepend_text=prepend_text, **kwargs)


def payment_verified() -> list[dict]:
    return text_card("✅ Payment received! Let me confirm your booking...")


def payment_claimed_check(lang: str = "English") -> list[dict]:
    """Guest claims payment made; checking bank records (Catalog v2.0 §4.4)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "⏳ Hum aapki payment bank records mein check kar rahe hain... "
        "Agar aapne direct UPI kiya hai, toh kripya screenshot yahan share karein taaki hum turant verify kar sakein."
        if is_hi
        else "⏳ Checking our bank records for your payment... "
        "If you paid via direct UPI, please share the payment screenshot here so we can verify right away."
    )
    return text_card(text)


def payment_received_pending_confirmation(
    amount: float | None = None,
    booking_id: str | None = None,
    lang: str = "English",
) -> list[dict]:
    """Payment received, awaiting front desk bank reconciliation (Catalog v2.0 §4.5)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    amt_str = f"₹{render_currency(amount)}" if amount else "payment"
    bid_str = f" (Booking: {booking_id})" if booking_id else ""
    text = (
        f"⏳ Humein aapka {amt_str} mil gaya hai{bid_str}! Hamari front desk team bank records se verify kar rahi hai "
        f"(aamtaur par 15 minute ke andar). Verify hote hi aapki confirmed booking slip yahan bhej di jayegi."
        if is_hi
        else f"⏳ We have received your {amt_str}{bid_str}! Our front desk team is verifying the transaction with bank records "
        f"(usually within 15 minutes). Your confirmed booking slip will follow as soon as verified."
    )
    return text_card(text)


def payment_receipt_summary(payment_data: dict, lang: str = "English") -> list[dict]:
    """
    P0 Guard: Show extracted payment receipt details with pending verification indicator (never premature ✅).
    Catalog v2.0 §4.6.
    """
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    header = (
        "⏳ *Payment screenshot mila — front desk verify kar raha hai.*\n"
        if is_hi
        else "⏳ *Payment screenshot received — verifying with front desk.*\n"
    )
    lines = [header]

    field_labels = {
        "amount": "💰 Amount",
        "transaction_id": "🔑 Transaction ID",
        "status": "📊 Status",
        "sender_name": "👤 Sender",
        "receiver_name": "🏨 Receiver",
        "payment_app": "📱 Payment App",
        "timestamp": "🕐 Date/Time",
        "customer_name": "👤 Guest Name",
        "customer_phone": "📞 Phone",
        "check_in": "📅 Check-in",
        "check_out": "📅 Check-out",
        "room_type": "🛏️ Room Type",
        "num_rooms": "🏨 Rooms",
        "num_adults": "👥 Adults",
        "num_children": "👶 Children",
        "meal_plan": "🍽️ Meal Plan",
    }

    for field, label in field_labels.items():
        val = payment_data.get(field)
        if val is not None and str(val).strip():
            if field == "amount":
                lines.append(f"{label}: ₹{render_currency(val)}")
            else:
                lines.append(f"{label}: {val}")

    eta_note = (
        "\n_Front desk verification aamtaur par 15 minute ke andar ho jati hai._"
        if is_hi
        else "\n_Front desk verification usually takes under 15 minutes._"
    )
    lines.append(eta_note)
    return text_card("\n".join(lines))


def payment_amount_mismatch(
    expected: Any,
    received: Any,
    booking_id: str | None = None,
    lang: str = "English",
) -> list[dict]:
    """P0 Mismatch Template: Amount variance detected (Catalog v2.0 §4.7)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    bid_str = f" for booking {booking_id}" if booking_id else ""
    text = (
        f"⚠️ Screenshot mein ₹{render_currency(received)} dikh raha hai, jabki booking amount ₹{render_currency(expected)} hai{bid_str}.\n\n"
        f"Hamari front desk team ko review ke liye inform kar diya gaya hai. Agar yeh partial advance hai, toh team balance check-in par confirm karegi."
        if is_hi
        else f"⚠️ Payment screenshot shows ₹{render_currency(received)}, but the expected amount is ₹{render_currency(expected)}{bid_str}.\n\n"
        f"Our front desk has been alerted to review this. If this was a partial advance, our team will confirm the remaining balance."
    )
    return button_card(text, [{"id": "menu_reception", "title": "Talk to reception"}])


def payment_screenshot_unclear(lang: str = "English") -> list[dict]:
    """P0 Mismatch Template: Unclear/blurry payment screenshot (Catalog v2.0 §4.8)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Aapka screenshot thoda blur hai ya UTR / amount saaf nahi dikh raha. 📸\n\n"
        "Kripya apne payment app se clear screenshot share karein jisme UTR number aur amount clearly visible ho."
        if is_hi
        else "The uploaded screenshot appears unclear or is missing the UTR/transaction ID and amount. 📸\n\n"
        "Please share a clear screenshot from your payment app showing the completed transfer details."
    )
    return text_card(text)


def payment_receiver_mismatch(
    receiver_name: str,
    hotel_name: str = "our hotel",
    lang: str = "English",
) -> list[dict]:
    """P0 Mismatch Template: Receiver account mismatch (Catalog v2.0 §4.9)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        f"⚠️ Payment receipt par receiver ka naam *{receiver_name}* dikh raha hai, jabki hotel ka account *{hotel_name}* hai.\n\n"
        f"Kripya check karein ki payment sahi account mein gayi hai, ya hamare front desk se baat karein."
        if is_hi
        else f"⚠️ The payment receipt shows recipient as *{receiver_name}*, while our hotel account is *{hotel_name}*.\n\n"
        f"Please verify if the payment was sent to the correct account, or speak with our front desk team."
    )
    return button_card(text, [{"id": "menu_reception", "title": "Talk to reception"}])


def payment_link_expired(booking_id: str | None = None, lang: str = "English") -> list[dict]:
    """Payment link expiration recovery (Catalog v2.0 §4.10)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "⏱️ Payment link expire ho chuki hai. Kya aap nayi payment link chahte hain?"
        if is_hi
        else "⏱️ The payment link has expired. Would you like me to generate a new payment link for your stay?"
    )
    buttons = [
        {"id": "quote_confirm", "title": "New payment link"},
        {"id": "menu_reception", "title": "Talk to reception"},
    ]
    return button_card(text, buttons)


def fallback_response(level: int = 1, lang: str = "English") -> list[dict]:
    """3-Level Progressive Fallback (Catalog v2.0 §5.1)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if level <= 1:
        text = (
            "Main samajh nahi paya. 🤔 Kya aap kamre dekhna chahte hain, apni booking check karna chahte hain, ya reception se baat karna chahte hain?"
            if is_hi
            else "I didn't quite catch that. 🤔 Would you like to check room options, check a booking, or talk to reception?"
        )
        buttons = [
            {"id": "menu_rooms", "title": "Kamre aur rates" if is_hi else "Room types & rates"},
            {"id": "menu_booking", "title": "Booking check karein" if is_hi else "Check booking"},
            {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
        ]
        return button_card(text, buttons)
    elif level == 2:
        text = (
            "Mujhe samajhne mein thodi dikkat ho rahi hai. Kya main aapko front desk team se connect kar doon?"
            if is_hi
            else "I'm having trouble understanding your request. Would you like me to connect you with our front desk team?"
        )
        buttons = [
            {"id": "menu_reception", "title": "Talk to reception"},
            {"id": "menu_rooms", "title": "Room options"},
        ]
        return button_card(text, buttons)
    else:
        text = (
            "Main aapko turant front desk team se connect kar raha hoon taaki sahi madad mil sake. 🛎️ Ek executive jaldi reply karega."
            if is_hi
            else "I want to ensure you get the right help! I am connecting you directly with our front desk team now. 🛎️ Someone will assist you shortly."
        )
        return text_card(text)


def handoff_to_reception(
    reason: str | None = None,
    hotel_name: str = "our hotel",
    lang: str = "English",
) -> list[dict]:
    """Reception handoff (Catalog v2.0 §5.2)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        f"Aapko {hotel_name} front desk team se connect kar raha hoon. 🛎️ Hamari team jaldi hi is chat par aapki madad karegi."
        if is_hi
        else f"Connecting you to the {hotel_name} front desk team. 🛎️ Someone will assist you shortly on this chat."
    )
    return text_card(text)


def handoff_delay_update(hotel_name: str = "our hotel", lang: str = "English") -> list[dict]:
    """Handoff delay update (Catalog v2.0 §5.3)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        f"Front desk team ko inform kar diya gaya hai aur woh jaldi hi reply karenge. Thoda intezar karne ke liye shukriya! 🙏"
        if is_hi
        else f"Our front desk team has been notified and will reply shortly. Thank you for your patience! 🙏"
    )
    return text_card(text)


def thank_you_response(lang: str = "English") -> list[dict]:
    """Thank you response (Catalog v2.0 §5.4)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Aapka swagat hai! 😊 Agar kisi aur cheez ki zaroorat ho, toh main yahi hoon. Have a wonderful day!"
        if is_hi
        else "You're welcome! 😊 If you need anything else, I'm right here to help. Have a wonderful day!"
    )
    return text_card(text)


def unrelated_query_response(lang: str = "English") -> list[dict]:
    """Out-of-scope query redirect (Catalog v2.0 §5.5)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Main ek hotel booking assistant hoon. 🏨 Main room booking, prices, amenities aur hotel details mein madad kar sakta hoon.\n\n"
        "Kisi aur sawal ke liye hamari front desk team se baat kar sakte hain."
        if is_hi
        else "I am a specialized hotel booking assistant. 🏨 I can help with room reservations, rates, amenities, and hotel information.\n\n"
        "For other queries, our front desk team will be happy to assist you!"
    )
    buttons = [
        {"id": "menu_rooms", "title": "Kamre aur rates" if is_hi else "Room types & rates"},
        {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
    ]
    return button_card(text, buttons)


def injection_detected_response(hotel_name: str = "our hotel", lang: str = "English") -> list[dict]:
    """Security injection detected guard response (Catalog v2.0 §5.6)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        f"Main {hotel_name} mein booking aur hotel jankari ke liye yahan hoon. Aaj main aapki kya madad kar sakta hoon? 🏨"
        if is_hi
        else f"I am here to help you with bookings and hotel inquiries at {hotel_name}. How can I assist you today? 🏨"
    )
    return text_card(text)


def session_update_confirmation(field: str, value: str, greeting: str = "") -> list[dict]:
    prefix = f"{greeting.strip()} " if greeting and greeting.strip() else ""
    return text_card(f"{prefix}Got it! I've updated {field} to {value}. ✅")


def rooms_pricing_response(rooms_text: str) -> list[dict]:
    return text_card(
        "Please note that the prices mentioned below are base rates and may vary depending on your selected dates and room availability.\n\n"
        "Here are our available rooms and standard pricing:\n\n"
        f"{rooms_text}\n\n"
        "To check live availability for these rooms, just let me know your **check-in and check-out dates**! 📅\n\n"
        "Would you like to start a booking?"
    )


def existing_booking_summary(
    booking_id: str, check_in: str, check_out: str,
    adults: int, children: int, num_rooms: int,
    room_name: str, total_price: float | None = None,
    payment_status: str | None = None,
    booking_status: str | None = None,
    *,
    source: str = "confirmed",
) -> list[dict]:
    if source == "in_progress":
        lines = ["📋 *Booking In Progress*\n"]
    elif booking_status and str(booking_status).lower() == "cancelled":
        lines = ["📋 *Your Last Booking* (cancelled)\n"]
    else:
        lines = ["📋 *Your Booking Details*\n"]

    from app.utils.helpers import format_display_date
    if booking_id:
        lines.append(f"🆔 Booking ID: *{booking_id}*")
    if booking_status and source != "in_progress":
        lines.append(f"📌 Status: {booking_status}")
    lines.append(f"📅 Check-in: {format_display_date(check_in) or check_in}")
    lines.append(f"📅 Check-out: {format_display_date(check_out) or check_out}")
    lines.append(f"👥 Guests: {adults} adults, {children} children")
    lines.append(f"🏨 Rooms: {num_rooms}")
    if room_name:
        lines.append(f"🛏️ Room: {room_name}")
    if total_price is not None:
        lines.append(f"💰 Total: ₹{total_price:,.0f}")
    if payment_status:
        lines.append(f"💳 Payment: {payment_status}")
    # This summary used to be a plain read-out: the guest could see their stay and
    # then had to think of something to type. Both things they actually want next
    # are now taps. A drafted booking (source="in_progress") is not yet a stay,
    # so "Book a room" would only start a second one — it gets the door to a
    # person and nothing else.
    buttons = [{"id": "menu_reception", "title": "Talk to reception"}]
    if source != "in_progress":
        buttons.insert(0, {"id": "menu_book", "title": "Book a room"})
    return button_card("\n\n".join(lines), buttons)


def existing_booking_not_found(lang: str = "English") -> list[dict]:
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    # This is the reply a guest gets when the lookup found nothing, which is
    # exactly when they most want a human. Same wording, plus the door.
    buttons = [{"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"}]
    if is_hi:
        return button_card(
            "Agar aapne MakeMyTrip, Booking.com, Agoda ya direct hotel se booking ki hai, "
            "toh aapka reservation front desk records par confirmed hai! 🏨\n\n"
            "Kripya apna **Booking ID**, **Guest Name**, ya **Room Number** share karein — "
            "hum aapke stay ke liye kya madad kar sakte hain? 😊",
            buttons,
        )
    return button_card(
        "If you booked through MakeMyTrip, Booking.com, Agoda, or directly at the hotel, "
        "your reservation is confirmed on our front desk records! 🏨\n\n"
        "Please share your **Booking ID**, **Guest Name**, or **Room Number** — "
        "what can we help you with regarding your stay today? 😊",
        buttons,
    )


def existing_booking_not_found_dated(date_phrase: str, lang: str = "English") -> list[dict]:
    """The lookup found nothing, and the guest answered with a date.

    The plain ``existing_booking_not_found`` is the wrong reply here: it asks
    the same question the guest just tried to answer, word for word, so the
    conversation cannot move. This one names what the guest said and what is
    still missing.

    ``date_phrase`` is only ever echoed back. It is never used to widen the
    lookup — a booking found by date alone would be someone else's.
    """
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    buttons = [{"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"}]
    buttons.insert(0, {"id": "menu_book", "title": "Book a room"})
    phrase = f"**{date_phrase}**" if date_phrase else "those dates"
    if is_hi:
        return button_card(
            f"Main is number par {phrase} ki koi booking nahi dhundh paaya. 🤔\n\n"
            "Sirf date se reservation identify nahi hoti — kripya apna **Booking ID** "
            "(jaise BK-20261020-ABC123), booking par diya gaya **Guest Name**, ya "
            "**Room Number** share karein, main turant check karta hoon. 😊",
            buttons,
        )
    return button_card(
        f"I couldn't find a booking under this number for {phrase}. 🤔\n\n"
        "A date on its own doesn't identify a reservation — please share your "
        "**Booking ID** (it looks like BK-20261020-ABC123), the **Guest Name** on the "
        "booking, or your **Room Number**, and I'll check right away. 😊",
        buttons,
    )


def draft_booking_summary_note() -> list[dict]:
    return text_card(
        "This isn't confirmed in our system yet — say **continue** to pick it up, "
        "or **start fresh** to clear it and begin again."
    )


def something_else_response(hotel_name: str = "our hotel", lang: str = "English") -> list[dict]:
    """M4 Something Else Menu (Catalog v2.0 §1.4)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Zaroor! Aur kis cheez mein help chahiye? Aap seedha apna sawal likh sakte hain ya option chun sakte hain:"
        if is_hi
        else "Sure! What else can I help you with? You can type your question directly or pick an option:"
    )
    buttons = [
        {"id": "menu_rooms", "title": "Kamre aur rates" if is_hi else "Room types & rates"},
        {"id": "menu_reception", "title": "Reception se baat" if is_hi else "Talk to reception"},
        {"id": "menu_booking", "title": "Booking check karein" if is_hi else "Check booking"},
    ]
    return button_card(text, buttons)


def opt_out_confirmation(lang: str = "English") -> list[dict]:
    """Opt-out confirmation (Catalog v2.0 §6.1)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Aapko promotional messages se unsubscribe kar diya gaya hai. Active bookings ke alawa messages nahi aayenge. Dobara shuru karne ke liye 'START' likhein."
        if is_hi
        else "You have been unsubscribed from promotional messages. We will only contact you regarding active bookings. Reply 'START' anytime to re-enable."
    )
    return text_card(text)


def discount_request(lang: str = "English") -> list[dict]:
    """Discount inquiry handler (Catalog v2.0 §6.2)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Diye gaye rates hamare best direct booking rates hain sabhi taxes ke saath! 🏨 Badi bookings ya extended stay ke liye front desk manager se baat kar sakte hain."
        if is_hi
        else "Our quoted rates are our best direct booking prices with all taxes included! 🏨 For large group bookings or extended stays, our front desk manager can review special pricing."
    )
    buttons = [
        {"id": "menu_reception", "title": "Talk to manager"},
        {"id": "quote_confirm", "title": "Proceed at best rate"},
    ]
    return button_card(text, buttons)


def call_request(phone: str | None = None, lang: str = "English") -> list[dict]:
    """Call request / phone inquiry handler (Catalog v2.0 §6.3)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    phone_txt = f"*{phone}*" if phone else "hamare front desk number"
    text = (
        f"Aap reception se seedha {phone_txt} par baat kar sakte hain, ya chahein toh team aapko yahan message karegi?"
        if is_hi
        else f"You can reach our reception directly at {f'*{phone}*' if phone else 'our front desk line'}, or would you like our team to message you here?"
    )
    buttons = [{"id": "menu_reception", "title": "Request callback"}]
    return button_card(text, buttons)


def still_there_nudge(lang: str = "English") -> list[dict]:
    """Session idle nudge (Catalog v2.0 §6.4)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Main bas check kar raha tha — kya aap booking aage badhana chahte hain? 🏨"
        if is_hi
        else "Just checking in — would you like to continue with your booking? 🏨"
    )
    buttons = [
        {"id": "nudge_continue", "title": "Continue booking"},
        {"id": "nudge_cancel", "title": "Not now"},
    ]
    return button_card(text, buttons)


def location_directions(address: str | None = None, gmaps_url: str | None = None, lang: str = "English") -> list[dict]:
    """Location & directions info (Catalog v2.0 §6.5)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    addr_line = address or "Our hotel"
    maps_line = f"\nGoogle Maps: {gmaps_url}" if gmaps_url else ""
    text = (
        f"📍 *Hotel ka Pata:*\n{addr_line}{maps_line}"
        if is_hi
        else f"📍 *Hotel Location:*\n{addr_line}{maps_line}"
    )
    return text_card(text)


def photos_amenities(amenities_text: str | None = None, photos_url: str | None = None, lang: str = "English") -> list[dict]:
    """Photos & amenities info (Catalog v2.0 §6.6)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    amenities = amenities_text or "Free Wi-Fi, 24/7 Room Service, Hot Water, AC rooms"
    photos = f"\nPhotos: {photos_url}" if photos_url else ""
    text = (
        f"📸 *Photos aur Suvidhayein:*\n{amenities}{photos}"
        if is_hi
        else f"📸 *Photos & Amenities:*\n{amenities}{photos}"
    )
    return text_card(text)


def early_checkin_request(standard_time: str = "12:00 PM", fee: float | None = None, lang: str = "English") -> list[dict]:
    """Early check-in inquiry (Catalog v2.0 §6.7)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    fee_part = f" Additional fee: ₹{render_currency(fee)}." if fee else ""
    text = (
        f"Hamara standard check-in time *{standard_time}* hai. Early check-in arrival par room availability par depend karta hai.{fee_part}"
        if is_hi
        else f"Our standard check-in time is *{standard_time}*. Early check-in is subject to room availability upon arrival.{fee_part}"
    )
    return text_card(text)


def cancel_request(booking_id: str | None = None, policy: str | None = None, lang: str = "English") -> list[dict]:
    """Cancellation confirmation prompt (Catalog v2.0 §6.8)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    bid = f" *{booking_id}*" if booking_id else ""
    pol = policy or "Free cancellation up to 24 hours prior to check-in."
    text = (
        f"Kya aap booking{bid} cancel karna chahte hain?\n\nPolicy: {pol}"
        if is_hi
        else f"Are you sure you want to cancel booking{bid}?\n\nPolicy: {pol}"
    )
    buttons = [
        {"id": "cancel_confirm", "title": "Yes, cancel"},
        {"id": "cancel_abort", "title": "Keep booking"},
    ]
    return button_card(text, buttons)


def cancel_confirm(booking_id: str | None = None, lang: str = "English") -> list[dict]:
    """Cancellation confirmed message (Catalog v2.0 §6.9)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    bid = f" *{booking_id}*" if booking_id else ""
    text = (
        f"Aapki booking{bid} cancel ho gayi hai. Refund hotel policy ke mutabiq process kiya jayega."
        if is_hi
        else f"Your booking{bid} has been cancelled. Any eligible refund will be processed per hotel policy."
    )
    return text_card(text)


def complaint_followup(ticket_id: str | None = None, lang: str = "English") -> list[dict]:
    """Complaint acknowledgement and handoff (Catalog v2.0 §6.10)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    ref = f" (Ref: #{ticket_id})" if ticket_id else ""
    text = (
        f"Batane ke liye shukriya. Hamari management team ko inform kar diya gaya hai{ref} aur woh jaldi hi assist karenge."
        if is_hi
        else f"Thank you for sharing your feedback. We take guest concerns seriously. Our management team has been alerted{ref} to assist you promptly."
    )
    return text_card(text)


def language_switched(new_lang: str = "Hinglish") -> list[dict]:
    """Language switch acknowledgement (Catalog v2.0 §6.11)."""
    is_hi = str(new_lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Bilkul! Hum Hinglish mein baat karenge. Main aapki kaise madad kar sakta hoon? 😊"
        if is_hi
        else "Sure! Switching to English. How can I help you today? 😊"
    )
    return text_card(text)


def voice_note_unsupported(lang: str = "English") -> list[dict]:
    """Voice note fallback (Catalog v2.0 §6.12)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Main abhi voice note sun nahi sakta. 🎤 Kripya apna sawal text mein likhein, main turant madad karunga!"
        if is_hi
        else "I cannot listen to voice notes directly yet. 🎤 Please type your message or question, and I'll be glad to help!"
    )
    return text_card(text)


def processing_wait(lang: str = "English") -> list[dict]:
    """Interim processing notice (Catalog v2.0 §6.13)."""
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    text = (
        "Aapke liye live availability aur rates check kar raha hoon, bas ek second... ⏳"
        if is_hi
        else "Checking live availability and best rates for you, please give me a moment... ⏳"
    )
    return text_card(text)


def non_payment_image_redirect(user_message: str = "") -> list[dict]:
    """
    Acknowledge a non-payment image upload and gently redirect toward room booking.
    Extracts the image type from the [IMAGE_ANALYSIS] block for a contextual response.
    """
    import re
    image_type = "image"
    type_match = re.search(r'Type:\s*(.+)', user_message)
    if type_match:
        image_type = type_match.group(1).strip()

    if image_type == "Room Image":
        return text_card(
            "Thanks for sharing this room photo! 📸\n\n"
            "I noticed this isn't a payment receipt. If you're comparing rooms, "
            "I'd love to show you what we have available — our rooms offer great value with "
            "premium amenities! 🏨\n\n"
            "Would you like to **book a room** with us? Just share your check-in dates and preferences!"
        )
    elif image_type == "Complaint":
        return text_card(
            "I'm sorry to see this issue. 😔 We take guest concerns very seriously.\n\n"
            "I've noted the problem shown in your photo. Our team will address this promptly. "
            "If you'd like to speak with the front desk directly, please let us know.\n\n"
            "Is there anything else I can help you with?"
        )
    elif image_type == "Document":
        return text_card(
            "Thank you for sharing this document. 📄\n\n"
            "I noticed this isn't a payment receipt. If you're looking to book a room with us, "
            "I'd be happy to help! Just share your check-in dates and room preferences. 🏨\n\n"
            "Or if you need help with something else, feel free to ask!"
        )
    else:
        # General / unrecognized image
        return text_card(
            "Thank you for sharing this image! 🙂\n\n"
            "It doesn't appear to be a payment receipt. If you were trying to submit a payment, "
            "please upload a clear screenshot of your UPI/bank payment confirmation.\n\n"
            "Otherwise, I'd love to help you explore our rooms and make a booking! "
            "Just share your check-in dates and preferences to get started. 🏨"
        )
