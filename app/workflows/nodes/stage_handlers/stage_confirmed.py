"""
Stage 2 Handler — Confirmed Booking & Pre-Arrival Concierge.

Sir's Specification:
- A booking enters Stage 2 only when Razorpay webhook succeeds, hotelier confirms, or manual payment marked.
- ZERO SELLING. Never restart the booking flow, never ask dates, never open a draft.
- Never call confirmation message a bill, receipt, or invoice — MUST be called 'confirmation slip'.
- S53: 'Rate kya hai?' -> Show what guest already paid. No room menu!
- S54: 'Meri booking confirm hai?' -> Show booking ID, dates, room, name, amount paid, balance.
- S55: 'Station se kaise aayein?' -> Route from City Central Station / airport, map pin, front desk phone. No selling.
- S56: 'Hi' -> Personal greeting + booking summary. No welcome menu, no continue/fresh start.
- S57: 'Ek aur room chahiye' -> Ask once if separate booking; if yes, open separate Lead draft.

Amended 2026-10-25 (product): ZERO SELLING means *no draft*, not *no
information*. A confirmed guest asking what the rooms cost is answered with the
hotel's price list, and ordinary questions are answered from the knowledge base.
Only an explicit booking request still goes through S57's "is this a separate
booking?" question. Nothing in this handler repeats "your booking is confirmed"
unless the guest asked about their booking.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.prompts.templates import text_card
from app.repositories import task_repository
from app.services.hotel_identity import HotelIdentity, resolve_hotel_identity
from app.services.second_booking import second_booking_gate_enabled
from app.workflows.nodes.stage_handlers.kb_answer import (
    is_selling_question,
    knowledge_base_answer,
)
from app.workflows.nodes.stage_handlers.room_price_answer import room_price_answer
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# The guest answering S57's "is this a separate booking?" with a yes.
#
# Word boundaries on purpose. This used to be `any(w in user_msg_lower ...)`, so
# "ha" matched inside "kya aapke paas hai" and started a draft the guest never
# asked for. The multi-word entries are the phrasings S57's own prompt suggests,
# so a guest who copies the instruction always matches.
_SEPARATE_BOOKING_YES = re.compile(
    r"\b(?:yes|yeah|yep|yup|haan|han|haa|ha|bilkul|new booking|naya booking|"
    r"separate booking|alag booking)\b",
    re.IGNORECASE,
)

# The guest answering S57's question with a no. Anchored at the start of the
# message so a refusal that happens to *contain* these words later — "no
# problem, but what is the wifi?" — is not read as an answer to a question
# about a booking. `\b` after each alternative keeps "not" out of "nothing".
_SEPARATE_BOOKING_NO = re.compile(
    r"^\s*(?:no|nope|nahi|nhi|na|not\s+now|abhi\s+nahi|not\s+needed|no\s+thanks)\b",
    re.IGNORECASE,
)

# A confirmed guest asking to move their stay. This is a regex, not the
# substring list it replaced, because that list held `"change date"` — singular,
# no words in between — and so missed the most natural phrasing there is.
# Measured on a replay turn 2026-10-10: "can you change the dates to next week?"
# raised no task at all, because "change the dates" is not "change date".
_CHANGE_DATE_RE = re.compile(
    r"\b(?:change|move|shift|resched\w*|postpone|prepone)\b[^.?!]{0,20}?"
    r"(?:dates?|stay|booking|reservations?|rooms?)\b"
    r"|\b(?:date|dates)\s+change\b"
)

# Everything else a guest says when they want the confirmed stay changed.
#
# Measured 2026-10-25 on a probe of twelve real phrasings against this clone:
# five raised no ticket and nothing else picked them up, so the guest was told
# "I'm not entirely sure about that" — or, for "add one more night to my
# booking", that everything was fine. The misses were "extend my stay",
# "shorten my stay", "postpone my trip", "check in a day earlier" and that last
# one. Each is an amendment in plain English; none carries the word "date".
_AMEND_OTHER_RE = re.compile(
    r"\b(?:extend|shorten|lengthen)\b[^.?!]{0,24}\b(?:stays?|bookings?|nights?|days?)\b"
    r"|\b(?:add|reduce|remove)\b[^.?!]{0,20}\b(?:nights?|days?)\b"
    r"|\b(?:more|fewer|less|extra)\s+nights?\b"
    r"|\b(?:postpone|prepone|resched\w*)\b"
    r"|\b(?:check[\s-]?in|check[\s-]?out)\b[^.?!]{0,24}\b(?:earlier|early|later|late)\b",
    re.IGNORECASE,
)

# A request to cancel that never says "cancel".
#
# Same probe, same day: "I don't want to come anymore" is the most natural way a
# guest withdraws, and it reached the knowledge base, which apologised for not
# understanding. Anchored on the withdrawal itself — "don't want to come",
# "won't be able to make it" — never on a bare "no", which would fire on
# "no, actually we'll be there by 3".
_CANCEL_OTHER_RE = re.compile(
    r"\b(?:don'?t|do\s+not|dont)\s+(?:want|wish)\s+to\s+(?:come|stay|visit|travel)\b"
    r"|\bwon'?t\s+be\s+able\s+to\s+(?:make\s+it|come)\b"
    r"|\bcan'?t\s+(?:come|make\s+it)\b"
    r"|\bnot\s+coming\b"
    r"|\bnahi\s+aa\s*(?:paayenge|payenge|unga|ungi|sakta|sakti)\b",
    re.IGNORECASE,
)

# "I don't want to cancel" is not a cancellation.
#
# `is_cancel` has always been a bare "cancel" substring test, so this sentence
# already raised a front-desk task to approve the very thing the guest ruled out
# — the same failure `_CHANGE_DATE_RE`'s comment records for "change the dates,
# don't cancel". The window is short on purpose: a negation must sit in the same
# clause as the cancel word, not three sentences earlier.
_CANCEL_NEGATED_RE = re.compile(
    r"\b(?:don'?t|do\s+not|dont|not|mat|nahi|nhi)\b[^.?!]{0,20}?\b(?:cancel|refund)\b",
    re.IGNORECASE,
)

# The rest of the amendment cues carry no such plural/word-order trap and stay
# plain substrings.
_AMENDMENT_CUES = (
    "modify", "early check in", "early checkin",
    "late checkout", "late check out", "tarikh badalna",
)

# A bare "ok" was matched as a substring until 2026-10-25, so *every* message
# containing the letters o-k was answered with "Ji! Hum aapke swagat ke liye
# taiyar hain" — including "add one more night to my booking", where "bo-ok-ing"
# is the offender. Anchoring the ack at the start of the message keeps the
# acknowledgement and stops it swallowing sentences that merely contain it; a
# guest who writes "ok ji" or "okay done" still matches.
_OK_ACK_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|k|theek|thik|achh?a|acch?a|got\s+it|samajh\s+gaya)\b",
    re.IGNORECASE,
)

# A question ABOUT the policy is not a request to act on it.
#
# `is_cancel` below is a plain substring test, and "cancellation" contains
# "cancel" — so "what's your cancellation policy?" from a confirmed guest
# raised a BOOKING_CANCELLATION task and woke the front desk for a guest who
# only wanted to read the terms. Same shape as the `"change date"` miss above:
# the cue and the thing it names are not the same word.
#
# The discrimination is adjacency, not a keyword list. A policy noun directly
# after the cancel/refund word ("cancellation policy", "refund rules",
# "cancellation fee") is a question; the same nouns sitting *before* it
# ("policy on cancellation") are caught by the second alternative. A genuine
# request never puts a policy noun next to the verb — "I want to cancel my
# booking, what's your policy?" does not match, because the two are not
# adjacent, and it stays a cancellation.
_POLICY_QUESTION_RE = re.compile(
    r"\b(?:cancel\w*|refund\w*)\s+(?:polic\w*|rules?|terms?|charges?|fees?|process)\b"
    r"|\b(?:polic\w*|rules?|terms?|charges?|fees?)\b[^.?!]{0,20}\b(?:cancel\w*|refund\w*)\b"
)


#: A request to see the rooms, in either of the two shapes it arrives in.
#:
#: The buttons are resolved through `app.utils.button_registry` — the same
#: declaration `button_card` validates against — so the tap, the id the CRM may
#: one day forward, and a guest typing the title all land on one answer. The
#: regex covers the phrasings no button carries ("show me available rooms").
#:
#: Deliberately not "room" on its own: "does the deluxe room have AC" is a
#: factual question the knowledge base answers, and reads nothing like a request
#: to browse.
_ROOMS_BROWSE_RE = re.compile(
    r"\broom\s*(?:types?|options?|list|catalogue|catalog)\b"
    r"|\bavailable\s+rooms?\b"
    r"|\bkambre?\s*(?:aur|ke|ka)\s*(?:rate|daam)\b"
    r"|\b(?:rate|price)\s*list\b",
    re.IGNORECASE,
)


def _wants_room_browse(user_msg: str) -> bool:
    """Whether this turn asks to see the rooms, rather than asking about them."""
    from app.utils.button_registry import resolve_button

    tapped = resolve_button(user_msg)
    if tapped is not None and tapped.action in ("SHOW_ROOMS", "BOOK_NOW"):
        return True
    return bool(_ROOMS_BROWSE_RE.search(user_msg or ""))


def _wants_price_list(user_msg: str) -> bool:
    """Whether this turn asks what the rooms cost.

    Deliberately borrows `is_selling_question` — the same regex `kb_answer`
    consults to *refuse* a selling question to a booked guest. Sharing the
    definition is what stops the two drifting: whatever the knowledge-base guard
    declines to answer is exactly what the price list has to take over, so a
    phrasing added to one is added to both.

    `BOOK_NOW` is not here. "Book a room" is a booking intent, not a price
    question, and it belongs to S57 — the branch that asks whether the new stay
    is separate from the confirmed one.
    """
    from app.utils.button_registry import resolve_button

    tapped = resolve_button(user_msg)
    if tapped is not None and tapped.action == "SHOW_ROOMS":
        return True
    if _ROOMS_BROWSE_RE.search(user_msg or ""):
        return True
    return is_selling_question(user_msg)


def _format_confirmation_slip(
    booking_id: str,
    guest_name: str,
    check_in: str | None,
    check_out: str | None,
    room_name: str,
    adults: int,
    amount_paid: float,
    total_amount: float,
    ident: HotelIdentity,
    has_payment_record: bool = True,
) -> str:
    balance = max(0.0, total_amount - amount_paid)
    # A row can be `confirmed` and still carry no stay: older writes left
    # `check_in`, `check_out`, `rooms` and `num_nights` null and kept a total on
    # its own (measured 2026-10-23 — BK-6E14274A, created June, reprinted as
    # "Dates: None se None", "0 Guest(s)", "Balance Due: Rs 50,000"). Printing
    # that at a guest is worse than printing nothing, and the total is exactly
    # the part we cannot stand behind: with no dates there is no quote behind it.
    # Flag off restores the old slip verbatim, empty lines and all.
    show_unknown = not get_settings().CONFIRMATION_SLIP_HIDES_UNKNOWN_STAY_FIELDS
    has_dates = show_unknown or bool(check_in and check_out)
    # The caller passes `has_payment_record=False` for a CRM-confirmed guest, who has no
    # row of ours — their `amount_paid` is a default 0.0, not a fact. Printing
    # "Advance Paid: Rs 0" to a guest who has paid is worse than saying nothing, and the
    # CRM is not a payment record to read one from instead.
    if not has_payment_record or not has_dates:
        balance_line = "• *Payment:* Front desk ke paas record hai — woh detail bata denge."
    elif balance > 0:
        balance_line = f"• *Balance Due at Check-in:* Rs {balance:,.0f}"
    else:
        balance_line = "• *Payment Status:* Fully Paid (Nil Balance)"

    # Address and phone come from the property's own record. Each line is
    # dropped entirely when the record has nothing, rather than printed empty
    # or borrowed from another hotel.
    address_line = f"📍 *Hotel Address:* {ident.address}\n" if ident.address else ""
    # Same reason as the caller's fragments: a CRM-confirmed guest has no id of ours,
    # and this slip is printed straight at the guest.
    booking_id_line = f"• *Booking ID:* `{booking_id}`\n" if booking_id else ""
    # Empty for a CRM-confirmed guest the CRM also has no room for — printing the label
    # with nothing after it reads worse than leaving the line out.
    room_line = f"• *Room Category:* {room_name}\n" if room_name else ""
    # Same rule as the balance: an advance against a stay with no dates is not a
    # number to put in front of a guest, whatever the row says.
    advance_line = (
        f"• *Advance Paid:* Rs {amount_paid:,.0f}\n"
        if has_payment_record and has_dates
        else ""
    )
    dates_line = f"• *Dates:* {check_in} se {check_out}\n" if has_dates else ""
    # `adults` defaults to None on a row that never recorded a party; "0 Guest(s)"
    # reads as a booking for nobody rather than as missing information.
    guests_line = (
        f"• *Guests:* {adults} Guest(s)\n" if (adults or show_unknown) else ""
    )

    return (
        f"🏨 *{ident.name} — Booking Confirmation Slip*\n\n"
        f"Aapka reservation fully confirmed hai! Swagat ki taiyari hai:\n\n"
        f"{booking_id_line}"
        f"• *Guest Name:* {guest_name}\n"
        f"{dates_line}"
        f"{room_line}"
        f"{guests_line}"
        f"{advance_line}"
        f"{balance_line}\n\n"
        f"{address_line}"
        f"📞 *{ident.front_desk_clause()}*\n\n"
        f"_Kisi bhi madad ke liye aap seedhe is chat par pooch sakte hain._"
    )


async def handle_stage_confirmed(state: ConversationState) -> ConversationState:
    hotel_id = state.get("hotel_id", 111111)
    phone = state.get("customer_identifier") or ""
    user_msg = (state.get("user_message") or "").strip()
    user_msg_lower = user_msg.lower()
    ident = await resolve_hotel_identity(hotel_id, state)
    phone_md = f"**{ident.phone}**" if ident.phone else "**the front desk at reception**"
    # The map pin is only offered when this property actually has one. It used
    # to default to Sample Hotel's pin for every tenant that was not Sample Property.
    map_line = f"🗺️ *Google Maps Pin:* {ident.map_link}\n\n" if ident.map_link else ""
    location_line = f"📍 *Hotel Location:* {ident.address}\n" if ident.address else ""

    active_booking = state.get("active_booking") or {}
    raw_booking_id = (
        active_booking.get("booking_id")
        or state.get("active_booking_id")
        or state.get("booking_id")
    )
    # A booking MUST have a valid, non-placeholder booking ID to be in Stage 2 Confirmed
    # — unless the stay was confirmed by staff in the CRM, which leaves no row of ours.
    from app.utils.helpers import guest_is_confirmed

    has_valid_booking = guest_is_confirmed(state, raw_booking_id)

    if not has_valid_booking:
        # Check if the guest has an active hold or provisional payment reference
        is_hold = bool(
            (raw_booking_id and str(raw_booking_id).startswith("PAY-"))
            or active_booking.get("payment_link")
            or state.get("payment_link")
            or (active_booking.get("check_in") and (active_booking.get("total_price") or state.get("total_price")))
        )
        if is_hold:
            logger.info(
                "STAGE_2_CONFIRMED_GHOST_GUARD | hotel=%s phone=%s hold detected (%s); prompting for payment",
                hotel_id, phone, raw_booking_id,
            )
            link = state.get("payment_link") or active_booking.get("payment_link")
            ci = str(active_booking.get("check_in") or state.get("check_in") or "")
            co = str(active_booking.get("check_out") or state.get("check_out") or "")
            stay_dates = f"{ci} se {co}" if co else ci
            r_name = str(active_booking.get("room_name") or active_booking.get("room_type") or state.get("room_type_preference") or "")
            room_disp = f" ({r_name})" if r_name else ""
            amt = float(active_booking.get("total_price") or state.get("total_price") or active_booking.get("amount") or 0.0)
            adv = float(active_booking.get("payment_link_amount") or state.get("payment_link_amount") or (amt * 0.3 if amt > 0 else 0.0))
            link_part = f"\n\n🔗 Advance payment link: {link}" if link else ""
            raw_name = str(active_booking.get("guest_name") or state.get("guest_name") or "").strip()
            if raw_name.lower() in ("thank", "thanks", "request", "guest", "valued guest", "none", "unknown"):
                raw_name = ""
            namaste_greeting = f"Namaste {raw_name}!" if raw_name else "Namaste!"

            resp = (
                f"{namaste_greeting} 🙏 Welcome to {ident.name}.\n\n"
                f"Aapka reservation{room_disp} {stay_dates} ke liye pending payment hai.\n"
                f"💰 Total Amount: ₹{amt:,.0f}\n"
                f"📌 Advance Required (30%): ₹{adv:,.0f}"
                f"{link_part}\n\n"
                f"Rooms hold karne ke liye kripya advance payment complete karein. Payment hone ke baad receipt yahan share karein to hum confirmation slip bhej denge! 🙏"
            )
            return {
                **state,
                "booking_stage": "payment_pending",
                "lifecycle_stage": "payment_pending",
                "next_action": "payment_pending",
                "bot_response": text_card(resp),
            }

        logger.warning(
            "STAGE_2_CONFIRMED_GHOST_GUARD | hotel=%s phone=%s has no active booking in DB; demoting to lead",
            hotel_id, phone,
        )
        demoted_state = {
            **state,
            "booking_stage": "lead",
            "lifecycle_stage": "lead",
            "active_booking": {},
            "active_booking_id": None,
            "booking_id": None,
        }
        # Check if guest message is a refusal / decline / negative response to a campaign or offer
        is_refusal = any(
            p in user_msg_lower for p in (
                "no thanks", "no thank you", "nahi chahiye", "nahi", "not interested",
                "not needed", "dont need", "cancel", "stop", "nope", "nhi"
            )
        ) or user_msg_lower in ("no", "na", "n")
        if is_refusal:
            resp = (
                "Dhanyawad! Agar bhavishya mein aapko a stay in the city ki zaroorat ho, "
                "toh humse zaroor sampark karein. Shubh din! 🙏"
            )
            return {
                **demoted_state,
                "next_action": None,
                "bot_response": text_card(resp),
            }

        is_greeting = any(user_msg_lower == g for g in ("hi", "hello", "hey", "namaste", "namaskar"))
        if is_greeting:
            resp = (
                f"🏨 *Namaste! Welcome to {ident.name}.* 🙏\n\n"
                "Main aapki booking mein kya madad kar sakta hoon? "
                "Kripya apni dates aur number of guests batayein."
            )
            return {
                **demoted_state,
                "next_action": "need_dates",
                "bot_response": text_card(resp),
            }

        resp = (
            f"Namaste! {ident.name} mein stay booking ya room enquiry ke liye kripya "
            f"apni check-in dates aur guests ki sankhya batayein. Main aapko best options dikhata hoon! 🏨"
        )
        return {
            **demoted_state,
            "next_action": "need_dates",
            "bot_response": text_card(resp),
        }

    booking_id = raw_booking_id
    if booking_id and str(booking_id).startswith("PAY-"):
        booking_id = None
    # A stay confirmed by staff in the CRM has no booking id of ours, so the strings
    # below would otherwise print "(None)" at the guest. Build the three display
    # fragments once, here, and use them at the guest-facing sites. `booking_id`
    # itself stays raw: the staff-task payloads take it as a field, and a null id
    # there is honest — staff can see this guest has no bot-side row.
    booking_ref = f" ({booking_id})" if booking_id else ""
    booking_ref_bold = f" **{booking_id}**" if booking_id else ""
    booking_id_line = f"• *Booking ID:* `{booking_id}`\n" if booking_id else ""
    raw_name = str(active_booking.get("guest_name") or state.get("guest_name") or "").strip()
    if raw_name.lower() in ("thank", "thanks", "request", "guest", "valued guest", "none", "unknown"):
        raw_name = ""
    guest_name = raw_name or "Guest"
    namaste_greeting = f"Namaste {guest_name}!" if raw_name else "Namaste!"
    name_phrase = f" {guest_name}" if raw_name else ""
    check_in = str(active_booking.get("check_in") or active_booking.get("check_in_date") or state.get("check_in") or "Upcoming Stay")
    check_out = str(active_booking.get("check_out") or active_booking.get("check_out_date") or state.get("check_out") or "")
    # "{check_in} se {check_out}" is only a sentence when both dates exist. A
    # date-less row made check_in the "Upcoming Stay" placeholder while
    # check_out stayed empty, so the guest read
    #   "… (BK-EVAL-SF1) Upcoming Stay se  (Deluxe Room King Bed) ke liye …"
    # — a sentence with a hole in it. Keep the pair in one fragment so it can
    # never be half-printed.
    stay_dates = f"{check_in} se {check_out}" if check_out else check_in
    # No hardcoded default. A guest with a booking row reads their room off it; a
    # CRM-confirmed guest has no row, so their room comes from the CRM's own record of
    # the stay. "Deluxe Room" used to stand here as the last resort and was shown to
    # real guests who had booked something else entirely.
    crm_room_type = state.get("crm_confirmed_room_type") if state.get("crm_confirmed_stay") else None
    room_name = str(
        active_booking.get("room_name")
        or active_booking.get("room_type")
        or state.get("room_type_preference")
        or crm_room_type
        or ""
    ).strip()
    room_disp = f" ({room_name})" if room_name else ""
    _dates_part = f", {stay_dates}" if check_out else ""
    room_with_dates = (
        f" ({room_name}{_dates_part})" if room_name
        else (f" ({stay_dates})" if check_out else "")
    )
    # A CRM-confirmed guest has no row of ours, so these used to fall through to
    # the literals — 2 guests for a party of 4, and Rs 0 paid for a guest who had
    # paid. The party size is read from the CRM's own record of the stay (it is
    # descriptive, like the room name); money is deliberately NOT, because the
    # CRM's `bookingAmount` is what was *discussed*, not what was received — see
    # `crm_confirmation_repository`'s docstring.
    crm_details = state.get("crm_confirmed_details") if state.get("crm_confirmed_stay") else None
    crm_details = crm_details if isinstance(crm_details, dict) else {}
    adults = int(
        active_booking.get("adults")
        or state.get("adults")
        or crm_details.get("adults")
        or 2
    )
    amount_paid = float(active_booking.get("amount_paid") or state.get("amount_paid") or 0.0)
    total_amount = float(active_booking.get("total_price") or active_booking.get("amount") or state.get("total_price") or amount_paid or 0.0)
    balance = max(0.0, total_amount - amount_paid)

    logger.info(
        "STAGE_2_CONFIRMED_DISPATCH | hotel=%s phone=%s booking_id=%s msg='%s'",
        hotel_id, phone, booking_id, user_msg[:60],
    )

    # ── Answer what the rooms cost ───────────────────────────────────────────
    #
    # Product, 2026-10-25: *"bot should be able to answer room prices and normal
    # questions"*. This stage used to refuse both ways round: `is_rate_ask`
    # below met every price word with "you already have a booking, no new quote
    # needed", and `kb_answer`'s selling guard turned a room-list question into
    # the directions paragraph at the tail. Neither answers the question.
    #
    # Before S57 on purpose. A "Room types & rates" tap used to be answered with
    # S57's "is this a separate booking?" question, which answers a price
    # question with a question. The list is the answer; the booking intent is
    # still S57's, because `_wants_price_list` does not accept `BOOK_NOW`.
    #
    # Showing prices sells nothing and creates no booking. The duplicate-booking
    # risk this stage's ZERO SELLING rule was written for lives in the draft,
    # and that is still gated behind S57's explicit question.
    if get_settings().CONFIRMED_STAGE_ANSWERS_ROOM_PRICES and _wants_price_list(user_msg):
        priced = await room_price_answer(
            state, stage="confirmed", log_tag="STAGE_2_CONFIRMED",
        )
        if priced is not None:
            return priced

    # ── S57: Explicit Request for Another / Second Room ──────────────────────
    # A guest who taps "Book a room" is asking the same thing S57 exists for —
    # they have not said "for my current stay". So the answer is the one already
    # designed for a second room: say the stay is safe, ask once whether this is
    # a separate booking. The tap used to fall through to the directions/parking
    # paragraph, answering a question the guest had not asked. Product,
    # 2026-10-18.
    #
    # "Room types & rates" used to arrive here too. Since 2026-10-25 the price
    # branch above answers that tap with the price list, so `wants_rooms` now
    # only carries it on deployments where CONFIRMED_STAGE_ANSWERS_ROOM_PRICES
    # is off.
    wants_rooms = (
        get_settings().CONFIRMED_ROOMS_ASK_SEPARATE_BOOKING
        and _wants_room_browse(user_msg)
    )
    is_separate_booking_trigger = wants_rooms or any(
        p in user_msg_lower for p in (
            "ek aur room", "another room", "one more room", "new booking", "separate booking",
            "naya room", "doosra room", "dusra room", "2nd room", "second room", "additional room"
        )
    )
    clarification_msg = (
        f"{namaste_greeting} 🙏 Aapki ek booking{booking_ref} already {stay_dates} ke liye confirmed hai.\n\n"
        f"Kya aap is confirmed stay ke alawa ek **naya alag booking** karna chahte hain?\n"
        f"👉 Kripya reply karein **'Yes, new booking'** ya **'No'**."
    )
    if is_separate_booking_trigger and not state.get("separate_booking_clarification_asked"):
        # Ask exactly ONCE: Is this a new booking, separate from your current stay?
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "separate_booking_clarification_asked": True,
            "bot_response": text_card(clarification_msg),
            "next_action": "awaiting_separate_booking_confirmation",
        }

    # Asked once, and the guest asked again without answering: repeat the
    # question. Falling through here would send the canned paragraph after a
    # second, explicit request for rooms — the non-answer this branch removes.
    if is_separate_booking_trigger and wants_rooms:
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(clarification_msg),
            "next_action": "awaiting_separate_booking_confirmation",
        }

    # If guest confirmed separate booking creation.
    #
    # Word-boundary, not `w in user_msg_lower`: the old check matched "ha" inside
    # any word carrying it, so "kya aapke paas ha..." answered S57's question on
    # the guest's behalf and started a draft they never asked for. The phrases
    # stay because that is what S57 itself tells the guest to reply with.
    if (
        state.get("separate_booking_clarification_asked")
        and _SEPARATE_BOOKING_YES.search(user_msg_lower)
    ):
        # Transition to new Lead draft alongside existing confirmed booking
        lead_start_msg = (
            f"Theek hai{name_phrase}! Hum naye booking ke liye details note kar rahe hain. "
            f"(Aapki pehli confirmed booking{booking_ref_bold} bilkul safe hai).\n\n"
            f"Naye room ke liye aap kis date par check-in karna chahenge aur kitne guests honge?"
        )
        opened = {
            **state,
            "booking_stage": "lead",
            "lifecycle_stage": "lead",
            "separate_booking_clarification_asked": False,
            "is_separate_booking_inquiry": True,
            "selected_rooms": [],
            "allocation_options": [],
            "price_breakdown": None,
            "total_price": None,
            "bot_response": text_card(lead_start_msg),
            "next_action": "need_dates",
        }
        # The latch is written only when the feature is actually on. With either
        # gate off the draft still opens — that is S57's existing behaviour — but
        # nothing is left in the session for a later gate-on deployment to inherit,
        # which is what stops a stale latch outliving the switch that created it.
        if second_booking_gate_enabled(state):
            # Durable, unlike the flags around it: `context_loader` reads this on
            # every later turn and the resolver holds the confirmed stay aside so
            # this draft can advance. Without it the stage snapped back to
            # `confirmed` on the very next message and the guest could never give
            # a date — measured 2026-10-23 on the bossman's own run.
            opened["separate_booking_active"] = True
            # Bounds the latch above. `context_loader` expires it so a guest who
            # walks away after "yes" gets their own confirmed stay back.
            opened["separate_booking_started_at"] = datetime.now(timezone.utc).isoformat()
        return opened

    # The guest answered S57's question with a no: back to ordinary conversation.
    #
    # Without this the "no" fell through every branch to `is_declined_ack` at
    # the bottom, which read as a polite brush-off ("Bilkul! ... Shubh din!")
    # rather than an answer to the question the bot had just asked. Cleared
    # here so the flag can never outlive the question — the same reason the
    # unconditional clear below exists.
    if (
        state.get("separate_booking_clarification_asked")
        and _SEPARATE_BOOKING_NO.match(user_msg_lower)
    ):
        return {
            **{**state, "separate_booking_clarification_asked": False},
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(
                f"Theek hai{name_phrase}! 🙏 Aapki existing booking{booking_ref_bold} "
                f"jaisi hai waisi rahegi — usme koi badlav nahi hua.\n\n"
                f"Koi aur sawaal ho (directions, check-in time, hotel ki suvidhaayein) "
                f"to bas poochh lijiye. 😊"
            ),
        }

    # S57's question lives for exactly one turn.
    #
    # Reaching here means the guest neither said yes (the branch above) nor asked
    # for rooms again (the repeat branch), so this message is not an answer to the
    # question. Leaving the flag set made it permanent: it is persisted, and the
    # only other place that clears it is the yes branch. A guest who replied "No"
    # kept it True forever, so months later a bare "haan ji" answering something
    # else opened a draft *and* armed the 24-hour second-booking latch — hiding a
    # paid, confirmed stay from the cancel, modify and in-house paths. Clearing it
    # here is what keeps a stale flag from outliving the question that set it.
    if state.get("separate_booking_clarification_asked"):
        state = {**state, "separate_booking_clarification_asked": False}

    # ── S53: "Rate kya hai?" / Price / Tariff Ask ────────────────────────────
    # If guest explicitly asks for total price of N nights or future duration:
    m_nights_ask = re.search(r'\b(\d+)\s*nights?\b|\btotal\s+(?:price|amount|rate)\s+of\s+(\d+)\s*nights?\b', user_msg_lower)
    if m_nights_ask:
        req_n = int(m_nights_ask.group(1) or m_nights_ask.group(2) or 1)
        nightly_rate = 0.0
        if state.get("selected_rooms"):
            nightly_rate = sum(float(r.get("price_per_night") or 0.0) for r in state["selected_rooms"])
        if nightly_rate == 0.0 and active_booking.get("total_price") and active_booking.get("num_nights"):
            nightly_rate = float(active_booking["total_price"]) / max(1, int(active_booking["num_nights"]))
        if nightly_rate == 0.0 and total_amount > 0:
            nightly_rate = total_amount / max(1, int(state.get("num_nights") or 3))
        if nightly_rate > 0.0:
            calc_sub = round((nightly_rate * req_n) / 10) * 10
            tax_sub = round((calc_sub * 0.05) / 10) * 10
            tot_sub = calc_sub + tax_sub
            resp = (
                f"{namaste_greeting} 🙏\n\n"
                f"{req_n} nights ke liye estimated total amount **₹{tot_sub:,.0f}** (incl. taxes) hai.\n"
                f"(Room rate: ₹{nightly_rate:,.0f} x {req_n} nights = ₹{calc_sub:,.0f} + 5% GST ₹{tax_sub:,.0f}).\n\n"
                f"Agar aap dates change ya naya booking karna chahte hain, toh kripya batayein! 😊"
            )
            return {
                **state,
                "booking_stage": "confirmed",
                "lifecycle_stage": "confirmed",
                "bot_response": text_card(resp),
            }

    # Must show what guest ALREADY paid for current booking. ZERO selling, NO room menu!
    is_rate_ask = any(
        p in user_msg_lower for p in (
            "rate kya hai", "rate kya h", "price kya hai", "kitna rate", "room rate", "tariff",
            "price list", "charges kya hai", "how much", "what is the rate", "price", "pricing",
            "total amount", "amount kitna", "kitna amount", "total kitna", "amount kya hai",
            "kitna balance", "balance kitna", "kitna baki", "due amount"
        )
    ) and not any(p in user_msg_lower for p in ("station", "airport", "taxi", "cab", "auto"))

    if is_rate_ask:
        if state.get("crm_confirmed_stay"):
            paid_line = "Payment ka record front desk ke paas hai — woh aapko exact detail bata denge.\n"
            resp = f"{namaste_greeting} 🙏\n\n{paid_line}Hum aapka swagat karne ke liye taiyar hain! 🏨"
        elif amount_paid > 0:
            bal_str = f" Check-in ke time remaining balance: **Rs {balance:,.0f}**." if balance > 0 else " Full payment received (Nil balance)."
            paid_line = f"Is booking ke liye aapne **Rs {amount_paid:,.0f}** pay kiya hai{room_with_dates}.{bal_str}\n"
            resp = (
                f"{namaste_greeting} 🙏\n\n"
                f"{paid_line}"
                f"Aapka stay secure hai. Hum aapka swagat karne ke liye taiyar hain! 🏨"
            )
        else:
            # Guest has NOT paid (amount_paid == 0) — never claim stay is secure or paid Rs 0
            resp = (
                f"{namaste_greeting} 🙏\n\n"
                f"Is booking ka total amount **Rs {total_amount:,.0f}** hai"
                f"{room_with_dates}.\n"
                f"Payment record ke anusaar abhi advance payment received nahi hua hai.\n"
                f"Booking confirm aur secure karne ke liye kripya advance payment complete karein."
            )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S53b: Bill / Invoice / Receipt Request ───────────────────────────────
    is_bill_ask = any(
        w in user_msg_lower for w in ("bill", "invoice", "receipt", "payment slip", "vej dijiye", "bhej dijiye", "hisab")
    ) and not any(w in user_msg_lower for w in ("room chahiye", "booking", "naya room", "rate kya"))
    if is_bill_ask:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Bill / Invoice Request: Booking {booking_id} ({guest_name})",
                description=f"Confirmed Guest {guest_name} ({phone}) requested bill/invoice: '{user_msg}'. Booking: {booking_id}. Stay dates: {check_in} to {check_out}.",
                task_type="FRONT_DESK",
                subtype="BILLING_INQUIRY",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                customer_name=guest_name,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "confirmed",
                    "task_type": "billing_inquiry",
                },
            )
        except Exception as e:
            logger.warning(f"Error creating confirmed bill task: {e}")

        resp = (
            f"{namaste_greeting} Humne aapke bill/receipt ka anurodh front desk team ko bhej diya hai.\n\n"
            f"{booking_id_line}"
            f"• *Stay Dates:* {stay_dates}\n\n"
            f"Humari front desk team jald hi aapse bill share karegi. Main aapki aur kya madad kar sakta hoon? 🙏"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S54: "Meri booking confirm hai?" / Status Ask ────────────────────────
    #
    # Bare "details" was in this list and matched far too much: "what are the
    # details of the evening aarti" came back as the booking slip. The phrasing
    # guests actually use to ask for the booking — "booking details" — is listed
    # in its own right, so dropping the bare word costs nothing and stops the
    # slip being repeated at guests who asked about something else.
    is_confirm_check = any(
        p in user_msg_lower for p in (
            "booking confirm", "confirm hai", "status kya hai", "check booking", "is my booking",
            "meri booking", "slip", "booking details"
        )
    )
    if is_confirm_check:
        slip = _format_confirmation_slip(
            booking_id=booking_id,
            guest_name=guest_name,
            check_in=check_in,
            check_out=check_out,
            room_name=room_name,
            adults=adults,
            amount_paid=amount_paid,
            total_amount=total_amount,
            ident=ident,
            has_payment_record=not state.get("crm_confirmed_stay"),
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(slip),
        }

    # ── S55: "Station se kaise aayein?" / Airport / Directions ───────────────
    # Provide directions, distance, map pin, front desk contact. NO sales pitch!
    is_directions_ask = any(
        p in user_msg_lower for p in (
            "station", "kaise aayein", "kaise pahunchein", "directions", "location", "address",
            "airport", "rasta", "route", "map", "how to reach", "cantt", "banaras"
        )
    )
    if is_directions_ask:
        resp = (
            f"{namaste_greeting} 🏨 {ident.name} pahunchna bohot aasan hai:\n\n"
            f"{location_line}"
            f"{map_line}"
            f"🚆 *City Central Railway Station Se:* (~3.5 km)\n"
            f"• Station ke bahar se direct Auto / E-Rickshaw Central Market Square ke liye milte hain (approx 15-20 mins, ₹20-30 shared ya ₹100-150 private).\n"
            f"• Central Market Square se Main Riverfront Road par paidal chal kar {ident.name} pahunch sakte hain.\n\n"
            f"✈️ *Lal Bahadur Shastri Airport (Babatpur) Se:* (~26 km)\n"
            f"• Airport se pre-paid taxi ya cab (Ola/Uber) lagbhag 45-50 minutes leti hai.\n\n"
            f"🚗 *Parking:* Hotel ke paas vehicle parking suvidha uplabdh hai.\n"
            f"📞 *Front Desk / Navigation Help:* {phone_md}\n\n"
            f"_Happy journey!_"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S56: "Hi" / Greeting ─────────────────────────────────────────────────
    # Personal greeting + booking summary. NO welcome menu, NO continue/fresh start.
    is_greeting = any(user_msg_lower == g for g in ("hi", "hello", "hey", "namaste", "namaskar", "pranam", "good morning", "good evening"))
    if is_greeting:
        if str(raw_booking_id or "").startswith("PAY-") or (amount_paid <= 0 and not state.get("crm_confirmed_stay")):
            link = state.get("payment_link") or active_booking.get("payment_link")
            amt = total_amount or state.get("total_price") or 0.0
            adv = float(active_booking.get("payment_link_amount") or state.get("payment_link_amount") or (amt * 0.3 if amt > 0 else 0.0))
            link_part = f"\n\n🔗 Advance payment link: {link}" if link else ""
            resp = (
                f"{namaste_greeting} 🙏 Welcome to {ident.name}.\n\n"
                f"Aapka reservation{room_disp} {stay_dates} ke liye pending payment hai.\n"
                f"💰 Total Amount: ₹{amt:,.0f}\n"
                f"📌 Advance Required (30%): ₹{adv:,.0f}"
                f"{link_part}\n\n"
                f"Rooms hold karne ke liye kripya advance payment complete karein. Payment hote hi hum confirmation slip bhej denge! 🙏"
            )
            return {
                **state,
                "booking_stage": "payment_pending",
                "lifecycle_stage": "payment_pending",
                "bot_response": text_card(resp),
                "next_action": "payment_pending",
            }

        resp = (
            f"{namaste_greeting} 🙏 Welcome to {ident.name}.\n\n"
            f"Aapki reservation{booking_ref} {stay_dates}{room_disp} ke liye confirmed hai.\n\n"
            f"Aapki aane ki taiyari ya city trip se judi kisi bhi jankari (directions, City Temple visit, Aarti timing) ke liye main aapki kya madad kar sakta hoon?"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── Cancel a confirmed booking ───────────────────────────────────────────
    #
    # Split out of a shared change/cancel branch on 2026-10-10 (product). Until
    # then a guest asking to cancel got the same generic acknowledgement as a
    # guest asking to move a date, filed under a CHECKOUT_REQUEST subtype — so an
    # operator could not tell "wants to leave early" from "wants their money
    # back" — and was told nothing at all about their refund.
    #
    # The booking is NOT cancelled here. Cancelling is a money decision the
    # front desk owns, so this raises the task and hands the guest the policy;
    # nothing in this branch writes to the bookings collection.

    # ── A question ABOUT the policy is not a request to act on it ────────────
    #
    # `_POLICY_QUESTION_RE` above carries the reasoning. Measured live against
    # this clone on 2026-10-11: "what is the cancellation policy" from a
    # confirmed guest raised a BOOKING_CANCELLATION task, so the front desk was
    # asked to approve a cancellation nobody had requested.
    #
    # The answer is the same text the cancel branch quotes, from the same
    # service — minus the task, and minus the "we've sent your request" framing,
    # which would be untrue here.
    if _POLICY_QUESTION_RE.search(user_msg_lower):
        from app.services.cancellation_policy_service import get_cancellation_policy_text

        policy = await get_cancellation_policy_text(hotel_id)
        resp = (
            f"{namaste_greeting} 🙏\n\n"
            f"📋 *Cancellation & Refund Policy*\n{policy}\n\n"
            f"Agar aap cancel karna chahte hain to bas bata dijiye — hum request "
            f"front desk team ko forward kar denge."
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # Amendment is decided first, on this one message, and cancellation defers
    # to it. "change the dates, don't cancel" contains "cancel", so the bare
    # substring test took it to the cancellation branch and raised a task asking
    # the front desk to approve the very thing the guest had ruled out.
    #
    # Deciding both here rather than reordering the two blocks keeps the
    # precedence in one place — the amendment block below reads this variable
    # instead of recomputing it, so the two cannot drift.
    # The model's own classification joins the regexes here rather than
    # replacing them. Measured 2026-10-25: `understand` labels "can I check in a
    # day earlier" MODIFY_BOOKING and "can I change my room" PAYMENT_STATUS, so
    # neither source is sufficient alone — but either one saying "amendment" is
    # a signal strong enough to raise the ticket, and the ticket is the thing
    # this branch exists to produce. `_POLICY_QUESTION_RE` above has already
    # taken the questions *about* this, so a MODIFY/CANCEL intent reaching here
    # is a request to act.
    intent_now = str(state.get("intent") or "")

    is_amendment = (
        intent_now == "MODIFY_BOOKING"
        or bool(_CHANGE_DATE_RE.search(user_msg_lower))
        or bool(_AMEND_OTHER_RE.search(user_msg_lower))
        or any(p in user_msg_lower for p in _AMENDMENT_CUES)
    )

    is_cancel = (
        (
            intent_now == "CANCEL_BOOKING"
            or any(p in user_msg_lower for p in ("cancel", "cancellation", "refund"))
            or bool(_CANCEL_OTHER_RE.search(user_msg_lower))
        )
        and not _CANCEL_NEGATED_RE.search(user_msg_lower)
        and not is_amendment
    )
    if is_cancel:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task

            await create_stage_task(
                hotel_id=hotel_id,
                title=f"Cancellation Request for Confirmed Booking {booking_id}",
                description=(
                    f"Guest {guest_name} ({phone}) requested cancellation: '{user_msg}'. "
                    f"Stay dates: {check_in} to {check_out}. Booking is still CONFIRMED — "
                    f"cancellation and any refund are for the front desk to approve."
                ),
                task_type="FRONT_DESK",
                subtype="BOOKING_CANCELLATION",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "confirmed",
                    "task_type": "cancellation",
                    "action_required": "approve_or_decline_cancellation",
                    "check_in": str(check_in),
                    "check_out": str(check_out),
                    "room_name": room_name,
                },
            )
        except Exception as e:
            logger.warning(f"Error creating cancellation task: {e}")

        # The policy is the whole point of this reply, so it is fetched and never
        # replaced by a generic line: the service returns a hotel's own bands, or
        # the default wording, and only ever an empty string would be a problem —
        # which it cannot be.
        from app.services.cancellation_policy_service import get_cancellation_policy_text

        policy = await get_cancellation_policy_text(hotel_id)

        resp = (
            f"Humne aapka cancellation request front desk team ko bhej diya hai.\n\n"
            f"{booking_id_line}"
            f"• *Dates:* {stay_dates}\n\n"
            f"📋 *Cancellation & Refund Policy*\n{policy}\n\n"
            f"Team aapke request ko verify karke refund process aage badhayegi. "
            f"Kisi urgent sahayata ke liye aap reception par call kar sakte hain: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── Amend a confirmed booking (dates, room, early/late, etc.) ────────────
    #
    # Also split out on 2026-10-10. The amendment itself is not applied here —
    # repricing and re-reserving a confirmed booking is the front desk's call —
    # but the request is now *recorded* on the task in structured form, so an
    # operator sees what the guest asked for against the dates currently held,
    # rather than a free-text sentence they have to parse.
    #
    # `is_amendment` is decided up at the cancellation block, which defers to
    # it. Recomputed here, the two answers could drift apart and a message could
    # be both or neither.
    if is_amendment:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task

            await create_stage_task(
                hotel_id=hotel_id,
                title=f"Amendment Request for Confirmed Booking {booking_id}",
                description=(
                    f"Guest {guest_name} ({phone}) requested a change: '{user_msg}'. "
                    f"Currently held: {check_in} to {check_out}{room_disp}. "
                    f"Booking is CONFIRMED — the change is for the front desk to "
                    f"reprice and confirm."
                ),
                task_type="FRONT_DESK",
                subtype="BOOKING_AMENDMENT",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "confirmed",
                    "task_type": "amendment",
                    "action_required": "reprice_and_confirm_change",
                    # The dates being changed *from*. Without these the task is a
                    # sentence and the operator has to open the booking to know
                    # what is actually being moved.
                    "current_check_in": str(check_in),
                    "current_check_out": str(check_out),
                    "room_name": room_name,
                },
            )
        except Exception as e:
            logger.warning(f"Error creating amendment task: {e}")

        resp = (
            f"Humne aapka change request front desk team ko note kar diya hai.\n\n"
            f"{booking_id_line}"
            f"• *Current Dates:* {stay_dates}\n"
            f"• *Request:* {user_msg}\n\n"
            f"Team availability aur rate check karke aapko naye dates aur amount ke saath "
            f"confirm karegi. Kisi urgent sahayata ke liye aap reception par call kar sakte hain: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── Sightseeing / Temple / Aarti in Advance ───────────────────────────────
    if any(p in user_msg_lower for p in ("aarti", "ganga", "darshan", "vishwanath", "temple", "ghat", "kashi")):
        resp = (
            f"{ident.name} se City Temple aur Riverfront Plaza bohot kareeb hain:\n\n"
            f"🛕 *Shri City Temple Temple:* Gate No. 4 sirf paidal doori par hai.\n"
            f"🌊 *Ganga Aarti:* Riverfront Plaza par daily sham 6:45 PM – 7:00 PM par shuru hoti hai ({ident.name} se sirf 5-10 minute walk).\n"
            f"🚤 *Boat Rides:* Morning sunrise boat tour aur evening aarti boat tour reception se arrange ho sakte hain.\n\n"
            f"Kisi aur jankari ke liye kripya batayein!"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S56b: Guest Acknowledgment / Polite Decline ───────────────────────────
    is_declined_ack = any(
        user_msg_lower == p or user_msg_lower.startswith(p + " ") or user_msg_lower.endswith(" " + p)
        for p in ("no thanks", "no thank you", "nahi chahiye", "not needed", "nope", "nahi", "no")
    )
    if is_declined_ack:
        resp = (
            f"Bilkul{name_phrase}! Swagat ki poori taiyari hai. "
            f"Kisi bhi sahayata (directions, Kashi darshan, aarti timing) ke liye hum yahan hain. Shubh din! 🙏"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    is_thanks_ack = any(p in user_msg_lower for p in ("thank you", "thanks", "dhanyawad", "shukriya"))
    if is_thanks_ack:
        resp = (
            f"You're most welcome{name_phrase}! Hum {ident.name} mein aapka swagat karne ke liye utsuk hain. "
            f"Kisi aur jankari ke liye batayein. Shubh yatra! 🏨"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # Anchored at the start of the message — see `_OK_ACK_RE`. The substring
    # form this replaced matched "ok" inside "booking", so a guest asking to
    # add a night was told their arrival was all set.
    is_ok_ack = bool(_OK_ACK_RE.match(user_msg_lower))
    if is_ok_ack:
        resp = (
            f"Ji{name_phrase}! Hum aapke swagat ke liye taiyar hain. "
            f"Front desk se sampark ke liye number hai: {phone_md}. Shubh din! 🙏"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S58: Everything Else — Answer From the Hotel's Knowledge Base ─────────
    #
    # This tail used to be a fixed paragraph naming directions, check-in time,
    # parking and Kashi darshan — so "what is the wifi password", "do you allow
    # pets" and "is there a restaurant" all got the same non-answer, and the
    # hotel's own knowledge base (14 articles plus structured transit details for
    # Sahu) was never read for a confirmed guest at all.
    #
    # The cause is routing, not this file: `routing.py:40` sends every
    # `lifecycle_stage == "confirmed"` turn here and returns, so the
    # `intent == "FAQ"` branch at `routing.py:99` is unreachable for a confirmed
    # guest. Answering means calling the FAQ node from here.
    #
    # Three things this call must NOT be allowed to do, all of them reverting to
    # the canned paragraph below instead:
    #
    #   * Move the guest out of the confirmed stage. `handle_faq` spreads the
    #     incoming state, so it preserves the stage today — but that is a
    #     property of a different module, and re-asserting it here is what keeps
    #     a confirmed guest confirmed.
    #   * Sell. `handle_faq`'s first branch redirects a room-pricing question to
    #     `handle_room_query`, which shows option cards. That is the right answer
    #     in the booking flow and a contradiction of this stage's ZERO SELLING
    #     contract. `next_action == "faq_answered"` is how an answer is
    #     distinguished from a redirect, and it is the only outcome accepted.
    #   * Raise. `handle_faq` reaches Gemini, ChromaDB, Mongo and the persona
    #     rewriter; a knowledge-base failure must cost the guest a fallback
    #     sentence, never a turn with no reply.
    # `knowledge_base_answer` owns all three of those, and its docstring carries
    # the reasoning. It lives in `kb_answer.py` rather than here because
    # `stage_in_house` now needs the identical call — Product, 2026-10-11: *"faq
    # mode till check out"* — and two copies would drift.
    #
    # It also guards a leak this block's own `faq_answered` check did not, and
    # that only showed up once the bot started answering at all: "show me
    # available rooms" came back as a room catalogue with rate plans. The
    # hotel's room list and rate plans are *in* the knowledge base, so it
    # arrives as a plain `faq_answered` and the check above waved it through.
    # Measured 2026-10-11; at HEAD the same message got the canned paragraph
    # below.
    #
    # Everything it declines comes back as `None` and lands on that paragraph —
    # which is why the paragraph is still here.
    kb_state = await knowledge_base_answer(
        state, stage="confirmed", log_tag="STAGE_2_CONFIRMED",
    )
    if kb_state is not None:
        return kb_state

    # ── Default Confirmed Guidance (Reassuring, NO Selling) ───────────────────
    resp = (
        f"{namaste_greeting} 🙏\n\n"
        f"Directions, check-in time (12:00 PM), parking, ya Kashi darshan se judi kisi bhi sahayata ke liye hum yahan hain. "
        f"Emergency ya direct baat karne ke liye front desk number hai: {phone_md}."
    )
    return {
        **state,
        "booking_stage": "confirmed",
        "lifecycle_stage": "confirmed",
        "bot_response": text_card(resp),
    }
