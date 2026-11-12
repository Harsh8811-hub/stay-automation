"""
Deterministic intent overrides for common misclassifications.

Also:
- Mid-booking escape hatch for policy side-questions
- Don't trap users in an in-progress quote/booking when they ask
  something new (rates catalog, new-date availability, greeting)
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from app.utils.ac_text import ac_preference_from_prompt_reply
from app.utils.booking_resume import (
    clear_in_progress_booking_fields,
    has_in_progress_draft,
    is_mid_booking_side_question,
    looks_like_booking_slot_answer,
    needs_start_fresh_confirmation,
    wants_to_abandon_in_progress_booking,
)

#: A month name, short or long. Written out rather than `(?:jan|...|dec)[a-z]*`
#: because the loose form matches "decide", "mayor" and "march past the desk",
#: and a false positive here routes a photo-only turn into the booking flow.
#:
#: It also has to match the LONG form, which a bare short form cannot do once a
#: trailing `\b` follows it: `dec\b` fails on "December" (no boundary between c
#: and e), so "12th December to 15 December" read as *not* a date search, the
#: photo branch returned SHOW_AMENITIES early, and every date and guest count in
#: the message was dropped. Measured 2026-10-29 on a real WhatsApp thread: the
#: guest sent photos + "12th December to 15 December" + "6 adult 2 child", and
#: the bot replied asking for check-in and check-out dates.
_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

#: The intents `understand` treats as booking intents — the ones that may write
#: dates, guests and rooms into the state. A deliberate copy of the set at
#: understand.py:1520 rather than an import, because importing the node into
#: this module would pull LangGraph into every caller of `apply_intent_overrides`.
#: If the two ever disagree the symptom is benign: this one is only read to
#: decide whether an override is needed at all.
_BOOKING_INTENTS = frozenset({
    "BOOK_ROOM", "MODIFY_BOOKING", "SELECT_ROOM", "CONFIRM_BOOKING",
    "CHECK_AVAILABILITY", "CANCEL_BOOKING", "PAYMENT_STATUS", "PRICE_MATCH",
    "BOOKING_SUMMARY", "BOOKING_DATES_PROVIDED", "BOOKING_AMENDMENT",
})

#: The guest is asking *about* a paid early check-in / late check-out, not
#: asking to book one. Every alternative names the direction — a plain
#: "what time is check-in?" is a different question with a different answer,
#: and must not be caught here.
_EARLY_LATE_ASK = re.compile(
    r"\b(?:"
    r"early\s*(?:check[\s-]?in|checkin)"
    r"|check[\s-]?in\s*early"
    r"|late\s*(?:check[\s-]?out|checkout)"
    r"|check[\s-]?out\s*late"
    r"|jaldi\s*(?:check|aa|pahunch)"
    r"|subah\s*(?:aana|aa\s*jau|pahunch|check)"
    r"|der\s*se\s*(?:check|nikal|jaa)"
    r"|late\s*nikal"
    r")\b",
    re.IGNORECASE,
)

_PRICING_PATTERN = re.compile(
    r"\b(price|prices|pricing|cost|costs|total|how much|rate|rates|amount|charge|fare)\b"
    r"|\b\d+\s*nights?\b"
    r"|\bhow many nights\b"
    r"|\bper night\b",
    re.IGNORECASE,
)

# Catalog / menu price list — NOT the current booking quote (includes Hinglish)
_GENERAL_RATES = re.compile(
    r"\b("
    r"room\s+rates?|room\s+prices?|your\s+(room\s+)?(rates?|prices?)"
    r"|price\s+list|rate\s+list|all\s+(room\s+)?(prices?|rates?)"
    r"|show\s+(me\s+)?(the\s+)?(prices?|rates?|pricing)"
    r"|what\s+(are|is)\s+(your\s+)?(room\s+)?(prices?|rates?|pricing)"
    r"|how\s+much\s+(do\s+)?(your\s+)?rooms?\s+cost"
    r"|prices?\s+&\s+availability|room\s+prices?\s+&\s+availability"
    r"|room\s+(ka\s+)?(rate|rates|price|kiraya|charges?)"
    r"|rate\s+kya\s+(chal\s+raha\s+)?hai|kitna\s+(rate|price|kiraya)\s+hai"
    r"|price\s+kya\s+hai|rate\s+card|rates?\s+list|price\s+list"
    r"|best\s+(?:direct\s+)?rates?|best\s+prices?|offers?\s+(?:&|and)?\s+availability"
    r"|share\s+offers|room\s+rent|room\s+tariff|tariff\s+chart|rent\s+chart"
    r")\b"
    r"|\b(?:bhaiya|sir|boss)\s+.*(?:rate|price|kiraya|charges)\b",
    re.IGNORECASE,
)

_EXPLICIT_COMPETITOR_PRICE_MATCH = re.compile(
    r"\b("
    r"cheaper\s+online|lower\s+(?:price|rate)\s+online|getting\s+cheaper"
    r"|makemytrip|mmt|agoda|goibibo|booking\.com|expedia|tripadvisor|yatra|easemytrip|cleartrip"
    r"|online\s+(?:sasta|kam|cheaper|kamti|portal|app)|online\s+dekha|competitor"
    r"|match\s+(?:the\s+)?price|price\s+match"
    r")\b",
    re.IGNORECASE,
)

_BANK_DETAILS_PATTERN = re.compile(
    r"\b("
    r"bank\s+details?|bank\s+account|account\s+(?:no\.?|number)|ifsc(?:\s+code)?"
    r"|bank\s+transfer|neft|rtgs|imps|khata\s+(?:no\.?|number)"
    r"|share\s+bank|hotel\s+bank|send\s+bank|account\s+details?"
    r"|bank\s+se\s+payment|bank\s+transfer\s+karna\s+hai|bank\s+me\s+transfer"
    r"|bank\s+information|hotel\s+account|hotel\s+bank\s+details?"
    r")\b",
    re.IGNORECASE,
)

_CURRENT_BOOKING_PRICE = re.compile(
    r"\b("
    r"my\s+(total|price|quote|booking)|what\s+(is|was)\s+my\s+(total|price|quote)"
    r"|what\s+did\s+i\s+select|booking\s+summary|current\s+quote"
    r")\b",
    re.IGNORECASE,
)

# "What was my last booking?" / check existing — not current quote summary
_EXISTING_OR_LAST_BOOKING = re.compile(
    r"\b("
    r"last\s+booking|previous\s+booking|past\s+booking|earlier\s+booking"
    r"|my\s+(existing|old|confirmed)\s+booking"
    r"|check\s+(my\s+)?(existing\s+)?booking"
    r"|do\s+i\s+have\s+a\s+booking|any\s+(existing\s+)?booking"
    r"|show\s+(me\s+)?my\s+booking|booking\s+history"
    r"|what\s+(was|is)\s+my\s+last\s+booking"
    r")\b",
    re.IGNORECASE,
)

_MENU_OPTION_2 = re.compile(r"^\s*(2|option\s*2)\s*$", re.IGNORECASE)

# In-stay operational requests — keep out of booking/FAQ loops
_HOTEL_TASK_HINT = re.compile(
    r"\b("
    r"ac\s+(is\s+)?(not\s+working|broken|off)|"
    r"(not\s+working|broken).{0,20}\bac\b|"
    r"extra\s+(pillows?|towels?|blankets?|water\s+bottle|water)|"
    r"water\s+bottle|pani\s+ki\s+bottle|pani\s+bhej\s+do|"
    r"towel\s+bhej\s+do|bottle\s+aur\s+towel|towel\s+aur\s+water|"
    r"room\s+service|order\s+.+\s+to\s+(the\s+|my\s+)?room|"
    r"butter\s+chicken|send\s+.+\s+to\s+(my\s+)?room|"
    r"wake[- ]?up\s+call|hot\s+water|tv\s+(is\s+)?(not\s+working|broken)|"
    r"wifi\s+(is\s+)?(slow|not\s+working|down)|"
    r"need\s+(extra\s+)?(towels?|pillows?|toiletries|water)|"
    r"room\s+clean(ing)?|housekeeping|"
    r"(?:bill|invoice|parchi|receipt|hisab)\s*(?:vej|bhej|send|de|do|dijiye|bhejiye|kijiye|karo)|"
    r"(?:vej|bhej|send)\s*(?:dijiye|do)?\s*(?:bill|invoice|parchi|receipt|hisab)"
    r")\b"
    r"|\broom\s+\d+\s+(?:me|mein)\b",
    re.IGNORECASE,
)

_SUGGESTION_REQUEST = re.compile(
    r"\b("
    r"suggest(\s+me)?(\s+room)?|recommend(\s+me)?(\s+room)?|what\s+(do\s+you\s+)?suggest"
    r"|what\s+(do\s+you\s+)?recommend|show\s+(me\s+)?options|room\s+options|show\s+(me\s+)?rooms"
    r"|any\s+suggestions?|give\s+(me\s+)?suggestions?"
    r")\b",
    re.IGNORECASE,
)


def _infer_task_extras(user_message: str) -> dict[str, Any]:
    """Best-effort task fields when Gemini misroutes but phrasing is clear."""
    msg = (user_message or "").lower()
    extras: dict[str, Any] = {
        "task_is_request": True,
        "task_urgency": "NORMAL",
        "task_confidence": 0.85,
    }
    if re.search(r"\bac\b", msg) and re.search(r"not\s+working|broken|off", msg):
        extras.update({
            "task_type": "MAINTENANCE",
            "task_subtype": "AC_REPAIR",
            "task_summary": "Guest reported AC not working.",
            "task_urgency": "HIGH",
        })
    elif re.search(r"room\s+service|order\s+|butter\s+chicken|naan|food", msg):
        extras.update({
            "task_type": "ROOM_SERVICE",
            "task_subtype": "FOOD_DELIVERY",
            "task_summary": "Guest requested room service / food order.",
        })
    elif re.search(r"pillow", msg):
        extras.update({
            "task_type": "HOUSEKEEPING",
            "task_subtype": "EXTRA_PILLOWS",
            "task_summary": "Guest requested extra pillows.",
        })
    elif re.search(r"towel", msg):
        extras.update({
            "task_type": "HOUSEKEEPING",
            "task_subtype": "EXTRA_TOWELS",
            "task_summary": "Guest requested extra towels.",
        })
    elif re.search(r"\b(?:bill|invoice|receipt|hisab|parchi)\b", msg):
        extras.update({
            "task_type": "FRONT_DESK",
            "task_subtype": "BILLING_INQUIRY",
            "task_summary": "Guest requested bill / invoice.",
        })
    elif re.search(r"(?:paid|payment|money\s+sent|transferred).*(?:brother|bhai|friend|other|another|different|account|neft|rtgs|imps|utr|cash|counter)", msg) or re.search(r"(?:neft|rtgs|imps|utr\s*[:#-]?\s*[0-9a-zA-Z]+|paid\s+cash)", msg):
        extras.update({
            "task_type": "FRONT_DESK",
            "task_subtype": "PAYMENT_CONFIRMATION",
            "task_summary": "Guest requested payment confirmation via other party / other modes.",
            "task_urgency": "HIGH",
        })
    else:
        extras.update({
            "task_type": "OTHER",
            "task_subtype": "OTHER",
            "task_summary": "Guest service request.",
        })
    return extras

_NIGHTS_PATTERN = re.compile(r"\b(\d+)\s*nights?\b", re.IGNORECASE)

_ROOM_COUNT_PATTERNS = (
    re.compile(
        r"\b(\d+)\s+(?:(?:deluxe|standard|premier|family|suite|triple|trippple|non\s*ac|ac)\s+)+rooms?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d+)\s+rooms?\b", re.IGNORECASE),
)

_BOOKING_VERB = re.compile(r"\b(book|booking|reserve|wanna|want|need)\b", re.IGNORECASE)

_ROOM_TYPE_KEYWORDS = (
    "deluxe", "standard", "premier", "family", "suite", "triple", "trippple", "ac", "non ac",
)

_AVAILABILITY_QUESTION = re.compile(
    r"\b(available|availability|weekend|weekends|saturday|sunday|open|vacancy|vacant)\b",
    re.IGNORECASE,
)

_RELATIVE_DATE_EXPR = re.compile(
    r"\b(this\s+weekend|next\s+weekend|coming\s+weekend"
    r"|this\s+saturday|next\s+saturday|coming\s+saturday"
    r"|this\s+sunday|next\s+sunday|coming\s+sunday"
    r"|this\s+friday|next\s+friday|coming\s+friday"
    r"|tomorrow|day\s+after\s+tomorrow"
    r"|tonight)\b",
    re.IGNORECASE,
)

# A goodbye is not a commitment. `_CONFIRM_PHRASE` has to admit the soft words
# a real confirmation uses ("ok", "haan", "theek hai"), and those are exactly
# the words a guest also uses while saying thank you and leaving. The model
# tells the two apart in `intent`, so when it says THANK_YOU and the message
# carries gratitude, the soft confirmation words are not evidence of anything.
_THANKS_WORD = re.compile(
    r"\b(thank\s*you|thanks|thanku|dhanyavad|dhanyawad|dhanyavaad|shukriya|thx)\b",
    re.IGNORECASE,
)

_CONFIRM_PHRASE = re.compile(
    r"\b("
    r"confirm(\s+booking)?"
    r"|yes\s*,?\s*confirm"
    r"|proceed(\s+to\s+pay)?"
    r"|pay\s+now"
    r"|i\s+confirm"
    r"|yes|ok|okay|sure|haan|han|theek(\s+hai|\s+h)?|thik(\s+hai|\s+h)?|done|pakka|go\s+ahead"
    r"|book\s+(?:this|it|now|room|kardo|karo|kijiye)"
    r"|please\s+book"
    r"|reserve\s+(?:this|it|now|room)"
    r"|lock\s+(?:this|it|room)"
    r"|finalize"
    r")\b",
    re.IGNORECASE,
)

# A message that is asking to negotiate or to change something — never a
# confirmation, however it ends. Shared by `routing.route_by_intent` (which
# decides payment-vs-booking) and `validator` (which decides quote-vs-payment),
# because the two drifted apart once already.
#
# This was a plain `any(w in message for w in (...))` substring test, and every
# short keyword in it fired inside an unrelated word: "kam" inside "kamra"
# (room), "less" inside "unless", "change" inside "exchange", "stop" inside
# "stop by the hotel at 5". So "book the kamra with two beds" and "unless it
# rains, confirm it" both routed away from payment. Word boundaries fix it;
# `non[\s-]?ac` is spelled out because "non ac" needs to match "non-ac" too.
_NEGOTIATION_OR_CHANGE = re.compile(
    r"\b(?:"
    r"kam|discount|offer|coupon|sasta|reduce|less|non[\s-]?ac|change|modify|badlo"
    r"|stop|cancel|actually\s+show"
    r")\b"
    r"|@\s*\d{3,6}\b"
    r"|\b(?:confirm\s+)?(?:at|@)\s*(?:rs\.?|inr|₹)?\s*\d{3,6}\b"
    r"|\b(?:in|for|pe|par)\s*(?:rs\.?|inr|₹)?\s*\d{3,6}\b"
    r"|\b\d{3,6}\s*(?:me|mein|pe|par)\s*(?:kardo|kar\s*do|confirm|chalega|dena|de\s*do|final)\b",
    re.IGNORECASE,
)


def resolve_negotiation_flag(state: Any, user_message: str) -> bool:
    """
    Whether this turn is a request to change or negotiate, rather than to confirm.

    The single definition, shared by `routing.route_by_intent` and `validator` —
    the two drifted apart once already (see the note above `_NEGOTIATION_OR_CHANGE`),
    and this is the second time the rule has needed to change in two places.

    Precedence:

      LLM says CHANGE   -> True,  even when the regex found nothing
      LLM says CONFIRM  -> False, even when the regex matched
      UNCLEAR / None    -> the regex decides (today's behaviour exactly)

    The model only reaches the first two when it returned one of its three
    enum values; a timeout, a malformed reply, an unset `PAYMENT_TURN_LLM_VERDICT`
    or a prompt-injected message all land on UNCLEAR and leave the regex in
    charge. So this can sharpen the rule but cannot disable it.

    Every disagreement is logged: the pairs where the two differ are exactly the
    evidence needed to decide whether the blocklist can be retired.
    """
    verdict = (state or {}).get("payment_turn_verdict")
    regex_hit = bool(_NEGOTIATION_OR_CHANGE.search(user_message or ""))

    if verdict == "CHANGE":
        if not regex_hit:
            logger.info(
                "PAYMENT_TURN_DISAGREEMENT | llm=CHANGE regex=no-match | msg=%r",
                (user_message or "")[:80],
            )
        return True
    if verdict == "CONFIRM":
        if regex_hit:
            logger.info(
                "PAYMENT_TURN_DISAGREEMENT | llm=CONFIRM regex=match | msg=%r",
                (user_message or "")[:80],
            )
        return False
    return regex_hit


_ESCAPE_FROM_INTENTS = frozenset({
    "BOOK_ROOM",
    "MODIFY_BOOKING",
    "CHECK_AVAILABILITY",
    "UNKNOWN",
    "SELECT_ROOM",
})

_ESCAPE_STAGES = frozenset({
    "collecting",
    "rooms_shown",
    "quote_shown",
    "awaiting_payment",
    "new",
})


def extract_nights_from_message(user_message: str) -> int | None:
    match = _NIGHTS_PATTERN.search(user_message)
    if match:
        return int(match.group(1))
    return None


def extract_num_rooms_from_message(user_message: str) -> int | None:
    """Extract requested room count — never confuse '2 nights' or child ages with 'room count'."""
    for pattern in _ROOM_COUNT_PATTERNS:
        match = pattern.search(user_message)
        if match:
            val = int(match.group(1))
            # Guard against child ages being matched (e.g. "age 4 and 9 ac room")
            if re.search(rf"\b(?:age|aged|saal|umar)\s+.*?\b{val}\b", user_message, re.IGNORECASE):
                continue
            # Guard against check-in/out date digits (e.g. "5 to 7 sep")
            if re.search(rf"\b{val}\s*(?:to|-|se|\.|\/)\b", user_message, re.IGNORECASE):
                continue
            return val
    return None


def extract_room_type_from_message(user_message: str) -> str | None:
    lower = user_message.lower()
    for keyword in _ROOM_TYPE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lower):
            return keyword
    return None


def get_room_pricing_query(
    user_message: str,
    state: dict[str, Any],
) -> tuple[str | None, int | None, int | None]:
    """Return (room_type, nights, num_rooms) for a room-type pricing/booking quantity query."""
    # Generic catalog rate questions are not structured pricing queries
    if _GENERAL_RATES.search(user_message) and not extract_room_type_from_message(user_message):
        return None, None, None

    room_type = (
        state.get("pricing_query_room_type")
        or extract_room_type_from_message(user_message)
        or state.get("room_type_preference")
    )
    nights = state.get("pricing_query_nights") or extract_nights_from_message(user_message)
    num_rooms = (
        state.get("pricing_query_num_rooms")
        or state.get("num_rooms")
        or extract_num_rooms_from_message(user_message)
    )

    has_pricing = bool(_PRICING_PATTERN.search(user_message))
    has_booking = bool(_BOOKING_VERB.search(user_message))
    if not room_type:
        return None, None, None
    if not nights and not has_pricing and not has_booking:
        return None, None, None
    return room_type, nights or 1, num_rooms


def _apply_pricing_extras(
    extras: dict[str, Any],
    user_message: str,
    state: dict[str, Any],
    room_type: str,
    nights: int | None,
) -> dict[str, Any]:
    num_rooms = (
        extract_num_rooms_from_message(user_message)
        or state.get("num_rooms")
    )
    extras["pricing_query_room_type"] = room_type
    extras["pricing_query_nights"] = nights or 1
    if num_rooms:
        extras["pricing_query_num_rooms"] = num_rooms
        extras["num_rooms"] = num_rooms
    pref = extract_room_type_from_message(user_message) or state.get("room_type_preference") or room_type
    extras["room_type_preference"] = pref
    return extras


def _force_new_search_extras(extras: dict[str, Any]) -> dict[str, Any]:
    """Clear quote/payment trap so availability searches run fresh."""
    extras["force_new_search"] = True
    extras["selected_rooms"] = []
    extras["allocation_options"] = []
    extras["price_breakdown"] = None
    extras["total_price"] = None
    extras["payment_link"] = None
    extras["payment_link_id"] = None
    extras["payment_link_amount"] = None
    extras["payment_status"] = None
    extras["booking_stage"] = "collecting"
    return extras


#: Whole-message matches for "throw this conversation away and start again".
#: Deliberately an exact-match set and not a regex: the regex form in
#: `booking_resume._START_FRESH_PHRASES` is unanchored and matched inside
#: ordinary sentences, which is a different, wider contract.
#
# "new booking" / "nayi booking" / "naya booking" are deliberately NOT in here,
# although this set was extracted from a tuple that had them. They do not mean
# "throw this conversation away": they mean "add a stay on top of the one I
# already have", and `understand` has a dedicated, tested branch for exactly
# that (S8 branch 1 — it files the current booking into `completed_bookings`
# and answers `booking_stage="collecting"`, `next_action="need_dates"`). While
# they counted as a start-fresh, that branch was unreachable: the earlier
# fast path returned the greeting first. `test_a_new_booking_still_starts` and
# `test_followup_new_booking_triggers_date_prompt` both caught it.
_EXPLICIT_START_FRESH: frozenset[str] = frozenset({
    "fresh start", "start fresh", "restart", "start new", "reset",
    # The one command guests actually type. WhatsApp has no command bar, so
    # "/start" is what people carry over from Telegram — measured on prod
    # 2026-10-22, phone 9958263793 sent "/start", the LLM labelled it GREETING,
    # and the turn went back into the checked-out feedback stage. A bare "start"
    # is included because it is the same intent and the match is whole-message,
    # so no ordinary sentence can reach it.
    "start",
})


def is_explicit_start_fresh(user_message: str) -> bool:
    """Whether the whole message is an explicit "start fresh" command.

    The leading "/" is how these get typed as commands — "/start fresh" — and an
    exact-string match on the raw text missed every one of them: the message fell
    through to `needs_start_fresh_confirmation`, whose unanchored regex *did*
    match, so the guest got "shall I start fresh?" instead of a reset. Stripping
    the command prefix is what makes the two agree.

    Exposed rather than inlined because `understand` has to ask the same question
    *before* its confirmed-stay branch, which returns without ever reaching
    `apply_intent_overrides`.
    """
    return (user_message or "").strip().lower().lstrip("/").strip() in _EXPLICIT_START_FRESH


#: Words that appear on every room name and in half of ordinary sentences.
#: The noun itself is ignored; only the distinctive part of a name counts.
_CARD_NAME_STOPWORDS = frozenset({
    "room", "rooms", "bed", "beds", "the", "with", "want", "need",
    "please", "book", "booking", "night", "nights", "adult", "adults",
    "guest", "guests", "stay", "date", "dates", "have", "this", "that",
})


#: A sentence about a room is not automatically a pick. These are the words
#: that turn one into something a person has to answer; when any is present the
#: message keeps the label it came with.
_COMPLAINT_CUE = re.compile(
    r"\b(dirty|broken|not\s+working|no\s+water|no\s+power|leak|leaking|smell|"
    r"complaint|problem|issue|worst|rude|cheat|refund)\b",
    re.IGNORECASE,
)


def _names_a_card_room(message: str, options: list[dict[str, Any]]) -> bool:
    """Does the message name a room that is on the card right now?

    Spelling is not the guest's problem: "standerd" and "delux" are how these
    rooms get typed, so a near-miss counts. The card's own names are the
    vocabulary — a word the card does not use returns False, which keeps
    "your deluxe room is dirty" out of the selection path.
    """
    card_words: set[str] = set()
    for option in options:
        for room in (option.get("rooms") or []):
            card_words.update(
                re.findall(r"[a-z]{4,}", str(room.get("room_name") or "").lower())
            )
    if not card_words:
        return False

    for word in re.findall(r"[a-z]{4,}", (message or "").lower()):
        if word in _CARD_NAME_STOPWORDS:
            continue
        if word in card_words or difflib.get_close_matches(
            word, card_words, n=1, cutoff=0.84
        ):
            return True
    return False


def apply_intent_overrides(
    user_message: str,
    intent: str,
    state: dict[str, Any],
    *,
    reset_booking: bool = False,
) -> tuple[str, str | None, dict[str, Any]]:
    """
    Return (intent, faq_question, state_extras).

    `reset_booking` comes from Gemini (semantic start-fresh). Short "No"
    after a continue prompt is still handled deterministically.
    """
    stage = state.get("booking_stage", "new")
    # The prompt pointer. `load_context` resets `next_action` to None at the start of
    # every turn, so every guard below that read it directly was dead in production —
    # including the change-menu reader, which is why a row tap meant nothing here.
    # `last_prompt_action` is the half that survives, and this is the same
    # next_action-or-last_prompt_action fallback `understand.get_active_prompt` uses.
    active_prompt_action = state.get("next_action") or state.get("last_prompt_action")
    expected_action = active_prompt_action
    has_dates = bool(state.get("check_in") and state.get("check_out"))
    has_options = bool(state.get("allocation_options"))
    extras: dict[str, Any] = {}

    msg_room_type = extract_room_type_from_message(user_message)
    nights = extract_nights_from_message(user_message)
    num_rooms = extract_num_rooms_from_message(user_message)
    has_booking = bool(_BOOKING_VERB.search(user_message))
    has_relative_dates = bool(_RELATIVE_DATE_EXPR.search(user_message))
    has_availability = bool(_AVAILABILITY_QUESTION.search(user_message))

    # ── Paid early check-in / late check-out ──
    #
    # The hotels sell this and the Control Centre prices it, but the fee is
    # priced off a clock the guest states, and the guest may state it at any
    # point: before the quote, on the same line as "early check in?", or on the
    # turn after the bot asks them for it. Resolving one of those turns at a
    # time is how the first attempt at this shipped a rule that only covered the
    # guest who said everything in one breath — so every way the clock can
    # arrive is resolved here, together.
    #
    # **This sits above the HOTEL_TASK bail, and has to.** The classifier really
    # does return HOTEL_TASK for "early check in at 3 am" and "i want early
    # check in, make sure" — both appear in live transcripts — and that bail,
    # now the block below this one, used to run first and hand every one of them
    # to the front desk with no price. It still guards everything else; only a
    # guest who is genuinely at the hotel is still left to it, because there the
    # front desk really is the one who decides.
    #
    # `awaiting_early_late_time` is cleared on every turn and re-set only where
    # the question is actually asked, so a stale marker can never turn an
    # unrelated sentence into a re-quote.
    extras["awaiting_early_late_time"] = False

    from app.utils.guest_time import extract_requested_times, is_only_a_clock

    _stated = extract_requested_times(user_message)
    _states_a_clock = bool(_stated.get("arrival") or _stated.get("departure"))
    _mentions = not looks_like_booking_slot_answer(user_message) and bool(
        _EARLY_LATE_ASK.search(user_message)
    )
    _bare_clock = is_only_a_clock(user_message)
    _clock_on_record = bool(
        state.get("requested_arrival_time") or state.get("requested_departure_time")
    )
    _quote_on_screen = bool(state.get("selected_rooms") or state.get("total_price"))
    _asked_already = bool(state.get("awaiting_early_late_time"))
    _in_stay = stage in ("confirmed", "in_house") or state.get("lifecycle_stage") in (
        "confirmed",
        "in_house",
    )

    if not _in_stay and (
        # "early check in at 3 am" — the whole request in one breath.
        (_mentions and _states_a_clock)
        # "10AM", one turn after the bot asked for the time. No early/late word
        # is present because the bot's own question already supplied the
        # context, and without this the guest loses the quote they were holding.
        or (_asked_already and _states_a_clock)
        # Asked again, but the clock is already on record: quote the fee rather
        # than ask for a time the bot already has.
        or (_mentions and _clock_on_record and _quote_on_screen)
        # Just a clock — "10AM", "at 7 in the morning" — with a priced quote in
        # front of the guest and an early/late time already in this conversation.
        # Nothing else this narrow needs saying, and the alternative measured on
        # the 2026-10-07 transcript is worse than the risk: the message is not a
        # room pick and not a confirmation, so `validator` reads it as a fresh
        # search and throws the guest's selected room away.
        #
        # This one is a judgement call, and deliberately the tightest of the
        # four. It does NOT fire for prose that merely mentions a time — "our
        # train is at 6pm", "we will reach by 8" — because `is_only_a_clock`
        # requires every other word to be filler; and it does not fire for a
        # guest who has never named a time for this stay, so the conversation
        # must already be about one.
        or (_bare_clock and _clock_on_record and _quote_on_screen)
    ):
        # Send the turn back through the quote so the fee lands on the total the
        # guest is already looking at, rather than letting `validator` start a
        # fresh search over it — "early check in 3 am" contains "check in",
        # which its new-date regex reads as new dates, and the selection they
        # just made is wiped.
        #
        # BOOK_ROOM rather than a new intent value: `feature_gate_node` gates a
        # fixed tuple of booking intents and an unknown string falls outside it.
        # The flag, not the intent, is what carries the meaning.
        extras["early_late_requote"] = True
        logger.info(
            "EARLY_LATE_REQUOTE | input='%s' arrival=%s departure=%s mentioned=%s "
            "asked=%s on_record=%s bare=%s",
            user_message,
            _stated.get("arrival"),
            _stated.get("departure"),
            _mentions,
            _asked_already,
            _clock_on_record,
            _bare_clock,
        )
        return "BOOK_ROOM", None, extras

    if _mentions:
        if _in_stay:
            # Room 204 asking for a late checkout is a service request, not a
            # change to a booking — leave the intent the classifier chose.
            return intent, None, extras

        # No clock anywhere to price. The only honest answer is the hotel's own
        # policy plus a request for the time — and the request is remembered, so
        # the guest's answer is understood when it arrives. Fetching the policy
        # text needs a Mongo read, so it happens in `menu_handler`, which is
        # async; that also keeps the reply honest, because a hotel with no active
        # row gets a generic availability answer rather than a fee it never set.
        extras["awaiting_early_late_time"] = True
        logger.info(
            "EARLY_LATE_POLICY | input='%s' -> SHOW_CHECKIN_POLICY", user_message
        )
        return "SHOW_CHECKIN_POLICY", None, extras

    # Never remap away from an in-stay task
    if intent == "HOTEL_TASK":
        return intent, None, extras

    # ── Explicit Bank Details Request ──
    if _BANK_DETAILS_PATTERN.search(user_message):
        logger.info("BANK_DETAILS_TRIGGER | input='%s' -> BANK_DETAILS", user_message)
        return "BANK_DETAILS", None, extras

    # ── In-chat discount & price negotiation inquiries ──
    _DISCOUNT_PATTERN = re.compile(
        r"\b(?:discount|discounts|offer|offers|concession|promo|coupon|price match|cheaper|kam karo|kam kijiye|sasta|rate kam|reduce price|less price|price kam|kam price|kam hoga|kam nahi hoga|best rate|best price|lowest price|better rate|aur kam|thoda aur|aur discount|aur sasta|kuch aur kam|aur thoda|thoda kam)\b"
        r"|@\s*\d{3,6}\b"
        r"|\b(?:confirm\s+)?(?:at|@)\s*(?:rs\.?|inr|₹)?\s*\d{3,6}\b"
        r"|\b(?:in|for|pe|par)\s*(?:rs\.?|inr|₹)?\s*\d{3,6}\b"
        r"|\b\d{3,6}\s*(?:me|mein|pe|par)\s*(?:kardo|kar\s*do|confirm|chalega|dena|de\s*do|final)\b",
        re.IGNORECASE,
    )
    if _DISCOUNT_PATTERN.search(user_message):
        from app.utils.room_requirement_parse import parse_room_requirements
        has_room_requirements = bool(parse_room_requirements(user_message))
        if not has_room_requirements and stage not in ("confirmed", "cancelled"):
            extras["next_action"] = None
            extras["payment_missing_fields"] = []
            return "PRICE_MATCH", None, extras

    # ── T5 Staff phone call handoff trigger & Missed-Call Booking Protection ──
    _CALL_UNANSWERED_CUE = re.compile(
        r"\b(?:answer|ans|uthaya|uthaye|receive|received|pick|picked|responded|response|lag|laga)\s*(?:nehi|nahi|not|ni|nhi)\b"
        r"|\b(?:no|nobody|no\s*one|kisi\s+ne)\s*(?:answer|response|uthaya|receive)\b"
        r"|\b(?:missed\s+call|call\s+(?:cut|disconnect))\b",
        re.IGNORECASE,
    )
    _BOOKING_OR_ROOM_CUE = re.compile(
        r"\b(?:hotel\s+)?book\s*(?:karn[aeio]|karein|chah[ei]|karo)|room\s+(?:book|chahiye|booking)|stay\b"
        r"|\b\d+\s*(?:log|guests?|adults?|persons?|people|pax)\b",
        re.IGNORECASE,
    )
    _CONTACT_FOR_BOOKING = re.compile(
        r"\bhow\s+(?:can\s+i|to)\s+(?:contact|call|reach|connect)\s+(?:for\s+)?(?:hotel\s+)?(?:booking|book)\b"
        r"|\b(?:hotel\s+)?(?:booking|book)\s+ke\s+liye\s+(?:kaise|kisko|kisse)\s+(?:contact|call|baat)\b",
        re.IGNORECASE,
    )

    if _CONTACT_FOR_BOOKING.search(user_message):
        extras["escalation_required"] = False
        return "BOOK_ROOM", None, extras

    has_unanswered_call = bool(_CALL_UNANSWERED_CUE.search(user_message))
    has_explicit_booking = bool(has_booking or _BOOKING_OR_ROOM_CUE.search(user_message))

    if has_explicit_booking and (has_unanswered_call or intent in ("COMPLAINT", "HUMAN_REQUEST", "ESCALATE")):
        # Guest mentions unanswered call or asks for booking after failed phone contact:
        # Prioritize room booking over human escalation
        extras["escalation_required"] = False
        extras.pop("escalation_reason", None)
        extras.pop("escalation_reason_code", None)
        extras.pop("bot_response", None)
        return "BOOK_ROOM", None, extras

    _STAFF_CALL_TRIGGER = re.compile(
        r"\b("
        r"(?:abhi\s+)?(?:maine|humne|hamne)\s+(?:call|phone)\s+(?:kiye?|kiya|kara)\s*(?:theee?|tha|hai)?"
        r"|(?:call|phone)\s+(?:kiya|kiye|kara)\s*(?:tha|hai|the)?"
        r"|already\s+called|i\s+(?:already\s+)?called|just\s+called"
        r"|spoke\s+(?:to|with)\s+(?:staff|manager|reception|hotel|sir)"
        r"|baat\s+hui\s+thi\s+(?:call|phone)\s+pe"
        r"|phone\s+pe\s+baat\s+hui"
        r")\b",
        re.IGNORECASE,
    )
    if _STAFF_CALL_TRIGGER.search(user_message) and not has_unanswered_call and not has_explicit_booking:
        logger.info("STAFF_CALL_TRIGGER | input='%s' -> immediate human escalation", user_message)
        extras["escalation_required"] = True
        extras["escalation_issue_type"] = "NORMAL"
        extras["escalation_reason"] = "guest_called_staff"
        extras["escalation_reason_code"] = "PHONE_CALL_HANDOFF"
        from app.prompts import templates
        extras["bot_response"] = templates.text_card(
            "Understood! 🙏 We have noted your call with our team. A staff member will assist you directly. Bot replies are paused so our team can handle your request."
        )
        return "ESCALATE", None, extras

    # ── Photo & gallery requests (mid-booking or general) ──
    _PHOTO_KW = r"(?:photos?|photoes|foto|fotos|pics?|pocs?|pictures?|images?|imgs?|gallery|tasveere?n?)"
    _PHOTO_PATTERN = re.compile(
        r"^\s*(?:all\s+)?(?:room\s+|hotel\s+|kamra\s+|kamre\s+)?(?:" + _PHOTO_KW + r")(?:\s+(?:please|plz|bhi|dikhao|bhejo|dekhna))?\s*$"
        r"|\b(?:show|send|share|give|need|want|see|view)\b.{0,25}\b(?:" + _PHOTO_KW + r")\b"
        r"|\b(?:" + _PHOTO_KW + r")\b.{0,25}\b(?:dikh?ao|dikh?ana|dikh?aye|bhej[on]|bhejiye|chahiye|dekhna|dekhe|milegi|hoga)\b"
        r"|\b(?:room|hotel|kamr[ae])\s+(?:ki\s+)?(?:" + _PHOTO_KW + r")\b"
        r"|\b(?:see\s+rooms?|view\s+rooms?|pura\s+pic|pure\s+photo)\b",
        re.IGNORECASE,
    )
    is_payment_proof = bool(
        re.search(r"\b(?:payment|screenshot|paid|screen\s*shot|transferred|upi\s*ref|utr|receipt)\b", user_message, re.I)
    )
    if not is_payment_proof and _PHOTO_PATTERN.search(user_message.strip()):
        has_photo_dates = (
            has_relative_dates
            or has_availability
            or bool(re.search(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH_NAME + r"|today|tomorrow|kal|aaj|\d{1,2}[/-]\d{1,2}|check\s*in|check\s*out|\d{4}-\d{2}-\d{2})\b", user_message, re.IGNORECASE))
        )
        extras["wants_photos"] = True
        feat_match = re.search(r"\b(washroom|bathroom|toilet|bath|restroom|reception|lobby|facade|entrance|pool|gym|restaurant|dining|balcony)\b", user_message, re.I)
        photo_tag = feat_match.group(1).lower() if feat_match else (msg_room_type or "")
        extras["photo_tag_preference"] = "washroom" if photo_tag in ("bathroom", "toilet", "bath", "restroom") else photo_tag
        if not has_photo_dates:
            return "SHOW_AMENITIES", None, extras
        else:
            return "BOOK_ROOM", None, extras

    # ── Amenity & Facility Inquiries (Deterministic Override) ──
    _AMENITY_QUESTION_CUES = re.compile(
        r"\b(?:"
        r"do\s+you\s+(?:have|provide|offer)|is\s+there(?:\s+a)?|are\s+there"
        r"|hai\s+kya|milega|milegi|hota\s+hai|hoti\s+hai|available\s+hai"
        r"|facility|facilities|amenity|amenities|service|services|suvidha|suvidhaye"
        r"|can\s+i\s+(?:get|have|use)|kya\s+hotel\s+me|hotel\s+me"
        r")\b|\?",
        re.IGNORECASE,
    )
    _AMENITY_TARGETS = [
        (re.compile(r"\b(swimming\s*pool|pool|swimming)\b", re.I), "swimming_pool"),
        (re.compile(r"\b(gym|fitness(?:\s*(?:center|centre))?|workout)\b", re.I), "gym_fitness_center"),
        (re.compile(r"\b(lift|elevator|escalator)\b", re.I), "elevator_lift"),
        (re.compile(r"\b(wifi|wi[- ]?fi|internet|lan)\b", re.I), "wifi"),
        (re.compile(r"\b(parking|car\s*parking|four[- ]?wheeler|two[- ]?wheeler|bike\s*parking|gaadi\s*(?:parking|park|khadi|lagane)|vehicle\s*parking)\b", re.I), "parking"),
        (re.compile(r"\b(spa|massage|wellness)\b", re.I), "spa"),
        (re.compile(r"\b(laundry|washing\s*machine|iron|ironing|dry\s*cleaning|press)\b", re.I), "laundry"),
        (re.compile(r"\b(power\s*backup|generator|bijli\s*backup|inverter)\b", re.I), "power_backup"),
        (re.compile(r"\b(doctor(?:\s*on\s*call)?|first[- ]?aid|medical\s*facility)\b", re.I), "doctor_on_call"),
        (re.compile(r"\b(wheelchair|ramp|handicap|disabled|specially\s*abled)\b", re.I), "wheelchair"),
        (re.compile(r"\b(non[- ]?veg(?:etarian)?|chicken|meat)\b", re.I), "non_vegetarian_food_allowed"),
        (re.compile(r"\b(outside\s*food|swiggy|zomato)\b", re.I), "outside_food_allowed"),
        (re.compile(r"\b(alcohol|liquor|wine|beer|daru|sharab)\b", re.I), "alcohol_served"),
        (re.compile(r"\b(smoking\s*rooms?|smoking\s*allowed|can\s*i\s*smoke|smoke)\b", re.I), "smoking_rooms"),
        (re.compile(r"\b(ev\s*charging|electric\s*vehicle)\b", re.I), "ev_charging_station"),
        (re.compile(r"\b(driver(?:\s*room|\s*stay|\s*accommodation)?)\b", re.I), "drivers_accommodation"),
        (re.compile(r"\b(refrigerator|fridge)\b", re.I), "refrigerator"),
        (
            re.compile(
                r"\b(?:"
                r"all\s*(?:amenities|facilities)"
                r"|what\s*(?:are\s*the\s*|all\s*)?(?:facilities|amenities)"
                r"|what\s*facilities(?:\s*do\s*you\s*have)?"
                r"|list\s*(?:of\s*)?(?:amenities|facilities)"
                r"|(?:amenities|facilities)\s*list"
                r"|show\s*(?:all\s*)?(?:amenities|facilities)"
                r"|hotel\s*(?:facilities|amenities)"
                r"|kya\s*(?:kya\s*)?suvidha"
                r"|facilities|amenities"
                r")\b",
                re.I,
            ),
            "all",
        ),
    ]

    is_booking_slot = looks_like_booking_slot_answer(user_message)
    is_ac_prompt = active_prompt_action == "need_ac_preference"
    if not is_booking_slot and not is_ac_prompt and not has_booking and not has_relative_dates:
        for pat, a_key in _AMENITY_TARGETS:
            if pat.search(user_message):
                if _AMENITY_QUESTION_CUES.search(user_message) or len(user_message.strip().split()) <= 4:
                    extras["amenity_query"] = {"amenity_name": a_key, "query_types": ["general"]}
                    logger.info("AMENITY_OVERRIDE | input='%s' -> AMENITY_QUERY (%s)", user_message, a_key)
                    return "AMENITY_QUERY", None, extras

    # ── Explicit Fresh Start (Sir's Spec / D9 / S27 / S46) ───────────────────
    # In Stage 1, an explicit 'fresh start' must immediately wipe the draft and restart.
    #
    msg_clean_lower = user_message.strip().lower()
    if is_explicit_start_fresh(user_message):
        extras.update(clear_in_progress_booking_fields(state))
        extras["booking_stage"] = "new"
        extras["lifecycle_stage"] = "lead"
        logger.info("FRESH_START_WIPED | Explicit fresh start in Stage 1 — cleared draft immediately")
        return "GREETING", None, extras

    # ── Lifecycle: semantic reset_booking with in-progress draft ─────────────
    if needs_start_fresh_confirmation(user_message, state, reset_booking=reset_booking):
        return "LIFECYCLE_RESET", None, extras

    # ── Session continuity: Gemini reset OR short "No" after continue ───────
    if wants_to_abandon_in_progress_booking(
        user_message, state, reset_booking=reset_booking
    ):
        extras.update(clear_in_progress_booking_fields(state))
        return "GREETING", None, extras

    # ── M3: Change Menu Trigger ──
    msg_clean_lower = user_message.strip().lower()
    if msg_clean_lower in ("change", "modify", "badlo", "change booking", "modify booking", "change details", "update details") and (stage in ("rooms_shown", "quote_shown") or has_options):
        return "SHOW_CHANGE_MENU", None, extras

    # ── Active Qualifier / Change Followup Prompt Guard ──
    active_prompt_or_change = active_prompt_action in (
        "change_menu_shown",
        "need_change_dates",
        "need_change_guests",
        "need_ac_preference",
        "need_meal_plan_preference",
        "need_child_ages",
        "need_composite_slots",
        "need_custom_rooms",
        "need_guest_name_for_payment",
    )

    # ── S3: Change Menu Selection Handling (fourth copy of this menu's reader) ──
    #
    # Reads the row titles through `resolve_button`, the same declaration `understand` uses
    # and the same one `list_card` validates the rows against. Before that this table was
    # kept by hand here *and* in `understand`, so the two could disagree about the same tap.
    #
    # The digits are gone (they were numbered until 2026-10-15) because they never worked:
    # the SELECT_ROOM handler immediately below claims every bare number at `rooms_shown`,
    # which is exactly the stage this menu fires at, so "1" here was selecting a room.
    if active_prompt_action == "change_menu_shown":
        from app.utils.button_registry import resolve_button
        _change_tap = resolve_button(user_message)
        _change_action = _change_tap.action if _change_tap else None
        if _change_action == "CHANGE_DATES" or msg_clean_lower in ("dates", "date", "tarikh", "change dates", "check in", "check out", "checkin", "checkout"):
            extras["next_action"] = "need_change_dates"
            return "BOOK_ROOM", None, extras
        elif _change_action == "CHANGE_GUESTS" or msg_clean_lower in ("guests", "guest", "room count", "rooms", "guests / room count", "log", "kamre", "change guests", "change rooms"):
            extras["next_action"] = "need_change_guests"
            return "BOOK_ROOM", None, extras
        elif _change_action == "CHANGE_AC" or msg_clean_lower in ("ac", "non-ac", "non ac", "ac / non-ac", "ac/non-ac", "change ac"):
            extras["next_action"] = "need_ac_preference"
            return "BOOK_ROOM", None, extras
        elif _change_action == "CHANGE_MEAL" or msg_clean_lower in ("meal plan", "meals", "meal", "khana", "breakfast", "change meal plan"):
            extras["next_action"] = "need_meal_plan_preference"
            return "BOOK_ROOM", None, extras

    # ── Deterministic SELECT_ROOM override at rooms_shown stage ───────
    # When at rooms_shown stage or allocation_options exist:
    # Option keys like "1B", "1A", "3A", bare numbers, "option 2", "cheapest", etc. MUST be treated as option selection,
    # NEVER as a guest count or room count!
    # Guard: If message contains explicit dates or new search parameters, treat as fresh search, not option selection.
    is_date_search = bool(
        re.search(r'\b(?:check\s*in|check\s*out|\d{4}-\d{2}-\d{2}|se\s+.*\s+tak|\d+\s+adults?)\b', user_message, re.IGNORECASE)
    )

    # ── Handle AC preference selection response explicitly ──
    #
    # Read by the shared matcher, so the words and the ordinals are one path
    # and "non - ac", "non_ac", "a/c" and "bina ac" resolve here as they do in
    # `understand`. A miss sets nothing and falls through: the guest gets asked
    # again rather than being handed the class they ruled out.
    if expected_action == "need_ac_preference":
        resolved_ac = ac_preference_from_prompt_reply(user_message)
        if resolved_ac:
            # This site's own vocabulary for "show me both" — `understand` calls
            # the same answer NO_PREFERENCE. The two are read identically
            # downstream (`room_searcher` treats both as unfiltered), so the
            # difference is naming, and it is left as it was found.
            extras["ac_preference"] = "SHOW_BOTH" if resolved_ac == "NO_PREFERENCE" else resolved_ac
            extras["ac_qualified"] = True
            extras["next_action"] = "search"
            logger.info("AC_PREFERENCE_OVERRIDE | input=\"%s\" -> %s", user_message, extras["ac_preference"])
            return "BOOK_ROOM", None, extras

    # ── Rooms-and-combinations card replies (1-4 / A-D) ──
    #
    # Deliberately ABOVE the numeric block below, because that block reads a
    # bare "3" as "the third priced option" while the card uses the numbers for
    # room *types*. It is deliberately BELOW the AC block above: under
    # `need_ac_preference` the same digits mean AC / Non-AC / Both, and the
    # resolver refuses to answer outside `rooms_shown`, so the two never
    # contend. On a non-card turn this returns None and nothing changes.
    from app.utils.room_card_reply import resolve_room_card_reply

    card_reply = resolve_room_card_reply(user_message, state, intent)
    if card_reply:
        card_intent, card_extras = card_reply
        extras.update(card_extras)
        logger.info(
            "ROOM_CARD_REPLY | input=\"%s\" | stage=%s | -> %s | extras=%s",
            user_message, stage, card_intent, sorted(card_extras),
        )
        return card_intent, None, extras

    if (
        (stage in ("rooms_shown", "quote_shown") or has_options)
        and not is_date_search
        and not active_prompt_or_change
        and intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "COMPLAINT", "HUMAN_REQUEST")
    ):
        from app.utils.room_card_reply import _normalize_keycaps
        msg_clean = _normalize_keycaps(user_message).strip()

        # 0. Check if user wants to see all room options / show more options
        more_options_match = re.search(
            r'^\s*(?:more|other\s+combos?|other\s+combinations?|more\s+options?|more\s+combos?|aur\s+options?|other\s+options?)\s*$'
            r'|\b(?:show|dikhao|aur|more|other|dusra|change|view|check|list|see|all)\s*(?:me\s*)?(?:more\s*)?(?:room\s*options?|rooms?|options?|combos?|combinations?)\b'
            r'|\b(?:room\s*options?|available\s*rooms?)\b',
            msg_clean,
            re.IGNORECASE
        )
        if more_options_match:
            extras["force_new_search"] = True
            extras["selected_rooms"] = []
            extras["room_type_preference"] = None
            extras["allocation_options"] = []
            extras["show_more_combinations"] = True
            extras["booking_stage"] = "collecting"
            extras["next_action"] = "search"
            logger.info("SHOW_ROOM_OPTIONS_OVERRIDE | input=\"%s\" | stage=%s", user_message, stage)
            return "BOOK_ROOM", None, extras

        # 0.05 Check for explicit list option tap / selection like "Option 1", "Opt 2"
        explicit_opt_match = re.search(r'^\s*(?:option|opt|choice|selection)\s*([1-9])\b', msg_clean, re.IGNORECASE)
        if explicit_opt_match:
            opt_num = int(explicit_opt_match.group(1))
            extras["room_type_preference"] = f"option {opt_num}"
            extras["selected_option_number"] = opt_num
            logger.info("SELECT_ROOM_EXPLICIT_OPTION | input=\"%s\" | stage=%s | option=%s", user_message, stage, opt_num)
            return "SELECT_ROOM", None, extras

        # 0.1 Check for explicit alphanumeric option key like "1A", "1B", "2A", "3A", "4B"
        alpha_match = re.search(r'\b(?:option|opt|choice|selection|no\.?)?\s*([1-9])\s*[-_]?\s*([a-zA-Z])\b', msg_clean, re.IGNORECASE)
        if not alpha_match:
            alpha_match = re.search(r'\b([1-9][a-zA-Z]|[a-zA-Z][1-9])\b', msg_clean, re.IGNORECASE)
        if alpha_match:
            if alpha_match.lastindex == 2:
                opt_label = f"{alpha_match.group(1)}{alpha_match.group(2)}".upper()
            else:
                raw = alpha_match.group(1).upper()
                if raw[0].isalpha() and raw[1].isdigit():
                    opt_label = f"{raw[1]}{raw[0]}"
                else:
                    opt_label = raw
            extras["room_type_preference"] = opt_label
            extras["selected_option_label"] = opt_label
            logger.info("SELECT_ROOM_ATTEMPT | input=\"%s\" | stage=%s | interpreted_as=option_label | label=%s", user_message, stage, opt_label)
            return "SELECT_ROOM", None, extras

        # 0.5. Check for cheapest / lowest price / sasta
        cheapest_match = re.search(r'\b(cheapest|chepest|lowest\s*price|most\s*affordable|best\s*value|sasta|sabse\s*sasta|budget\s*option)\b', msg_clean, re.IGNORECASE)
        if cheapest_match:
            extras["room_type_preference"] = "cheapest"
            extras["recommendation_preference"] = "CHEAPEST"
            logger.info("SELECT_ROOM_ATTEMPT | input=\"%s\" | stage=%s | interpreted_as=cheapest", user_message, stage)
            return "SELECT_ROOM", None, extras

        # 0.6. Check for most expensive / luxury / premium
        expensive_match = re.search(r'\b(most\s*expensive|luxury|premium|highest\s*price|sabse\s*mahnga|mehenga)\b', msg_clean, re.IGNORECASE)
        if expensive_match:
            extras["room_type_preference"] = "expensive"
            extras["recommendation_preference"] = "EXPENSIVE"
            logger.info("SELECT_ROOM_ATTEMPT | input=\"%s\" | stage=%s | interpreted_as=expensive", user_message, stage)
            return "SELECT_ROOM", None, extras

        num_match = re.search(r'\b(?:option|opt|choice|selection|no\.?|room)\s*(\d+)\b', msg_clean, re.IGNORECASE)
        bare_match = re.search(r'^\s*(?:(?:i\s+)?(?:want|choose|select|book|take)|give\s+me)?\s*(?:option|opt|choice|selection|room|no\.?)?\s*(\d+)\s*(?:please|pls|wala|wali)?\.?\s*$', msg_clean, re.IGNORECASE)
        ordinal_match = re.search(r'\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\s*(?:option|wala|choice)?\b', msg_clean, re.IGNORECASE)
        hindi_ordinal = re.search(r'\b(pehla|pahla|dusra|doosra|teesra|tisra|chautha|panchwa)\s*(?:option|wala|wali|choice)?\b', msg_clean, re.IGNORECASE)
        
        # A number is only an option number if a line with that number was
        # actually drawn. Unbounded, every digit string at `rooms_shown` became
        # a pick: guest 919088198242 sent "2500" (a budget) and "1550" (a rate)
        # and the override read them as option 2500 and option 1550 — one
        # matched a room by accident, the other produced SELECT_ROOM_NO_MATCH
        # and the same card back. Past the last line the turn is not a pick,
        # so it falls through and the model's own verdict stands.
        from app.services.room_service import ROOM_CARD_MAX_COMBOS

        opts_on_screen = state.get("allocation_options") or []
        shown = min(len(opts_on_screen), ROOM_CARD_MAX_COMBOS)

        opt_num = None
        if bare_match and 1 <= int(bare_match.group(1)) <= shown:
            opt_num = int(bare_match.group(1))
        elif num_match and 1 <= int(num_match.group(1)) <= shown:
            opt_num = int(num_match.group(1))
        elif ordinal_match:
            ord_map = {
                "1st": 1, "first": 1,
                "2nd": 2, "second": 2,
                "3rd": 3, "third": 3,
                "4th": 4, "fourth": 4,
                "5th": 5, "fifth": 5,
            }
            opt_num = ord_map.get(ordinal_match.group(1).lower())
        elif hindi_ordinal:
            hindi_map = {
                "pehla": 1, "pahla": 1,
                "dusra": 2, "doosra": 2,
                "teesra": 3, "tisra": 3,
                "chautha": 4,
                "panchwa": 5
            }
            opt_num = hindi_map.get(hindi_ordinal.group(1).lower())

        if opt_num is not None and not (1 <= opt_num <= shown):
            opt_num = None  # "5th" when three lines were drawn

        if opt_num is not None:
            extras["room_type_preference"] = f"option {opt_num}"
            extras["selected_option_number"] = opt_num
            logger.info("SELECT_ROOM_ATTEMPT | input=\"%s\" | stage=%s | interpreted_as=option_num | option=%s", user_message, stage, opt_num)
            return "SELECT_ROOM", None, extras

        # Check room name match against allocation_options
        opts = state.get("allocation_options") or []
        for idx, o in enumerate(opts[:3], 1):
            for r in o.get("rooms", []):
                r_nm = str(r.get("room_name") or "").strip().lower()
                clean_lower = msg_clean.lower()
                if r_nm and (clean_lower == r_nm or (len(clean_lower) >= 4 and clean_lower in r_nm)):
                    extras["room_type_preference"] = f"option {idx}"
                    extras["selected_option_number"] = idx
                    logger.info("SELECT_ROOM_ATTEMPT | input=\"%s\" | stage=%s | interpreted_as=room_name | option=%s", user_message, stage, idx)
                    return "SELECT_ROOM", None, extras

    # ── Bengali / Hindi / Hinglish Billing / Quote Request Override ──
    _BILL_SEND_REQUEST = re.compile(
        r"\b("
        r"(?:bill|invoice|parchi|receipt|hisab)\s*(?:vej|bhej|send|de|do|dijiye|bhejiye|kijiye|karo)"
        r"|(?:vej|bhej|send)\s*(?:dijiye|do)?\s*(?:bill|invoice|parchi|receipt|hisab)"
        r"|send\s*bill|give\s*bill"
        r")\b",
        re.IGNORECASE,
    )
    _BILLING_QUOTE_HINT = re.compile(
        r"\b("
        r"bill\s*kore\s*din|bill\s*kore|bill\s*banao|bill\s*banaye|bill\s*karo"
        r"|make\s*bill|generate\s*bill|total\s*bill|quote\s*banao|quote\s*do"
        r"|hisab\s*karo|hisab\s*batao|invoice\s*banao"
        r")\b",
        re.IGNORECASE,
    )
    if _BILL_SEND_REQUEST.search(user_message):
        logger.info("BILL_SEND_REQUEST_OVERRIDE | input=\"%s\" -> HOTEL_TASK", user_message)
        extras.update(_infer_task_extras(user_message))
        return "HOTEL_TASK", None, extras

    if _BILLING_QUOTE_HINT.search(user_message):
        has_dates_and_guests = bool(state.get("check_in") and state.get("check_out") and state.get("adults"))
        is_in_stay = stage in ("confirmed", "in_house") or state.get("lifecycle_stage") in ("confirmed", "in_house")
        if has_dates_and_guests and not is_in_stay:
            logger.info("BILLING_QUOTE_OVERRIDE | input=\"%s\" -> BOOK_ROOM", user_message)
            extras["force_new_search"] = True
            extras["booking_stage"] = "collecting"
            extras["next_action"] = "search"
            return "BOOK_ROOM", None, extras
        else:
            logger.info("BILLING_QUOTE_OVERRIDE | input=\"%s\" -> HOTEL_TASK", user_message)
            extras.update(_infer_task_extras(user_message))
            return "HOTEL_TASK", None, extras

    # ── In-stay service requests (don't fall into booking loops) ──
    if _HOTEL_TASK_HINT.search(user_message) and intent in (
        "FAQ",
        "UNKNOWN",
        "BOOKING_SUMMARY",
        "CHECK_EXISTING_BOOKING",
        "BOOK_ROOM",
        "SHOW_PRICES",
        "SOMETHING_ELSE",
    ):
        extras.update(_infer_task_extras(user_message))
        return "HOTEL_TASK", None, extras

    # ── Photo Requests (e.g. "send proper pictures", "photo bhejo", "room photos") ──
    _PHOTO_HINT = re.compile(r"\b(?:photo|photos|pics?|pictures?|images?|tasveer|tasveere|tasvir)\b", re.IGNORECASE)
    if not is_payment_proof and _PHOTO_HINT.search(user_message):
        has_photo_dates = (
            has_relative_dates
            or has_availability
            or bool(re.search(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH_NAME + r"|today|tomorrow|kal|aaj|\d{1,2}[/-]\d{1,2}|check\s*in|check\s*out|\d{4}-\d{2}-\d{2})\b", user_message, re.IGNORECASE))
        )
        extras["wants_photos"] = True
        feat_match = re.search(r"\b(washroom|bathroom|toilet|bath|restroom|reception|lobby|facade|entrance|pool|gym|restaurant|dining|balcony)\b", user_message, re.I)
        photo_tag = feat_match.group(1).lower() if feat_match else (msg_room_type or "")
        extras["photo_tag_preference"] = "washroom" if photo_tag in ("bathroom", "toilet", "bath", "restroom") else photo_tag
        if not has_photo_dates:
            return "SHOW_AMENITIES", None, extras
        else:
            return "BOOK_ROOM", None, extras

    # ── S3: AC Preference Qualifier Response ──
    if active_prompt_action == "need_ac_preference":
        msg_clean = user_message.strip().lower()
        resolved_ac = ac_preference_from_prompt_reply(user_message)
        if resolved_ac == "AC":
            extras["ac_preference"] = "AC"
            extras["ac_qualified"] = True
            feat_list = list(set((state.get("feature_preferences") or []) + ["AC"]))
            extras["feature_preferences"] = feat_list
            return "BOOK_ROOM", None, extras
        elif resolved_ac == "NON_AC":
            extras["ac_preference"] = "NON_AC"
            extras["ac_qualified"] = True
            feat_list = [f for f in (state.get("feature_preferences") or []) if f != "AC"] + ["Non-AC"]
            extras["feature_preferences"] = list(set(feat_list))
            return "BOOK_ROOM", None, extras
        elif resolved_ac == "NO_PREFERENCE":
            extras["ac_preference"] = "NO_PREFERENCE"
            extras["ac_qualified"] = True
            return "BOOK_ROOM", None, extras

    # ── Deterministic Payment Receipt / Confirmation Claim Override ──
    _PAYMENT_CLAIM_PHRASE = re.compile(
        r"(?:payment\s*(?:done|ho\s*gaya|completed?|successful?|bhej\s*diya?|kar\s*diya?)"
        r"|\bpaid\b|i\s*have\s*paid|maine\s*pay(?:ment)?\s*(?:bhi\s*)?(?:kar\s*diya|bhej\s*diya)|money\s*sent|advance\s*(?:paid|bheja|diya)"
        r"|amount\s*transferred|screenshot\s*(?:sent|attached|bheja|dekho|receipt|hai|check|please)"
        r"|(?:bhej\s*diya|bheja\s*hai)\s*(?:bhi\s*)?(?:screenshot|payment|receipt)"
        r"|check\s*payment|payment\s*receipt|here\s*is\s*(?:the\s*)?(?:payment|receipt|screenshot)|transferred|utr\s*[:\s#]*[a-zA-Z0-9]+)",
        re.IGNORECASE,
    )
    if _PAYMENT_CLAIM_PHRASE.search(user_message):
        has_any_booking = bool(
            state.get("booking_id")
            or state.get("total_price")
            or state.get("check_in")
            or stage in ("awaiting_payment", "ready_for_payment", "quote_shown", "rooms_shown", "confirmed", "collecting")
        )
        if has_any_booking:
            return "PAYMENT_STATUS", None, extras

    # ── Explicit "Continue" / "Resume" in-progress booking ──
    _CONTINUE_PHRASE = re.compile(
        r"^\s*(?:continue|resume|continue\s+booking|resume\s+booking|pick\s+up|aage\s+badho|ha\s+continue|haan\s+continue|yes\s+continue|okay\s+continue|keep\s+going)\s*[.!]?\s*$",
        re.IGNORECASE,
    )
    if _CONTINUE_PHRASE.match(user_message) and has_in_progress_draft(state):
        stage = state.get("booking_stage", "new")
        if stage == "quote_shown":
            return "BOOKING_SUMMARY", None, extras
        elif stage == "awaiting_payment":
            return "PAYMENT_STATUS", None, extras
        elif stage == "rooms_shown" or has_options:
            extras["force_new_search"] = False
            return "BOOK_ROOM", None, extras
        else:
            return "BOOK_ROOM", None, extras

    # ── S0/S1 Root Menu Words (third copy of the same table) ──
    # ONLY applies at the initial greeting/menu stage when no active prompt and no booking details exist
    #
    # Bare digits removed 2026-10-15 for the reason `understand.py` gives at its C1 table:
    # they answered a 1-6 menu (`inq_greet`) that nothing sends. The words stay.
    #
    # This is the third copy of one table, and it agrees with `understand.py`'s C1 copy on
    # every word except one — both read "room prices & availability" as `SHOW_PRICES` while
    # `understand.py`'s V3 copy reads it as `BOOK_ROOM`. Left as found; see the note there.
    active_prompt = active_prompt_action in (
        "need_ac_preference",
        "need_child_ages",
        "need_composite_slots",
        "need_custom_rooms",
        "need_guest_name_for_payment",
        "change_menu_shown",
    )
    has_dates_or_guests = bool(state.get("check_in") or state.get("adults"))
    if stage == "new" and not has_options and not active_prompt and not has_dates_or_guests:
        msg_clean = user_message.strip().lower()
        if msg_clean in ("make a new booking", "new booking"):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "BOOK_ROOM", None, extras
        elif msg_clean in ("check my existing booking", "my booking"):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "CHECK_EXISTING_BOOKING", None, extras
        elif msg_clean in ("room prices & availability", "price and availability"):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "SHOW_PRICES", None, extras
        elif msg_clean in ("hotel location & directions", "location", "directions"):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "SHOW_LOCATION", None, extras
        elif msg_clean in ("room photos & amenities", "photos", "amenities"):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "SHOW_AMENITIES", None, extras
        elif msg_clean in ("something else", "other"):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "SOMETHING_ELSE", None, extras

    if _EXISTING_OR_LAST_BOOKING.search(user_message):
        # Avoid stealing "my booking total" / current quote questions
        if not _CURRENT_BOOKING_PRICE.search(user_message) or re.search(
            r"\b(last|previous|past|existing|history)\b", user_message, re.I
        ):
            if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
                return "CHECK_EXISTING_BOOKING", None, extras

    # ── Suggestion request ("suggest me", "recommend room", "show options") ──
    if _SUGGESTION_REQUEST.search(user_message):
        if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS", "CONFIRM_BOOKING"):
            return "BOOK_ROOM", None, _force_new_search_extras(extras)

    # ── T02: General room rates catalog → SHOW_PRICES (never current quote) ──
    if _GENERAL_RATES.search(user_message) and not msg_room_type:
        has_who_and_when = bool(
            (state.get("check_in") or extras.get("check_in"))
            and (state.get("adults") or state.get("selected_rooms") or state.get("allocation_options"))
        )
        if not has_who_and_when and intent not in ("CONFIRM_BOOKING", "CANCEL_BOOKING", "PAYMENT_STATUS"):
            return "SHOW_PRICES", None, extras

    # ── T03: New-date availability → fresh booking search (not payment/FAQ) ──
    if has_relative_dates and has_availability:
        if intent not in ("CANCEL_BOOKING", "PAYMENT_STATUS"):
            return "BOOK_ROOM", None, _force_new_search_extras(extras)

    # ── Deterministic CONFIRM_BOOKING override when quote or rooms are shown ──
    # Guest 918918122852, 2026-10-19 07:13: "Ok... thank you sir." became
    # CONFIRM_BOOKING at stage=lead. The model had called it THANK_YOU; the
    # `ok` inside the farewell matched `_CONFIRM_PHRASE`, and `total_price`
    # was still on the state from a session the ghost-confirmed guard had
    # already demoted to lead. A farewell, called a farewell by the model, is
    # never a confirmation — whatever old price is still lying around.
    is_farewell = bool(intent == "THANK_YOU" and _THANKS_WORD.search(user_message))
    if (
        stage in ("quote_shown", "rooms_shown") or state.get("total_price")
    ) and _CONFIRM_PHRASE.search(user_message):
        if (
            not is_farewell
            and not _DISCOUNT_PATTERN.search(user_message)
            and msg_clean_lower not in ("change", "modify", "badlo")
            and not is_date_search
            and not active_prompt_or_change
        ):
            logger.info("CONFIRM_PHRASE_OVERRIDE | input=\"%s\" | stage=%s -> CONFIRM_BOOKING", user_message, stage)
            return "CONFIRM_BOOKING", None, extras

    # Misclassified CONFIRM while asking something else
    if intent == "CONFIRM_BOOKING" and not _CONFIRM_PHRASE.search(user_message):
        if has_availability or has_relative_dates or _GENERAL_RATES.search(user_message):
            if has_relative_dates or has_availability:
                return "BOOK_ROOM", None, _force_new_search_extras(extras)
            if _GENERAL_RATES.search(user_message):
                return "SHOW_PRICES", None, extras

    # ── Mid-booking escape hatch (policy side-questions only) ──
    if (
        stage in _ESCAPE_STAGES
        and intent in _ESCAPE_FROM_INTENTS
        and not looks_like_booking_slot_answer(user_message)
        and is_mid_booking_side_question(user_message)
    ):
        return "FAQ", user_message, extras

    # Structured pricing only when a room type is IN THE MESSAGE (or current-booking price ask).
    #
    # A room *pick* is not a price question. `has_booking` matches the verb
    # "want", so a SELECT_ROOM that names a type — guest 916297020216,
    # 2026-10-19 10:34, "we want standered non ac" while the card was on
    # screen — was rewritten into BOOKING_SUMMARY and the pick was thrown
    # away. Keep the rewrite for every other intent, and for SELECT_ROOM only
    # when the guest actually asked about money.
    if msg_room_type and (has_booking or _PRICING_PATTERN.search(user_message)) and not (
        intent == "SELECT_ROOM" and not _PRICING_PATTERN.search(user_message)
    ):
        if intent not in ("CONFIRM_BOOKING", "CANCEL_BOOKING"):
            if has_dates and (stage in ("rooms_shown", "quote_shown") or has_options):
                # Specific type pricing while options shown — summary of options OK
                if _CURRENT_BOOKING_PRICE.search(user_message):
                    return "BOOKING_SUMMARY", None, extras
                return "BOOKING_SUMMARY", None, _apply_pricing_extras(
                    extras, user_message, state, msg_room_type, nights
                )
            return "BOOKING_SUMMARY", None, _apply_pricing_extras(
                extras, user_message, state, msg_room_type, nights
            )

    if _PRICING_PATTERN.search(user_message) and msg_room_type:
        if intent not in ("CONFIRM_BOOKING", "CANCEL_BOOKING"):
            return "BOOKING_SUMMARY", None, _apply_pricing_extras(
                extras, user_message, state, msg_room_type, nights
            )

    if num_rooms and has_booking and _PRICING_PATTERN.search(user_message):
        if intent not in ("CONFIRM_BOOKING", "CANCEL_BOOKING"):
            extras["pricing_query_num_rooms"] = num_rooms
            extras["num_rooms"] = num_rooms
            extras["pricing_query_nights"] = nights or 1
            return "BOOKING_SUMMARY", None, extras

    # "what's my total?" while quote/options on screen
    if _CURRENT_BOOKING_PRICE.search(user_message) and has_dates:
        if intent not in ("CONFIRM_BOOKING", "CANCEL_BOOKING"):
            if stage in ("rooms_shown", "quote_shown") or has_options or state.get("selected_rooms"):
                return "BOOKING_SUMMARY", None, extras

    # Pricing words alone with dates + options, but NO room type in message and NOT general catalog:
    # only treat as current booking summary if clearly about "my" booking — else SHOW_PRICES
    if _PRICING_PATTERN.search(user_message) and not msg_room_type and not _GENERAL_RATES.search(user_message):
        if intent not in ("CONFIRM_BOOKING", "CANCEL_BOOKING"):
            if _CURRENT_BOOKING_PRICE.search(user_message) and (
                stage in ("rooms_shown", "quote_shown") or has_options
            ):
                return "BOOKING_SUMMARY", None, extras

    # Relative dates without availability keyword still → BOOK_ROOM for date extraction
    if has_relative_dates and intent in ("FAQ", "CHECK_AVAILABILITY", "UNKNOWN", "SHOW_PRICES"):
        if not is_mid_booking_side_question(user_message):
            return "BOOK_ROOM", None, _force_new_search_extras(extras) if stage in (
                "quote_shown", "rooms_shown", "awaiting_payment"
            ) else extras

    # Generic availability/policy while collecting dates — FAQ (not re-ask)
    if has_availability and not has_dates and not has_relative_dates:
        if stage in ("new", "collecting") and intent in (
            "BOOK_ROOM",
            "CHECK_AVAILABILITY",
            "MODIFY_BOOKING",
            "UNKNOWN",
        ):
            return "FAQ", user_message, extras

    # ── A room named while the card is on screen is a pick, whatever the label ──
    #
    # The label is the model's opinion, and on a spelling it does not recognise
    # it changes. Guest 919088198242 (2026-10-19 11:16) answered the card with
    # "Standerd non ac 1 room" — the same room they had named twice already — and
    # the turn was labelled UNKNOWN. `route_by_intent` only excuses a repeat from
    # the escalation guard when the label is SELECT_ROOM, so the fourth identical
    # card became SUP-20261019-000004. Naming a line that is on the card is not
    # ambiguous, so the message decides here, before the label does.
    #
    # Deliberately narrow: only while rooms are on screen, only when the message
    # carries a word from a room name that is actually in front of the guest, and
    # never for the intents that must reach a person or that settle a quote.
    if stage == "rooms_shown" and has_options and intent not in (
        "HUMAN_REQUEST",
        "ESCALATE",
        "COMPLAINT",
        "CONFIRM_BOOKING",
        "CANCEL_BOOKING",
    ) and not _COMPLAINT_CUE.search(user_message or ""):
        if _names_a_card_room(user_message, state.get("allocation_options") or []):
            return "SELECT_ROOM", None, extras

    return intent, None, extras
