"""
Understand Node — combined intent classification + entity extraction.

This is the ONLY node that calls Gemini for understanding.
One call extracts BOTH intent and booking entities.
"""

import datetime
import logging
import re
from typing import Any

from app.config.settings import get_settings
from app.services.gemini_service import (
    classify_payment_turn,
    extract_room_request,
    understand_message,
)
from app.utils.ac_text import (
    ac_preference_from_prompt_reply,
    ac_preference_from_text,
    is_bare_ac_class,
    is_plausible_ac_answer,
    mentions_both_classes,
)
from app.utils.button_registry import resolve_button
from app.utils.competitor_mention import mentions_competitor
from app.utils.date_masking import mask_dates
from app.utils.helpers import today_str
from app.utils.intent_overrides import apply_intent_overrides, is_explicit_start_fresh
from app.utils.pax_resolution import resolve_pax_counts
from app.utils.person_name import extract_person_name, looks_like_a_name_answer
from app.utils.room_requirement_parse import is_an_explicit_split
from app.workflows.nodes import understand_fast_paths as fast_paths
from app.workflows.nodes.understand_slices.active_prompt import (
    handle_active_prompt,
    handle_change_menu_prompt,
)
from app.workflows.nodes.understand_slices.button_tap import handle_tap_routed
from app.workflows.nodes.understand_slices.confirmed_booking import (
    handle_confirmed_reservation,
)
from app.workflows.nodes.understand_slices.name_expected import (
    apply_expected_action_reply,
)
from app.workflows.nodes.understand_slices.booking_intent import (
    _apply_split_party,
    apply_booking_intent,
)
from app.workflows.session_mapper import state_to_session
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# ── OPT-OUT / OPT-IN COMPLIANCE (ZERO-LLM FAST PATH) ─────────────────────────
# A STOP must be the whole message. The first version anchored with `^` but
# tested with `search`, so any message *beginning* with "stop" opted the guest
# out — "stop by the hotel at 5" wiped `selected_rooms`, `allocation_options`,
# `total_price` and `payment_link`, marked the session `opted_out`, and every
# later message got the unsubscribed reply. The tolerated tail is a short fixed
# list of fillers, deliberately not `.*`: "stop sending me photos" and "stop at
# the hotel" must reach the normal flow. Bare "cancel" is absent on purpose —
# cancelling a booking is a different flow, not a marketing opt-out.
STOP_PATTERN = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:stop(?:\s+all)?|unsubscribe|unsub|opt\s*out|optout|opt-out|halt|quit|cancel\s+subscription)"
    r"(?:\s+(?:please|now|messages?|msg|it|this))?"
    r"[\s.!]*$",
    re.IGNORECASE,
)

START_PATTERN = re.compile(r"^\s*(?:start|unstop|subscribe|opt\s*in)\s*[.!]?\s*$", re.IGNORECASE)


# ── Language cues ─────────────────────────────────────────────────────────
#
# Moved to module level because the node rebuilt all of these on every message,
# and because the legacy and expanded Hinglish sets need to sit side by side so
# the `LANGUAGE_DETECTION_WEIGHTS_HINDI` gate can be read as a diff.
#
# Which set is in play is decided in `understand`; see `settings.py` for why.
_HINGLISH_CUES_LEGACY = {
    "chahiye", "hai", "hain", "kardo", "karna", "bhai", "bhaiya", "batao", "bataye", "batayiye",
    "kitna", "kitne", "mera", "meri", "hoga", "hogi", "shukriya", "dhanyawad", "alag",
    "kamre", "kamra", "kharcha", "aaram", "sasta", "dono", "dikhaye", "dikhaiye",
    "kaisa", "tarah", "zyada", "sabko", "bade", "bacche", "apna", "saath", "seedha", "jaise",
    "theek", "haan", "nahi", "naam", "namaste"
}

#: The legacy set plus the function words Roman Hindi is actually built out of.
#: Their absence is why "room ki photo bhejo" scored zero Hindi cues and fell
#: through to the English default: `ki`, `ka`, `ke`, `kya`, `mujhe`, `aur` and
#: `wala` were simply not in the list.
_HINGLISH_CUES_EXPANDED = _HINGLISH_CUES_LEGACY | {
    "kya", "mujhe", "ka", "ki", "ke", "ko", "mein", "se", "bhi", "toh", "aur", "ya",
    "aap", "hum", "tum", "dena", "karo", "kar", "milega", "paisa", "paise", "din", "raat",
    "aaj", "kal", "abhi", "accha", "achha", "thik", "kuch", "koi", "wala", "wali", "wale",
    "bhejo", "bhej", "dikhao", "bata", "kaise", "log", "liye", "sirf", "phir", "lekin",
    "magar", "kyun", "kab", "kahan", "jaldi", "raha", "rahi", "gaya", "gayi", "kiya",
    "diya", "liya", "hua", "hui", "sakta", "sakte", "chalo", "bahut", "kam",
    "pehle", "baad", "saal", "baje", "band", "khula", "chalu",
}

#: English cues. Note these are *content* words Hinglish routinely borrows —
#: "room", "book", "family", "price". That is fine under a proportional rule,
#: and it is exactly what made the old `e_count > h_count` rule misfire.
_ENGLISH_CUES = {
    "want", "need", "please", "book", "rooms", "adults", "adult", "children", "child", "checkin", "checkout",
    "night", "nights", "hello", "hi", "sorry", "we", "are", "i", "am", "how", "much", "can", "you",
    "available", "cost", "price", "comfortable", "cheapest", "option", "split", "suggest",
    "separate", "family", "decide", "best"
}

#: Words that answer a question without saying anything about language. A bare
#: "thank you" or "ok" mid-Hinglish conversation carried one English cue, won
#: the English branch, and flipped the rest of the conversation to English.
_LANGUAGE_NEUTRAL_WORDS = {
    "thanks", "thank", "thankyou", "thx", "ty", "ok", "okay", "k", "done",
    "yes", "no", "yep", "yeah", "sure", "cool", "great", "fine", "good",
    "nice", "bye", "welcome", "hmm", "oh", "ah", "please",
    "and", "the", "a", "an", "is", "it", "that", "this", "for", "of", "to",
    "my", "me", "you", "we",
}


# ── Confirmed-session side questions ──────────────────────────────────────
#
# A guest whose booking is already confirmed is not shopping any more. The
# phrases below are the ones that should still be allowed through to the
# amenity / location / FAQ branches further down the node; everything else in
# a confirmed session is answered in place and the bot stops there.
#
# Deliberately generous. A false positive here costs a room card shown to a
# guest who did not want one — the bug being fixed. A false negative costs a
# canned reply instead of a real answer, which is the worse trade.
_CONFIRMED_SIDE_QUERIES = (
    # amenities and facilities
    "amenit", "facilit", "wifi", "wi-fi", "internet", "parking", "pool",
    "gym", "laundry", "restaurant", "breakfast", "lunch", "dinner", "food",
    "hot water", "ac ", "air condition", "lift", "elevator", "tv",
    # getting here
    "location", "direction", "address", "map", "how to reach", "distance",
    "landmark", "nearby", "taxi", "airport", "station",
    # arrival admin
    "check in", "check-in", "checkin", "id card", "id proof", "aadhaar",
    "aadhar", "pan card", "passport", "early check", "late check",
    # after-sales
    "cancel", "modify", "refund", "invoice", "receipt", "bill", "gst",
    "parchi", "hisab", "vej", "bhej",
    "payment", "paid", "utr", "screenshot",
    # anything that is plainly a question
    "?", "help", "kya", "kaise", "where", "what", "how",
)


def _is_confirmed_side_query(user_msg_clean: str) -> bool:
    """Whether a message in a confirmed session is a side question rather than
    an attempt to book again. See `_CONFIRMED_SIDE_QUERIES`."""
    low = (user_msg_clean or "").lower()
    return any(cue in low for cue in _CONFIRMED_SIDE_QUERIES)


#: Intents that record something the guest has done, or asked to have done, to a
#: booking. A price phrase must never overwrite one of these — see
#: `_booking_act_intent` and its use in the price-negotiation override below.
_BOOKING_ACT_INTENTS = frozenset({
    "CANCEL_BOOKING",
    "MODIFY_BOOKING",
    "PAYMENT_STATUS",
})


#: Phrases that ask for a better price. Module-level, not a local of `understand`,
#: so a test can assert what they do and do not match — see
#: `tests/test_booking_act_beats_price_phrase.py`. The last alternation is the one
#: that read a cancellation as a discount demand; the test says so in writing.
QUOTE_NEGOTIATION_PATTERNS = [
    r"\b(?:discount|discounts|offer|offers|concession|promo|coupon|price\s*match|cheaper)\b",
    r"\b(?:give\s+(?:me\s+)?discount|need\s+discount|can\s+you\s+reduce|reduce\s+price|less\s+price|price\s+kam|kam\s+price|kam\s+karo|kam\s+kijiye|rate\s+kam|kam\s+hoga|kam\s+nahi\s+hoga)\b",
    r"\b(?:aur\s+kam|thoda\s+aur|aur\s+discount|aur\s+sasta|kuch\s+aur\s+kam|aur\s+kam\s+karo|aur\s+thoda|thoda\s+kam|kamti|sasta)\b",
    r"\b(?:discount\s*milega|kuch\s*discount|koi\s*offer|offer\s*hai|kya\s*offer)\b",
    r"@\s*\d{3,6}\b",
    r"\b(?:confirm\s+)?(?:at|@)\s*(?:rs\.?|inr|₹)?\s*\d{3,6}\b",
    r"\b(?:in|for|at|pe|par)\s*\d{3,6}\b",
    r"\b\d{3,6}\s*(?:me|mein|pe|par)\s*(?:kardo|kar\s*do|confirm|chalega|dena|de\s*do|final)\b",
    # A guest holding our own number up against our own number: "here is
    # showing 2000 then how can be 2250". Measured 2026-10-31 (Sample Property): the
    # model read that as CONFIRM_BOOKING, so nothing above fired and the turn
    # re-printed the identical summary — three times, until staff closed it by
    # hand. It is not a discount request and it is not consent; it is a price
    # question, which is what the price path answers. Kept numeric on purpose:
    # a cue word alone ("why is the price high") is a question, not a dispute
    # we can name a figure for.
    r"\b(?:showing|shows|shown|says|said|quoted|listed|mentioned|advertised|displayed)\b[^.?!]{0,24}?\d{3,6}\b",
    r"\bhow\s+can\s+(?:it|this|that|you\s+)?be\s*(?:rs\.?|inr|₹)?\s*\d{3,6}\b",
    # Last on purpose: `tests/test_booking_act_beats_price_phrase.py` pins this
    # alternation as the one the cancellation message hits, and the veto above
    # is what keeps it from claiming a booking act.
    r"\b(?:in\s+|for\s+|me\s+|mein\s+|pe\s+|to\s+|at\s+)?(?:rs\.?|inr|₹)\s*\d{3,6}\b",
]


def _booking_act_intent(updates_intent: str | None, result_intent: str | None) -> str | None:
    """The booking act this turn is, or None if it is not one.

    The set is closed on purpose, and that is the whole design. The false-positive
    space of money phrases is not enumerable — "advance of Rs. 3000", "refund of
    Rs. 3000", a receipt's "₹3,000", "budget is 4000", "3000 mein ho jayega" — so
    no regex can be narrowed far enough to be safe. The set of acts a price phrase
    must not overwrite is small and closed, so this enumerates that instead.

    `updates_intent` is checked first: the word-lists earlier in `understand` are
    deterministic and already claim the turn for a concrete flow (the image
    analysis writes PAYMENT_STATUS, for one). `result_intent` is the model's
    verdict on the same message, which the routing below already reads beside
    `updates` for exactly this reason.
    """
    if updates_intent in _BOOKING_ACT_INTENTS:
        return updates_intent
    if result_intent in _BOOKING_ACT_INTENTS:
        return result_intent
    return None


def get_active_prompt(state: dict[str, Any]) -> str:
    """Return the active waiting prompt action from state or last prompt action."""
    action = state.get("next_action") or state.get("last_prompt_action") or ""
    return str(action).strip()


#: The card's trailing "Tell me how you want to split — I'll quote it" line,
#: spelled out whenever a guest types it rather than tapping the number.
_BUILD_YOUR_OWN_ALIASES = (
    "custom", "build your own", "build own", "custom room",
    "apna banao", "build your own room", "build my own",
)


def _build_your_own_number(state: dict[str, Any]) -> str | None:
    """The number the card printed for its build-your-own line, or None.

    One definition, because there were about to be two that disagreed. The card
    draws this line one past the priced rows (`room_service`), so the number is
    derived from the same live list here. None when no card is on screen, since
    "1" then means anything but a build-your-own pick.
    """
    options = state.get("allocation_options") or []
    if not options:
        return None
    from app.services.room_service import ROOM_CARD_MAX_COMBOS

    return str(min(len(options), ROOM_CARD_MAX_COMBOS) + 1)


def _is_build_your_own_reply(user_msg_clean: str, state: dict[str, Any]) -> bool:
    """Whether this message is the guest picking the card's build-your-own line.

    Every site that reads a digit off the rooms card has to agree on which digit
    this is, and they did not: the fast path below accepted any digit 1-5 and
    claimed this one, so the guest who tapped it was read as choosing an option
    that did not exist. Keycaps are normalised here because the card prints the
    line as "3️⃣" while a guest usually types "3".
    """
    from app.utils.room_card_reply import _normalize_keycaps

    number = _build_your_own_number(state)
    if number is None:
        return False
    # Lowercased here as well as by the caller: this reads two call sites, and
    # one of them must not be the only reason "Option 3" is recognised.
    msg = _normalize_keycaps(user_msg_clean or "").strip().lower()
    return msg in (
        number,
        f"{number}.",
        f"option {number}",
        f"option{number}",
        *_BUILD_YOUR_OWN_ALIASES,
    )


#: How a guest adds a second kind of room to what they already picked:
#: "and 1 ac room", "aur ek non ac", "+ 1 deluxe". Only a leading marker counts
#: — "and 4 adults" is a party change and must not be read as rooms.
_ADDITIVE_LEAD_RE = re.compile(
    r'^\s*(?:and|aur|plus|also|&|\+)\b[\s,:\-]*', re.IGNORECASE
)

#: Something in the message has to name a room; see `_additive_request_text`.
_ROOM_SIGNAL_RE = re.compile(
    r'\b(?:rooms?|kamra|kamre|ac|non[\s-]?ac|deluxe|delux|standard|standerd|'
    r'premier|premium|family|suite|bed|triple|tripple|double)\b',
    re.IGNORECASE,
)


def _selection_phrase(state: dict[str, Any]) -> str | None:
    """What the guest has already chosen, as a phrase a parser can read.

    Counted from `selected_rooms`, because that is the only field that knows
    how many rooms the guest is holding. `room_type_preference` is the fallback
    for a turn where the rooms were not built yet.
    """
    counts: dict[str, int] = {}
    for room in state.get("selected_rooms") or []:
        name = room.get("room_name") if isinstance(room, dict) else None
        name = (name or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    if counts:
        return " and ".join(f"{n} {name}" for name, n in counts.items())

    pref = (state.get("room_type_preference") or "").strip()
    if pref and _ROOM_SIGNAL_RE.search(pref):
        return f"1 {pref}"
    return None


def _additive_request_text(state: dict[str, Any]) -> str | None:
    """The whole mixed request, when this turn adds a room to the guest's pick.

    Guest 919088198242, 2026-10-19: they picked a Standard Non-AC room, then
    wrote "and 1 ac room" and "and ac delux room king bed". Both turns arrived
    as BOOK_ROOM, wiped the earlier pick and redrew the card, so the two halves
    of a split were never priced together. The card's own "tell me how you want
    to split" line works; nothing recognised a split the guest started
    unprompted.

    The earlier selection is put in front of the message with its count, so the
    custom-room reader sees both halves. It returns None for anything that is
    not that shape, and the caller then behaves exactly as before.
    """
    if not get_settings().AUTO_CUSTOM_ROOMS_FOR_ADDITIVE_ROOM:
        return None

    msg = (state.get("user_message") or "").strip()
    if not msg or not _ADDITIVE_LEAD_RE.match(msg):
        return None
    if not _ROOM_SIGNAL_RE.search(msg):
        return None

    prior = _selection_phrase(state)
    if not prior:
        return None
    # Drop the guest's own "and" — the join supplies one, and "and and 1 ac
    # room" is what the parser reads as noise.
    tail = _ADDITIVE_LEAD_RE.sub("", msg, count=1).strip()
    return f"{prior} and {tail}" if tail else None


def _deterministic_date_result(user_message: str, *, today=None, via: str = "outage_fallback"):
    """The regex date extractor, reachable through two narrow doors only.

    Door 1 (always): the model is unavailable and the guest has just been asked
    for dates. Losing the answer is worse than a parser that is occasionally
    wrong, and the bot has just asked the question.

    Door 2 (`DATES_FROM_MODEL_ONLY=False`): the pre-model short-circuit, kept as
    a rollback. Off by default.

    Date parsing on a healthy turn belongs to `understand_message` — the regex
    reassigned the wrong field whenever a guest corrected one date out of a pair
    (prod 2026-10-23 06:30:40 and 06:36:05).

    Deliberately narrower than the block it replaces: no one-date slot-shuffling
    heuristic (that is what moved a check-out into check-in), and it only fires
    when the extractor found a date in the guest's own words.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app.models.escalation_schemas import EscalationClassification
    from app.models.schemas import ExtractedBookingData, GeminiUnderstandingResponse
    from app.services.date_extractor import extract_stay_dates

    if today is None:
        # Same clock as `today_str()`, so the fallback and the model are told
        # the same "today". `today_str()` returns a string; parsing it back
        # keeps one source of truth for the date.
        today = datetime.strptime(today_str(), "%Y-%m-%d").date()
    found = extract_stay_dates(user_message, today=today)
    if not found.found:
        return None

    ci = found.check_in.strftime("%Y-%m-%d") if found.check_in else None
    co = found.check_out.strftime("%Y-%m-%d") if found.check_out else None
    logger.info(
        "DETERMINISTIC_DATES_%s | check_in=%s check_out=%s nights=%s",
        via.upper(),
        ci,
        co,
        found.nights,
    )
    return GeminiUnderstandingResponse(
        intent="CHECK_AVAILABILITY",
        extracted_data=ExtractedBookingData(check_in=ci, check_out=co),
        escalation=EscalationClassification(
            escalation_required=False,
            issue_type="NORMAL",
            reason_code="OTHER",
            urgency="LOW",
            sentiment="NEUTRAL",
            reason="normal_query",
            confidence=1.0,
        ),
    )


async def understand(state: ConversationState) -> ConversationState:
    """
    Call Gemini to understand the user's message.

    Extracts:
    - Intent (GREETING, BOOK_ROOM, SELECT_ROOM, etc.)
    - Booking entities (adults, children, dates, room type, etc.)

    Then merges extracted entities into state (null = not mentioned = no change).
    """
    # ── 1. PRIVACY SCRUBBER (DPDP Act 2023) ──
    from app.utils.privacy_scrubber import contains_id_number, scrub_id_numbers
    user_msg_raw = state.get("user_message", "")
    id_volunteered = contains_id_number(user_msg_raw)
    if id_volunteered:
        # Scrub message so Aadhaar / Passport number never reaches logs or persistent store
        state["user_message"] = scrub_id_numbers(user_msg_raw)
        logger.info("PRIVACY_SCRUBBER: Inbound Aadhaar/Passport pattern scrubbed from user message.")

    # Build session object for Gemini context (shared mapper)
    session = state_to_session(state, include_messages=True)

    # ── Extract AI config from hotel config (only AI-relevant fields) ──
    ai_config = None
    config_dict = state.get("hotel_config")
    if config_dict and isinstance(config_dict, dict):
        ai_raw = config_dict.get("ai_config")
        if ai_raw:
            from app.config.hotel_config import AIPromptConfig
            ai_config = AIPromptConfig(**ai_raw)

    user_msg_clean = state.get("user_message", "").strip().lower()
    stage = state.get("booking_stage", "new")
    has_options = bool(state.get("allocation_options"))
    active_prompt_action = get_active_prompt(state)
    logger.info(f"DEBUG understand | stage={stage} | booking_id={state.get('booking_id')} | user_msg={user_msg_clean}")

    # ── 1. GUARDS THAT END THE TURN WITHOUT A MODEL CALL (defect #25) ──
    # Each returns a finished turn or None; the order between them is the
    # precedence, and it lives here in one readable list instead of 300 lines
    # of nested branches.
    ended = fast_paths.opt_in_or_out(state, user_msg_clean)
    if ended is not None:
        return ended

    ended = fast_paths.fresh_start(state, user_msg_clean)
    if ended is not None:
        return ended


    ended = fast_paths.acknowledgment(state, user_msg_clean)
    if ended is not None:
        return ended


    ended = fast_paths.already_booked(state, user_msg_clean)
    if ended is not None:
        return ended

    # Before any selling: a guest asking for another hotel has the wrong
    # number, and every turn below this line would book them into ours.
    ended = fast_paths.other_hotel(state, user_msg_clean)
    if ended is not None:
        return ended


    # ── Extra Bed Preference ──
    # Three answers, not two: refuse, request, or say nothing about a bed. The
    # classifier changes nothing on "nothing", and that is the half that was
    # missing — until 2026-09-24 this block only ever wrote "NO", so a guest who
    # declined the bed once could never buy one back: `quote_generator.py:59`
    # waived the charge for the rest of the session, silently. The pattern for
    # "no" moved to `app/utils/extra_bed_intent.py` unchanged.
    from app.utils.extra_bed_intent import apply_to_state as apply_extra_bed
    apply_extra_bed(state, user_msg_clean)

    # ── Pre-extract Child Ages if present in message ──
    from app.utils.helpers import extract_child_ages_from_text
    early_child_ages = extract_child_ages_from_text(user_msg_clean)
    if early_child_ages:
        existing_ca = list(state.get("child_ages") or [])
        merged_ca = list(dict.fromkeys(existing_ca + early_child_ages))
        state["child_ages"] = merged_ca

    ended = fast_paths.bank_details_request(state, user_msg_clean)
    if ended is not None:
        return ended

    # ── ZERO-LLM FAST PATH: INTERACTIVE BUTTON TAPS ──
    #
    # The ladder itself lives in its own module (DEFECTS.md #25); it hands back
    # the reply and `_meal_plan_from_button`, the flag the extractor is held to
    # further down. `None` means "not a routed tap, carry on".
    _tap_reply, _meal_plan_from_button = handle_tap_routed(
        state=state,
        user_msg_clean=user_msg_clean,
        active_prompt_action=active_prompt_action,
    )
    if _tap_reply is not None:
        return _tap_reply

    # The five zero-LLM prompt handlers live in their own module now
    # (DEFECTS.md #25). `None` means the guest did not answer the open prompt,
    # which is exactly what running off the end of the block used to mean.
    _prompt_reply = handle_active_prompt(state, user_msg_clean, active_prompt_action)
    if _prompt_reply is not None:
        return _prompt_reply

    # ── S7.2: ZERO-LLM FAST PATH FOR ISOLATED MONTH FOLLOW-UP ──
    _MONTH_NAMES = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12
    }
    if user_msg_clean in _MONTH_NAMES:
        target_m = _MONTH_NAMES[user_msg_clean]
        ci = state.get("check_in")
        co = state.get("check_out")
        updates_month: dict[str, Any] = {"intent": "BOOK_ROOM"}
        if ci:
            try:
                dt_ci = datetime.datetime.strptime(ci, "%Y-%m-%d")
                updates_month["check_in"] = dt_ci.replace(month=target_m).strftime("%Y-%m-%d")
            except Exception:
                pass
        if co:
            try:
                dt_co = datetime.datetime.strptime(co, "%Y-%m-%d")
                updates_month["check_out"] = dt_co.replace(month=target_m).strftime("%Y-%m-%d")
            except Exception:
                pass
        if "check_in" in updates_month:
            logger.info("UNDERSTAND_FAST_PATH | Month follow-up applied (%s) -> check_in=%s, check_out=%s", user_msg_clean, updates_month.get("check_in"), updates_month.get("check_out"))
            return {
                **state,
                **updates_month,
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── S7.5: ZERO-LLM FAST PATH FOR PAYMENT RECEIPT / CLAIMS ──
    _PAYMENT_CLAIM_RE = re.compile(
        r"\b(?:payment\s*(?:done|ho\s*gaya|completed?|successful?|bhej\s*diya?|kar\s*diya?)"
        r"|paid|i\s*have\s*paid|maine\s*pay(?:ment)?\s*(?:bhi\s*)?(?:kar\s*diya|bhej\s*diya)|money\s*sent|advance\s*paid"
        r"|screenshot\s*(?:sent|attached|bheja|dekho|receipt)|check\s*payment|payment\s*receipt"
        r"|here\s*is\s*(?:the\s*)?(?:payment|receipt|screenshot)|transferred|utr\s*[:\s#]?[a-zA-Z0-9]+)\b",
        re.IGNORECASE,
    )
    if _PAYMENT_CLAIM_RE.search(user_msg_clean) and stage in ("awaiting_payment", "ready_for_payment", "quote_shown", "rooms_shown", "collecting"):
        logger.info("UNDERSTAND_FAST_PATH | Payment claim detected -> PAYMENT_STATUS")
        return {
            **state,
            "intent": "PAYMENT_STATUS",
            "booking_stage": stage,
            "next_action": "payment_followup",
            "bot_response": "Thank you! 🙏 If you've completed the payment, please share a screenshot of your payment receipt or UTR / reference number here so we can instantly verify and confirm your booking! 📲",
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── S7.8: ZERO-LLM FAST PATH FOR ROOM OPTION SELECTION ──
    # When options are on screen, digits (1, 2, 3), letters (A, B, C), or "Option X"
    # are selections, NEVER unknown messages.
    #
    # But the card's LAST line is a number too — "N. Tell me how you want to
    # split — I'll quote it" — and it is not a room option. This path claimed it
    # anyway, because the pattern below accepts any digit 1-5 and this block
    # runs before the build-your-own block that owns that line. A guest who
    # tapped it sent "option 3", which matched neither of the card's two
    # options, so the card came back with "reply with the option number (1 to
    # 2)" — a list that excludes the number they had just sent — and nothing
    # they could reply would get them out. Measured on a replay turn 2026-10-10.
    if (has_options or stage in ("rooms_shown", "quote_shown")) and not (
        _is_build_your_own_reply(user_msg_clean, state)
    ):
        from app.utils.room_card_reply import _normalize_keycaps
        user_msg_norm = _normalize_keycaps(user_msg_clean).strip()
        opt_choice_match = re.match(
            r'^\s*(?:(?:i\s+)?(?:want|choose|select|book|take)|give\s+me|option|opt|choice|selection|room|no\.?)?\s*([1-5]|[a-dA-D]|1[a-b]|2[a-b]|3[a-b]|4[a-b])\s*(?:please|pls|wala|wali)?\s*[.,!]?\s*$',
            user_msg_norm,
            re.IGNORECASE,
        )
        is_ordinal_choice = user_msg_norm in (
            "first", "second", "third", "first option", "second option", "third option",
            "1st", "2nd", "3rd", "last option", "pehla", "doosra", "teesra",
            "1st option", "2nd option", "3rd option", "pehla option", "doosra option",
        )
        is_superlative = any(w in user_msg_norm for w in ("cheapest", "sasta", "sabse sasta", "lowest price", "most comfortable", "luxury", "custom", "build your own"))

        is_room_name_match = False
        if not (opt_choice_match or is_ordinal_choice or is_superlative):
            opts = state.get("allocation_options") or []
            for o in opts[:3]:
                for r in o.get("rooms", []):
                    r_nm = str(r.get("room_name") or "").strip().lower()
                    if r_nm and (user_msg_norm == r_nm or (len(user_msg_norm) >= 4 and user_msg_norm in r_nm)):
                        is_room_name_match = True
                        break
                if is_room_name_match:
                    break

        if (opt_choice_match or is_ordinal_choice or is_superlative or is_room_name_match) and not any(w in user_msg_norm for w in ("photo", "discount", "offer", "coupon", "cancel")):
            logger.info("UNDERSTAND_FAST_PATH | Room option choice detected (%r) -> SELECT_ROOM", user_msg_clean)
            pref_val = state.get("user_message", "").strip()
            if opt_choice_match and opt_choice_match.group(1):
                pref_val = f"option {opt_choice_match.group(1)}"
            return {
                **state,
                "intent": "SELECT_ROOM",
                "room_type_preference": pref_val,
                "extracted_data": None,
                "extraction_result": None,
            }

    # The confirmed-reservation router lives in its own module (DEFECTS.md #25).
    # It hands back `state` because the ghost path replaces it with a demoted
    # copy and then falls through, so the rest of the node must use that one.
    _confirmed_reply, state = handle_confirmed_reservation(
        state,
        user_msg_clean,
        stage,
        _PAYMENT_CLAIM_RE,
        _is_confirmed_side_query,
    )
    if _confirmed_reply is not None:
        return _confirmed_reply

    # ── C1: ZERO-LLM FAST PATH FOR THE ROOT MENU'S TYPED WORDS ──
    # ONLY applies at root new session when no prompt or booking details are active
    #
    # The bare digits are gone as of 2026-10-15, and the words stay. Those digits numbered a
    # 1-6 text menu (`journey_templates/inquiry_templates.py:inq_greet`) that is registered
    # as `INQ_GREET` and called by nothing — the live greeting is the three-button
    # `greeting_response`, so no guest has been sent that menu for a long time. A cold "1"
    # was therefore being answered from a menu the bot had stopped sending, and it is not
    # even a settled answer: the same digit meant `SHOW_PRICES` here and `BOOK_ROOM` at the
    # second copy of this table further down, and a third copy in
    # `app/utils/intent_overrides.py` agreed with the first one.
    #
    # The words are real intents and are kept, because a guest who types "location" or
    # "photos" is asking for something the bot can do. What is dropped is only the shorthand
    # that existed to answer a menu nobody is shown; a guest typing "1" now reaches the model,
    # which already resolves "book a room" and "make a new booking" to the same intent.
    active_prompt = active_prompt_action in (
        "need_ac_preference",
        "need_extra_bed",
        "need_composition_preference",
        "need_subgroups",
        "need_child_ages",
        "need_composite_slots",
        "need_custom_rooms",
        "need_guest_name_for_payment",
        "need_change_dates",
        "need_change_guests",
        "need_meal_plan_preference",
        "change_menu_shown",
        "ask_booking_or_checkin",
        "payment_followup",
    )
    has_dates_or_guests = bool(state.get("check_in") or state.get("adults"))
    has_existing_booking = bool(state.get("booking_id") or (state.get("completed_bookings") and stage == "confirmed"))
    if stage == "new" and not has_options and not active_prompt and not has_dates_or_guests and not has_existing_booking:
        if user_msg_clean in ("hotel location & directions", "location", "directions"):
            return {
                **state,
                "intent": "SHOW_LOCATION",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("room photos & amenities", "photos", "amenities"):
            return {
                **state,
                "intent": "SHOW_AMENITIES",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("make a new booking", "new booking"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "collecting",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("check my existing booking", "my booking"):
            return {
                **state,
                "intent": "CHECK_EXISTING_BOOKING",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("room prices & availability", "price and availability"):
            return {
                **state,
                "intent": "SHOW_PRICES",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("something else", "other"):
            return {
                **state,
                "intent": "SOMETHING_ELSE",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── M3: 'change' KEYWORD NUMBERED MENU ──
    if user_msg_clean in ("change", "modify", "badlo", "update", "change booking", "modify booking", "change details", "update details") and (stage in ("rooms_shown", "quote_shown") or has_options):
        from app.prompts import templates
        return {
            **state,
            "intent": "SHOW_CHANGE_MENU",
            "next_action": "change_menu_shown",
            "updated_fields": [],
            # `lang` was missing here while every other template in this block passes it, so
            # the menu's Hinglish half was unreachable and a Hinglish guest got the English
            # menu. Passing it is the whole fix; the branch itself was always written.
            "bot_response": templates.change_options_menu(
                lang=state.get("preferred_language") or "English"
            ),
        }

    # The change menu and the meal-plan answer keep their own entry point: they
    # run after the confirmed block, so merging them into `handle_active_prompt`
    # would reorder them (DEFECTS.md #25).
    _menu_reply = handle_change_menu_prompt(state, user_msg_clean, active_prompt_action)
    if _menu_reply is not None:
        return _menu_reply

    # ── S4 CUSTOM / BUILD YOUR OWN OPTION SELECTION ──
    avail_opts = state.get("allocation_options", [])
    is_custom_selected = _is_build_your_own_reply(user_msg_clean, state)
    if is_custom_selected:
        from app.prompts import templates
        avail_names = []
        for opt in avail_opts:
            for r in opt.get("rooms", []):
                rn = r.get("room_name", "")
                short = rn.replace("ROOM", "").replace("Room", "").strip()
                if short and short not in avail_names:
                    avail_names.append(short)
        return {
            **state,
            "intent": "BOOK_ROOM",
            "next_action": "need_custom_rooms",
            "booking_stage": "collecting_custom_room",
            "room_type_not_available": False,
            "updated_fields": [],
            "bot_response": templates.ask_split_one_liner(avail_names),
        }

    # ── M3: STAY DATES COME FROM THE MODEL, AND ONLY THE MODEL ──
    #
    # A deterministic regex extractor (`app/services/date_extractor.py`) used to
    # run first here and, whenever it matched, the model was never asked what the
    # guest meant. It matched more than it understood:
    #
    #   "7th, 8th november"  -> check-in 8 Nov   (`_DM` only sees the day that
    #                                             carries the month name)
    #   "check out is 9th november" -> a *check-in* of 9 Nov, which the
    #                             one-date heuristic below then shuffled into
    #                             whatever slot was empty
    #
    # Prod 2026-10-23 06:28–06:36 is one guest correcting the same stay four
    # times in a row while the parser reassigned the wrong field each time
    # (`DETERMINISTIC_DATES_FOUND` at 06:30:40 and 06:36:05). Removed on
    # purpose: `understand_message` is now the only source of check_in and
    # check_out. The module stays where it is for its unit tests; nothing on
    # this path skips the model any more.
    from app.services.llm_errors import LLMUnavailableError

    user_msg_raw = state.get("user_message", "")

    # Call Gemini (SINGLE call for both intent + entities + escalation)
    # Only ai_config (tone, language, system prompt) is passed — never business config
    result = state.get("extraction_result")
    # Rollback gate — see `DATES_FROM_MODEL_ONLY` in settings. Off by default:
    # the model parses dates, and the regex only answers outages below.
    if not result and not get_settings().DATES_FROM_MODEL_ONLY:
        result = _deterministic_date_result(user_msg_raw, via="pre_model")
    if not result:
        try:
            result = await understand_message(
                user_message=user_msg_raw,
                session=session,
                today_str=today_str(),
                ai_config=ai_config,
                persona_prompt=state.get("persona_prompt"),
                hotel_rules=state.get("hotel_rules"),
            )
        except LLMUnavailableError as err:
            logger.warning("LLM_UNAVAILABLE in understand node | %s", err)
            # Outage-only fallback: keep a date the guest actually typed even
            # though the model is down. Returns None (→ the outage reply) for
            # every message that is not a date reply.
            result = _deterministic_date_result(user_msg_raw)
            if result is not None:
                logger.error(
                    "LLM_UNAVAILABLE_TURN | session=%s hotel=%s | answered from the "
                    "date fallback, model down: %s",
                    state.get("session_id"),
                    state.get("hotel_id"),
                    err,
                )
            else:
                from app.prompts.templates import text_card
                from app.services.ticket_service import get_formatted_contact_info

                # This is the one reply in the bot that must not need the model
                # in order to be said: it names the fault as ours and gives a
                # number that reaches a human, because the next turn is answered
                # by the same outage. `llm_unavailable` then keeps routing and
                # escalation off this turn — see routing.route_by_intent and
                # escalation_engine.
                contact = await get_formatted_contact_info(state.get("hotel_id") or 0)
                logger.error(
                    "LLM_UNAVAILABLE_TURN | session=%s hotel=%s | %s",
                    state.get("session_id"),
                    state.get("hotel_id"),
                    err,
                )
                degraded_text = (
                    "Sorry — I'm having trouble connecting to my assistant right now, and that's on my side, "
                    "not you. Your message has been noted; please try again in a few minutes.\n\n"
                    f"If it's urgent: {contact}"
                )
                current_failures = state.get("escalation_failure_count", 0)
                return {
                    **state,
                    "intent": "UNKNOWN",
                    "llm_unavailable": True,
                    "bot_response": text_card(degraded_text),
                    "escalation_failure_count": current_failures,
                    "escalation_required": False,
                    "escalation_issue_type": "NORMAL",
                    "escalation_sentiment": "NEUTRAL",
                    "escalation_reason": "infra_outage_suppressed",
                }

    # `is True`, not truthiness: tests hand this node a MagicMock, and any
    # `getattr` on one returns a truthy mock. Only a real bool means an outage.
    if getattr(result, "llm_unavailable", False) is True:
        # The model never saw this turn. Every field on `result` is a
        # placeholder, so the overrides, extraction and routing below would all
        # be reading an empty answer as if the model had judged it. Answer
        # honestly instead — this is the one reply in the bot that must not
        # need the model in order to be said. Same outage-only date fallback as
        # the exception path above.
        fallback = _deterministic_date_result(user_msg_raw)
        if fallback is not None:
            result = fallback
        else:
            from app.prompts.templates import text_card
            from app.services.ticket_service import get_formatted_contact_info

            contact = await get_formatted_contact_info(state.get("hotel_id") or 0)
            logger.error(
                "LLM_UNAVAILABLE_TURN | session=%s hotel=%s — replied with the outage message",
                state.get("session_id"),
                state.get("hotel_id"),
            )
            return {
                **state,
                "intent": "UNKNOWN",
                "next_action": "llm_unavailable",
                # The turn-scoped flag routing and the escalation suppressor read.
                "llm_unavailable": True,
                "extraction_result": None,
                "escalation_failure_count": state.get("escalation_failure_count", 0),
                "escalation_required": False,
                "escalation_issue_type": "NORMAL",
                "escalation_sentiment": "NEUTRAL",
                "escalation_reason": "infra_outage_suppressed",
                "bot_response": text_card(
                    "Sorry — I'm having trouble connecting to my assistant right now, and that's on my side, "
                    "not you. Your message has been noted; please try again in a few minutes.\n\n"
                    f"If it's urgent: {contact}"
                ),
            }

    logger.info(f"Intent: {result.intent} | Extracted: {result.extracted_data.model_dump(exclude_none=True)}")

    resolved_intent, override_faq, override_extras = apply_intent_overrides(
        user_message=state.get("user_message", ""),
        intent=result.intent,
        state=state,
        reset_booking=bool(getattr(result, "reset_booking", False)),
    )
    if resolved_intent != result.intent:
        logger.info(f"Intent override: {result.intent} -> {resolved_intent}")
    if override_extras.get("abandoned_draft"):
        logger.info("Draft booking cleared (reset_booking / decline continue)")

    updates: dict[str, Any] = {
        "intent": resolved_intent,
        "faq_question": override_faq or result.faq_question,
        **override_extras,
    }
    # ── STAY DATES RETURNED BY THE MODEL ──
    #
    # `menu_handler` parks a date prompt by setting `awaiting="stay_dates"`, and
    # a turn that answers it must clear the latch. Two guards on purpose:
    #
    # * The prompt carries the whole `CURRENT BOOKING STATE`, dates included
    #   (`gemini_service` builds it that way), so the model can echo dates the
    #   guest already gave while answering something else — e.g. "what name for
    #   the booking?". An echo must not silently drop the parked prompt, so the
    #   turn only counts when the model also classified it as booking-ish.
    # * A date the range validator rejects is not an answer either; the restore
    #   happens next to that rejection below.
    #
    # The failure counter is NOT reset here: it is computed once, further down,
    # from `state`, which would overwrite anything written at this point.
    _DATE_INTENTS = {
        "CHECK_AVAILABILITY",
        "BOOKING_DATES_PROVIDED",
        "BOOK_ROOM",
        "SELECT_ROOM",
        "CONFIRM_BOOKING",
        "MODIFY_BOOKING",
        "BOOKING_AMENDMENT",
    }
    _extracted_dates = getattr(result, "extracted_data", None)
    dates_returned = bool(
        _extracted_dates is not None
        and (
            getattr(_extracted_dates, "check_in", None)
            or getattr(_extracted_dates, "check_out", None)
        )
    )
    if dates_returned and resolved_intent in _DATE_INTENTS:
        updates["awaiting"] = None

    # ── PAYMENT-TURN VERDICT (LLM, with the regex as fallback) ───────────────
    #
    # Computed only on the turns where a payment gate will actually read it —
    # `route_by_intent` consults it under `intent == "CONFIRM_BOOKING"`, and
    # `validator` consults it in the `quote_shown` stage. Any other turn skips
    # the call entirely, so this costs one extra model round-trip on the handful
    # of turns that can reach the payment step and nothing on the rest.
    #
    # Always written, including as `None`. The key is declared in
    # `ConversationState` (an undeclared key would be silently dropped at the
    # node boundary) and classified EPHEMERAL in `session_contract` — writing it
    # unconditionally means a turn that did not classify cannot inherit the
    # previous turn's verdict from graph state.
    payment_verdict: str | None = None
    _stage_now = str(state.get("booking_stage") or "")
    if get_settings().PAYMENT_TURN_LLM_VERDICT and (
        resolved_intent == "CONFIRM_BOOKING" or _stage_now == "quote_shown"
    ):
        try:
            payment_verdict = await classify_payment_turn(
                state.get("user_message", ""),
                context_hint=f"stage={_stage_now or 'unknown'}",
            )
        except LLMUnavailableError:
            payment_verdict = None
        logger.info(
            f"PAYMENT_TURN_VERDICT | intent={resolved_intent} stage={_stage_now or '-'} "
            f"-> {payment_verdict}"
        )
    updates["payment_turn_verdict"] = payment_verdict

    # ── EXPECTED PROMPT ANSWERS OVERRIDE (V3) ──
    active_prompt_action = get_active_prompt(state)
    expected_action = None
    if active_prompt_action == "need_ac_preference":
        expected_action = "need_ac_preference"
    elif active_prompt_action == "need_composition_preference":
        expected_action = "need_composition_preference"
    elif active_prompt_action == "need_subgroups":
        expected_action = "need_subgroups"
    elif active_prompt_action == "need_child_ages" or ((state.get("children") or 0) > 0 and len(state.get("child_ages") or []) < int(state.get("children") or 0)):
        expected_action = "need_child_ages"
    elif state.get("booking_stage") == "collecting_custom_room" or active_prompt_action == "need_custom_rooms":
        expected_action = "need_custom_rooms"
    elif _additive_request_text(state):
        # The guest added a second room type to one they already picked. It is
        # the same request the custom-room prompt would have collected, so it
        # takes the same path — otherwise it lands as a preference change and
        # the earlier half is wiped. See `_additive_request_text`.
        expected_action = "need_custom_rooms"
    elif (
        state.get("booking_stage") == "quote_shown"
        and not state.get("guest_name")
        and not get_settings().NAME_AFTER_PAYMENT_ENABLED
    ):
        # With NAME_AFTER_PAYMENT_ENABLED the name is asked after payment, so a
        # quote-stage message is never a name answer — reading it as one is how
        # "ok" became a guest called "Ok".
        expected_action = "need_guest_name_for_payment"
    elif active_prompt_action == "need_guest_name_for_payment":
        expected_action = "need_guest_name_for_payment"
    elif active_prompt_action in ("group_booking_lead", "lead_created") or state.get("booking_stage") == "lead_created":
        expected_action = "group_booking_lead"
    elif active_prompt_action == "need_meal_plan_preference":
        expected_action = "need_meal_plan_preference"
    elif not state.get("check_in") or not state.get("check_out") or state.get("adults") is None:
        expected_action = "need_composite_slots"

    user_msg_clean = state.get("user_message", "").strip().lower()

    GREETING_WORDS = {
        "hi", "hello", "hey", "namaste", "hola", "good morning", 
        "good evening", "good afternoon", "good night", "hii", "helo", "hiii", 
        "start", "restart", "menu", "help", "sup", "yo",
        "good evening sir", "good morning sir", "good afternoon sir",
        "good evening madam", "good morning madam", "good afternoon madam",
        "good evening sir/madam", "good morning sir/madam", "good afternoon sir/madam",
        "good evening sir / madam", "good morning sir / madam", "good afternoon sir / madam",
        "hello sir", "hi sir", "namaste sir", "hello madam", "hi madam", "dear sir", "dear team",
        "sir", "madam", "sir/madam", "sir / madam"
    }
    NON_NAME_WORDS = GREETING_WORDS.union({
        "confirm", "yes", "no", "ok", "book", "book room", "1", "2", "3", "4", "5", "6",
        "ac", "non ac", "non-ac", "english", "hindi", "hinglish",
        "vacancy", "job", "jobs", "hiring", "resume", "cv", "apply", "interview",
        "manager", "front office", "receptionist", "staff", "salary", "document",
        "photo", "photos", "photoes", "foto", "pic", "pics", "pocs", "picture", "pictures",
        "image", "images", "img", "gallery", "dikhao", "bhejo", "dekhna",
        "thank", "thanks", "thank you", "thanku", "dhanyawad", "dhanyavad", "shukriya", "thx"
    })

    _PHOTO_KW_LIST = ("photo", "photos", "photoes", "foto", "pic", "pics", "pocs", "picture", "pictures", "image", "images", "img", "gallery", "tasveer")
    is_photo_query = (
        updates.get("intent") == "SHOW_AMENITIES"
        or any(pw in user_msg_clean for pw in _PHOTO_KW_LIST)
        or any(w in user_msg_clean for w in ("see rooms", "view rooms"))
    )

    is_inquiry_or_question = (
        "?" in state.get("user_message", "")
        or is_photo_query
        or any(user_msg_clean.startswith(q) for q in (
            "is there", "are there", "do you have", "can i", "can you", "what is", "how much", 
            "where is", "when is", "tell me", "i want to apply", "any vacancy", "vacancy for",
            "show me", "send me", "send", "show"
        ))
        or any(k in user_msg_clean for k in ("vacancy", "hiring", "resume", "curriculum vitae", "job application", "front office manager"))
    )

    if user_msg_clean in GREETING_WORDS:
        updates["intent"] = "GREETING"
        updates.pop("guest_name", None)

    # ── V3 Language Switch at any stage (Non-blocking) ──
    if user_msg_clean in ("hindi", "hinglish", "hindi me", "hindi mein", "hindi please"):
        updates["preferred_language"] = "Hinglish"
        updates["conversation_language"] = "Hindi"
    elif user_msg_clean in ("english", "english please", "in english"):
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"

    # ── V3 Root Menu Words — zero model spend ──
    # ONLY applies at root new session when no prompt or booking details are active
    #
    # The bare digits are gone, for the reason the table at C1 gives: they answered a 1-6
    # menu nothing sends. The words are kept.
    #
    # The "3" arm is the one to look at before changing anything here. It read
    # `BOOK_ROOM` with a comment asserting "MENU_03 routes to the same S2_SLOTS flow as
    # MENU_01", while the C1 table above and the third copy in `intent_overrides.py` both
    # read the same word as `SHOW_PRICES`. Two of the three, and the printed label — "Room
    # prices & availability" — say prices. It is NOT changed here: which of the two is
    # right is a behaviour call, not a refactor, and this commit only takes the digits away.
    stage = state.get("booking_stage", "new")
    has_options = bool(state.get("allocation_options"))
    has_dates_or_guests = bool(state.get("check_in") or state.get("adults"))
    if stage == "new" and not has_options and not expected_action and not has_dates_or_guests:
        if user_msg_clean in ("make a new booking", "new booking"):
            updates["intent"] = "BOOK_ROOM"
        elif user_msg_clean in ("check my existing booking", "my booking"):
            updates["intent"] = "CHECK_EXISTING_BOOKING"
        elif user_msg_clean in ("room prices & availability", "price and availability"):
            updates["intent"] = "BOOK_ROOM"
        elif user_msg_clean in ("hotel location & directions", "location", "directions"):
            updates["intent"] = "SHOW_LOCATION"
        elif user_msg_clean in ("room photos & amenities", "photos", "amenities"):
            updates["intent"] = "SHOW_AMENITIES"
        elif user_msg_clean in ("something else", "other"):
            updates["intent"] = "SOMETHING_ELSE"

    # ── S7: Lead guest name capture at quote confirmation ──
    is_side_intent = updates.get("intent") in (
        "SHOW_AMENITIES", "AMENITY_QUERY", "FAQ", "PRICE_MATCH", "SHOW_PRICES",
        "SHOW_LOCATION", "CHECK_EXISTING_BOOKING", "CANCEL_BOOKING", "MODIFY_BOOKING"
    )
    is_name_expected = (
        expected_action == "need_guest_name_for_payment"
        or (
            state.get("booking_stage") in ("quote_shown", "awaiting_payment")
            and not state.get("guest_name")
        )
        or (
            bool(state.get("selected_rooms") or state.get("total_price"))
            and not state.get("guest_name")
            and looks_like_a_name_answer(state.get("user_message") or "")
        )
    )
    # The name / expected-action ladder lives in its own module (DEFECTS.md
    # #25). It writes `updates` and awaits; it has nothing to return.
    await apply_expected_action_reply(
        state=state,
        updates=updates,
        user_msg_clean=user_msg_clean,
        expected_action=expected_action,
        is_name_expected=is_name_expected,
        is_inquiry_or_question=is_inquiry_or_question,
        is_side_intent=is_side_intent,
        GREETING_WORDS=GREETING_WORDS,
        NON_NAME_WORDS=NON_NAME_WORDS,
        additive_request_text=_additive_request_text,
        extract_room_request=extract_room_request,
    )


    # ── Extract escalation classification from Gemini ──────────
    esc = result.escalation.model_dump() if result.escalation else {}
    updates["escalation_required"] = esc.get("escalation_required", False)
    updates["escalation_issue_type"] = esc.get("issue_type", "NORMAL")
    updates["escalation_reason_code"] = esc.get("reason_code", "OTHER")
    updates["escalation_urgency"] = esc.get("urgency", "LOW")
    updates["escalation_sentiment"] = esc.get("sentiment", "NEUTRAL")
    updates["escalation_reason"] = esc.get("reason", "normal_query")
    updates["escalation_confidence"] = esc.get("confidence", 0.9)

    if esc.get("escalation_required"):
        logger.info(
            f"Escalation flagged: type={esc.get('issue_type')}, "
            f"reason_code={esc.get('reason_code')}, urgency={esc.get('urgency')}"
        )

    # ── Extract task classification from Gemini ────────────────
    task = result.task_classification or {}
    updates["task_is_request"] = task.get("is_task_request", False)
    updates["task_type"] = task.get("task_type", "OTHER")
    updates["task_subtype"] = task.get("subtype", "OTHER")
    updates["task_urgency"] = task.get("urgency", "NORMAL")
    updates["task_confidence"] = task.get("confidence", 0.0)
    updates["task_summary"] = task.get("summary", "")

    if task.get("is_task_request"):
        logger.info(
            f"Task request detected: type={task.get('task_type')}, "
            f"subtype={task.get('subtype')}, urgency={task.get('urgency')}"
        )

    # ── Extract room service extraction from Gemini ────────────
    updates["room_service_extraction"] = getattr(result, "room_service_extraction", None)

    # ── Extract amenity query from Gemini ─────────────────────
    if not updates.get("amenity_query"):
        updates["amenity_query"] = getattr(result, "amenity_query", None)

    # ── Extract explicit preferences unconditionally ─────────────
    ext = getattr(result, "extracted_data", None)
    if ext:
        if getattr(ext, "ac_preference", None) is not None:
            updates["ac_preference"] = ext.ac_preference
            feat_list = list(updates.get("feature_preferences") or state.get("feature_preferences") or [])
            if ext.ac_preference == "AC" and "AC" not in feat_list:
                feat_list.append("AC")
            elif ext.ac_preference == "NON_AC" and "Non-AC" not in feat_list:
                feat_list.append("Non-AC")
            updates["feature_preferences"] = feat_list
        if getattr(ext, "occupancy_preference", None) is not None:
            updates["occupancy_preference"] = ext.occupancy_preference
        if getattr(ext, "photo_tag_preference", None) is not None:
            updates["photo_tag_preference"] = ext.photo_tag_preference

    # ── Extract language tracking from Gemini ──────────────────
    # ── Continuous Language Detection & Adaptation ──────────────────

    user_msg_text = state.get("user_message", "").strip()
    user_msg_lower = user_msg_text.lower()
    
    # Deterministic lexical cues. The sets are module-level constants now — the
    # node used to rebuild both on every single message.
    _full_weights = get_settings().LANGUAGE_DETECTION_WEIGHTS_HINDI
    hinglish_cues = _HINGLISH_CUES_EXPANDED if _full_weights else _HINGLISH_CUES_LEGACY
    english_cues = _ENGLISH_CUES

    words = set(re.findall(r'\b\w+\b', user_msg_lower))
    h_count = len(words.intersection(hinglish_cues))
    e_count = len(words.intersection(english_cues))

    detected_lang = getattr(result, "detected_language", None)
    base_lang = getattr(result, "base_language", None)
    requested_lang = getattr(result, "requested_language", None)
    # The reply vocabulary of the room card is 1-4 and A-D, so "d" belongs in
    # this set with "a"/"b"/"c". Without it a bare "d" — the card's
    # "tell me how you want to split" — failed the subtraction below, was not
    # recognised as a short choice, and was read as a language signal instead.
    _SHORT_CHOICES = {
        "option", "opt", "choice", "selection", "room", "rooms",
        "1", "2", "3", "4", "5", "a", "b", "c", "d",
        "1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b",
    }
    is_numeric_or_short_choice = bool(re.match(r'^(?:option|opt|choice|selection)?\s*[0-9a-zA-Z\s.,]*$', user_msg_lower.strip())) and len(words - _SHORT_CHOICES) == 0

    # ── Which way the cues point ─────────────────────────────────────
    #
    # Product, 2026-09-30: *"detection of Hindi, Hinglish (Roman Hindi) is not
    # consistent, it still replies in english most of the time."*
    #
    # The old rule was `h_count >= e_count` for Hinglish and `e_count >
    # h_count` for English — a straight head-count. It misfired because the two
    # sides are not comparable: Hinglish borrows English *content* words
    # ("room", "book", "price", "family") while English borrows no Hindi, so a
    # single borrowed noun could outvote real Hindi. "2 adults aur 1 child" was
    # read as English; "room ki photo bhejo" scored zero and fell through to
    # the English default.
    #
    # The proportional rule encodes the asymmetry: a Hindi cue is strong
    # evidence, an English cue is weak evidence, so Hindi wins ties and
    # near-ties and English must clearly dominate to take the turn.
    #
    # The acknowledgement guard is separate and deliberate. "ok", "thank you"
    # and "done" carry one English cue each and no language intent at all —
    # under either rule they flipped an established Hinglish session to
    # English, which is the "keeps replying in English" complaint in one line.
    # A message made only of those words now changes nothing.
    _has_content = bool(words - _LANGUAGE_NEUTRAL_WORDS - _SHORT_CHOICES)
    _neutral_ack = _full_weights and not _has_content

    _has_devanagari = bool(re.search(r'[\u0900-\u097F]', user_msg_text))

    if _full_weights:
        _hindi_wins = _has_devanagari or (h_count > 0 and e_count <= h_count * 2)
        _english_wins = not _has_devanagari and (e_count > 0 and e_count > h_count * 2 and _has_content)
    else:
        _hindi_wins = _has_devanagari or (h_count > 0 and h_count >= e_count)
        _english_wins = not _has_devanagari and (e_count > 0 and e_count > h_count)

    if requested_lang or getattr(result, "language_switch_requested", False):
        target_style = requested_lang or detected_lang or "English"
        updates["preferred_language"] = "English" if "english" in target_style.lower() else "Hinglish"
        updates["conversation_language"] = "English" if "english" in target_style.lower() else "Hindi"
    elif is_numeric_or_short_choice and state.get("preferred_language"):
        updates["preferred_language"] = state["preferred_language"]
        updates["conversation_language"] = state.get("conversation_language", "English" if state["preferred_language"] == "English" else "Hindi")
    elif _hindi_wins:
        updates["preferred_language"] = "Hinglish"
        updates["conversation_language"] = "Hindi"
    elif _english_wins:
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"
    elif detected_lang and not _neutral_ack:
        # `_neutral_ack` blocks the model's read as well as the cue counts: a
        # bare "thank you" is not evidence of a language change, whichever
        # signal is asking. Guarded so that an explicit switch (branch 1) and
        # real Devanagari still get through — those carry content.
        if "hinglish" in detected_lang.lower() or "hindi" in detected_lang.lower() or (base_lang and "hindi" in base_lang.lower()):
            updates["preferred_language"] = "Hinglish"
            updates["conversation_language"] = "Hindi"
        elif "english" in detected_lang.lower() or (base_lang and "english" in base_lang.lower()):
            updates["preferred_language"] = "English"
            updates["conversation_language"] = "English"
    elif not state.get("preferred_language"):
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"


    # ── Progress and repetition failure counter management (Issue 3 Root Cause C) ──
    from app.services.repetition_guard import count_trailing_user_repeats

    user_msg = state.get("user_message", "")
    current_failures = state.get("escalation_failure_count", 0)
    trailing_user_repeats = count_trailing_user_repeats(
        user_msg,
        state.get("messages_history", []),
    )

    PROGRESS_SLOTS = (
        "check_in",
        "check_out",
        "adults",
        "children",
        "num_rooms",
        "selected_rooms",
        "guest_name",
        "customer_identifier",
        "room_type_preference",
        "occupancy_preference",
        "special_requests",
        "subgroups",
    )
    slot_progress = any(
        updates.get(slot) is not None and updates.get(slot) != state.get(slot)
        for slot in PROGRESS_SLOTS
    )
    stage_progress = bool(
        updates.get("booking_stage")
        and updates.get("booking_stage") != state.get("booking_stage")
    )
    action_progress = bool(
        updates.get("group_lead_submitted")
        or updates.get("amenity_query")
        or resolved_intent in ("FAQ", "POLICY", "HOTEL_TASK", "ROOM_SERVICE_ORDER")
    )

    # When the guest is repeating themselves (trailing_repeats >= 2), the conversation
    # is looping; re-extracting slots from the same text does not constitute progress.
    is_progress = (slot_progress or stage_progress or action_progress) and trailing_user_repeats < 2

    if trailing_user_repeats >= 3:
        # A guest saying the same thing 3 times is the clearest possible signal of failure.
        # Immediately set to >= 3 to trigger escalation in routing.py.
        updates["escalation_failure_count"] = max(current_failures + 1, 3)
        logger.info(
            "Guest repetition threshold reached (%s times) | msg=%r | failure_count=%s",
            trailing_user_repeats,
            user_msg,
            updates["escalation_failure_count"],
        )
    elif trailing_user_repeats == 2:
        # Guest repeated for the 2nd time -> increment failure count
        updates["escalation_failure_count"] = current_failures + 1
        logger.info(
            "Guest repetition detected (2nd time) | msg=%r | failure_count=%s",
            user_msg,
            updates["escalation_failure_count"],
        )
    elif resolved_intent == "UNKNOWN":
        if dates_returned:
            # The guest did give a date; only the intent label came back empty.
            # Before the deterministic branch was narrowed to outages this turn
            # was forced to CHECK_AVAILABILITY, so it never reached the counter.
            updates["escalation_failure_count"] = 0
            logger.info(
                "UNKNOWN intent but stay dates were extracted: counter reset to 0"
            )
        else:
            updates["escalation_failure_count"] = current_failures + 1
            logger.info(f"Comprehension failure (UNKNOWN intent): counter incremented to {current_failures + 1}")
    elif is_progress:
        updates["escalation_failure_count"] = 0
        logger.info("Progress detected in conversation: failure counter reset to 0")
    else:
        # Stalled turn without progress or repetition: preserve existing counter
        updates["escalation_failure_count"] = current_failures

    # Force intent to PAYMENT_STATUS if a payment screenshot is detected in the image analysis
    is_silent_upload = user_msg.startswith("[Analysis from ")

    if "[IMAGE_ANALYSIS]" in user_msg:

        import json

        if "Type: Payment Screenshot" in user_msg or "[Image Analysis: Payment Proof]" in user_msg:
            logger.info("Payment screenshot detected in user message. Forcing intent to PAYMENT_STATUS.")
            updates["intent"] = "PAYMENT_STATUS"
                
        elif "Type: Room Image" in user_msg:
            if not is_silent_upload:
                logger.info("Room image detected. Forcing intent to ROOM_FEATURE_LOOKUP.")
                updates["intent"] = "ROOM_FEATURE_LOOKUP"
                updates["escalation_required"] = False
            else:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"
                
        elif "Type: Competitor Price Match" in user_msg:
            if not is_silent_upload:
                logger.info("Competitor price match screenshot detected. Forcing intent to PRICE_MATCH.")
                updates["intent"] = "PRICE_MATCH"
            else:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"
                
            details_match = re.search(r"Details:\s*(.*?)\s*\[/IMAGE_ANALYSIS\]", user_msg, re.DOTALL)
            if details_match:
                try:
                    updates["price_match_data"] = json.loads(details_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON from price match details")
                    
        elif "Type: Invalid Price Match" in user_msg:
            if not is_silent_upload:
                logger.info("Invalid price match document detected.")
                updates["intent"] = "PRICE_MATCH"
            else:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"
                
            details = "Uploaded image is not related to a hotel booking."
            details_match = re.search(r"Details:\s*(.*)", user_msg)
            if details_match:
                details = details_match.group(1).strip()
            updates["price_match_data"] = {"invalid": True, "reason": details}
            
        else:
            if is_silent_upload:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"

    # Detect in-chat discount / price negotiation requests
    user_lower = (state.get("user_message") or "").lower()
    # The pattern moved to `app/utils/competitor_mention.py` on 2026-10-14, when
    # `price_match_handler` needed the same answer to decide whether it may
    # compare our rate to a portal's. Moved verbatim: the behaviour here is
    # unchanged, and a test pins each alternation.
    explicit_competitor = mentions_competitor(user_lower)

    user_booking_stage = state.get("booking_stage", "new")
    has_active_quote = (
        user_booking_stage in ("quote_shown", "rooms_shown", "rooms_selected", "awaiting_payment")
        or float(state.get("total_price") or 0.0) > 0
        or bool(state.get("allocation_options"))
        or bool(state.get("selected_rooms"))
    )

    # ── A booking act outranks a price phrase ─────────────────────────────────
    # The money alternation in `QUOTE_NEGOTIATION_PATTERNS` matches a *mention* of
    # an amount, and a guest who is cancelling, amending or paying almost always
    # mentions one — "advance of Rs. 3000", "refund of Rs. 3000", the "₹3,000" on a
    # receipt. On 2026-10-16 that read a cancellation as a discount demand and
    # answered it with the direct-booking pitch, so the guest was never asked
    # whether he wanted to cancel. The veto is on the act, not on the phrase,
    # because the phrase side cannot be enumerated — see `_booking_act_intent`.
    _booking_act = _booking_act_intent(updates.get("intent"), result.intent)

    if _booking_act:
        logger.info(
            f"Price/discount override skipped: turn is a booking act ({_booking_act})."
        )
        # Only when nothing has claimed the turn yet. If a word-list above already
        # wrote an intent, the veto stands aside rather than adjudicating between
        # two answers neither of which it can see the message for. `next_action` is
        # deliberately left alone: this guard exists to stop the override, not to
        # re-drive the flow.
        if updates.get("intent") is None:
            updates["intent"] = _booking_act
    elif explicit_competitor:
        logger.info("Explicit competitor price match inquiry detected. Mapping to PRICE_MATCH.")
        updates["intent"] = "PRICE_MATCH"
        updates["next_action"] = None
    elif has_active_quote and any(re.search(pat, user_lower) for pat in QUOTE_NEGOTIATION_PATTERNS):
        has_specific_room_request = bool(re.search(r'\b\d+\s*(?:deluxe|standard|suite|family|classic|super|triple|tripple|king|queen|ac|non[\s-]?ac)\b', user_lower))
        if user_booking_stage not in ("confirmed", "cancelled") and not has_specific_room_request:
            logger.info("Direct quote negotiation / discount inquiry detected on active quote/options. Mapping to PRICE_MATCH.")
            updates["intent"] = "PRICE_MATCH"
            updates["next_action"] = None
            updates["payment_missing_fields"] = []
    elif not has_active_quote and (updates.get("intent") == "PRICE_MATCH" or result.intent == "PRICE_MATCH"):
        # If user explicitly asked about discount/offers without dates yet, keep as PRICE_MATCH
        # so price_match_handler can explain direct booking benefits!
        is_discount_inquiry = any(re.search(pat, user_lower) for pat in QUOTE_NEGOTIATION_PATTERNS)
        if is_discount_inquiry:
            logger.info("General discount/offer inquiry without active quote -> keeping PRICE_MATCH to show benefits.")
            updates["intent"] = "PRICE_MATCH"
            updates["next_action"] = None
        elif state.get("check_in") or state.get("adults"):
            logger.info("General inquiry with dates misclassified as PRICE_MATCH -> routing to BOOK_ROOM.")
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        else:
            logger.info("General rate inquiry misclassified as PRICE_MATCH -> routing to SHOW_PRICES.")
            updates["intent"] = "SHOW_PRICES"
            updates["next_action"] = None

    # Force intent to PAYMENT_STATUS if we are in a payment follow-up flow and user is answering payment questions
    active_missing = state.get("payment_missing_fields") or []
    current_intent = updates.get("intent", result.intent)
    non_payment_intents = (
        "GREETING", "CANCEL_BOOKING", "FAQ", "HOTEL_TASK", "COMPLAINT",
        "HUMAN_REQUEST", "PRICE_MATCH", "MODIFY_BOOKING", "BOOK_ROOM",
        "CHECK_AVAILABILITY", "LIFECYCLE_RESET"
    )
    is_opt_out = bool(re.search(r"^\s*(stop|halt|quit|unsubscribe|cancel|ruko|ruk\s*jao|chhod\s*do|rahne\s*do|rehne\s*do)\s*[.!]?$", user_lower))
    if is_opt_out:
        updates["payment_missing_fields"] = []
    elif active_missing and current_intent not in non_payment_intents:
        logger.info(f"Payment follow-up active (missing: {active_missing}). Forcing intent to PAYMENT_STATUS.")
        updates["intent"] = "PAYMENT_STATUS"

        # Merge follow-up answers into existing payment_details
        extracted = result.extracted_data
        existing_pd = dict(state.get("payment_details") or {})

        # Map top-level extracted fields to payment_details fields
        field_mapping = {
            "guest_name": "customer_name",
            "guest_phone": "customer_phone",
            "guest_email": "customer_email",
            "adults": "num_adults",
            "children": "num_children",
            "num_rooms": "num_rooms",
            "check_in": "check_in",
            "check_out": "check_out",
        }

        # Pull from payment_details sub-object first
        if extracted.payment_details:
            pd_dict = extracted.payment_details.model_dump(exclude_none=True)
            for k, v in pd_dict.items():
                if v is not None:
                    existing_pd[k] = v

        # Also pull from top-level extracted fields
        extracted_dict = extracted.model_dump(exclude_none=True)
        for top_field, pd_field in field_mapping.items():
            if top_field in extracted_dict and extracted_dict[top_field] is not None:
                existing_pd[pd_field] = extracted_dict[top_field]

        updates["payment_details"] = existing_pd

        # Recalculate which fields are still missing
        from app.services.image_vision import CRITICAL_FIELDS, IMPORTANT_FIELDS
        still_missing = []
        for f in active_missing:
            val = existing_pd.get(f)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                # Check if we can fill from session context
                session_map = {"check_in": "check_in", "check_out": "check_out",
                               "num_adults": "adults", "num_children": "children",
                               "num_rooms": "num_rooms", "customer_name": "guest_name",
                               "customer_phone": "guest_phone", "customer_email": "guest_email"}
                session_field = session_map.get(f)
                session_val = state.get(session_field) if session_field else None
                if session_val is not None:
                    existing_pd[f] = session_val
                else:
                    still_missing.append(f)
        updates["payment_missing_fields"] = still_missing if still_missing else None
        updates["payment_details"] = existing_pd

    # If the user is in the middle of a cancellation request, do not parse booking parameter changes
    is_cancellation = state.get("booking_stage") == "cancellation_pending"

    # ── CRITICAL: Only modify booking state for BOOKING-RELATED intents ──
    # Non-booking intents (FAQ, ROOM_FEATURE_LOOKUP, SHOW_AMENITIES, etc.)
    # should NEVER wipe dates, guests, rooms, or booking progress.
    # Also skip when we just cleared the draft (reset / decline continue).
    booking_intents = {
        "BOOK_ROOM", "MODIFY_BOOKING", "SELECT_ROOM", "CONFIRM_BOOKING",
        "CHECK_AVAILABILITY", "CANCEL_BOOKING", "PAYMENT_STATUS", "PRICE_MATCH", "BOOKING_SUMMARY",
        "BOOKING_DATES_PROVIDED",
    }
    
    draft_cleared = bool(override_extras.get("abandoned_draft"))
    if draft_cleared:
        # ── MULTI-BOOKING QUEUE CONTINUATION (ABANDONED) ──
        from app.utils.queue_manager import advance_booking_queue
        q_updates = advance_booking_queue(state, "abandoned")
        if q_updates.get("queue_advanced"):
            # Merge the new active booking into override_extras so it gets loaded instead of staying cleared
            for k, v in q_updates.items():
                override_extras[k] = v
            override_extras["abandoned_draft"] = False # Un-abandon it since we are continuing!
            draft_cleared = False
            updates["multi_booking_detected"] = True # Force response_generator to say something
            
    is_booking_intent = (
        not draft_cleared
        and (resolved_intent in booking_intents or result.intent in booking_intents)
    )

    if not is_cancellation and is_booking_intent:
        apply_booking_intent(
            state=state,
            updates=updates,
            result=result,
            user_msg_clean=user_msg_clean,
            active_prompt_action=active_prompt_action,
            expected_action=expected_action,
            is_inquiry_or_question=is_inquiry_or_question,
            _full_weights=_full_weights,
            _meal_plan_from_button=_meal_plan_from_button,
            NON_NAME_WORDS=NON_NAME_WORDS,
            resolved_intent=resolved_intent,
        )
    else:
        updates["updated_fields"] = []
        # For non-booking intents, still extract photo_tag_preference if present
        if not is_cancellation:
            extracted = result.extracted_data
            if getattr(extracted, "photo_tag_preference", None) is not None:
                updates["photo_tag_preference"] = extracted.photo_tag_preference
            # Keep room type + nights context for pricing summaries
            if resolved_intent == "BOOKING_SUMMARY":
                if extracted.room_type_preference is not None:
                    updates["room_type_preference"] = extracted.room_type_preference
                if extracted.num_rooms is not None:
                    updates["num_rooms"] = extracted.num_rooms
                    updates["pricing_query_num_rooms"] = extracted.num_rooms
                if not updates.get("pricing_query_nights"):
                    from app.utils.intent_overrides import extract_nights_from_message
                    nights = extract_nights_from_message(state.get("user_message", ""))
                    if nights:
                        updates["pricing_query_nights"] = nights
                if not updates.get("pricing_query_num_rooms"):
                    from app.utils.intent_overrides import extract_num_rooms_from_message
                    num_rooms = extract_num_rooms_from_message(state.get("user_message", ""))
                    if num_rooms:
                        updates["pricing_query_num_rooms"] = num_rooms
                        updates["num_rooms"] = num_rooms
    # A turn may not replace a valid future stay with an impossible one. Guest
    # 917376603689 typed "2" (guests) and the state's 03–05 Nov became
    # (None, 2024-11-05); every tap after that was answered "dates sahi nahi
    # lag rahi" against a card that had just offered rooms. See
    # `app/utils/date_sanity.py` for the full measurement and the two limits
    # that keep this from overruling a guest who is actually changing dates.
    from app.utils.date_sanity import drop_impossible_dates
    drop_impossible_dates(updates, state, user_msg_clean)

    # Photos plus dates in one message = a booking
    is_photo_req = any(pw in user_msg_clean for pw in _PHOTO_KW_LIST) or any(w in user_msg_clean for w in ("see rooms", "view rooms"))
    # The dates must be *this message's* — "send me photos for 5-7 Nov" is a
    # booking, "send me photos of the deluxe room" is a photo request.
    # Reading `state.get("check_in")` made every photo request from a guest who
    # already had a card on screen look like the second kind: measured
    # 2026-11-05 (`week-sahu-photo-request-repeats`), "Please send Pic of Delux
    # Room King Bed" was turned into BOOK_ROOM + a forced new search, so the
    # guest got the same room card three times and was never told about the
    # photo. The stay they are looking at is not a date they just gave.
    has_dates_for_photo = bool(updates.get("check_in") or updates.get("check_out"))
    if is_photo_req:
        updates["wants_photos"] = True
        if has_dates_for_photo:
            updates["intent"] = "BOOK_ROOM"
            updates["force_new_search"] = True
            updates["next_action"] = "search"
            updates["booking_stage"] = "collecting"

    if updates.get("wants_photos") and not state.get("selected_rooms") and not state.get("allocation_options"):
        updates["force_new_search"] = True
        updates["next_action"] = "search"
        updates["booking_stage"] = "collecting"
        updates["allocation_options"] = []
        updates["selected_rooms"] = []

    # ── The guest's own clock ──
    # The arrival or departure time they volunteered, if they volunteered one.
    # Deliberately read from `user_message` here and then *persisted* onto the
    # session: `user_message` is turn-scoped, so a guest who says "I'll reach at
    # 7am" and picks a room two turns later must still be priced for the 7am.
    #
    # Never asked for. If a guest never states a time, no early/late fee is ever
    # quoted — which is the whole point, because a misread time is a wrong
    # number on someone's bill.
    #
    # Written after every extraction branch and before the funnel, so it lands
    # in `updates` like any other field and survives `validate_updates` (these
    # two keys are not governed fields, so the contract leaves them alone).
    from app.utils.guest_time import extract_requested_times

    _guest_times = extract_requested_times(state.get("user_message"))
    if _guest_times.get("arrival"):
        updates["requested_arrival_time"] = _guest_times["arrival"]
    if _guest_times.get("departure"):
        updates["requested_departure_time"] = _guest_times["departure"]

    return fast_paths.finalise(state, updates)