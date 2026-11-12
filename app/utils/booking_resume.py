"""
Mid-booking escape hatch helpers.

When a guest asks a side question during booking collection
(e.g. "what's check-out time?"), answer it and resume with the
*actual* missing booking field — never blindly re-ask guest count.
"""

from __future__ import annotations

import re
from typing import Any


# Clear hotel-policy / amenity / info side questions
_SIDE_QUESTION = re.compile(
    r"\b("
    r"check[- ]?in\s+time|check[- ]?out\s+time|checkout\s+time|checkin\s+time"
    r"|what\s+time\b|by\s+the\s+way|quick\s+question"
    r"|parking|wifi|wi[- ]?fi|pool|gym|spa|pets?|smoking|elevator|lift"
    r"|cancellation\s+polic|refund\s+polic|late\s+check[- ]?out|early\s+check[- ]?in"
    r"|do\s+you\s+(have|allow|provide|offer)|is\s+there|are\s+there"
    r"|where\s+(is|are)|how\s+far|distance|airport|location|address|directions?"
    r"|amenities|facilities|breakfast\s+time|dinner\s+time|lunch\s+time"
    r"|id\s+proof|documents?\s+required|payment\s+methods?"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_LEAD = re.compile(
    r"^\s*(what|when|where|how|why|which|do\s+you|does\s+it|is\s+there|are\s+there|"
    r"can\s+i|could\s+i|by\s+the\s+way|also|one\s+more\s+thing|quick\s+q)",
    re.IGNORECASE,
)

_RELATIVE_DATE_FOR_AVAIL = re.compile(
    r"\b(this\s+weekend|next\s+weekend|coming\s+weekend"
    r"|this\s+saturday|next\s+saturday|coming\s+saturday"
    r"|this\s+sunday|next\s+sunday|coming\s+sunday"
    r"|this\s+friday|next\s+friday|coming\s+friday"
    r"|tomorrow|day\s+after\s+tomorrow|tonight)\b",
    re.IGNORECASE,
)

_AVAIL_ACTION = re.compile(
    r"\b(available|availability|vacancy|vacant|any\s+rooms?|rooms?\s+available)\b",
    re.IGNORECASE,
)

# Short answers that are clearly filling a booking slot — never escape these
_SLOT_ANSWER = re.compile(
    r"^("
    r"\d{1,2}(\s*(adults?|children|child|kids?|guests?|people|pax|persons?|rooms?))?"
    r"|(\d{1,2}\s*)(\+|and|&|/)\s*\d{1,2}(\s*(adults?|children|kids?))?"
    r"|option\s*\d+|opt\s*\d+|\d{1,2}"
    r"|couple|solo|just\s+me|only\s+me|family\s+of\s+\d+"
    r"|yes|yeah|yep|yup|confirm|ok|okay|sure|proceed|go\s+ahead|haan|han|theek(\s+hai|\s+h)?|thik(\s+hai|\s+h)?|done|pakka"
    r"|tomorrow|today|tonight"
    r"|\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?"
    r"|\d{4}-\d{2}-\d{2}"
    r")([\s!.]*)?$",
    re.IGNORECASE,
)

_ACTIVE_STAGES = frozenset({
    "collecting", "rooms_shown", "quote_shown", "awaiting_payment", "new",
    "payment_pending", "payment_check", "id_verification",
})

_DRAFT_STAGES = frozenset({
    "collecting", "rooms_shown", "quote_shown", "awaiting_payment", "ready_for_payment",
})

# Explicit abandon phrases kept only as a thin safety net; Gemini owns natural language.
# Prefer short declines after a continue/confirm prompt (regex), plus Gemini `reset_booking`.
_ABANDON_SHORT = re.compile(
    r"^\s*(no|nope|nah|no\s+thanks?|don'?t|do\s+not|not\s+now|cancel|stop|halt|quit|unsubscribe|ruko|ruk\s*jao|chhod\s*do|rahne\s*do|rehne\s*do)\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Explicit "start fresh / new booking" — triggers the confirmation flow, NOT immediate clear.
#
# Every alternative here must name what is being restarted. The verb-only forms
# used to carry an optional suffix — `stop(...)?`, `cancel(...)?`, `reset(...)?`,
# `restart\s+(booking)?`, `naya(...)?`, `phir\s+se\s*(karo)?` — so the bare word
# matched *anywhere* in the message. "stop by the hotel at 5" and "can I cancel
# my spa booking" both opened the start-fresh prompt mid-booking. The object is
# now required, and `forget` / `clear` must name one too, so "forget the towels"
# is no longer read as starting over.
#
# Bare "cancel" is deliberately NOT here. It still reaches
# `wants_to_abandon_in_progress_booking` (anchored, and only after the bot has
# actually asked whether to continue) and the CANCEL_BOOKING intent; widening
# this pattern to catch it is what caused the misfires.
_START_FRESH_PHRASES = re.compile(
    r"\b("
    r"start\s+(fresh|over|again|a\s+new)|new\s+booking|new\s+reservation|new\s+conversation"
    r"|forget\s+(previous\s+|my\s+|the\s+|this\s+|last\s+|old\s+)?(booking|reservation|it)"
    r"|restart\s+(booking|reservation)"
    r"|clear\s+(the\s+|my\s+)?(booking|reservation|all|everything|draft)"
    r"|reset\s+(booking|reservation|all|kar\s+do)"
    r"|discard\s+(booking|draft|all)"
    r"|book\s+another\s+(room|stay|booking|hotel)"
    r"|make\s+another\s+(booking|reservation)"
    r"|another\s+booking|different\s+booking"
    r"|fresh\s+start|sab\s+clear|naya(\s+se)?\s+(shuru|booking)|cancel\s+this\s+and\s+start|phir\s+se\s+(karo|shuru)"
    r"|stop\s+(booking|reservation|this)"
    r"|cancel\s+(this|booking|reservation|kar\s+do)"
    r")\b",
    re.IGNORECASE,
)

# Only a prompt that actually asks about CONTINUING THE DRAFT counts.
#
# This pattern used to contain a bare `continue`. Every room-options card ends
# with "👉 Reply *1*, *2*, *3*, or *4* to continue", so that alternative matched
# the footer of every room list — and, measured across the production store on
# 2026-09-16, 491 of 4306 (11.4%) of all assistant messages. Because
# last_bot_asked_to_continue() scans the last 8 stored messages, one room list
# armed wants_to_abandon_in_progress_booking() for the next 8 turns: any bare
# "No" / "nope" / "cancel" in that window then wiped the whole booking draft and
# dumped the guest on the welcome menu. That is defect #9 — Chat A 06:38:48
# "No" -> 06:38:55 the full welcome menu, dates and guests gone.
#
# The alternatives below each name the draft explicitly, so a "reply 1 to
# continue" footer cannot arm it. Markdown is stripped before matching (see
# _strip_emphasis), which is why these can be written as plain words.
_CONTINUE_PROMPT_HINT = re.compile(
    r"continue\s+(?:your|the|current|this|my)\s+(?:\w+\s+)?(?:booking|reservation|session|draft)"
    r"|(?:say|reply|type|send|write)\s+continue\b"
    r"|continue\s+(?:bolo|boliye|likho|likhiye|karo|kar\s*do|kijiye)\b"
    r"|pick\s+it\s+up"
    r"|booking\s+in\s+progress"
    r"|booking\s+abhi\s+chal\s+rahi"
    r"|would\s+you\s+like\s+to\s+proceed"
    r"|confirm\s+this\s+booking"
    r"|proceed\s+and\s+confirm"
    r"|whenever\s+you.?re\s+ready.*confirm",
    re.IGNORECASE,
)

_EMPHASIS = re.compile(r"[*_`]")


def _strip_emphasis(text: str) -> str:
    """Drop WhatsApp markdown so the phrase patterns match plain words."""
    return _EMPHASIS.sub("", text)


# ── What a reset must NOT touch ───────────────────────────────────────────
#
# This wipe used to be a hand-written list of 41 keys, and it drifted away from
# the session it was supposed to clear. Measured on the live bot 2026-09-17,
# right after "start fresh" -> "Yes, start fresh", these were still set:
#
#   ac_preference, ac_qualified, composite_template_shown, effective_adults,
#   effective_children, child_ages, preferred_language, conversation_language,
#   quote_presented, room_options_all, declined_upsells, applied_coupon, ...
#
# Two of them changed what the guest saw on the next booking. `search_rooms`
# skips the AC question once `ac_qualified` is true, so the new booking never
# asked it; and room sizing prefers `effective_adults` over `adults`, so the old
# party size quietly carried over. A guest who reset was not a new guest.
#
# So the list is now derived from the session contract (`PERSISTED`): a field
# added to `BookingSession` is reset without anyone remembering to add it here.
# Every field resets to its **pydantic default**, which is not a coincidence —
# `state_to_session` builds a fresh `BookingSession` from graph state every
# turn, so a field *absent* from state was already being written back as its
# default. The default is exactly "as if this field had never been set".
#
# Only the fields named below survive a reset. Everything else goes.
_KEEP_ON_RESET: dict[str, str] = {
    # ── identity: without these the session cannot find its own document ──
    "session_id": "the conversation document's own key",
    "hotel_id": "which hotel this session belongs to",
    "customer_identifier": "the guest's phone — the key the session is looked up by",
    # ── the hotel's persona, not this guest's draft ──
    "active_personality_mode": "chosen by the hotel, not by this booking",
    "personality_mode_switched_at": "when that persona was chosen",
    # ── other bookings the guest asked for in the SAME message ──
    # "2 rooms, one for each family" produces a queue. A reset ends the draft
    # being built, not the ones still waiting behind it.
    "completed_bookings": "bookings from the same message that are already confirmed",
    "booking_queue": "the bookings from the same message still to be made",
    "queue_id": "the id of that queue",
    "active_booking_index": "which entry of the queue the cleared draft was",
    "failed_bookings": "the record of earlier attempts in the same message",
    # ── handled separately, at the bottom of this function ──
    "conversation_overrides": (
        "clear_overrides_in_state ends staff *decisions* at the booking "
        "boundary without ending a human takeover"
    ),
}


def clear_in_progress_booking_fields(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Every field a brand-new session would not have. Identity is kept.

    Derived from the session contract rather than typed out; `_KEEP_ON_RESET`
    explains why the hand-written version was the bug.
    """
    from app.models.schemas import BookingSession
    from app.models.session_contract import PERSISTED

    cleared: dict[str, Any] = {}
    for name in sorted(PERSISTED - set(_KEEP_ON_RESET)):
        field = BookingSession.model_fields.get(name)
        if field is None:  # contract drift; the contract test catches it
            continue
        if field.default_factory is not None:
            cleared[name] = field.default_factory()
        elif not field.is_required():
            cleared[name] = field.default
        else:
            # A required field with no default. There is no sane value to reset
            # it to, and guessing None would blank something the session cannot
            # work without — so fail here, loudly, where a test will see it,
            # rather than silently corrupt the session of a real guest.
            # Add the field to _KEEP_ON_RESET if it should survive a reset.
            raise RuntimeError(
                f"{name} is a required BookingSession field with no default; "
                "add it to _KEEP_ON_RESET with a reason, or give it a default"
            )

    cleared.update({
        # an enum on the model, a str on the wire
        "booking_stage": "new",
        # a one-turn signal: response_generator reads it to speak the
        # new-guest greeting, then clears it (response_generator.py:406)
        "abandoned_draft": True,
        # ── truncate the transcript ──
        # `messages_history` is a projection rebuilt from the stored session on
        # every load, so clearing it here only reaches the database because
        # save_state runs later in this same turn and rebuilds `messages` from
        # it. Without this the previous booking's turns stay in front of the
        # model — and `last_bot_asked_to_continue()` scans the last 8 messages,
        # so a leftover "booking in progress" prompt can arm the abandon guard
        # against the NEXT booking, wiping it on a bare "no".
        "messages_history": [],
    })

    # Staff decisions must not leak into the next booking attempt
    from app.services.override_resolver import clear_overrides_in_state
    if state is not None:
        cleared.update(clear_overrides_in_state(state, reason="start_fresh"))
    else:
        from app.models.conversation_overrides import ConversationOverrides
        cov = ConversationOverrides()
        cov.clear_decision_overrides(reason="start_fresh")
        cleared["conversation_overrides"] = cov.model_dump(mode="json")
    return cleared



def last_bot_asked_to_continue(state: dict[str, Any]) -> bool:
    """True if a recent bot message asked to continue / confirm the draft."""
    hist = state.get("messages_history") or []
    for msg in reversed(hist[-8:]):
        role = (msg.get("role") or msg.get("sender") or "").lower()
        if role not in ("assistant", "bot", "model", "ai"):
            # Also check text cards without role
            if role and role in ("user", "human", "customer"):
                continue
        text = msg.get("content") or msg.get("text") or msg.get("message") or ""
        if isinstance(text, list):
            text = " ".join(
                (c.get("text") if isinstance(c, dict) else str(c)) for c in text
            )
        if _CONTINUE_PROMPT_HINT.search(_strip_emphasis(str(text))):
            return True
    # Soft footnote / quote stage also counts as continue context
    stage = state.get("booking_stage", "new")
    if stage in ("quote_shown", "awaiting_payment", "ready_for_payment"):
        return True
    return False


def has_in_progress_draft(state: dict[str, Any]) -> bool:
    """True when there is draft booking progress worth clearing."""
    stage = state.get("booking_stage", "new")
    if stage in ("cancellation_pending", "confirmed", "cancelled"):
        return False
    return bool(
        state.get("check_in")
        or state.get("check_out")
        or state.get("selected_rooms")
        or state.get("allocation_options")
        or state.get("total_price")
        or state.get("payment_link")
        or (stage in _DRAFT_STAGES)
    )


def needs_start_fresh_confirmation(
    user_message: str,
    state: dict[str, Any],
    *,
    reset_booking: bool = False,
) -> bool:
    """
    Return True when the guest asks to start a new booking / start fresh AND there is
    an in-progress booking draft worth protecting.

    When this returns True, the caller routes to lifecycle_handler (LIFECYCLE_RESET intent)
    to present an explicit confirmation prompt before clearing any session state.
    """
    if not has_in_progress_draft(state):
        return False  # Nothing to protect — no confirmation needed

    msg = (user_message or "").strip()

    # Gemini's semantic reset_booking signal OR explicit start-fresh phrase
    if reset_booking or _START_FRESH_PHRASES.search(msg):
        return True

    return False


def wants_to_abandon_in_progress_booking(
    user_message: str,
    state: dict[str, Any],
    *,
    reset_booking: bool = False,
) -> bool:
    """
    STRICT: Never silently abandon an in-progress draft.
    All explicit or semantic requests to start a new booking when a draft exists
    must go through needs_start_fresh_confirmation() -> lifecycle_handler.

    This function only returns True for short declines ('no', 'nope') when replying
    directly to a prompt asking if the guest wants to continue their draft.
    """
    if not has_in_progress_draft(state):
        return False

    msg = (user_message or "").strip()
    if not msg:
        return False

    # Short declines ONLY when replying to a prompt asking to continue/confirm
    if _ABANDON_SHORT.match(msg) and last_bot_asked_to_continue(state):
        return True

    return False


def looks_like_booking_slot_answer(user_message: str) -> bool:
    """True for short replies that fill dates/guests/option/confirm slots."""
    msg = (user_message or "").strip()
    if not msg or len(msg) > 80:
        return False
    # Strip common fillers
    cleaned = re.sub(r"^(please|pls|we\s+are|we're|i\s+need|need)\s+", "", msg, flags=re.I).strip()
    return bool(_SLOT_ANSWER.match(cleaned) or _SLOT_ANSWER.match(msg))


def is_mid_booking_side_question(user_message: str) -> bool:
    """
    True when the message is clearly a hotel info question,
    not a booking-slot answer.
    """
    msg = (user_message or "").strip()
    if not msg:
        return False
    if looks_like_booking_slot_answer(msg):
        return False

    # Availability for a specific/relative date is a booking action, not a side FAQ
    if _AVAIL_ACTION.search(msg) and (
        _RELATIVE_DATE_FOR_AVAIL.search(msg)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", msg)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", msg)
    ):
        return False

    if _SIDE_QUESTION.search(msg):
        return True

    # Soft: question form mid-booking ("what's the checkout?")
    if ("?" in msg or _QUESTION_LEAD.search(msg)) and len(msg.split()) >= 3:
        # Avoid treating "can I book deluxe" as side FAQ
        if re.search(r"\b(book|reserve|booking)\b", msg, re.I) and re.search(
            r"\b(deluxe|standard|suite|room|adults?|guests?|nights?)\b", msg, re.I
        ):
            return False
        # Availability questions without dates can still be FAQ ("are you open on holidays?")
        # but "rooms available?" alone mid-booking often means search — treat as side only
        # if clearly policy-ish (already covered by _SIDE_QUESTION).
        if _AVAIL_ACTION.search(msg) and not _SIDE_QUESTION.search(msg):
            return False
        return True

    return False


def is_real_booking_id(booking_id: Any) -> bool:
    """Return True if booking_id is a genuine reference, False if pending/placeholder/empty."""
    if not booking_id:
        return False
    val = str(booking_id).strip()
    return bool(val and val.upper() not in ("BK-PENDING", "BK-PAST", "PENDING", "NONE", "YOUR BOOKING", "DRAFT"))


def is_hindi_or_hinglish(state: dict[str, Any], user_msg: str = "") -> bool:
    """Return True if conversational context or user message is Hindi/Hinglish."""
    pref = str(state.get("preferred_language") or state.get("conversation_language") or "").strip().lower()
    if pref in ("english", "en"):
        msg = (user_msg or state.get("user_message") or "").strip().lower()
        hindi_tokens = {
            "kya", "hai", "hain", "ka", "ki", "ke", "ko", "batao", "bataiye", "bhejo",
            "chahiye", "kar", "karo", "diya", "nahi", "nahin", "kitna", "kitne", "kab",
            "kahan", "paas", "namaste", "aapka", "aapki", "dhanyawad", "shukriya", "bhi"
        }
        words = set(re.findall(r"\b\w+\b", msg))
        return len(words & hindi_tokens) >= 2

    if pref in ("hinglish", "hindi", "hi"):
        return True

    msg = (user_msg or state.get("user_message") or "").strip().lower()
    english_tokens = {
        "do", "you", "have", "what", "is", "the", "when", "can", "how", "much",
        "swimming", "pool", "parking", "checkout", "check-in", "wifi", "facilities",
        "available", "time", "please", "send"
    }
    hindi_tokens = {
        "kya", "hai", "hain", "ka", "ki", "ke", "ko", "batao", "bataiye", "bhejo",
        "chahiye", "kar", "karo", "diya", "nahi", "nahin", "kitna", "kitne", "kab",
        "kahan", "paas", "namaste", "aapka", "aapki", "dhanyawad", "shukriya", "bhi"
    }
    words = set(re.findall(r"\b\w+\b", msg))
    if words & hindi_tokens:
        return True
    if len(words & english_tokens) >= 3 and not (words & hindi_tokens):
        return False
    return True


def get_booking_resume_nudge(state: dict[str, Any]) -> str | None:
    """
    Resume prompt based on real missing booking state or pending verification stage.
    Returns None if there's nothing useful to nudge.
    """
    stage = state.get("booking_stage", "new")
    lifecycle_stage = state.get("lifecycle_stage") or stage

    if stage not in _ACTIVE_STAGES and lifecycle_stage not in _ACTIVE_STAGES:
        return None

    # Pending payment verification reminder (one-liner, language mirrored, real ref or omitted)
    if stage in ("payment_pending", "payment_check") or lifecycle_stage in ("payment_pending", "payment_check"):
        raw_b_id = state.get("active_booking_id") or state.get("booking_id")
        has_real_ref = is_real_booking_id(raw_b_id)
        b_id = str(raw_b_id).strip() if has_real_ref else ""
        is_hi = is_hindi_or_hinglish(state)
        if is_hi:
            if has_real_ref:
                return f"_(Aapki booking *{b_id}* ka payment verification abhi chal raha hai. Verify hote hi confirmation bhej denge. 🙏)_"
            return "_(Aapka payment verification abhi chal raha hai. Verify hote hi confirmation bhej denge. 🙏)_"
        else:
            if has_real_ref:
                return f"_(Your booking *{b_id}* payment verification is currently in progress. We'll send your confirmation slip once verified. 🙏)_"
            return "_(Your payment verification is currently in progress. We'll send your confirmation slip once verified. 🙏)_"

    # Pending ID verification reminder
    if stage == "id_verification" or lifecycle_stage == "id_verification":
        raw_b_id = state.get("active_booking_id") or state.get("booking_id")
        has_real_ref = is_real_booking_id(raw_b_id)
        b_id = str(raw_b_id).strip() if has_real_ref else ""
        is_hi = is_hindi_or_hinglish(state)
        if is_hi:
            if has_real_ref:
                return f"_(Aapki booking *{b_id}* ka ID verification abhi chal raha hai. Verify hote hi front desk team bata degi. 🙏)_"
            return "_(Aapka ID verification abhi front desk team ke paas chal raha hai. Verify hote hi bata denge. 🙏)_"
        else:
            if has_real_ref:
                return f"_(Your ID verification for booking *{b_id}* is currently in progress with our front desk team. 🙏)_"
            return "_(Your ID verification is currently in progress with our front desk team. 🙏)_"

    # Confirmed stays never receive a booking resume nudge
    lifecycle_stage = state.get("lifecycle_stage") or stage
    if stage == "confirmed" or lifecycle_stage == "confirmed" or state.get("crm_confirmed_stay"):
        return None

    # If no booking details have been collected yet, do not nudge dates
    has_any_booking_data = bool(state.get("check_in") or state.get("check_out") or state.get("adults") is not None or state.get("selected_rooms"))
    if not has_any_booking_data:
        return None

    if stage == "rooms_shown":
        return (
            "By the way, you still have room options waiting — "
            "just tell me which one you'd like (e.g. *Option 1*)."
        )
    if state.get("next_action") == "need_custom_rooms":
        return "Whenever you're ready, tell me how many rooms and which type you would like!"

    if stage == "quote_shown":
        if state.get("next_action") == "need_guest_name_for_payment" or not state.get("guest_name"):
            return "Whenever you're ready, please share the *guest's full name* (as on ID for check-in) to proceed with your booking."
        return (
            "Your quote is still ready — say *confirm* whenever you'd like to proceed."
        )
    if stage == "awaiting_payment":
        link = state.get("payment_link")
        if link:
            return "_Your payment link is still active - reply *link* to get it again._"
        return "Whenever you're ready, we can continue with payment for your booking."

    # collecting / new — nudge the actual missing field
    has_check_in = bool(state.get("check_in"))
    has_check_out = bool(state.get("check_out"))
    has_adults = state.get("adults") is not None

    if not has_check_in and not has_check_out:
        return (
            "Whenever you're ready, share your *check-in and check-out dates* "
            "and we'll continue your booking."
        )
    if has_check_in and not has_check_out:
        return (
            "Whenever you're ready, share your *check-out date* "
            "and we'll continue your booking."
        )
    if has_check_out and not has_check_in:
        return (
            "Whenever you're ready, share your *check-in date* "
            "and we'll continue your booking."
        )
    if not has_adults:
        return (
            "Whenever you're ready, let me know how many *guests* will be staying "
            "and we'll continue your booking."
        )
    if state.get("ambiguous_guests"):
        return (
            "Just to confirm — is that extra guest an *adult* or a *child*?"
        )
    if state.get("room_type_preference") or state.get("feature_preferences"):
        return "We can continue searching for rooms whenever you're ready!"
    return "We can continue with your booking whenever you're ready!"


def append_booking_resume(text: str, state: dict[str, Any]) -> str:
    """
    Append a resume nudge to a FAQ/info answer when mid-booking.
    Capped at maximum 2 times across turns for active booking collection,
    but always appended for pending verification stages.
    """
    stage = state.get("booking_stage") or state.get("lifecycle_stage") or ""
    lifecycle_stage = state.get("lifecycle_stage") or stage
    pending_stages = {"payment_pending", "payment_check", "id_verification"}
    is_pending = stage in pending_stages or lifecycle_stage in pending_stages

    # Suppress booking resume footer on general non-booking queries (e.g. Sparsh darshan, amenities, policy)
    intent = str(state.get("intent") or "").upper()
    non_booking_intents = {"FAQ", "AMENITIES", "POLICY", "LOCATION", "GREETING", "HUMAN_REQUEST", "COMPLAINT", "ESCALATION"}
    if intent in non_booking_intents and not (state.get("check_in") and state.get("check_out")):
        return text

    if not is_pending:
        count = int(state.get("booking_resume_nudge_count") or 0)
        if count >= 2:
            return text

    nudge = get_booking_resume_nudge(state)
    if not nudge:
        return text

    # Avoid duplicate nudges if already present in text
    if nudge.strip() in text:
        return text

    if not is_pending:
        state["booking_resume_nudge_count"] = count + 1

    return f"{text.rstrip()}\n\n{nudge}"

