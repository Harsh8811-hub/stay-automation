"""Zero-LLM fast paths, lifted out of `understand` (defect #25).

Each function here decides one thing about a message without calling a model,
and either returns the finished turn or `None` to mean "not mine, carry on".
`understand()` reads as an ordered list of these guards; before the split they
were 300 lines of nested `if` blocks interleaved with extraction code, and the
ordering between them was only visible by reading the whole function.

Nothing in this module may mutate `state` — a guard that returns None must
leave the turn exactly as it found it.
"""

import logging
import re
from datetime import datetime as _datetime_cls, timezone as _timezone
from typing import Any, cast

from app.config.settings import get_settings
from app.models.extraction_contract import validate_updates
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


# ── OPT-OUT / OPT-IN COMPLIANCE ──────────────────────────────────────────────
# The patterns themselves live in `understand.py`, next to the argument for why
# they are anchored the way they are; a test imports them from there.
def _patterns():
    from app.workflows.nodes.understand import START_PATTERN, STOP_PATTERN

    return START_PATTERN, STOP_PATTERN


def opt_in_or_out(
    state: ConversationState, user_msg_clean: str
) -> ConversationState | None:
    """Handle START / STOP / an already opted-out guest. Entirely zero-LLM.

    A STOP is destructive (it wipes the draft and the payment link), so it wins
    over everything below: the guest asked not to be messaged, and nothing in
    the rest of this turn should resurrect that.
    """
    START_PATTERN, STOP_PATTERN = _patterns()

    from app.prompts import templates

    if START_PATTERN.match(user_msg_clean):
        logger.info("UNDERSTAND_FAST_PATH | START / Opt-in requested by guest")
        return {
            **state,
            "intent": "GREETING",
            "opted_out": False,
            "booking_stage": "new",
            "next_action": None,
            "bot_response": templates.text_card(
                "Welcome back! You have been re-subscribed to StayInn updates. How can I help you with your booking today?"
            ),
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
        }

    if state.get("opted_out"):
        return {
            **state,
            # NOT "CANCEL_BOOKING". These two branches set the state and the
            # reply they want, but the intent they stamped was read downstream by
            # `feature_gate_node`, whose `check_intent` maps CANCEL_BOOKING to the
            # cancellation gate. That gate is blocked on this hotel, and its
            # response replaces `bot_response` wholesale — so a guest who typed
            # STOP was told "Booking cancellations are processed by our
            # reservations team", with reception's phone number under it. The
            # opt-out had already worked; the reply was the lie. `OPT_OUT` is in
            # no gate map and routes on `next_action` instead (routing.py:148).
            "intent": "OPT_OUT",
            "opted_out": True,
            "booking_stage": "opted_out",
            "next_action": "opted_out",
            "bot_response": templates.text_card(
                "You are currently unsubscribed from messages. Reply START to opt back in."
            ),
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
        }

    # The old bare-tuple check (`user_msg_clean in ("stop", ...)`) is subsumed:
    # every entry it held is a whole-message match for `STOP_PATTERN`.
    if STOP_PATTERN.match(user_msg_clean):
        logger.info("UNDERSTAND_FAST_PATH | STOP / Opt-out requested by guest")
        return {
            **state,
            "intent": "OPT_OUT",
            "opted_out": True,
            "booking_stage": "opted_out",
            "next_action": "opted_out",
            "selected_rooms": [],
            "allocation_options": [],
            "total_price": None,
            "payment_link": None,
            "payment_link_id": None,
            "bot_response": templates.text_card(
                "You have been unsubscribed and will not receive further messages. Reply START to opt back in."
            ),
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
        }

    return None


def stay_check_out(active_booking: Any) -> str | None:
    """The check-out that names the stay holding this turn's confirmed stage.

    Read from the booking row the resolver chose, never from the guest's draft
    `check_out` (which may be a date typed for a second stay). Normalised to
    `YYYY-MM-DD` because it is stored on a session field typed `str` and Mongo
    hands back a `datetime`.
    """
    if not isinstance(active_booking, dict):
        return None
    val = active_booking.get("check_out") or active_booking.get("check_out_date")
    if not val:
        return None
    text = str(val).strip()
    return text[:10] if len(text) >= 10 else text


def fresh_start(
    state: ConversationState, user_msg_clean: str
) -> ConversationState | None:
    """An explicit "start fresh", answered ahead of every stage branch.

    A guest whose stay is confirmed — usually by the CRM, which leaves no
    booking row of ours — is answered by the confirmed branch further down, and
    that branch returns before `apply_intent_overrides` is ever called. So the
    reset that function implements was unreachable for exactly the guests most
    likely to ask for it. Measured 2026-10-17: "/start fresh" logged
    `STAGE_RESOLVED_FROM_CRM | confirmed stay, no booking row of ours`, then
    `STAGE_2_CONFIRMED_DISPATCH`, with no FRESH_START_WIPED — the message was
    answered as a side question.

    The reservation is deliberately NOT touched: a reset ends the conversation,
    not the booking.
    """
    from app.utils.booking_resume import clear_in_progress_booking_fields
    from app.utils.intent_overrides import is_explicit_start_fresh

    if not (
        get_settings().START_FRESH_RESETS_CONFIRMED_STAY
        and is_explicit_start_fresh(user_msg_clean)
    ):
        return None

    logger.info(
        "FRESH_START_FAST_PATH | explicit reset ahead of the stage branches | msg='%s'",
        user_msg_clean,
    )
    # Dropping this turn's draft is only half a reset. context_loader resolves
    # the stage from the database before `understand` runs, so on the NEXT turn
    # the stay is found again and the guest is back where they were — the reset
    # would last exactly one message. Recording which stay was dismissed is what
    # makes it stick, and there are two kinds of stay to name:
    #
    #   * one staff confirmed in the CRM, which never lands in our `bookings`
    #     collection. Its identity is the check-out date, so a later
    #     confirmation for a different stay compares unequal and still applies.
    #   * one the bot itself booked, which does have a row — and that row
    #     resolves the stage on every turn regardless of the session. Measured
    #     on prod 2026-10-17: a guest with such a row reset, was saved
    #     `Stage: new`, and was back to `stage=confirmed` on the next message.
    #     Its identity is the booking id.
    #
    # A latch that is already set is kept when this turn has nothing new to
    # dismiss. `crm_confirmed_check_out` reads None for a stay that is *already*
    # dismissed — the resolver reports no opinion about it — so overwriting the
    # latch with it would quietly un-dismiss on a second "start fresh".
    dismissed_stay = state.get("crm_confirmed_check_out") or state.get(
        "crm_confirmation_dismissed_check_out"
    )
    # The stay that set this turn's stage is not always CRM-sourced. For a guest
    # who booked through the bot, `crm_confirmed_check_out` is None by
    # construction — the resolver only asks the CRM when a booking row does not
    # exist — so the latch above stayed empty and dismissing the row un-masked
    # the CRM confirmation for the *same* stay on the next message.
    #
    # Measured on prod 2026-10-17 (the GUNU transcript): `/start fresh` logged
    # `FRESH_START_DISMISSED crm_stay_check_out=None booking_id=BK-…`, and the
    # reply after it logged `STAGE_BOOKING_ROW_DISMISSED rows=1→0` followed by
    # `STAGE_RESOLVED_FROM_CRM check_out=2026-10-19`. One stay, two sources; the
    # reset has to name the stay, not the source it was found in.
    #
    # `active_booking` is the confirmed row itself — never the draft `check_out`,
    # which may be a date the guest typed for a second stay.
    if not dismissed_stay:
        dismissed_stay = stay_check_out(state.get("active_booking"))
    dismissed_booking = None
    if get_settings().START_FRESH_DISMISSES_BOOKING_ROW:
        dismissed_booking = state.get("active_booking_id") or state.get("booking_id")
    dismissed_booking = dismissed_booking or state.get("dismissed_booking_id")
    logger.info(
        "FRESH_START_DISMISSED | crm_stay_check_out=%s booking_id=%s | "
        "the reset now outlives this turn",
        dismissed_stay,
        dismissed_booking,
    )
    # "Treat this number as new" is stronger than naming the one stay this turn
    # happened to find. This turn's `active_booking` can be empty (nothing
    # resolved, or a stay the guest did not mean), and a phone can hold more than
    # one row — in both cases an id-shaped dismissal names nothing and the guest
    # is confirmed again on the very next message. A timestamp cannot miss:
    # `resolve_guest_stage` drops every row created at or before it, and a
    # booking made after it still applies normally.
    reset_fields: dict[str, Any] = (
        {"guest_reset_at": _datetime_cls.now(_timezone.utc).isoformat()}
        if get_settings().START_FRESH_FORGETS_ALL_BOOKINGS
        else {}
    )
    return {
        **state,
        **clear_in_progress_booking_fields(state),
        **reset_fields,
        "intent": "GREETING",
        "booking_stage": "new",
        "lifecycle_stage": "lead",
        "next_action": None,
        "last_prompt_action": None,
        "extracted_data": None,
        "extraction_result": None,
        "updated_fields": [],
        # This turn must not read as confirmed either, or the reply is built
        # from the stage we just told the guest we were dropping.
        "crm_confirmed_stay": False,
        "crm_confirmed_room_type": None,
        "crm_confirmation_dismissed_check_out": dismissed_stay,
        "dismissed_booking_id": dismissed_booking,
        # A reset is a reset: the guest is no longer adding a second booking
        # beside the old one, they are starting over. Leaving this on would keep
        # the resolver holding the dismissed stay aside.
        "separate_booking_active": False,
        "separate_booking_started_at": None,
    }


# ── USER ACKNOWLEDGMENT & FILLER MESSAGES ("Oo", "Ok", "Coming", "Wait") ─────
_ACKNOWLEDGMENT_FILLER = re.compile(
    r"^(?:"
    r"o+|ok+|okay|coming|wait|wait\s+a\s+(?:sec|minute)|wait\s+karo|hold\s+on|"
    r"acha+|achha+|cool"
    r")[.!?\s]*$",
    re.IGNORECASE,
)


def acknowledgment(
    state: ConversationState, user_msg_clean: str
) -> ConversationState | None:
    """A filler message gets one warm line instead of an LLM round trip."""
    if not (
        _ACKNOWLEDGMENT_FILLER.match(user_msg_clean)
        and not re.search(r"\d", user_msg_clean)
    ):
        return None

    is_hi = str(state.get("preferred_language") or "").lower() in (
        "hinglish",
        "hindi",
        "hi",
    ) or any(
        w in user_msg_clean
        for w in ("ji", "haan", "acha", "achha", "thik", "theek", "karo")
    )
    txt = (
        "Ji bilkul! Aap aaram se details share kar dijiye, main yahi hoon. 😊"
        if is_hi
        else "Sure! Take your time. Whenever you're ready, please share your details. 😊"
    )
    logger.info(
        "FAST_PATH_ACKNOWLEDGMENT | msg='%s' -> USER_ACKNOWLEDGMENT", user_msg_clean
    )
    from app.prompts import templates

    return {
        **state,
        "intent": "USER_ACKNOWLEDGMENT",
        "bot_response": templates.text_card(txt),
        "extracted_data": None,
        "extraction_result": None,
    }


# ── GUEST CLAIMS AN EXISTING BOOKING ("Already booked hai") ──────────────────
_ALREADY_BOOKED_PATTERN = re.compile(
    r"\b("
    r"already\s+booked|booking\s+already\s+done|already\s+have\s+a\s+booking|"
    r"already\s+confirm|pehle\s+se\s+book\s*(?:hai|tha)?|pehle\s+se\s+booking\s*(?:hai|kar\s+rakhi)?|"
    r"booking\s+ho\s+chuki\s+hai|book\s+ho\s+gaya\s+hai|booking\s+hai\s+meri|"
    r"humari\s+booking\s+hai|meri\s+booking\s+hai|kitni\s+baa?r\s+b(?:a|ta)taya"
    r")\b",
    re.IGNORECASE,
)


def already_booked(
    state: ConversationState, user_msg_clean: str
) -> ConversationState | None:
    """The guest says they already hold a booking — look it up, do not sell."""
    if not _ALREADY_BOOKED_PATTERN.search(user_msg_clean):
        return None
    logger.info(
        "FAST_PATH_ALREADY_BOOKED | msg='%s' -> CHECK_EXISTING_BOOKING", user_msg_clean
    )
    return {
        **state,
        "intent": "CHECK_EXISTING_BOOKING",
        "claimed_existing_booking": True,
        "extracted_data": None,
        "extraction_result": None,
    }


# ── THE GUEST NAMED SOMEONE ELSE'S HOTEL ────────────────────────────────────
def other_hotel(
    state: ConversationState, user_msg_clean: str
) -> ConversationState | None:
    """The guest wrote to the wrong number — say so, do not sell.

    Measured 2026-11-05 (Sample Hotel, guest asking for Hotel Samman): the
    message carried rooms and dates, the model extracted them happily, and
    the reply was "I've updated your details to …" — a booking with the
    wrong hotel, silently. The detector lives in
    `app/utils/other_hotel_mention.py`; this is only where it is answered.

    Runs *after* `already_booked` on purpose. "I already booked at Hotel X"
    is a claim about an existing reservation, and the lookup is the right
    answer to it, not a refusal to talk about the property.

    Details in the same message are deliberately not salvaged. Half a draft
    built from a booking the guest meant for someone else is worse than an
    empty one: it would quote our rooms against their dates and look like we
    had accepted. The guest restates them in one line, and that is the cheap
    half of this trade.
    """
    from app.utils.other_hotel_mention import other_property_named

    other = other_property_named(
        state.get("user_message") or user_msg_clean, state.get("hotel_name")
    )
    if not other:
        return None

    from app.prompts import templates

    ours = state.get("hotel_name") or "our hotel"
    logger.info(
        "FAST_PATH_OTHER_HOTEL | guest named %r, this number is %r", other, ours
    )
    return {
        **state,
        # No stage in `routing.py` knows this intent, so the turn falls
        # through to `generate_response`, which uses the `bot_response` set
        # here. The same route every zero-LLM fast path takes.
        "intent": "OTHER_HOTEL",
        "other_hotel_mentioned": other,
        "bot_response": templates.text_card(
            f"You've mentioned *{other}* — this number belongs to *{ours}*, "
            f"so I can't check availability or rates for {other}.\n\n"
            f"If you meant to book with us at {ours}, send your check-in and "
            f"check-out dates with the number of rooms and I'll price it right "
            f"away. If you meant {other}, please message them directly."
        ),
        "extracted_data": None,
        "extraction_result": None,
        "updated_fields": [],
    }


def bank_details_request(
    state: ConversationState, user_msg_clean: str
) -> ConversationState | None:
    """An explicit "send me your bank details" needs no model call."""
    from app.utils.intent_overrides import _BANK_DETAILS_PATTERN

    if not _BANK_DETAILS_PATTERN.search(user_msg_clean):
        return None
    logger.info("UNDERSTAND_FAST_PATH | Bank details inquiry detected -> BANK_DETAILS")
    return {
        **state,
        "intent": "BANK_DETAILS",
        "extracted_data": None,
        "extraction_result": None,
    }


# ── THE FUNNEL ───────────────────────────────────────────────────────────────
# Every extraction branch in `understand` writes into `updates`, and this is the
# only place they all pass through. Lifted verbatim from the node (defect #25)
# so a rule written once — here — is a rule the whole function is subject to,
# instead of one reachable only from the last three lines of a 3,400-line body.
def finalise(
    state: ConversationState, updates: dict[str, Any]
) -> ConversationState:
    # Every extraction branch writes into `updates`, and this line is the
    # only place they all pass through. A value that cannot be what it claims —
    # the year in "21.10.2026" as a headcount, a guest's whole sentence as a
    # room type — is nulled here, once, instead of at each of the seventeen
    # call sites that would each have to remember. See
    # app/models/extraction_contract.py.
    #
    # Validated over the merged result, not over `updates` alone. A value that
    # is wrong does not stop being wrong by being a turn old: in chat
    # 919748533727 the poisoned room_type_preference="ac" was extracted once and
    # then carried silently through later turns, which never mentioned a room
    # type at all — the branch that wrote it never ran again, so an
    # updates-only check would never have seen it. re-reading it from state is
    # what makes the next turn clean rather than the turn after next.
    merged: dict[str, Any] = {**state, **updates}
    merged, rejected_fields = validate_updates(state, merged)

    # A field we just refused must not be announced. The update-confirmation
    # prefix names each changed field and prints the value from state, so
    # leaving `adults` in this list while the contract has nulled it produced
    # "I've updated your details to number of adults to **None**" — the guest
    # told about a change that did not happen. Dropping it here lets the
    # missing-field path ask for it properly instead.
    if rejected_fields:
        declared = merged.get("updated_fields")
        if isinstance(declared, list):
            merged["updated_fields"] = [
                f for f in declared if f not in rejected_fields
            ]
        queued = merged.get("booking_queue")
        if isinstance(queued, list):
            # Rebuilt rather than edited in place: `merged` shares its nested
            # lists with `state`, and this function must not mutate its input.
            merged["booking_queue"] = [
                {
                    **entry,
                    "updated_fields": [
                        f for f in entry["updated_fields"] if f not in rejected_fields
                    ],
                }
                if isinstance(entry, dict)
                and isinstance(entry.get("updated_fields"), list)
                else entry
                for entry in queued
            ]

    # Authoritative lifecycle invariant (Sir's spec):
    # LLM understanding must NEVER downgrade or override an active lifecycle stage.
    #
    # `abandoned_draft` is the one exception, and it has to be here rather than in
    # the caller. It is set by `clear_in_progress_booking_fields`, which runs when
    # the guest explicitly asks to start over — and that override has just written
    # `lifecycle_stage: "lead"` into `merged`. Without this guard the invariant
    # read `state` (which context_loader had already set to the booking's own
    # stage) and put it straight back, so "start fresh" reset the draft and then
    # the very same turn routed to `stage_payment_check` as if nothing had
    # happened. A reset is not the LLM downgrading a stage; it is the guest
    # ending it, and the booking row itself is deliberately left alone.
    current_lifecycle = state.get("lifecycle_stage") or state.get("booking_stage")
    if (
        current_lifecycle in ("payment_check", "payment_pending", "id_verification", "confirmed", "in_house", "checked_out")
        and not merged.get("abandoned_draft")
    ):
        merged["lifecycle_stage"] = state.get("lifecycle_stage") or current_lifecycle
        merged["booking_stage"] = state.get("booking_stage") or current_lifecycle

    return cast(ConversationState, merged)
