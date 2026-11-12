"""
Response Generator Node — decides bot response.

Uses TEMPLATES for standard flows (70%+ of cases).
Falls back to Gemini only for complex/ambiguous situations.
"""

import logging
from typing import Any, cast
from app.config.settings import get_settings
from app.models.schemas import SelectedRoom
from app.prompts import templates
from app.services.room_service import format_room_options
from app.models.schemas import RoomType
from app.workflows.state import ConversationState
from app.observability.metrics import record_translation_failure
from app.services.confirmed_guard import confirmed_guest_reply, guest_should_not_be_sold_to
from app.utils.helpers import format_display_date, guest_is_confirmed

logger = logging.getLogger(__name__)

#: `next_action`s whose template asks the guest for stay details or re-renders the
#: room card. Anything not listed here is an answer (a thank-you, a greeting, a
#: status line) and is left alone even for a confirmed guest.
_SELL_NEXT_ACTIONS = frozenset({
    "need_composite_slots", "need_dates", "need_guests", "need_num_rooms",
    "need_child_ages", "need_ac_preference", "need_meal_plan_preference",
    "need_composition_preference", "need_subgroups", "need_custom_rooms",
    "need_extra_bed", "invalid_dates", "no_rooms", "invalid_room_selection",
    "rooms_shown", "show_quote",
})


def _turn_would_sell(intent: str, next_action: str, booking_stage: str, state: dict) -> bool:
    """Whether this turn's reply would offer rooms or ask for stay details.

    Two shapes, and both were live: an explicit `next_action` from the validator,
    and the `UNKNOWN` fallback — which asks for dates when it has none, and asks
    the guest to pick a combination when a card is already on screen.
    """
    if next_action in _SELL_NEXT_ACTIONS:
        return True
    if intent == "UNKNOWN":
        return not state.get("check_in") or booking_stage in ("rooms_shown", "quote_shown")
    return False


#: Language tags a hotel may declare in `ai_config.supported_languages`, grouped by
#: the two the templates actually speak. Anything else falls back to English.
_HINDI_TAGS = {"hinglish", "hindi", "hi", "hin", "en-in-hi"}
_ENGLISH_TAGS = {"english", "en", "en-us", "en-gb", "en-in"}


def _offered_language(preferred: str | None, hotel_config: dict | None) -> str:
    """The guest's language, unless this hotel does not offer it.

    Sample Hotel declares `supported_languages: ['english']`, yet a guest who wrote
    entirely in English was answered in Hinglish on 2026-10-17 at 16:31:08 — the
    language had been set from the conversation and nothing checked it against
    the hotel's own list. Falling back to English keeps the bot inside what the
    hotel has said it can serve; a hotel that lists Hinglish is unaffected.
    """
    offered = {
        str(s).strip().lower()
        for s in (((hotel_config or {}).get("ai_config") or {}).get("supported_languages") or [])
        if str(s).strip()
    }
    wants = (preferred or "English").strip().lower()
    if offered:
        if wants in _HINDI_TAGS and not (offered & _HINDI_TAGS):
            logger.info(
                "LANGUAGE_NOT_OFFERED | preferred=%s offered=%s — replying in English",
                preferred, sorted(offered),
            )
            return "English"
        if wants in _ENGLISH_TAGS and not (offered & _ENGLISH_TAGS):
            # The hotel lists only Hindi variants; saying "English" would flip the
            # whole reply for a guest who asked for it *and* a hotel that cannot.
            return preferred or "Hinglish"
    return preferred or "English"


def _format_room_gap(gap: dict) -> str:
    """Turn `nearest_room_alternatives` into the message a guest reads.

    Names the half of the request that could not be met, then offers the rooms
    that meet the other half — "these have the king bed, but in AC; these are
    non-AC, but with a double bed". Returns "" when there is no bed or AC class
    to talk about, so the caller falls through to the existing wording and
    nothing changes for the generic case.

    Kept here rather than in the resolver because the resolver returns structure
    and this is prose: the wording is presentation, and it belongs next to the
    other message this branch builds.
    """
    if not gap:
        return ""

    bed = gap.get("bed")
    ac_class = gap.get("ac_class")
    if not bed or not ac_class:
        return ""

    asked = f"{ac_class} {bed}"
    article = "an" if asked[:1].lower() in "aeiou" else "a"
    other_class = "AC" if ac_class == "Non-AC" else "Non-AC"

    from app.services.room_alias_resolver import room_bed_label

    # Two per half. A hotel with six beds that miss the same way produces six
    # near-identical bullet points, which buries the one distinction the message
    # exists to make.
    lines: list[str] = []
    for room in (gap.get("same_bed") or [])[:2]:
        lines.append(f"• **{room.name}** — {bed}, but {other_class}")
    for room in (gap.get("same_class") or [])[:2]:
        other = room_bed_label(room)
        lines.append(f"• **{room.name}** — {ac_class}, but {'a ' + other if other else 'a different bed'}")
    if not lines:
        return ""

    return (
        f"I'm sorry, we don't have {article} **{asked}** room. 😔\n\n"
        f"Closest we have:\n" + "\n".join(lines) +
        "\n\nWhich room type would you prefer?"
    )


#: The only fields a guest may be told about, keyed by the state key
#: `understand` writes into `updated_fields`. This is an allow-list, not a
#: lookup with a fallback: a field with no label here is internal state and is
#: skipped, where the old `field_labels.get(f, f)` printed the raw key.
#:
#: `room_requirements` is deliberately absent, and adding it is the wrong fix.
#: It is a list of dicts (`{'room_type': ..., 'count': ...}`) with no
#: guest-facing one-line form, and the old code printed both the raw key and
#: the dict — so a real guest read `room_requirements to **{'room_type': 'bed',
#: 'count': 2, 'is_foc': False, ...}**` in WhatsApp on 2026-09-26. Naming it
#: would only make the leak prettier. See `tests/test_state_leak_guard.py`.
_SESSION_UPDATE_LABELS = {
    "check_in": "check-in date",
    "check_out": "check-out date",
    "adults": "number of adults",
    "children": "number of children",
    "child_ages": "child ages",
    "num_rooms": "number of rooms",
    "meal_plan_preference": "meal plan",
    "room_type_preference": "room preference",
    "ambiguous_guests": "additional guest count",
    "feature_preferences": "room features",
}


def _renderable(v) -> bool:
    """A value may join the sentence only if it reads as text.

    A dict, or a list holding one, is structured state; interpolating it
    produces a Python repr. The label allow-list should already exclude those,
    so this is the second lock on the same door — the 2026-09-26 leak happened
    because there was no first one.
    """
    if isinstance(v, dict):
        return False
    if isinstance(v, (list, tuple)):
        return all(not isinstance(i, (dict, list, tuple)) for i in v)
    return True


def _format_update_value(field: str, v):
    if isinstance(v, list):
        return ", ".join(str(i) for i in v)
    if field in ("check_in", "check_out"):
        return format_display_date(str(v))
    return str(v)


def _session_update_prefix(updated_fields, state) -> list | None:
    """
    "Got it! I've updated …" for the fields the guest actually set, or None.

    A field must be both named in `_SESSION_UPDATE_LABELS` and renderable to be
    echoed. Filtering here rather than at write time means the guest still hears
    about the fields that do have a name, instead of the whole sentence
    vanishing because one of them could not be shown.

    Returns None — never an empty list — when nothing qualifies, so the caller
    can tell "nothing to say" from "something to say" and leave its own
    accumulated prefix alone.
    """
    echoable = [
        f for f in updated_fields
        if f in _SESSION_UPDATE_LABELS and _renderable(state.get(f))
    ]
    # A check-out is half a stay. When the model reports only the new check-out
    # ("28 november to 5 decemeber" — the typo costs it the check-in, but the
    # same happens whenever the stated check-in equals the one already held),
    # the guest was told "check-out date to 05/12/2026" and nothing about the
    # check-in the bot is keeping. Name the pair. Measured 2026-11-05 on
    # `week-sahu-confirmed-guest-new-booking`.
    if (
        "check_out" in echoable
        and "check_in" not in echoable
        and _renderable(state.get("check_in"))
    ):
        echoable.insert(0, "check_in")
    if not echoable:
        return None

    has_date_change = any(f in ("check_in", "check_out") for f in echoable)
    greeting = "Hello! " if has_date_change else ""

    if len(echoable) == 1:
        f = echoable[0]
        return templates.session_update_confirmation(
            _SESSION_UPDATE_LABELS[f],
            f"**{_format_update_value(f, state.get(f))}**",
            greeting=greeting,
        )

    details = ", ".join(
        f"{_SESSION_UPDATE_LABELS[f]} to **{_format_update_value(f, state.get(f))}**"
        for f in echoable
    )
    return templates.session_update_confirmation("your details", details, greeting=greeting)


async def generate_response(state: ConversationState) -> ConversationState:
    """
    Generate the final bot response.

    Priority:
    1. If THIS turn's room search failed, return custom unavailability response.
       A stale `room_type_not_available` from an earlier turn must not claim the
       turn; see `NO_ROOMS_REQUIRES_FRESH_SEARCH_ENABLED`.
    2. If bot_response is already set (by a previous node), use it.
    3. Use deterministic templates for standard flows.
    4. Gemini as last resort for complex responses.
    """
    if state.get("llm_unavailable"):
        degraded_text = (
            "Thanks for your message. I'm having a little trouble right now. "
            "Please share your check-in date, check-out date and number of guests, "
            "and our team will confirm availability shortly."
        )
        return cast(ConversationState, {
            **state,
            "bot_response": state.get("bot_response") or templates.text_card(degraded_text),
        })

    intent = state.get("intent", "UNKNOWN")
    next_action = state.get("next_action") or ""
    booking_stage = state.get("booking_stage", "new")
    preferred_lang = _offered_language(
        state.get("preferred_language"), state.get("hotel_config")
    )
    bot_response: list[dict] = []
    template_just_rendered = False
    awaiting_dates = False

    # Generate update confirmation prefix if any core fields were updated
    updated_fields = state.get("updated_fields", [])
    update_prefix = state.get("bot_response_prepend", [])
    if isinstance(update_prefix, dict):
        update_prefix = [update_prefix]
    
    # The multi-booking acknowledgement is not built here any more: it is put
    # in `bot_response_prepend` when `understand` creates the queue, so the
    # card path in `room_searcher` — which never reaches this node — shows it
    # too. Both renderers now read the same sentence, from one place.

    if updated_fields:
        # See `_session_update_prefix`. Anything it declines to name stays
        # unsaid, and the prefix accumulated above is left as it was.
        prefix = _session_update_prefix(updated_fields, state)
        if prefix:
            # Both, not either. This used to *replace* `update_prefix`, which
            # silently threw away the multi-booking acknowledgement above on
            # any turn that also set a date or a party — i.e. on every
            # multi-stay message, since the first turn always carries dates.
            # Measured 2026-11-07: two stays were queued and the guest was
            # shown only the first stay's card, with no hint that the second
            # had been read at all. "Got it! …" reads first, then the list.
            update_prefix = list(prefix) + list(update_prefix)

        child_notes = state.get("child_age_notes")
        if child_notes and "child_ages" in updated_fields:
            notes_str = " ".join(child_notes)
            if update_prefix and isinstance(update_prefix, list) and update_prefix[0].get("text"):
                update_prefix[0]["text"] += f"\n_{notes_str}_"
            elif not update_prefix:
                update_prefix = templates.text_card(f"_{notes_str}_")

    # If response was already generated by a previous node, keep it (and prepend update prefix if any)
    if state.get("bot_response"):
        resp = state["bot_response"]
        
        # If we fell back to other rooms because the requested type was full/unavailable
        if state.get("room_type_not_available") and state.get("room_type_preference") and next_action != "need_custom_rooms":
            pref = state.get("room_type_preference")
            fallback_msg = templates.text_card(f"I'm sorry, our **{pref}** rooms are fully booked or unavailable for these dates. However, here are some great alternatives:")
            if isinstance(resp, list):
                resp = fallback_msg + resp
            else:
                resp = fallback_msg + [resp]
                
        if isinstance(resp, str):
            resp = templates.text_card(resp)
        bot_response = update_prefix + resp
        return {**state, "bot_response": bot_response}

    # ── Zero selling for a guest we already know is confirmed ────────────
    # Stage routing should never have let this turn reach a selling branch, but
    # resolution can miss (CRM timeout, a session stage written before the
    # confirmation landed, an older revision). The branches below are the ones
    # that ask for stay details or put a card back on screen, so this is the
    # point at which a missed confirmed guest is stopped rather than sold to.
    # See `app/services/confirmed_guard.py`.
    if guest_should_not_be_sold_to(state) and _turn_would_sell(intent, next_action, booking_stage, state):
        logger.info(
            "RESPONSE_SUPPRESSED_CONFIRMED_GUEST | intent=%s next_action=%s hotel=%s — "
            "selling reply replaced with the concierge one",
            intent, next_action, state.get("hotel_id"),
        )
        bot_response = update_prefix + confirmed_guest_reply(state)
        return {**state, "bot_response": bot_response, "next_action": "confirmed_guest_no_sell"}

    # Handle requested room type not available in DB AND no fallback options could be generated
    pref = str(state.get("room_type_preference") or "")
    is_option_pref = pref.lower().startswith("option ") or pref.isdigit() or pref.lower() in ("custom", "build your own")
    # `room_type_not_available` is a latch, not a turn result: `room_searcher`
    # sets it when a search falls back and nothing clears it. Read without a
    # freshness check it claimed, on every later turn that produced no reply of
    # its own, that the hotel had no rooms — including turns that never asked
    # about availability. Live transcript, 2026-10-20: the guest asked about hot
    # water and got the answer, then wrote "Ok... thank you sir." and got "I'm
    # sorry, we don't have rooms available that match your requirements" twice.
    #
    # A failed search is the only turn entitled to report the failure, and it is
    # the only turn that leaves `next_action == "no_rooms"` behind — `load_context`
    # resets `next_action` on every turn, so this test is turn-scoped by
    # construction. The flag restores the old unconditional branch without a
    # redeploy of any other line.
    requires_fresh_search = get_settings().NO_ROOMS_REQUIRES_FRESH_SEARCH_ENABLED
    if (
        state.get("room_type_not_available")
        and (not requires_fresh_search or next_action == "no_rooms")
        and not is_option_pref
        and next_action not in (
            "need_custom_rooms", "show_quote", "need_dates", "need_guests",
            "need_ac_preference", "need_meal_plan_preference", "need_composition_preference",
            "need_subgroups", "need_change_dates", "need_change_guests", "need_extra_bed"
        )
    ):
        hotel_id = state.get("hotel_id", 0)
        from app.services import inventory_service
        rooms = await inventory_service.get_available_rooms(hotel_id, state.get("check_in"), state.get("check_out"))
        room_names = sorted(list(set(r.name for r in rooms))) if rooms else []
        room_list_str = ", ".join(f"**{name}**" for name in room_names)

        # Only quote the preference when the guest's words actually name a room
        # the hotel sells. `room_type_preference` is free text, so echoing it
        # verbatim produced "we don't have **one room with attached bathroom**
        # rooms available" — the guest's own sentence, repeated back as though
        # it named a room. The extraction contract rejects the obvious
        # sentences at the source; this is the last line of defence for the
        # rest, and it needs the catalogue, which is why it is not there.
        # `room_names` — the names, which is what this wants. It used to be
        # handed `await get_all_rooms(hotel_id)`, a list of `RoomType` objects,
        # and `names_a_catalogue_room` does `name.lower()` on each entry: it
        # raised `AttributeError: 'RoomType' object has no attribute 'lower'`
        # every time this branch was reached. So a guest whose custom request
        # parsed cleanly but could not be allocated was told "a brief issue"
        # instead of this message, which is the one case the branch exists for.
        # `room_names` is built above and is already what the text quotes.
        from app.models.extraction_contract import names_a_catalogue_room
        pref_str = pref if names_a_catalogue_room(pref, room_names) else ""
        # Before the flat dead end, try to say WHICH half of the request the
        # hotel cannot meet. A guest who asks for a "non-AC king bed" at a hotel
        # whose king rooms are all AC was told "we don't have rooms available
        # that match your requirements" and handed the entire catalogue — which
        # never says that the bed was fine and only the AC class was the problem,
        # or the reverse (Sample Hotel, 2026-09-21).
        from app.services.room_alias_resolver import nearest_room_alternatives
        gap_text = _format_room_gap(
            nearest_room_alternatives(
                pref, rooms, feature_preferences=state.get("feature_preferences")
            )
        )
        if room_list_str and gap_text:
            response_text = gap_text
        elif room_list_str:
            if pref_str and get_settings().PREFERRED_ROOM_HANDOFF_ENABLED:
                # A named room we cannot build is not a "here are other rooms"
                # problem: the guest asked for one thing, and a list answers a
                # question they did not ask. The front desk is the other path,
                # so raise the ticket in this turn — routing only reaches
                # `handle_escalation` on the *next* dispatch, and a guest who
                # never replies would otherwise get no ticket at all (the same
                # reason `confirm_booking` calls `raise_availability_ticket`).
                from app.services.availability_guard import (
                    AvailabilityVerdict,
                    escalation_updates,
                    raise_availability_ticket,
                )

                verdict = AvailabilityVerdict(
                    "preferred_room_unavailable",
                    f"{pref_str} could not be allocated for "
                    f"{state.get('check_in')} to {state.get('check_out')}",
                )
                await raise_availability_ticket(state, verdict)
                logger.info(
                    "PREFERRED_ROOM_HANDOFF | hotel=%s pref=%s rooms=%s — front desk asked to take over",
                    state.get("hotel_id"), pref_str, room_list_str,
                )
                bot_response = update_prefix + templates.button_card(
                    "🙏 Please hold on — a senior staff will reach out to you "
                    "regarding your booking.\n\n"
                    f"I'm sorry, I couldn't confirm **{pref_str}** for these dates just now.",
                    buttons=[{"id": "menu_reception", "title": "Talk to reception"}],
                )
                return {
                    **state,
                    "bot_response": bot_response,
                    "next_action": "preferred_room_handoff",
                    **escalation_updates(verdict),
                }
            if pref_str:
                # `pref_str` survives `names_a_catalogue_room` only when the
                # preference names a room the hotel has open for these dates —
                # so "we don't have **premier room** rooms available" printed
                # directly above a list containing PREMIER ROOM KING BED was a
                # live contradiction, three turns in a row, to guest
                # 919064956832 on 2026-10-18. The real failure there was
                # allocational, not inventory: 5 rooms were requested and the
                # hotel could allocate at most 3 for those dates, whatever the
                # room type. Say the thing that actually failed.
                response_text = (
                    f"I'm sorry, we couldn't build your stay from **{pref_str}** rooms "
                    f"for these dates. 😔\n\n"
                    f"We currently have the following room types available: {room_list_str}.\n\n"
                    f"Would any of these work for you?"
                )
            else:
                response_text = f"I'm sorry, we don't have rooms available that match your requirements. 😔\n\nWe currently have the following room options available: {room_list_str}.\n\nWhich room type would you prefer?"
        else:
            response_text = f"I'm sorry, we are completely fully booked for those dates. 😔"
        bot_response = update_prefix + templates.text_card(response_text)
        return {**state, "bot_response": bot_response}

    # ── Template-based responses (no Gemini needed) ──────────────
    hotel_name = state.get("hotel_name", "our hotel")
    raw_b_id = state.get("booking_id") or state.get("active_booking_id")

    if intent == "GREETING":
        # A stay confirmed by staff in the CRM has no booking id of ours — the guest is
        # still confirmed, we just have no reference to quote at them.
        has_real_booking = guest_is_confirmed(state, raw_b_id)
        if (booking_stage in ("confirmed", "checked_in") or state.get("booking_id")) and has_real_booking:
            b_id = raw_b_id
            id_disp = f" (`{b_id}`)" if b_id else ""
            g_name = state.get("guest_name") or "Guest"
            pref_lang = state.get("preferred_language") or "English"
            is_hi = str(pref_lang).lower() in ("hinglish", "hindi", "hi")
            if is_hi:
                greet_txt = (
                    f"🏨 *Namaste {g_name}! {hotel_name} mein aapka swagat hai.* 🙏\n\n"
                    f"Aapki booking{id_disp} confirmed hai! Main aapki kya sahayata kar sakta hoon? "
                    f"Aap express check-in ke liye ID card bhej sakte hain, ya hotel ki location, amenities ke baare mein pooch sakte hain."
                )
            else:
                greet_txt = (
                    f"🏨 *Namaste {g_name}! Welcome back to {hotel_name}.* 🙏\n\n"
                    f"Your reservation{id_disp} is confirmed with us! How can I assist you today? "
                    f"You can upload your ID card for express check-in, or ask for directions, amenities, or stay assistance."
                )
            return {**state, "bot_response": templates.text_card(greet_txt)}

        is_pending = (
            booking_stage in ("payment_pending", "payment_check", "id_verification", "awaiting_payment")
            or state.get("lifecycle_stage") in ("payment_pending", "payment_check", "id_verification", "awaiting_payment")
            or bool(state.get("payment_link") or (raw_b_id and str(raw_b_id).startswith("PAY-")))
        )
        if is_pending:
            from app.utils.booking_resume import append_booking_resume, is_hindi_or_hinglish
            is_hi = is_hindi_or_hinglish(state)
            g_name = state.get("guest_name") or "Guest"
            name_str = f" {g_name}" if g_name != "Guest" else ""
            if is_hi:
                greet_txt = f"Namaste{name_str}! 🙏 {hotel_name} mein aapka swagat hai. Aaj main aapki kya sahayata kar sakta hoon?"
            else:
                greet_txt = f"Hello{name_str}! 🙏 Welcome to {hotel_name}. How can I assist you today?"
            greet_txt = append_booking_resume(greet_txt, state)
            return {**state, "bot_response": templates.text_card(greet_txt)}

        has_active_booking = bool(
            state.get("check_in")
            or state.get("selected_rooms")
            or booking_stage in ["rooms_shown", "quote_shown", "awaiting_payment", "ready_for_payment"]
        )
        if state.get("abandoned_draft") or state.get("reset_booking") or not has_active_booking:
            guest_name = state.get("guest_name")
            pref_lang = state.get("preferred_language")
            if not guest_name and not pref_lang:
                bot_response = templates.ask_name_and_language(hotel_name, lang=pref_lang or "English")
            elif not guest_name:
                bot_response = templates.ask_guest_name(hotel_name, lang=pref_lang or "English")
            elif not pref_lang:
                bot_response = templates.ask_preferred_language(guest_name)
            else:
                bot_response = templates.show_menu(hotel_name)
        else:
            # Stale Session Detection (Fix 2)
            updated_at = state.get("session_updated_at")
            age_hours = 0.0
            is_stale = False
            if updated_at:
                from datetime import datetime, UTC
                dt = None
                if isinstance(updated_at, str):
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    except Exception:
                        pass
                elif isinstance(updated_at, datetime):
                    dt = updated_at
                if dt:
                    if not dt.tzinfo:
                        dt = dt.replace(tzinfo=UTC)
                    age_hours = (datetime.now(UTC) - dt).total_seconds() / 3600.0
                    is_stale = age_hours > 4.0

            sess_id = state.get("session_id", "-")
            date_str = format_display_date(state.get("check_in")) or "date not set"

            if is_stale:
                logger.info(
                    "GREETING_STALE_SESSION | sess=%s | stage=%s | age_hours=%.1f | action=offer_reset",
                    sess_id,
                    booking_stage,
                    age_hours,
                )
                bot_response = [{
                    "type": "text",
                    "text": (
                        f"Welcome back! 👋 You have a previous booking session for check-in *{date_str}*.\n\n"
                        f"Would you like to:\n"
                        f"• *Continue* your existing booking\n"
                        f"• *Start fresh* with a new booking"
                    ),
                }]
            else:
                logger.info(
                    "GREETING_ACTIVE_SESSION | sess=%s | stage=%s | age_hours=%.1f | action=remind_continue",
                    sess_id,
                    booking_stage,
                    age_hours,
                )
                bot_response = [{
                    "type": "text",
                    "text": (
                        f"Welcome back! You have a booking in progress. "
                        f"Say *continue* to pick up where you left off, or *start fresh* for a new booking."
                    ),
                }]
    elif intent == "THANK_YOU":
        is_pending = (
            booking_stage in ("payment_pending", "payment_check", "id_verification", "awaiting_payment")
            or state.get("lifecycle_stage") in ("payment_pending", "payment_check", "id_verification", "awaiting_payment")
            or bool(state.get("payment_link") or (raw_b_id and str(raw_b_id).startswith("PAY-")))
        )
        if is_pending:
            from app.utils.booking_resume import append_booking_resume, is_hindi_or_hinglish
            is_hi = is_hindi_or_hinglish(state)
            if is_hi:
                ty_txt = "Aapka swagat hai! Agar aur koi sahayata chahiye to zaroor batayein. 🙏"
            else:
                ty_txt = "You're very welcome! Feel free to ask if you need anything else. 🙏"
            ty_txt = append_booking_resume(ty_txt, state)
            bot_response = templates.text_card(ty_txt)
        else:
            bot_response = templates.thank_you_response()
    elif next_action in ("need_composite_slots", "need_dates", "need_guests", "need_num_rooms"):
        missing = state.get("missing_fields") or []
        if "check_in" in missing or "check_out" in missing or not state.get("check_in") or not state.get("check_out"):
            awaiting_dates = True
        lang = preferred_lang or state.get("preferred_language") or "English"
        template_shown = state.get("composite_template_shown", False)
        # First time ask when no slots have been given yet
        if not template_shown and not state.get("check_in") and not state.get("check_out") and state.get("adults") is None:
            bot_response = templates.ask_composite_slots(lang=lang)
            template_just_rendered = True
        else:
            bot_response = templates.ask_missing_slot_one_liner(
                missing_fields=missing,
                check_in=state.get("check_in"),
                check_out=state.get("check_out"),
                adults=state.get("adults"),
                children=state.get("children"),
                lang=lang,
            )
    elif next_action == "need_child_ages":
        # The hotel's own bands, so the sentence quotes its policy rather than a
        # hardcoded 10. Same source the validator reads its buckets from.
        _limits = ((state.get("hotel_config") or {}).get("business_config") or {}).get("limits") or {}
        _free_limit = int(_limits.get("child_free_age_limit") or 10)
        _adult_limit = int(_limits.get("child_as_adult_age_limit") or 11)
        if _adult_limit <= _free_limit:
            _free_limit = max(0, _adult_limit - 1)
        bot_response = templates.ask_child_ages_followup(
            lang=preferred_lang or "English",
            known_ages=state.get("child_ages"),
            total_children=int(state.get("children") or 0),
            free_limit=_free_limit,
            adult_limit=_adult_limit,
        )
    elif next_action in ("group_booking_lead", "lead_created"):
        if state.get("group_lead_submitted") or state.get("booking_stage") == "lead_created":
            bot_response = templates.text_card(
                "Thank you! 🙏 Our reservations team has received your group request and contact details. We will reach out to you directly with the best rates.\n\nIs there anything else I can assist you with?"
            )
        else:
            bot_response = templates.button_card(
                "For group bookings of 5+ rooms or 10+ guests, our reservations team will assist you directly with special group rates and dedicated coordination.\n\nPlease share your contact details and group requirements, or connect with our front desk below.",
                buttons=[{"id": "menu_reception", "title": "Talk to reception"}],
            )
    elif next_action == "need_guest_name":
        bot_response = templates.ask_guest_name(hotel_name)
    elif next_action == "need_preferred_language":
        bot_response = templates.ask_preferred_language(state.get("guest_name", ""))
    elif next_action == "need_guest_type":
        bot_response = templates.ask_guest_type()
    elif next_action == "need_ac_preference":
        bot_response = templates.ask_ac_preference(lang=preferred_lang or "English")
    elif next_action == "need_extra_bed":
        # The charge is not passed in: the question is the client's fixed line
        # and names no amount. The card the guest just picked from prints the
        # mattress on its own row, so the figure is already on their screen.
        bot_response = templates.ask_extra_bed()
    elif next_action == "need_change_dates":
        bot_response = templates.ask_change_dates(lang=preferred_lang or "English")
    elif next_action == "need_change_guests":
        bot_response = templates.ask_change_guests(lang=preferred_lang or "English")
    elif next_action == "need_meal_plan_preference":
        bot_response = templates.ask_meal_plan_preference(lang=preferred_lang or "English")
    elif next_action == "need_composition_preference":
        from app.utils.helpers import format_quote_date_range
        from app.services.booking_engine import calculate_num_nights
        check_in = state.get("check_in")
        check_out = state.get("check_out")
        adults = state.get("adults") or 2
        children = state.get("children") or 0
        try:
            num_nights = calculate_num_nights(check_in, check_out) if check_in and check_out else 1
        except Exception:
            num_nights = 1
        dates_str = format_quote_date_range(check_in, check_out) if (check_in and check_out) else ""
        night_word = "night" if num_nights == 1 else "nights"
        nights_str = f"{num_nights} {night_word}"
        adult_word = "adult" if adults == 1 else "adults"
        if children and children > 0:
            child_word = "child" if children == 1 else "children"
            pax_str = f"{adults} {adult_word} + {children} {child_word}"
        else:
            pax_str = f"{adults} {adult_word}"
        lang = preferred_lang or "English"
        bot_response = templates.ask_composition_preference(dates_str, nights_str, pax_str, lang=lang)
    elif next_action == "need_subgroups":
        lang = preferred_lang or "English"
        bot_response = templates.ask_family_subgroups_followup(lang=lang)
    elif next_action == "need_guest_name_for_payment":
        bot_response = templates.ask_guest_name_for_payment(state.get("prior_guest_name"))
    elif next_action == "need_custom_rooms":
        all_opts = state.get("allocation_options", [])
        avail_names = []
        for opt in all_opts:
            for r in opt.get("rooms", []):
                rn = r.get("room_name", "")
                short = rn.replace("ROOM", "").replace("Room", "").strip()
                if short and short not in avail_names:
                    avail_names.append(short)
        bot_response = templates.build_your_own_prompt(
            avail_names, problem=state.get("custom_room_problem")
        )
    elif next_action == "invalid_dates":
        bot_response = templates.invalid_dates()
    elif next_action == "no_rooms":
        bot_response = templates.no_rooms_available()
    elif next_action == "invalid_room_selection":
        candidates = state.get("room_match_candidates") or []
        num_opts = len(state.get("allocation_options") or [])
        range_str = f"1 to {num_opts}" if num_opts > 1 else "1"
        if len(candidates) >= 2:
            if len(candidates) == 2:
                msg = f"Did you mean **{candidates[0]}** or **{candidates[1]}**?"
            else:
                shown = candidates[:4]
                msg = (
                    "Did you mean "
                    + ", ".join(f"**{n}**" for n in shown[:-1])
                    + f", or **{shown[-1]}**?"
                )
            bot_response = templates.text_card(msg)
        else:
            bot_response = templates.text_card(
                f"Please reply with the option number ({range_str}) or the exact room name from the list above."
            )
            if state.get("allocation_options"):
                from app.services.room_service import build_room_options_card
                options_card = build_room_options_card(state.get("allocation_options"), state)
                bot_response.append(options_card)
    elif intent == "UNKNOWN":
        user_msg = state.get("user_message", "")
        # Check if user uploaded a non-payment image
        if "[IMAGE_ANALYSIS]" in user_msg and "Type: Payment Screenshot" not in user_msg:
            bot_response = templates.non_payment_image_redirect(user_msg)
        elif not state.get("check_in") and not state.get("check_out"):
            bot_response = templates.text_card("Could you please share your check-in and check-out dates so I can check room availability for you?")
            awaiting_dates = True
        elif state.get("check_in") and not state.get("check_out"):
            bot_response = templates.text_card(f"Got your check-in date ({state.get('check_in')})! Could you please share your **check-out date**?")
            awaiting_dates = True
        elif not state.get("adults"):
            ci = state.get("check_in")
            co = state.get("check_out")
            dates_hint = f" for {ci} to {co}" if ci and co else ""
            bot_response = templates.text_card(f"Could you please let me know how many adults and children will be staying{dates_hint}?")
        elif state.get("booking_stage") in ("rooms_shown", "quote_shown"):
            # The range is derived from the options actually offered. The old
            # literal "*1*, *2*, or *3*" named a number that was not on screen
            # whenever fewer than two options rendered — Chat B 06:47:45 and
            # 06:51:00, where only two entries existed.
            #
            # Numbers, not letters, since 2026-10-14. The card itself stopped
            # being lettered on 2026-09-23 (`room_service.card_rows`), but these
            # two prompts kept asking for a "combination letter" — so a guest on
            # this path was told to reply with an A while the card above them
            # showed 1s. `format_option_letters` stays where it is: a guest
            # holding an old lettered card in their scrollback can still type
            # one, and `room_card_reply` still resolves it.
            from app.services.room_service import format_option_numbers, option_number_count
            n = option_number_count(state.get("allocation_options"), state)
            bot_response = templates.text_card(
                f"Please reply with a combination number ({format_option_numbers(n)}) to continue with your booking."
            )
        else:
            bot_response = templates.unrelated_query_response()
    else:
        # ── Fallback ─────────────────────────────────────────────────
        bot_response = templates.fallback_response()

    bot_response = update_prefix + bot_response

    # ── LANGUAGE TRANSLATION CHOKE-POINT ──────────────────────────────
    # Production safeguard: if translation fails for any reason (timeout,
    # rate limit, malformed JSON), the guest receives the original English
    # response rather than an error. Guests seeing English is always better
    # than the bot failing entirely.
    preferred_lang = state.get("preferred_language")
    if preferred_lang and preferred_lang.lower() not in ("english", "en", "casual english"):
        from app.services.gemini_service import translate_bot_response
        import logging
        log = logging.getLogger(__name__)

        # Extract ai_config from state if present
        ai_config = None
        config_dict = state.get("hotel_config")
        if config_dict and isinstance(config_dict, dict):
            ai_raw = config_dict.get("ai_config")
            if ai_raw:
                from app.config.hotel_config import AIPromptConfig
                ai_config = AIPromptConfig(**ai_raw)

        original_response = bot_response  # Always keep original as fallback
        try:
            log.info(f"Translating bot response — style={preferred_lang}")
            bot_response = await translate_bot_response(bot_response, preferred_lang, ai_config)
        except Exception as e:
            # Fallback to English — never surface a translation error to the guest
            bot_response = original_response
            log.warning(
                "Translation failed — falling back to English",
                extra={
                    "hotel_id": state.get("hotel_id"),
                    "session_id": state.get("session_id"),
                    "preferred_language": preferred_lang,
                    "conversation_language": state.get("conversation_language"),
                    "error": str(e),
                },
            )
            record_translation_failure(
                preferred_language=preferred_lang,
                conversation_language=state.get("conversation_language"),
                hotel_id=state.get("hotel_id"),
            )

    # Strip any leaked prompt/context artifacts (e.g. "Stage: collecting", "Stage: new")
    if bot_response:
        import re
        for card in bot_response:
            if isinstance(card, dict) and card.get("type") == "text" and isinstance(card.get("text"), str):
                cleaned_text = re.sub(r'^(?:Stage|Context):\s*[a-zA-Z_]+\s*\n*', '', card["text"], flags=re.IGNORECASE).strip()
                if cleaned_text != card["text"]:
                    card["text"] = cleaned_text

    from typing import Any
    updates: dict[str, Any] = {"bot_response": bot_response}
    if template_just_rendered:
        updates["composite_template_shown"] = True
    if awaiting_dates:
        updates["awaiting"] = "stay_dates"
    if state.get("abandoned_draft"):
        updates["abandoned_draft"] = False
    return cast(ConversationState, {**state, **updates})
