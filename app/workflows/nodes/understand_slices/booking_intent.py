"""The booking-intent branch of `understand`, out of the node.

Extracted from `app/workflows/nodes/understand.py` on 2026-10-28. This is a
**pure move**: the statements are the original lines dedented by one level, in
the same order, with the same names — verified by comparing the AST of the
moved block against the AST of this function before the swap.

Why it was worth moving. `understand` was a 3,600-line function with a single
rule applied at the bottom (the stage invariant) and many exits above it; the
819-line `if not is_cancellation and is_booking_intent:` branch was the largest
single reason the function could not be read. The branch has no early `return`,
no `await`, and never assigns into `state` — it only reads `state` and writes
`updates` — which is what makes it movable without changing behaviour.

The contract is deliberately narrow and the same as before:
  * read `state`, write `updates`;
  * return nothing;
  * no side effects beyond the task/activity writes the branch always made.

`_apply_split_party` moved with it. It was only ever called from this branch and
from the custom-room prompt handler, and the node now imports it from here.

Nothing here is re-implemented, tidied, or renamed. Behaviour changes belong in a
separate commit, so that a regression can be attributed to one or the other.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config.settings import get_settings
from app.utils.ac_text import (
    ac_preference_from_text,
    is_bare_ac_class,
    mentions_both_classes,
)
from app.utils.date_masking import mask_dates
from app.utils.pax_resolution import resolve_pax_counts
from app.utils.room_requirement_parse import (
    is_an_explicit_split,
    reconcile_split_with_party,
    split_party_from_message,
)

logger = logging.getLogger(__name__)


def _apply_split_party(
    updates: dict[str, Any],
    state: dict[str, Any],
    split_items: list[dict],
    split_adults: int,
) -> None:
    """Make the guest's own split the party, and keep it from being undone.

    Three things, and each one is a defect if it is left out.

    **The head count.** A guest who writes "room 1 2 people, room 2 4 people"
    has stated eight adults without writing "8" anywhere, and the model will
    often have read a different number from somewhere else in the message.
    `effective_adults` is the field the search and the validator actually read —
    both take it in preference to `adults` — so writing `adults` alone would be
    inert on any session where a previous turn had set it.

    **The room-type preference.** This branch sets `room_type_preference` to the
    guest's raw sentence. Left as the sentence, `validator` finds no room name
    in it, flags `pref_mismatch`, wipes the rooms we just selected, and searches
    again — forever. "custom" is already one of the values it treats as
    generic, which is exactly what this is.

    **The seats.** `guests` is the adult ceiling per room, already recorded on
    each line by the allocator through `line_seats`.
    """
    updates["adults"] = split_adults
    updates["effective_adults"] = split_adults
    # A single line holding one room is the *shape* of a party that was never
    # split, not a statement about how many rooms it needs. Written blindly,
    # it overwrote a room count the guest had already given: on 2026-11-06
    # "3 rooms" then "8 adults" produced one auto-generated line, `num_rooms`
    # became 1, and the three rooms were gone (eval case
    # collecting-party-and-rooms-survive). A guest who really names the rooms
    # is unaffected — "3 rooms" writes `num_rooms` through the room-keyword
    # branch long before this, and a genuine split has more than one line.
    derived_rooms = sum(int(item.get("count") or 1) for item in split_items)
    if len(split_items) > 1 or derived_rooms > 1:
        updates["num_rooms"] = derived_rooms
    updates["room_type_preference"] = "custom"
    updates["room_selection_resolved"] = True
    logger.info(
        "SPLIT_PARTY | sess=%s | %d lines, %d adults | stage=%s",
        state.get("session_id"),
        sum(int(item["count"]) for item in split_items),
        split_adults,
        updates.get("booking_stage"),
    )



_OPTION_MARKER = re.compile(r"^(?:option|opt|no\.?)\s*\d+$|^\d+$", re.IGNORECASE)


def _norm_room_name(value: Any) -> str:
    """Lower-case, punctuation-free form of a room name, for comparing names."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _names_the_room_already_selected(value: Any, state: dict[str, Any]) -> bool:
    """True when the model's room type is the room the guest already has.

    `state["room_type_preference"]` stops being a room name the moment the
    guest picks an option by number: it becomes the literal string
    "option 1" (written by the option-number branch). Every later turn in
    which the model helpfully repeats the actual room type — which it does
    from context on almost any message — then compares
    "Superior Room Double Bed" against "option 1", looks like a *new*
    preference, and the preference-change branch below wipes
    `total_price`, `selected_rooms` and `allocation_options` and sends the
    guest back to a fresh search.

    Measured 2026-11-05 on Sample Property (guest 919999000222's shape): pick a
    room, then ask "here is showing 2000 then how can be 2250" — the quote
    the guest was looking at vanished, and the price question was answered
    with "let me pull up the rooms and rates for those nights".

    So a room type only counts as a *change* when it names a room the guest
    does not already hold. A genuine switch ("actually the family room")
    still fails this test and still triggers the fresh search.
    """
    if _OPTION_MARKER.match(_norm_room_name(value)):
        return False
    new_name = _norm_room_name(value)
    if not new_name:
        return False
    held = []
    for room in state.get("selected_rooms") or []:
        name = room.get("room_name") if isinstance(room, dict) else getattr(room, "room_name", None)
        held.append(_norm_room_name(name))
    held = [h for h in held if len(h) >= 4]
    return any(h in new_name or new_name in h for h in held)


def apply_booking_intent(
    state: dict[str, Any],
    updates: dict[str, Any],
    result: Any,
    user_msg_clean: str,
    active_prompt_action: str | None,
    expected_action: str | None,
    is_inquiry_or_question: bool,
    _full_weights: bool,
    _meal_plan_from_button: bool,
    NON_NAME_WORDS: set[str],
    resolved_intent: str,
) -> None:
    """Apply a booking-intent turn to `updates`.

    Every argument is a name the original block closed over in `understand`.
    They are passed explicitly so this module has no import cycle with the node
    and so the next person can see, at the call site, exactly what the branch
    depends on.

    `_apply_split_party` lives in this module because this branch is where it
    was born. The custom-room prompt handler in the node still calls it, so the
    node imports it from here rather than keeping a second copy.
    """
    if state.get("booking_stage") == "lead_created" and not updates.get("group_lead_submitted"):
        updates["booking_stage"] = "new"
        updates["group_lead_submitted"] = False
        if updates.get("next_action") == "lead_created":
            updates["next_action"] = None

    extracted = result.extracted_data

    # ── MULTI-BOOKING QUEUE LOGIC ──
    if getattr(result, "extracted_bookings", None):
        queue_len = len(result.extracted_bookings)
        if queue_len > 1:
            from app.observability.metrics import metrics
            import random
            from datetime import datetime, timezone, timedelta
        
            _IST = timezone(timedelta(hours=5, minutes=30))
            now_iso = datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")
            queue_id = f"QUEUE-{datetime.now(_IST).strftime('%Y%m%d')}-{random.randint(10000, 99999)}"
            logger.info(f"MULTI_BOOKING_DETECTED: {queue_len} bookings. QUEUE_CREATED: {queue_id}")
            metrics.incr("queue.created")
        
            queue_data = []
            for b in result.extracted_bookings:
                d = b.model_dump(exclude_none=True)
                d["queue_status"] = "pending"
                d["queue_id"] = queue_id
                d["created_at"] = now_iso
                d["last_updated"] = now_iso
                d["completed_at"] = None
                queue_data.append(d)
            
            # Mark first as immediately processing
            queue_data[0]["queue_status"] = "processing"
            queue_data[0]["last_updated"] = now_iso
        
            updates["booking_queue"] = queue_data
            updates["active_booking_index"] = 0
            updates["multi_booking_detected"] = True
            updates["queue_id"] = queue_id
            # Say so, on this turn, in both renderers. The card reply is drawn
            # by `room_searcher` and returns straight to `save_state`, never
            # passing through `response_generator` — so a sentence built there
            # was never shown on the one turn it mattered. Measured 2026-11-07:
            # a two-stay message queued both stays and the guest saw only the
            # first stay's card. `bot_response_prepend` is read by both paths.
            from app.prompts import templates
            from app.utils.queue_manager import describe_queue

            updates["bot_response_prepend"] = templates.text_card(
                describe_queue(queue_data)
            )
        # Pull the first booking to process right now
        extracted = result.extracted_bookings[0]
    
    # ── TARGETED QUEUE MODIFICATIONS ──
    if getattr(result, "target_queue_index", None) is not None:
        t_idx = result.target_queue_index
        active_idx = updates.get("active_booking_index", state.get("active_booking_index", 0))
        queue = updates.get("booking_queue", state.get("booking_queue", []))
    
        if 0 <= t_idx < len(queue) and t_idx != active_idx:
            logger.info(f"Targeted queue modification for index {t_idx} (active is {active_idx})")
            target_booking = dict(queue[t_idx])
        
            if resolved_intent == "CANCEL_BOOKING" or result.intent == "CANCEL_BOOKING":
                target_booking["queue_status"] = "cancelled"
                target_booking["terminal_status"] = "cancelled"
                updates["intent"] = "QUEUE_MODIFICATION"
                updates["bot_response"] = f"✅ The booking for {target_booking.get('guest_name', 'that guest')} has been cancelled."
            else:
                # It's a modification
                ex_dict = extracted.model_dump(exclude_none=True)
                for k, v in ex_dict.items():
                    target_booking[k] = v
                updates["intent"] = "QUEUE_MODIFICATION"
                updates["bot_response"] = f"✅ The details for {target_booking.get('guest_name', 'that guest')}'s booking have been updated."
            
            queue[t_idx] = target_booking
            updates["booking_queue"] = queue
        
            # Prevent these extracted fields from polluting the active booking
            extracted = result.extracted_data.__class__() # Empty it out so no changed_fields are detected
            resolved_intent = updates["intent"]
    
    # Clean feature preferences early to prevent false change detection
    if extracted.feature_preferences is not None:
        cleaned_features = [
            f for f in extracted.feature_preferences 
            if f and f.lower().strip() not in ["none", "null", "n/a", ""]
        ]
        extracted.feature_preferences = cleaned_features if cleaned_features else None

    core_fields = ["adults", "children", "child_ages", "num_rooms", "check_in", "check_out", "room_type_preference", "ambiguous_guests", "budget", "feature_preferences", "meal_plan_preference"]
    changed_fields = []
    extracted_dict = extracted.model_dump(exclude_none=True)

    # ── HARD DATE VALIDATION & ORDER ENFORCEMENT ──
    if "check_in" in extracted_dict and "check_out" not in extracted_dict:
        new_ci = extracted_dict["check_in"]
        old_ci = state.get("check_in")
        old_co = state.get("check_out")
        if old_ci and old_co:
            try:
                from datetime import datetime, timedelta
                d_ci = datetime.strptime(old_ci, "%Y-%m-%d")
                d_co = datetime.strptime(old_co, "%Y-%m-%d")
                duration = max((d_co - d_ci).days, 1)
                new_ci_dt = datetime.strptime(new_ci, "%Y-%m-%d")
                new_co = (new_ci_dt + timedelta(days=duration)).strftime("%Y-%m-%d")
                extracted_dict["check_out"] = new_co
                extracted.check_out = new_co
            except Exception:
                pass
        elif not old_co:
            try:
                from datetime import datetime, timedelta
                new_ci_dt = datetime.strptime(new_ci, "%Y-%m-%d")
                new_co = (new_ci_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                extracted_dict["check_out"] = new_co
                extracted.check_out = new_co
            except Exception:
                pass

    if "check_in" not in extracted_dict and "check_out" not in extracted_dict:
        # Fallback date recovery if model omitted dates on a booking turn
        from app.workflows.nodes.understand import _deterministic_date_result
        _det = _deterministic_date_result(state.get("user_message", ""))
        if _det and _det.extracted_data:
            if _det.extracted_data.check_in:
                extracted.check_in = _det.extracted_data.check_in
                extracted_dict["check_in"] = _det.extracted_data.check_in
            if _det.extracted_data.check_out:
                extracted.check_out = _det.extracted_data.check_out
                extracted_dict["check_out"] = _det.extracted_data.check_out

    if "check_in" in extracted_dict or "check_out" in extracted_dict:
        # Protect existing valid stay dates from being wiped/corrupted by non-date messages (e.g. "2", "option 1", etc.)
        user_msg = state.get("user_message", "")
        has_existing_dates = bool(state.get("check_in") and state.get("check_out"))
        from app.utils.intent_overrides import _RELATIVE_DATE_EXPR
        has_rel_date = bool(_RELATIVE_DATE_EXPR.search(user_msg))
        has_month_word = bool(re.search(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', user_msg, re.I))
        has_date_pattern = bool(re.search(r'\b\d{1,2}(?:st|nd|rd|th)?\s*(?:to|-|/|\s)\s*\d{1,2}(?:st|nd|rd|th)?\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b(?:today|tomorrow|tonight|aaj|kal|parso)\b', user_msg, re.I))
        has_date_cues = has_rel_date or has_month_word or has_date_pattern
        if has_existing_dates and not has_date_cues:
            extracted_dict.pop("check_in", None)
            extracted_dict.pop("check_out", None)
            extracted.check_in = None
            extracted.check_out = None

    if "check_in" in extracted_dict or "check_out" in extracted_dict:
        from app.utils.helpers import normalize_date, validate_proposed_date_range
    
        raw_ci = extracted_dict.get("check_in")
        raw_co = extracted_dict.get("check_out")
    
        norm_ci = normalize_date(raw_ci) if raw_ci else None
        norm_co = normalize_date(raw_co) if raw_co else None
    
        if norm_ci:
            extracted_dict["check_in"] = norm_ci
            extracted.check_in = norm_ci
        if norm_co:
            extracted_dict["check_out"] = norm_co
            extracted.check_out = norm_co

        proposed_ci = extracted_dict.get("check_in") or state.get("check_in")
        proposed_co = extracted_dict.get("check_out") or state.get("check_out")

        is_valid_dates, date_err_msg = validate_proposed_date_range(proposed_ci, proposed_co)
        if not is_valid_dates:
            logger.warning(f"Invalid date proposal detected: check_in={proposed_ci}, check_out={proposed_co}. Error: {date_err_msg}")
            # Do NOT mutate state with invalid dates
            extracted_dict.pop("check_in", None)
            extracted_dict.pop("check_out", None)
            extracted.check_in = None
            extracted.check_out = None
            updates["bot_response"] = date_err_msg
            updates["next_action"] = "invalid_dates"
            # The date prompt was cleared before validation ran; a rejected
            # date is not an answer, so put the parked prompt back.
            updates["awaiting"] = state.get("awaiting")


    # ── BOOKING MODIFICATION INTERCEPTION ──
    booking_status = state.get("booking_status", "new")
    stage = state.get("booking_stage", "new")
    is_confirmed_or_checked_in = (
        booking_status in ("confirmed", "checked_in")
        or stage in ("confirmed", "cancellation_pending")
    )

    if updates.get("intent") == "PRICE_MATCH":
        pass
    elif result.intent == "MODIFY_BOOKING" and not is_confirmed_or_checked_in:
        # Pre-confirmation draft modification: update active search parameters directly
        updates["intent"] = "BOOK_ROOM"
        resolved_intent = "BOOK_ROOM"
        result.intent = "BOOK_ROOM"
    elif result.intent == "MODIFY_BOOKING":
        # Load existing pending modification state for confirmed bookings
        mod_fields = list(state.get("modified_fields") or [])
        prev_vals = dict(state.get("previous_values") or {})
        req_vals = dict(state.get("requested_values") or {})
    
        # We check all core fields that can be modified
        mod_track_fields = ["check_in", "check_out", "adults", "children", "num_rooms", "room_type_preference", "meal_plan_preference", "special_requests"]
    
        new_changes_detected = False
        for field in mod_track_fields:
            if field in extracted_dict:
                new_val = extracted_dict[field]
                old_val = state.get(field)
            
                # Ignore no-op updates (same as current active booking)
                if old_val == new_val:
                    continue
                if isinstance(old_val, str) and isinstance(new_val, str) and old_val.lower().strip() == new_val.lower().strip():
                    continue
                
                # Ignore if this exact change is already pending
                if req_vals.get(field) == new_val:
                    continue

                if field not in mod_fields:
                    mod_fields.append(field)
                    prev_vals[field] = old_val
                req_vals[field] = new_val
                new_changes_detected = True
                
        if mod_fields:
            lines = ["**Booking Changes Detected**"]
            for f in mod_fields:
                formatted_field = f.replace("_", " ").title()
                old_str = prev_vals[f] if prev_vals[f] is not None else "None"
                lines.append(f"• {formatted_field}: {old_str} → {req_vals[f]}")
        
            # Maintain Modification Metadata
            from datetime import datetime, timezone
            now_str = datetime.now(timezone.utc).isoformat()
            meta = dict(state.get("modification_metadata") or {})
            if not meta.get("modification_id"):
                import random
                meta["modification_id"] = f"MOD-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{random.randint(100, 999)}"
                meta["created_at"] = now_str
            meta["updated_at"] = now_str
            meta["status"] = "pending"

            updates["modified_fields"] = mod_fields
            updates["previous_values"] = prev_vals
            updates["requested_values"] = req_vals
            updates["booking_change_summary"] = "\n".join(lines)
            updates["modification_metadata"] = meta
        
            logger.info(f"Booking Modification Intercepted/Updated: {mod_fields} | Meta: {meta['modification_id']}")
            # We skip applying these directly to state to protect the current booking!
            extracted_dict = {} # Clear it so normal apply logic does nothing
            extracted = result.extracted_data.__class__() # Empty out extracted
        else:
            if updates.get("intent") != "PRICE_MATCH":
                # If no actual changes detected, fallback to BOOK_ROOM so the flow continues smoothly
                updates["intent"] = "BOOK_ROOM"
                result.intent = "BOOK_ROOM"
        
    # ── PENDING MODIFICATION SELECTION INTERCEPTION ──
    if result.intent == "SELECT_ROOM" and state.get("requested_values"):
        logger.info("User selected a room while modification is pending. Flushing requested values into active state.")
        req_vals = state.get("requested_values")
        for k, v in req_vals.items():
            # We put them directly into extracted_dict so they trigger the normal change detection!
            extracted_dict[k] = v
        
        # Clear pending modification state so it doesn't pollute future turns
        updates["requested_values"] = None
        updates["modified_fields"] = None
        updates["booking_change_summary"] = None
        updates["modification_metadata"] = None

    # Guard against option number / menu selection / continue phrases being extracted as num_rooms / adults
    #
    # Every count regex in this block reads `countable_text` — this message
    # with its dates blanked — and never `user_msg_clean`. A date puts
    # numbers in exactly the position a count would occupy, and on
    # 2026-09-24 "…check out 15/11/26 person 5+1" had its year read as the
    # party size: the regex below matched "26 person" and a guest who asked
    # for six people was told the bot had 26, then that no room could seat
    # them. See app/utils/date_masking.py, and the second, independent
    # guard in extraction_contract Rule 1.
    countable_text = mask_dates(user_msg_clean)

    is_comp_option = bool(
        expected_action == "need_composition_preference" 
        and not re.search(r'\b\d+\s*(?:rooms?|kamre?|kamra)\b', countable_text)
    )
    is_menu_option = bool(
        expected_action in ("need_ac_preference", "need_meal_plan_preference", "change_menu_shown", "need_composition_preference")
        and not re.search(r'\b\d+\s*(?:adults?|persons?|people|pax|guests?|log|vyakti|jana|jane|rooms?|kamre?|bachche?|children)\b', countable_text)
    )
    is_continue_msg = bool(re.search(r'\b(?:continue|resume|pick\s+up|aage\s+badho)\b', user_msg_clean))
    has_explicit_capacity_phrase = bool(re.search(r'\b\d+\s*(?:adults?|persons?|people|pax|guests?|members?|log|vyakti|jana|jane|rooms?|kamre?|kamra|bachche?|children)\b', countable_text))

    is_option_number_selection = bool(
        not has_explicit_capacity_phrase
        and (
            is_comp_option
            or is_menu_option
            or is_continue_msg
            or (
                (updates.get("intent") == "SELECT_ROOM" or result.intent == "SELECT_ROOM")
                and (
                    updates.get("selected_option_number")
                    or updates.get("selected_option_label")
                    or updates.get("room_type_preference") in ("cheapest", "expensive")
                    or re.search(r'^\s*(?:option|opt|no\.?|choice)?\s*([1-9][a-zA-Z]?|[a-zA-Z][1-9])\.?\s*$', (state.get("user_message") or "").strip(), re.I)
                    or re.search(r'\b(cheapest|chepest|sasta|lowest\s*price|best\s*value|most\s*affordable|budget\s*option)\b', (state.get("user_message") or "").strip(), re.I)
                )
            )
        )
    )
    if has_explicit_capacity_phrase and (updates.get("intent") == "SELECT_ROOM" or result.intent == "SELECT_ROOM"):
        updates["intent"] = "BOOK_ROOM"

    # Perform change detection before updating state for NORMAL bookings
    for field in core_fields:
        if field in extracted_dict:
            new_val = extracted_dict[field]
            old_val = state.get(field)
        
            # If child_ages was extracted as empty/falsy but old_val exists, default extraction didn't specify ages; keep old
            if field == "child_ages" and not new_val and old_val:
                continue

            old_val_clean = old_val if old_val else None
            new_val_clean = new_val if new_val else None
        
            if old_val_clean != new_val_clean:
                # If they are actively selecting a room or confirming a booking, don't flag capacity or preference fields as changes!
                if (
                    result.intent in ("SELECT_ROOM", "CONFIRM_BOOKING")
                    or updates.get("intent") in ("SELECT_ROOM", "CONFIRM_BOOKING")
                    or is_option_number_selection
                ):
                    if field in ("room_type_preference", "recommendation_preference", "selected_option_label", "selected_option_number", "num_rooms", "adults", "children", "child_ages"):
                        continue
                changed_fields.append(field)

    # Naming the room the guest already holds is an echo, not a preference
    # change — see `_names_the_room_already_selected`. It is dropped from
    # `changed_fields` here rather than inside the loop because the party may
    # have changed in the *same* message: in that case the restated room type
    # has to stay in `changed_fields`, or the capacity-change branch below
    # drops the carried-over `room_type_preference` the guest just re-stated
    # (`test_a_preference_the_guest_restates_is_kept`).
    _HARD_RESET_FIELDS = ("check_in", "check_out", "adults", "children", "num_rooms", "child_ages")
    if "room_type_preference" in changed_fields and not any(
        f in changed_fields for f in _HARD_RESET_FIELDS
    ):
        if _names_the_room_already_selected(extracted_dict.get("room_type_preference"), state):
            logger.info(
                "ROOM_TYPE_ECHO | keeping %r — it is the room already selected (state held %r)",
                extracted_dict.get("room_type_preference"), state.get("room_type_preference"),
            )
            changed_fields.remove("room_type_preference")

    # Deterministic extraction of multi-family / person counts (e.g. "6 person(3/3)-2 family", "6 people 2 rooms")

    m_fam = re.search(r'(\d+)\s*(?:person|people|pax|members?|guests?)?\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*[- ]*(\d+)\s*famil', countable_text)
    if m_fam:
        total_pax = int(m_fam.group(1))
        num_fams = int(m_fam.group(4))
        updates["adults"] = total_pax
        updates["effective_adults"] = total_pax
        updates["children"] = 0
        updates["effective_children"] = 0
        updates["child_ages"] = []
        updates["child_age_ask_count"] = 0
        updates["num_rooms"] = num_fams
    else:
        has_child_kw = bool(re.search(r'\b(?:child|children|kids?|bachche?|bacha|baby|infant)\b', user_msg_clean))

        # A guest-stated split is the party as well as the rooms: nine
        # guests is "2 people 1 room, 3 people 1 room, 4 people 1 room",
        # and the split has already written that sum by the time we get
        # here — on the custom branch. The regex below would read the "2"
        # out of the first line and shrink the party to two. The main
        # branch writes its split *after* this point rather than before, so
        # without this guard the two branches answer the same sentence
        # differently. Requirements carrying `guests` are the better
        # reader; see `is_an_explicit_split`.
        if not is_an_explicit_split(updates.get("room_requirements")):
            # Check structured heading lists first (e.g. Adults:- ... Children:- ...)
            from app.utils.pax_resolution import extract_structured_heading_pax
            struct_adults, struct_children, struct_child_ages = extract_structured_heading_pax(state.get("user_message", ""))
            if struct_children is not None:
                has_child_kw = True
                extracted.children = max(extracted.children or 0, struct_children)
                if struct_child_ages:
                    extracted.child_ages = list(dict.fromkeys((extracted.child_ages or []) + struct_child_ages))
            if struct_adults is not None:
                extracted.adults = max(extracted.adults or 0, struct_adults)

            # Check if guest wrote a price calculation: e.g. "Rs 2,000 x 3 rooms x 3 nights = Rs 18,000"
            m_calc = re.search(
                r'(?:(?:rs\.?|inr|₹)\s*)?(\d[\d,]*)\s*(?:[xX*]|per\s*(?:room|night))\s*(\d+)\s*(?:rooms?|kamre?)?\s*(?:[xX*]|per\s*(?:room|night)|\s*for\s*)?\s*(\d+)?\s*(?:nights?)?\s*=\s*(?:(?:rs\.?|inr|₹)\s*)?(\d[\d,]*)',
                state.get("user_message", ""),
                re.IGNORECASE
            )
            if m_calc:
                try:
                    g_rate = float(m_calc.group(1).replace(",", ""))
                    g_total = float(m_calc.group(4).replace(",", ""))
                    updates["guest_written_price"] = {
                        "rate": g_rate,
                        "total": g_total,
                        "applied": False,
                        "text": m_calc.group(0).strip(),
                    }
                    updates["guest_quoted_budget"] = g_total
                except Exception:
                    pass

            _PER_ROOM_CAPACITY_RE = re.compile(
                r'\b(?:(?:max|maximum|at\s*most|upto|up\s*to)\s*)?(\d+)\s*(?:adults?|guests?|persons?|people|pax|log)\s*(?:in\s+each\s+room|per\s+room|in\s+a\s+room|each\s+room|a\s+room|har\s+kamre\s*(?:me|mein)?)\b'
                r'|\b(?:in\s+each\s+room|per\s+room|har\s+kamre\s*(?:me|mein)?)\s*(?:(?:max|maximum|at\s*most|upto|up\s*to)\s*)?(\d+)\s*(?:adults?|guests?|persons?|people|pax|log)\b',
                re.IGNORECASE,
            )
            m_cap = _PER_ROOM_CAPACITY_RE.search(countable_text)
            if m_cap:
                cap_val = int(m_cap.group(1) or m_cap.group(2))
                updates["max_adults_per_room"] = cap_val
                countable_text = _PER_ROOM_CAPACITY_RE.sub(" ", countable_text)
                if state.get("adults") and int(state.get("adults")) > cap_val:
                    if extracted.adults == cap_val:
                        extracted.adults = int(state.get("adults"))

            m_pax = re.search(r'\b(\d+)\s*(?:persons?|people|pax|members?|guests?|adults?|log|vyakti|jana|jane)\b', countable_text, re.IGNORECASE)
            if not m_pax and re.search(r'^\s*(\d+)\s*$', user_msg_clean):
                if (
                    "adults" in (state.get("missing_fields") or [])
                    or expected_action in ("need_composite_slots", "need_guests")
                    # `expected_action` already covers `need_composite_slots`; only
                    # `need_guests` is reachable through this clause, and it was dead
                    # for the same reason as the two readers above — `next_action` is
                    # None by the time a turn runs. It reads the surviving pointer.
                    or active_prompt_action in ("need_composite_slots", "need_guests")
                ):
                    m_pax = re.search(r'^\s*(\d+)\s*$', user_msg_clean)

            # The model reads first; the regex is the fallback, not a rival.
            # The decision is a pure function so the ordering is testable
            # without a model call — see app/utils/pax_resolution.py, which
            # carries the 2026-09-24 defect and the rule it produced.
            resolved, used_regex = resolve_pax_counts(
                model_adults=extracted.adults,
                model_children=extracted.children,
                regex_pax=int(m_pax.group(1)) if m_pax else None,
                has_child_kw=has_child_kw,
                is_option_number_selection=is_option_number_selection,
                expected_action=expected_action,
            )
            updates.update(resolved)
            if used_regex:
                # The room count beside the pax phrase belongs to the regex
                # branch only; the model's own `num_rooms` is applied later.
                m_rms = re.search(r'\b(\d+)\s*(?:rooms?|kamre?|famil(?:y|ies))\b', countable_text)
                if m_rms:
                    updates["num_rooms"] = int(m_rms.group(1))
    if not updates.get("child_ages"):
        # The model reads first, but a PARTIAL read does not end the question.
        # `extracted.child_ages` is list[int], so a 2.5-year-old cannot be
        # expressed in it and the model drops the child rather than rounding:
        # the 2026-10-29 brief naming a 5-year-old and a 2.5-year-old arrived
        # as [5]. This `if` used to accept that as final and never run the
        # deterministic pass below, which does handle decimals ("2.5 saal" ->
        # 2) because it is written for the guest's own words.
        #
        # So the deterministic pass always runs, and it wins only when the
        # model returned FEWER ages than the party declares children. That
        # condition is the guard, not a nicety: the age regex is happy to read
        # "coming here for 3 years" as a three-year-old, and it must not be
        # allowed to overrule a model that already answered the question.
        model_ages = list(getattr(extracted, "child_ages", None) or [])
        declared_children = int(
            updates.get("children") or getattr(extracted, "children", None) or 0
        )

        # A guest who lists their children under a heading has COUNTED them for
        # us. The model lost a two-and-a-half-year-old from both the count and
        # the ages on the 2026-10-29 brief ("6 guests + 1 children" for a party
        # of eight), and no amount of age reconciliation can put a child back
        # who was never counted. See `app/utils/child_ages.py`.
        from app.utils.child_ages import child_section_ages
        section_ages = child_section_ages(user_msg_clean)
        if len(section_ages) > declared_children:
            logger.info(
                "CHILD_COUNT_FROM_SECTION | declared=%s section=%s",
                declared_children, section_ages,
            )
            declared_children = len(section_ages)
            updates["children"] = declared_children

        age_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:saal|years?|yrs?)\b', user_msg_clean)
        regex_ages = [int(float(a)) for a in age_matches if int(float(a)) <= 17]

        # Inside a children section the ages are unambiguous, so they are read
        # from there; outside it the conservative whole-message regex stands.
        preferred_ages = section_ages or regex_ages

        model_is_short = (
            bool(model_ages)
            and declared_children > len(model_ages)
            and len(preferred_ages) > len(model_ages)
        )

        if model_ages and not model_is_short and len(model_ages) >= declared_children:
            updates["child_ages"] = model_ages
        elif preferred_ages:
            # One age per child, in the order the guest named them. Capping at
            # the declared count stops a stray "3 years" elsewhere in the
            # message from becoming a fourth child.
            updates["child_ages"] = (
                preferred_ages[:declared_children] if declared_children else preferred_ages
            )
            if model_is_short:
                logger.info(
                    "CHILD_AGES_RECONCILED | model=%s detected=%s declared=%s",
                    model_ages, updates["child_ages"], declared_children,
                )

        if not updates.get("child_ages"):
            age_group_match = re.search(r'(?:age|aged|saal|umar)\s*([\d\s,and&yrolds]+)', user_msg_clean)
            if age_group_match:
                found_nums = [int(float(n)) for n in re.findall(r'\b\d+(?:\.\d+)?\b', age_group_match.group(1)) if int(float(n)) <= 17]
                if found_nums:
                    updates["child_ages"] = found_nums
        if not updates.get("child_ages"):
            paren_matches = re.findall(r'\((?:age\s*)?([\d\s,and&a-zA-Z]+)\)', user_msg_clean)
            if paren_matches:
                for pm in paren_matches:
                    if any(w in pm.lower() for w in ("age", "yr", "year", "saal", "child", "kid")):
                        found_nums = [int(float(n)) for n in re.findall(r'\b\d+(?:\.\d+)?\b', pm) if int(float(n)) <= 17]
                        if found_nums:
                            updates["child_ages"] = found_nums
                            break

    # Deterministic extraction of AC preference from user message.
    #
    # The comparison clause asks whether the guest named BOTH classes, and
    # it asks that with `mentions_both_classes` rather than by checking the
    # two vocabularies side by side — because "without ac" contains the
    # word AC without mentioning a second class, and read carelessly it
    # looks like a guest weighing the two up.
    #
    # `has_both_keyword` stays narrower than the shared no-preference
    # vocabulary on purpose: "any ac?" is a question about AC, and only the
    # words below mean the guest wants to see both classes. The class
    # decision itself is the shared resolver's, which is what makes
    # "non - ac" and "a/c" resolve here as they do everywhere else.
    has_both_keyword = bool(re.search(r'\b(?:show\s*both|both|dono|all|sab)\b', user_msg_clean))
    is_ac_comparison_or_question = (
        (mentions_both_classes(user_msg_clean) and user_msg_clean not in ("non ac", "non-ac", "non_ac"))
        or " or " in user_msg_clean
        or " ya " in user_msg_clean
        or "how do" in user_msg_clean
        or "difference" in user_msg_clean
        or (user_msg_clean.endswith("?") and not has_both_keyword)
        or "kaise" in user_msg_clean
        or "kya" in user_msg_clean
    )

    if has_both_keyword and not any(w in user_msg_clean for w in ("how do", "difference", "kaise", "kya")):
        updates["ac_preference"] = "SHOW_BOTH"
        updates["ac_qualified"] = True
    elif not is_ac_comparison_or_question:
        resolved_ac = ac_preference_from_text(user_msg_clean)
        if resolved_ac:
            updates["ac_preference"] = "SHOW_BOTH" if resolved_ac == "NO_PREFERENCE" else resolved_ac
            if resolved_ac == "AC":
                updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if "non" not in f.lower()] + ["AC"]
            elif resolved_ac == "NON_AC":
                updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f != "AC"] + ["Non-AC"]
            elif resolved_ac == "NO_PREFERENCE":
                updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f not in ("AC", "Non-AC")]
            updates["ac_qualified"] = True
    if extracted.ambiguous_guests is not None and not is_option_number_selection:
        updates["ambiguous_guests"] = extracted.ambiguous_guests
    if extracted.num_rooms is not None and not is_option_number_selection:
        # Guard against date digits (e.g. 5 from "5 to 7 sep") or child ages (e.g. 9 from "age 4 and 9") being wrongly parsed as num_rooms
        has_room_kw = bool(re.search(r'\b\d+\s*(?:rooms?|kamre?|kamra|famil(?:y|ies))\b', user_msg_clean))
        is_date_digit = bool(re.search(rf'\b{extracted.num_rooms}\s*(?:to|-|se|\.|\/)\b', user_msg_clean))
        is_child_age = extracted.num_rooms in (updates.get("child_ages") or []) or bool(re.search(rf'\b(?:age|aged|saal|umar)\s*.*?\b{extracted.num_rooms}\b', user_msg_clean))
        # A message that counts *guests* and names no rooms is not a statement
        # about rooms. Hosted on 2026-11-06: the guest sent "3 rooms", then
        # "8 adults", and the model returned `num_rooms: 1` with the second --
        # the reply acknowledged "number of rooms to *1*" and the three rooms
        # the guest had asked for were gone. The date/child-age guards below
        # both passed, because "8 adults" is neither.
        mentions_party = bool(re.search(
            r'\b\d+\s*(?:adults?|people|persons?|guests?|pax|log|aadmi|admi|bande)\b',
            user_msg_clean,
        ))
        if has_room_kw:
            updates["num_rooms"] = extracted.num_rooms
        elif mentions_party:
            # The extraction still differs from state, so `changed_fields`
            # already names num_rooms; left there, the reply announces a room
            # count the guest never mentioned. Drop it with the write.
            if "num_rooms" in changed_fields:
                changed_fields.remove("num_rooms")
        elif not is_date_digit and not is_child_age:
            updates["num_rooms"] = extracted.num_rooms
    if getattr(extracted, "room_requirements", None) and not updates.get("room_requirements"):
        reqs = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in extracted.room_requirements]
        updates["room_requirements"] = reqs
        updates["force_new_search"] = True
        updates["booking_stage"] = "collecting"
        updates["next_action"] = "search"
        updates["selected_rooms"] = []
        updates["room_type_not_available"] = False
        changed_fields.append("room_requirements")
    elif not updates.get("room_requirements"):
        from app.utils.room_requirement_parse import parse_room_requirements, parse_split_occupancies
        det_reqs = parse_room_requirements(state.get("user_message", "")) or parse_split_occupancies(state.get("user_message", ""))
        if det_reqs:
            updates["room_requirements"] = det_reqs
            updates["force_new_search"] = True
            updates["booking_stage"] = "collecting"
            updates["next_action"] = "search"
            updates["selected_rooms"] = []
            updates["room_type_not_available"] = False
            changed_fields.append("room_requirements")

    # A guest-stated split is the party as well as the rooms. Must run whenever
    # room_requirements is present (from model extraction or deterministic parse).
    active_reqs = updates.get("room_requirements")
    if active_reqs and is_an_explicit_split(active_reqs):
        m_tr = re.search(
            r'\b(?:no\.?\s*(?:of\s*)?rooms?|number\s*of\s*rooms?|total\s*rooms?)\s*[:-]?\s*(\d+)\b|\b(\d+)\s*rooms?\b',
            state.get("user_message", ""),
            re.I,
        )
        total_rooms = int(m_tr.group(1) or m_tr.group(2)) if m_tr else updates.get("num_rooms")
        if total_rooms and len(active_reqs) == total_rooms and sum(int(r.get("count") or 1) for r in active_reqs) > total_rooms:
            for r in active_reqs:
                r["count"] = 1

        # Who decides the party, in order of authority.
        #
        # A guest who wrote "2 people 1 room, 3 people 1 room" stated the total
        # without writing it, so the split is the answer. When they instead
        # wrote a total — "3 rooms ... for 8 adults" — the extractor's `guests`
        # is a rounded-up quotient (three rooms of three, i.e. nine) and must
        # not be summed over the 8 the guest actually said. Everything is
        # re-cut to whichever number won, so the seats and the party agree.
        stated_split = split_party_from_message(state.get("user_message", ""))
        stated_total = int(updates.get("adults") or 0)
        party = stated_split or stated_total or sum(
            int(r.get("count") or 0) * int(r.get("guests") or 0)
            for r in active_reqs
        )
        if stated_split or stated_total:
            active_reqs = reconcile_split_with_party(active_reqs, party)
            updates["room_requirements"] = active_reqs
        _apply_split_party(updates, state, active_reqs, party)
    if extracted.budget is not None:
        updates["budget"] = extracted.budget
    
    if extracted.feature_preferences is not None:
        updates["feature_preferences"] = extracted.feature_preferences
    if extracted.check_in is not None:
        updates["check_in"] = extracted.check_in
    if extracted.meal_plan_preference is not None and not _meal_plan_from_button:
        # A tap on a meal button must not be overwritten here. Measured 2026-10-15: the
        # fast path stores the clean code ("MAP"), then — because the meal arm does not
        # return — the turn reaches this extraction, and the model is free to answer with
        # the button's own label echoed back ("MAP (Bkfast+Din)"). The merge at the end of
        # this function lets `updates` win, so the raw label landed in state instead.
        #
        # It was harmless only by luck: `get_preferred_rate_plan` keyword-matches, so
        # `"map" in "map (bkfast+din)"` still picked the MAP plan. It is a trap for
        # anything that exact-matches, and the lifecycle summary prints the raw string to
        # staff. Typed text is unaffected — the flag is only set by a resolved tap.
        updates["meal_plan_preference"] = extracted.meal_plan_preference
    if getattr(extracted, "recommendation_preference", None) is not None:
        updates["recommendation_preference"] = extracted.recommendation_preference
    if getattr(extracted, "display_preference", None) is not None:
        updates["display_preference"] = extracted.display_preference
    if getattr(extracted, "room_count_preference", None) is not None:
        updates["room_count_preference"] = extracted.room_count_preference
    if getattr(extracted, "comfort_preference", None) is not None:
        updates["comfort_preference"] = extracted.comfort_preference
    if extracted.check_out is not None:
        updates["check_out"] = extracted.check_out
    if extracted.room_type_preference is not None:
        clean_rt = extracted.room_type_preference.strip().lower()
        resolved_ac = ac_preference_from_text(clean_rt)
        if resolved_ac:
            updates["ac_preference"] = resolved_ac
            updates["ac_qualified"] = True
        # "AC Deluxe" is a class AND a room name: the class is banked above,
        # and the name still has to go through or the alias resolver never
        # sees the room the guest actually asked for. A bare class
        # ("non - ac", "a/c room") is not a name and must not become one.
        if (
            not is_bare_ac_class(clean_rt)
            and not clean_rt.startswith("option ")
            and not clean_rt.isdigit()
            and clean_rt not in ("custom", "build your own")
        ):
            updates["room_type_preference"] = extracted.room_type_preference
        elif re.fullmatch(r"(?:option|opt|choice)\s*[1-9]\d*", clean_rt):
            # "Option 1" is not a room name, which is why it is kept out of
            # `room_type_preference` — but it is the model telling us which
            # line the guest picked. Guest 919999000111, 2026-10-19 10:54:
            # "it's best" -> `room_type_preference: 'Option 1'`, dropped
            # here, so the validator had nothing to match and redrew the
            # card. `selected_option_label` is per-turn state, so this
            # cannot leak into a later turn.
            updates["selected_option_label"] = clean_rt.split()[-1]
    if extracted.guest_name is not None:
        g_name = str(extracted.guest_name).strip()
        inquiry_words = ("vacancy", "job", "jobs", "manager", "hiring", "interview", "receptionist", "resume", "cv", "?")
        if (
            g_name.lower() not in NON_NAME_WORDS
            and len(g_name.split()) <= 4
            and not any(w in g_name.lower().split() for w in inquiry_words)
            and g_name.lower() not in ("hotel", "hotel sahu")
            and not is_inquiry_or_question
        ):
            updates["guest_name"] = g_name.title()
    if extracted.guest_email is not None:
        updates["guest_email"] = extracted.guest_email
    if extracted.guest_phone is not None:
        updates["guest_phone"] = extracted.guest_phone

    # Dynamic Language Switching / Detection (English is Primary Default)
    #
    # ── There are two language blocks in this function, and this one runs
    #    LAST, so whatever it writes wins ──────────────────────────────
    #
    # The newer block sits ~800 lines above and does the real lexical work.
    # This is the older one, never removed when the newer one was added,
    # and it carried a *second*, smaller copy of the cue logic plus a
    # *second* English default. On the turn a language is first
    # established `state["preferred_language"]` is still unset — so that
    # default overwrote the Hinglish the block above had just chosen.
    #
    # That is the mechanism behind "it still replies in english most of the
    # time": the newer block was doing its job and being clobbered here.
    # "mera naam Rahul hai" and "mujhe room book karna hai" matched nothing
    # in the regex below, fell through to the English default, and threw
    # away the Hinglish decision made 800 lines earlier.
    #
    # Under `LANGUAGE_DETECTION_WEIGHTS_HINDI` this block keeps only what
    # it alone can do — an explicit request, and Devanagari — and leaves
    # the cues and the default to the block that owns them.
    clean_raw_msg = (state.get("user_message") or "").strip().lower()
    if any(w in clean_raw_msg for w in ("in english", "english please", "speak in english", "english mein", "talk in english")) or clean_raw_msg == "english":
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"
    elif clean_raw_msg in ("hindi", "in hindi", "hindi please", "hindi mein", "hindi me", "हिंदी"):
        updates["preferred_language"] = "Hindi"
        updates["conversation_language"] = "Hindi"
    elif clean_raw_msg in ("hinglish", "in hinglish", "hinglish please"):
        updates["preferred_language"] = "Hinglish"
        updates["conversation_language"] = "Hindi"
    elif re.search(r'[\u0980-\u09FF]', state.get("user_message", "")) or clean_raw_msg in ("bengali", "in bengali", "bangla", "বাংলা"):
        updates["preferred_language"] = "Bengali"
        updates["conversation_language"] = "Bengali"
    elif re.search(r'[\u0900-\u097F]', state.get("user_message", "")):
        updates["preferred_language"] = "Hindi"
        updates["conversation_language"] = "Hindi"
    elif not _full_weights and re.search(r'\b(chahiye|kripya|bataiye|dijiye|hoga|rahega|kitna|kitne|kamra|kamre|dono|sasta|mehenga|batao|karo|karein|hain|nahi|haan)\b', clean_raw_msg):
        updates["preferred_language"] = "Hinglish"
        updates["conversation_language"] = "Hindi"
    elif not _full_weights and not state.get("preferred_language"):
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"
    if extracted.special_requests is not None:
        updates["special_requests"] = extracted.special_requests
    if extracted.payment_details is not None:
        pd_dict = extracted.payment_details.model_dump(exclude_none=True)
        if pd_dict:
            # Is this turn a receipt? Decided *before* the write, because it
            # decides what the write is allowed to do. It used to be computed
            # after, guarding everything except the overwrite it should have
            # guarded.
            user_msg = state.get("user_message", "")
            is_payment_image = (
                ("[IMAGE_ANALYSIS]" in user_msg and "Type: Payment Screenshot" in user_msg)
                or "[Image Analysis: Payment Proof]" in user_msg
                or "Type: Payment Screenshot" in user_msg
            )

            # A receipt replaces; a sentence fills gaps.
            #
            # Guest 919999000111 typed "Total payable: 17500/-" into a
            # complaint. The model tagged it as payment details and the whole
            # object was replaced: amount 4400 -> 17500 and transaction_id
            # T2610261142497376769073 -> None. Every later reply then quoted
            # "your uploaded payment of ₹17,500" and demanded the ₹380 gap.
            #
            # So a prose turn may only fill fields we do not already have. A
            # real receipt upload may still overwrite — that is a guest
            # correcting us with a document, not with arithmetic.
            from app.utils.payment_details_merge import kept_over_prose, merge_payment_details

            existing_pd = state.get("payment_details")
            if get_settings().PAYMENT_DETAILS_MERGE_GUARD_ENABLED:
                merged_pd = merge_payment_details(existing_pd, pd_dict, is_payment_image)
            else:
                # Rollback: last writer wins, which is the defect itself.
                merged_pd = pd_dict
            if merged_pd != (existing_pd or {}):
                updates["payment_details"] = merged_pd
                if not is_payment_image:
                    kept = kept_over_prose(existing_pd, pd_dict)
                    if kept:
                        logger.info(
                            "PAYMENT_DETAILS_KEPT | sess=%s fields=%s",
                            state.get("session_id"), sorted(kept),
                        )
            if is_payment_image:
                # Downstream Payment Verification Logic (Prompt 2 Task 4)
                p_amount = float(pd_dict.get("amount") or 0.0)
                p_status = str(pd_dict.get("status") or "").strip().lower()
                tot_price = float(state.get("total_price") or 0.0)
                exp_advance = round(tot_price * 0.30, 2) if tot_price > 0 else 0.0
                logger.info("PAYMENT_VERIFY_START | amount=%s | status=%s | expected_advance=%s | sess=%s", p_amount, p_status, exp_advance, state.get("session_id"))

                gap_adv = abs(p_amount - exp_advance) if exp_advance > 0 else 0.0
                gap_full = abs(p_amount - tot_price) if tot_price > 0 else 0.0
                min_gap = min(gap_adv, gap_full)

                is_success_status = p_status in ("success", "completed", "paid", "successful")

                if is_success_status and (min_gap <= 10.0 or p_amount >= (exp_advance - 10.0 if exp_advance > 0 else 1.0)):
                    logger.info("PAYMENT_VERIFY_RESULT | outcome=auto_confirmed | p_amount=%s | exp_advance=%s | gap=%s", p_amount, exp_advance, min_gap)
                    updates["booking_stage"] = "confirmed"
                    updates["payment_status"] = "paid"
                    updates["amount_paid"] = p_amount
                    updates["intent"] = "PAYMENT_STATUS"
                elif 10.0 < min_gap <= 500.0 and is_success_status:
                    logger.info("PAYMENT_VERIFY_RESULT | outcome=staff_review | p_amount=%s | exp_advance=%s | gap=%s", p_amount, exp_advance, min_gap)
                    updates["intent"] = "PAYMENT_STATUS"
                    updates["payment_status"] = "review_needed"
                elif p_status in ("pending", "processing", "failed"):
                    logger.info("PAYMENT_VERIFY_RESULT | outcome=pending_or_failed | p_status=%s", p_status)

                if not state.get("payment_missing_fields"):
                    from app.services.image_vision import _compute_missing_fields, CRITICAL_FIELDS, IMPORTANT_FIELDS
                    all_missing = _compute_missing_fields(pd_dict)
                    trackable = [f for f in all_missing if f in CRITICAL_FIELDS or f in IMPORTANT_FIELDS]
                    if trackable:
                        updates["payment_missing_fields"] = trackable
                        updates["payment_followup_index"] = 0
                        logger.info(f"Payment receipt missing fields: {trackable}")
    if extracted.partial_payment_amount is not None:
        updates["partial_payment_amount"] = extracted.partial_payment_amount
    if extracted.coupon_code is not None:
        updates["coupon_code"] = extracted.coupon_code
        changed_fields.append("coupon_code")
    
    if extracted.extra_services is not None:
        # Append newly extracted services to existing
        current_extras = state.get("extra_services", [])
        new_extras = [s for s in extracted.extra_services if s not in current_extras]
        if new_extras:
            updates["extra_services"] = current_extras + new_extras
            changed_fields.append("extra_services")

    # Process ignored/declined upsells
    last_upsell = state.get("last_upsell_offered")
    if last_upsell and (updates.get("intent") in ["CONFIRM_BOOKING", "PAYMENT_STATUS", "BOOK_ROOM"]):
        # If they didn't add it in this turn, consider it declined/ignored
        if last_upsell not in (extracted.extra_services or []):
            declined = state.get("declined_upsells", [])
            if last_upsell not in declined:
                updates["declined_upsells"] = declined + [last_upsell]
            
                from app.repositories.activity_repository import log_activity, ActivityLog
                import asyncio
                asyncio.create_task(log_activity(ActivityLog(
                    hotel_id=str(state.get("hotel_id", 0)),
                    session_id=state.get("session_id", ""),
                    booking_id=state.get("booking_id"),
                    activity_type="SYSTEM_EVENT",
                    description=f"UPSELL_DECLINED: {last_upsell}"
                )))
        updates["last_upsell_offered"] = None # consume

    # If any core search parameter changed, invalidate downstream context
    is_payment_receipt = (
        ("[IMAGE_ANALYSIS]" in state.get("user_message", "") and "Type: Payment Screenshot" in state.get("user_message", ""))
        or "[Image Analysis: Payment Proof]" in state.get("user_message", "")
        or "Type: Payment Screenshot" in state.get("user_message", "")
    )
    current_stage = state.get("booking_stage")
    ignore_changes = current_stage == "awaiting_payment" and (is_payment_receipt or updates.get("intent") == "PAYMENT_STATUS")

    if changed_fields and not ignore_changes:
        logger.info(f"Booking query change detected in: {changed_fields}.")
        updates["updated_fields"] = changed_fields

        # True structural/capacity changes that invalidate existing pricing & selected room
        hard_reset_fields = {"check_in", "check_out", "adults", "children", "num_rooms", "child_ages"}
        is_confirming = (updates.get("intent") in ("SELECT_ROOM", "CONFIRM_BOOKING") or result.intent in ("SELECT_ROOM", "CONFIRM_BOOKING"))
        requires_hard_reset = any(f in changed_fields for f in hard_reset_fields) and not is_option_number_selection and not is_confirming

        # Preference/taste changes (room_type_preference, feature_preferences,
        # meal_plan_preference, etc.).
        #
        # `budget` is deliberately NOT in this list. A guest who names a figure
        # mid-quote ("can you give me this room in 1800") has not changed their
        # taste in rooms — they have made an offer, and the answer to an offer
        # is a price, not a fresh search. While `budget` was listed here, every
        # such message wiped `total_price`/`selected_rooms`/`allocation_options`
        # and the bot replied "let me pull up the rooms and rates for those
        # nights" to a guest holding our quote. Measured 2026-11-05 (Sample Property,
        # the three challenges in `week-san-price-challenge`). `budget` is still
        # recorded on `updates`; it just no longer destroys the quote.
        preference_only_changed = not requires_hard_reset and any(
            f in changed_fields for f in ["room_type_preference", "feature_preferences", "meal_plan_preference", "recommendation_preference", "display_preference", "room_count_preference", "comfort_preference"]
        )

        # If user changed room preference, AC preference, or budget, wipe selected rooms and trigger fresh search
        if preference_only_changed and not is_option_number_selection and not is_confirming:
            logger.info("Room preference change detected. Triggering fresh room search and wiping stale selections.")
            updates["selected_rooms"] = []
            updates["allocation_options"] = []
            updates["price_breakdown"] = None
            updates["total_price"] = None
            updates["payment_link"] = None
            updates["payment_link_id"] = None
            updates["payment_link_amount"] = None
            updates["force_new_search"] = True
            updates["booking_stage"] = "collecting"
            updates["next_action"] = "search"
        elif requires_hard_reset:
            # Always invalidate pricing and payment link when dates or capacity change
            updates["price_breakdown"] = None
            updates["total_price"] = None
            updates["payment_link"] = None
            updates["payment_status"] = None
            updates["booking_id"] = None
            updates["room_type_not_available"] = False
            updates["allocation_options"] = []
            updates["force_new_search"] = True

            # Check if TRUE capacity parameters changed (adults, children, num_rooms, child_ages)
            capacity_fields = {"adults", "children", "num_rooms", "child_ages"}
            capacity_changed = any(f in changed_fields for f in capacity_fields) and not is_option_number_selection and not is_confirming

            if capacity_changed:
                logger.info("Capacity parameters changed. Wiping selected rooms and resetting upsell.")
                updates["selected_rooms"] = []
                updates["available_rooms"] = []
                # A room preference is pinned against a party size, so a
                # change to the party makes it stale: "a deluxe room" names a
                # different room for two people than for six, and one the
                # guest chose for five says nothing about six. Left in place
                # it kept narrowing the search that followed — `room_searcher`
                # reads it and `filter_rooms_by_reference` allocates over
                # only the rooms it matched — which is how a party that grew
                # by one child went on being quoted the room it had outgrown.
                #
                # Not when the guest restated it in this same message: then
                # it is in `changed_fields` too, and it is current.
                if (
                    get_settings().PARTY_CHANGE_DROPS_ROOM_PREFERENCE
                    and state.get("room_type_preference")
                    and "room_type_preference" not in changed_fields
                ):
                    logger.info(
                        "Capacity changed; dropping the carried-over room_type_preference=%r.",
                        state.get("room_type_preference"),
                    )
                    updates["room_type_preference"] = None
                updates["booking_stage"] = "collecting"
                if updates.get("next_action") != "need_subgroups":
                    updates["next_action"] = "search"
                updates["ambiguous_guests"] = None
                updates["upsell_shown"] = False

                # If room_requirements existed from a previous turn, but user changed guests/rooms without providing new requirements
                if getattr(extracted, "room_requirements", None) is None:
                    current_num_rooms = state.get("num_rooms") or 0
                    current_adults = updates.get("adults", state.get("adults", 0))
                    if current_adults and current_num_rooms and current_adults < current_num_rooms:
                        logger.info(f"Guest count ({current_adults}) is less than previous room count ({current_num_rooms}). Resetting room_requirements & num_rooms.")
                        updates["room_requirements"] = None
                        updates["num_rooms"] = None
            else:
                is_checking_avail = (updates.get("intent") == "CHECK_AVAILABILITY" or result.intent == "CHECK_AVAILABILITY")
                if is_checking_avail:
                    logger.info("Availability check query detected; wiping stale selected rooms for fresh search.")
                    updates["selected_rooms"] = []
                    updates["allocation_options"] = []
                    updates["force_new_search"] = True
                    updates["booking_stage"] = "collecting"
                    updates["next_action"] = "search"
                    if "room_type_preference" not in changed_fields:
                        updates["room_type_preference"] = None
                    if "ac_preference" not in changed_fields and not ac_preference_from_text(user_msg_clean):
                        updates["ac_preference"] = None
                        updates["ac_qualified"] = False
                elif state.get("selected_rooms"):
                    logger.info("Only date parameters changed on existing draft. Keeping selected rooms and showing quote.")
                    updates["booking_stage"] = "rooms_shown"
                    updates["next_action"] = "show_quote"
                else:
                    updates["booking_stage"] = "collecting"
                    updates["next_action"] = "search"

        if (updates.get("intent") == "CHECK_AVAILABILITY" or result.intent == "CHECK_AVAILABILITY") and not is_confirming:
            updates["selected_rooms"] = []
            updates["allocation_options"] = []
            updates["force_new_search"] = True
            updates["booking_stage"] = "collecting"
            updates["next_action"] = "search"
            if "room_type_preference" not in changed_fields:
                updates["room_type_preference"] = None
            if "ac_preference" not in changed_fields and not ac_preference_from_text(user_msg_clean):
                updates["ac_preference"] = None
                updates["ac_qualified"] = False

        if updates.get("intent") not in ("PRICE_MATCH", "CONFIRM_BOOKING", "CANCEL_BOOKING", "CHECK_AVAILABILITY") and (updates.get("intent") in ("BOOKING_SUMMARY", "UNKNOWN", "GREETING") or resolved_intent in ("BOOKING_SUMMARY", "UNKNOWN", "GREETING")):
            logger.info(f"Changed intent from {resolved_intent} to BOOK_ROOM because core booking parameters were modified.")
            updates["intent"] = "BOOK_ROOM"
    else:
        if ignore_changes and changed_fields:
            logger.info(f"Ignoring booking query changes {changed_fields} because user uploaded a payment receipt.")
        updates["updated_fields"] = []