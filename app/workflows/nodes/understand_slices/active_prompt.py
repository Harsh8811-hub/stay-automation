"""The zero-LLM handlers for an open prompt, out of `understand`.

Extracted from `app/workflows/nodes/understand.py` on 2026-10-28. Both entry
points are **pure moves**: the statements are the original lines, in the same
order, with the same names and the same comments — verified by comparing the
AST of each moved block against the AST of its function before the swap.

There are two functions rather than one because the handlers are not all
evaluated at the same point in `understand`. `handle_active_prompt` runs early,
before the confirmed / checked-in block; `handle_change_menu_prompt` runs after
it. Merging them would reorder two prompt answers against a guest who is
already booked, which is a behaviour change and not part of a move.

Why it was worth moving. `understand` was a 3,600-line function with a single
rule applied at the bottom (the stage invariant) and many exits above it. These
blocks are twenty-two of those exits, and they are all the same shape: the bot
asked the guest a question, the guest answered it, and the answer is read
without the model. Keeping them beside the caller made it look as though the
rest of the node could run on a turn where a prompt was open; it cannot — every
one of these returns.

The contract:
  * read `state` and the cleaned message, assign nothing into `state`;
  * return the update dict to hand back to the graph, or `None` for "not an
    answer to the open prompt, carry on with the node";
  * no `await`.

`None` is the sentinel because every one of these branches returns a dict. The
caller treats `None` as "fall through", which is what running off the end of the
block did before.

Nothing here is re-implemented, tidied, or renamed. Behaviour changes belong in
a separate commit, so that a regression can be attributed to one or the other.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.utils.ac_text import ac_preference_from_prompt_reply, is_bare_ac_class
from app.utils.button_registry import resolve_button

# The same logger object the node uses, by name rather than by import: importing
# it from `understand` would be a cycle, and a fresh `getLogger(__name__)` would
# move these records to a different name, which log routing keys on.
logger = logging.getLogger("app.workflows.nodes.understand")


def handle_active_prompt(
    state: dict[str, Any],
    user_msg_clean: str,
    active_prompt_action: str | None,
) -> dict[str, Any] | None:
    """Answer an open prompt without the model. `None` means "not an answer"."""
    # ── S3: ZERO-LLM FAST PATH FOR AC PREFERENCE QUALIFIER ──
    #
    # This path RETURNs, so it is the only handler that runs on the live channel.
    # The `expected_action == "need_ac_preference"` block in `understand` catches
    # the turns this one lets through — and lets through is the point: a miss
    # here is not "assume AC", it is "ask again", which is why the two blocks
    # have to agree on what counts as an answer.
    #
    # The two are now written in terms of the same reader
    # (`app.utils.ac_text`), which is what stops them drifting apart again.
    #
    # All three answers set `ac_preference` and `ac_qualified` and go straight to
    # `search`; the priced card is what the guest sees next.
    #
    # The ordinals and the words are read by the same matcher, so "non - ac",
    # "non_ac", "a/c" and "bina ac" arrive here exactly as "2" does. An answer
    # it cannot read does not become AC by default — control falls through to
    # the `expected_action` block in `understand`, which asks again.
    if (
        active_prompt_action == "need_ac_preference"
        and not state.get("ac_qualified")
        and is_bare_ac_class(user_msg_clean)
    ):
        resolved = ac_preference_from_prompt_reply(user_msg_clean)
        if resolved == "AC":
            feat_list = list(set((state.get("feature_preferences") or []) + ["AC"]))
            return {
                **state,
                "ac_preference": "AC",
                "ac_qualified": True,
                "feature_preferences": feat_list,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif resolved == "NON_AC":
            feats = [f for f in (state.get("feature_preferences") or []) if f != "AC"] + ["Non-AC"]
            return {
                **state,
                "ac_preference": "NON_AC",
                "ac_qualified": True,
                "feature_preferences": list(set(feats)),
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif resolved == "NO_PREFERENCE":
            return {
                **state,
                "ac_preference": "NO_PREFERENCE",
                "ac_qualified": True,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── The mattress question, answered ──
    #
    # Asked by `validator` after the guest picks a card row whose rooms can take
    # a mattress. This was a disclosure until 2026-09-25 — the bed was already in
    # the row's price and the question existed to let the guest take it off. It
    # is an offer now, because the row no longer contains one: `distribute_guests`
    # seats adults to the room's `max_adults` and prices a mattress only when the
    # guest asks. So both answers re-price the same set, and only the Rs 600 line
    # moves.
    #
    #   "no"  → the same rooms, without the mattress line.
    #   "yes" → the same rooms, with it.
    #   None  → unreadable. Keep the row, no mattress, no charge: the cheap
    #           direction, and the one that cannot bill a guest for something
    #           they did not ask for. Silence is not consent to being charged.
    #
    # "The same set either way" is the fix. On "no" this used to re-search
    # without mattresses, which also removed the berth the row was using, so a
    # party of three in a Deluxe King was answered with two rooms Rs 2,250 more
    # expensive. A mattress is not a berth — see `distribute_guests`.
    if active_prompt_action == "need_extra_bed":
        from app.utils.extra_bed_intent import REFUSE, answer_bed_question

        answer = answer_bed_question(user_msg_clean)
        if answer is None:
            # Unreadable — "Final", an emoji, a question about something else.
            # This used to drop `next_action` and fall through with no bed, which
            # let the *next* node re-derive the guest's room type from nothing:
            # measured on prod 2026-10-17, a guest who had just picked PREMIER
            # ROOM KING BED said "Final" and was told "we don't have an AC king
            # bed room". Ask the question again instead — it is still the one
            # thing standing between them and a quote, and no bed is charged
            # while it goes unanswered.
            #
            # Once, not forever. Guest 919088198242 (2026-10-19 11:15–11:16)
            # answered this prompt with "1550", "And 1 Ac room", "2500",
            # "Standerd non ac 1 room" and "And ac delux room king bed" — five
            # booking turns, none of them a yes/no. Three identical re-asks in
            # a row then tripped the repeat guard, and the next turn — a room
            # the guest was naming — was answered with a support ticket. The
            # question may interrupt a guest once; it may not hold them hostage.
            if state.get("extra_bed_reasked"):
                logger.info(
                    "EXTRA_BED_UNREADABLE_TWICE | input=%r — proceeding without the bed",
                    user_msg_clean,
                )
                return {
                    **state,
                    "no_extra_bed": True,
                    "wants_bed": False,
                    "extra_bed_preference": "unreadable",
                    "intent": "BOOK_ROOM",
                    "booking_stage": "rooms_shown",
                    "next_action": "show_quote",
                    "extracted_data": None,
                    "extraction_result": None,
                }
            logger.info(
                "EXTRA_BED_UNREADABLE | input=%r — re-asking once, rather than guessing",
                user_msg_clean,
            )
            return {
                **state,
                "wants_bed": False,
                "next_action": "need_extra_bed",
                "last_prompt_action": "need_extra_bed",
                "extra_bed_reasked": True,
                "extracted_data": None,
                "extraction_result": None,
            }
        wants = answer != REFUSE
        logger.info(
            "EXTRA_BED_ANSWERED | input=%r wants_bed=%s",
            user_msg_clean,
            wants,
        )

        if wants:
            # Accepting has to go back through the allocator, and only for the
            # reason that the charge is not a field this node can write: it is
            # what `distribute_guests(..., wants_bed=True)` puts on the option
            # the guest then picks, read off the room's own `extra_adult_rate`
            # (allocation_engine.py:656). The rooms the guest is holding were
            # allocated without a bed, so adding one here would mean
            # reimplementing that pricing rule in a second place — the drift
            # that put two copies of the calendar lookup in the codebase once
            # already. Re-searching with `wants_bed=True` returns the same set
            # with the bed priced on its rows, which is what the guest picks
            # from. This is unchanged from before: the accept path is the one
            # way the mattress has ever reached a bill.
            return {
                **state,
                "no_extra_bed": False,
                "extra_bed_preference": answer,
                "wants_bed": True,
                "selected_rooms": [],
                "allocation_options": [],
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }

        # Declining needs no allocation at all — `quote_generator` zeroes the
        # mattress line straight off `no_extra_bed` (quote_generator.py:82), so
        # the guest's own pick can be priced exactly as it stands. Keeping it is
        # the fix. Clearing `selected_rooms` / `allocation_options` and setting
        # `next_action: "search"` sent the turn back through `room_searcher`,
        # which re-rendered the whole room card the guest had just chosen from:
        # they answered a yes/no and got the menu again. `extra_bed_asked` is
        # already True — `validator` set it on the turn it asked — so returning
        # to `validator` with the pick intact takes the `else` arm at
        # validator.py:448 and goes straight to `show_quote`.
        return {
            **state,
            "no_extra_bed": True,
            "extra_bed_preference": answer,
            "wants_bed": False,
            "intent": "BOOK_ROOM",
            "booking_stage": "rooms_shown",
            "next_action": "show_quote",
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── S5: ZERO-LLM FAST PATH FOR COMPOSITION PREFERENCE QUALIFIER ──
    if active_prompt_action == "need_composition_preference":
        room_match = re.search(r'\b(\d+)\s*(?:rooms?|kamre?|kamra)\b', user_msg_clean)
        budget_match = re.search(r'\b(?:budget|within|under|tak)?\s*(\d{3,6})\s*(?:budget|tak|rs|inr)?\b', user_msg_clean)
        if room_match:
            return {
                **state,
                "composition_asked": True,
                "num_rooms": int(room_match.group(1)),
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif "budget" in user_msg_clean and budget_match:
            return {
                **state,
                "composition_asked": True,
                "budget": float(budget_match.group(1)),
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("1", "1.", "option 1", "option1", "opt 1") or any(w in user_msg_clean for w in ("kam kharcha", "best price", "cheapest", "sasta", "extra bed", "kam room", "fewer room", "low budget")):
            return {
                **state,
                "composition_asked": True,
                "occupancy_preference": "MAX",
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("2", "2.", "option 2", "option2", "opt 2") or any(w in user_msg_clean for w in ("aaram se", "most comfortable", "comfortable", "comfort", "zyada room", "more room", "apna bed", "own bed", "dedicated bed", "no extra bed")):
            return {
                **state,
                "composition_asked": True,
                "occupancy_preference": "BASE",
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        # "Split by family" is no longer on the menu (2026-09-19), so the bare
        # numbers "3" / "option 3" that used to select it are gone — we do not
        # honour a number we do not print. The free-text wording stays, because
        # the intent is real and a guest who types it is asking for something we
        # can still do; they simply have to say it rather than number it.
        elif any(w in user_msg_clean for w in ("family alag", "split", "separate", "alag", "bade alag", "subgroups")):
            from app.prompts import templates
            lang = state.get("preferred_language") or "Hinglish"
            return {
                **state,
                "composition_asked": True,
                "grouping_mode": "SEGMENTED",
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "need_subgroups",
                "bot_response": templates.ask_family_subgroups_followup(lang=lang),
                "extracted_data": None,
                "extraction_result": None,
            }
        # The old option 4 ("You decide") is gone from the menu, and the branch
        # is gone with it: it set `occupancy_preference = None` and searched,
        # which is exactly what an unrecognised reply does by falling through.
        # Keeping it would only have re-routed "suggest" / "options" / "aap
        # decide karo" into a branch that does the same thing as no branch.

    # ── S6: ZERO-LLM FAST PATH FOR ID CHECK-IN VS NEW BOOKING ──
    if active_prompt_action == "ask_booking_or_checkin":
        import datetime
        guest_name = state.get("guest_name") or "Guest"
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        date_display = now_dt.strftime("%d %b %Y")

        # 1. Check if user provides stay details (dates, guest count, nights)
        has_booking_criteria = bool(re.search(r'\b(?:adults?|children|child|sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug|\d{1,2}[/-]\d{1,2}|night|nights|room\s*only|with\s*breakfast)\b', user_msg_clean))
        is_new_booking_phrase = any(w in user_msg_clean for w in (
            "new booking", "make a booking", "want to book", "book a room",
            "new reservation", "booking", "book", "reserve", "reservation"
        ))
        room_match = (
            re.search(r'\b(?:room|room\s*no\.?|check\s*in|checking\s*in|checkin|kamra|kamre)\s*(?:is\s*|no\.?\s*|number\s*)?([0-9]{1,4}[a-zA-Z]?)\b', user_msg_clean)
            or re.search(r'^\s*([0-9]{1,4}[a-zA-Z]?)\s*[.,!]?\s*$', user_msg_clean)
        )
        hotel_name = state.get("hotel_name") or (state.get("hotel_config") or {}).get("hotel_name") or "Sample Hotel"
        
        if not has_booking_criteria and not is_new_booking_phrase and room_match and (room_match.group(1).isdigit() or len(room_match.group(1)) <= 4):
            room_num = room_match.group(1).strip().upper()
            # Double check it's not "1" or "2" meant as option choice unless preceded by explicit room keyword
            has_room_word = bool(re.search(r'\b(room|kamra|kamre|check\s*in|checking\s*in|checkin)\b', user_msg_clean))
            if room_num in ("1", "2") and not has_room_word:
                pass  # let standard logic evaluate
            else:
                formatted_room = f"Room {room_num}" if not room_num.lower().startswith("room") else room_num
                verified_ids = state.get("verified_ids") or []
                if len(verified_ids) > 1:
                    cards_bullets = "\n".join([
                        f"• **Guest {i+1}:** {c.get('guest_name', 'Guest')} — {c.get('doc_label', 'Government ID')}" + (f" ({c.get('id_masked_number')})" if c.get('id_masked_number') else "")
                        for i, c in enumerate(verified_ids)
                    ])
                    msg = (
                        f"✅ **Check-in Recorded!**\n\n"
                        f"Thank you, **{guest_name}**! {len(verified_ids)} IDs have been linked to **{formatted_room}** on {date_display}:\n\n"
                        f"{cards_bullets}\n\n"
                        f"Have a comfortable and pleasant stay at {hotel_name}! 🏨\n"
                        f"If you need room service, WiFi info, or anything else, feel free to ask."
                    )
                else:
                    msg = (
                        f"✅ **Check-in Recorded!**\n\n"
                        f"Thank you, **{guest_name}**! Your ID has been linked to **{formatted_room}** on {date_display}.\n\n"
                        f"Have a comfortable and pleasant stay at {hotel_name}! 🏨\n"
                        f"If you need room service, WiFi info, or anything else, feel free to ask."
                    )
                logger.info("CHECKIN_RECORDED | name=%s | room=%s | date=%s | total_ids=%d", guest_name, formatted_room, now_str, len(verified_ids))
                return {
                    **state,
                    "intent": "GREETING",
                    "room_number": formatted_room,
                    "id_received_date": now_str,
                    "booking_stage": "checked_in",
                    "next_action": None,
                    "bot_response": msg,
                    "extracted_data": None,
                    "extraction_result": None,
                }

        # 2. User says "checking in" / "check in" without room number
        if any(w in user_msg_clean for w in ("check in", "checking in", "checkin", "check-in")) and not has_booking_criteria:
            return {
                **state,
                "intent": "GREETING",
                "next_action": "ask_booking_or_checkin",
                "bot_response": f"Welcome, **{guest_name}**! Please share your **Booking ID** so I can link your ID for check-in. 🏨",
                "extracted_data": None,
                "extraction_result": None,
            }

        # 3. User says "new booking" / "booking" without dates yet
        is_new_booking_phrase = any(w in user_msg_clean for w in ("new booking", "make a booking", "want to book", "book a room", "new reservation", "booking", "book"))
        if is_new_booking_phrase and not has_booking_criteria:
            return {
                **state,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "need_dates",
                "bot_response": "Great! Please share your **stay dates** and **number of guests** (e.g., *22-23 Sep, 2 adults*) so I can show you our best room options! 🏨",
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── S7: ZERO-LLM FAST PATH FOR PAYMENT RECEIPT FOLLOWUP ──
    if active_prompt_action == "payment_followup":
        declined_followup = any(w in user_msg_clean for w in ("not now", "no", "nahi", "skip", "none", "no thanks", "later", "nhi", "bad me", "baad me", "not needed", "dont need"))
        
        # Extract meal plan if guest specified one
        meal_plan_val = None
        if "breakfast" in user_msg_clean:
            meal_plan_val = "CP"
        elif any(w in user_msg_clean for w in ("dinner", "lunch", "map")):
            meal_plan_val = "MAP"
        elif any(w in user_msg_clean for w in ("all meals", "ap", "full board")):
            meal_plan_val = "AP"
        elif any(w in user_msg_clean for w in ("room only", "ep", "no food")):
            meal_plan_val = "EP"
        elif declined_followup:
            meal_plan_val = "EP"

        booking_id = state.get("booking_id") or state.get("active_booking_id")
        from app.utils.helpers import guest_is_confirmed

        if not guest_is_confirmed(state, booking_id):
            # Ghost booking guard
            return {
                **state,
                "booking_stage": "lead",
                "lifecycle_stage": "lead",
                "next_action": None,
                "bot_response": "Thank you! If you need any assistance with a room booking or stay in the city, please let us know. 🙏",
                "extracted_data": None,
                "extraction_result": None,
            }
        guest_name = state.get("guest_name") or "Guest"
        check_in = state.get("check_in") or "Your stay"
        check_out = state.get("check_out") or ""
        room_name = state.get("room_type_preference") or "Standard Room"
        selected_rooms = state.get("selected_rooms") or []
        if selected_rooms and isinstance(selected_rooms[0], dict):
            room_name = selected_rooms[0].get("room_name", room_name)

        dates_disp = f"{check_in} to {check_out}" if check_out else check_in
        # NOT `or "Sample Hotel"`. This used to name Sample Hotel for every tenant
        # whose session had not stamped `hotel_name` yet, so a Hotel Paradise
        # guest was congratulated on booking somewhere else. A missing name is
        # smaller than a wrong one — the copy below reads correctly without it.
        h_name = str(state.get("hotel_name") or "").strip()
        welcome_line = f"We look forward to welcoming you to {h_name}! 🏨" if h_name else "We look forward to welcoming you! 🏨"
        msg = (
            f"✅ **Booking Confirmed!**\n\n"
            f"*Booking ID:* `{booking_id}`\n"
            f"*Guest:* **{guest_name}**\n"
            f"*Dates:* {dates_disp}\n"
            f"*Room:* {room_name}\n"
            f"*Meal Plan:* {'Room Only' if meal_plan_val in ('EP', None) else meal_plan_val}\n\n"
            f"{welcome_line}\n"
            f"You can upload your ID card anytime for express check-in, or ask me for directions, amenities, or stay assistance."
        )

        return {
            **state,
            "intent": "PAYMENT_STATUS",
            "booking_stage": "confirmed",
            "next_action": None,
            "payment_missing_fields": [],
            "meal_plan_preference": meal_plan_val,
            "bot_response": msg,
            "extracted_data": None,
            "extraction_result": None,
        }


def handle_change_menu_prompt(
    state: dict[str, Any],
    user_msg_clean: str,
    active_prompt_action: str | None,
) -> dict[str, Any] | None:
    """The two prompt handlers that sit below the confirmed-booking block.

    They cannot be part of `handle_active_prompt`: that one runs before the
    confirmed / checked-in block, these run after it, and merging them would
    reorder the two against a guest who is already booked. `None` means "not
    one of these prompts, carry on".

    Moved here verbatim on 2026-10-28 (DEFECTS.md #25).
    """
    # ── M3: RESPONSES TO 'change' MENU ──
    #
    # Four rows, so the menu is a list and a tap comes back as the row's **title**
    # ("Stay Dates"), not a number. `resolve_button` is the one place that says what that
    # title means, so this reader no longer keeps its own copy of the wording — the copy
    # that used to be here accepted four numbers and a handful of typed words, and the
    # template's labels were a third list that nothing checked against either of them.
    #
    # The digits are not kept as an alias, though the rows were numbered until today. They
    # would not have worked: a bare "1" at `rooms_shown` is claimed by the SELECT_ROOM
    # handler in `apply_intent_overrides`, which runs first and reads any bare number as a
    # room option. So a guest tapping "1" on the old change menu was selecting a room, not
    # changing dates — measured 2026-10-15, and it is the reason this menu became a list
    # rather than a fourth place to keep numbers in step.
    #
    # A guest with the old numbered menu still on screen is no worse off than they were
    # yesterday, because that reply never reached this branch either.
    #
    # This guard read `state["next_action"]`, which `load_context` resets to None at the
    # start of every turn — so the branch was unreachable in production for as long as it
    # has existed (`9282250`), and every row and every typed word on the menu fell through
    # to the model. Measured 2026-10-15: tapping "Meal Plan" came back as a re-quoted room
    # card, and `next_action` on disk said "change_menu_shown" while this expression was
    # None. It reads `active_prompt_action` now — the same next_action-or-last_prompt_action
    # fallback the working prompt readers (AC, extra bed, composition) have always used.
    if active_prompt_action == "change_menu_shown":
        from app.prompts import templates
        _change_tap = resolve_button(user_msg_clean)
        _change_action = _change_tap.action if _change_tap else None
        if _change_action == "CHANGE_DATES" or user_msg_clean in ("1", "1.", "dates", "date", "tarikh", "change dates", "check in", "check out", "checkin", "checkout"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_change_dates",
                "updated_fields": [],
                "bot_response": templates.ask_change_dates(lang=state.get("preferred_language") or "English"),
            }
        elif _change_action == "CHANGE_GUESTS" or user_msg_clean in ("2", "2.", "guests", "guest", "room count", "rooms", "guests / room count", "log", "kamre", "change guests", "change rooms"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_change_guests",
                "updated_fields": [],
                "bot_response": templates.ask_change_guests(lang=state.get("preferred_language") or "English"),
            }
        elif _change_action == "CHANGE_AC" or user_msg_clean in ("3", "3.", "ac", "non-ac", "non ac", "ac / non-ac", "ac/non-ac", "change ac", "ac change", "ac room", "ac rooms"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_ac_preference",
                "updated_fields": [],
                "bot_response": templates.ask_ac_preference(lang=state.get("preferred_language") or "English"),
            }
        elif _change_action == "CHANGE_MEAL" or user_msg_clean in ("4", "4.", "meal plan", "meals", "meal", "khana", "breakfast", "change meal plan"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_meal_plan_preference",
                "updated_fields": [],
                "bot_response": templates.ask_meal_plan_preference(lang=state.get("preferred_language") or "English"),
            }

    # ── M3: MEAL PLAN PREFERENCE QUALIFIER RESPONSE ──
    # Reads the surviving pointer, not `next_action` — see the note on the change-menu
    # reader above. This is where the change menu's "Meal Plan" row lands, so it was dead
    # for the same reason and by the same mechanism.
    if active_prompt_action == "need_meal_plan_preference":
        if user_msg_clean in ("1", "1.", "room only", "ep", "only room", "without breakfast", "no meals"):
            return {
                **state,
                "meal_plan_preference": "EP",
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "force_new_search": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "booking_stage": "collecting",
                "updated_fields": ["meal_plan_preference"],
            }
        elif user_msg_clean in ("2", "2.", "with breakfast", "breakfast", "cp", "nashta", "bed and breakfast"):
            return {
                **state,
                "meal_plan_preference": "CP",
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "force_new_search": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "booking_stage": "collecting",
                "updated_fields": ["meal_plan_preference"],
            }
        elif user_msg_clean in ("3", "3.", "breakfast & dinner", "breakfast and dinner", "map", "dinner", "all meals", "ap"):
            return {
                **state,
                "meal_plan_preference": "MAP",
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "force_new_search": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "booking_stage": "collecting",
                "updated_fields": ["meal_plan_preference"],
            }

    return None
