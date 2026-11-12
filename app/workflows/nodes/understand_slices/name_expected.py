"""The reply ladder for a turn where the bot is waiting on a name, out of
`understand`.

Extracted from `app/workflows/nodes/understand.py` on 2026-10-28 (DEFECTS.md
#25). This is a **pure move**: the statements are the original lines, in the
same order, with the same names and the same comments -- verified by comparing
the AST of the moved block against the body of this function before the swap.

Unlike the earlier slices this one returns nothing: it has no `return` and no
`state` writes. It writes `updates`, which is the dict the caller already holds,
so the mutation crosses the boundary for free. It does, however, `await` three
times (`extract_room_request`, `generate_lead_id`, `create_lead`), so the
function is `async` and the caller awaits it.

The guard moves with the body because the `elif`s pair with it -- the first
branch contains a nested `if`, and `elif expected_action == "need_child_ages"`
belongs to the outer statement, not to that nested one. Splitting the guard off
would move every `elif` a level deeper and change what they are tested against.

`_additive_request_text` is a module-level helper of `understand`; it is passed
in and bound to its own name at the top, so every moved line stays byte
identical and no import cycle is created.

`extract_room_request` is passed in as well, and deliberately *not* imported
here. Nine tests in `tests/` spy on that name by patching it on the
`understand` module -- patching only intercepts calls made through the
module's own namespace. Importing it directly into this file would leave those
spies blind and send the real Gemini call out during the suite: that is
exactly what happened when this slice first landed, and only one of the nine
failed loudly. Taking the callable as a parameter means the caller resolves it
from its own globals on every turn, so `patch.object(understand,
"extract_room_request", stub)` keeps working.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from app.config.settings import get_settings
from app.services.llm_errors import LLMUnavailableError
from app.utils.ac_text import ac_preference_from_prompt_reply, is_plausible_ac_answer
from app.utils.person_name import extract_person_name
from app.utils.room_requirement_parse import (
    is_an_explicit_split,
    reconcile_split_with_party,
    split_party_from_message,
)
from app.workflows.nodes.understand_slices.booking_intent import _apply_split_party

# Same logger object as the node, by name: importing it would be a cycle.
logger = logging.getLogger("app.workflows.nodes.understand")


async def _get_full_catalogue(state: dict[str, Any]) -> list[str]:
    from app.services.room_service import room_names_from_options
    names = room_names_from_options(state.get("allocation_options"))
    try:
        from app.repositories import room_repository
        hotel_rooms = await room_repository.get_all_rooms(state.get("hotel_id", 0))
        for hr in hotel_rooms or []:
            short = str(getattr(hr, "name", "") or "").replace("ROOM", "").replace("Room", "").strip()
            if short and short not in names:
                names.append(short)
    except Exception:
        pass
    return names


async def apply_expected_action_reply(
    state: dict[str, Any],
    updates: dict[str, Any],
    user_msg_clean: str,
    expected_action: str | None,
    is_name_expected: bool,
    is_inquiry_or_question: bool,
    is_side_intent: bool,
    GREETING_WORDS: set[str],
    NON_NAME_WORDS: set[str],
    additive_request_text: Any,
    extract_room_request: Callable[..., Awaitable[Any]],
) -> None:
    """Read a reply to a pending name / child-age / room-request prompt."""
    _additive_request_text = additive_request_text
    if is_name_expected and not is_inquiry_or_question and not is_side_intent and user_msg_clean not in GREETING_WORDS:
        if user_msg_clean in ("confirm", "confirm booking", "confirm kar do", "kar do", "yes", "haan", "ha", "y", "correct", "theek hai", "ji", "proceed", "pay", "ok", "okay", "done"):
            existing_name = state.get("guest_name") or state.get("prior_guest_name") or state.get("customer_name")
            if existing_name:
                updates["guest_name"] = existing_name
            updates["intent"] = "CONFIRM_BOOKING"
        elif updates.get("guest_name") and user_msg_clean not in NON_NAME_WORDS and not any(user_msg_clean.startswith(w) for w in ("book ", "i want", "need room", "room from", "can i book")):
            updates["intent"] = "CONFIRM_BOOKING"
        elif (
            get_settings().NAME_CAPTURE_READS_ONLY_THE_NAME
            and user_msg_clean not in NON_NAME_WORDS
            and not any(user_msg_clean.startswith(w) for w in ("book ", "i want", "need room", "room from", "can i book"))
        ):
            # ── Read the name, not the message ───────────────────────────
            # Product, 2026-09-30: the name prompt "takes the whole thing no
            # matter what the customer sent". `extract_person_name` scans for a
            # name and returns `None` when there is not one, where the branch
            # below accepted any short digit-free message — which is how
            # "hi i am rahul" was saved as "Hi I Am Rahul".
            #
            # `None` is left as `None`: no name is captured and the prompt
            # stands, so the guest is asked again. That is the safe direction —
            # a guest called "Looking" is not recoverable by asking again.
            _name = extract_person_name(state.get("user_message", "") or "")
            if _name:
                updates["guest_name"] = _name
                updates["intent"] = "CONFIRM_BOOKING"
        elif user_msg_clean not in NON_NAME_WORDS and not any(user_msg_clean.startswith(w) for w in ("book ", "i want", "need room", "room from", "can i book")):
            clean_name = state.get("user_message", "").strip()
            for prefix in ("my name is", "i am", "i'm", "naam hai", "name is", "name:"):
                if clean_name.lower().startswith(prefix):
                    clean_name = clean_name[len(prefix):].strip(" :,-")
            
            # Strict human name validator (Sir's Spec / D11)
            # Rejects sentences, payment phrases, numbers, and inquiry words
            cand_lower = clean_name.lower()
            payment_or_sentence_words = (
                "payment", "paid", "diya", "kar diya", "bhej diya", "transfer", "screenshot", 
                "gpay", "phonepe", "paytm", "upi", "qr", "receipt", "advance", "rupaye", "rs", "inr",
                "hai", "hain", "karna", "karein", "chahiye", "chahta", "room", "hotel", "stay", 
                "booking", "please", "help", "batao", "batayein", "cancel", "change", "what", "where",
                "vacancy", "job", "jobs", "hiring", "resume", "cv", "?"
            )
            words = cand_lower.split()
            is_valid_name = (
                bool(clean_name)
                and 1 <= len(words) <= 4
                and not any(ch.isdigit() for ch in clean_name)
                and not any(w in words for w in payment_or_sentence_words)
                and not any(pw in cand_lower for pw in ("kar diya", "payment", "bhej diya", "hotel sahu"))
                and bool(re.match(r"^[A-Za-z.\s\'-]{2,50}$", clean_name))
            )
            if is_valid_name and cand_lower not in NON_NAME_WORDS:
                updates["guest_name"] = clean_name.title()
                updates["intent"] = "CONFIRM_BOOKING"

    elif expected_action == "need_child_ages":
        # Check if user replied with child ages (e.g. "8", "13", "4 and 10", "child is 6", "1.5 saal", "4 years", "1 child 4 years")
        from app.utils.helpers import extract_child_ages_from_text
        found_ages = extract_child_ages_from_text(user_msg_clean)
        if not found_ages:
            nums = [int(float(n)) for n in re.findall(r'\b\d+(?:\.\d+)?\b', user_msg_clean)]
            # If user wrote "1 child 4 years", nums is [1, 4]. If state already knows children=1, the age is 4!
            total_kids = int(state.get("children") or 1)
            if len(nums) > total_kids and nums[0] == total_kids:
                nums = nums[1:]
            found_ages = [n for n in nums if 0 <= n <= 17]

        total_kids = int(state.get("children") or len(found_ages) or 1)
        if found_ages:
            existing_ages = list(state.get("child_ages") or [])
            if len(found_ages) >= total_kids:
                updates["child_ages"] = found_ages[:total_kids]
            elif existing_ages and (len(existing_ages) + len(found_ages) <= total_kids):
                updates["child_ages"] = existing_ages + found_ages
            else:
                updates["child_ages"] = found_ages
            updates["intent"] = "BOOK_ROOM"
        elif user_msg_clean not in GREETING_WORDS and not any(w in user_msg_clean for w in ("job", "vacancy", "hiring", "resume")):
            # Guest replied with something else or ignored age -> assume 4 (free) to avoid endless loop
            updates["child_ages"] = [4] * total_kids
            updates["intent"] = "BOOK_ROOM"

    elif expected_action == "need_ac_preference":
        discount_kws = ("discount", "kam karo", "sasta", "offer", "coupon", "bargain", "best rate", "reduce", "price match", "less price", "kam price", "price kam")
        resolved_ac = ac_preference_from_prompt_reply(user_msg_clean)
        if any(w in user_msg_clean for w in discount_kws):
            updates["intent"] = "PRICE_MATCH"
            updates["next_action"] = None
            updates["ac_preference"] = "AC"
            updates["ac_qualified"] = True
            updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f not in ("AC", "Non-AC")]
            feat_list = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if "non" not in f.lower()] + ["AC"]
            updates["feature_preferences"] = feat_list
        elif resolved_ac == "AC":
            updates["ac_preference"] = "AC"
            updates["ac_qualified"] = True
            updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f not in ("AC", "Non-AC")]
            feat_list = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if "non" not in f.lower()] + ["AC"]
            updates["feature_preferences"] = feat_list
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif resolved_ac == "NON_AC":
            updates["ac_preference"] = "NON_AC"
            updates["ac_qualified"] = True
            updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f not in ("AC", "Non-AC")]
            feats = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f != "AC"] + ["Non-AC"]
            updates["feature_preferences"] = list(set(feats))
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif resolved_ac == "NO_PREFERENCE" or user_msg_clean.strip() == "no":
            updates["ac_preference"] = "NO_PREFERENCE"
            updates["ac_qualified"] = True
            updates["feature_preferences"] = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f not in ("AC", "Non-AC")]
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif not updates.get("ac_preference") and is_plausible_ac_answer(user_msg_clean):
            # The guest answered and we could not read it. Only `next_action` is
            # pinned — deliberately not `intent`, so a message that was really
            # an escalation is still allowed to escalate, and deliberately not
            # `ac_qualified`, which means "the guest has answered" and belongs
            # to the branches above. Leaving both alone is what lets
            # `room_searcher` put the question again instead of quietly
            # assuming AC and selling the class the guest ruled out.
            logger.info(
                "AC_REASK | input=%r resolved to no AC preference — asking again",
                user_msg_clean,
            )
            updates["next_action"] = "need_ac_preference"

    elif expected_action == "need_composition_preference":

        updates["composition_asked"] = True

        # Check free-text shortcuts first:
        # 1. "3 rooms", "3 room", "3 rooms chahiye"
        room_match = re.search(r'\b(\d+)\s*(?:rooms?|kamre?|kamra)\b', user_msg_clean)
        # 2. "budget 6000", "budget 6000 tak", "6000 budget"
        budget_match = re.search(r'\b(?:budget|within|under|tak)?\s*(\d{3,6})\s*(?:budget|tak|rs|inr)?\b', user_msg_clean)

        if room_match:
            updates["num_rooms"] = int(room_match.group(1))
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif "budget" in user_msg_clean and budget_match:
            try:
                updates["budget"] = float(budget_match.group(1))
                updates["intent"] = "BOOK_ROOM"
                updates["next_action"] = "search"
            except ValueError:
                updates["intent"] = "BOOK_ROOM"
                updates["next_action"] = "search"
        elif user_msg_clean in ("1", "1.", "option 1", "option1", "opt 1") or any(w in user_msg_clean for w in ("kam kharcha", "best price", "cheapest", "sasta", "extra bed", "kam room", "fewer room", "low budget")):
            # Option 1: Kam kharcha / Best price / cheapest / extra bed
            updates["occupancy_preference"] = "MAX"
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif user_msg_clean in ("2", "2.", "option 2", "option2", "opt 2") or any(w in user_msg_clean for w in ("aaram se", "most comfortable", "comfortable", "comfort", "zyada room", "more room", "apna bed", "own bed", "dedicated bed")):
            # Option 2: Aaram se / Most comfortable / own bed / more rooms / dedicated bed
            updates["occupancy_preference"] = "BASE"
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif any(w in user_msg_clean for w in ("family alag", "split", "separate", "alag", "bade alag", "subgroups")):
            # "Split by family" is off the menu as of 2026-09-19, so the bare
            # numbers are gone — but the intent is real and still routed, for a
            # guest who says it in words. Mirrors the zero-LLM fast path above;
            # the two must agree or the same reply would mean two things
            # depending on whether the fast path caught it.
            # (`composition_asked` is already set at the top of this handler.)
            updates["grouping_mode"] = "SEGMENTED"
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "need_subgroups"
            from app.prompts import templates
            lang = state.get("preferred_language") or "Hinglish"
            updates["bot_response"] = templates.ask_family_subgroups_followup(lang=lang)
        else:
            # Anything else — including the "3" and "4" numbers the old
            # four-option menu used to offer, and "you decide" — falls through
            # to the standard ranking. That is the same outcome the old option 4
            # produced, and it is why cutting the menu to two choices needed no
            # new branch here: the guest can no longer ask for something we do
            # not offer, and if they type it anyway they get the default rather
            # than a dead end.
            #
            # "Never ask twice" still holds: whatever they said, we search now.
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"

    elif expected_action == "need_subgroups":
        updates["intent"] = "BOOK_ROOM"
        updates["next_action"] = "search"
        updates["grouping_mode"] = "SEGMENTED"
        updates["subgroups"] = [{"description": state.get("user_message", "").strip()}]

    elif expected_action == "need_custom_rooms":

        combined_request = _additive_request_text(state)
        # A split names more than one room type, so no single one of them is
        # "the" preference. Pinning the second half would narrow the search to
        # it and drop the half the guest picked a turn earlier.
        updates["room_type_preference"] = None if combined_request else user_msg_clean
        updates["intent"] = "BOOK_ROOM"
        updates["booking_stage"] = "collecting"
        updates["next_action"] = "search"

        from app.services.room_service import room_names_from_options
        from app.utils.room_requirement_parse import (
            parse_room_requirements,
            validate_room_requirements,
        )

        # On the additive path the guest's line is only half the request; the
        # room they already picked is the other half, and it was put in front
        # of the message by `_additive_request_text`.
        raw_message = combined_request or state.get("user_message", "")

        # The model reads the request; the deterministic scan is the net under
        # it. This branch is the one place in the funnel where the guest writes
        # without a template — the card asks "how many rooms, which type" and
        # takes the answer in their own words, in their own language, usually
        # mixed into the rest of the booking ("Cheack in 24/10/26 at 5pm ...
        # TOTAL HEAD 10+ 2 Child ... Room category- 3 kashi classic Room in 4th
        # floor"). Reading that is what the model is for. The scan reads the
        # same message by rule and has to be taught, guard by guard, not to take
        # 26 rooms out of a date; told which numbers are the dates, the model
        # does not have the problem.
        #
        # So `parse_room_requirements` runs only if the call itself fails, and
        # that is deliberate: an outage degrades to the previous behaviour
        # rather than to "no room request". A clean empty read is not an outage
        # — it means the message was read and holds no rooms — and running the
        # scan over that would hand back exactly the misparse the guards above
        # exist to stop.
        try:
            extracted = await extract_room_request(
                user_message=raw_message,
                catalogue=await _get_full_catalogue(state),
                check_in=state.get("check_in"),
                check_out=state.get("check_out"),
            )
        except LLMUnavailableError:
            extracted = None
        if extracted is None:
            logger.warning(
                "CUSTOM_ROOM_READ_FALLBACK | sess=%s | the normaliser did not "
                "answer; reading with the deterministic scan",
                state.get("session_id"),
            )
            extracted = parse_room_requirements(raw_message)

        # A per-room occupancy split — "2 people 1 room" — arrives from the
        # normaliser as lines carrying `guests` and naming no room type. It
        # states the party as well as the rooms: the adults are the sum of the
        # per-room counts, so the guest does not have to have said "9" anywhere
        # for the request to be answerable.
        # The same authority order as the main path: what the guest split, then
        # the total they stated, and only then the extractor's derived per-room
        # figures — which are a rounded-up quotient and can claim more people
        # than the guest brought. See `split_party_from_message`.
        is_split = is_an_explicit_split(extracted)
        split_adults = 0
        if is_split:
            split_adults = (
                split_party_from_message(raw_message)
                or int(updates.get("adults") or 0)
                or int(state.get("adults") or 0)
                or sum(
                    int(item.get("count") or 0) * int(item.get("guests") or 0)
                    for item in (extracted or [])
                )
            )
            extracted = reconcile_split_with_party(extracted, split_adults)
        party_size = split_adults or (
            int(state.get("adults") or 0) + int(state.get("children") or 0)
        )

        # Whatever read it, one bound applies: a request for more rooms than
        # there are guests is a misparse, not a booking, and is refused rather
        # than priced.
        reqs, problem = validate_room_requirements(extracted, party_size=party_size)
        if reqs:
            updates["room_requirements"] = reqs
            updates["booking_stage"] = "custom_rooms_selected"
            updates["custom_room_problem"] = None
            if is_split:
                _apply_split_party(updates, state, extracted, split_adults)
        else:
            # Ask again, and say which part did not land. This branch used to
            # leave `booking_stage` as "collecting" and search anyway, so an
            # answer it could not read was silently replaced by whatever the
            # allocator would have offered regardless — the guest's request
            # disappeared with nothing said about it.
            updates["booking_stage"] = "collecting_custom_room"
            updates["next_action"] = "need_custom_rooms"
            updates["custom_room_problem"] = problem or (
                "I couldn't find a room count in that — could you write it the "
                "way the example does?"
            )

    elif expected_action == "group_booking_lead":

        raw_msg = state.get("user_message", "").strip()
        msg_clean = raw_msg.lower()

        # 1. Check if user is clarifying that this is NOT a group booking (Correction)
        not_group_keywords = (
            "group requirement nahi", "group requirement nehi", "group nahi", "group nehi", "not a group",
            "not group", "no group", "individual", "alag alag", "alag-alag", "separate room", "personal", "individual room"
        )
        is_clarifying_not_group = any(k in msg_clean for k in not_group_keywords)

        # Check if user mentioned room count (e.g. "Three alag alag room", "3 rooms", "2 rooms")
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "chhe": 6
        }
        room_match = re.search(r'\b(one|two|three|four|five|six|ek|do|teen|char|paanch|chhe|\d+)\s*(?:alag\s*alag\s*)?(?:rooms?|kamre?|kamra)\b', msg_clean)

        if is_clarifying_not_group or (room_match and int(word_to_num.get(room_match.group(1), int(room_match.group(1)) if room_match.group(1).isdigit() else 1)) < 5):
            new_rooms = None
            if room_match:
                val = room_match.group(1)
                new_rooms = word_to_num.get(val, int(val) if val.isdigit() else None)

            if new_rooms:
                updates["num_rooms"] = new_rooms
            else:
                new_rooms = int(state.get("num_rooms") or 1)

            cur_adults = int(state.get("adults") or 2)
            if cur_adults >= 10:
                updates["adults"] = min(cur_adults, max(2, new_rooms * 2))
                updates["effective_adults"] = updates["adults"]

            updates["group_lead_submitted"] = False
            updates["next_action"] = "search"
            updates["booking_stage"] = "new"
            updates["intent"] = "BOOK_ROOM"

        else:
            # 2. Check if user sent contact details (Phone / Name)
            phone_match = re.search(r'(?:\+?91[\s-]?)?([6-9]\d{9})\b', raw_msg)
            detected_phone = phone_match.group(1) if phone_match else None

            # Extract name from lines or message
            lines = [l.strip() for l in raw_msg.split("\n") if l.strip()]
            detected_name = None
            for line in lines:
                clean_l = re.sub(r'[^\w\s]', '', line).strip()
                if clean_l and not re.search(r'\d', clean_l):
                    l_lower = clean_l.lower()
                    if not any(w in l_lower for w in ("contact", "detail", "details", "mera", "meri", "number", "phone", "hai", "yehi", "ye", "this", "my")):
                        if len(clean_l.split()) <= 4:
                            detected_name = clean_l.title()
                            break

            if detected_phone or detected_name or any(w in msg_clean for w in ("contact", "details", "phone", "number", "mera", "meri")):
                existing_name = state.get("guest_name") or state.get("customer_name")
                existing_phone = state.get("customer_identifier") or ""
                
                final_name = detected_name or (existing_name if existing_name != "Guest" else None) or ""
                final_phone = detected_phone or (existing_phone if (len(existing_phone) == 10 or existing_phone.startswith("+91")) else "")

                if final_name:
                    updates["guest_name"] = final_name
                if detected_phone:
                    updates["customer_identifier"] = detected_phone
                    final_phone = detected_phone

                try:
                    from app.repositories.lead_repository import create_lead, generate_lead_id
                    hotel_id = state.get("hotel_id", 0)
                    lead_id = await generate_lead_id(hotel_id)
                    lead_data = {
                        "lead_id": lead_id,
                        "hotel_id": hotel_id,
                        "session_id": state.get("session_id", ""),
                        "customer_phone": final_phone or state.get("customer_identifier") or "",
                        "guest_name": final_name or "Guest",
                        "check_in": state.get("check_in"),
                        "check_out": state.get("check_out"),
                        "adults": state.get("adults"),
                        "children": state.get("children") or 0,
                        "num_rooms": state.get("num_rooms") or 5,
                        "lead_type": "GROUP_BOOKING",
                        "notes": raw_msg,
                    }
                    await create_lead(lead_data)
                except Exception as e:
                    logger.warning(f"Failed to save group lead: {e}")

                name_greeting = f" {final_name}" if final_name and final_name != "Guest" else ""
                phone_mention = f" on **+91 {final_phone}**" if final_phone else ""

                updates["group_lead_submitted"] = True
                updates["booking_stage"] = "lead_created"
                updates["next_action"] = "lead_created"
                from app.prompts import templates
                
                if not detected_phone and not detected_name and any(w in msg_clean for w in ("contact", "details", "mera", "meri")):
                    updates["bot_response"] = templates.text_card(
                        "Ji bilkul, kripya apna **naam** aur **contact number** share kar dijiye."
                    )
                else:
                    updates["bot_response"] = templates.text_card(
                        f"Thank you{name_greeting}! 🙏\n\n"
                        f"We have received your group requirements and contact details. "
                        f"Our reservations team will review special group pricing and connect with you directly{phone_mention} shortly.\n\n"
                        f"Is there anything else we can assist you with in the meantime?"
                    )
