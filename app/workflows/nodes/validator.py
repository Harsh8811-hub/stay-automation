"""
Validator Node — checks what booking info is present/missing.

Pure Python logic — no AI. Decides what the graph should do next based on state
completeness.

No DB calls either. A 2026-10-10 attempt to have this node ask the guest about
early check-in / late check-out did add one (to see whether the hotel sells it),
and it was removed the same day: the bot does not raise that question. The guest
raises it, and `intent_overrides` answers it — see `_EARLY_LATE_ASK` there.
"""

import difflib
import logging
import re
from typing import cast

from app.config.settings import get_settings
from app.utils.helpers import (
    is_valid_date,
    is_future_date,
    check_in_before_check_out,
    is_multi_room_party,
)
from app.utils.intent_overrides import _CONFIRM_PHRASE, resolve_negotiation_flag
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def validate(state: ConversationState) -> ConversationState:
    """
    - "ready_for_payment"  → Quote confirmed, send payment
    """
    if state.get("next_action") in (
        "need_subgroups",
        "need_change_dates",
        "need_change_guests",
        "need_ac_preference",
        "need_meal_plan_preference",
        "need_composition_preference",
        "need_custom_rooms",
        "group_booking_lead",
    ):
        return {
            **state,
            "missing_fields": [],
            "booking_stage": "collecting",
        }

    missing: list[str] = []

    # Check dates
    ci = state.get("check_in")
    co = state.get("check_out")

    if ci:
        if not is_valid_date(ci) or not is_future_date(ci):
            return {**state, "check_in": None, "missing_fields": ["check_in"], "next_action": "invalid_dates", "updated_fields": []}
            
    if co:
        if not is_valid_date(co):
            return {**state, "check_out": None, "missing_fields": ["check_out"], "next_action": "invalid_dates", "updated_fields": []}
            
    if ci and co:
        if not check_in_before_check_out(ci, co):
            return {**state, "check_in": None, "check_out": None, "missing_fields": ["check_in", "check_out"], "next_action": "invalid_dates", "updated_fields": []}

    # ── Group Booking Guard ──────────────────────────────────────────────
    # A request for 5 or more rooms (e.g. 5, 10, 20, 80 rooms) or 15+ adults is
    # handled via group lead capture / reception coordination rather than
    # self-serve — UNLESS the hotel sells group bookings itself, which is what
    # `GROUP_BOOKING_SELF_SERVE` says. With the flag on this guard never fires:
    # the party is searched and priced by the ordinary path, and
    # `allocation_engine` lifts its own four-room ceiling to match.
    num_rooms_val = state.get("num_rooms")
    adults_val = int(state.get("effective_adults") if state.get("effective_adults") is not None else (state.get("adults") or 0))
    if (
        not get_settings().GROUP_BOOKING_SELF_SERVE
        and ((num_rooms_val is not None and num_rooms_val >= 5) or adults_val >= 15)
        and not state.get("group_lead_submitted")
        # Sent once, not every turn. `group_lead_submitted` above is an in-turn
        # flag the old path never set, which is why the card repeated.
        and not state.get("group_handoff_sent")
        and not state.get("selected_rooms")
        and state.get("booking_stage") not in ("quote_shown", "payment_pending", "payment_attempted", "booking_confirmed", "lead_created")
        and state.get("next_action") not in ("rooms_shown", "show_quote", "ready_for_payment")
    ):
        return {
            **state,
            "missing_fields": [],
            "next_action": "group_booking_lead",
            "booking_stage": "collecting",
            "group_handoff_sent": True,
        }

    # Child age policy: if children > 0 and we don't have explicit ages for ALL children
    num_children = int(state.get("children") or 0)
    child_ages = list(state.get("child_ages") or [])
    hotel_config_dict = state.get("hotel_config") or {}
    biz_limits = (hotel_config_dict.get("business_config") or {}).get("limits", {}) if isinstance(hotel_config_dict, dict) else {}

    # The hotel's own two limits, resolved once, up here — the unknown-age
    # assumption below needs them and the occupancy math further down reads the
    # same two values. Resolving them in two places is how they drift apart.
    free_limit = int(biz_limits.get("child_free_age_limit") or 10)
    adult_limit = int(biz_limits.get("child_as_adult_age_limit") or 11)
    if adult_limit <= free_limit:
        # Crossed limits (free=12, adult=11) would put an 11-year-old in BOTH
        # buckets — counted as an adult AND as a child, inflating the party —
        # and leave the paid band empty. Pull the free limit down so the three
        # buckets stay disjoint and the adult line stays where the hotel wrote
        # it: the child who is not free is the one it named.
        logger.warning(
            "child age limits are crossed (free=%s, adult=%s) for hotel_id=%s "
            "— children at %s and under stay free, %s and over count as adults",
            free_limit, adult_limit, state.get("hotel_id"),
            max(0, adult_limit - 1), adult_limit,
        )
        free_limit = max(0, adult_limit - 1)

    # An unknown child age is ASSUMED, never asked for twice, and never blocks
    # the booking. The guest is treated as inside the hotel's free band until
    # they say otherwise, and the assumption is disclosed on the card via
    # `child_age_assumed` so it can still be corrected. Assuming `free_limit` is
    # free at every hotel by construction; the hardcoded 4 this replaces was a
    # guess at one hotel's policy, and it was wrong at any hotel whose free band
    # does not end at 4 — there an assumed child was silently put in the PAID
    # band and charged. The blocking `return` waited a turn for an answer the
    # guest may have already ignored, and held the whole booking on it.
    if num_children > 0 and len(child_ages) < num_children:
        missing_count = num_children - len(child_ages)
        child_ages = child_ages + [free_limit] * missing_count
        state = {**state, "child_ages": child_ages, "child_age_assumed": True}

    # Apply the hotel's child age policy to occupancy math.
    #
    # THREE buckets, not two. A child policy has three outcomes and this code
    # had two, which is why it could not represent Sample Hotel's policy at all:
    #
    #   age <= free_limit              -> free, stays with parents
    #   free_limit < age < adult_limit -> PAID child: takes a berth, is charged
    #   age >= adult_limit             -> counts as an adult
    #
    # Sahu is free till 7, paid till 12, so 8-to-12 year olds are paid children
    # and 13+ are adults. The two limits there are five years apart.
    #
    # `adult_ages` used to be "age > free_limit OR age >= adult_limit". Under
    # the old 10/11 defaults those limits were adjacent, so the first clause
    # never did anything a correct policy would notice. Under a real policy it
    # would swallow the whole paid band and call a nine-year-old an adult.
    if child_ages:
        free_ages = [age for age in child_ages if age <= free_limit]
        paid_ages = [age for age in child_ages if free_limit < age < adult_limit]
        adult_ages = [age for age in child_ages if age >= adult_limit]

        orig_adults = int(state.get("adults") or 0)
        effective_adults = orig_adults + len(adult_ages)
        # A paid child takes a berth exactly as a free one does — what differs
        # is the bill, not the bed. Counting only the free ones would let a
        # party of paid children look smaller than it is when rooms are chosen.
        counted_children = len(free_ages) + len(paid_ages)
        child_notes = []

        pref_lang = (state.get("preferred_language") or "").lower()
        conv_lang = (state.get("conversation_language") or "").lower()
        is_hindi = pref_lang in ("hindi", "hinglish") or conv_lang == "hindi"

        # One note per bucket, and they no longer exclude each other. The old
        # if/elif chain could only describe a party whose children were ALL
        # free or ALL adult; three free-or-paid-or-adult notes stacked is what
        # a real party with a 5-year-old and a 14-year-old actually needs, and
        # notes are what the guest sees attached to the card.
        if free_ages:
            is_assumed = bool(state.get("child_age_assumed"))
            if len(free_ages) == 1:
                if is_assumed:
                    note = (
                        f"{free_ages[0]} saal ka bachcha (hotel policy ke hisaab se umar assume ki gayi hai) free rahega."
                        if is_hindi
                        else f"Child ({free_ages[0]} yrs — age assumed as per hotel's child policy) stays free with parents."
                    )
                else:
                    note = f"{free_ages[0]} saal ka bachcha free rahega." if is_hindi else f"Child ({free_ages[0]} yrs) stays free with parents."
            else:
                ages_str = ", ".join(str(a) for a in free_ages)
                if is_assumed:
                    note = (
                        f"{len(free_ages)} bachche ({ages_str} saal — hotel policy ke hisaab se umar assume ki gayi hai) free rahenge."
                        if is_hindi
                        else f"{len(free_ages)} children (ages {ages_str} yrs — age assumed as per hotel's child policy) stay free with parents."
                    )
                else:
                    note = f"{len(free_ages)} bachche ({ages_str} saal) free rahenge." if is_hindi else f"{len(free_ages)} children (ages {ages_str}) stay free with parents."
            child_notes.append(note)
        if paid_ages:
            # Deliberately does not name an amount. The hotel's policy says a
            # child in this band is charged; nothing in this codebase yet turns
            # that into a line item (the requirements path never charges
            # children, and `extra_child_rate` is a 100.0 placeholder no room
            # carries). Saying "Rs 100" would invent a price, and saying
            # "stays free" would contradict the hotel. Deferring to the policy
            # is the only sentence that is true today either way.
            if len(paid_ages) == 1:
                note = f"{paid_ages[0]} saal ke bachche ka charge hotel policy ke hisaab se lagega." if is_hindi else f"Child ({paid_ages[0]} yrs) is chargeable under this hotel's child policy."
            else:
                ages_str = ", ".join(str(a) for a in paid_ages)
                note = f"{len(paid_ages)} bachchon ({ages_str} saal) ka charge hotel policy ke hisaab se lagega." if is_hindi else f"{len(paid_ages)} children (ages {ages_str}) are chargeable under this hotel's child policy."
            child_notes.append(note)
            logger.info(
                "child_age_policy: %s child(ren) in the paid band %s<age<%s for hotel_id=%s "
                "— counted for berths, but no child charge is added to the quote yet",
                len(paid_ages), free_limit, adult_limit, state.get("hotel_id"),
            )
        if adult_ages:
            if len(adult_ages) == 1:
                note = f"{adult_ages[0]} saal wale ko adult count kiya gaya hai." if is_hindi else f"Child ({adult_ages[0]} yrs) is counted as an adult."
            else:
                ages_str = ", ".join(str(a) for a in adult_ages)
                note = f"{len(adult_ages)} bachche ({ages_str} saal) adult count kiye gaye hain." if is_hindi else f"{len(adult_ages)} children (ages {ages_str}) are counted as adults."
            child_notes.append(note)
        
        state = {
            **state,
            "effective_adults": effective_adults,
            "effective_children": counted_children,
            "paid_child_count": len(paid_ages),
            "child_age_notes": child_notes,
        }

    if not state.get("check_in"):
        missing.append("check_in")
    if not state.get("check_out"):
        missing.append("check_out")

    has_adults = state.get("adults") is not None
    if not has_adults:
        missing.append("adults")

    # If any core hard slots are missing, prompt composite ask template (S2)
    if missing:
        return {
            **state,
            "missing_fields": missing,
            "next_action": "need_composite_slots",
            "booking_stage": "collecting",
        }

    # Gated AC Preference (Ticket 9):
    # Only ask if hotel has mixed AC inventory AND ac_preference is None
    mixed_ac = (hotel_config_dict.get("business_config") or {}).get("features", {}).get("has_mixed_ac_inventory", True) if isinstance(hotel_config_dict, dict) else True
    if mixed_ac and not state.get("ac_preference"):
        # Default silently to AC or ask if explicitly needed
        pass

    # If the user asked to add an ambiguous number of guests
    ambiguous_guests = state.get("ambiguous_guests")
    if ambiguous_guests and ambiguous_guests > 0:
        return {**state, "missing_fields": ["guest_type"], "next_action": "need_guest_type"}

    # If AC preference was answered or present, clear need_ac_preference
    if state.get("ac_preference") and state.get("next_action") == "need_ac_preference":
        state = {**state, "next_action": "search_rooms"}

    # If next_action is already set to search, bypass room selection
    if state.get("next_action") in ("search", "search_rooms"):
        return {
            **state,
            "allocation_options": [],
            "selected_rooms": [],
            "missing_fields": [],
            "next_action": "search_rooms",
            "booking_stage": "collecting",
            "wants_photos": False,
        }

    # The guest named a clock for an early check-in / late check-out while a
    # quote was already on screen: re-render it so the fee lands on the total
    # they are looking at.
    #
    # The damage this prevents, measured: with the flag off, this exact message
    # comes back `search_rooms` with `selected_rooms` emptied, because the
    # `quote_shown` block below reads "check in" as new dates via
    # `has_new_date_or_search`. With the flag on it comes back `show_quote` with
    # the Deluxe room the guest picked still attached.
    #
    # It sits above `_resolve_room_selection` rather than beside that block for
    # defence: that call re-resolves the guest's message as a room selection, and
    # a message about a stay's timing has no business being read as a pick. (In
    # every phrasing tried, "early check in at 3 am", "…option 2", "…option 3",
    # it returned None — so this ordering is a guard against a hazard, not a
    # demonstrated fix.)
    #
    # `force_new_search` still wins: a guest who genuinely starts over gets their
    # search back, not a re-quote.
    #
    # It deliberately beats an explicit "confirm" too — re-quoting costs one
    # turn, while taking payment on a total missing a fee the hotel charges does
    # not undo. The flag is turn-scoped, so a "confirm" on any other turn is
    # unaffected.
    if (
        state.get("booking_stage") == "quote_shown"
        and state.get("early_late_requote")
        and state.get("selected_rooms")
        and not state.get("force_new_search")
    ):
        return {**state, "missing_fields": [], "next_action": "show_quote"}

    # Check if user is trying to select a room
    intent = state.get("intent", "UNKNOWN")
    is_selection_stage = state.get("booking_stage") in ("rooms_shown", "quote_shown") or intent == "SELECT_ROOM"
    if is_selection_stage and (intent in ("SELECT_ROOM", "CONFIRM_BOOKING", "BOOK_ROOM") or state.get("room_type_preference")) and not state.get("force_new_search") and state.get("allocation_options"):
        # Don't re-resolve if user is confirming and rooms are already selected
        if intent == "CONFIRM_BOOKING" and state.get("selected_rooms") and state.get("booking_stage") == "quote_shown":
            pass  # Skip re-resolution, proceed to payment flow below
        else:
            resolved_state = await _resolve_room_selection(state)
            if resolved_state:
                if resolved_state.get("next_action") == "need_custom_rooms":
                    return resolved_state
                state = resolved_state

    # All basic info present — decide based on booking stage
    booking_stage = state.get("booking_stage", "new")
    has_selected_rooms = bool(state.get("selected_rooms"))

    # Fresh availability / date change must not re-show quote or push payment (T03)
    if state.get("force_new_search"):
        return {
            **state,
            "selected_rooms": [],
            "allocation_options": [],
            "price_breakdown": None,
            "total_price": None,
            "payment_link": None,
            "payment_link_id": None,
            "payment_link_amount": None,
            "force_new_search": False,
            "missing_fields": [],
            "next_action": "search_rooms",
            "booking_stage": "collecting",
        }

    # Validate that currently selected rooms can accommodate the guests
    if has_selected_rooms:
        eff_a = state.get("effective_adults") if state.get("effective_adults") is not None else state.get("adults")
        eff_c = state.get("effective_children") if state.get("effective_children") is not None else (state.get("children") or 0)
        req_adults = int(eff_a or 2)
        req_children = int(eff_c or 0)
        total_guests = req_adults + req_children
        
        total_max_capacity = 0
        total_max_adults = 0
        for r in state.get("selected_rooms", []):
            if isinstance(r, dict):
                r_assigned_a = r.get("assigned_adults") or 0
                r_assigned_c = r.get("assigned_children") or 0
                r_max = r.get("max_occupancy") or (r.get("occupancy") or {}).get("max_occupancy") or (r_assigned_a + r_assigned_c) or 2
                r_max_a = r.get("max_adults") or (r.get("occupancy") or {}).get("max_adults") or r_assigned_a or 2
            elif hasattr(r, "assigned_adults"):
                r_assigned_a = getattr(r, "assigned_adults", 0)
                r_assigned_c = getattr(r, "assigned_children", 0)
                r_max = getattr(r, "max_occupancy", None) or (r_assigned_a + r_assigned_c) or 2
                r_max_a = getattr(r, "max_adults", None) or r_assigned_a or 2
            else:
                r_assigned_a = 0
                r_assigned_c = 0
                r_max = 2
                r_max_a = 2
            total_max_capacity += max(r_max, r_assigned_a + r_assigned_c)
            total_max_adults += max(r_max_a, r_assigned_a)
            
        req_num_rooms = int(state.get("num_rooms") or 0)
        # A room count is a preference, not a cage — the same rule
        # `room_service.search_rooms` applies when it nulls the count. For a 5+
        # party the card on the guest's screen deliberately carries more rooms
        # than they asked for, so enforcing the count here rejects the card the
        # guest is looking at, re-searches, and renders it again — forever. The
        # capacity checks above are what actually guard against under-seating;
        # this one is redundant for these parties and harmful.
        #
        # The same is true of any selection that came off the card. A guest who
        # asked "1 room" for four adults was shown "2 x Deluxe Room King Bed",
        # picked it, and had the pick thrown away because `num_rooms` still said
        # 1 from an earlier message — the identical card went back out and the
        # repeat counter did the rest (guest 919088198242, 2026-10-19
        # 11:17:05). Choosing a priced line *is* choosing its room count, so
        # the stale number loses to the guest's pick.
        if is_multi_room_party(req_adults, req_children) or state.get("room_selection_resolved"):
            req_num_rooms = 0
        room_count_mismatch = (req_num_rooms > 0 and req_num_rooms != len(state.get("selected_rooms", [])))
        
        rt_pref = (state.get("room_type_preference") or "").strip().lower()
        pref_mismatch = False
        is_generic_or_option_pref = (
            not rt_pref
            or state.get("room_selection_resolved")
            or rt_pref in ("cheapest", "expensive", "best", "any", "no preference", "custom", "build your own")
            or bool(re.match(r'^(?:option|opt|no\.?|choice)?\s*\d+\.?$', rt_pref))
        )
        if not is_generic_or_option_pref:
            pref_words = [w for w in re.findall(r'[a-zA-Z]+', rt_pref) if w not in ("room", "rooms", "kamra", "kamre", "bed", "extra", "plus", "with", "option", "opt")]
            if pref_words:
                pref_mismatch = not any(
                    any(w in (r.get("room_name") if isinstance(r, dict) else getattr(r, "room_name", "")).lower() for w in pref_words)
                    for r in state.get("selected_rooms", [])
                )

        has_capacity_shortfall = (total_max_capacity < total_guests or total_max_adults < req_adults)
        has_other_mismatch = (room_count_mismatch or pref_mismatch) and intent != "CONFIRM_BOOKING"
        if has_capacity_shortfall or has_other_mismatch:
            logger.info(
                f"VALIDATOR_SELECTION_MISMATCH: capacity={total_max_capacity}/{total_guests}, "
                f"rooms={len(state.get('selected_rooms', []))}/{req_num_rooms}, pref_mismatch={pref_mismatch}. Triggering fresh search."
            )
            return {
                **state,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "payment_link": None,
                "payment_link_id": None,
                "payment_link_amount": None,
                "missing_fields": [],
                "next_action": "search_rooms",
                "booking_stage": "collecting",
            }

    if not has_selected_rooms:
        if state.get("room_requirements") or state.get("next_action") == "search":
            return {
                **state,
                "allocation_options": [],
                "missing_fields": [],
                "next_action": "search_rooms",
                "booking_stage": "collecting",
            }
        if state.get("wants_photos"):
            return {
                **state,
                "missing_fields": [],
                "next_action": "search_rooms",
                "booking_stage": "collecting",
            }

        user_msg = (state.get("user_message") or "").lower()
        is_photo_query = any(w in user_msg for w in ("photo", "photos", "pic", "pics", "image", "images", "picture", "pictures"))
        if not is_photo_query and state.get("allocation_options") and intent in ("SELECT_ROOM", "CONFIRM_BOOKING", "BOOK_ROOM"):
            from app.services.stop_sell_guard import check_and_handle_stop_sell_escalation
            stop_sell_res = await check_and_handle_stop_sell_escalation(state)
            if stop_sell_res:
                return cast(ConversationState, {**state, **stop_sell_res})
            # They tried to select from options but we couldn't resolve it uniquely
            return {**state, "missing_fields": [], "next_action": "invalid_room_selection"}

        # (AC and Occupancy preferences are now enforced earlier in the validation flow)

        # Need to search and show rooms
        return {**state, "missing_fields": [], "next_action": "search_rooms", "booking_stage": "collecting"}

    if booking_stage in ("rooms_shown", "collecting") and has_selected_rooms:
        # Room selected, generate quote — unless the rooms can take a mattress
        # and the guest has not said either way. Ask once per session; the
        # answer is read in `understand`, which re-prices the same set with or
        # without the bed.
        #
        # This was a disclosure until 2026-09-25: the row's price already
        # contained a mattress the allocator had added on the guest's behalf, so
        # the question existed to let them take it off. The allocator no longer
        # adds one unasked (`distribute_guests`), so it is an offer now — the
        # same question, with nothing hidden behind it.
        if _offers_an_extra_bed(state) and not state.get("extra_bed_asked"):
            return {
                **state,
                "missing_fields": [],
                "next_action": "need_extra_bed",
                "extra_bed_asked": True,
            }
        return {**state, "missing_fields": [], "next_action": "show_quote"}

    if booking_stage == "quote_shown":
        user_msg = str(state.get("user_message", ""))

        # Don't treat negotiation, stop, change, or questions as confirmation.
        # The intent test stays authoritative — it is the model's own earlier
        # classification of the message as a modify/price/FAQ/cancel ask, and
        # the payment-turn verdict should not be able to override it.
        # `user_msg`, not a lowercased copy: the shared resolver is IGNORECASE.
        is_negotiation_or_change = (
            intent in ("PRICE_MATCH", "FAQ", "CANCEL_BOOKING", "MODIFY_BOOKING")
            or resolve_negotiation_flag(state, user_msg)
        )

        # Explicit affirmative confirmation.
        #
        # The plain-substring tuple that used to sit here was fully redundant:
        # every phrase in it is already matched by `_CONFIRM_PHRASE`, which does
        # the same job with word boundaries. Its only *additional* effect was
        # the bare "confirm" entry, whose substring test fired inside
        # "confirmation" — so "send me a confirmation email" read as a booking
        # confirmation and pushed the guest to payment. Verified entry by entry
        # against the regex before removing.
        is_confirming_explicit = (
            intent == "CONFIRM_BOOKING" or bool(_CONFIRM_PHRASE.search(user_msg))
        )

        if is_confirming_explicit and not is_negotiation_or_change:
            # The name used to be a gate here, in front of the payment link —
            # see NAME_AFTER_PAYMENT_ENABLED. "Confirm" now goes straight to
            # payment and the name is collected on the confirmation turn.
            if (
                not get_settings().NAME_AFTER_PAYMENT_ENABLED
                and (not state.get("guest_name") or state.get("guest_name") == "Guest")
            ):
                return {
                    **state,
                    "missing_fields": ["guest_name"],
                    "next_action": "need_guest_name_for_payment",
                }
            return {**state, "missing_fields": [], "next_action": "ready_for_payment"}

        from app.workflows.nodes.understand import get_active_prompt
        was_prompted_for_name = (
            state.get("next_action") == "need_guest_name_for_payment"
            or get_active_prompt(state) == "need_guest_name_for_payment"
        )
        if was_prompted_for_name and not is_negotiation_or_change and state.get("guest_name") and state.get("guest_name") != "Guest":
            return {**state, "missing_fields": [], "next_action": "ready_for_payment"}

        # New availability / booking ask must NOT re-show quote or push payment (T03)
        if intent in ("BOOK_ROOM", "CHECK_AVAILABILITY", "MODIFY_BOOKING"):
            user_msg = str(state.get("user_message", ""))
            # If the user is affirming/booking the current selection without specifying new dates or changing pax:
            # Same redundancy as above — `_CONFIRM_PHRASE` covers every phrase.
            is_confirming_current = bool(_CONFIRM_PHRASE.search(user_msg))
            has_new_date_or_search = (
                bool(state.get("force_new_search"))
                or bool(re.search(r"\b(next|coming|another|different|change\s+dates?|other\s+dates?|check\s+in|check\s+out|guests?)\b", user_msg, re.IGNORECASE))
            )
            if is_confirming_current and not has_new_date_or_search and state.get("selected_rooms") and state.get("total_price"):
                if (
                    not get_settings().NAME_AFTER_PAYMENT_ENABLED
                    and not state.get("guest_name")
                ):
                    return {
                        **state,
                        "missing_fields": ["guest_name"],
                        "next_action": "need_guest_name_for_payment",
                    }
                return {**state, "missing_fields": [], "next_action": "ready_for_payment"}

            return {
                **state,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "payment_link": None,
                "payment_link_id": None,
                "payment_link_amount": None,
                "force_new_search": False,
                "missing_fields": [],
                "next_action": "search_rooms",
                "booking_stage": "collecting",
            }
        return {**state, "missing_fields": [], "next_action": "show_quote"}

    # Default: search rooms
    return {**state, "missing_fields": [], "next_action": "search_rooms"}


def _selection_carries_a_bed(state: ConversationState) -> bool:
    """Is a mattress already priced into the row the guest just picked?

    Read off `extra_bed_count`, the same field `room_service._card_candidates`
    reads to lay a bed into the card's row. A guest who said "with extra bed" up
    front reaches the quote already paying for one, and there is nothing to offer
    them.

    It used to be read off `extra_adult_charge`, and those were the same thing
    until a CP / MAP guest began being charged for an adult above base occupancy
    without a mattress ever being offered. Reading the charge here would find
    that guest "already has a bed", decline to ask, and leave them paying for a
    mattress the hotel never sold them.

    `None` is "nobody said" — a selection written before the field — and keeps
    the old charge-based read, which is what that selection was built against.
    A missing or unparseable charge reads as "no bed": not asking is the safe
    direction, and every option that predates `assigned_*` arrives that way.
    """
    for room in state.get("selected_rooms") or []:
        try:
            room = room or {}
            bed_count = room.get("extra_bed_count")
            if bed_count is not None:
                if int(bed_count) > 0:
                    return True
                continue
            if float(room.get("extra_adult_charge") or 0) > 0:
                return True
        except (TypeError, ValueError, AttributeError):
            continue
    return False


def _selected_option(state: ConversationState) -> dict | None:
    """The allocation option the guest's current `selected_rooms` came from.

    `selected_rooms` is exactly one option's `rooms` list (every writer in
    `validator` sets it from `cand["opt"]["rooms"]`), so the option is recovered
    by matching the room ids. Returns None when nothing matches, which is the
    honest answer for a session whose options have since been cleared.
    """
    selected = state.get("selected_rooms") or []
    if not selected:
        return None
    want = sorted(str((r or {}).get("room_id") or "") for r in selected)
    for opt in state.get("allocation_options") or []:
        got = sorted(str((r or {}).get("room_id") or "") for r in (opt.get("rooms") or []))
        if got == want:
            return opt
    return None


def _offers_an_extra_bed(state: ConversationState) -> bool:
    """Is there a mattress left to *sell* the guest — offered but not yet bought?

    Two conditions, and both matter:

    - **The set can take one.** `max_adults > base_adults` over the option is
      the hotel saying a mattress exists for these rooms. Rooms whose two
      figures are equal (every Sample Property room is base 2 / max 2) have none, and
      asking there would promise something the allocator will refuse to supply.
    - **The party is above base.** A couple in a Deluxe King is not a mattress
      conversation. This is deliberately the same trigger the old disclosure
      question fired on — a party the rooms' *base* occupancy does not cover —
      so the question appears at the same moment it always did, and only its
      meaning changed.

    A mattress already priced short-circuits to False: `understand` re-prices
    the set on the answer, so asking again would be asking about a purchase the
    guest has already made.
    """
    if _selection_carries_a_bed(state):
        return False
    opt = _selected_option(state)
    if not opt:
        return False
    try:
        if int(opt.get("max_adults") or 0) <= int(opt.get("base_adults") or 0):
            return False
        # Adults only. A child fits in with the adults and never needs a
        # mattress (Product, 2026-09-25: *"Adults only count toward beds"*), so
        # counting children here would put the question to a family of four in
        # a base-2 room whose two adults the room already seats — asking them
        # to buy a mattress for a child. The old disclosure gate summed the
        # same way, because a child never produced a bed charge either.
        seated_adults = sum(
            int((r or {}).get("assigned_adults") or 0) for r in (opt.get("rooms") or [])
        )
        return seated_adults > int(opt.get("base_adults") or 0)
    except (TypeError, ValueError):
        return False


_NAME_STOPWORDS = {"room", "rooms", "the", "a", "an", "with"}


def _name_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (text or "").lower())} - _NAME_STOPWORDS


def _same_word(a: str, b: str) -> bool:
    """Equal allowing for the spelling a guest actually types.

    "standerd" / "standard" and "delux" / "deluxe" are one word to a reader and
    two to a string comparison. Prefix covers the truncated form; the ratio
    covers the typo.
    """
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _names_an_offered_room(user_msg: str, options: list[dict]) -> bool:
    """Did the guest write out a room name that is on the card, near enough?

    Deliberately strict: *every* distinctive word of the room name has to be
    present. A bare preference ("non ac", "ac chahiye") names none of the
    rooms, and must keep hitting the guard.
    """
    said = _name_tokens(user_msg)
    if not said:
        return False
    for option in options or []:
        for room in (option or {}).get("rooms") or []:
            wanted = _name_tokens(str(room.get("room_name") or ""))
            if wanted and all(any(_same_word(w, s) for s in said) for w in wanted):
                return True
    return False


async def _resolve_room_selection(state: ConversationState) -> ConversationState | None:
    preference = (state.get("room_type_preference") or "").strip()
    user_msg = (state.get("user_message") or "").strip()
    options = state.get("allocation_options", [])

    if not options:
        return None

    user_msg_lower = user_msg.lower()
    # Does the guest's own words name a room that is on the card? Decided once,
    # for two readers below: the preference guard, and the loose number
    # fallback that would otherwise read "…king bed 1" as "option 1".
    named_room = _names_an_offered_room(user_msg, options)
    # ── Strict Guard (Sir's Spec): Never treat language request, preference change, or vague text as room selection ──
    # The bare verb `bataiye` is deliberately NOT in this list. On its own it is
    # the guest asking us to choose — `aap bataiye` / `aap batayein` is the
    # recommendation trigger below — and matching it here would return None
    # before that branch is ever reached, silently killing recommendation.
    # Every genuine language request pairs the verb with a language
    # (`hindi mein`, `english mein`), which this list still catches.
    is_language_request = any(w in user_msg_lower for w in (
        "english mein", "hindi mein", "in english", "in hindi", "english me", "hindi me",
        "english please", "hindi please", "speak in", "talk in", "boliye"
    ))
    is_preference_change = any(w in user_msg_lower for w in (
        "non ac", "non-ac", "ac chahiye", "non ac chahiye", "ac wala", "without ac", "with ac",
        "thoda kam", "discount", "offer", "fresh start", "start fresh"
    ))
    # ...unless the guest just named a room that is on the card. "Standerd non
    # ac 1 room" and "1standerd non ac" contain "non ac" and are nonetheless
    # the most exact selection a guest can make: the Standard Non-AC row, by
    # name. The guard read the preference word, threw the pick away, and the
    # card came back asking for "the exact room name" the guest had just typed
    # (guest 919088198242, 2026-10-19, three turns running).
    if is_preference_change and named_room:
        logger.info(
            "SELECT_ROOM_PREFERENCE_WORD_IS_A_ROOM_NAME | msg='%s' names a room on the card",
            user_msg,
        )
        is_preference_change = False
    if is_language_request or is_preference_change:
        logger.info("SELECT_ROOM_GUARD_TRIGGERED | msg='%s' is language/preference/discount, NOT room selection", user_msg)
        return None

    import re
    from app.utils.room_card_reply import _normalize_keycaps
    selected_index = -1
    opt_num_explicit = None

    user_msg_norm = _normalize_keycaps(user_msg).strip()
    pref_norm = _normalize_keycaps(preference).strip()
    search_target = f"{user_msg_norm} {state.get('selected_option_label') or ''} {pref_norm}".strip()

    # 0. Check for explicit alphanumeric option key like "1A", "1B", "2A", "3A", "4B"
    alpha_match = re.search(r'\b(?:option|opt|choice|selection|no\.?)?\s*([1-9])\s*[-_]?\s*([a-zA-Z])\b', search_target, re.IGNORECASE)
    if not alpha_match:
        alpha_match = re.search(r'\b([1-9][a-zA-Z]|[a-zA-Z][1-9])\b', search_target, re.IGNORECASE)
    if alpha_match:
        if alpha_match.lastindex == 2:
            target_label = f"{alpha_match.group(1)}{alpha_match.group(2)}".upper()
        else:
            raw = alpha_match.group(1).upper()
            if raw[0].isalpha() and raw[1].isdigit():
                target_label = f"{raw[1]}{raw[0]}"
            else:
                target_label = raw

        for idx, opt in enumerate(options):
            if str(opt.get("option_label") or "").upper() == target_label:
                selected_index = idx
                break

        # If option_label wasn't set on options, map by category number + sub-option letter
        if selected_index == -1 and target_label and len(target_label) >= 2 and target_label[0].isdigit():
            cat_num = int(target_label[0])
            sub_letter = target_label[1].upper()
            sub_idx = ord(sub_letter) - ord('A')  # 'A' -> 0, 'B' -> 1
            cat_options = [opt for opt in options if str(opt.get("option_label", "")).startswith(str(cat_num))]
            if 0 <= sub_idx < len(cat_options):
                chosen_opt = cat_options[sub_idx]
                for idx, opt in enumerate(options):
                    if opt is chosen_opt:
                        selected_index = idx
                        break

    # 0.1 Check for single letter like "A", "B", "C", "D" or "Option A"
    if selected_index == -1:
        single_letter_match = re.search(r'^\s*(?:option|opt|choice|selection|room)?\s*([a-dA-D])\s*[.,!]?\s*$', search_target, re.IGNORECASE)
        if single_letter_match:
            let = single_letter_match.group(1).upper()
            let_idx = ord(let) - ord('A')
            if 0 <= let_idx < len(options):
                selected_index = let_idx

    # 0.2 Check for ordinal words like "first", "second", "third", "pehla", "doosra"
    if selected_index == -1:
        if any(w in search_target.lower() for w in ("first", "1st", "pehla", "first option")):
            if len(options) >= 1:
                selected_index = 0
        elif any(w in search_target.lower() for w in ("second", "2nd", "doosra", "dusra", "second option")):
            if len(options) >= 2:
                selected_index = 1
        elif any(w in search_target.lower() for w in ("third", "3rd", "teesra", "tisra", "third option")):
            if len(options) >= 3:
                selected_index = 2

    # 0.5. Check for cheapest / lowest price / sasta
    if selected_index == -1:
        is_cheapest = any(w in search_target.lower() for w in ("cheapest", "chepest", "sasta", "lowest price", "most affordable", "best value", "budget", "sabse sasta"))
        if is_cheapest:
            min_price = float("inf")
            min_idx = -1
            for idx, opt in enumerate(options):
                price = opt.get("total_price_for_stay") or opt.get("total_price_per_night") or float("inf")
                if price < min_price:
                    min_price = price
                    min_idx = idx
            if min_idx >= 0:
                selected_index = min_idx

    # 0.6. Check for most expensive / luxury / premium
    if selected_index == -1:
        is_expensive = any(w in search_target.lower() for w in ("most expensive", "expensive", "luxury", "premium", "highest price", "sabse mahnga", "mehenga"))
        if is_expensive:
            max_price = -1
            max_idx = -1
            for idx, opt in enumerate(options):
                price = opt.get("total_price_for_stay") or opt.get("total_price_per_night") or 0
                if price > max_price:
                    max_price = price
                    max_idx = idx
            if max_idx >= 0:
                selected_index = max_idx

    # Category indices for Custom / Build your own detection
    category_indices = set()
    for opt in options:
        lbl = str(opt.get("option_label") or "")
        if lbl and lbl[0].isdigit():
            category_indices.add(int(lbl[0]))
    custom_category_idx = max(category_indices) + 1 if category_indices else 5

    # 0.7 A line the model named outright. When the guest says something the
    # card text cannot match ("it's best") but the model still returns
    # `room_type_preference: 'Option 1'`, `understand` keeps that as the
    # per-turn `selected_option_label`. Until now the only reader of that key
    # was the free-text `search_target` above, which no matcher parses for a
    # bare digit — so the pick was dropped and the card redrawn.
    if selected_index == -1:
        stated = str(state.get("selected_option_label") or "").strip()
        stated_match = re.fullmatch(r"(?:option\s*)?([1-9]\d*)", stated, re.IGNORECASE)
        if stated_match and 1 <= int(stated_match.group(1)) <= len(options):
            selected_index = int(stated_match.group(1)) - 1

    # 1. Try pattern on preference first (e.g. "option 3", "3", "select 1", "1 please")
    if selected_index == -1 and pref_norm:
        if pref_norm.strip().isdigit():
            opt_num_explicit = int(pref_norm.strip())
        else:
            opt_match = re.search(r'\b(?:(?:i\s+)?(?:want|choose|select|book|take)|give\s+me|option|opt|no\.?|choice|selection|room)\s*(\d+)\b', pref_norm.lower())
            if not opt_match and not named_room and len(pref_norm) <= 25 and not any(w in pref_norm.lower() for w in ("adult", "child", "night", "pax", "guest")):
                opt_match = re.search(r'\b(\d+)\b', pref_norm)
            if opt_match:
                opt_num_explicit = int(opt_match.group(1))

        if opt_num_explicit is not None:
            for idx, opt in enumerate(options):
                lbl = str(opt.get("option_label") or "")
                if lbl.startswith(str(opt_num_explicit)):
                    selected_index = idx
                    break
            if selected_index == -1 and opt_num_explicit != custom_category_idx and 0 <= opt_num_explicit - 1 < len(options):
                selected_index = opt_num_explicit - 1

    # 2. If no match, try pattern on raw user message (e.g. "3", "option 3", "3.", "select 1", "1 please")
    if selected_index == -1 and user_msg_norm:
        opt_num_explicit = None
        cleaned_msg = user_msg_norm.strip().rstrip(".").strip()
        if cleaned_msg.isdigit():
            opt_num_explicit = int(cleaned_msg)
        else:
            opt_match = re.search(r'\b(?:(?:i\s+)?(?:want|choose|select|book|take)|give\s+me|option|opt|no\.?|choice|selection|room)\s*(\d+)\b', user_msg_norm.lower())
            if not opt_match and not named_room and len(user_msg_norm) <= 25 and not any(w in user_msg_norm.lower() for w in ("adult", "child", "night", "pax", "guest")):
                opt_match = re.search(r'\b(\d+)\b', user_msg_norm)
            if opt_match:
                opt_num_explicit = int(opt_match.group(1))

        if opt_num_explicit is not None:
            for idx, opt in enumerate(options):
                lbl = str(opt.get("option_label") or "")
                if lbl and lbl.startswith(str(opt_num_explicit)):
                    selected_index = idx
                    break
            if selected_index == -1 and opt_num_explicit != custom_category_idx and 0 <= opt_num_explicit - 1 < len(options):
                selected_index = opt_num_explicit - 1

    is_build_your_own = (
        opt_num_explicit == custom_category_idx
        or (opt_num_explicit == 5 and 5 not in category_indices)
        or any(w in user_msg.lower() for w in ("build your own", "build my own", "custom", "custom room", "apne hisab se", "khud choose", "manpasand"))
        or any(w in preference.lower() for w in ("build your own", "build my own", "custom", "custom room"))
    )
    if is_build_your_own:
        return {
            **state,
            "booking_stage": "collecting",
            "next_action": "need_custom_rooms",
            "room_type_preference": None,
            "room_type_not_available": False,
            "selected_rooms": [],
            "room_requirements": None,
        }

    # If the user confirms/proceeds or asks for recommendation while seeing room options:
    intent = state.get("intent", "UNKNOWN")
    if selected_index == -1:
        is_recommendation_request = any(w in user_msg.lower() or w in preference.lower() for w in (
            "recommend", "recommendation", "suggest", "suggestion", "best option", "best fit", "you decide", "self choice", "guest recommendation", "aap batayein", "aap choose karein", "aap bataiye", "kaun sa", "kon sa", "yehi", "yahi", "same"
        ))
        if is_recommendation_request and options:
            selected_index = 0
        elif (intent == "CONFIRM_BOOKING" or bool(_CONFIRM_PHRASE.search(user_msg))) and state.get("booking_stage") == "rooms_shown" and options:
            if len(options) == 1:
                selected_index = 0

    # Build candidate display labels and room names for each option
    candidate_options = []
    for i, opt in enumerate(options, 1):
        rms = opt.get("rooms", [])
        names = []
        for r in rms:
            r_name = str(r.get("room_name") or "").strip()
            if r_name and r_name not in names:
                names.append(r_name)
        if len(rms) == 1:
            disp_label = names[0] if names else f"Option {i}"
        else:
            counts = {}
            for r in rms:
                nm = str(r.get("room_name") or "Room").strip()
                counts[nm] = counts.get(nm, 0) + 1
            disp_label = ", ".join(f"{cnt}x {nm}" if cnt > 1 else nm for nm, cnt in counts.items())

        candidate_options.append({
            "option_index": i - 1,
            "option_number": i,
            "display_label": disp_label,
            "display_label_lower": disp_label.lower(),
            "room_names": names,
            "room_names_lower": [n.lower() for n in names],
            "opt": opt,
        })

    # Index-based selection wins immediately
    if selected_index >= 0:
        if 0 <= selected_index < len(options):
            matched = candidate_options[selected_index]
            logger.info(
                "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=index",
                user_msg,
                matched["display_label"],
                matched["option_number"],
            )
            selection = {
                **state,
                "selected_rooms": matched["opt"].get("rooms", []),
                "room_match_candidates": None,
                "room_selection_resolved": True,
            }
            # `display_label` describes a *set* — "2x DELUXE ROOM KING BED" — and
            # `room_type_preference` is read downstream as a room *type*:
            # `room_searcher.py:73` hands it to `room_service.search_rooms`, which
            # runs it through `filter_rooms_by_reference` and allocates over only
            # the rooms it matched. Writing the label here therefore pinned every
            # later search to the rooms this one set happened to contain, and
            # overwrote whatever the guest had actually asked for.
            #
            # Measured 2026-09-29 (hotel 111111, six adults): after "child is 15
            # years old" moved the party, the pool held `DELUXE ROOM FOUR BED`
            # alone, so `2 x King` at Rs 3,600 could not be offered against
            # `2 x Four Bed` at Rs 6,800. The selection is already recorded in
            # `selected_rooms`, and `room_selection_resolved` above is what tells
            # this validator the choice was the guest's, so the preference is
            # neither needed nor wanted here.
            if not get_settings().SELECTION_DOES_NOT_PIN_ROOM_TYPE:
                selection["room_type_preference"] = matched["display_label"]
            return selection
        else:
            logger.warning(
                "SELECT_ROOM_OUT_OF_RANGE | input=\"%s\" | option_num=%s | total_options=%s",
                user_msg,
                opt_num_explicit,
                len(options),
            )
            return None

    # If in rooms_shown / quote_shown stage or intent indicates selection
    is_rooms_shown = state.get("booking_stage") in ("rooms_shown", "quote_shown") or state.get("next_action") == "rooms_shown"
    if selected_index == -1 and not is_rooms_shown and intent not in ("SELECT_ROOM", "BOOK_ROOM", "CONFIRM_BOOKING"):
        return None

    # 3. Text-based matching against display labels and room names
    guest_input = (preference or user_msg).strip().lower()
    
    # Check cheapest / budget preference
    if candidate_options and (state.get("recommendation_preference") == "CHEAPEST" or any(w in guest_input for w in ("cheapest", "chepest", "lowest price", "lowest", "sasta", "sabse sasta", "budget", "kam dam", "kam daam"))):
        cheapest_cand = min(candidate_options, key=lambda c: c["opt"].get("total_price_per_night", 999999))
        logger.info(
            "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=cheapest",
            user_msg,
            cheapest_cand["display_label"],
            cheapest_cand["option_number"],
        )
        return {
            **state,
            "selected_rooms": cheapest_cand["opt"].get("rooms", []),
            "room_type_preference": cheapest_cand["display_label"],
            "room_match_candidates": None,
            "room_selection_resolved": True,
        }

    # Check expensive / luxury preference
    if candidate_options and (state.get("recommendation_preference") == "EXPENSIVE" or any(w in guest_input for w in ("expensive", "most expensive", "luxury", "premium", "highest price", "mehenga", "sabse mehenga"))):
        expensive_cand = max(candidate_options, key=lambda c: c["opt"].get("total_price_per_night", -1))
        logger.info(
            "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=expensive",
            user_msg,
            expensive_cand["display_label"],
            expensive_cand["option_number"],
        )
        return {
            **state,
            "selected_rooms": expensive_cand["opt"].get("rooms", []),
            "room_type_preference": expensive_cand["display_label"],
            "room_match_candidates": None,
            "room_selection_resolved": True,
        }

    # Check exact match
    for cand in candidate_options:
        if guest_input == cand["display_label_lower"] or any(guest_input == n_lower for n_lower in cand["room_names_lower"]):
            logger.info(
                "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=exact",
                user_msg,
                cand["display_label"],
                cand["option_number"],
            )
            return {
                **state,
                "selected_rooms": cand["opt"].get("rooms", []),
                "room_type_preference": cand["display_label"],
                "room_match_candidates": None,
                # Every path that reads a pick off the card says so, not just
                # the index match. The capacity check downstream uses this to
                # let the picked line's room count beat an older `num_rooms`.
                "room_selection_resolved": True,
            }

    # Check semantic match for room type + extra bed / room count modifiers
    wants_extra = any(w in guest_input for w in ("extra bed", "extra", "kam room", "sasta", "cheaper", "b option", "sub option b", "b wala"))
    wants_comfort = any(w in guest_input for w in ("apna bed", "own bed", "zyada room", "more rooms", "without extra bed", "a option", "sub option a", "a wala"))
    
    # Extract any specific room count mentioned (e.g. "3 room", "2 rooms")
    room_cnt_match = re.search(r'(\d+)\s*(?:room|kamra|rooms)', guest_input)
    target_room_cnt = int(room_cnt_match.group(1)) if room_cnt_match else None

    scored_cands = []
    for cand in candidate_options:
        score = 0
        opt_data = cand["opt"]
        rms = opt_data.get("rooms", [])
        extra_a = opt_data.get("extra_adult_count", 0)
        
        # Match room type
        for r_name in cand["room_names_lower"]:
            if "deluxe" in r_name and "deluxe" in guest_input:
                score += 10
            elif "premier" in r_name and "premier" in guest_input:
                score += 10
            elif "family" in r_name and "family" in guest_input:
                score += 10
            elif "four bed" in r_name and "four" in guest_input:
                score += 10
            elif "standard" in r_name and "standard" in guest_input:
                score += 10
                
        if score > 0:
            if target_room_cnt and len(rms) == target_room_cnt:
                score += 8
            if wants_extra and extra_a > 0:
                score += 5
            elif wants_comfort and extra_a == 0:
                score += 5
            scored_cands.append((score, cand))

    if scored_cands:
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        if scored_cands[0][0] >= 10:
            best_cand = scored_cands[0][1]
            logger.info(
                "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=semantic_scorer",
                user_msg,
                best_cand["display_label"],
                best_cand["option_number"],
            )
            return {
                **state,
                "selected_rooms": best_cand["opt"].get("rooms", []),
                "room_type_preference": best_cand["display_label"],
                "room_match_candidates": None,
                "room_selection_resolved": True,
            }

    # Check substring match
    for cand in candidate_options:
        if (guest_input in cand["display_label_lower"] or cand["display_label_lower"] in guest_input) or any(
            guest_input in n_lower or n_lower in guest_input for n_lower in cand["room_names_lower"]
        ):
            logger.info(
                "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=substring",
                user_msg,
                cand["display_label"],
                cand["option_number"],
            )
            return {
                **state,
                "selected_rooms": cand["opt"].get("rooms", []),
                "room_type_preference": cand["display_label"],
                "room_match_candidates": None,
                "room_selection_resolved": True,
            }

    # Check word overlap match (>= 60%)
    guest_words = set(re.findall(r'\w+', guest_input))
    for filler in ("room", "the", "option", "please", "i", "want", "book", "take", "show", "give", "me"):
        guest_words.discard(filler)

    best_overlap = 0.0
    best_cand = None
    if guest_words:
        for cand in candidate_options:
            for label in [cand["display_label"]] + cand["room_names"]:
                label_words = set(re.findall(r'\w+', label.lower()))
                label_words.discard("room")
                label_words.discard("the")
                if not label_words:
                    continue
                overlap = len(guest_words & label_words) / max(len(label_words), 1)
                if overlap >= 0.6 and overlap > best_overlap:
                    best_overlap = overlap
                    best_cand = cand

    if best_cand:
        logger.info(
            "SELECT_ROOM_MATCH | input=\"%s\" | matched_label=\"%s\" | option=%s | match_type=word_overlap | overlap=%.2f",
            user_msg,
            best_cand["display_label"],
            best_cand["option_number"],
            best_overlap,
        )
        return {
            **state,
            "selected_rooms": best_cand["opt"].get("rooms", []),
            "room_type_preference": best_cand["display_label"],
            "room_match_candidates": None,
            "room_selection_resolved": True,
        }

    # 4. If no match found
    candidate_labels = [c["display_label"] for c in candidate_options]
    logger.info(
        "SELECT_ROOM_NO_MATCH | input=\"%s\" | candidates=%s | reason=no_overlap",
        user_msg,
        candidate_labels,
    )
    return None
