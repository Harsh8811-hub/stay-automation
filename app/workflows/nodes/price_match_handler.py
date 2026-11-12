"""
Price Match Handler Node.
Processes competitor price match screenshots and validation.
"""
import logging
import re
from typing import cast
from app.workflows.state import ConversationState
from app.prompts.templates import button_card, text_card
from app.services.gemini_service import translate_cards_if_needed
from app.services.price_match_service import validate_price_match
from app.repositories import price_match_repository
from app.models.activity_schemas import ActivityLog
from app.repositories.activity_repository import log_activity
from app.utils.competitor_mention import mentions_competitor

from app.repositories.season_repository import get_season_negotiation_limit

logger = logging.getLogger(__name__)

#: A figure the guest put in front of us with no rupee marker on it. Only read
#: when the message quotes a number back at us (see `_QUOTE_BACK_CUE`).
_PLAIN_AMOUNT = re.compile(r'\b(\d{3,6})\b')

#: Words a guest uses when they are comparing our figure to another one.
#: "Here is showing 2000 then how can be 2250" (Sample Property, 2026-10-31).
_QUOTE_BACK_CUE = re.compile(
    r'\b(?:showing|shows|shown|says|said|quoted|listed|mentioned|advertised|displayed)\b'
    # "here superior room price is 2000" — the other shape a guest uses to hold
    # our rate up against a figure. Same list, one more verb.
    r'|\b(?:price|rate)\s*(?:is|was|be)\b',
    re.IGNORECASE,
)

#: A figure with a rupee marker or a preposition in front of it — the original
#: read, kept verbatim so nothing that matched before stops matching.
#:
#: The third alternative is postfix Hinglish — "3000 mein kar do", "2000 pe kar
#: do" — which the original missed because it only looked for the preposition
#: *before* the number. It is the commonest way a guest in this market makes an
#: offer, and the only cost of reading it is that their own figure is printed
#: back in the note under our price.
_ANCHORED_AMOUNT = re.compile(
    r'(?:@|(?:in|for|at|me|mein|pe|to)\s*(?:rs\.?|inr|₹)?)\s*(\d{3,6})\b'
    r'|\b(?:rs\.?|inr|₹)\s*(\d{3,6})\b'
    r'|\b(\d{3,6})\s*(?:rs\.?|inr|₹)?\s*(?:mein|me|pe|to|par|for)\b',
    re.IGNORECASE,
)


def target_price_from_message(message: str) -> float | None:
    """The figure the guest is holding us to, or None if they named none.

    Two reads, in order. An anchored amount ("3000 mein", "@2500", "rs 2000")
    is unambiguous and comes first. A *bare* number is read only when the
    message quotes one back at us — "here is showing 2000 then how can be
    2250" — because a bare number in any other sentence is far more likely to
    be a date, a room count or a night count. The guest's own figure is the
    first such number: when they say "showing 2000, how can be 2250", 2000 is
    the rate they saw and 2250 is ours.

    The handler uses this only to *acknowledge* the guest's number (it is
    printed back in the note under our price). It never becomes the price we
    quote, so a wrong read costs a strange sentence, not a bad rate.
    """
    text = message or ""
    anchored = _ANCHORED_AMOUNT.search(text)
    if anchored:
        return float(anchored.group(1) or anchored.group(2) or anchored.group(3))
    if _QUOTE_BACK_CUE.search(text):
        plain = _PLAIN_AMOUNT.search(text)
        if plain:
            return float(plain.group(1))
    return None


async def handle_price_match(state: ConversationState) -> ConversationState:
    """
    Handle Price Match feature.
    1. Check if price_match_data was extracted from an image.
    2. Validate against session and policy.
    3. Update DB and respond to user.
    """
    price_match_data = state.get("price_match_data")
    session_id = state.get("session_id")
    hotel_id = state.get("hotel_id", 0)
    customer_identifier = state.get("customer_identifier")
    booking_id = state.get("booking_id", "UNKNOWN")

    # If the user just uploaded an image and it was a price match
    if price_match_data is not None:
        updated_state = {**state, "price_match_data": None}
        
        if price_match_data.get("invalid"):
            reason = price_match_data.get("reason", "The uploaded file doesn't appear to be related to a hotel booking or reservation, so I can't verify a price match from it.")
            return cast(ConversationState, {
                **updated_state,
                "next_action": "need_info",
                "bot_response": text_card(
                    f"{reason}\n\nIf you're requesting a price match, please upload a clear hotel booking screenshot or reservation PDF showing the booking details."
                ),
            })
            
        # Create request in MongoDB
        request_id = await price_match_repository.create_price_match_request({
            "session_id": session_id,
            "hotel_id": hotel_id,
            "customer_phone": customer_identifier,
            "booking_id": booking_id,
            "our_price": state.get("total_price", 0.0),
            "status": "PENDING",
            "extracted_data": price_match_data
        })
        
        # Check missing fields
        missing_raw = price_match_data.get("missing_fields") or []
        missing = [m.lower() for m in missing_raw]
        if missing:
            if "price" in missing and len(missing) == 1:
                bot_text = "Everything looks good, but I couldn't find the price on this screenshot. Please upload one showing the final payable amount."
            else:
                bot_text = "I need a screenshot that clearly shows the hotel name, room details, dates, and price so I can verify the offer."
                
            # We don't reject it outright, we just ask for a better picture
            return cast(ConversationState, {
                **updated_state,
                "next_action": "need_info",
                "bot_response": text_card(bot_text),
            })
            
        # Validate logic
        is_valid, errors, final_price, discount = await validate_price_match(price_match_data, state, str(hotel_id))
        
        if is_valid:
            # Approve it!
            await price_match_repository.update_request_status(request_id, "COMPLETED", final_price=final_price)
            
            # Update state with new price
            new_amount_due = final_price - state.get("amount_paid", 0.0)
            updated_state["total_price"] = final_price
            updated_state["amount_due"] = new_amount_due
            
            # Log activity
            ref_id = str(booking_id or session_id or "UNKNOWN")
            await log_activity(ActivityLog(
                hotel_id=hotel_id,
                activity_type="SYSTEM_ACTION",
                reference_id=ref_id,
                booking_id=booking_id or None,
                customer_phone=customer_identifier,
                description=f"Price match approved! Discount of ₹{discount:,.0f} applied. New price: ₹{final_price:,.0f}"
            ))
            
            return cast(ConversationState, {
                **updated_state,
                "next_action": "need_info",
                "bot_response": text_card(
                    f"Great news! 🎉\n\nWe've successfully verified your uploaded offer.\n"
                    f"Your booking qualifies under our Price Match policy.\n\n"
                    f"💰 **Original Price:** ₹{state.get('total_price', 0.0):,.0f}\n"
                    f"🎁 **Discount Applied:** ₹{discount:,.0f}\n"
                    f"✨ **Final Price:** ₹{final_price:,.0f}\n\n"
                    f"Would you like to proceed with the payment for this new price?"
                )
            })
        else:
            # Low confidence or manual review needed?
            confidence = price_match_data.get("confidence") or 0.0
            if confidence < 0.80:
                await price_match_repository.update_request_status(request_id, "MANUAL_REVIEW", errors)
                # Create a support ticket automatically (dummy log for now)
                ref_id = str(booking_id or session_id or "UNKNOWN")
                await log_activity(ActivityLog(
                    hotel_id=hotel_id,
                    activity_type="ESCALATION",
                    reference_id=ref_id,
                    booking_id=booking_id or None,
                    customer_phone=customer_identifier,
                    description=f"Manual review requested for Price Match due to low confidence/mismatch."
                ))
                return cast(ConversationState, {
                    **updated_state,
                    "next_action": "need_info",
                    "bot_response": text_card(
                        "Thank you for the screenshot. The system couldn't automatically verify all details with high confidence. "
                        "I have escalated this to our team for a manual review. They will get back to you shortly!"
                    )
                })
            else:
                # Rejected
                await price_match_repository.update_request_status(request_id, "REJECTED", errors)
                error_bullets = "\n".join([f"• {e}" for e in errors])
                return cast(ConversationState, {
                    **updated_state,
                    "next_action": "need_info",
                    "bot_response": text_card(
                        f"I have reviewed your price match request. Unfortunately, it does not qualify for a price match for the following reason(s):\n\n"
                        f"{error_bullets}\n\n"
                        f"Please let me know if you would like to proceed with the original booking."
                    )
                })
                
    # If there was no image uploaded just now, but the user asked about discounts or price match
    active_req = await price_match_repository.get_active_request_for_session(session_id)
    if active_req:
        return cast(ConversationState, {
            **state,
            "next_action": "need_info",
            "bot_response": text_card("Your price match request is currently under manual review by our team. We will notify you once it's completed.")
        })
        
    current_total = float(state.get("total_price") or 0.0)
    already_discounted = float(state.get("discount_amount") or 0.0)
    base_total = float(state.get("base_total_price") or 0.0)
    if base_total == 0.0:
        base_total = current_total + already_discounted
    if base_total == 0.0 and current_total > 0:
        base_total = current_total

    # The discount ladder below is policy, not mechanism: the client does not
    # offer a direct-booking discount (2026-10-18). Gated rather than deleted so
    # the policy can be restored from an environment variable, which is the only
    # undo this project's operator account can perform on Cloud Run.
    from app.config.settings import get_settings

    discounts_enabled = get_settings().DIRECT_BOOKING_DISCOUNT_ENABLED

    # Fetch seasonal negotiation limit from Season page in Control Centre (MongoDB 'seasons').
    # Only consulted by the ladder — with discounts off it is never read, so it
    # is not fetched.
    check_in = state.get("check_in") or state.get("check_in_date") or ""
    check_out = state.get("check_out") or state.get("check_out_date") or ""

    if discounts_enabled:
        res_limit = await get_season_negotiation_limit(hotel_id, str(check_in), str(check_out))
        if isinstance(res_limit, (tuple, list)):
            max_limit_pct = float(res_limit[0])
            season_name = res_limit[1]
        else:
            max_limit_pct = float(res_limit)
            season_name = None
    else:
        max_limit_pct = 0.0
        season_name = None

    # If guest is asking for discount while room options are shown (no room selected yet):
    # DO NOT auto-select option 1! Explain the direct booking discount and prompt choice.
    if discounts_enabled and current_total == 0 and not state.get("selected_rooms") and state.get("allocation_options"):
        msg = (
            f"🎉 *Exclusive Direct Booking Privilege!*\n\n"
            f"Yes, absolutely! We guarantee better rates than MakeMyTrip or Agoda. "
            f"When you book directly with us, we offer an instant *5% Direct Booking Discount* (up to {max_limit_pct:.0f}% OFF)!\n\n"
            f"Please let me know which room option you prefer (e.g. reply **1** or **2**), "
            f"and I will apply your direct discount immediately! 🏨"
        )
        return cast(ConversationState, {
            **state,
            "next_action": "rooms_shown",
            "bot_response": text_card(msg),
        })

    if current_total == 0 and state.get("selected_rooms"):
        try:
            from app.services.pricing_engine import PricingEngine
            from app.models.schemas import SelectedRoom
            from app.services.hotel_config_service import get_hotel_config

            sel_rooms = state.get("selected_rooms") or []
            if not sel_rooms and state.get("allocation_options"):
                first_opt = state["allocation_options"][0]
                sel_rooms = first_opt.get("rooms") or []
                state["selected_rooms"] = sel_rooms
            elif not sel_rooms and state.get("available_rooms"):
                avail = state["available_rooms"]
                first_r = avail[0]
                sel_rooms = [SelectedRoom(
                    room_id=first_r.get("room_id") or first_r.get("id") or "1",
                    room_name=first_r.get("name") or first_r.get("room_name") or "Deluxe Room",
                    rate_plan_id="EP",
                    rate_plan_name="Room Only",
                    price_per_night=float(first_r.get("base_price") or first_r.get("price") or 2300.0),
                    currency="INR"
                )]
                state["selected_rooms"] = [r.model_dump() if hasattr(r, "model_dump") else r for r in sel_rooms]

            if sel_rooms:
                cfg = await get_hotel_config(hotel_id)
                room_models = [r if isinstance(r, SelectedRoom) else SelectedRoom(**r) for r in sel_rooms]
                bk = await PricingEngine.get_price(
                    selected_rooms=room_models,
                    check_in=str(check_in),
                    check_out=str(check_out),
                    hotel_id=hotel_id,
                    config=cfg,
                    adults=state.get("adults") or 2,
                    children=state.get("children") or 0,
                    session_id=session_id
                )
                base_total = bk.total
                current_total = bk.total
        except Exception as err:
            logger.warning(f"Error auto-calculating base quote during negotiation: {err}")

    if base_total == 0 and state.get("allocation_options"):
        first_opt = state["allocation_options"][0]
        opt_price = float(first_opt.get("total_price_for_stay") or first_opt.get("total_price_per_night") or 0.0)
        if opt_price > 0:
            base_total = opt_price
            current_total = opt_price
            if not state.get("selected_rooms") and first_opt.get("rooms"):
                state["selected_rooms"] = first_opt.get("rooms")

    current_discount_pct = float(state.get("negotiated_discount_pct") or 0.0)
    if current_discount_pct == 0.0 and already_discounted > 0 and base_total > 0:
        current_discount_pct = round((already_discounted / base_total) * 100.0)

    # If guest has a quote and is negotiating
    user_msg = state.get("user_message", "")
    import re
    target_val = target_price_from_message(user_msg)

    if discounts_enabled and current_total > 0:
        next_discount_pct = current_discount_pct + 5.0

        if next_discount_pct <= max_limit_pct:
            # Reduce price by 5% at a time (or 10% on second ask)
            disc_val = float(round((base_total * (next_discount_pct / 100.0)) / 10) * 10)
            disc_val = max(100.0, disc_val)  # Minimum ₹100 incentive
            new_total = float(round((base_total - disc_val) / 10) * 10)

            target_note = ""
            if target_val and target_val < new_total:
                target_note = f"\n\n_Note: While ₹{target_val:,.0f} is below our permissible system limit, I have applied our maximum immediate direct discount for you._"

            if current_discount_pct == 0:
                body_msg = (
                    f"🎉 *Exclusive Direct Booking Privilege!*\n\n"
                    f"We love having you stay with us! To guarantee you get the best possible deal compared to MakeMyTrip/Agoda, I've applied an instant *5% Direct Booking Discount* to your reservation:\n\n"
                    f"💰 *Standard Total:* ₹{base_total:,.0f}\n"
                    f"🎁 *Direct Discount:* -₹{disc_val:,.0f} (5% OFF)\n"
                    f"✨ *Special Direct Total:* ₹{new_total:,.0f}\n\n"
                    f"🌟 *Direct Booking Perks:*\n"
                    f"• Priority early check-in (subject to availability)\n"
                    f"• 100% direct front desk support{target_note}\n\n"
                    f"Would you like to confirm your stay at this special rate of *₹{new_total:,.0f}*?"
                )
            else:
                season_str = f" for {season_name}" if season_name else ""
                headline = f"🤝 *Additional Seasonal Discount Applied{season_str}!*" if season_name else "🤝 *Additional Direct Booking Discount Applied!*"
                body_msg = (
                    f"{headline}\n\n"
                    f"I understand! As a special gesture for booking directly with us, I've reduced the price by another 5% (Total *{next_discount_pct:.0f}% OFF*):\n\n"
                    f"💰 *Standard Total:* ₹{base_total:,.0f}\n"
                    f"🎁 *Total Discount:* -₹{disc_val:,.0f} ({next_discount_pct:.0f}% OFF)\n"
                    f"✨ *Special Direct Total:* ₹{new_total:,.0f}{target_note}\n\n"
                    f"This is our best direct rate for your dates. Would you like to confirm your stay at *₹{new_total:,.0f}*?"
                )

            updated_state = {
                **state,
                "base_total_price": base_total,
                "negotiated_discount_pct": next_discount_pct,
                "total_price": new_total,
                "amount_due": new_total - float(state.get("amount_paid") or 0.0),
                "discount_amount": disc_val,
                "discount_reason": f"Direct Booking Privilege ({next_discount_pct:.0f}% OFF)",
                "booking_stage": "quote_shown",
                "next_action": "show_quote",
                "bot_response": button_card(
                    body_msg,
                    [
                        {"id": "quote_confirm", "title": "Confirm Booking"},
                        {"id": "menu_reception", "title": "Talk to reception"},
                    ],
                ),
            }
            return cast(ConversationState, updated_state)
        else:
            # Minimum limit set in Control Centre reached!
            season_str = f" for {season_name}" if season_name else ""
            limit_msg = (
                f"We have already applied our maximum direct discount of *{current_discount_pct:.0f}% OFF*{season_str}.\n\n"
                f"💰 *Best Direct Total:* ₹{current_total:,.0f}\n\n"
                f"This is our absolute lowest rate — guaranteed lower than online portals (MMT/Agoda/Booking.com) with direct front desk support.\n\n"
                f"Shall I confirm your booking at *₹{current_total:,.0f}*?"
            )
            return cast(ConversationState, {
                **state,
                "booking_stage": "quote_shown",
                "next_action": "show_quote",
                "bot_response": button_card(
                    limit_msg,
                    [
                        {"id": "quote_confirm", "title": "Confirm Booking"},
                        {"id": "menu_reception", "title": "Talk to reception"},
                    ],
                ),
            })

    # Two versions of the same offer, and the difference is one clause: the
    # comparison to OTAs by name. `understand` routes here on three conditions
    # and only two of them involve a portal — the third is a bare "any
    # discount?" with no dates, which reaches this line with no screenshot and
    # nobody having mentioned MakeMyTrip. Naming a portal the guest did not
    # raise is an unasked-for claim about a competitor's price; naming one they
    # did raise is the answer to their question. product rule, 2026-10-14.
    #
    # The hotel's own switch rides along: a property that has turned OTA
    # comparison off must not have portals named for it at all, whatever the
    # guest asked. Default is True, so on every hotel today only the guest's
    # own words decide — which is what makes this a fix rather than a dark flag.
    # Imported here rather than at module scope because the sibling import at
    # line 193 sits inside a `try` that only runs on the screenshot path, so it
    # is not bound by the time this branch is reached.
    from app.services.hotel_config_service import get_hotel_config

    cfg = await get_hotel_config(hotel_id)
    # The path is `business_config.features`, not `.features` — `HotelConfig` has
    # no such attribute, so a `getattr(cfg, "features", None)` chain here would
    # have returned None, defaulted to True, and quietly gated nothing. Verified
    # against a live config 2026-10-14 (Sample Hotel: True).
    ota_ok = bool(cfg.business_config.features.ota_comparison_enabled)

    # A guest who already holds a quote is shown it, not asked for their dates.
    # The reply below used to end every branch with "let me know your travel
    # dates and guest count" no matter what `state` held, because the whole
    # priced ladder above sits behind `discounts_enabled` and the fall-through
    # never looked at `current_total`. Measured 2026-10-28 (Prit's screenshot):
    # a guest mid-conversation typed "thoda aur kam karo please, aur discount
    # do" and was asked for dates and guest count the bot already had — the
    # date parser had run, `missing_fields` was empty.
    #
    # The ladder is off by policy (2026-10-18), so there is no percentage to
    # claim. What can be said truthfully is the price we hold and that it is
    # fixed — which is the answer to "give me a discount", and the sentence
    # Prit expected, minus the 5% line that only holds when the ladder is on.
    if current_total > 0:
        _dates_phrase = ""
        if check_in and check_out:
            _dates_phrase = f" for *{check_in}* to *{check_out}*"
        # The portal sentence obeys the same rule as the offer below (Product,
        # 2026-10-14): name a competitor only when the guest named one, and
        # only when the hotel's own OTA switch is on. A guest who asked for
        # "a discount" gets the price and nothing about MakeMyTrip.
        if ota_ok and mentions_competitor(state.get("user_message")):
            lowest_clause = (
                "This is our absolute lowest rate — guaranteed at or below the "
                "online portals (MMT/Agoda/Booking.com), with direct front desk "
                "support."
            )
        else:
            lowest_clause = (
                "This is our lowest book-direct rate, and it includes direct "
                "front desk support."
            )
        target_note = ""
        if target_val and target_val < current_total:
            target_note = f"\n\n_Note: While ₹{target_val:,.0f} is below our permissible system limit, we have already applied our best direct rate for you._"
        price_msg = (
            f"💰 *Best Direct Total{_dates_phrase}:* ₹{current_total:,.0f}\n\n"
            f"{lowest_clause}{target_note}\n\n"
            f"Shall I confirm your booking at *₹{current_total:,.0f}*?"
        )
        return cast(ConversationState, {
            **state,
            "booking_stage": "quote_shown",
            "next_action": "show_quote",
            "bot_response": await translate_cards_if_needed(
                button_card(
                    price_msg,
                    [
                        {"id": "quote_confirm", "title": "Confirm Booking"},
                        {"id": "menu_reception", "title": "Talk to reception"},
                    ],
                ),
                state,
            ),
        })

    # No quote yet. Ask only for what is actually missing: when the guest has
    # already given the dates, naming them and moving on to rooms is the reply
    # DEFECTS item 2 asks for; asking again is the defect.
    adults = state.get("adults")
    if check_in and check_out:
        if adults:
            next_step = (
                f"I have your dates — *{check_in}* to *{check_out}* ({adults} guests). Let me pull up "
                f"the rooms and rates for those nights."
            )
            next_action = "search_rooms"
        else:
            next_step = (
                f"I have your dates — *{check_in}* to *{check_out}*. How many guests (adults and children) will be staying?"
            )
            next_action = "need_guests"
    elif adults:
        next_step = (
            f"Please share your check-in and check-out dates for {adults} guests, and I'll quote you our best direct rate."
        )
        next_action = "need_dates"
    else:
        next_step = (
            "Let me know your travel dates and guest count (adults and children) and I'll quote you "
            "our best direct rate."
        )
        next_action = "need_dates"

    if ota_ok and mentions_competitor(state.get("user_message")):
        if not discounts_enabled:
            offer = (
                "We match any lower rate you find on MakeMyTrip, Agoda or "
                "Booking.com! Send a screenshot showing the dates and the price "
                "and we will match it.\n\n" + next_step
            )
        else:
            offer = (
                f"We offer exclusive direct booking discounts (up to {max_limit_pct:.0f}% lower than OTAs like MakeMyTrip/Agoda) and match any lower online rate!\n\n"
                f"If you found a cheaper price elsewhere, simply send a screenshot showing the dates and price, and I'll match it immediately.\n\n"
                + next_step
            )
    else:
        # Per product guidance: When user asks for best direct rate, offers, or discounts without
        # mentioning a competitor, do NOT ask for a screenshot. Ask for dates and number of persons!
        if discounts_enabled:
            offer = (
                f"We offer exclusive direct booking discounts of up to {max_limit_pct:.0f}% off our best rate!\n\n"
                + next_step
            )
        else:
            offer = next_step

    new_stage = state.get("booking_stage")
    if new_stage in ("initial", "new", None):
        new_stage = "collecting"

    return cast(ConversationState, {
        **state,
        "booking_stage": new_stage,
        "next_action": next_action,
        # Translated, like every other guest-facing node builds its cards
        # (`room_searcher`, `room_query_handler`). Without this line the reply
        # was English-only whatever the guest spoke — Prit's screenshot: a
        # Hinglish "thoda aur kam karo" answered in English.
        "bot_response": await translate_cards_if_needed(text_card(offer), state),
    })
