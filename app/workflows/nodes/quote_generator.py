"""
Quote Generator Node — deterministic pricing calculation.
"""

from app.models.schemas import SelectedRoom
from app.services.booking_engine import (
    apply_early_late_fee,
    apply_staff_total,
    calculate_price_breakdown,
    format_price_quote,
)
from app.services.early_late_checkout_service import resolve_early_late_fees
from app.workflows.state import ConversationState


def _decline_the_mattress(room_models: list[SelectedRoom]) -> list[SelectedRoom]:
    """Apply a guest's "no extra bed" to the selected rooms, in place.

    "No extra bed" declines a *mattress*. On a CP / MAP plan it does not decline
    the extra-person rate — that adult is still in the room, still eating the
    breakfast the plan was sold to cover — so a charge the allocator recorded
    with `extra_bed_count == 0` is not a mattress and is left standing. On EP,
    and on any selection written before that field existed, the charge *was* the
    mattress and declining it removes the money exactly as it always did.

    The child charge is cleared either way: nothing here sells a child a
    mattress, and refusing one has always meant refusing that too.
    """
    for room_model in room_models:
        if room_model.extra_bed_count != 0:
            room_model.extra_adult_charge = 0.0
            room_model.extra_bed_count = 0
        room_model.extra_child_charge = 0.0
    return room_models


async def generate_quote(state: ConversationState) -> ConversationState:
    """
    Calculate pricing from selected rooms. Pure math — no AI.

    Uses basePrice from the selected rate plan × num_rooms × num_nights + GST.
    """
    # Using text_card for missing selection
    from app.prompts.templates import text_card

    selected_rooms = state.get("selected_rooms", [])
    if not selected_rooms:
        return {**state, "bot_response": text_card("Please select a room first."), "next_action": "need_selection"}

    # Convert dicts back to models
    room_models = [SelectedRoom(**r) for r in selected_rooms]

    # If the user requested a specific meal plan upgrade, apply it to the selected rooms.
    # Note: the allocation engine already applies meal_pref during room search, so this block
    # acts as a fallback for when meal_pref is set after a room has already been selected.
    meal_pref = state.get("meal_plan_preference")
    if meal_pref:
        from app.repositories import room_repository
        from app.services.allocation_engine import get_preferred_rate_plan
        hotel_id = state.get("hotel_id", 0)
        all_rooms = await room_repository.get_all_rooms(hotel_id)

        updated = False
        for room_model in room_models:
            # Find the full room object
            full_room = next((r for r in all_rooms if r.room_id == room_model.room_id), None)
            if full_room:
                pref_rp = get_preferred_rate_plan(full_room, meal_pref)
                if pref_rp and pref_rp.rate_plan_id != room_model.rate_plan_id:
                    room_model.rate_plan_id = pref_rp.rate_plan_id
                    room_model.rate_plan_name = pref_rp.name
                    room_model.price_per_night = pref_rp.base_price
                    room_model.currency = pref_rp.currency
                    updated = True

        if updated:
            # Save the updated room models back to state
            state["selected_rooms"] = [r.model_dump() for r in room_models]
            # Keep meal_plan_preference in state so modifications/requotes remember the preference

    # If the guest requested no extra bed, zero out extra bed charges.
    if state.get("no_extra_bed") or state.get("extra_bed_preference") == "NO":
        room_models = _decline_the_mattress(room_models)
        state["selected_rooms"] = [r.model_dump() for r in room_models]

    # Get config
    from app.config.hotel_config import HotelConfig
    config_dict = state.get("hotel_config")
    config = HotelConfig(**config_dict) if config_dict else HotelConfig(hotel_id=state.get("hotel_id", 0))

    from app.services.feature_gate import FeatureGate
    gate = FeatureGate(config, session_id=state.get("session_id"))

    from app.services.pricing_engine import PricingEngine
    from app.services.discount_engine import DiscountEngine
    from app.repositories.activity_repository import log_activity, ActivityLog
    from app.services.override_resolver import resolve_payable_override
    import logging
    import asyncio

    # The deterministic pricing engine ALWAYS runs. A staff-negotiated total is an
    # overlay applied below (apply_staff_total), never a replacement.
    #
    # This call used to sit in an `else:` branch guarded by `price_override is not
    # None`. Because context_loader wrote the calculated total back into
    # price_override, that guard was true on every re-quote, so the engine was
    # skipped and the summary was built from a stub with num_nights hardcoded to 1
    # and every room line carrying the whole total.
    breakdown = await PricingEngine.get_price(
        selected_rooms=room_models,
        check_in=state.get("check_in", ""),
        check_out=state.get("check_out", ""),
        hotel_id=state.get("hotel_id", 0),
        config=config,
        adults=state.get("adults") or 2,
        children=state.get("children") or 0,
        extra_services=state.get("extra_services", []),
        session_id=state.get("session_id")
    )

    coupon_msg = ""
    # Process Coupon if provided
    provided_coupon = state.get("coupon_code")
    applied_coupon = state.get("applied_coupon")

    if gate.can_use_coupon() and provided_coupon and provided_coupon != applied_coupon:
        coupon_obj = await DiscountEngine.get_coupon(provided_coupon, state.get("hotel_id", 0))
        if coupon_obj:
            is_valid, reason = DiscountEngine.validate_coupon(coupon_obj, breakdown, state.get("check_in", ""))
            if is_valid:
                breakdown = DiscountEngine.apply_discount(breakdown, coupon_obj)
                state["applied_coupon"] = provided_coupon
                state["coupon_code"] = None # consume it
                coupon_msg = f"🎉 Success! Promotional code **{provided_coupon}** has been applied.\n\n"

                # Log success
                asyncio.create_task(log_activity(ActivityLog(
                    hotel_id=int(state.get("hotel_id", 0)),
                    reference_id=str(state.get("booking_id") or state.get("session_id") or "SYS"),
                    session_id=state.get("session_id", ""),
                    booking_id=state.get("booking_id"),
                    activity_type="SYSTEM_EVENT",
                    description=f"COUPON_APPLIED: {provided_coupon}"
                )))
            else:
                state["coupon_code"] = None # consume invalid coupon to avoid loops
                coupon_msg = f"This promotional code is currently unavailable. ({reason})\n\n"

                # Log rejection
                asyncio.create_task(log_activity(ActivityLog(
                    hotel_id=int(state.get("hotel_id", 0)),
                    reference_id=str(state.get("booking_id") or state.get("session_id") or "SYS"),
                    session_id=state.get("session_id", ""),
                    booking_id=state.get("booking_id"),
                    activity_type="SYSTEM_EVENT",
                    description=f"COUPON_REJECTED: {provided_coupon} - {reason}"
                )))
        else:
            state["coupon_code"] = None
            coupon_msg = "This promotional code is currently unavailable. Please verify the code and try again.\n\n"
            asyncio.create_task(log_activity(ActivityLog(
                hotel_id=int(state.get("hotel_id", 0)),
                reference_id=str(state.get("booking_id") or state.get("session_id") or "SYS"),
                session_id=state.get("session_id", ""),
                booking_id=state.get("booking_id"),
                activity_type="SYSTEM_EVENT",
                description=f"COUPON_REJECTED: {provided_coupon} - Not Found"
            )))
    elif gate.can_use_coupon() and applied_coupon and (breakdown.discount_amount or 0.0) == 0.0:
        coupon_obj = await DiscountEngine.get_coupon(applied_coupon, state.get("hotel_id", 0))
        if not coupon_obj:
            nights_count = breakdown.rooms[0].num_nights if (breakdown and breakdown.rooms) else 1
            coupon_obj = await DiscountEngine.get_active_default_promotion(
                state.get("hotel_id", 0),
                state.get("check_in"),
                check_out_date=state.get("check_out"),
                num_nights=nights_count,
            )
        if coupon_obj:
            is_valid, _ = DiscountEngine.validate_coupon(coupon_obj, breakdown, state.get("check_in", ""))
            if is_valid:
                breakdown = DiscountEngine.apply_discount(breakdown, coupon_obj)
                state["discount_amount"] = breakdown.discount_amount
                state["negotiated_discount_pct"] = (
                    coupon_obj.discount_value if coupon_obj.discount_type == "percentage" else 0.0
                )
    elif gate.can_use_coupon() and not applied_coupon and (breakdown.discount_amount or 0.0) == 0.0:
        nights_count = breakdown.rooms[0].num_nights if (breakdown and breakdown.rooms) else 1
        default_promo = await DiscountEngine.get_active_default_promotion(
            state.get("hotel_id", 0),
            state.get("check_in"),
            check_out_date=state.get("check_out"),
            num_nights=nights_count,
        )
        if default_promo:
            is_valid, _ = DiscountEngine.validate_coupon(default_promo, breakdown, state.get("check_in", ""))
            if is_valid:
                breakdown = DiscountEngine.apply_discount(breakdown, default_promo)
                state["applied_coupon"] = default_promo.coupon_code
                state["discount_amount"] = breakdown.discount_amount
                state["negotiated_discount_pct"] = (
                    default_promo.discount_value if default_promo.discount_type == "percentage" else 0.0
                )
                promo_label = (
                    f"{default_promo.discount_value:.0f}%"
                    if default_promo.discount_type == "percentage"
                    else f"₹{default_promo.discount_value:,.0f}"
                )
                p_type = (default_promo.promotion_type or "").lower()
                p_name = (default_promo.promotion_name or default_promo.coupon_code or "").lower()

                if "early-bird" in p_type or "early bird" in p_name:
                    if default_promo.base_discount and default_promo.add_on_discount:
                        coupon_msg = (
                            f"🎉 Early Bird Special: **{promo_label} Combined Discount** "
                            f"({default_promo.base_discount:.0f}% Base + {default_promo.add_on_discount:.0f}% Early Bird) "
                            f"applied automatically for booking in advance!\n\n"
                        )
                    else:
                        coupon_msg = (
                            f"🎉 Early Bird Special: **{promo_label} Discount** "
                            f"applied automatically for booking in advance!\n\n"
                        )
                elif "long-stay" in p_type or "long stay" in p_name:
                    nights_label = f" for your extended stay of {nights_count} nights" if nights_count > 1 else ""
                    if default_promo.base_discount and default_promo.add_on_discount:
                        coupon_msg = (
                            f"🎉 Long Stay Deal: **{promo_label} Combined Discount** "
                            f"({default_promo.base_discount:.0f}% Base + {default_promo.add_on_discount:.0f}% Long Stay) "
                            f"applied automatically{nights_label}!\n\n"
                        )
                    else:
                        coupon_msg = (
                            f"🎉 Long Stay Deal: **{promo_label} Discount** "
                            f"applied automatically{nights_label}!\n\n"
                        )
                elif "last-minute" in p_type or "last minute" in p_name:
                    if default_promo.base_discount and default_promo.add_on_discount:
                        coupon_msg = (
                            f"🎉 Last Minute Deal: **{promo_label} Combined Discount** "
                            f"({default_promo.base_discount:.0f}% Base + {default_promo.add_on_discount:.0f}% Last Minute) "
                            f"applied automatically!\n\n"
                        )
                    else:
                        coupon_msg = (
                            f"🎉 Last Minute Deal: **{promo_label} Discount** "
                            f"applied automatically!\n\n"
                        )
                else:
                    friendly_title = (
                        default_promo.promotion_name
                        if default_promo.promotion_name and not any(k in default_promo.promotion_name.lower() for k in ["basic-promotion", "basic_promotion"])
                        else "Direct Booking"
                    )
                    coupon_msg = f"🎉 Special Offer: **{promo_label} {friendly_title} Discount** applied automatically!\n\n"

                asyncio.create_task(log_activity(ActivityLog(
                    hotel_id=int(state.get("hotel_id", 0)),
                    reference_id=str(state.get("booking_id") or state.get("session_id") or "SYS"),
                    session_id=state.get("session_id", ""),
                    booking_id=state.get("booking_id"),
                    activity_type="SYSTEM_EVENT",
                    description=f"DEFAULT_PROMOTION_APPLIED: {default_promo.coupon_code} ({promo_label})"
                )))
    elif not gate.can_use_coupon() and provided_coupon:
        state["coupon_code"] = None
        coupon_msg = "Promotional offers are currently unavailable.\n\n"

    if (breakdown.discount_amount or 0.0) == 0.0 and float(state.get("negotiated_discount_pct") or 0.0) > 0.0:
        disc_pct = float(state["negotiated_discount_pct"])
        disc_val = float(round((breakdown.subtotal * (disc_pct / 100.0)) / 10) * 10)
        breakdown.discount_amount = disc_val
        breakdown.total = max(0.0, breakdown.subtotal - disc_val + breakdown.tax_amount)
        breakdown.discount_reason = state.get("discount_reason") or f"Direct Booking Privilege ({disc_pct:.0f}% OFF)"
        state["discount_amount"] = disc_val

    # Upsell disabled per design — clean price breakdown only
    upsell_msg = ""
    upsell_cards = []

    # Prepare dynamic assumptions based on selected rooms
    assumptions = state.get("assumptions") or []
    if room_models:
        has_non_ac = any("non ac" in r.room_name.lower() or "non-ac" in r.room_name.lower() for r in room_models)
        has_ac = any("non ac" not in r.room_name.lower() and "non-ac" not in r.room_name.lower() for r in room_models)

        sample_plan = (room_models[0].rate_plan_name or "").upper()
        meal_pref = (state.get("meal_plan_preference") or "").upper()
        if "CP" in sample_plan or "BREAKFAST" in sample_plan or meal_pref in ("CP", "BREAKFAST"):
            meal_label = "Breakfast Included"
        elif "MAP" in sample_plan or "DINNER" in sample_plan or meal_pref in ("MAP", "HALF_BOARD", "DINNER"):
            meal_label = "Breakfast & Dinner"
        else:
            meal_label = "Room Only"

        if has_non_ac and not has_ac:
            assumptions = ["Non-AC", meal_label]
        elif has_ac and not has_non_ac:
            assumptions = ["AC", meal_label]
        elif has_ac and has_non_ac:
            assumptions = ["Mixed (AC + Non-AC)", meal_label]
        else:
            assumptions = [meal_label]

    # Paid early check-in / late check-out, when the guest themselves stated a
    # time. Sits between the coupon and the staff overlay on purpose: a coupon
    # rebuilds `total` from `subtotal` and would wipe the fee, while a staff
    # total replaces `total` outright and must be the last word — so it has to
    # see the fee already in place for its adjustment line to reconcile.
    #
    # Priced off one night of the rooms actually booked, at their catalogue
    # rate, so a coupon or a staff discount never changes what the extra hours
    # cost the hotel.
    early_late_fees = await resolve_early_late_fees(
        state.get("hotel_id", 0),
        check_in_date=state.get("check_in"),
        check_out_date=state.get("check_out"),
        room_names=[r.room_name for r in room_models],
        one_night_rate=sum(
            (r.price_per_night or 0.0) * (r.quantity or 1) for r in breakdown.rooms
        ),
        requested_arrival_time=state.get("requested_arrival_time"),
        requested_departure_time=state.get("requested_departure_time"),
    )
    breakdown = apply_early_late_fee(breakdown, early_late_fees)

    # A staff-negotiated total wins over the computed one, including over any coupon
    # applied above. Applied as an overlay so the line items, nights, quantities and
    # GST stay real and the printed summary still reconciles.
    #
    # product rule, 2026-11-09: what staff quoted takes top priority, even when
    # it is *above* the catalogue total. The old guard discarded any staff total
    # >= the computed one on the theory that it must be a stale override from a
    # costlier room or longer stay. That theory could not tell "stale" from
    # "staff priced a bigger party than the bot allocated" — and the latter is
    # exactly what happened live on 2026-11-09, where staff said ₹50,000 for six
    # rooms and the bot silently fell back to its own four-room ₹44,996. A stale
    # override is corrected by staff overriding again; a live staff price is
    # never corrected by guessing. So: apply it, and log loudly when it sits
    # above catalogue so the case is findable in production.
    staff_total = resolve_payable_override(state)
    if staff_total is not None:
        computed_total = float(breakdown.total or 0.0)
        breakdown = apply_staff_total(breakdown, staff_total)
        if staff_total >= computed_total:
            logging.getLogger(__name__).info(
                "STAFF_TOTAL_ABOVE_CATALOGUE | staff_total=%s >= computed=%s | "
                "applying staff price (top priority)",
                staff_total, computed_total,
            )

    # "No extra bed for N people", for a room that seats the party without a
    # mattress. product case of 2026-09-29: Sample Hotel's Deluxe Room King Bed
    # has base_adults 2 and max_adults 3, so three guests sleep there with no bed
    # added and nothing extra charged. The summary showed a room rate and no bed
    # line, and a guest cannot tell from an absent line whether the bot
    # understood, quietly dropped a charge, or is about to bill them at
    # check-in.
    #
    # Computed here rather than in `format_price_quote` because the breakdown
    # carries no occupancy -- `RoomLineItem` has a rate, a quantity and the two
    # bed charges, and nothing about how many the room sleeps. The catalogue
    # does, and only the caller can reach it.
    #
    # The count is `adults + children` -- the same party the header above prints
    # and the same source the room card's own note uses, so the two cards cannot
    # disagree about how many people are being seated.
    no_extra_bed_note = None
    if len(breakdown.rooms) == 1 and int(breakdown.rooms[0].quantity or 0) == 1:
        _line = breakdown.rooms[0]
        # A mattress, specifically — not merely a charge. A CP / MAP guest can
        # owe the extra-person rate and still have been offered no bed, and the
        # note exists to tell exactly that guest they are being seated without
        # one. `None` keeps the old charge-based read.
        _has_a_mattress = (
            bool(_line.extra_bed_count)
            if _line.extra_bed_count is not None
            else (_line.extra_adult_charge or 0) > 0
        )
        if not (_has_a_mattress or (_line.extra_child_charge or 0) > 0):
            try:
                from app.repositories import room_repository

                _catalogue = await room_repository.get_all_rooms(state.get("hotel_id", 0))
                _full = next((r for r in _catalogue if r.room_id == room_models[0].room_id), None)
                _seats = int(getattr(getattr(_full, "occupancy", None), "base_adults", 0) or 0)
                # `effective_*` first, exactly as `_card_party` reads them. Raw
                # `adults` is the number the guest first typed; `effective_adults`
                # is that plus any child the hotel's policy counts as an adult,
                # and it is the one the room card's own note uses. Reading the
                # raw counts here let the two cards name two different party
                # sizes for the same booking.
                _party = int(
                    state.get("effective_adults")
                    if state.get("effective_adults") is not None
                    else (state.get("adults") or 0)
                ) + int(
                    state.get("effective_children")
                    if state.get("effective_children") is not None
                    else (state.get("children") or 0)
                )
                if _seats and _party > _seats:
                    no_extra_bed_note = f"🛏️ *No extra bed for {_party} people*"
            except Exception:
                # A missing catalogue must not cost the guest their quote, and a
                # row that quietly claims the wrong number is worse than a
                # silent one. Same posture as the room card: say nothing.
                logging.getLogger(__name__).warning(
                    "no-extra-bed note skipped: catalogue lookup failed for hotel_id=%s",
                    state.get("hotel_id"),
                    exc_info=True,
                )

    quote_text = format_price_quote(
        breakdown=breakdown,
        check_in=state.get("check_in", ""),
        check_out=state.get("check_out", ""),
        adults=state.get("adults") or 2,
        children=state.get("children") or 0,
        option_label=state.get("selected_option_label"),
        assumptions=assumptions,
        no_extra_bed_note=no_extra_bed_note,
    )

    # ── Several bookings in one conversation ──────────────────────────────
    # A queue of two or more stays is collected before anything is paid. This
    # stay's price is computed exactly as it always is (above); what changes is
    # what the guest is shown next. A stay that is not the last one is stashed
    # and the conversation moves to the next stay's card; the last one closes
    # the collection with a single quote naming every stay and a grand total.
    # Gated — `is_collecting` is False whenever MULTI_STAY_COMBINED_ENABLED is
    # off, so the old path below runs untouched.
    from app.utils import multi_stay

    if multi_stay.is_collecting(state):
        # `total_price`/`price_breakdown` are written to the result below, not
        # to `state`, so the stash is fed a state that already carries them.
        # Stashing from the raw state saved `None` and the combined quote
        # totalled zero — the failure the tests below pin.
        priced = {
            **state,
            "total_price": breakdown.total,
            "price_breakdown": breakdown.model_dump(),
            "selected_rooms": [r.model_dump() for r in room_models],
        }
        updates = multi_stay.stash_and_advance(priced, quote_text) or multi_stay.stash_and_combine(
            priced, quote_text
        )
        return {**priced, **updates}

    quote_card = {
        "type": "quote",
        "text": quote_text,
        "data": breakdown.model_dump(),
        "buttons": [
            {"id": "quote_confirm", "title": "Confirm Booking", "label": "Confirm Booking", "action": "postback", "value": "confirm booking"}
        ]
    }

    min_advance = round((breakdown.total * 0.30) / 10) * 10

    if gate.can_direct_book():
        follow_up_msg = coupon_msg + (
            f"Would you like to proceed and confirm this booking?\n\n"
            f"✅ Type **'Confirm'** or **'Yes'** to get your advance payment link (30% = ₹{min_advance:,.0f}).\n"
            f"💳 Or mention a custom advance amount you'd like to pay now (minimum ₹{min_advance:,.0f}).\n"
            f"   _Example: \"I'll pay {min_advance:,.0f} now\"_"
        ) + upsell_msg
    else:
        follow_up_msg = coupon_msg + (
            f"Would you like to proceed and confirm this booking?\n\n"
            f"✅ Type **'Confirm'** or **'Yes'** to send a booking inquiry to our team."
        ) + upsell_msg

    # Mention meal plan upgrades footnote if not explicitly requested
    meal_pref = state.get("meal_plan_preference")
    if not meal_pref:
        follow_up_msg += "\n\n_Breakfast and meal plan upgrades available — ask me for details._"

    # Mention coupon availability only once during pricing if enabled
    if gate.can_use_coupon() and not state.get("applied_coupon"):
        follow_up_msg += "\n\nIf you have a promotional code, you may share it before completing payment."

    media_cards = []
    already_sent_photos = bool(state.get("quote_photos_sent") or state.get("room_previews_sent"))
    if room_models and (not already_sent_photos or state.get("wants_photos")):
        from app.services.hotel_media_service import media_delivery_card

        seen_ids = set()
        for rm in room_models:
            rid = str(rm.room_id or "").strip()
            rname = str(rm.room_name or "Room").strip()
            if not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)
            aliases = rm.aliases if hasattr(rm, "aliases") and rm.aliases else []
            media_cards.append(
                media_delivery_card(
                    hotel_id=state.get("hotel_id"),
                    entity_type="room",
                    entity_id=rid,
                    tags=[rname],
                    aliases=aliases,
                    caption=f"{rname}",
                    limit=2,
                    is_preview=True,
                )
            )

    res = {
        **state,
        "applied_coupon": state.get("applied_coupon"),
        "coupon_code": state.get("coupon_code"),
        "discount_amount": breakdown.discount_amount,
        "discount_reason": breakdown.discount_reason,
        "negotiated_discount_pct": state.get("negotiated_discount_pct", 0.0),
        "upsell_shown": state.get("upsell_shown", False),
        "price_breakdown": breakdown.model_dump(),
        "total_price": breakdown.total,
        "booking_stage": "quote_shown",
        "quote_photos_sent": True,
        "wants_photos": False,
        "bot_response": [
            quote_card,
            *text_card(follow_up_msg),
            *upsell_cards,
            *media_cards,
        ],
        "next_action": "quote_shown",
    }
    return res
