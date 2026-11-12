import logging
from datetime import datetime, timezone
from typing import cast

from app.workflows.state import ConversationState
from app.services.room_service import find_suitable_rooms
from app.workflows.nodes.quote_generator import generate_quote
from app.services.room_alias_resolver import aliases_from_hotel_config
from app.prompts import templates
from app.services.lifecycle_manager import (
    build_change_summary,
    build_final_modification_confirmation,
    detect_early_checkout,
    detect_stay_extension,
    format_early_checkout_summary,
    format_stay_extension_summary,
    invalidate_payment_links,
    log_modification_audit,
    requires_confirmation,
)

logger = logging.getLogger(__name__)


async def handle_modification(state: ConversationState) -> ConversationState:
    """
    Core Validation & Recalculation Engine for modifications.
    Safely applies pending changes (requested_values) by validating availability, 
    rechecking constraints, updating quotes, and invalidating old payment links.
    """
    req_vals = state.get("requested_values")
    if not req_vals:
        return state
        
    hotel_id = state.get("hotel_id", 0)
    
    # ── LIFECYCLE INTERCEPT: In-stay checkout modifications ───────────
    booking_status = state.get("booking_status", "new")
    if booking_status == "checked_in" and "check_out" in req_vals:
        # User is already staying and wants to change their checkout date
        new_co = req_vals["check_out"]
        orig_co = state.get("check_out", "")
        
        # Determine if it's an extension or early checkout
        if new_co > orig_co:
            res_ext = detect_stay_extension(state, new_co)
            if res_ext.valid:
                summary = format_stay_extension_summary(res_ext, state)
                return cast(ConversationState, {
                    **state,
                    "lifecycle_action_pending": "stay_extension",
                    "lifecycle_action_data": {
                        "new_check_out": res_ext.new_check_out,
                        "extra_nights": res_ext.extra_nights,
                        "additional_amount": res_ext.additional_amount,
                    },
                    "bot_response": templates.text_card(summary),
                    "next_action": "lifecycle_awaiting_confirmation",
                    "intent": "LIFECYCLE_RESET",
                    "requested_values": None,  # clear pending mod
                })
        elif new_co < orig_co:
            res_ec = detect_early_checkout(state, new_co)
            if res_ec.valid:
                summary = format_early_checkout_summary(res_ec, state)
                return cast(ConversationState, {
                    **state,
                    "lifecycle_action_pending": "early_checkout",
                    "lifecycle_action_data": {
                        "new_check_out": res_ec.new_check_out,
                        "nights_removed": res_ec.nights_removed,
                        "revised_total": res_ec.revised_total,
                    },
                    "bot_response": templates.text_card(summary),
                    "next_action": "lifecycle_awaiting_confirmation",
                    "intent": "LIFECYCLE_RESET",
                    "requested_values": None,  # clear pending mod
                })
    
    # 1. Shadow State Construction
    shadow_state = dict(state)
    for k, v in req_vals.items():
        shadow_state[k] = v
        
    adults = shadow_state.get("adults", 2)
    children = shadow_state.get("children", 0)
    num_rooms = shadow_state.get("num_rooms")
    room_type_pref = shadow_state.get("room_type_preference")
    feature_preferences = shadow_state.get("feature_preferences")
    budget = shadow_state.get("budget")
    check_in = shadow_state.get("check_in")
    check_out = shadow_state.get("check_out")
    meal_pref = shadow_state.get("meal_plan_preference")
    
    hotel_cfg = shadow_state.get("hotel_config") or {}
    aliases = aliases_from_hotel_config(hotel_cfg)
    threshold = 90.0
    if isinstance(hotel_cfg, dict) and hotel_cfg.get("room_match_threshold") is not None:
        try:
            threshold = float(hotel_cfg["room_match_threshold"])
        except (TypeError, ValueError):
            pass

    # 2. Inventory Re-Validation
    allocations, fell_back = await find_suitable_rooms(
        hotel_id=hotel_id,
        adults=adults,
        children=children,
        num_rooms=num_rooms,
        room_type_preference=room_type_pref,
        budget=budget,
        check_in=check_in,
        check_out=check_out,
        meal_plan_preference=meal_pref,
        room_aliases=aliases,
        room_match_threshold=threshold,
        feature_preferences=feature_preferences,
        room_requirements=shadow_state.get("room_requirements"),
        wants_bed=bool(state.get("wants_bed")),
    )
    
    # 3. Outcome 1: Complete Failure
    if not allocations:
        failure_msg = "I'm sorry, we don't have any rooms available for those updated details."
        return cast(ConversationState, {
            **state,
            "bot_response": templates.text_card(failure_msg),
            "requested_values": None,
            "modified_fields": None,
            "booking_change_summary": None,
            "modification_metadata": None,
            "intent": "BOOK_ROOM"
        })
        
    # Check if we were forced to allocate more rooms than requested/currently held
    forced_more_rooms = False
    if num_rooms is not None:
        forced_more_rooms = all(len(opt.rooms) > num_rooms for opt in allocations)
    else:
        # Check against previously selected rooms if any
        prev_rooms = state.get("selected_rooms")
        if prev_rooms and len(prev_rooms) > 0:
            forced_more_rooms = all(len(opt.rooms) > len(prev_rooms) for opt in allocations)
            
    # Check if the exact room type matching failed
    # But only if they had a selected room! If they were just changing dates before selecting, 
    # we don't have a "previously selected" room to protect.
    had_selection = bool(state.get("selected_rooms"))
    
    # We cannot auto-apply if:
    # A) They need more rooms to fit the guests
    # B) Their previously selected room type is completely sold out (fell_back is True) 
    #    AND they actually had a selected room.
    cannot_auto_apply = forced_more_rooms or (fell_back and had_selection)

    # 4. Outcome 2: Capacity Exceeded / Auto-Apply Failed
    if cannot_auto_apply:
        # Generate graceful options response
        options_cards = _room_preview_media_cards(allocations, state)
        
        from app.repositories.hotel_repository import get_hotel_contact_info
        contact = await get_hotel_contact_info(hotel_id) or {}
        phone = contact.get("phone") or "the hotel front desk"
            
        reason = "Your requested dates or room type are no longer available"
        if forced_more_rooms:
            reason = f"Your requested group size exceeds the maximum capacity of your current selection"
            
        text = f"I'm sorry, {reason}. Here are some available options that accommodate your request:"
        
        # We return the original state (so active booking remains untouched),
        # but we present the new options and wait for selection.
        return cast(ConversationState, {
            **state,
            "allocation_options": [a.model_dump() for a in allocations],
            "room_type_not_available": fell_back,
            "booking_stage": "rooms_shown",
            "next_action": "need_selection",
            "bot_response": templates.text_card(text) + options_cards,
            "intent": "BOOK_ROOM"
            # Note: We do NOT clear requested_values here. They remain pending 
            # until the guest chooses an option.
        })

    # 5. Outcome 3: Perfect Match — decide auto-apply vs. confirmation gate
    # We auto-select the best matching allocation
    top_allocation = allocations[0]
    shadow_state["selected_rooms"] = [r.model_dump() for r in top_allocation.rooms]

    # Recalculate Quote on shadow state
    shadow_state = await generate_quote(shadow_state)

    old_price: float | None = state.get("total_price")
    new_price: float | None = shadow_state.get("total_price")
    req_vals = state.get("requested_values") or {}
    new_rooms: list[dict] = shadow_state.get("selected_rooms") or []

    # ── Confirmation Gate ────────────────────────────────────────────────────
    if requires_confirmation(req_vals, state, old_price, new_price):
        # Park the fully-validated shadow state and route to lifecycle_handler
        summary = build_change_summary(
            prev_state=state,
            req_vals=req_vals,
            new_price=new_price,
            old_price=old_price,
            new_rooms=new_rooms,
        )
        logger.info(
            "Modification requires confirmation | mod_id=%s fields=%s",
            (state.get("modification_metadata") or {}).get("modification_id", ""),
            list(req_vals.keys()),
        )
        return cast(ConversationState, {
            **state,
            # Store the pre-validated state so lifecycle_handler can apply it atomically
            "modification_candidate_state": shadow_state,
            "lifecycle_action_pending": "booking_modification",
            "lifecycle_action_data": {
                "req_vals": req_vals,
                "old_price": old_price,
                "new_price": new_price,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "bot_response": templates.text_card(summary),
            "next_action": "lifecycle_awaiting_confirmation",
            "intent": "LIFECYCLE_RESET",
        })

    # ── Auto-Apply (minor changes: special_requests, guest contact only) ─────
    payment_clears = invalidate_payment_links(shadow_state)
    for k, v in payment_clears.items():
        shadow_state[k] = v

    # Emit audit log (fire-and-forget)
    try:
        await log_modification_audit(state, shadow_state, req_vals, old_price, new_price)
    except Exception:
        logger.warning("Modification audit log failed (non-fatal)", exc_info=True)

    # Clear modification state
    shadow_state["requested_values"] = None
    shadow_state["modified_fields"] = None
    shadow_state["booking_change_summary"] = None
    shadow_state["modification_metadata"] = None
    shadow_state["modification_candidate_state"] = None

    # Format final confirmation as receipt-style checklist
    confirmation_text = build_final_modification_confirmation(shadow_state)
    # Append price delta note for auto-applied minor changes with same price
    shadow_state["bot_response"] = templates.text_card(confirmation_text)
    shadow_state["intent"] = "BOOK_ROOM"

    return cast(ConversationState, shadow_state)



def _room_preview_media_cards(options: list, state: dict | None = None) -> list[dict]:
    """Helper to generate media cards for allocations, copied from room_searcher."""
    from app.services.hotel_media_service import MAX_IMAGES_PER_RESPONSE, media_delivery_card
    from app.services.room_service import _friendly_rate_plan_name

    cards: list[dict] = []
    seen_room_ids: set[str] = set()
    hotel_id = (state or {}).get("hotel_id")
    for option in options:
        for room in option.rooms:
            room_id = str(room.room_id or "").strip()
            room_name = str(room.room_name or "Room").strip()
            if not room_id or room_id in seen_room_ids:
                continue
            seen_room_ids.add(room_id)

            caption = room_name

            cards.append(media_delivery_card(
                hotel_id=hotel_id,
                entity_type="room",
                entity_id=room_id,
                tags=[room_name],
                aliases=room.aliases if hasattr(room, 'aliases') else [],
                caption=caption,
                limit=1,
                is_preview=True,
            ))
            if len(cards) >= MAX_IMAGES_PER_RESPONSE:
                return cards
    return cards
