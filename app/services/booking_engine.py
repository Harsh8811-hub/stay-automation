"""
Booking Engine — deterministic pricing calculations.

CRITICAL: No AI involved in pricing. All calculations are pure Python.
This ensures prices are always accurate and auditable.
"""

import logging
from datetime import datetime

from app.models.schemas import (
    BookingSession,
    PriceBreakdown,
    RoomLineItem,
    RoomType,
    SelectedRoom,
    TaxDetails,
)
from app.config.settings import get_settings
from app.repositories.hotel_repository import get_hotel_settings
from app.repositories.rate_threshold_repository import get_rate_threshold
from app.utils.rate_threshold import clamp_nightly_rate

logger = logging.getLogger(__name__)


def calculate_num_nights(check_in: str, check_out: str) -> int:
    """Calculate number of nights between check-in and check-out."""
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d")
        co = datetime.strptime(check_out, "%Y-%m-%d")
        delta = (co - ci).days
        return max(delta, 1)  # Minimum 1 night
    except (ValueError, TypeError):
        return 1


async def calculate_price_breakdown(
    selected_rooms: list[SelectedRoom],
    check_in: str,
    check_out: str,
    hotel_id: int,
    extra_services: list[str] | None = None  # List of selected upsell names
) -> PriceBreakdown:
    """
    Build a complete price breakdown.
    Calculates per-room pricing including extra guest charges.
    """
    num_nights = calculate_num_nights(check_in, check_out)
    
    # 1. Fetch settings
    settings_doc = await get_hotel_settings(hotel_id)
    
    # 2. Tax Type: Default to "inclusive" (GST included in catalog price) unless explicitly set to "exclusive"
    tax_config = None
    tax_type = "inclusive"
    taxes_applicable = True
    
    if settings_doc:
        settings_data = settings_doc.get("settings_data")
        if isinstance(settings_data, dict):
            tax_config = settings_data.get("taxConfig") or settings_data.get("tax_config")
        if not tax_config:
            tax_config = settings_doc.get("taxConfig") or settings_doc.get("tax_config")

        if isinstance(tax_config, dict):
            reg_type = str(tax_config.get("pricing_type") or tax_config.get("type") or "inclusive").lower()
            tax_type = "exclusive" if reg_type in ("exclusive", "exclude") else "inclusive"
        elif isinstance(tax_config, str):
            tax_type = "exclusive" if tax_config.lower() in ("exclusive", "exclude") else "inclusive"
            
    # 3. Seasonal multiplier — step 3 of the price-resolution ladder, written out
    #    in full at the top of inventory_service.py. It scales the nightly rate
    #    that steps 1-2 resolved, which is why it lives here and not among the
    #    rate rungs: it is a modifier on a price, not a source of one.
    #
    #    ACTIVE. Scales base rates by season_multiplier. It does NOT floor or cap
    #    anything itself — the 0.50 floor in season_repository bounds the
    #    multiplier, and the hotel's own rate threshold (below) bounds the
    #    result. An earlier version of this comment claimed otherwise.
    from app.repositories.season_repository import get_active_season_modifier
    season_multiplier, season_name = await get_active_season_modifier(hotel_id, check_in, check_out)

    # 3b. Rate threshold protection — the hotel's own floor and ceiling on a
    #     nightly rate. The reference is the PRE-season rate: the window exists
    #     to bound how far the season may drag a price, not to overrule the
    #     calendar. `inventory_service._apply_calendar_rates` applies the same
    #     clamp with the same reference, so the room card and this stay summary
    #     cannot disagree. See app/utils/rate_threshold.py.
    threshold = await get_rate_threshold(hotel_id)

    taxes_list = settings_doc.get("taxes") if settings_doc else None
    
    line_items: list[RoomLineItem] = []
    rooms_subtotal = 0.0
    rooms_tax = 0.0
    rooms_total = 0.0

    # Ladder steps 1-2: pull the calendar for the stay dates. Rung 3 (the
    # catalogue) is applied per night inside the room loop below. The order and
    # the reasoning behind it: see THE PRICE-RESOLUTION LADDER, inventory_service.py.
    from app.services.inventory_service import (
        _nights,
        average_nightly,
        calendar_rate_for_night,
        fetch_daily_rates,
    )
    nights = _nights(check_in, check_out)
    daily_rates_data = await fetch_daily_rates(hotel_id, nights) if nights else {"by_plan": {}, "by_type": {}}
    rates_by_plan = daily_rates_data.get("by_plan", {})
    rates_by_type = daily_rates_data.get("by_type", {})

    # Group identical rooms so multi-room bookings appear as a single line item with quantity=N
    from collections import defaultdict
    room_groups: dict[tuple, list[SelectedRoom]] = defaultdict(list)
    for room in selected_rooms:
        room_id = str(room.room_id)
        plan_id = str(room.rate_plan_id) if room.rate_plan_id else ""
        plan_name = str(room.rate_plan_name or "").upper()

        inferred_type = None
        if "CP" in plan_name or "BREAKFAST" in plan_name:
            inferred_type = "RT_CP"
        elif "MAP" in plan_name or "DINNER" in plan_name or "LUNCH" in plan_name:
            inferred_type = "RT_MAP"
        elif "EP" in plan_name or "ONLY" in plan_name:
            inferred_type = "RT_EP"

        if nights:
            resolved_rates: list[float] = []
            for dt in nights:
                # Shares `calendar_rate_for_night` with inventory_service so this
                # path cannot drift from the one that prices the room card. It
                # used to be a hand-copied second version without the
                # sibling-plan guard, which is why the stay summary could quote
                # a different number than the card it was summarising.
                night = calendar_rate_for_night(
                    rates_by_plan,
                    rates_by_type,
                    room_id,
                    plan_id,
                    dt,
                    (inferred_type,),
                )
                # No calendar row for this plan on this night: this room's own
                # price stands, which is how a stay straddling the calendar's
                # coverage blends the two rather than failing.
                raw_fallback = float(getattr(room, "raw_price_per_night", None) or room.price_per_night)
                resolved_rates.append(
                    night if night is not None else raw_fallback
                )
            effective_nightly_base = (
                average_nightly(resolved_rates) if resolved_rates else float(getattr(room, "raw_price_per_night", None) or room.price_per_night)
            )
        else:
            effective_nightly_base = float(getattr(room, "raw_price_per_night", None) or room.price_per_night)

        # Step 3: Seasonal multiplier. `pre_season_base` is kept as the reference
        # the threshold measures against, so the bound is the hotel's window on
        # the calendar rate rather than on the season's output.
        #
        # Caveat, deliberate: on a night with no calendar row the fallback above
        # is `room.price_per_night`, which for a card-built room has ALREADY been
        # season-scaled by inventory_service. On those nights this reference is
        # post-season, so the window is measured from a slightly high base. The
        # guest's price is unaffected (the season is applied once either way);
        # only the bound is loose on unlisted nights. Fixing it properly means
        # carrying a pre-season value on the room through the card, which is a
        # larger change than this feature.
        pre_season_base = effective_nightly_base
        if season_multiplier and season_multiplier != 1.0:
            effective_nightly_base = float(round((effective_nightly_base * season_multiplier) / 10) * 10)

        effective_nightly_base, was_clamped = clamp_nightly_rate(
            pre_season_base, effective_nightly_base, threshold
        )
        if was_clamped:
            logger.info(
                "RATE_THRESHOLD | hotel=%s room=%s base=%s -> %s (season x%s)",
                hotel_id, room.room_name, pre_season_base, effective_nightly_base,
                season_multiplier,
            )

        room.price_per_night = effective_nightly_base

        key = (
            room.room_id,
            room.room_name,
            room.rate_plan_id,
            room.rate_plan_name,
            effective_nightly_base,
            room.extra_adult_charge,
            room.extra_child_charge,
            room.currency,
        )
        room_groups[key].append(room)
    
    for key, group in room_groups.items():
        qty = len(group)
        room = group[0]
        adjusted_base = room.price_per_night
        nightly_base_rate = adjusted_base + room.extra_adult_charge + room.extra_child_charge
        room_rate = nightly_base_rate
        
        # Determine tax rate for this room
        if not taxes_applicable:
            tax_rate = 0.0
        else:
            # Find matching slab in taxes array
            matched_slab = None
            if taxes_list:
                for tax in taxes_list:
                    app_range = tax.get("applicableRange")
                    if not isinstance(app_range, dict):
                        continue
                    min_val = app_range.get("min")
                    max_val = app_range.get("max")
                    
                    min_f = float(min_val) if min_val is not None else None
                    max_f = float(max_val) if max_val is not None else None
                    
                    if min_f is not None and room_rate < min_f:
                        continue
                    if max_f is not None and room_rate > max_f:
                        continue
                        
                    matched_slab = tax
                    break
            
            if matched_slab:
                if matched_slab.get("active", False):
                    tax_rate = float(matched_slab.get("value", 0.0))
                else:
                    tax_rate = 0.0
            else:
                # Fall back to default room revenue slabs (5% < 7500, 18% >= 7500).
                #
                # The high rate is 18, not 12. It read 12 because 12 was the LOW
                # band until the September 2025 GST reform cut that band to 5 --
                # when `rate_low` moved, `rate_high` was left holding the retired
                # low-band number. Above Rs 7,500 a room has been taxed at 18
                # since 22 September 2025, and `hotel_config.py` has declared 18
                # in both of its copies throughout.
                #
                # This is not a rare branch. No hotel in the live `hotel_settings`
                # collection carries a `taxes` array, and none carries
                # `gstRateHigh`, so `matched_slab` above is always None and these
                # three constants price every quote the bot produces. Pricing is
                # inclusive, so the guest pays the same either way -- the rate
                # only decides how the invoice splits between taxable value and
                # tax. That is why a number that is too low under-remits GST and
                # overstates revenue rather than over-charging the guest.
                thresh = float((settings_doc or {}).get("gstThresholdInr") or (settings_doc or {}).get("gst_threshold_inr") or 7500.0)
                rate_low = float((settings_doc or {}).get("gstRateLow") or (settings_doc or {}).get("gst_rate_low") or 5.0)
                rate_high = float((settings_doc or {}).get("gstRateHigh") or (settings_doc or {}).get("gst_rate_high") or 18.0)
                if room_rate < thresh:
                    tax_rate = rate_low
                else:
                    tax_rate = rate_high

            # A blanket rate for every property, applied *after* the slab
            # resolution above so it wins over both a per-hotel `taxes` slab and
            # the built-in fallback. `None` restores the normal behaviour.
            # See GST_RATE_OVERRIDE_PERCENT in settings.py for why it exists.
            _gst_override = get_settings().GST_RATE_OVERRIDE_PERCENT
            if _gst_override is not None:
                tax_rate = float(_gst_override)

        # Calculate nightly and total room tax details
        if tax_type == "inclusive":
            nightly_final = float(round(room_rate / 10) * 10)
            if tax_rate > 0:
                nightly_base = float(round((nightly_final / (1 + tax_rate / 100)) / 10) * 10)
                nightly_tax = float(round((nightly_final - nightly_base) / 10) * 10)
            else:
                nightly_base = nightly_final
                nightly_tax = 0.0
        else:
            nightly_tax_exact = room_rate * (tax_rate / 100)
            nightly_tax = float(round(nightly_tax_exact / 10) * 10) if tax_rate > 0 else 0.0
            nightly_base = float(round(room_rate / 10) * 10)
            nightly_final = float(round((nightly_base + nightly_tax) / 10) * 10)
            
        base_amount = float(round((nightly_base * num_nights * qty) / 10) * 10)
        final_amount = float(round((nightly_final * num_nights * qty) / 10) * 10)
        tax_amount = float(round((final_amount - base_amount) / 10) * 10)
        
        tax_details = TaxDetails(
            base_amount=base_amount,
            tax_amount=tax_amount,
            tax_rate=tax_rate,
            tax_type=tax_type,
            final_amount=final_amount
        )

        line_items.append(RoomLineItem(
            room_name=room.room_name,
            rate_plan=room.rate_plan_name,
            quantity=qty,
            price_per_night=nightly_final,
            num_nights=num_nights,
            subtotal=base_amount,
            tax_details=tax_details,
            extra_adult_charge=float(round(room.extra_adult_charge / 10) * 10),
            extra_child_charge=float(round(room.extra_child_charge / 10) * 10),
            # Carried, not re-derived, so the printed quote can say "Extra
            # person" instead of claiming a mattress. See `_is_a_mattress`.
            extra_bed_count=room.extra_bed_count,
        ))
        
        rooms_subtotal += base_amount
        rooms_tax += tax_amount
        rooms_total += final_amount
        
    # 4. Upsell items calculation
    upsell_items_list = []
    upsells_subtotal = 0.0
    upsells_tax = 0.0
    upsells_total = 0.0
    
    if extra_services:
        from app.services.extra_services import extra_service_manager
        catalog = await extra_service_manager.get_services(hotel_id)
        for service_query in extra_services:
            detail = extra_service_manager.calculate_service_price_detail(
                service_query,
                catalog,
                tax_type=tax_type,
                taxes_applicable=taxes_applicable
            )
            if not detail:
                continue
            upsell_items_list.append(detail)
            upsells_subtotal += float(detail["base_amount"])
            upsells_tax += float(detail["tax_amount"])
            upsells_total += float(detail["final_amount"])

    overall_subtotal = float(round((rooms_subtotal + upsells_subtotal) / 10) * 10)
    raw_total = float(round((rooms_total + upsells_total) / 10) * 10)
    overall_total = raw_total
    overall_tax = float(round((overall_total - overall_subtotal) / 10) * 10)
    
    # Blended tax percentage
    # The 5.0 fallback is only reached on a zero subtotal, where no tax was
    # computed and the label just needs to agree with the blanket rate in
    # settings.py rather than the retired 12%.
    overall_tax_percent = round((overall_tax / overall_subtotal) * 100, 2) if overall_subtotal > 0 else 5.0

    return PriceBreakdown(
        rooms=line_items,
        upsells=upsell_items_list,
        subtotal=overall_subtotal,
        tax_percent=overall_tax_percent,
        tax_amount=overall_tax,
        total=overall_total,
        currency=selected_rooms[0].currency if selected_rooms else "INR",
        season_name=season_name,
        season_multiplier=season_multiplier,
    )


def _is_a_mattress(room: RoomLineItem) -> bool:
    """Is this room's extra charge for a bed, or for an unseated adult?

    `RoomLineItem.extra_bed_count` is the answer. `None` is "nobody said", and
    falls back to the reading this code used before the field existed — any
    charge above zero was a mattress — so a breakdown built before the
    extra-person rule prints exactly as it did.
    """
    if room.extra_bed_count is not None:
        return bool(room.extra_bed_count)
    return bool(room.extra_adult_charge and room.extra_adult_charge > 0)


def format_price_quote(
    breakdown: PriceBreakdown,
    check_in: str,
    check_out: str,
    adults: int,
    children: int = 0,
    option_label: str = "Selected Option",
    assumptions: list[str] | None = None,
    no_extra_bed_note: str | None = None,
) -> str:
    """
    Format the price breakdown as a clean, readable V3 quote with exact arithmetic & alignment.

    `no_extra_bed_note` is the caller's "No extra bed for N people" line, printed
    in the place the `➕ Extra bed` line would occupy had one been charged. The
    caller computes it because only it holds the room's occupancy and the party
    size -- `RoomLineItem` carries neither, and this function deliberately takes
    nothing but the finished breakdown so the numbers it prints always reconcile
    to the total it was handed. It is rendered only when no bed was charged, so
    the two lines can never both appear.
    """
    from app.prompts.templates import format_assumption_line
    from app.utils.helpers import format_quote_date_range

    room_names = ", ".join(r.room_name for r in breakdown.rooms) if breakdown.rooms else "Selected Room"
    num_nights = breakdown.rooms[0].num_nights if breakdown.rooms else 1
    dates_str = format_quote_date_range(check_in, check_out)

    night_str = f"{num_nights} night" if num_nights == 1 else f"{num_nights} nights"
    adult_str = f"{adults} adult" if adults == 1 else f"{adults} adults"
    guest_str = adult_str
    if children and children > 0:
        ch_str = f"{children} child" if children == 1 else f"{children} children"
        guest_str += f", {ch_str}"

    opt_label_clean = str(option_label or "").strip()
    if opt_label_clean and opt_label_clean.lower() != "none" and opt_label_clean.lower() != "selected option":
        header_title = f"🧾 *Booking Summary — {opt_label_clean} ({room_names})*"
    else:
        header_title = f"🧾 *Booking Summary — {room_names}*"

    lines: list[str] = [
        header_title,
        f"📅 *{dates_str}* ({night_str}) | 👥 *{guest_str}*\n",
    ]

    total_extra_bed_qty = 0
    total_extra_bed_cost = 0.0
    # The same money as a bed, but a different sentence: on a CP / MAP plan an
    # adult above base occupancy is charged whether or not a mattress was
    # offered. Kept apart so the quote cannot tell a guest they bought a bed
    # they never got.
    total_extra_person_qty = 0
    total_extra_person_cost = 0.0
    accumulated_room_sub = 0.0

    tax_rate = breakdown.tax_percent or 5.0
    is_inclusive = any(r.tax_details and r.tax_details.tax_type == "inclusive" for r in breakdown.rooms)

    for r in breakdown.rooms:
        qty = r.quantity
        if is_inclusive:
            gross_bed = (r.extra_adult_charge or 0.0) * r.num_nights * qty
            base_bed = float(round((gross_bed / (1 + tax_rate / 100.0)) / 10) * 10) if tax_rate > 0 else gross_bed
            raw_room_sub = max(0.0, r.subtotal - base_bed)
            base_nightly_rate = float(round((raw_room_sub / max(1, r.num_nights * qty)) / 10) * 10)
            base_room_sub = base_nightly_rate * r.num_nights * qty
            accumulated_room_sub += base_room_sub

            if qty > 1:
                calc_str = f"Rs {base_nightly_rate:,.0f} x {night_str} ({qty} rooms)"
            else:
                calc_str = f"Rs {base_nightly_rate:,.0f} x {night_str}"
            room_amt_str = f"Rs {base_room_sub:>5,.0f}"
            lines.append(f"{'🛏️ Room rate':<14}{calc_str:<25} {room_amt_str}")

            if r.extra_adult_charge and r.extra_adult_charge > 0:
                if _is_a_mattress(r):
                    total_extra_bed_cost += base_bed
                    total_extra_bed_qty += qty
                else:
                    total_extra_person_cost += base_bed
                    total_extra_person_qty += qty
        else:
            base_nightly_rate = float(round((r.subtotal / max(1, r.num_nights * qty)) / 10) * 10)
            base_room_sub = base_nightly_rate * r.num_nights * qty
            accumulated_room_sub += base_room_sub

            if qty > 1:
                calc_str = f"Rs {base_nightly_rate:,.0f} x {night_str} ({qty} rooms)"
            else:
                calc_str = f"Rs {base_nightly_rate:,.0f} x {night_str}"
            room_amt_str = f"Rs {base_room_sub:>5,.0f}"
            lines.append(f"{'🛏️ Room rate':<14}{calc_str:<25} {room_amt_str}")

            if r.extra_adult_charge and r.extra_adult_charge > 0:
                extra_c = float(round((r.extra_adult_charge * r.num_nights * qty) / 10) * 10)
                if _is_a_mattress(r):
                    total_extra_bed_cost += extra_c
                    total_extra_bed_qty += qty
                else:
                    total_extra_person_cost += extra_c
                    total_extra_person_qty += qty

    if total_extra_bed_cost > 0:
        per_bed = float(round((total_extra_bed_cost / max(1, total_extra_bed_qty * num_nights)) / 10) * 10)
        total_extra_bed_cost = per_bed * total_extra_bed_qty * num_nights
        bed_calc = f"{total_extra_bed_qty} x Rs {per_bed:,.0f}" if num_nights == 1 else f"{total_extra_bed_qty} x Rs {per_bed:,.0f} x {night_str}"
        bed_amt_str = f"Rs {total_extra_bed_cost:>5,.0f}"
        lines.append(f"{'➕ Extra bed':<14}{bed_calc:<25} {bed_amt_str}")
    elif no_extra_bed_note:
        # No mattress was charged. If the caller worked out that the party is
        # being seated without one, say so here rather than leaving the guest to
        # infer it from a bed line's absence -- a missing line reads as an
        # oversight, or as a charge still to come at check-in.
        lines.append(no_extra_bed_note)

    if total_extra_person_cost > 0:
        # The extra-person charge, said in its own words. This is the CP / MAP
        # case: the plan the guest bought does not cover an adult above the
        # room's base occupancy, so the rate is owed -- but no mattress was
        # offered, and a quote that said "Extra bed" here would be selling them
        # something they never got.
        per_person = float(
            round((total_extra_person_cost / max(1, total_extra_person_qty * num_nights)) / 10) * 10
        )
        total_extra_person_cost = per_person * total_extra_person_qty * num_nights
        person_calc = (
            f"{total_extra_person_qty} x Rs {per_person:,.0f}"
            if num_nights == 1
            else f"{total_extra_person_qty} x Rs {per_person:,.0f} x {night_str}"
        )
        lines.append(
            f"{'👤 Extra person':<14}{person_calc:<25} {f'Rs {total_extra_person_cost:>5,.0f}'}"
        )

    # Separator
    lines.append(f"{' ' * 40}---------")
    computed_subtotal = accumulated_room_sub + total_extra_bed_cost + total_extra_person_cost
    lines.append(f"{'Subtotal':<40}Rs {computed_subtotal:>5,.0f}")

    disc_val = float(breakdown.discount_amount or 0.0)
    if disc_val > 0:
        disc_reason = breakdown.discount_reason or "Promotional Discount"
        lines.append(f"{'🎉 ' + disc_reason:<40}-Rs {disc_val:>5,.0f}")

    computed_taxable = max(0.0, computed_subtotal - disc_val)
    computed_tax = float(round((computed_taxable * (tax_rate / 100.0)) / 10) * 10) if tax_rate > 0 else 0.0
    gst_label = f"GST @ {tax_rate:.0f}%" if tax_rate > 0 else "GST (Exempt)"
    lines.append(f"{'🏛️ ' + gst_label:<40}Rs {computed_tax:>5,.0f}")

    # Paid early check-in / late check-out. Printed here, after GST and before
    # the total, because it is added straight to the total and never to the
    # taxable subtotal: the room rate it was derived from already includes GST.
    # Each fee carries its own disclaimer so the number can never be shown
    # without the conditions attached to it.
    fee_total = 0.0
    for fee in breakdown.early_late_fees or []:
        amount = float(fee.get("amount") or 0.0)
        if amount <= 0:
            continue
        fee_total += amount
        lines.append(f"{'⏰ ' + (fee.get('label') or 'Early check-in / late check-out'):<40}"
                     f"{'+Rs ' + format(amount, ',.0f'):>9}")
        if fee.get("disclaimer"):
            lines.append(fee["disclaimer"])

    adj_val = breakdown.adjustment_amount or 0.0
    if adj_val:
        adj_reason = breakdown.adjustment_reason or "Adjustment"
        if adj_val < 0:
            adj_label = f"🎁 {adj_reason}"
            adj_sign = "-"
        else:
            adj_label = adj_reason
            adj_sign = "+"
        lines.append(f"{adj_label:<40}{adj_sign}Rs {abs(adj_val):>5,.0f}")

    lines.append(f"{' ' * 40}=========")
    final_total = computed_taxable + computed_tax + fee_total + adj_val
    final_total = max(0.0, float(round(final_total / 10) * 10))
    lines.append(f"💰 *Total Amount{' ' * 24}Rs {final_total:>5,.0f}*\n")
    lines.append("👉 Reply *confirm* to proceed with your booking.\n")
    lines.append(format_assumption_line(assumptions))

    return "\n".join(lines)


def apply_staff_total(
    breakdown: PriceBreakdown,
    payable: float,
    *,
    reason: str | None = None,
) -> PriceBreakdown:
    """
    Overlay a staff-negotiated total onto a computed breakdown.

    Line items, nights, quantities, subtotal and GST all keep their computed
    values. Only `total` is replaced, and the signed difference is recorded on
    `adjustment_amount` so the printed summary still reconciles:

        subtotal + tax_amount + adjustment_amount == total

    `payable` is treated as all-in (GST included), matching the
    "_Rates final · GST included_" contract on every quote the bot sends.

    Mutates and returns the same object, in step with DiscountEngine.apply_discount
    so callers can treat both the same way.
    """
    payable = float(payable)
    computed = float(breakdown.total or 0.0)
    diff = float(round(payable - computed, 2))
    breakdown.adjustment_amount = diff
    if reason:
        breakdown.adjustment_reason = reason
    elif diff < 0:
        breakdown.adjustment_reason = "Special rate"
    elif diff > 0:
        breakdown.adjustment_reason = "Rate adjustment"
    else:
        breakdown.adjustment_reason = None
    breakdown.total = payable
    return breakdown


def apply_early_late_fee(breakdown: PriceBreakdown, fees: list[dict] | None) -> PriceBreakdown:
    """
    Attach one-off early check-in / late check-out fees and fold them into
    `total`.

    Mirrors exactly what `format_price_quote` adds to the printed chain, so for
    a fee-bearing quote:

        subtotal + tax_amount + fee_total + adjustment_amount == total

    The fees are added to `total` alone — `subtotal` and `tax_amount` keep their
    computed values, because the room rate a percentage was taken of already
    includes GST.

    **Call this after the coupon and before `apply_staff_total`.** A coupon
    recomputes `total` from `subtotal` and would drop the fee; a staff total
    replaces `total` outright and must be the last word, so it has to see the
    fee already in place for its adjustment to reconcile.

    Mutates and returns the same object, in step with `apply_staff_total` and
    `DiscountEngine.apply_discount`.
    """
    if not fees:
        return breakdown
    breakdown.early_late_fees = list(fees)
    breakdown.total = float(
        breakdown.total + sum(float(f.get("amount") or 0.0) for f in fees)
    )
    return breakdown


def generate_booking_id() -> str:
    """Generate a human-readable, unique booking ID."""
    import secrets
    date_part = datetime.utcnow().strftime("%Y%m%d")
    random_part = secrets.token_hex(4).upper()  # 8 hex chars = 4 billion possibilities
    return f"BK-{date_part}-{random_part}"


def generate_payment_reference() -> str:
    """
    The key for a payment that has no booking id yet.

    Deliberately not a `BK-…` string: every reader that treats a `BK-` id as a
    reservation (CRM lists, staff hand-off, the guest's own copy) must not be
    able to pick this up by accident. It keys the gateway order and the
    inventory hold only.
    """
    import secrets
    date_part = datetime.utcnow().strftime("%Y%m%d")
    return f"PAY-{date_part}-{secrets.token_hex(4).upper()}"


def generate_mock_payment_link(booking_id: str, amount: float) -> str:
    """
    Generate a mock payment link.
    In production, replace with Razorpay/Stripe integration.
    """
    return f"https://pay.stayinn.com/{booking_id}?amount={amount:.0f}"


def build_booking_document(session: BookingSession) -> dict:
    """Build a booking document from the finalized session."""
    return {
        "booking_id": session.booking_id,
        "hotel_id": session.hotel_id,
        "session_id": session.session_id,
        "customer_phone": session.customer_identifier,
        "guest_name": session.guest_name,
        "guest_email": session.guest_email,
        "adults": session.adults,
        "children": session.children or 0,
        "num_rooms": session.num_rooms,
        "check_in": session.check_in,
        "check_out": session.check_out,
        "rooms": [r.model_dump() for r in session.selected_rooms],
        "num_nights": calculate_num_nights(session.check_in, session.check_out),
        "price_breakdown": session.price_breakdown.model_dump() if session.price_breakdown else None,
        "total_price": session.total_price,
        "currency": "INR",
        "payment_status": session.payment_status or "pending",
        "payment_link": session.payment_link,
        "booking_status": "confirmed",
    }
