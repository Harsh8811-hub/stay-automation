"""
Discount Engine — validates and applies coupons and other discounts deterministically.

Centralized discount handling ensures coupons do not stack improperly with 
other discounts (e.g. price match) and always follow deterministic business rules.
"""

import logging
import re
from datetime import date, datetime, timezone
from app.models.schemas import Coupon, PriceBreakdown
from app.db.mongodb import coupons_collection

logger = logging.getLogger(__name__)

class DiscountEngine:
    @staticmethod
    async def get_coupon(coupon_code: str, hotel_id: int | str) -> Coupon | None:
        """Fetch a coupon or promotion by code and hotel_id."""
        if not coupon_code or not str(coupon_code).strip():
            return None
        coll = coupons_collection()
        # `re.escape`: the code is whatever the guest typed, and this is a regex.
        # Unescaped, ".*" matched every promotion the hotel had — the widened
        # `$or` below reaches `coupon_name`/`name`, where promotions actually
        # keep their code, so a guest who was never offered a code was handed a
        # 10-15% discount. Escaping also means a code containing "." or "+" only
        # ever matches itself, which is what a coupon lookup should do.
        code_regex = {"$regex": f"^{re.escape(coupon_code.strip())}$", "$options": "i"}
        variants = [hotel_id, str(hotel_id)]
        if str(hotel_id).isdigit():
            variants.append(int(hotel_id))

        doc = await coll.find_one({
            "$or": [
                {"coupon_code": code_regex},
                {"coupon_name": code_regex},
                {"name": code_regex},
                {"title": code_regex},
            ],
            "hotel_id": {"$in": variants}
        })
        if not doc:
            return None

        return DiscountEngine._doc_to_coupon(doc, hotel_id, default_code=coupon_code)

    @staticmethod
    def _doc_to_coupon(doc: dict, hotel_id: int | str, default_code: str = "PROMO") -> Coupon:
        """Normalize document from promotions collection into Coupon schema."""
        c_code = (
            doc.get("coupon_code")
            or doc.get("coupon_name")
            or doc.get("name")
            or doc.get("title")
            or default_code
        )
        d_val = float(doc.get("discount_value") or doc.get("discount") or doc.get("value") or 0.0)
        raw_type = doc.get("discount_type") or doc.get("type")
        if raw_type:
            raw_str = str(raw_type).strip().lower()
            if raw_str in ("percentage", "percent", "%"):
                d_type = "percentage"
            elif raw_str in ("fixed", "flat", "amount", "inr"):
                d_type = "fixed"
            else:
                d_type = raw_str
        else:
            d_type = "fixed" if d_val > 100 else "percentage"

        is_act = doc.get("active")
        if is_act is None:
            is_act = doc.get("is_active") if doc.get("is_active") is not None else (doc.get("status") == "ACTIVE")

        v_from = doc.get("valid_from")
        if not v_from and doc.get("start_date"):
            try:
                v_from = datetime.fromisoformat(str(doc["start_date"]))
            except Exception:
                pass
        if not v_from and isinstance(doc.get("booking_dates"), dict):
            b_s = doc["booking_dates"].get("start")
            if b_s:
                try:
                    v_from = datetime.fromisoformat(str(b_s))
                except Exception:
                    pass

        v_until = doc.get("valid_until")
        if not v_until and doc.get("end_date"):
            try:
                v_until = datetime.fromisoformat(str(doc["end_date"]))
                if v_until.time() == datetime.min.time():
                    v_until = v_until.replace(hour=23, minute=59, second=59)
            except Exception:
                pass
        if not v_until and isinstance(doc.get("booking_dates"), dict):
            b_e = doc["booking_dates"].get("end")
            if b_e:
                try:
                    v_until = datetime.fromisoformat(str(b_e))
                    if v_until.time() == datetime.min.time():
                        v_until = v_until.replace(hour=23, minute=59, second=59)
                except Exception:
                    pass

        s_from = None
        s_until = None
        stay_dates = doc.get("stay_dates")
        if isinstance(stay_dates, dict):
            if stay_dates.get("start"):
                try:
                    s_from = datetime.fromisoformat(str(stay_dates["start"]))
                except Exception:
                    pass
            if stay_dates.get("end"):
                try:
                    s_until = datetime.fromisoformat(str(stay_dates["end"]))
                    if s_until.time() == datetime.min.time():
                        s_until = s_until.replace(hour=23, minute=59, second=59)
                except Exception:
                    pass

        adv = doc.get("advanced_settings") or {}
        min_nights = (
            adv.get("minNights")
            or adv.get("min_nights")
            or adv.get("minimumStay")
            or adv.get("lengthOfStay")
            or doc.get("min_nights")
        )
        if not min_nights:
            m_stay = re.search(r'(\d+)\s*nights?', str(doc.get("name") or doc.get("title") or ""), re.IGNORECASE)
            if m_stay:
                min_nights = int(m_stay.group(1))
        if min_nights:
            try:
                min_nights = int(min_nights)
            except Exception:
                min_nights = None

        promo_type = str(doc.get("promotion_type") or doc.get("promotion_kind") or "").strip().lower()
        promo_name = str(doc.get("name") or doc.get("title") or "").strip()
        base_disc = adv.get("baseDiscount")
        addon_disc = adv.get("addOnDiscount")

        return Coupon(
            coupon_code=str(c_code),
            hotel_id=hotel_id,
            active=bool(is_act),
            discount_type=str(d_type),
            discount_value=d_val,
            min_booking_value=doc.get("min_booking_value"),
            max_discount_amount=doc.get("max_discount_amount"),
            min_nights=min_nights,
            applicable_room_types=doc.get("applicable_room_types"),
            valid_from=v_from,
            valid_until=v_until,
            stay_start=s_from,
            stay_end=s_until,
            usage_limit=doc.get("usage_limit"),
            times_used=doc.get("times_used", 0),
            promotion_type=promo_type or None,
            promotion_name=promo_name or None,
            base_discount=float(base_disc) if base_disc is not None else None,
            add_on_discount=float(addon_disc) if addon_disc is not None else None,
        )

    @staticmethod
    async def get_active_default_promotion(
        hotel_id: int | str,
        check_in_date: str | None = None,
        check_out_date: str | None = None,
        num_nights: int | None = None,
    ) -> Coupon | None:
        """
        Fetch the active hotel-wide promotion that applies by default (e.g. basic-promotion-10).
        These promotions do not require a guest to enter a promo code.
        """
        coll = coupons_collection()
        str_hid = str(hotel_id)
        variants = [str_hid, int(str_hid)] if str_hid.isdigit() else [str_hid]

        try:
            cursor = coll.find({
                "hotel_id": {"$in": variants},
                "$or": [
                    {"is_active": True},
                    {"active": True},
                    {"status": "ACTIVE"},
                ],
            })
            docs = await cursor.to_list(length=50)
        except Exception as e:
            logger.warning(f"Error fetching active default promotions for hotel {hotel_id}: {e}")
            return None

        if not docs:
            return None

        if num_nights is None and check_in_date and check_out_date:
            try:
                ci = datetime.strptime(str(check_in_date).strip().split("T")[0].split(" ")[0], "%Y-%m-%d").date()
                co = datetime.strptime(str(check_out_date).strip().split("T")[0].split(" ")[0], "%Y-%m-%d").date()
                num_nights = max(1, (co - ci).days)
            except Exception:
                num_nights = 1
        stay_duration = num_nights if num_nights is not None else 1

        # In case promotions collection stores array per hotel document:
        flattened_docs = []
        for d in docs:
            if isinstance(d.get("promotions"), list):
                for p in d["promotions"]:
                    if isinstance(p, dict):
                        p_copy = dict(p)
                        p_copy.setdefault("hotel_id", d.get("hotel_id"))
                        flattened_docs.append(p_copy)
            else:
                flattened_docs.append(d)
        docs = flattened_docs

        now_utc = datetime.now(timezone.utc)
        def _is_doc_active(d: dict) -> bool:
            is_act = d.get("is_active")
            if is_act is None:
                is_act = d.get("active") if d.get("active") is not None else (str(d.get("status") or "").upper() == "ACTIVE")
            if not is_act:
                return False
            if d.get("end_date"):
                try:
                    ed = datetime.fromisoformat(str(d["end_date"])).replace(tzinfo=timezone.utc)
                    if now_utc > ed:
                        return False
                except Exception:
                    pass
            return True

        # Check if an active basic promotion actually exists for this hotel
        has_active_basic = any(
            (str(d.get("promotion_type") or d.get("type") or "").strip().lower() == "basic-promotion"
             or "basic promotion" in str(d.get("name") or d.get("title") or "").lower()
             or "basic-promotion" in str(d.get("coupon_name") or "").lower())
            and _is_doc_active(d)
            for d in docs
        )

        candidates: list[Coupon] = []
        for doc in docs:
            c_name = str(doc.get("coupon_name") or "").strip()
            promo_type = str(doc.get("promotion_type") or doc.get("type") or "").strip().lower()
            adv = doc.get("advanced_settings") or {}
            apply_all = adv.get("applyAll") is True

            # If coupon_name is non-empty, and applyAll is not True, and not basic-promotion,
            # this is a code-gated coupon requiring explicit guest entry
            if c_name and not apply_all and promo_type != "basic-promotion":
                continue

            coupon = DiscountEngine._doc_to_coupon(doc, hotel_id, default_code="PROMO")
            if not coupon.active or coupon.discount_value <= 0:
                continue

            # Per product guidance: If there is no active basic promotion, do not assume a base discount.
            # Long stay, early bird, and last minute discounts must be pure add-on value (e.g. pure 7%).
            is_tier1_addon = any(k in promo_type or k in str(doc.get("name") or "").lower() for k in ["long-stay", "early-bird", "last-minute"])
            if is_tier1_addon and not has_active_basic:
                if coupon.add_on_discount is not None and coupon.add_on_discount > 0:
                    coupon.discount_value = coupon.add_on_discount
                else:
                    m_add = re.search(r'\+(\d+(?:\.\d+)?)%', str(doc.get("name") or doc.get("title") or ""))
                    if m_add:
                        coupon.discount_value = float(m_add.group(1))
                    elif adv.get("baseDiscount") and coupon.discount_value > float(adv.get("baseDiscount")):
                        coupon.discount_value = max(0.0, coupon.discount_value - float(adv.get("baseDiscount")))
                coupon.base_discount = None

            now = datetime.now(timezone.utc)
            if coupon.valid_until:
                vu = coupon.valid_until if coupon.valid_until.tzinfo else coupon.valid_until.replace(tzinfo=timezone.utc)
                if now > vu:
                    continue
            if coupon.valid_from:
                vf = coupon.valid_from if coupon.valid_from.tzinfo else coupon.valid_from.replace(tzinfo=timezone.utc)
                if now < vf:
                    continue

            if check_in_date:
                try:
                    ci_clean = str(check_in_date).strip().split("T")[0].split(" ")[0]
                    ci_date = datetime.strptime(ci_clean, "%Y-%m-%d").date()
                    if coupon.stay_start and ci_date < coupon.stay_start.date():
                        continue
                    if coupon.stay_end and ci_date > coupon.stay_end.date():
                        continue
                except Exception:
                    pass

            # If last-minute promotion, only apply if booking is within bookablePeriod days of check-in
            if "last-minute" in promo_type or "last minute" in str(doc.get("name") or "").lower():
                if check_in_date:
                    try:
                        ci_clean = str(check_in_date).strip().split("T")[0].split(" ")[0]
                        ci_dt = datetime.strptime(ci_clean, "%Y-%m-%d").date()
                        today = date.today()
                        diff_days = (ci_dt - today).days
                        bookable_period = int(adv.get("bookablePeriod") or 1)
                        m_days = re.search(r'(\d+)\s*days?', str(doc.get("name") or ""), re.IGNORECASE)
                        if m_days:
                            bookable_period = max(bookable_period, int(m_days.group(1)))
                        if diff_days < 0 or diff_days > bookable_period:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            # If early-bird promotion, only apply if booking is well in advance
            if "early-bird" in promo_type or "early bird" in str(doc.get("name") or "").lower():
                if check_in_date:
                    try:
                        ci_clean = str(check_in_date).strip().split("T")[0].split(" ")[0]
                        ci_dt = datetime.strptime(ci_clean, "%Y-%m-%d").date()
                        today = date.today()
                        diff_days = (ci_dt - today).days
                        min_advance = int(adv.get("advancePeriod") or 14)
                        if diff_days < min_advance:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            # If long-stay promotion, only apply if stay duration meets min_nights
            if "long-stay" in promo_type or "long stay" in str(doc.get("name") or doc.get("title") or "").lower():
                min_n = coupon.min_nights or int(adv.get("minNights") or adv.get("minimumStay") or 3)
                if stay_duration < min_n:
                    continue

            candidates.append(coupon)

        if not candidates:
            return None

        # Prefer percentage discounts, higher discount value, and specific promotions over generic fallback
        def _candidate_rank(c: Coupon) -> tuple:
            p_type = (c.promotion_type or "").lower()
            p_name = (c.promotion_name or c.coupon_code or "").lower()
            is_specific = any(k in p_type or k in p_name for k in ["long-stay", "early-bird", "last-minute"])
            return (
                1 if c.discount_type == "percentage" else 0,
                c.discount_value,
                1 if is_specific else 0,
            )

        candidates.sort(key=_candidate_rank, reverse=True)
        return candidates[0]

    @staticmethod
    def validate_coupon(coupon: Coupon, breakdown: PriceBreakdown, check_in_date: str) -> tuple[bool, str]:
        """
        Validate if the coupon can be applied to the current booking.
        Returns (is_valid, reason_if_invalid).
        """
        if not coupon.active:
            return False, "This coupon is no longer active."

        # Check dates
        now = datetime.now(timezone.utc)
        if coupon.valid_from:
            vf = coupon.valid_from if coupon.valid_from.tzinfo else coupon.valid_from.replace(tzinfo=timezone.utc)
            if now < vf:
                return False, "This coupon is not valid yet."
        if coupon.valid_until:
            vu = coupon.valid_until if coupon.valid_until.tzinfo else coupon.valid_until.replace(tzinfo=timezone.utc)
            if now > vu:
                return False, "This coupon has expired."

        # Check stay dates
        if check_in_date:
            try:
                ci_clean = str(check_in_date).strip().split("T")[0].split(" ")[0]
                ci_date = datetime.strptime(ci_clean, "%Y-%m-%d").date()
                if coupon.stay_start and ci_date < coupon.stay_start.date():
                    return False, f"This coupon is only valid for stays starting {coupon.stay_start.strftime('%d %b %Y')} or later."
                if coupon.stay_end and ci_date > coupon.stay_end.date():
                    return False, f"This coupon is only valid for stays up to {coupon.stay_end.strftime('%d %b %Y')}."
            except Exception:
                pass

        # Check usage limits
        if coupon.usage_limit is not None and coupon.times_used >= coupon.usage_limit:
            return False, "This coupon has reached its usage limit."

        # Check minimum booking value
        if coupon.min_booking_value and breakdown.subtotal < coupon.min_booking_value:
            return False, f"This coupon requires a minimum booking value of ₹{coupon.min_booking_value:,.2f}."

        # Check minimum nights for long-stay promotion
        if coupon.min_nights:
            stay_nights = breakdown.rooms[0].num_nights if (breakdown and breakdown.rooms) else 1
            if stay_nights < coupon.min_nights:
                return False, f"This promotion requires a minimum stay of {coupon.min_nights} nights."

        # Check room type restrictions
        if coupon.applicable_room_types:
            booked_room_types = [room.room_name for room in breakdown.rooms]
            if not any(rt in coupon.applicable_room_types for rt in booked_room_types):
                return False, "This coupon is not applicable for the selected room type."

        return True, "Valid"

    @staticmethod
    def apply_discount(breakdown: PriceBreakdown, coupon: Coupon) -> PriceBreakdown:
        """
        Apply the coupon discount to the PriceBreakdown.
        Note: Modifies the breakdown in place and returns it.
        """
        # Calculate discount amount based on subtotal (before tax)
        if coupon.discount_type == "percentage":
            discount = breakdown.subtotal * (coupon.discount_value / 100.0)
        else:
            discount = coupon.discount_value

        if coupon.max_discount_amount:
            discount = min(discount, coupon.max_discount_amount)
            
        # Ensure discount doesn't exceed subtotal
        discount = min(discount, breakdown.subtotal)

        # Update breakdown
        breakdown.discount_amount = round(discount, 2)
        label = (
            f"{coupon.discount_value:.0f}% OFF"
            if coupon.discount_type == "percentage"
            else f"₹{coupon.discount_value:,.0f} OFF"
        )
        p_type = (coupon.promotion_type or "").lower()
        p_name = (coupon.promotion_name or coupon.coupon_code or "").lower()

        is_early_bird = "early-bird" in p_type or "early bird" in p_name
        is_long_stay = "long-stay" in p_type or "long stay" in p_name
        is_last_minute = "last-minute" in p_type or "last minute" in p_name
        is_basic = "basic-promotion" in p_type or "basic promotion" in p_type or "basic-promotion" in coupon.coupon_code.lower()

        if is_early_bird:
            if coupon.base_discount and coupon.add_on_discount:
                breakdown.discount_reason = (
                    f"Early Bird Deal ({label}: {coupon.base_discount:.0f}% Base + {coupon.add_on_discount:.0f}% Early Bird)"
                )
            else:
                breakdown.discount_reason = f"Early Bird Special ({label})"
        elif is_long_stay:
            if coupon.base_discount and coupon.add_on_discount:
                breakdown.discount_reason = (
                    f"Long Stay Deal ({label}: {coupon.base_discount:.0f}% Base + {coupon.add_on_discount:.0f}% Long Stay)"
                )
            else:
                breakdown.discount_reason = f"Long Stay Offer ({label})"
        elif is_last_minute:
            if coupon.base_discount and coupon.add_on_discount:
                breakdown.discount_reason = (
                    f"Last Minute Deal ({label}: {coupon.base_discount:.0f}% Base + {coupon.add_on_discount:.0f}% Last Minute)"
                )
            else:
                breakdown.discount_reason = f"Last Minute Deal ({label})"
        elif is_basic:
            if coupon.promotion_name and not any(k in coupon.promotion_name.lower() for k in ["basic-promotion", "basic_promotion"]):
                breakdown.discount_reason = f"{coupon.promotion_name} ({label})"
            else:
                breakdown.discount_reason = f"Special Offer ({label})"
        elif "coupon" in coupon.coupon_code.lower():
            breakdown.discount_reason = f"Coupon: {coupon.coupon_code} ({label})"
        else:
            breakdown.discount_reason = f"{coupon.coupon_code} ({label})"
        breakdown.coupon_code = coupon.coupon_code

        # Recalculate tax and total
        discounted_subtotal = breakdown.subtotal - breakdown.discount_amount
        # Assuming tax is proportional to discounted subtotal
        breakdown.tax_amount = round(discounted_subtotal * (breakdown.tax_percent / 100.0), 2)
        breakdown.total = float(round(round(discounted_subtotal + breakdown.tax_amount, 2) / 10) * 10)

        return breakdown
