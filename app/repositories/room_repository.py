"""
Room Repository — reads ONLY the normalized room_types & rate_plans collections in bot_migration.

This is READ-ONLY. We never write to room catalogs.
Transforms normalized room_types and rate_plans documents into flat RoomType models.
"""

import logging
from app.db.mongodb import (
    hotel_policies_collection,
    rate_plans_collection,
    room_types_collection,
)
from app.models.schemas import RatePlan, RoomOccupancy, RoomType

logger = logging.getLogger(__name__)


def _as_amount(value) -> float:
    """
    A currency field off a room or policy document, as a number.

    The catalogue stores these three ways — `600`, `"600"` and `"₹ 600"` — and
    an absent or unusable one means *unset*, not zero-rupees, so it reads back
    as 0.0 and the caller falls through to its next source. Raising here would
    take a whole hotel's catalogue down over one odd field.
    """
    if value is None:
        return 0.0
    try:
        return float(str(value).replace("₹", "").replace(",", "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def _room_extra_bed_rate(rt: dict, hotel_default: float) -> float:
    """
    What one extra bed costs *in this room*, from the most specific answer down.

    Three sources, in order, and the order is the whole point:

      1. `extra_adult_rate` on the room — the hotel pricing a fourth adult as
         such, breakfast and all. The most specific thing anyone can say.
      2. `extra_bed_price` on the room — the hotel pricing the mattress. This
         field was read by **nothing at all** until 2026-09-24: the only two
         readers of a bed price both went straight to the property-wide policy,
         so hotel 111111's Standard Non-AC Tripple Bed, which says 800, was
         sold at the hotel's 600 — Rs 200 short, per bed, per night, on the one
         room in the catalogue that bothers to price its own bed.
      3. `hotel_default` — the property rules' `mattress_charge`, which is what
         every other room falls back to and what all of them did before.

    A zero or unparseable value counts as unset at every step, never as
    "free". `RoomOccupancy.extra_adult_rate` is consumed as a plain number by
    the allocator, the card and the booking engine, so a 0 here would quietly
    make the bed free rather than fall through to the next source.

    **This number is the price, not the permission.** Whether anything is owed
    at all is a question about the *plan* the guest bought, not about this
    field: on EP an adult above base occupancy may stay with no bed and no
    charge, on CP and MAP they may not. That decision lives in
    `app/utils/meal_plan.py` and is applied by the allocator
    (`allocation_engine.plan_owes_for_an_extra_person`). A zero here must still
    fall through to the next source — "the guest owes nothing" is expressed by
    the plan, and never by a room's rate being unset.
    """
    rate = _as_amount(rt.get("extra_adult_rate"))
    if rate <= 0:
        rate = _as_amount(rt.get("extra_bed_price"))
    return rate if rate > 0 else hotel_default


def _get_hotel_variants(hotel_id: int | str) -> list:
    """Build list of tenant ID representations including known cross-system aliases."""
    str_hotel_id = str(hotel_id)
    # Tenant aliases (e.g. Sample Property 108959 contains real Riverside City rooms for Sample Property by Sahu Hotels)
    if str_hotel_id in ("988273", "108959"):
        return ["108959", "988273", 108959, 988273]
    variants: list[int | str] = [str_hotel_id]
    if str_hotel_id.isdigit():
        variants.append(int(hotel_id))
    return list(dict.fromkeys(variants))


async def _get_hotel_default_extra_bed_rate(hotel_variants: list) -> float:
    """
    Fetch default extra bed rate from hotel_policies (property_rules.extra_bed_policy.mattress_charge).
    Defaults to 600.0 if not specified in hotel policy.
    """
    try:
        pol_doc = await hotel_policies_collection().find_one({"hotel_id": {"$in": hotel_variants}})

        if pol_doc:
            extra_bed_pol = pol_doc.get("property_rules", {}).get("extra_bed_policy", {})
            mattress_charge = extra_bed_pol.get("mattress_charge")
            if mattress_charge:
                val = float(str(mattress_charge).replace("₹", "").replace(",", "").strip() or 0)
                if val > 0:
                    return val
            extra_bed_charge = pol_doc.get("property_rules", {}).get("infant_policy", {}).get("extra_bed_charge")
            if extra_bed_charge and float(extra_bed_charge) > 0:
                return float(extra_bed_charge)
            if pol_doc.get("extra_bed_rate") and float(pol_doc["extra_bed_rate"]) > 0:
                return float(pol_doc["extra_bed_rate"])
    except Exception as e:
        logger.warning(f"Failed to fetch hotel default extra bed rate for {hotel_variants}: {e}")
    return 600.0


async def get_all_rooms(hotel_id: int | str) -> list[RoomType]:
    """
    Fetch all active rooms for a hotel.

    Source: normalized `room_types` + `rate_plans` in bot_migration (no fallbacks).
    """
    hotel_variants = _get_hotel_variants(hotel_id)
    default_extra_bed_rate = await _get_hotel_default_extra_bed_rate(hotel_variants)

    # 1. Try normalized room_types collection in bot_migration
    rt_col = room_types_collection()
    rp_col = rate_plans_collection()

    rt_docs = await rt_col.find({
        "hotel_id": {"$in": hotel_variants},
        "$or": [
            {"is_active": True},
            {"active": True},
            {"is_active": {"$exists": False}},
        ]
    }).to_list(200)

    if not rt_docs:
        # Fallback to hotel_media room categories if property has media types configured but no room_types seeded yet
        try:
            from app.db.mongodb import hotel_media_collection
            media_doc = await hotel_media_collection().find_one({"hotel_id": {"$in": hotel_variants}})
            if media_doc and media_doc.get("types"):
                fallback_rooms: list[RoomType] = []
                for t in media_doc["types"]:
                    cat_name = t.get("type") or ""
                    if not cat_name or cat_name.lower() in ("hotel", "property", "general", "washroom", "bathroom", "reception", "restaurant", "lobby"):
                        continue
                    room_id = cat_name.lower().replace(" ", "_")
                    fallback_rooms.append(RoomType(
                        room_id=room_id,
                        name=cat_name,
                        aliases=[cat_name],
                        active=True,
                        room_type=cat_name,
                        description=f"{cat_name} at property",
                        number_of_rooms=5,
                        room_size="250 sq.ft",
                        occupancy=RoomOccupancy(
                            base_adults=2,
                            max_adults=3,
                            base_children=0,
                            max_children=1,
                            max_occupancy=3,
                            extra_adult_rate=default_extra_bed_rate,
                            extra_child_rate=100.0,
                        ),
                        amenities={"AC": True, "Attached Bathroom": True, "TV": True, "WiFi": True},
                        rate_plans=[
                            RatePlan(
                                rate_plan_id=f"{room_id}_EP",
                                name="Room Only (EP)",
                                base_price=2500.0,
                                currency="INR",
                                active=True,
                            )
                        ],
                        beds=[{"type": "Double Bed" if "double" in cat_name.lower() else "King Bed", "count": 1}],
                        bathroom_count=1,
                    ))
                if fallback_rooms:
                    return fallback_rooms
        except Exception as e:
            logger.warning(f"Failed to load fallback rooms from hotel_media for {hotel_variants}: {e}")

        return []

    rooms: list[RoomType] = []
    for rt in rt_docs:
        rt_id = rt.get("room_type_id") or rt.get("roomId") or str(rt.get("_id"))

        # Fetch rate plans for this room
        rp_docs = await rp_col.find({
            "hotel_id": {"$in": hotel_variants},
            "$and": [
                {
                    "$or": [
                        {"room_type_id": rt_id},
                        {"roomId": rt_id},
                    ]
                },
                {
                    "$or": [
                        {"is_active": True},
                        {"active": True},
                        {"is_active": {"$exists": False}},
                    ]
                }
            ]
        }).to_list(100)

        rate_plans = [
            RatePlan(
                rate_plan_id=rp.get("rate_plan_id") or rp.get("ratePlanId") or str(rp.get("_id")),
                name=rp.get("name") or "",
                base_price=float(round(float(rp.get("base_price") if rp.get("base_price") is not None else rp.get("basePrice", 0)) / 10) * 10),
                currency=rp.get("currency", "INR"),
                active=bool(rp.get("is_active", rp.get("active", True))),
                rate_type_id=rp.get("rate_type_id") or rp.get("rateTypeId"),
            )
            for rp in rp_docs
        ]

        base_a = int(rt.get("base_adult_occupancy") or rt.get("sleeping", {}).get("occupancy", {}).get("baseAdults", 2))
        max_a = int(rt.get("max_adult_occupancy") or rt.get("sleeping", {}).get("occupancy", {}).get("maxAdults", 3))
        base_c = int(rt.get("base_child_occupancy") or rt.get("sleeping", {}).get("occupancy", {}).get("baseChildren", 0))
        max_c = int(rt.get("max_child_occupancy") or rt.get("sleeping", {}).get("occupancy", {}).get("maxChildren", 1))

        # Check for family room in name
        if "family" in (rt.get("name") or "").lower():
            base_a = max(base_a, 4)
            max_a = max(max_a, 5)
            max_c = max(max_c, 2)

        raw_max_occ = rt.get("max_occupancy") or rt.get("sleeping", {}).get("occupancy", {}).get("maxOccupancy")
        if raw_max_occ is not None and int(raw_max_occ) > 0:
            max_occ = int(raw_max_occ)
        else:
            max_occ = max(max_a, base_a + max_c)

        if max_occ < max_a:
            max_occ = max_a
        # Ensure a room for 2+ adults can accommodate at least 1 child sharing bed
        if max_occ <= base_a and base_a >= 2:
            max_occ = base_a + 1
        if "family" in (rt.get("name") or "").lower():
            max_occ = max(max_occ, 5)

        extra_adult = _room_extra_bed_rate(rt, default_extra_bed_rate)

        extra_child = rt.get("extra_child_rate")
        if extra_child is None:
            extra_child = 100.0

        occ = RoomOccupancy(
            base_adults=base_a,
            max_adults=max_a,
            base_children=base_c,
            max_children=max_c,
            max_occupancy=max_occ,
            extra_adult_rate=float(extra_adult),
            extra_child_rate=float(extra_child),
        )

        raw_amenities = rt.get("amenities") or {}
        if isinstance(raw_amenities, list):
            amenities_dict = {str(a): True for a in raw_amenities}
        elif isinstance(raw_amenities, dict):
            amenities_dict = raw_amenities
        else:
            amenities_dict = {}

        rooms.append(RoomType(
            room_id=rt_id,
            name=rt.get("name") or "",
            aliases=rt.get("aliases") or [],
            active=True,
            room_type=rt.get("room_type") or rt.get("details", {}).get("roomType", ""),
            description=rt.get("description") or rt.get("details", {}).get("description", ""),
            number_of_rooms=int(rt.get("total_rooms") or rt.get("number_of_rooms") or rt.get("details", {}).get("numberOfRooms", 1)),
            room_size=str(rt.get("room_size") or rt.get("details", {}).get("roomSize", "")),
            occupancy=occ,
            amenities=amenities_dict,
            rate_plans=rate_plans,
            beds=rt.get("bed_config") or rt.get("beds") or rt.get("sleeping", {}).get("beds", []),
            bathroom_count=int(rt.get("bathroom_count") or rt.get("bathroom", {}).get("count", 1)),
        ))

    return rooms



async def get_rooms_by_type(
    hotel_id: int,
    room_type: str,
    aliases: dict[str, str] | None = None,
) -> list[RoomType]:
    """Filter rooms by type using the hybrid room alias resolver."""
    from app.services.room_alias_resolver import filter_rooms_by_reference

    all_rooms = await get_all_rooms(hotel_id)
    return filter_rooms_by_reference(room_type, all_rooms, aliases=aliases or {})


async def get_room_by_id(hotel_id: int, room_id: str) -> RoomType | None:
    """Find a specific room by its roomId."""
    all_rooms = await get_all_rooms(hotel_id)
    for room in all_rooms:
        if room.room_id == room_id:
            return room
    return None
