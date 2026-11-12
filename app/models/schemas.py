"""
Pydantic v2 models for the entire application.

Covers:
- API request/response schemas
- Gemini structured output schemas
- MongoDB document models
- Internal domain objects
"""

from __future__ import annotations

import enum
import json
from datetime import datetime, UTC
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.escalation_schemas import EscalationClassification
from app.models.conversation_overrides import ConversationOverrides


# ═══════════════════════════════════════════════════════════════════
# 1. ENUMS
# ═══════════════════════════════════════════════════════════════════

class Intent(str, enum.Enum):
    """All intents the bot can recognise."""
    GREETING = "GREETING"
    BOOK_ROOM = "BOOK_ROOM"
    SELECT_ROOM = "SELECT_ROOM"
    CONFIRM_BOOKING = "CONFIRM_BOOKING"
    MODIFY_BOOKING = "MODIFY_BOOKING"
    CANCEL_BOOKING = "CANCEL_BOOKING"
    CHECK_AVAILABILITY = "CHECK_AVAILABILITY"
    PAYMENT_STATUS = "PAYMENT_STATUS"
    PRICE_MATCH = "PRICE_MATCH"
    FAQ = "FAQ"
    THANK_YOU = "THANK_YOU"
    SELECTED_ROOM_DETAILS = "SELECTED_ROOM_DETAILS"
    BOOKING_SUMMARY = "BOOKING_SUMMARY"
    ROOM_FEATURE_LOOKUP = "ROOM_FEATURE_LOOKUP"
    SHOW_PRICES = "SHOW_PRICES"
    SHOW_LOCATION = "SHOW_LOCATION"
    SHOW_AMENITIES = "SHOW_AMENITIES"
    CHECK_EXISTING_BOOKING = "CHECK_EXISTING_BOOKING"
    SOMETHING_ELSE = "SOMETHING_ELSE"
    COMPLAINT = "COMPLAINT"
    HUMAN_REQUEST = "HUMAN_REQUEST"
    PERSONALITY_SWITCH = "PERSONALITY_SWITCH"
    HOTEL_TASK = "HOTEL_TASK"
    ROOM_SERVICE_ORDER = "ROOM_SERVICE_ORDER"
    LIFECYCLE_RESET = "LIFECYCLE_RESET"  # Start-fresh confirmation flow
    BANK_DETAILS = "BANK_DETAILS"
    UNKNOWN = "UNKNOWN"


class BookingStage(str, enum.Enum):
    """Deterministic stage — drives graph routing, not the LLM."""
    # ── Canonical Booking Lifecycle Stages (Sir's Spec) ──
    LEAD = "lead"
    PAYMENT_CHECK = "payment_check"
    CONFIRMED = "confirmed"
    IN_HOUSE = "in_house"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"

    # ── Granular / Conversational sub-stages within Stage 1 (Lead) ──
    NEW = "new"
    COLLECTING = "collecting"
    COLLECTING_CUSTOM_ROOM = "collecting_custom_room"
    CUSTOM_ROOMS_SELECTED = "custom_rooms_selected"
    COMPOSITE_SELECTED = "composite_selected"
    ROOMS_SHOWN = "rooms_shown"
    ROOMS_SELECTED = "rooms_selected"
    QUOTE_SHOWN = "quote_shown"
    AWAITING_PAYMENT = "awaiting_payment"
    CHECKED_IN = "checked_in"
    CANCELLATION_PENDING = "cancellation_pending"
    LEAD_CREATED = "lead_created"
    INQUIRY_CREATED = "inquiry_created"


# ═══════════════════════════════════════════════════════════════════
# 2. GEMINI STRUCTURED OUTPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class PaymentDetails(BaseModel):
    # Payment Info
    amount: float | None = None
    transaction_id: str | None = None
    status: str | None = None
    sender_name: str | None = None
    receiver_name: str | None = None
    payment_app: str | None = None
    timestamp: str | None = None

    # Room / Stay Details
    room_type: str | None = None
    room_number: str | None = None
    num_rooms: int | None = None
    num_adults: int | None = None
    num_children: int | None = None
    num_nights: int | None = None
    check_in: str | None = None
    check_out: str | None = None
    meal_plan: str | None = None
    
    # Customer Details
    customer_name: str | None = None
    customer_phone: str | None = None
    customer_email: str | None = None

    # Booking Reference
    booking_id: str | None = None
    hotel_name: str | None = None

    # LLM Notes (Max 5 points)
    notes: list[str] | str | None = None

    @field_validator("notes", mode="before")
    @classmethod
    def normalize_notes(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            v_str = v.strip()
            return [v_str] if v_str else []
        if isinstance(v, list):
            return [str(item) for item in v]
        return v


class RoomRequirementItem(BaseModel):
    room_type: str         # e.g., "Double", "Triple", "Manager", "Single", "Suite"
    count: int             # e.g., 10, 3, 1
    is_foc: bool = False   # True if complimentary / FOC / Free of Charge
    with_extra_bed: bool = False  # True if user explicitly asked for extra bed in this room
    # Set by the extraction when a guest states how many people go in each room
    # ("room 1 2 people, room 2 4 people", "2 people 1 room") without naming a
    # room type. Then `room_type` is empty and this is the **adult seats** the
    # room must provide — the allocator picks the cheapest room that seats them.
    # It is also the marker `is_an_explicit_split` reads, which is why it must be
    # declared here: Pydantic's default `extra="ignore"` would drop it silently
    # at the `model_dump()` round-trip in `understand`, leaving a request that
    # parsed correctly and then behaved as though it never had.
    guests: int | None = None
    # The AC class for this line, when the guest named one. None and
    # "NO_PREFERENCE" both mean "do not filter" — see `ac_text`.
    ac: str | None = None


class RoomRequestLine(BaseModel):
    """
    One room type in a "build your own" request, as the normaliser reads it.

    Every field has a default on purpose. Downstream, `RoomRequirementItem`
    requires both a type and a count — the right contract once the pipeline has
    decided a line is real. Here it is still deciding, so a line the model
    half-read must be droppable by the caller rather than failing the whole
    reply. One bad line should cost one line, not the request.
    """
    room_type: str = ""
    count: int = 0
    is_foc: bool = False
    with_extra_bed: bool = False
    # Mirrors `RoomRequirementItem`: a per-room occupancy line from the split
    # reader, rather than a room type. Inert on the normaliser path today — the
    # reader runs first — but declared so the two models stay interchangeable.
    guests: int | None = None
    ac: str | None = None


class RoomRequestExtraction(BaseModel):
    """The `extract_room_request` reply — the guest's request, normalised."""
    rooms: list[RoomRequestLine] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_list(cls, data):
        """
        Wrap a top-level JSON array into ``{"rooms": [...]}``.

        Asked for "one entry per distinct room type", the model returns a bare
        array — it did so on all five live cases (2026-09-21), under
        `response_mime_type="application/json"` and with this object described
        in the prompt. That is the right answer in the wrong envelope, and
        refusing it would send every single request down the regex fallback for
        a formatting difference while the read itself was correct.

        So this validates the *content*, not the packaging. The prompt asks for
        the object shape; this is what happens when it declines.
        """
        if isinstance(data, list):
            return {"rooms": data}
        return data


class ExtractedBookingData(BaseModel):
    """
    What Gemini extracts from a single user message.
    Fields are Optional — null means "not mentioned in this message".
    We merge non-null values into the session.
    """
    adults: int | None = None
    children: int | None = None
    child_ages: list[int] = Field(default_factory=list)
    num_rooms: int | None = None
    check_in: str | None = None        # YYYY-MM-DD or natural language date
    check_out: str | None = None
    room_type_preference: str | None = None
    guest_name: str | None = None
    guest_email: str | None = None
    guest_phone: str | None = None
    special_requests: str | list[str] | None = None
    total_price: float | None = None
    pricing_quote: dict | None = None

    ambiguous_guests: int | None = None
    budget: float | None = None
    feature_preferences: list[str] | None = None
    ac_preference: str | None = None               # e.g. "AC", "NON_AC", "NO_PREFERENCE"
    occupancy_preference: str | None = None        # e.g. "BASE", "MAX", "NO_PREFERENCE"
    occupancy_mode: str | None = None              # "MODE_BASE", "MODE_MAX", "MODE_CUSTOM"
    payment_details: PaymentDetails | None = None
    photo_tag_preference: str | None = None
    partial_payment_amount: float | None = None
    meal_plan_preference: str | None = None
    recommendation_preference: str | None = None   # e.g. "CHEAPEST"
    display_preference: str | None = None           # e.g. "SHOW_ALL"
    room_count_preference: str | None = None        # e.g. "FEWER_ROOMS"
    comfort_preference: str | None = None           # e.g. "COMFORTABLE_STAY"
    coupon_code: str | None = None
    extra_services: list[str] | None = None
    room_requirements: list[RoomRequirementItem] | None = Field(default=None, description="List of multi-category room requirements with count and FOC status, e.g. 10 Double, 3 Triple, 1 FOC")
    composite_template_shown: bool = False
    active_persona: str | None = None

class GeminiUnderstandingResponse(BaseModel):
    """
    The single structured JSON that Gemini returns per user message.
    Combines intent classification + entity extraction + escalation assessment in ONE call.
    """
    intent: str = Field(description="One of: GREETING, BOOK_ROOM, SELECT_ROOM, CONFIRM_BOOKING, MODIFY_BOOKING, CANCEL_BOOKING, CHECK_AVAILABILITY, PAYMENT_STATUS, PRICE_MATCH, FAQ, AMENITY_QUERY, THANK_YOU, HUMAN_REQUEST, HOTEL_TASK, PERSONALITY_SWITCH, ROOM_SERVICE_ORDER, LIFECYCLE_RESET, UNKNOWN")
    target_queue_index: int | None = Field(default=None, description="If acting on a specific booking in a multi-booking queue, specify the 0-based index.")
    detected_language: str | None = Field(default=None, description="The conversational style used by the guest, e.g., Hinglish, Formal Hindi, Casual English, Kannada.")
    base_language: str | None = Field(default=None, description="The underlying language family, e.g., Hindi, English, Telugu, Kannada. For Hinglish this is Hindi.")
    language_confidence: float | None = Field(default=None, description="Confidence of language detection 0.0 to 1.0")
    language_switch_requested: bool = Field(default=False, description="True if the user explicitly asked to change the language.")
    requested_language: str | None = Field(
        default=None,
        description=(
            "When language_switch_requested is true, the language the guest asked to switch TO "
            "(e.g. Kannada for 'can you speak in kannada?'), not the language of the message itself."
        ),
    )
    extracted_data: ExtractedBookingData = Field(default_factory=ExtractedBookingData)
    extracted_bookings: list[ExtractedBookingData] | None = Field(default=None, description="If multiple independent bookings are detected in one message, put them here.")
    faq_question: str | None = Field(default=None, description="If intent is FAQ, the actual question to look up")
    reset_booking: bool = Field(
        default=False,
        description=(
            "True when the user wants to abandon the current in-progress draft and start fresh "
            "(any phrasing: start over, fresh booking, bro start again, decline continue, etc.). "
            "False for normal booking updates or cancelling a confirmed booking."
        ),
    )
    escalation: EscalationClassification | None = Field(default=None, description="Escalation classification — Gemini classifies, Python decides")
    task_classification: dict | None = Field(default=None, description="Hotel task classification — Gemini classifies, Python handles routing/creation")
    room_service_extraction: dict | None = Field(default=None, description="Structured extraction for Room Service cart operations")
    amenity_query: dict | None = Field(default=None, description="Amenity query extraction: {amenity_name: str, query_types: list[str]}")
    llm_unavailable: bool = Field(
        default=False,
        exclude=True,
        description=(
            "Set by the bot, never by the model: the provider was unreachable, so every "
            "other field on this object is a placeholder rather than an answer."
        ),
    )

# ═══════════════════════════════════════════════════════════════════
# 3. ROOM / RATE PLAN DOMAIN MODELS
# ═══════════════════════════════════════════════════════════════════

class RoomOccupancy(BaseModel):
    base_adults: int = 2
    max_adults: int = 3
    base_children: int = 0
    max_children: int = 2
    max_occupancy: int = 4
    extra_adult_rate: float = 0.0
    extra_child_rate: float = 0.0
    extra_bed_allowed: bool = True
    max_extra_beds: int = 1


class RatePlan(BaseModel):
    rate_plan_id: str
    name: str
    base_price: float
    raw_base_price: float | None = None
    currency: str = "INR"
    active: bool = True
    rate_type_id: str | None = None


class RoomType(BaseModel):
    """Flattened view of a room from normalized room_types/rate_plans for internal use."""
    room_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    active: bool = True
    room_type: str = ""             # "Deluxe", "Standard", "Premier", etc.
    description: str = ""
    number_of_rooms: int = 1        # Total physical rooms of this type
    room_size: str = ""
    occupancy: RoomOccupancy = Field(default_factory=RoomOccupancy)
    amenities: dict[str, bool] = Field(default_factory=dict)
    rate_plans: list[RatePlan] = Field(default_factory=list)
    beds: list[dict] = Field(default_factory=list)
    bathroom_count: int = 1

    def adult_seats(self) -> int:
        """How many adults this room holds, with or without a mattress.

        `max(base_adults, max_adults)`, and the `max` is not decoration: the
        schema defaults `max_adults` to 3, so a room stating a base of 4 would
        otherwise report 3 seats and be capped below what it advertises.

        This is the *adult* ceiling. `max_occupancy` is a different number —
        adults **plus** children — and using it here was the ₹600 mattress
        defect: a third adult in a base-2/max-3 room read as "one over base",
        was charged for a mattress, and the room advertised "Sleeps 2" to a
        party of three. Product, 2026-09-25: "max occupancy means, adults +
        children, and max_adults is a hard adult ceiling".

        Shared so the third copy of this expression cannot drift from the
        first two — `room_service._card_candidates` and this.
        """
        return max(
            int(getattr(self.occupancy, "base_adults", 0) or 0),
            int(getattr(self.occupancy, "max_adults", 0) or 0),
        )


class SelectedRoom(BaseModel):
    """A room + rate plan the user has chosen."""
    room_id: str
    room_name: str
    aliases: list[str] = Field(default_factory=list)
    rate_plan_id: str
    rate_plan_name: str
    # The catalogue's own identifier for the meal plan (`RT_EP` / `RT_CP` /
    # `RT_MAP`), carried through so a charge can be gated on the plan the guest
    # actually bought. It already rode on `RatePlan` (`room_repository.py:204`)
    # and was dropped here, which left every downstream reader to recover the
    # plan by substring-matching `rate_plan_name` — see `app/utils/meal_plan.py`.
    rate_type_id: str | None = None
    price_per_night: float
    raw_price_per_night: float | None = None
    currency: str = "INR"
    assigned_adults: int = 2
    assigned_children: int = 0
    extra_adult_charge: float = 0.0
    extra_child_charge: float = 0.0
    # How many mattresses are on this room, as distinct from how much was charged
    # for the people above base occupancy. The two used to be the same number —
    # `extra_adult_charge > 0` *was* the test for "there is a bed here" at
    # `room_service.py:1738`, `booking_engine.py:391` and
    # `validator.py:487` — and they come apart the moment a CP guest is charged
    # for a person without being offered a bed. Reading the charge as a bed
    # count would then print "Extra bed" for a bed the hotel never sold, and
    # suppress the mattress question (`validator._selection_carries_a_bed`).
    #
    # `None` is "nobody said", and it keeps the old fallback — the charge itself
    # — so a selection written before this field existed behaves exactly as it
    # did. Same convention as `max_adults` below.
    extra_bed_count: int | None = None
    # The catalogue's two ceilings for the room, carried onto the selection so
    # `validator` can check the set against them.
    #
    # `validator.py:282-283` has always read these two field names off a selected
    # room, and until 2026-09-29 nothing ever wrote them — so the read fell
    # through to `assigned_adults + assigned_children`, which is the allocator's
    # own distribution and not the room's capacity. Those two numbers disagree
    # for every room whose `max_adults` sits above its `base_adults`, because the
    # requirements allocator records a named room at its base: a Deluxe King Bed
    # (base 2 / max 3 / occupancy 4) went on the selection as 2, and two of them
    # then totalled 4 seats against a party of six.
    #
    # `None` is "the catalogue did not say", which keeps the old fallback — the
    # combination allocator is left unfilled deliberately, since its fallback is
    # already the correct number and changing it would suppress wipes that are
    # doing real work.
    #
    # `max_adults` is the *adult* ceiling; `max_occupancy` is adults **plus**
    # children. Two different numbers, and the validator needs both — see
    # `RoomType.adult_seats`.
    max_adults: int | None = None
    max_occupancy: int | None = None


class AllocationOption(BaseModel):
    """A generated multi-room combination."""
    option_id: str
    rooms: list[SelectedRoom]
    total_price_per_night: float = 0.0
    total_price_for_stay: float | None = None
    base_adults: int = 0
    base_children: int = 0
    max_adults: int = 0
    max_children: int = 0
    meal_upgrades: list[dict] = Field(default_factory=list)
    allocation_type: str = "COMFORT"
    extra_adult_count: int = 0
    extra_child_count: int = 0
    recommendation_labels: list[str] = Field(default_factory=list)
    option_label: str | None = None
    display_label: str | None = None


# ═══════════════════════════════════════════════════════════════════
# 4. PRICING
# ═══════════════════════════════════════════════════════════════════

class TaxDetails(BaseModel):
    base_amount: float
    tax_amount: float
    tax_rate: float
    tax_type: str  # "inclusive" or "exclusive"
    final_amount: float


class RoomLineItem(BaseModel):
    room_name: str
    rate_plan: str
    quantity: int
    price_per_night: float
    num_nights: int
    subtotal: float
    tax_details: TaxDetails | None = None
    extra_adult_charge: float = 0.0
    extra_child_charge: float = 0.0
    # Carried from `SelectedRoom` so the printed quote can tell an extra *bed*
    # from an extra *person* — the two are the same money on the line above but
    # different sentences to the guest. See `SelectedRoom.extra_bed_count`.
    extra_bed_count: int | None = None


class PriceBreakdown(BaseModel):
    rooms: list[RoomLineItem] = Field(default_factory=list)
    upsells: list[dict] = Field(default_factory=list)  # List of upsell TaxDetails with item name
    subtotal: float = 0.0
    discount_amount: float = 0.0
    discount_reason: str | None = None
    coupon_code: str | None = None
    # Only a fallback for a breakdown built by hand — the engine always sets
    # this from the rates it actually applied. It tracks the 5% blanket in
    # settings.GST_RATE_OVERRIDE_PERCENT so the label cannot disagree with the
    # numbers; it read 12% until 2026-09-29.
    tax_percent: float = 5.0
    tax_amount: float = 0.0
    # Staff-negotiated overlay. Signed: negative reduces the computed total.
    # Invariant: subtotal + tax_amount + adjustment_amount + fee_total == total,
    # where fee_total is the sum of `early_late_fees` below.
    adjustment_amount: float = 0.0
    adjustment_reason: str | None = None
    # One-off paid early check-in / late check-out charges, already final: each
    # entry is {"kind", "amount", "label", "disclaimer"} and its `amount` is
    # added to `total` but NOT to `subtotal`, because the rate it was derived
    # from is GST-inclusive and taxing it again would double-charge the GST.
    # Declared here rather than left to the dict: pydantic v2 is extra="ignore"
    # and `session_mapper` rebuilds this model with `PriceBreakdown(**pb)`, so an
    # undeclared field would be silently dropped on every save/load round-trip.
    early_late_fees: list[dict] = Field(default_factory=list)
    total: float = 0.0
    currency: str = "INR"
    season_name: str | None = None
    season_multiplier: float = 1.0

    @property
    def early_late_fee_total(self) -> float:
        """Sum of the one-off early/late fees. Used by the printed quote chain."""
        return float(sum(float(f.get("amount") or 0.0) for f in self.early_late_fees or []))


class Coupon(BaseModel):
    coupon_code: str
    hotel_id: int | str
    active: bool = True
    discount_type: str  # "percentage" or "fixed"
    discount_value: float
    min_booking_value: float | None = None
    max_discount_amount: float | None = None
    min_nights: int | None = None
    applicable_room_types: list[str] | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    stay_start: datetime | None = None
    stay_end: datetime | None = None
    usage_limit: int | None = None
    times_used: int = 0
    promotion_type: str | None = None
    promotion_name: str | None = None
    base_discount: float | None = None
    add_on_discount: float | None = None

# ═══════════════════════════════════════════════════════════════════
# 5. SESSION (MongoDB document model)
# ═══════════════════════════════════════════════════════════════════

#: What each `role` implies about who spoke. `role` is the old field and is
#: always present; `sender_type` was added later with a default, and that
#: default is where the history went wrong (see `MessageRecord`).
_ROLE_TO_SENDER: dict[str, str] = {
    "assistant": "bot",
    "bot": "bot",
    "ai": "bot",
    "system": "system",
    "tool": "system",
    "staff": "staff",
    "agent": "staff",
    "user": "guest",
    "human": "guest",
}


class MessageRecord(BaseModel):
    role: str                       # "user" or "assistant"
    content: str
    sender_type: str = "guest"      # "guest", "staff", "bot", "system"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def _derive_sender_type(cls, data: Any) -> Any:
        """Keep `sender_type` in step with `role`.

        `sender_type` was added after `role` with a default of "guest", and
        every writer that appended a bot reply passed only `role="assistant"`.
        So the default stuck, and the whole saved history ended up stamped as
        the guest speaking. Measured on live sessions (2026-10-20, 15,174
        messages): 7,632 bot replies and 94 staff system notes, every one of
        them `sender_type="guest"`, and not one `sender_type="bot"`.

        That is not a cosmetic label. The bot's replies are the only place the
        guest's *questions* live, and the model reads this history to work out
        what a message means. With the bot's own lines attributed to the guest,
        "28 october" arriving after "What is your check-in date?" looks like
        the guest talking to themselves, so the date has no direction and gets
        read as both ends of the stay.

        Two things are fixed here, and the difference matters:

        * Future writes are correct, because the role says who spoke.
        * Documents already in Mongo are repaired **on load**, because
          `MessageRecord(**msg)` re-validates them. There is no migration to
          run and no backfill to wait for — the next read of a session fixes it.

        A staff turn is the one case where the sender is more specific than
        the role: staff type through the guest's own channel, so a `role="user"`
        message that says it is staff keeps saying so, and a staff summary
        written back into the session keeps its `role="system"`.
        """
        if not isinstance(data, dict):
            return data
        current = str(data.get("sender_type") or "").strip().lower()
        if current in ("staff", "agent", "user_staff"):
            # A human is speaking. The role is our own coarse bucket and
            # cannot see this.
            return data
        role = str(data.get("role") or "").strip().lower()
        implied = _ROLE_TO_SENDER.get(role)
        if implied is None:
            # An unrecognised role. Keep whatever the row says, and never let
            # a null through a field that is not optional.
            return data if current else {**data, "sender_type": "guest"}
        if current == implied:
            return data
        # A new dict rather than an in-place edit: Pydantic hands this
        # validator the caller's own dict when it validates a document, and
        # mutating it would rewrite the raw row that was read out of Mongo.
        return {**data, "sender_type": implied}


class BookingSession(BaseModel):
    """
    Full conversation state — stored in MongoDB Booking_Sessions.
    This is the SINGLE SOURCE OF TRUTH for a conversation.
    """
    session_id: str
    hotel_id: str | int
    customer_identifier: str = ""   # phone or temp ID
    preferred_language: str | None = None      # Conversational style, e.g. "Hinglish", "Formal Hindi"
    conversation_language: str | None = None   # Base language family, e.g. "Hindi", "English", "Telugu"

    # Booking stage (deterministic state machine)
    booking_stage: BookingStage = BookingStage.NEW

    @field_validator("booking_stage", mode="before")
    @classmethod
    def _coerce_booking_stage(cls, v: Any) -> BookingStage:
        if isinstance(v, str):
            try:
                return BookingStage(v)
            except ValueError:
                return BookingStage.NEW
        return v or BookingStage.NEW

    intent: str = ""

    # Multi-Booking Queue Management
    queue_id: str | None = None
    booking_queue: list[dict] = Field(default_factory=list)
    active_booking_index: int = 0
    completed_bookings: list[dict] = Field(default_factory=list)
    failed_bookings: list[dict] = Field(default_factory=list)
    # Set once a multi-stay queue has been quoted as a whole. From then on the
    # queue is read one booking at a time again (index 0 is payable), so a
    # later re-quote must not re-enter the collection flow.
    combined_quote_shown: bool = False

    # Accumulated booking data
    adults: int | None = None
    children: int | None = None
    child_ages: list[int] = Field(default_factory=list)
    child_age_ask_count: int = 0
    child_age_assumed: bool = False
    effective_adults: int | None = None
    effective_children: int | None = None
    # How many of `effective_children` the hotel's own child policy makes
    # chargeable (strictly between its free_till and paid_till). Stored rather
    # than recomputed on read so it cannot drift from `effective_children`: a
    # turn that reloads the party but not the policy would otherwise seat two
    # children and charge for none.
    paid_child_count: int = 0
    child_age_notes: list[str] = Field(default_factory=list)
    num_rooms: int | None = None
    room_requirements: list[dict] | None = None
    check_in: str | None = None
    check_out: str | None = None
    # The arrival/departure time the guest volunteered, as "HH:MM" 24-hour.
    # Only ever written from the guest's own words — the bot never asks for
    # these, and a quote is never charged an early/late fee without one.
    # Persisted rather than read off `user_message` because that field is
    # turn-scoped: a guest who says "I'll reach at 7am" and then picks a room
    # two turns later must still be priced for the 7am.
    requested_arrival_time: str | None = None
    requested_departure_time: str | None = None
    # True for the single turn after the bot has asked the guest for that time.
    # The bot's request ends one turn and the answer arrives on the next, so the
    # question has to be remembered across the gap or the answer is just an
    # unexplained "10AM" to whatever node sees it next. Cleared at the top of
    # every turn and re-set only when the question is asked again, so it can
    # never outlive the question it belongs to.
    awaiting_early_late_time: bool = False
    room_type_preference: str | None = None
    ac_preference: str | None = None
    ac_qualified: bool = False
    occupancy_preference: str | None = None
    occupancy_mode: str | None = None          # "MODE_BASE", "MODE_MAX", "MODE_CUSTOM"
    grouping_mode: str | None = None
    composition_asked: bool = False
    composition_options_count: int | None = None
    subgroups: list[dict] | None = None
    next_action: str | None = None
    last_prompt_action: str | None = None
    selected_rooms: list[SelectedRoom] = Field(default_factory=list)
    available_rooms: list[dict] = Field(default_factory=list)
    allocation_options: list[dict] = Field(default_factory=list)
    # T2/T3 paging — see ConversationState for the invariant these three maintain.
    room_options_all: list[dict] = Field(default_factory=list)
    room_option_offset: int = 0
    room_options_key: str = ""
    # B1. Whether the photos for the quote in room_options_key have already been sent.
    quote_presented: bool = False
    room_previews_sent: bool = False
    quote_photos_sent: bool = False
    guest_name: str | None = None
    guest_email: str | None = None
    guest_phone: str | None = None
    special_requests: str | list[str] | None = None
    ambiguous_guests: int | None = None
    budget: float | None = None
    feature_preferences: list[str] | None = None
    photo_tag_preference: str | None = None
    meal_plan_preference: str | None = None
    applied_coupon: str | None = None
    gst_slab_applied: float | None = None
    soft_hold_expires_at: datetime | None = None
    upsell_shown: bool = False
    last_upsell_offered: str | None = None
    declined_upsells: list[str] = Field(default_factory=list)
    updated_fields: list[str] = Field(default_factory=list)
    room_type_not_available: bool = False
    active_personality_mode: str | None = None
    personality_mode_switched_at: datetime | None = None

    # ── Promoted from state.py (see app/models/session_contract.py) ──
    # Each of these was declared on ConversationState and read on a turn after
    # the one that set it, but had no home here — and because upsert_session
    # replaces the whole workflow_state document and state_to_session rebuilds
    # the session from scratch, "no home here" meant "reset every turn".
    composite_template_shown: bool = False      # latch: don't show the template twice
    extra_services: list[str] = Field(default_factory=list)          # add-ons chosen so far
    pending_extra_services: list[dict] = Field(default_factory=list)  # awaiting payment
    booking_resume_nudge_count: int = 0                               # Capped at 2 resume nudges per active booking stage

    # ── Stage-lifecycle latches ──
    # Same reasoning as the block above: every one of these is read on a turn
    # *after* the one that set it. They were declared on ConversationState by
    # the stage-lifecycle commit without a column here, which meant LangGraph
    # dropped them and each read returned None — so the bot re-asked the
    # question, re-sent the welcome, or forgot the claim every single turn.
    separate_booking_clarification_asked: bool = False  # asked "is this a separate booking?" — don't ask twice
    # The guest said yes to that question: a new booking is being built while the
    # confirmed stay still stands. Read by `context_loader` on every later turn and
    # handed to `resolve_guest_stage`, which holds the standing stay aside so the
    # draft can advance instead of being dragged back to `confirmed`.
    #
    # Persisted because it is read on turns *after* the one that set it, exactly
    # like the latch above — the same omission that made those fields read back as
    # None every turn. Cleared when the session reaches `confirmed` again (the new
    # booking is done) and by `/start fresh`.
    separate_booking_active: bool = False
    # When that "yes" arrived, ISO-8601 UTC. `context_loader` expires the latch
    # after SECOND_BOOKING_LATCH_HOURS; without a bound a guest who asks about a
    # second booking and then walks away would resolve as a lead forever, with
    # their own standing booking unreachable from every stage path.
    #
    # A str rather than a datetime because it round-trips through graph state,
    # which is JSON: a datetime would have to be coerced on both sides, and this
    # value is only ever compared, never rendered.
    separate_booking_started_at: str | None = None
    checkin_welcome_sent: bool = False                  # in-stay welcome already sent
    payment_proof_submitted: bool = False               # guest claims payment; read by routing to route the claim
    feedback_recorded: bool = False                     # rating captured at check-out — don't ask again
    # The check-out date of the CRM stay this guest dismissed with an explicit
    # "start fresh". A latch with a value: `resolve_guest_stage` ignores the CRM's
    # confirmation only while it names this same stay. A bare bool would be
    # simpler and wrong — it would blind the bot to a *later* confirmation for a
    # new booking, which is a thing guests do in the same conversation.
    crm_confirmation_dismissed_check_out: str | None = None
    # The booking row this guest dismissed with an explicit "start fresh".
    #
    # The latch above covers a stay staff confirmed in the CRM, which has no row of
    # ours. A guest who booked *through the bot* has a row, and that row resolves
    # the stage on every turn regardless of what the session says — so their
    # "start fresh" dropped the draft and nothing else, and the very next message
    # came back confirmed (measured on prod 2026-10-17: the reset logged
    # `check_out=None` and the stage snapped back on the next turn).
    #
    # A booking id rather than a date, because this path knows exactly which row it
    # is. Scoped to that one id, so a new booking is a different id and applies
    # normally — a guest who resets and then books again is not left blind.
    dismissed_booking_id: str | None = None
    # When this phone last asked to start over, as an ISO timestamp.
    #
    # `dismissed_booking_id` names exactly one row, and only the row that happened
    # to set the stage on the turn the guest typed "/start". That is not the same
    # thing as "this phone is new": a guest with two rows, or one whose session
    # was lost, came straight back to `confirmed` on the next message with a stale
    # row still naming them (measured 2026-10-23 on the bossman's own account,
    # whose row carries no dates and a Rs 50,000 total it cannot justify).
    #
    # So the reset is a timestamp, not an id: every row created at or before it
    # stops setting the stage. A booking made *after* the reset has a later
    # `created_at` and applies normally, which is what keeps the reset from
    # blinding the bot to a genuine new stay.
    guest_reset_at: str | None = None
    # The "our reservations team will assist you directly" card has already been
    # sent for this conversation's group enquiry.
    #
    # Only read when `GROUP_BOOKING_SELF_SERVE` is off — the flag's default makes
    # the handoff unreachable. It is persisted, not an in-turn flag, because the
    # bug it fixes was across turns: nothing set `group_lead_submitted` (the
    # in-turn flag the old guard checked), so a 7-room / 17-adult enquiry on
    # 2026-10-17 received the same paragraph four times.
    group_handoff_sent: bool = False
    awaiting: str | None = None
    context_refresh_status: str | None = None
    # The run of unanswered guest messages the hand-back message already answered,
    # as a short fingerprint, and when it was sent.
    #
    # The CRM's override scheduler calls `/api/admin/context-refresh` repeatedly —
    # measured: the same phone four to eleven times a day, minutes apart — and each
    # call used to end in a fresh message to the guest. That is what got the feature
    # switched off on 2026-10-27. `_bot_spoke_since` only compares against the
    # window's own start, so it lapses the moment the CRM sends a newer start time.
    # This latch keys on the *content* instead: the same unanswered guest turn
    # cannot be answered twice, however often the refresh runs, while a guest who
    # writes again produces a different fingerprint and is answered normally.
    takeover_resume_fingerprint: str | None = None
    takeover_resume_at: datetime | None = None

    # Booking Modification State
    modified_fields: list[str] | None = None
    previous_values: dict | None = None
    requested_values: dict | None = None
    booking_change_summary: str | None = None
    modification_metadata: dict | None = None
    
    # ── WARNING: Temporary Pending State Only ──
    # This field holds a shadow state during the 2-turn modification confirmation flow.
    # It must NEVER be treated as the active booking state.
    # It should be cleared immediately after the guest confirms or aborts the change.
    modification_candidate_state: dict | None = None

    # Lifecycle State (Confirmation tracking)
    lifecycle_action_pending: str | None = None
    lifecycle_action_data: dict | None = None

    # ID Card & Identity Verification State
    id_verified: bool = False
    id_type: str | None = None
    id_masked_number: str | None = None
    id_card_details: dict | None = None
    id_received_date: str | None = None
    room_number: str | None = None
    verified_ids: list[dict] = Field(default_factory=list)
    ac_preference: str | None = None
    ac_qualified: bool = False
    extra_bed_preference: str | None = None
    no_extra_bed: bool = False
    # The guest is buying a mattress. Set with `no_extra_bed` and
    # `extra_bed_preference` in `extra_bed_intent.apply_to_state`, and read by
    # the allocator (`wants_bed`) on every later re-price — a date change three
    # turns on must not quietly drop the ₹600 the guest agreed to.
    wants_bed: bool = False
    # Asked once per session, after the guest picks a row whose price already
    # contains a mattress. Without it `validator` would put the question again
    # on every later turn that resolves the same selection.
    extra_bed_asked: bool = False
    #: The re-ask, sent once when the first answer could not be read. Set so a
    #: guest who keeps saying something else is never asked a third time: the
    #: second unreadable answer proceeds without a mattress instead of blocking
    #: the booking. Measured on guest 919088198242 (2026-10-19): the question
    #: stood in front of five consecutive turns, three of which produced the
    #: same reply, which then tripped the repeat guard into a support ticket.
    extra_bed_reasked: bool = False

    # Pricing
    price_breakdown: PriceBreakdown | None = None
    total_price: float | None = None
    base_total_price: float | None = None
    discount_amount: float = 0.0
    discount_reason: str | None = None
    negotiated_discount_pct: float | None = None

    # ── Negotiated Overrides (legacy flat fields — prefer conversation_overrides) ──
    price_override: float | None = None
    included_breakfast: bool = False
    late_checkout: bool = False
    airport_pickup: bool = False
    special_notes: str | None = None

    # ── Conversation-level overrides (Human Takeover + future domains) ──
    conversation_overrides: ConversationOverrides = Field(
        default_factory=ConversationOverrides
    )

    def update_overrides(self, overrides: dict):
        """
        Legacy helper — kept for compatibility.
        Phase-1 Human Takeover no longer calls this; decisions go to
        conversation_overrides.pending_decisions instead.
        """
        if "price_override" in overrides and isinstance(overrides["price_override"], (int, float)):
            self.price_override = float(overrides["price_override"])
        if "room_type" in overrides and overrides["room_type"]:
            self.room_type_preference = str(overrides["room_type"])
        if "breakfast" in overrides:
            self.included_breakfast = bool(overrides["breakfast"])
        if "late_checkout" in overrides:
            self.late_checkout = bool(overrides["late_checkout"])
        if "airport_pickup" in overrides:
            self.airport_pickup = bool(overrides["airport_pickup"])
        if "special_notes" in overrides and overrides["special_notes"]:
            if self.special_notes:
                self.special_notes += f" | {overrides['special_notes']}"
            else:
                self.special_notes = str(overrides["special_notes"])

    # Payment & Booking
    payment_link: str | None = None
    payment_link_id: str | None = None  # Razorpay plink_... (required for verify)
    # The Razorpay locked-amount UPI QR issued alongside the hold. Persisted so
    # the "I've paid" turn and the reconciliation sweep can ask the backend
    # about *this* QR; the webhook finds the same row through its notes.
    qr_order_id: str | None = None   # ord_...
    qr_id: str | None = None         # qr_...
    qr_link: str | None = None       # https://rzp.io/rzp/...
    payment_status: str | None = None
    booking_id: str | None = None
    # The `PAY-…` key behind the gateway order and the inventory hold while the
    # guest has not yet sent proof of payment. Renamed to the real `BK-…` id when
    # the proof arrives; persisted so the next turn can find the hold.
    payment_reference: str | None = None
    # The confirmation turn asked for the name to print on the slip, so the next
    # message is read as the answer — see NAME_AFTER_PAYMENT_ENABLED. Persisted
    # because the answer arrives on the following turn.
    name_after_payment_pending: bool | None = None
    payment_details: PaymentDetails | None = None
    amount_paid: float = 0.0
    amount_due: float = 0.0
    # Amount expected on the current payment link (may differ from remaining balance)
    payment_link_amount: float | None = None

    # Payment follow-up tracking
    payment_missing_fields: list[str] | None = None
    payment_followup_index: int = 0

    # Escalation tracking
    escalation_failure_count: int = 0
    # B3b — repetition tracking. See app/services/repetition_guard.py.
    last_response_hash: str = ""
    repeat_response_count: int = 0

    # ID Verification Documents (DPDP Compliant)
    id_docs: list[dict] = Field(default_factory=list)

    # Conversation history (only last N messages stored here)
    messages: list[MessageRecord] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved: bool = False

    def get_recent_messages(self, n: int = 20) -> list[dict]:
        """
        Return last N messages as simple dicts for Gemini context.

        The default mirrors `settings.HISTORY_WINDOW_MESSAGES`, which every caller
        actually passes; it is kept here as a stand-in because `app.models` does
        not import config. Keep the two in step.
        """
        return [
            {
                "role": m.role,
                "content": m.content,
                # `role` is the fallback, not `sender_type`: the model reads
                # this to tell the bot's questions from the guest's answers.
                "sender_type": getattr(m, "sender_type", None)
                or ("guest" if m.role == "user" else "bot"),
            }
            for m in self.messages[-n:]
        ]

    def get_booking_context_summary(self) -> str:
        """Compact text summary of current booking state for Gemini."""
        parts = [f"Stage: {self.booking_stage.value}"]
        if getattr(self, "preferred_language", None):
            parts.append(f"Language: {self.preferred_language}")
        cov = getattr(self, "conversation_overrides", None)
        if cov is not None:
            try:
                from app.models.conversation_overrides import ConversationOverrides
                if isinstance(cov, dict):
                    cov = ConversationOverrides(**cov)
                summary = cov.prompt_summary()
                if summary:
                    parts.append(summary)
            except Exception:
                pass
        if self.adults is not None:
            parts.append(f"Adults: {self.adults}")
        if self.children is not None:
            parts.append(f"Children: {self.children}")
        if self.check_in:
            parts.append(f"Check-in: {self.check_in}")
        if self.check_out:
            parts.append(f"Check-out: {self.check_out}")
            if self.check_in:
                from app.services.booking_engine import calculate_num_nights
                nights = calculate_num_nights(self.check_in, self.check_out)
                parts.append(f"Nights: {nights}")
        if self.allocation_options and self.booking_stage.value == "rooms_shown":
            parts.append(f"Room options shown: {len(self.allocation_options)} (awaiting selection)")
        if self.selected_rooms:
            from collections import Counter
            counts = Counter(
                (r.room_name if hasattr(r, "room_name") else r.get("room_name"))
                for r in self.selected_rooms
                if (hasattr(r, "room_name") and r.room_name) or (isinstance(r, dict) and r.get("room_name"))
            )
            room_details = ", ".join(f"{cnt} x {name}" for name, cnt in counts.items())
            parts.append(f"Booked/Selected Rooms: {room_details} ({len(self.selected_rooms)} rooms total)")
        else:
            if self.num_rooms is not None:
                parts.append(f"Rooms needed: {self.num_rooms}")
            if self.room_type_preference:
                parts.append(f"Room preference: {self.room_type_preference}")
        if self.total_price is not None:
            parts.append(f"Total: ₹{self.total_price:,.0f}")
        if self.booking_id:
            parts.append(f"Booking ID: {self.booking_id}")
        if getattr(self, "budget", None) is not None:
            parts.append(f"Budget: ₹{self.budget:,.0f}")
        if getattr(self, "feature_preferences", None):
            parts.append(f"Feature Preferences: {', '.join(self.feature_preferences)}")
        if getattr(self, "payment_details", None):
            pd = self.payment_details
            pd_str = f"Uploaded Payment: {pd.status} ₹{pd.amount} via {pd.payment_app} (TXN: {pd.transaction_id})"
            parts.append(pd_str)
        if getattr(self, "payment_missing_fields", None):
            parts.append(f"payment_missing_fields: {', '.join(self.payment_missing_fields)}")
        if getattr(self, "amount_paid", 0) > 0:
            parts.append(f"Amount Paid: ₹{self.amount_paid:,.0f}")
        if getattr(self, "amount_due", 0) > 0:
            parts.append(f"Amount Due: ₹{self.amount_due:,.0f}")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 6. API REQUEST / RESPONSE
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str | None = None   # None = create new session
    message: str
    hotel_id: str | int | None = None     # Override default hotel
    customer_phone: str | None = None


class CardType(str, enum.Enum):
    TEXT = "text"
    ROOM_OPTIONS = "room_options"
    QUOTE = "quote"
    PAYMENT = "payment"
    IMAGE = "image"
    OPTIONS = "options"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    UPSELL_OPTIONS = "upsell_options"
    ADDON_CATALOG = "addon_catalog"
    # Internal request card — WhatsApp resolves & sends images; HTTP clients may ignore
    MEDIA_DELIVERY = "media_delivery"
    # A Meta `interactive` payload (reply buttons or a list), built by `to_interactive`
    # from the buttons the templates author. The CRM hands `data` straight to the WhatsApp
    # Cloud API, so the payload must survive this model untouched — which is why the type
    # has to be declared here and not only in `_collapse_for_crm_whatsapp`'s pass-through
    # tuple. Omitting it raised on `response.0.type` and 500'd every menu turn in
    # production (2026-10-14), because this enum and that whitelist are two separate gates.
    INTERACTIVE = "interactive"


class ActionButton(BaseModel):
    label: str
    action: str  # e.g., "postback", "url"
    value: str


class UICard(BaseModel):
    type: CardType
    text: str | None = None
    data: dict[str, Any] | list[Any] | None = None
    buttons: list[ActionButton] | None = None
    url: str | None = None
    caption: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: list[UICard]
    booking_stage: str
    booking_summary: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    mongodb: bool = False
    chromadb: bool = False

# ═══════════════════════════════════════════════════════════════════
# 7. HUMAN INTERVENTION
# ═══════════════════════════════════════════════════════════════════

class InterventionMessage(BaseModel):
    sender: str
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class HumanIntervention(BaseModel):
    intervention_id: str
    hotel_id: str | int
    session_id: str
    booking_id: str | None = None
    status: str = "ACTIVE"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_staff_message: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    staff_id: int | str
    reason: str = "NEGOTIATION"
    transcript: list[InterventionMessage] = Field(default_factory=list)
    ended_at: datetime | None = None
    duration_seconds: int | None = None
    gemini_summary: str | None = None
    overrides_applied: dict | None = None

