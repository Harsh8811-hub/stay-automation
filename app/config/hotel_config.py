"""
Hotel Configuration Models — separated into AI Prompt Config and Business Config.

Design Principle:
- AIPromptConfig: Only data that affects language generation (injected into Gemini)
- BusinessConfig: Feature flags, limits, rules (enforced entirely in Python)
- HotelConfig: Combined container, loaded from MongoDB

The LLM NEVER sees BusinessConfig. Python is the authoritative source
for all business logic, permissions, and feature enforcement.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)


class AIPromptConfig(BaseModel):
    """
    Configuration that directly affects how the AI communicates.
    This is the ONLY config injected into the Gemini prompt.

    Keep this minimal — every field here adds tokens to every request.
    """
    system_prompt: str = ""
    conversation_tone: str = "professional"
    supported_languages: list[str] = Field(default_factory=lambda: ["english"])
    response_length: int = 40
    custom_instructions: dict[str, list[str]] = Field(default_factory=dict)
    bot_name: str = "Hotel Assistant"
    fallback_message: str = ""


class FeatureFlags(BaseModel):
    """
    Feature toggles enforced entirely in Python.
    NEVER exposed to the AI model.
    """
    ai_enabled: bool = True
    direct_booking: bool = True
    lead_only_mode: bool = False
    dynamic_pricing: bool = True
    coupon_visibility: bool = True
    auto_upsell: bool = False
    auto_upgrade: bool = True
    price_match_enabled: bool = True
    payment_verification_enabled: bool = True
    # These two are False in the parser AND here, deliberately. They were True here and False
    # there, so a hotel resolved through from_mongo_doc and a hotel built straight from this
    # class disagreed about the same settings. Failing closed is the choice: self-serve
    # modification and cancellation are opt-in per hotel. See the parser for the mirror comment.
    booking_modification_enabled: bool = False
    booking_cancellation_enabled: bool = False
    task_creation_enabled: bool = True
    # ON by default, for the same reason as human_escalation_enabled below: an unconfigured
    # hotel that cannot raise a ticket for a complaint has no way to route it anywhere, and the
    # guest gets a dead-end reply instead. A hotel that does not want tickets must opt out.
    support_ticket_enabled: bool = True
    # ON by default. A hotel that never configures this flag is a hotel whose guests have no
    # way to reach a human: the gate swaps the reply for "please contact our reception
    # directly" — copy that carries no phone number and no next step, so the guest asks "how"
    # and gets the identical sentence back. Failing closed here does not protect the hotel, it
    # strands the guest. A hotel that genuinely does not want escalation has to say so.
    human_escalation_enabled: bool = True
    document_understanding_enabled: bool = True
    vision_classification_enabled: bool = True
    id_verification_enabled: bool = True
    ota_comparison_enabled: bool = True
    document_ocr_enabled: bool = True
    staff_command_intelligence_enabled: bool = True
    same_day_booking_allowed: bool = True
    early_check_in_allowed: bool = False
    late_checkout_allowed: bool = False
    # Let a confirmed guest build a second booking while the first still stands.
    # ON by default because the flag exists to bound a change that already ships
    # on, and because failing closed here reproduces the complaint it answers: a
    # confirmed guest asking for another room was dragged back to `confirmed` on
    # every turn and could never start the booking. Its purpose is to be the
    # OPERABLE off-switch — it lives in the hotel's own Hotel-Settings document,
    # so the DB team can turn it off for one hotel without a deploy, which the
    # env flag in app/config/settings.py cannot do from this account.
    second_booking_alongside_enabled: bool = True
    has_mixed_ac_inventory: bool = True
    # Open Decisions (§12 / D1-D4) — Config-ready flags (Option A defaults preserved)
    default_language_mode: str = "english_default"      # D1: 'english_default' vs 'hinglish_for_plus91'
    greeting_style: str = "numbered_menu"              # D2: 'numbered_menu' vs 'open_greeting'
    meal_plan_display: str = "silent_ep_default"        # D3: 'silent_ep_default' vs 'show_ep_and_cp_side_by_side'
    advance_pct_mode: str = "fixed_30"                  # D4: 'fixed_30' vs 'per_property_config'


class BusinessLimits(BaseModel):
    """
    Business constraints enforced in Python validation.
    NEVER exposed to the AI model.
    """
    maximum_price_match_discount: float = 10.0
    maximum_adults_per_booking: int = 10
    maximum_rooms_per_booking: int = 5
    maximum_children_per_booking: int = 6
    group_threshold: int = 5
    # Child age thresholds (per-property config)
    child_free_age_limit: int = 5       # Age < 5 is free
    child_as_adult_age_limit: int = 12  # Age >= 12 is counted as adult
    # OPEN DECISION D4 — needs product sign-off, see spec §6: fixed 30% advance
    advance_percentage: float = 0.30
    # GST slabs (per room per night)
    gst_threshold_inr: float = 7500.0
    gst_rate_low: float = 5.0
    gst_rate_high: float = 18.0
    supported_booking_platforms: list[str] = Field(
        default_factory=lambda: ["booking.com", "agoda", "goibibo", "makemytrip", "expedia", "hotels.com"]
    )


class BusinessConfig(BaseModel):
    """
    All business logic configuration — enforced entirely in Python.
    NEVER injected into the AI prompt.
    """
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    limits: BusinessLimits = Field(default_factory=BusinessLimits)


class HotelConfig(BaseModel):
    """
    Complete hotel configuration container.

    Loaded from MongoDB and cached in memory.
    Only ai_config is exposed to Gemini — business_config stays in Python.
    """
    hotel_id: int
    hotel_name: str = "our hotel"
    ai_config: AIPromptConfig = Field(default_factory=AIPromptConfig)
    business_config: BusinessConfig = Field(default_factory=BusinessConfig)
    # Hotel-specific room aliases — keys are guest shorthand, values are canonical names
    # Example: {"dlx": "Deluxe Room", "exec": "Executive Suite"}
    room_aliases: dict[str, str] = Field(default_factory=dict)
    # Fuzzy match threshold for room alias resolver (0–100)
    room_match_threshold: float = 90.0

    @classmethod
    def from_mongo_doc(cls, doc: dict, hotel_id: int, hotel_name: str = "our hotel") -> HotelConfig:
        """
        Parse a MongoDB Hotel-Settings document into typed config.

        Handles both the current schema (automation_ai_settings style)
        and future expanded schemas gracefully.
        """
        if not doc:
            return cls(hotel_id=hotel_id, hotel_name=hotel_name)

        # ── AI Prompt Config ─────────────────────────────────
        behavior = doc.get("behavior") or {}
        if not isinstance(behavior, dict):
            logger.warning(f"CONFIG_VALIDATION_FAILED | hotel_id={hotel_id} | reason=behavior is not a dict (got {type(behavior).__name__})")
            behavior = {}

        system_prompt = doc.get("systemPrompt") or doc.get("system_prompt") or ""
        conversation_tone = (
            behavior.get("conversationTone")
            or behavior.get("ai_tone")
            or behavior.get("conversation_tone")
            or "professional"
        )
        response_length = behavior.get("responseLength") or behavior.get("response_length") or 40

        # Parse languages
        languages_raw = doc.get("languages") or {}
        supported_languages = [
            lang for lang, enabled in languages_raw.items()
            if enabled is True
        ]
        if not supported_languages:
            supported_languages = ["english"]

        # Parse custom instructions (dos & donts from ai_config)
        custom_instructions_raw = doc.get("custom_instructions") or doc.get("customInstructions") or {}
        custom_instructions = {}
        if isinstance(custom_instructions_raw, dict):
            dos = [str(x).strip() for x in custom_instructions_raw.get("dos", []) if str(x).strip()]
            donts = [str(x).strip() for x in custom_instructions_raw.get("donts", []) if str(x).strip()]
            if dos:
                custom_instructions["dos"] = dos
            if donts:
                custom_instructions["donts"] = donts

        persona_raw = doc.get("persona") or {}
        raw_b_name = persona_raw.get("bot_name") or persona_raw.get("name") or "Hotel Assistant"
        import re
        bot_name = re.sub(r'\b(stayinn\s*ai|stayinnai|ai|bot)\b', '', str(raw_b_name), flags=re.IGNORECASE).strip()
        bot_name = re.sub(r'\s+', ' ', bot_name).strip() or "Hotel Assistant"
        fallback_msg = persona_raw.get("fallback") or ""

        ai_config = AIPromptConfig(
            system_prompt=system_prompt,
            conversation_tone=conversation_tone,
            supported_languages=supported_languages,
            response_length=response_length,
            custom_instructions=custom_instructions,
            bot_name=str(bot_name),
            fallback_message=str(fallback_msg),
        )

        # ── Business Config ──────────────────────────────────
        features_raw = behavior.get("features") or {}

        def _safe_bool(key: str, data: dict, default: bool = False) -> bool:
            val = data.get(key, default)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower == "true":
                    return True
                if val_lower == "false":
                    return False
            logger.warning(f"CONFIG_VALIDATION_FAILED | hotel_id={hotel_id} | feature={key} | reason=Invalid value {val} (type {type(val)}) | Using default={default}")
            return default

        # Payment verification: True by default unless explicitly disabled
        payment_verification = True
        for k in ("paymentVerificationEnabled", "paymentVerification", "payment_verification_enabled", "payment_verification"):
            if k in features_raw:
                payment_verification = _safe_bool(k, features_raw, True)
                break

        # Document understanding: True by default unless explicitly disabled
        doc_understanding = True
        for k in ("documentUnderstandingEnabled", "documentUnderstanding", "document_understanding_enabled", "document_understanding"):
            if k in features_raw:
                doc_understanding = _safe_bool(k, features_raw, True)
                break

        feature_flags = FeatureFlags(
            ai_enabled=_safe_bool("ai_enabled", doc, _safe_bool("aiEnabled", doc, True)),
            direct_booking=_safe_bool("direct_booking", features_raw, _safe_bool("directBooking", features_raw, True)),
            lead_only_mode=_safe_bool("lead_only_mode", features_raw, _safe_bool("leadOnlyMode", features_raw, False)),
            dynamic_pricing=_safe_bool("dynamic_pricing", features_raw, _safe_bool("dynamicPricing", features_raw, True)),
            coupon_visibility=_safe_bool("coupon_visibility", features_raw, _safe_bool("couponVisibility", features_raw, True)),
            auto_upsell=_safe_bool("auto_upsell", features_raw, _safe_bool("autoUpsell", features_raw, False)),
            auto_upgrade=_safe_bool("auto_upgrade", features_raw, _safe_bool("autoUpgrade", features_raw, False)),
            price_match_enabled=_safe_bool("price_match_enabled", features_raw, _safe_bool("priceMatchEnabled", features_raw, True)),
            payment_verification_enabled=payment_verification,
            # These two are False here and False in FeatureFlags above — deliberately mirroring
            # rather than drifting. They used to disagree (True there, False here), so the same
            # hotel had different permissions depending on whether it resolved through this
            # parser or was built straight from the class. Opt-in is the chosen side.
            booking_modification_enabled=_safe_bool("booking_modification_enabled", features_raw, _safe_bool("bookingModificationEnabled", features_raw, False)),
            booking_cancellation_enabled=(
                _safe_bool("booking_cancellation_enabled", features_raw, False)
                or _safe_bool("bookingCancellationEnabled", features_raw, False)
                or _safe_bool("cancellationEnabled", features_raw, False)
                or _safe_bool("cancelBooking", features_raw, False)
            ),
            task_creation_enabled=_safe_bool("task_creation_enabled", features_raw, _safe_bool("taskCreationEnabled", features_raw, True)),
            # ON by default, matching the FeatureFlags field above — an unconfigured hotel must
            # be able to raise a ticket, or a complaint routes nowhere. Opt out explicitly.
            support_ticket_enabled=_safe_bool("support_ticket_enabled", features_raw, _safe_bool("supportTicketEnabled", features_raw, True)),
            # Innermost fallback is what a hotel with NO configured value gets, and it must match
            # the FeatureFlags default above. The outer _safe_bool only supplies its own default
            # when the snake_case key is absent; it then delegates to the inner result, so the
            # inner value is the one every unconfigured hotel actually receives.
            human_escalation_enabled=_safe_bool("human_escalation_enabled", features_raw, _safe_bool("humanEscalationEnabled", features_raw, True)),
            document_understanding_enabled=doc_understanding,
            vision_classification_enabled=_safe_bool("vision_classification_enabled", features_raw, _safe_bool("visionClassificationEnabled", features_raw, True)),
            id_verification_enabled=_safe_bool("id_verification_enabled", features_raw, _safe_bool("idVerificationEnabled", features_raw, True)),
            ota_comparison_enabled=_safe_bool("ota_comparison_enabled", features_raw, _safe_bool("otaComparisonEnabled", features_raw, True)),
            document_ocr_enabled=_safe_bool("document_ocr_enabled", features_raw, _safe_bool("documentOcrEnabled", features_raw, True)),
            staff_command_intelligence_enabled=_safe_bool("staff_command_intelligence_enabled", features_raw, _safe_bool("staffCommandIntelligenceEnabled", features_raw, True)),
            same_day_booking_allowed=_safe_bool("same_day_booking_allowed", features_raw, _safe_bool("sameDayBookingAllowed", features_raw, True)),
            early_check_in_allowed=_safe_bool("early_check_in_allowed", features_raw, _safe_bool("earlyCheckInAllowed", features_raw, False)),
            late_checkout_allowed=_safe_bool("late_checkout_allowed", features_raw, _safe_bool("lateCheckoutAllowed", features_raw, False)),
            has_mixed_ac_inventory=_safe_bool("has_mixed_ac_inventory", features_raw, _safe_bool("hasMixedAcInventory", features_raw, _safe_bool("has_mixed_ac_inventory", doc, True))),
            # ON by default, mirroring the FeatureFlags field above. This is the
            # operable off-switch for the second-booking latch: the DB team can
            # set it false in one hotel's Hotel-Settings document and the latch
            # stops for that hotel within the 5-minute config cache, with no
            # deploy and no Cloud Run access.
            second_booking_alongside_enabled=_safe_bool("second_booking_alongside_enabled", features_raw, _safe_bool("secondBookingAlongsideEnabled", features_raw, True)),
        )

        limits_raw = doc.get("limits") or doc.get("businessLimits") or {}
        business_limits = BusinessLimits(
            maximum_price_match_discount=float(limits_raw.get("maximumPriceMatchDiscount", 10.0)),
            maximum_adults_per_booking=int(limits_raw.get("maximumAdultsPerBooking", 10)),
            maximum_rooms_per_booking=int(limits_raw.get("maximumRoomsPerBooking", 5)),
            maximum_children_per_booking=int(limits_raw.get("maximumChildrenPerBooking", 6)),
            group_threshold=int(limits_raw.get("groupThreshold") or limits_raw.get("group_threshold") or doc.get("group_threshold") or 5),
            child_free_age_limit=int(limits_raw.get("childFreeAgeLimit") or limits_raw.get("child_free_age_limit") or doc.get("child_free_age_limit") or 10),
            child_as_adult_age_limit=int(limits_raw.get("childAsAdultAgeLimit") or limits_raw.get("child_as_adult_age_limit") or doc.get("child_as_adult_age_limit") or 11),
            advance_percentage=float(limits_raw.get("advancePercentage") or limits_raw.get("advance_percentage") or doc.get("advance_percentage") or 0.30),
            gst_threshold_inr=float(limits_raw.get("gstThresholdInr") or limits_raw.get("gst_threshold_inr") or doc.get("gst_threshold_inr") or 7500.0),
            gst_rate_low=float(limits_raw.get("gstRateLow") or limits_raw.get("gst_rate_low") or doc.get("gst_rate_low") or 5.0),
            gst_rate_high=float(limits_raw.get("gstRateHigh") or limits_raw.get("gst_rate_high") or doc.get("gst_rate_high") or 18.0),
            supported_booking_platforms=limits_raw.get(
                "supportedBookingPlatforms",
                ["booking.com", "agoda", "goibibo", "makemytrip", "expedia", "hotels.com"],
            ),
        )

        business_config = BusinessConfig(
            features=feature_flags,
            limits=business_limits,
        )

        # Room aliases (hotel-specific shorthand → canonical room name)
        aliases_raw = (
            doc.get("room_aliases")
            or doc.get("roomAliases")
            or behavior.get("room_aliases")
            or behavior.get("roomAliases")
            or {}
        )
        room_aliases: dict[str, str] = {}
        if isinstance(aliases_raw, dict):
            for k, v in aliases_raw.items():
                if k is not None and v is not None and str(k).strip() and str(v).strip():
                    room_aliases[str(k).strip()] = str(v).strip()

        threshold_raw = (
            doc.get("room_match_threshold")
            or doc.get("roomMatchThreshold")
            or behavior.get("room_match_threshold")
            or 90.0
        )
        try:
            room_match_threshold = float(threshold_raw)
        except (TypeError, ValueError):
            room_match_threshold = 90.0

        return cls(
            hotel_id=hotel_id,
            hotel_name=hotel_name,
            ai_config=ai_config,
            business_config=business_config,
            room_aliases=room_aliases,
            room_match_threshold=room_match_threshold,
        )
