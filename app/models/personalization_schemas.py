from datetime import datetime, UTC
from typing import Literal, Optional
from pydantic import BaseModel, Field


class BotPersona(BaseModel):
    name: str = "Hotel Assistant"
    property_type: Literal[
        "luxury",
        "boutique",
        "budget",
        "heritage",
        "resort",
        "business",
        "family"
    ] = "budget"

    target_guest_profile: str = "all travelers"

    signature_greeting: str = (
        "Namaste! 🙏 Welcome to our hotel. How may I assist you today?"
    )

    signature_sign_off: str = "Looking forward to hosting you!"


class ToneProfile(BaseModel):
    formality_level: int = Field(default=3, ge=1, le=5)
    response_length: Literal["brief", "balanced", "detailed"] = "balanced"
    emoji_enabled: bool = False
    emoji_density: Literal["none", "light", "moderate"] = "none"
    honorifics_enabled: bool = False
    humor_allowed: bool = False
    empathy_level: Literal["high", "medium", "standard"] = "standard"
    upsell_aggressiveness: Literal["passive", "moderate", "proactive"] = "passive"


class LanguageConfig(BaseModel):
    primary_language: str = "hinglish"
    fallback_language: str = "en"
    regional_flavor: Optional[
        Literal["gujarat", "rajasthan", "mumbai", "delhi"]
    ] = None
    use_guest_name: bool = True


class HotelCustomization(BaseModel):
    hotel_id: str
    bot_persona: BotPersona = Field(default_factory=BotPersona)
    tone_profile: ToneProfile = Field(default_factory=ToneProfile)
    language_config: LanguageConfig = Field(default_factory=LanguageConfig)
    usp_highlights: list[str] = Field(default_factory=list)
    suppressed_topics: list[str] = Field(default_factory=list)
    default_personality_mode: str = "default"
    system_prompt: Optional[str] = None
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToneOverrides(BaseModel):
    formality_level: Optional[int] = Field(default=None, ge=1, le=5)
    emoji_enabled: Optional[bool] = None
    response_length: Optional[
        Literal["brief", "balanced", "detailed"]
    ] = None
    humor_allowed: Optional[bool] = None


class HotelPersonalityMode(BaseModel):
    hotel_id: str
    mode_id: str
    mode_name: str
    is_default: bool = False
    trigger_phrases: list[str] = Field(default_factory=list)
    prompt_overlay: str
    tone_overrides: ToneOverrides = Field(default_factory=ToneOverrides)
    switch_acknowledgment: str
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HotelRule(BaseModel):
    rule_id: str
    type: Literal["do", "dont"]
    category: Literal[
        "competitor",
        "pricing",
        "upsell",
        "escalation",
        "content",
        "language",
        "service",
        "legal",
        "custom",
    ] | str
    rule_text: str
    priority: int = Field(default=3, ge=1, le=5)
    applies_to_stages: list[str] = Field(
        default_factory=lambda: ["all"]
    )
    active: bool = True


class HotelRulesDoc(BaseModel):
    hotel_id: str
    rules: list[HotelRule] = Field(default_factory=list)
    last_updated_by: str = "system"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PersonalizationContext(BaseModel):
    customization: HotelCustomization
    active_mode: HotelPersonalityMode
    rules: list[HotelRule]