import logging
from pymongo import ASCENDING
from app.db.mongodb import (
    hotel_customization_collection,
    hotel_personality_modes_collection,
    hotel_rules_collection,
)
from app.models.personalization_schemas import (
    HotelCustomization,
    HotelPersonalityMode,
    HotelRulesDoc,
    HotelRule,
)

logger = logging.getLogger(__name__)

async def ensure_indexes() -> None:
    cust_col = hotel_customization_collection()
    modes_col = hotel_personality_modes_collection()
    rules_col = hotel_rules_collection()

    await cust_col.create_index([("hotel_id", ASCENDING)], unique=True)
    await modes_col.create_index([("hotel_id", ASCENDING), ("mode_id", ASCENDING)], unique=True)
    await modes_col.create_index([("hotel_id", ASCENDING), ("is_default", ASCENDING)])
    await rules_col.create_index([("hotel_id", ASCENDING)], unique=True)
    await rules_col.create_index([("hotel_id", ASCENDING), ("rules.active", ASCENDING), ("rules.priority", ASCENDING)])

async def _seed_hotel_customization(hotel_id: str) -> HotelCustomization:
    logger.warning(f"Hotel {hotel_id} is running on default personalization and has not been customized.")
    
    # 1. Customization
    cust = HotelCustomization(hotel_id=hotel_id)
    
    # 2. Modes
    modes = [
        HotelPersonalityMode(
            hotel_id=hotel_id,
            mode_id="default",
            mode_name="Default",
            is_default=True,
            trigger_phrases=["default mode", "normal mode"],
            prompt_overlay="",
            switch_acknowledgment="Returning to default mode."
        ),
        HotelPersonalityMode(
            hotel_id=hotel_id,
            mode_id="human",
            mode_name="Human-Like",
            trigger_phrases=["behave like human", "talk normally", "stop being a bot", "be natural", "ek insaan ki tarah baat kar", "robot mat ban", "talk like a human"],
            prompt_overlay="Respond as if you are a warm, witty human hotel receptionist. Use contractions, occasional light humor, natural pauses like hmm or sure!. Never break character by mentioning AI or chatbot.",
            switch_acknowledgment="Sure! I'll keep it real with you from here. What do you need?",
            tone_overrides={"formality_level": 2, "emoji_enabled": True, "humor_allowed": True}
        ),
        HotelPersonalityMode(
            hotel_id=hotel_id,
            mode_id="formal",
            mode_name="Formal Concierge",
            trigger_phrases=["be professional", "speak formally", "formal mode"],
            prompt_overlay="Respond as a highly trained, professional hotel concierge. Use complete sentences, elegant phrasing, and utmost respect.",
            switch_acknowledgment="Certainly. I shall communicate formally. How may I assist you?",
            tone_overrides={"formality_level": 5, "emoji_enabled": False, "humor_allowed": False}
        ),
        HotelPersonalityMode(
            hotel_id=hotel_id,
            mode_id="sales",
            mode_name="Friendly Sales",
            trigger_phrases=["sales mode"],
            prompt_overlay="Proactively suggest upgrades and highlight the best premium options. Frame everything positively and create a gentle sense of urgency.",
            switch_acknowledgment="Let's find you the absolute best stay possible!"
        )
    ]
    
    # 3. Rules
    rules = [
        HotelRule(rule_id="R001", type="dont", category="competitor", rule_text="Never name competitor hotels, OTAs like OYO or Booking.com. If asked, say direct booking gives the best rate.", priority=1),
        HotelRule(rule_id="R002", type="dont", category="pricing", rule_text="Never quote a price from memory or estimate. Always use the exact figure from the booking engine.", priority=1),
        HotelRule(rule_id="R003", type="dont", category="legal", rule_text="Never make promises about refund timelines or legal rights. Escalate any legal questions to staff.", priority=1),
        HotelRule(rule_id="R004", type="do", category="escalation", rule_text="If a guest says agent, human, manager, staff, or front desk — immediately trigger escalation.", priority=1),
        HotelRule(rule_id="R005", type="do", category="upsell", rule_text="Always mention breakfast package availability once during the booking confirmation stage.", priority=3),
        HotelRule(rule_id="R006", type="dont", category="content", rule_text="Never discuss hotel ownership, internal staff matters, or operational issues with guests.", priority=2),
        HotelRule(rule_id="R007", type="do", category="service", rule_text="If a guest mentions a special occasion (anniversary, birthday, honeymoon), acknowledge it warmly and mention that the team can arrange special setups.", priority=3)
    ]
    rules_doc = HotelRulesDoc(hotel_id=hotel_id, rules=rules)

    # Insert
    await hotel_customization_collection().insert_one(cust.model_dump())
    await hotel_personality_modes_collection().insert_many([m.model_dump() for m in modes])
    await hotel_rules_collection().insert_one(rules_doc.model_dump())
    
    return cust

def _get_variants(hotel_id: str | int) -> list[str | int]:
    str_hid = str(hotel_id).strip()
    variants: list[str | int] = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))
    return list(dict.fromkeys(variants))


async def get_hotel_customization(hotel_id: str | int) -> HotelCustomization:
    variants = _get_variants(hotel_id)
    doc = await hotel_customization_collection().find_one({"hotel_id": {"$in": variants}})
    if not doc:
        cust = await _seed_hotel_customization(str(hotel_id))
    else:
        doc.pop("_id", None)
        if "hotel_id" in doc:
            doc["hotel_id"] = str(doc["hotel_id"])
        cust = HotelCustomization(**doc)

    # Bridge Control Center configs (ai_config / chatbot_config / hotel_settings / hotels)
    try:
        from app.db.mongodb import get_database
        import re
        db = get_database()
        ai_doc = await db["ai_config"].find_one({"hotel_id": {"$in": variants}})
        cb_doc = await db["chatbot_config"].find_one({"hotel_id": {"$in": variants}})
        hotel_doc = await db["hotels"].find_one({"hotel_id": {"$in": variants}}) or await db["hotels"].find_one({"id": {"$in": variants}})
        settings_doc = await db["hotel_settings"].find_one({"hotel_id": {"$in": variants}})

        b_name = (hotel_doc.get("name") if hotel_doc else None) or (settings_doc.get("business", {}).get("name") if settings_doc else None)
        hotel_brand = b_name.title() if b_name else "Hotel"

        persona_src = (ai_doc.get("persona") if ai_doc else None) or cb_doc or {}
        raw_name = persona_src.get("bot_name") or persona_src.get("name") or cust.bot_persona.name

        # Strip 'AI', 'Bot', 'StayInn AI' terms to support white-label control center personalization
        clean_name = re.sub(r'\b(stayinn\s*ai|stayinnai|stayinn|ai|bot)\b', '', str(raw_name), flags=re.IGNORECASE).strip()
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        if not clean_name or clean_name.lower() in ("hotel assistant", "assistant"):
            cust.bot_persona.name = f"{hotel_brand} Assistant"
        else:
            cust.bot_persona.name = clean_name

        if persona_src:
            allow_emojis = persona_src.get("allow_emojis")
            if allow_emojis is not None:
                cust.tone_profile.emoji_enabled = bool(allow_emojis)
                if cust.tone_profile.emoji_enabled and cust.tone_profile.emoji_density == "none":
                    cust.tone_profile.emoji_density = "light"

            resp_len = persona_src.get("response_length")
            if resp_len in ("brief", "balanced", "detailed"):
                cust.tone_profile.response_length = resp_len
            elif resp_len == "short":
                cust.tone_profile.response_length = "brief"
            elif resp_len in ("long", "medium"):
                cust.tone_profile.response_length = "detailed" if resp_len == "long" else "balanced"

        # Sync conversation tone from ai_config behavior / onboarding
        if ai_doc:
            beh = ai_doc.get("behavior") or {}
            tone_name = str(beh.get("conversation_tone") or ai_doc.get("onboarding", {}).get("ai_tone") or "").lower()
            if tone_name in ("luxury", "formal", "professional", "casual", "warm"):
                if tone_name == "formal":
                    cust.tone_profile.formality_level = 5
                elif tone_name in ("casual", "warm"):
                    cust.tone_profile.formality_level = 2
                    cust.tone_profile.humor_allowed = True
                elif tone_name == "luxury":
                    cust.bot_persona.property_type = "luxury"
                    cust.tone_profile.formality_level = 4
                elif tone_name == "professional":
                    cust.tone_profile.formality_level = 3

        # Clean generic / legacy AI greeting if not customized
        sig = cust.bot_persona.signature_greeting
        if any(bad in sig.lower() for bad in ("stayinn", "stayinnai", "ai bot", "hotel assistant")):
            cust.bot_persona.signature_greeting = f"Namaste! 🙏 Welcome to {hotel_brand}. How may I assist you today?"

        # Bridge system_prompt from Control Center
        sys_prompt = (ai_doc.get("system_prompt") if ai_doc else None) or (cb_doc.get("description") if cb_doc else None)
        if sys_prompt:
            cleaned_sys = re.sub(r'\b(stayinn\s*ai|stayinnai)\b', hotel_brand, str(sys_prompt), flags=re.IGNORECASE)
            cust.system_prompt = cleaned_sys
    except Exception as e:
        logger.warning(f"Failed to merge ai_config into hotel_customization: {e}")

    return cust


async def get_personality_modes(hotel_id: str | int) -> list[HotelPersonalityMode]:
    variants = _get_variants(hotel_id)
    docs = await hotel_personality_modes_collection().find({"hotel_id": {"$in": variants}, "active": True}).to_list(length=100)
    modes = []
    for doc in docs:
        d = {k: v for k, v in doc.items() if k != "_id"}
        if "hotel_id" in d:
            d["hotel_id"] = str(d["hotel_id"])
        modes.append(HotelPersonalityMode(**d))
    return modes


async def get_hotel_rules(hotel_id: str | int) -> list[HotelRule]:
    variants = _get_variants(hotel_id)
    doc = await hotel_rules_collection().find_one({"hotel_id": {"$in": variants}})
    rules: list[HotelRule] = []
    if doc:
        cleaned = {k: v for k, v in doc.items() if k != "_id"}
        if "hotel_id" in cleaned:
            cleaned["hotel_id"] = str(cleaned["hotel_id"])
        rules_doc = HotelRulesDoc(**cleaned)
        rules_doc.rules.sort(key=lambda r: r.priority)
        rules.extend([r for r in rules_doc.rules if r.active])

    # Also merge custom_instructions (dos and donts) from ai_config if present
    try:
        from app.db.mongodb import get_database
        ai_doc = await get_database()["ai_config"].find_one({"hotel_id": {"$in": variants}})
        if ai_doc and ai_doc.get("custom_instructions"):
            c_inst = ai_doc["custom_instructions"]
            if isinstance(c_inst, dict):
                existing_texts = {r.rule_text.lower() for r in rules}
                GREETING_STOPWORDS = ("always greet", "greet the guest", "mention hotel name in first response")
                for do_text in c_inst.get("dos", []):
                    clean = str(do_text).strip()
                    if any(sw in clean.lower() for sw in GREETING_STOPWORDS):
                        continue
                    if clean and clean.lower() not in existing_texts:
                        rules.append(HotelRule(rule_id=f"AI_DO_{len(rules)+1}", type="do", category="custom", rule_text=clean, priority=1))
                        existing_texts.add(clean.lower())
                for dont_text in c_inst.get("donts", []):
                    clean = str(dont_text).strip()
                    if clean and clean.lower() not in existing_texts:
                        rules.append(HotelRule(rule_id=f"AI_DONT_{len(rules)+1}", type="dont", category="custom", rule_text=clean, priority=1))
                        existing_texts.add(clean.lower())
    except Exception as e:
        logger.warning(f"Failed to load custom instructions from ai_config: {e}")

    return rules
