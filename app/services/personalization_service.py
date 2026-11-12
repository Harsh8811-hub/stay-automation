import logging
import time
from typing import Optional

from app.models.personalization_schemas import PersonalizationContext, HotelPersonalityMode, HotelCustomization
from app.repositories.personalization_repository import get_hotel_customization, get_personality_modes, get_hotel_rules

logger = logging.getLogger(__name__)

# TTL Cache dictionary
# Format: {hotel_id: {"context": PersonalizationContext, "timestamp": float}}
_cache = {}
CACHE_TTL = 30  # 30 seconds for quick reflection of Control Center changes


def invalidate_personalization_cache(hotel_id: Optional[str | int] = None) -> None:
    """Clear in-memory personalization cache for a hotel or all hotels."""
    if hotel_id is not None:
        _cache.pop(str(hotel_id), None)
        if str(hotel_id).isdigit():
            _cache.pop(int(hotel_id), None)
    else:
        _cache.clear()


class PersonalizationService:
    async def load_context(self, hotel_id: str) -> PersonalizationContext:
        now = time.time()
        if hotel_id in _cache:
            entry = _cache[hotel_id]
            if now - entry["timestamp"] < CACHE_TTL:
                return entry["context"]
        
        # Cache miss
        cust = await get_hotel_customization(hotel_id)
        modes = await get_personality_modes(hotel_id)
        rules = await get_hotel_rules(hotel_id)
        
        default_mode = next((m for m in modes if m.is_default), modes[0] if modes else None)
            
        context = PersonalizationContext(
            customization=cust,
            active_mode=default_mode, # actual active mode is computed per session
            rules=rules
        )
        
        # Save to cache
        _cache[hotel_id] = {"context": context, "timestamp": now}
        return context

    async def get_active_mode(self, hotel_id: str, session: dict) -> HotelPersonalityMode:
        context = await self.load_context(hotel_id)
        modes = await get_personality_modes(hotel_id)
        
        session_mode_id = session.get("active_personality_mode")
        target_mode_id = session_mode_id or context.customization.default_personality_mode
        
        active_mode = next((m for m in modes if m.mode_id == target_mode_id), None)
        if not active_mode:
            active_mode = next((m for m in modes if m.is_default), modes[0] if modes else None)
            
        return active_mode

    async def detect_mode_switch(self, message: str, hotel_id: str) -> Optional[HotelPersonalityMode]:
        words = message.split()
        if len(words) >= 15:
            return None # Length gate prevents false positives mid-booking
            
        msg_lower = message.lower().strip()
        modes = await get_personality_modes(hotel_id)
        
        # Tier 1: Substring match
        for mode in modes:
            for trigger in mode.trigger_phrases:
                if trigger.lower().strip() in msg_lower:
                    return mode
                    
        # Tier 2: Fuzzy match (>= 65% overlap)
        msg_tokens = set(msg_lower.split())
        for mode in modes:
            for trigger in mode.trigger_phrases:
                trigger_tokens = set(trigger.lower().split())
                if not trigger_tokens:
                    continue
                intersection = len(msg_tokens.intersection(trigger_tokens))
                if intersection / len(trigger_tokens) >= 0.65:
                    return mode
        
        # Tier 3: LLM-based classification for arbitrary persona requests
        # Only invoke if message contains behavior-related keywords (cost gate)
        behavior_keywords = {
            "behave", "behavior", "behaviour", "talk", "speak", "act", "mode",
            "style", "tone", "personality", "formal", "casual", "friendly",
            "professional", "human", "robot", "bot", "natural", "pirate",
            "baat", "tarah", "jaise", "bolo", "bol", "change"
        }
        if msg_tokens.intersection(behavior_keywords):
            result = await self._llm_classify_persona(msg_lower, modes, hotel_id)
            if result:
                return result
                    
        return None

    async def _llm_classify_persona(self, message: str, modes: list, hotel_id: str) -> Optional[HotelPersonalityMode]:
        """Tier 3: Use a small focused Gemini call to classify arbitrary persona requests."""
        try:
            from app.services.llm import compat as genai
            from app.config.settings import get_settings
            
            settings = get_settings()
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(settings.GEMINI_MODEL)
            
            mode_descriptions = "\n".join([
                f"- mode_id: \"{m.mode_id}\" | name: \"{m.mode_name}\" | description: \"{m.prompt_overlay}\""
                for m in modes if not m.is_default
            ])
            
            prompt = f"""Analyze this message from a hotel chatbot user. Is the user asking the bot to change its personality, behavior, or communication style?

MESSAGE: "{message}"

AVAILABLE PERSONALITY MODES:
{mode_descriptions}

If the user IS asking for a behavior/personality/style change, respond with the mode_id of the BEST matching mode from the list above.
If the user is NOT asking for a behavior change (e.g., they're asking a hotel question, booking, FAQ), respond with "NONE".

IMPORTANT:
- "talk like a human" or "stop being a bot" → match to the closest human/casual mode
- "be professional" or "formal please" → match to formal mode
- "be more friendly" or "act casual" → match to the most casual/human mode
- Normal hotel questions like "do you have rooms?" → NONE
- Booking requests → NONE

Respond with ONLY the mode_id or NONE. No explanation."""

            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=50,
                ),
            )
            
            if response and response.text:
                result = response.text.strip().strip('"').strip("'").lower()
                if result == "none":
                    return None
                # Find the matching mode
                matched = next((m for m in modes if m.mode_id.lower() == result), None)
                if matched:
                    logger.info(f"LLM_PERSONA_MATCH | message='{message}' | matched_mode={matched.mode_name}")
                    return matched
                    
            return None
        except Exception as e:
            logger.warning(f"LLM persona classification failed: {e}")
            return None

    def validate_rules(self, rules: list) -> list:
        # Strip rules that attempt to override deterministic logic
        valid_rules = []
        forbidden_keywords = [
            "razorpay", "stripe", "never escalate", "do not escalate", 
            "override price", "custom price", "ignore booking engine"
        ]
        for r in rules:
            text_lower = r.rule_text.lower()
            if any(k in text_lower for k in forbidden_keywords):
                logger.warning(f"Stripped hotel rule due to protected keyword violation: {r.rule_text}")
                continue
            valid_rules.append(r)
        return valid_rules

    def _build_persona_block(self, cust: HotelCustomization, active_mode: HotelPersonalityMode) -> str:
        property_type_desc = {
            "luxury": "You represent a premium luxury property. Maintain impeccable composure. Never rush a guest. Always offer alternatives before declining any request. Avoid informal phrases, abbreviations, or emoji entirely unless hotel config specifies otherwise.",
            "budget": "You represent a value-focused property where efficiency and friendliness matter most. Keep responses concise. Guests are often price-conscious — be direct about pricing and don't over-explain. Light warmth is appreciated; formality is not.",
            "heritage": "You represent a heritage or legacy property with a rich history. Weave in the property's character and story naturally. Guests choosing heritage properties value authenticity — avoid corporate-sounding language.",
            "family": "You represent a family-friendly property. Always check for children's ages when guests mention kids — this affects room allocation, facilities, and meal options. Use warm, inclusive language that respects both adults and acknowledges children as guests too.",
            "boutique": "You represent a boutique property where personalized attention is the core differentiator. Remember and reference details the guest mentions. Make them feel they are speaking to someone who genuinely knows the property and cares about their specific visit.",
            "resort": "You represent a resort property where the experience extends far beyond the room. Proactively surface activity, spa, dining, and leisure options. Guests at resorts often want to be inspired about their stay, not just check availability.",
            "business": "You represent a business hotel. Prioritize speed, efficiency, and clarity. Emphasize connectivity, meeting facilities, and express services."
        }
        
        type_desc = property_type_desc.get(cust.bot_persona.property_type, property_type_desc["budget"])
        
        usp = ", ".join(cust.usp_highlights) if cust.usp_highlights else "None"
        suppressed = ", ".join(cust.suppressed_topics) if cust.suppressed_topics else "None"
        
        block = f"[HOTEL PERSONA]\nYour name is {cust.bot_persona.name}. {type_desc}\nTarget Guest Profile: {cust.bot_persona.target_guest_profile}\nYour USPs to highlight naturally when relevant: {usp}\nTopics you must never raise or discuss: {suppressed}\n"
        return block

    def _build_style_block(self, cust: HotelCustomization, active_mode: HotelPersonalityMode | None) -> str:
        # `get_active_mode` returns None when the hotel has no personality-mode
        # rows at all. Dereferencing that killed the whole turn — every guest got
        # "I'm sorry, I encountered a brief issue processing that" — so a missing
        # mode now falls back to the hotel's own tone profile, which is what the
        # per-field fallbacks below already do when a mode leaves a field unset.
        # Measured 2026-10-25: three different eval cases died here across three
        # 124-case runs, and the same session sometimes took the other branch, so
        # the null arrives intermittently rather than for one fixed hotel.
        overrides = active_mode.tone_overrides if active_mode else None
        formality = (overrides.formality_level if overrides else None) or cust.tone_profile.formality_level
        emoji_enabled = overrides.emoji_enabled if overrides and overrides.emoji_enabled is not None else cust.tone_profile.emoji_enabled
        resp_length = (overrides.response_length if overrides else None) or cust.tone_profile.response_length
        humor = overrides.humor_allowed if overrides and overrides.humor_allowed is not None else cust.tone_profile.humor_allowed
        
        emoji_desc = f"Enabled ({cust.tone_profile.emoji_density})" if emoji_enabled else "Disabled"
        honorifics = "Use honorifics (Sir, Ma'am, ji) appropriately." if cust.tone_profile.honorifics_enabled else "Do not use honorifics unnecessarily."
        humor_desc = "Light humor is allowed." if humor else "Keep it strictly professional, no humor."
        
        block = f"[COMMUNICATION STYLE]\nFormality level: {formality}/5\nResponse length: {resp_length}\nEmoji: {emoji_desc}\nLanguage: Respond primarily in {cust.language_config.primary_language}. Fallback: {cust.language_config.fallback_language}.\n{f'Regional flavor: {cust.language_config.regional_flavor}' if cust.language_config.regional_flavor else ''}\n{honorifics}\n{humor_desc}\n"
        return block

    def _build_rules_block(self, rules: list) -> str:
        valid_rules = self.validate_rules(rules)
        if not valid_rules:
            return ""
            
        block = "[HOTEL RULES — FOLLOW THESE EXACTLY]\n"
        for i, r in enumerate(valid_rules):
            prefix = "[DO]" if r.type == "do" else "[DO NOT]"
            block += f"{i+1}. {prefix} {r.rule_text}\n"
        return block

    def build_system_prompt(self, base_prompt: str, context: PersonalizationContext, active_mode: HotelPersonalityMode) -> str:
        persona_block = self._build_persona_block(context.customization, active_mode)
        style_block = self._build_style_block(context.customization, active_mode)
        rules_block = self._build_rules_block(context.rules)

        continuity_block = (
            "[CONVERSATION CONTINUITY & GREETINGS RULE — CRITICAL]\n"
            "In ongoing conversation turns, NEVER start your response with greetings like 'Hi', 'Hello', 'Namaste', 'Hey', or self-introductions like 'I am Hotel Assistant' / 'StayInn AI'.\n"
            "Never refer to yourself as an AI or mention StayInn.\n"
            "Dive directly into answering the question or fulfilling the guest's request without repeating pleasantries.\n"
            "The only exception is if the customer explicitly changes their booking dates, in which case acknowledge with a brief 'Hello!' or 'Hi!'.\n"
        )

        custom_prompt_block = ""
        if context.customization.system_prompt:
            custom_prompt_block = f"[HOTEL CUSTOM INSTRUCTIONS FROM CONTROL CENTER]\n{context.customization.system_prompt}\n\n"

        mode_block = ""
        if active_mode and active_mode.prompt_overlay:
            mode_block = f"[ACTIVE PERSONALITY MODE: {active_mode.mode_name}]\n{active_mode.prompt_overlay}\n"

        return f"[BASE HOTEL BOT INSTRUCTIONS]\n{base_prompt}\n\n{persona_block}\n{style_block}\n{rules_block}\n{continuity_block}\n{custom_prompt_block}{mode_block}"

personalization_service = PersonalizationService()
