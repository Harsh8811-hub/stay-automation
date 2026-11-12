"""
Prompt Builder — assembles minimal Gemini prompts from AI configuration only.

Design Principle:
    The AI receives ONLY what it needs for language generation:
    - Base system prompt (intent classification rules)
    - Hotel-specific system prompt
    - Conversation tone
    - Supported languages

    It NEVER receives:
    - Feature flags
    - Business limits
    - Pricing rules
    - Security configuration
    - Workflow routing logic

This ensures:
    - Lower token usage
    - Lower API cost
    - Faster inference
    - More deterministic behavior
    - Reduced hallucination
    - Simpler prompts
"""

from __future__ import annotations

import logging

from app.config.hotel_config import AIPromptConfig

logger = logging.getLogger(__name__)

# ── Tone Instruction Map ─────────────────────────────────────────
_TONE_INSTRUCTIONS = {
    "professional": "Maintain a professional, courteous, and efficient tone.",
    "friendly": "Be warm, approachable, and conversational. Use a friendly tone.",
    "formal": "Use formal language. Be polite, respectful, and structured.",
    "casual": "Be relaxed, informal, and approachable. Keep it conversational.",
    "luxury": "Use elegant, refined language befitting a luxury hospitality experience.",
}


def build_understand_prompt_header(ai_config: AIPromptConfig) -> str:
    """
    Build the dynamic header portion of the understanding prompt.

    This prepends hotel-specific AI configuration to the base
    system prompt (intent classification rules remain static).

    Only includes information that affects language generation:
    - Hotel system prompt (identity)
    - Conversation tone
    - Supported languages

    Args:
        ai_config: The AI prompt configuration from hotel settings.

    Returns:
        A prompt header string to prepend to the base system prompt.
    """
    parts: list[str] = []

    # 1. Hotel-specific system prompt (identity)
    if ai_config.system_prompt:
        parts.append(f"HOTEL CONTEXT:\n{ai_config.system_prompt}")

    # 2. Conversation tone
    tone = ai_config.conversation_tone.lower()
    tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS["professional"])
    parts.append(f"TONE: {tone_instruction}")

    # 3. Supported languages (hints only — always mirror the guest's language)
    if ai_config.supported_languages:
        langs = ", ".join(lang.capitalize() for lang in ai_config.supported_languages)
        parts.append(
            f"PREFERRED LANGUAGES: {langs}. "
            "Always detect and follow the guest's actual language "
            "(including Hindi, Kannada, Telugu, Tamil, etc.) even if not listed."
        )

    return "\n\n".join(parts)


def build_response_prompt_header(ai_config: AIPromptConfig, hotel_name: str = "our hotel") -> str:
    """
    Build the dynamic header for the response generation prompt.

    Used when Gemini generates natural language responses (FAQ, complex replies).

    Args:
        ai_config: The AI prompt configuration.
        hotel_name: Hotel name for personalization.

    Returns:
        A prompt header string for response generation.
    """
    parts: list[str] = []

    # Hotel identity
    if ai_config.system_prompt:
        parts.append(f"You are the AI assistant for {hotel_name}.")
    else:
        parts.append(f"You are a friendly, professional hotel assistant for {hotel_name}.")

    # Tone
    tone = ai_config.conversation_tone.lower()
    tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS["professional"])
    parts.append(tone_instruction)

    # Languages (hints only — always mirror the guest's language)
    if ai_config.supported_languages:
        langs = ", ".join(lang.capitalize() for lang in ai_config.supported_languages)
        parts.append(
            f"Preferred languages: {langs}. "
            "Always reply in the guest's language (including Hindi, Kannada, Telugu, etc.)."
        )

    return " ".join(parts)


def build_faq_prompt_header(ai_config: AIPromptConfig, hotel_name: str = "our hotel") -> str:
    """
    Build the dynamic header for FAQ response generation.

    Args:
        ai_config: The AI prompt configuration.
        hotel_name: Hotel name for personalization.

    Returns:
        A prompt header string for FAQ responses.
    """
    tone = ai_config.conversation_tone.lower()
    tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS["professional"])

    return f"You are a helpful hotel assistant for {hotel_name}. {tone_instruction}"
