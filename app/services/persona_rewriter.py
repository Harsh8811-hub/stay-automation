"""
Persona Rewriter — rewrites template/static responses to match the active personality mode.

Design:
- Only invoked when a NON-DEFAULT persona is active (zero overhead for default mode).
- Uses a single low-temperature Gemini call to adapt tone/style while preserving all facts.
- Called by response_generator and faq_handler before returning template text.
"""

import logging
from app.services.llm import compat as genai

from app.config.settings import get_settings
from app.models.personalization_schemas import HotelPersonalityMode

logger = logging.getLogger(__name__)

# Reuse the same model instance from gemini_service
_model = None

def _get_model():
    global _model
    if _model is None:
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _model = genai.GenerativeModel(settings.GEMINI_MODEL)
    return _model


async def rewrite_for_persona(template_text: str, active_mode: HotelPersonalityMode) -> str:
    """
    Rewrite a template response to match the active personality mode.
    
    Returns the original text unchanged if:
    - The mode is the default mode (zero overhead)
    - The Gemini call fails (graceful fallback)
    """
    # Skip rewriting for default mode — no extra Gemini call needed
    if active_mode.is_default or not active_mode.prompt_overlay:
        return template_text

    # Skip very short texts (e.g., "✅ Payment received!")
    if len(template_text.strip()) < 15:
        return template_text

    model = _get_model()

    prompt = f"""You are a tone adapter for a hotel chatbot. Rewrite this message to match the specified personality.

PERSONALITY: {active_mode.mode_name}
STYLE INSTRUCTIONS: {active_mode.prompt_overlay}
FORMALITY LEVEL: {active_mode.tone_overrides.formality_level or 'default'}/5
EMOJI ALLOWED: {active_mode.tone_overrides.emoji_enabled if active_mode.tone_overrides.emoji_enabled is not None else 'keep as-is'}
HUMOR ALLOWED: {active_mode.tone_overrides.humor_allowed if active_mode.tone_overrides.humor_allowed is not None else 'keep as-is'}

ORIGINAL MESSAGE:
{template_text}

STRICT RULES:
1. Keep ALL factual content, dates, prices, booking IDs, and action items IDENTICAL.
2. Only change tone, word choice, phrasing style, and emoji usage.
3. Keep it roughly the same length — do NOT add new information.
4. Do NOT add greetings like "Hello!" or "Hi!" if the original doesn't have one.
5. Do NOT mention that you're an AI, bot, or assistant unless the original does.
6. If the original has markdown formatting (**bold**, bullet points), keep the formatting.
7. Return ONLY the rewritten message — no explanation, no quotes, no prefix.

Rewritten message:"""

    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1024,
            ),
        )
        if response and response.text:
            rewritten = response.text.strip()
            # Strip any accidental quotes the model might wrap around it
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            logger.debug(f"PERSONA_REWRITE | mode={active_mode.mode_name} | original_len={len(template_text)} | rewritten_len={len(rewritten)}")
            return rewritten
        return template_text
    except Exception as e:
        logger.warning(f"Persona rewrite failed (using original): {e}")
        return template_text
