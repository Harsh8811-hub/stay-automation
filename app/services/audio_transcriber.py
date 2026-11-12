"""
Audio transcriber using Google Gemini 1.5 Flash.
"""
import logging
from app.services.llm import compat as genai
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Re-use model initialization logic from gemini_service to keep it clean
_model = None

def _get_model():
    global _model
    if _model is None:
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Using 1.5 flash as it supports audio natively
        _model = genai.GenerativeModel(settings.GEMINI_MODEL)
    return _model

def transcribe_audio(file_bytes: bytes, mime_type: str) -> str:
    """
    Transcribes audio using Gemini.
    Returns the transcription text.
    Raises ValueError on failure.
    """
    # DeepSeek has no audio input, and the provider switch for the week is a
    # deliberate trade: voice notes are skipped, not half-transcribed. Checked
    # before the `try` because that block rewrites every exception into one
    # generic sentence, which would throw this message away.
    if get_settings().LLM_PROVIDER != "gemini":
        raise ValueError(
            "Voice notes are temporarily unavailable. Please type your message, or send a photo — "
            "both reach us normally."
        )

    try:
        model = _get_model()
        
        # Determine strict MIME type if needed by Gemini, though usually standard mimes work.
        # e.g., audio/mp3, audio/ogg, audio/wav, audio/m4a, audio/webm
        if not mime_type.startswith("audio/"):
            mime_type = f"audio/{mime_type}" if mime_type in ["mp3", "wav", "m4a", "ogg", "webm"] else "audio/mp3"
            
        logger.info(f"Sending audio to Gemini for transcription (mime_type: {mime_type}, size: {len(file_bytes)} bytes)")
        
        # Construct the part manually
        prompt = """Accurately transcribe this audio. 
CRITICAL RULE: Preserve the EXACT language spoken. 
- If the user speaks in Hindi, transcribe in Hindi. 
- If the user speaks in English, transcribe in English.
- If the user speaks in Hinglish (a mix of Hindi and English), transcribe naturally in Hinglish.
Do not translate unless necessary. Return only the exact transcription without any extra commentary. If it is empty or unintelligible, say 'UNINTELLIGIBLE'."""
        
        response = model.generate_content([
            {
                "mime_type": mime_type,
                "data": file_bytes
            },
            prompt
        ])
        
        transcription = response.text.strip()
        
        if transcription == "UNINTELLIGIBLE":
            raise ValueError("The audio was unintelligible or empty. Please send a clearer voice message.")
            
        logger.info(f"Successfully transcribed audio: {transcription[:50]}...")
        return transcription
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio with Gemini: {e}", exc_info=True)
        raise ValueError("Failed to transcribe the audio. Please try again or type your message.")
