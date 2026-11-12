"""
Orchestrator for media file processing.
Reads file bytes, delegates to the appropriate processor based on category,
and returns extracted text.
"""
import logging
from fastapi import UploadFile

from app.utils.file_validation import validate_file, get_file_extension
from app.services.image_vision import process_image
from app.services.pdf_processor import process_pdf
from app.services.audio_transcriber import transcribe_audio

logger = logging.getLogger(__name__)

async def process_upload(file: UploadFile, user_message: str | None = None) -> str:
    """
    Process an uploaded file and extract its text/transcript or detailed visual context.
    Throws ValueError with a user-friendly message if processing fails.
    """
    # 1. Validate extension and determine category
    is_valid, error_msg, category = validate_file(file)
    if not is_valid:
        raise ValueError(error_msg)
        
    try:
        # 2. Read bytes
        file_bytes = await file.read()
        
        # 3. Validate size now that we have the bytes
        from app.config.settings import get_settings
        settings = get_settings()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise ValueError(f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB.")
            
        if not file_bytes:
            raise ValueError("The uploaded file is empty.")
            
        # 4. Route to specific processor
        ext = get_file_extension(file.filename)
        
        logger.info(f"Processing {category} file: {file.filename} ({size_mb:.2f} MB)")
        
        if category == "image":
            extracted_text = process_image(file_bytes, user_message)
            if not extracted_text:
                raise ValueError("Could not analyze the image. Please upload a clearer image or type your details.")
            return extracted_text
            
        elif category == "pdf":
            from app.utils.privacy_scrubber import scrub_id_numbers
            raw_pdf = process_pdf(file_bytes)
            return scrub_id_numbers(raw_pdf)
            
        elif category == "audio":
            # Pass content_type if available, otherwise guess from extension
            mime_type = file.content_type if file.content_type else f"audio/{ext}"
            return transcribe_audio(file_bytes, mime_type=mime_type)
            
        else:
            raise ValueError(f"Unsupported file category: {category}")
            
    except ValueError:
        # Re-raise known validation/processing errors directly
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file {file.filename}: {e}", exc_info=True)
        raise ValueError("An unexpected error occurred while reading the file. Please try again or type your message.")
    finally:
        # Ensure file pointer is closed
        await file.close()
