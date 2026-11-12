"""
File validation utilities for media uploads (WhatsApp-compatible).
"""

import logging
from fastapi import UploadFile
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def detect_mime_from_bytes(file_bytes: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if not file_bytes:
        return "application/octet-stream"
    if file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif file_bytes.startswith(b'\xff\xd8'):
        return "image/jpeg"
    elif file_bytes.startswith(b'%PDF'):
        return "application/pdf"
    elif file_bytes.startswith(b'ID3') or file_bytes.startswith(b'\xff\xfb') or file_bytes.startswith(b'\xff\xf3'):
        return "audio/mpeg"
    elif file_bytes.startswith(b'OggS'):
        return "audio/ogg"
    elif file_bytes.startswith(b'RIFF') and b'WAVE' in file_bytes[8:12]:
        return "audio/wav"
    elif file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[8:12]:
        return "image/webp"
    return "application/octet-stream"


def get_file_extension(filename: str) -> str:
    """Extracts the lowercase file extension."""
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def validate_file(file: UploadFile, declared_mime: str | None = None) -> tuple[bool, str, str]:
    """
    Validates the uploaded file against allowed types and sizes.
    Supports files without extensions and generic application/octet-stream (e.g. raw WhatsApp media downloads).
    Returns: (is_valid: bool, error_message: str, file_category: str)
    file_category will be one of: "image", "pdf", "audio", or "unknown"
    """
    settings = get_settings()
    
    # 1. Validate file extension or Content-Type header
    ext = get_file_extension(file.filename)
    content_type = (getattr(file, "content_type", "") or declared_mime or "").lower().split(";")[0].strip()

    category = "unknown"
    if ext in settings.ALLOWED_IMAGE_TYPES or content_type in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        category = "image"
    elif ext in settings.ALLOWED_PDF_TYPES or content_type == "application/pdf":
        category = "pdf"
    elif ext in settings.ALLOWED_AUDIO_TYPES or content_type.startswith("audio/"):
        category = "audio"
    else:
        # Fallback to inspecting binary magic bytes
        file_obj = getattr(file, "file", None)
        head = b""
        if file_obj:
            try:
                if hasattr(file_obj, "peek"):
                    head = file_obj.peek(32)[:32]
                elif hasattr(file_obj, "read"):
                    cur_pos = file_obj.tell() if hasattr(file_obj, "tell") else 0
                    head = file_obj.read(32)
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(cur_pos)
            except Exception as e:
                logger.warning(f"Error sniffing magic bytes from upload: {e}")

        detected_mime = detect_mime_from_bytes(head)
        logger.info(f"FILE_VALIDATION_SNIFF | filename='{file.filename}' content_type='{content_type}' detected_mime='{detected_mime}'")

        if detected_mime in ("image/jpeg", "image/png", "image/webp"):
            category = "image"
        elif detected_mime == "application/pdf":
            category = "pdf"
        elif detected_mime.startswith("audio/"):
            category = "audio"
        else:
            return False, f"Unsupported file format '{ext or content_type}'. WhatsApp files must be JPEG, PNG, or PDF.", "unknown"
        
    # 2. Validate file size if size property available
    if hasattr(file, "size") and file.size is not None:
        size_mb = file.size / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            return False, f"File size ({size_mb:.1f}MB) exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB}MB.", category
            
    return True, "", category


def validate_whatsapp_media_bytes(
    file_bytes: bytes,
    filename: str = "",
    declared_mime: str = ""
) -> tuple[bool, str, str, str]:
    """
    Validates raw media bytes received from WhatsApp Cloud API or multipart upload.
    Returns: (is_valid: bool, error_message: str, detected_mime: str, file_category: str)
    """
    settings = get_settings()
    if not file_bytes:
        return False, "Empty media payload received.", "application/octet-stream", "unknown"

    size_mb = len(file_bytes) / (1024 * 1024)
    # Immediate security block for executables / archives
    if (
        file_bytes.startswith(b'MZ')
        or file_bytes.startswith(b'\x7fELF')
        or file_bytes.startswith(b'\xca\xfe\xba\xbe')
        or file_bytes.startswith(b'PK\x03\x04')
    ):
        return False, "Unsupported or unsafe media format. Please send your document or photo as JPEG, PNG, or PDF.", "application/octet-stream", "unsupported"

    detected_mime = detect_mime_from_bytes(file_bytes)
    
    # Fallback to declared_mime if byte detection returned generic octet-stream
    if detected_mime == "application/octet-stream" and declared_mime:
        declared_clean = declared_mime.split(";")[0].strip().lower()
        if declared_clean in ("image/jpeg", "image/jpg"):
            detected_mime = "image/jpeg"
        elif declared_clean == "image/png":
            detected_mime = "image/png"
        elif declared_clean == "application/pdf":
            detected_mime = "application/pdf"
        elif "audio" in declared_clean:
            detected_mime = declared_clean

    if detected_mime in ("image/jpeg", "image/png"):
        category = "image"
        return True, "", detected_mime, category
    elif detected_mime == "application/pdf":
        category = "pdf"
        return True, "", detected_mime, category
    elif detected_mime.startswith("audio/"):
        category = "audio"
        return True, "", detected_mime, category
    elif detected_mime == "image/webp":
        return False, "WebP images are not supported for processing. Please send as JPEG or PNG.", detected_mime, "unsupported"
    else:
        return False, "Unsupported media format. Please send your document or photo as JPEG, PNG, or PDF.", detected_mime, "unknown"

