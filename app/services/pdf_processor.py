"""
PDF Processor for extracting text from PDF files.
Uses pypdf for lightweight text extraction.
"""
import io
import logging

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("pymupdf not installed. PDF text extraction will not work.")

def process_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF file.
    Returns the extracted text.
    Raises ValueError if pypdf is not available, PDF is corrupt, or if it's purely scanned.
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("pymupdf is not installed.")
        raise ValueError("PDF processing is currently disabled (missing dependencies).")
        
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        extracted_text = []
        
        for page in doc:
            text = page.get_text()
            if text:
                extracted_text.append(text)
                
        full_text = "\n".join(extracted_text).strip()
        
        if not full_text:
            logger.info("PDF extraction yielded no text (likely a scanned document).")
            # If we wanted to fallback to OCR, we would convert PDF pages to images here 
            # and call image_vision.py. Since that requires poppler and pdf2image,
            # we will return a friendly error telling the user to upload as an image.
            raise ValueError("This PDF appears to be a scanned document or image. Please upload a clearer text-based PDF or send it as an image (JPG/PNG).")
            
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF.")
        return full_text
        
    except ValueError as ve:
        # Re-raise the scanned document error
        raise ve
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise ValueError("I couldn't read this PDF. Please upload a readable, unlocked version.")
