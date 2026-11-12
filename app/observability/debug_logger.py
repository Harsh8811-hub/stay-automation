import contextvars
from typing import Any, Dict

_debug_logs = contextvars.ContextVar("debug_logs", default=None)

def enable_debug_logging() -> contextvars.Token:
    """Enable debug logging for the current context and return the token."""
    return _debug_logs.set([])

def disable_debug_logging(token: contextvars.Token):
    """Disable debug logging and clean up."""
    _debug_logs.reset(token)

def add_debug_log(event_type: str, details: Dict[str, Any]):
    """Add a debug log entry if debug logging is enabled in this context."""
    logs = _debug_logs.get()
    if logs is not None:
        logs.append({
            "type": event_type,
            "details": details
        })

def get_debug_logs() -> list:
    """Return the accumulated debug logs for the current context."""
    logs = _debug_logs.get()
    return logs if logs is not None else []

import google.generativeai as genai

_original_generate = genai.GenerativeModel.generate_content_async

async def _patched_generate(self, contents, *args, **kwargs):
    prompt_log = str(contents)
    add_debug_log("llm_prompt", {"prompt": prompt_log})
    
    try:
        response = await _original_generate(self, contents, *args, **kwargs)
        try:
            resp_text = response.text
        except Exception:
            resp_text = "<no text in response>"
        add_debug_log("llm_response", {"response": resp_text})
        return response
    except Exception as e:
        add_debug_log("llm_error", {"error": str(e)})
        raise

# Apply the patch immediately when this module is imported
genai.GenerativeModel.generate_content_async = _patched_generate

