"""
LLM usage logging — one log line per Gemini call, carrying the token counts.

WHY THIS EXISTS
LLM spend came in far above expectation. Nothing in this codebase read
`response.usage_metadata`, so the real per-call token breakdown was invisible —
including `cached_content_token_count`, which is the only way to tell whether
Google's implicit context caching is actually hitting on the large static
prefixes (e.g. `UNDERSTAND_SYSTEM_PROMPT`, ~8.1k tokens) that this bot resends
on nearly every turn.

WHY A CLASS-LEVEL PATCH INSTEAD OF A LINE AT EACH CALL SITE
Gemini is called from roughly twenty places across fifteen modules —
`gemini_service`, the `vision/*` services, `takeover_engine`, `persona_rewriter`,
`personalization_service`, `knowledge_gap_engine`, `audio_transcriber`, and the
API routes. Wrapping the class once covers every one of them, including the ones
added later; a line per call site would miss both the ones I cannot see and the
ones nobody has written yet.

`app/observability/debug_logger.py` uses the same trick for prompt capture, but
it is imported lazily from a debug route, so it is dormant on production traffic.
This module is imported by `app.main`, so it installs on every process start.

CONTRACT: the wrapper never raises, never alters the response, and never changes
the arguments. A bug in here must cost a missing log line, never a lost booking.
"""

import inspect
import logging

import google.generativeai as genai

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL = None


def _caller_label() -> str:
    """Return `module.function` for the code that issued the Gemini call.

    A function name alone is ambiguous across this codebase (`answer_faq` exists
    in more than one place); the module qualifier is what makes a line
    attributable. Best-effort by design — an unreadable stack returns "unknown"
    rather than breaking the call.
    """
    try:
        frame = inspect.currentframe()
        # _caller_label <- _patched_generate_content_async <- the real caller
        caller = frame.f_back.f_back if frame is not None else None
        if caller is None:
            return "unknown"
        module = caller.f_globals.get("__name__", "?")
        return f"{module.rsplit('.', 1)[-1]}.{caller.f_code.co_name}"
    except Exception:
        return "unknown"


def _log_usage(where: str, response) -> None:
    """Emit one GEMINI_USAGE line. Never raises."""
    try:
        if not get_settings().GEMINI_USAGE_LOGGING:
            return
        usage = getattr(response, "usage_metadata", None)
        # A streamed response is an async iterator with no usage_metadata yet;
        # nothing in this codebase streams today, but the guard keeps the
        # eventual `stream=True` from turning into an AttributeError.
        if usage is None:
            return
        logger.info(
            "GEMINI_USAGE | at=%s prompt_tokens=%s cached_tokens=%s "
            "output_tokens=%s total_tokens=%s",
            where,
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "cached_content_token_count", None),
            getattr(usage, "candidates_token_count", None),
            getattr(usage, "total_token_count", None),
        )
    except Exception:
        logger.debug("GEMINI_USAGE logging failed", exc_info=True)


import asyncio
from google.api_core import exceptions as google_exceptions

from app.services.llm_errors import LLMUnavailableError

_INFRA_EXCEPTIONS = (
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
    google_exceptions.GatewayTimeout,
    asyncio.TimeoutError,
    TimeoutError,
)


async def _patched_generate_content_async(self, contents, *args, **kwargs):
    try:
        response = await _ORIGINAL(self, contents, *args, **kwargs)
    except _INFRA_EXCEPTIONS as exc:
        raise LLMUnavailableError(str(exc)) from exc

    _log_usage(_caller_label(), response)
    return response


def install_llm_usage_logging() -> bool:
    """Wrap `GenerativeModel.generate_content_async` once per process.

    Returns True when this call did the installing, False when it was already
    installed. Idempotent: wrapping twice would log every call twice, so the
    guard is load-bearing, not tidiness.
    """
    global _INSTALLED, _ORIGINAL

    if _INSTALLED:
        return False

    try:
        if not get_settings().GEMINI_USAGE_LOGGING:
            logger.info("GEMINI_USAGE logging disabled by GEMINI_USAGE_LOGGING")
            return False

        _ORIGINAL = genai.GenerativeModel.generate_content_async
        genai.GenerativeModel.generate_content_async = _patched_generate_content_async
        _INSTALLED = True
        logger.info("GEMINI_USAGE logging and error mapping installed")
        return True
    except Exception as e:
        logger.warning("GEMINI_USAGE logging NOT installed: %s", e)
        return False

