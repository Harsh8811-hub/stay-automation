"""LLM exception classes."""


class LLMUnavailableError(Exception):
    """Raised when the LLM provider is unavailable (quota exceeded, 5xx, timeout)."""
    pass
