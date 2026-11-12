"""
Request-scoped tracing for StayInn.

Every HTTP request (and WhatsApp webhook turn) gets a short ``trace_id`` that
is:
  - stored in contextvars (readable from any service layer)
  - injected into every log line via ``TraceContextFilter``
  - returned on response headers as ``X-Trace-Id``

Never put raw PII in labels — use ``mask_phone`` for guest identifiers.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_session_id: ContextVar[str] = ContextVar("session_id", default="-")
_hotel_id: ContextVar[str] = ContextVar("hotel_id", default="-")
_channel: ContextVar[str] = ContextVar("channel", default="-")


def new_trace_id() -> str:
    """Short opaque id — easy to grep in Cloud Run logs."""
    return uuid.uuid4().hex[:12]


def get_trace_id() -> str:
    return _trace_id.get()


def get_session_id() -> str:
    return _session_id.get()


def get_hotel_id() -> str:
    return _hotel_id.get()


def get_channel() -> str:
    return _channel.get()


def bind_context(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    hotel_id: str | int | None = None,
    channel: str | None = None,
) -> list[Any]:
    """
    Bind correlation fields for the current async task.
    Returns tokens so callers can reset (prefer ``trace_scope``).
    """
    tokens: list[Any] = []
    if trace_id is not None:
        tokens.append((_trace_id, _trace_id.set(str(trace_id))))
    if session_id is not None:
        tokens.append((_session_id, _session_id.set(str(session_id) or "-")))
    if hotel_id is not None:
        tokens.append((_hotel_id, _hotel_id.set(str(hotel_id) if hotel_id != "" else "-")))
    if channel is not None:
        tokens.append((_channel, _channel.set(str(channel) or "-")))
    return tokens


def reset_context(tokens: list[Any]) -> None:
    for var, token in reversed(tokens):
        var.reset(token)


@contextmanager
def trace_scope(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    hotel_id: str | int | None = None,
    channel: str | None = None,
) -> Iterator[str]:
    """Context manager that binds fields and yields the active trace_id."""
    tid = trace_id or get_trace_id()
    if tid == "-":
        tid = new_trace_id()
    tokens = bind_context(
        trace_id=tid,
        session_id=session_id,
        hotel_id=hotel_id,
        channel=channel,
    )
    try:
        yield tid
    finally:
        reset_context(tokens)


def mask_phone(phone: str | None) -> str:
    """Log-safe phone: last 4 digits only."""
    digits = "".join(c for c in (phone or "") if c.isdigit())
    if not digits:
        return "-"
    if len(digits) <= 4:
        return f"…{digits}"
    return f"…{digits[-4:]}"


def context_dict() -> dict[str, str]:
    return {
        "trace_id": get_trace_id(),
        "session_id": get_session_id(),
        "hotel_id": get_hotel_id(),
        "channel": get_channel(),
    }


class TraceContextFilter(logging.Filter):
    """Attach correlation fields onto every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()  # type: ignore[attr-defined]
        record.session_id = get_session_id()  # type: ignore[attr-defined]
        record.hotel_id = get_hotel_id()  # type: ignore[attr-defined]
        record.channel = get_channel()  # type: ignore[attr-defined]
        return True


def install_trace_logging(level: int | str = logging.INFO) -> None:
    """
    Reconfigure root logging so every line includes trace/session/hotel/channel.
    Safe to call once at app startup.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        "trace=%(trace_id)s sess=%(session_id)s hotel=%(hotel_id)s ch=%(channel)s | "
        "%(name)-28s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)

    # Ensure filter on all existing handlers; add one if none
    trace_filter = TraceContextFilter()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        handler.addFilter(trace_filter)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            # Avoid duplicate filters
            if not any(isinstance(f, TraceContextFilter) for f in handler.filters):
                handler.addFilter(trace_filter)
            handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # Pre-populate default attributes so early logs before filter don't crash
    logging.LogRecord.trace_id = "-"  # type: ignore[attr-defined]
    logging.LogRecord.session_id = "-"  # type: ignore[attr-defined]
    logging.LogRecord.hotel_id = "-"  # type: ignore[attr-defined]
    logging.LogRecord.channel = "-"  # type: ignore[attr-defined]
