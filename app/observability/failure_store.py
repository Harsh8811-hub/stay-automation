"""
In-process recent-failure ring buffer for ops debugging.

Surfaced via GET /api/admin/debug/failures.
Survives only for the lifetime of the Cloud Run instance — pair with
Cloud Logging (grep by trace_id) for durable history.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from typing import Any

from app.observability.tracing import context_dict, mask_phone

_MAX = 200
_lock = threading.Lock()
_failures: deque[dict[str, Any]] = deque(maxlen=_MAX)


def record_failure(
    *,
    component: str,
    error: str | Exception,
    severity: str = "error",
    hotel_id: int | str | None = None,
    session_id: str | None = None,
    phone: str | None = None,
    details: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    """
    Store a failure snapshot. Safe to call from any thread / async task.
    Does not raise.
    """
    ctx = context_dict()
    entry: dict[str, Any] = {
        "ts": time.time(),
        "severity": severity,
        "component": component,
        "error": str(error)[:800],
        "trace_id": ctx.get("trace_id") or "-",
        "session_id": session_id or ctx.get("session_id") or "-",
        "hotel_id": str(hotel_id) if hotel_id is not None else (ctx.get("hotel_id") or "-"),
        "channel": ctx.get("channel") or "-",
        "phone": mask_phone(phone),
        "details": _safe_details(details),
    }
    if exc is not None:
        entry["exc_type"] = type(exc).__name__
        entry["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:]
    elif isinstance(error, BaseException):
        entry["exc_type"] = type(error).__name__

    with _lock:
        _failures.appendleft(entry)
    return entry


def recent_failures(
    *,
    limit: int = 50,
    component: str | None = None,
    hotel_id: str | int | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    with _lock:
        items = list(_failures)
    if component:
        items = [i for i in items if i.get("component") == component]
    if hotel_id is not None:
        hid = str(hotel_id)
        items = [i for i in items if str(i.get("hotel_id")) == hid]
    if session_id:
        items = [i for i in items if i.get("session_id") == session_id]
    return items[: max(1, min(limit, _MAX))]


def clear_failures() -> None:
    with _lock:
        _failures.clear()


def _safe_details(details: dict[str, Any] | None) -> dict[str, Any]:
    if not details:
        return {}
    out: dict[str, Any] = {}
    for k, v in list(details.items())[:30]:
        try:
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[str(k)] = v if not isinstance(v, str) else v[:500]
            else:
                out[str(k)] = str(v)[:500]
        except Exception:
            out[str(k)] = "<unserializable>"
    return out
