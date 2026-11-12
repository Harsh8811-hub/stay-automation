"""
Run async coroutines from sync Streamlit code on one persistent event loop.

Streamlit reruns the script synchronously; calling asyncio.run() per message
creates and destroys loops. Motor (MongoDB) binds to the first loop — reuse one.
"""

from __future__ import annotations

import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()

            def _run_forever() -> None:
                asyncio.set_event_loop(_loop)
                _loop.run_forever()

            _thread = threading.Thread(
                target=_run_forever,
                daemon=True,
                name="streamlit-async",
            )
            _thread.start()
    return _loop


def run_async(coro):
    """Block until `coro` completes on the shared background event loop."""
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
