"""
In-process production metrics for StayInn.

Lightweight counters + latency stats (no external deps).
Exposed via GET /api/admin/metrics for ops dashboards / scraping.

Never store PII in metric labels — use hotel_id / node / outcome only.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class _LatencyStat:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = float("inf")

    def observe(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        self.min_ms = min(self.min_ms, ms)

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "avg_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
        return {
            "count": self.count,
            "avg_ms": round(self.total_ms / self.count, 2),
            "max_ms": round(self.max_ms, 2),
            "min_ms": round(self.min_ms if self.min_ms != float("inf") else 0.0, 2),
        }


class MetricsRegistry:
    """Thread-safe process-wide metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latency: dict[str, _LatencyStat] = defaultdict(_LatencyStat)
        self._started_at = time.time()

    def incr(self, name: str, amount: int = 1, **labels: Any) -> None:
        key = _metric_key(name, labels)
        with self._lock:
            self._counters[key] += amount

    def observe_ms(self, name: str, duration_ms: float, **labels: Any) -> None:
        key = _metric_key(name, labels)
        with self._lock:
            self._latency[key].observe(duration_ms)

    @contextmanager
    def timer(self, name: str, **labels: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.observe_ms(name, ms, **labels)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            latency = {k: v.as_dict() for k, v in sorted(self._latency.items())}
        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": counters,
            "latency_ms": latency,
        }

    def reset(self) -> None:
        """Test helper — clear all series."""
        with self._lock:
            self._counters.clear()
            self._latency.clear()
            self._started_at = time.time()


def _metric_key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    parts = ",".join(f"{k}={labels[k]}" for k in sorted(labels) if labels[k] is not None)
    return f"{name}{{{parts}}}" if parts else name


# Global singleton
metrics = MetricsRegistry()


# ── Convenience helpers used across the app ──────────────────────

def record_feature_gate(feature: str, decision: str, hotel_id: int | None = None) -> None:
    metrics.incr(
        "feature_gate_decisions_total",
        feature=feature,
        decision=decision,
        hotel_id=hotel_id,
    )


def record_node_error(node: str, hotel_id: str | int | None = None) -> None:
    metrics.incr("workflow_node_errors_total", node=node, hotel_id=hotel_id)


def record_chat_turn(outcome: str, hotel_id: str | int | None = None) -> None:
    metrics.incr("chat_turns_total", outcome=outcome, hotel_id=hotel_id)


def record_translation_failure(
    preferred_language: str | None = None,
    conversation_language: str | None = None,
    hotel_id: int | None = None,
) -> None:
    """Increment counter when bot falls back to English due to a translation error."""
    metrics.incr(
        "translation_failures_total",
        hotel_id=hotel_id,
        preferred_language=preferred_language,
        conversation_language=conversation_language,
    )


def record_outbound_send(
    *,
    channel: str,
    message_type: str,
    outcome: str,
    hotel_id: int | str | None = None,
) -> None:
    """
    Track WhatsApp / CRM / Twilio send attempts.
    outcome: ok | error | skipped | missing_creds
    """
    metrics.incr(
        "outbound_sends_total",
        channel=channel,
        message_type=message_type,
        outcome=outcome,
        hotel_id=hotel_id,
    )


def record_webhook_event(
    *,
    event: str,
    outcome: str,
    hotel_id: int | str | None = None,
) -> None:
    metrics.incr(
        "webhook_events_total",
        event=event,
        outcome=outcome,
        hotel_id=hotel_id,
    )
