"""
Health check routes.
"""

import logging
import time

from fastapi import APIRouter

from app.db.mongodb import get_database
from app.models.schemas import HealthResponse
from app.observability.tracing import get_trace_id
from app.rag.chroma_service import get_collection

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — verifies MongoDB and ChromaDB connectivity."""
    mongo_ok = False
    chroma_ok = False

    t0 = time.perf_counter()
    try:
        db = get_database()
        await db.command("ping")
        mongo_ok = True
        logger.debug("HEALTH_MONGO_OK | ms=%.1f", (time.perf_counter() - t0) * 1000)
    except Exception as e:
        logger.warning("HEALTH_MONGO_FAIL | %s", e)

    t1 = time.perf_counter()
    try:
        coll = get_collection()
        coll.count()
        chroma_ok = True
        logger.debug("HEALTH_CHROMA_OK | ms=%.1f", (time.perf_counter() - t1) * 1000)
    except Exception as e:
        logger.warning("HEALTH_CHROMA_FAIL | %s", e)

    status = "healthy" if (mongo_ok and chroma_ok) else "degraded"
    if status != "healthy":
        logger.error("HEALTH_DEGRADED | mongo=%s chroma=%s", mongo_ok, chroma_ok)

    return HealthResponse(
        status=status,
        mongodb=mongo_ok,
        chromadb=chroma_ok,
    )


@router.get("/health/detailed")
async def health_detailed():
    """
    Extended health for ops — dependency checks + metrics uptime + failure count.
    Does not change the lightweight /health contract used by Cloud Run.
    """
    from app.observability.failure_store import recent_failures
    from app.observability.metrics import metrics

    mongo_ok = False
    chroma_ok = False
    mongo_ms = None
    chroma_ms = None
    errors: dict = {}

    t0 = time.perf_counter()
    try:
        db = get_database()
        await db.command("ping")
        mongo_ok = True
        mongo_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        errors["mongodb"] = str(e)[:200]

    t1 = time.perf_counter()
    try:
        coll = get_collection()
        coll.count()
        chroma_ok = True
        chroma_ms = round((time.perf_counter() - t1) * 1000, 1)
    except Exception as e:
        errors["chromadb"] = str(e)[:200]

    snap = metrics.snapshot()
    failures = recent_failures(limit=5)

    return {
        "status": "healthy" if (mongo_ok and chroma_ok) else "degraded",
        "trace_id": get_trace_id(),
        "mongodb": {"ok": mongo_ok, "latency_ms": mongo_ms},
        "chromadb": {"ok": chroma_ok, "latency_ms": chroma_ms},
        "errors": errors or None,
        "uptime_seconds": snap.get("uptime_seconds"),
        "recent_failure_count_sample": len(failures),
        "recent_failures_preview": [
            {"component": f.get("component"), "error": f.get("error"), "trace_id": f.get("trace_id")}
            for f in failures
        ],
        "endpoints": {
            "metrics": "/api/admin/metrics",
            "failures": "/api/admin/debug/failures",
            "session_dump": "/api/admin/debug/session/{session_id}",
            "phone_lookup": "/api/admin/debug/phone/{phone}",
        },
    }
