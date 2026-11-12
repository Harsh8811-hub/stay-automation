"""
StayInn AI Hotel Booking Chatbot — FastAPI Application Entry Point.
[Reload Triggered: Installed google-cloud-vision on Windows Store Python]

Startup sequence:
1. Connect to MongoDB
2. Create database indexes
3. Seed ChromaDB knowledge base (if empty)
4. Compile LangGraph workflow
5. Start serving

Shutdown:
1. Close MongoDB connection
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.whatsapp import router as whatsapp_router
from app.api.routes.admin import router as admin_router
from app.api.routes.amenities import router as amenities_router
from app.api.routes.payment_webhook import router as payment_webhook_router
from app.api.routes.smart_reply import router as smart_reply_router
from app.api.routes.tasks import router as tasks_router
from app.api.routes.escalations import router as escalations_router, tickets_router
from app.api.routes.email import router as email_router
from app.config.settings import get_settings
from app.db.mongodb import close_mongo, connect_mongo
from app.rag.chroma_service import sync_knowledge_base_from_mongo
from app.repositories import (
    activity_repository,
    amenity_repository,
    booking_repository,
    guest_repository,
    inbound_dedupe_repository,
    lead_repository,
    payment_receipt_repository,
    personalization_repository,
    price_match_repository,
    session_repository,
    task_repository,
    ticket_repository,
)
from app.services.hotel_media_service import hotel_media_service
from app.services import escalation_notification_service, task_notification_service
from app.workflows.graph import get_compiled_graph

# ── Logging (includes trace/session/hotel/channel on every line) ─
settings = get_settings()
from app.observability.tracing import install_trace_logging

install_trace_logging(settings.LOG_LEVEL)

# Wrap Gemini's generate_content_async once so every call logs its token usage.
# Class-level on purpose: Gemini is called from ~20 sites across 15 modules, so
# there is no single call site to instrument. See app/observability/llm_usage.py.
from app.observability.llm_usage import install_llm_usage_logging

install_llm_usage_logging()

logger = logging.getLogger(__name__)


def _index_initializers() -> list[tuple[str, object]]:
    """Every index set startup is responsible for, in the order it runs them.

    A function rather than a literal inside the lifespan so a test can assert
    that a collection's setup is actually wired in. The `task_notification`
    entry matters most: its unique `task_id` index is the only thing that makes
    "tell the guest once" true, and it went uncreated in production for a day
    because nothing checked that the list was complete.
    """
    return [
        ("guest", guest_repository.ensure_indexes),
        ("session", session_repository.ensure_indexes),
        ("booking", booking_repository.ensure_indexes),
        ("lead", lead_repository.ensure_indexes),
        ("payment_receipt", payment_receipt_repository.ensure_indexes),
        ("ticket", ticket_repository.ensure_indexes),
        ("task", task_repository.ensure_indexes),
        ("activity", activity_repository.ensure_indexes),
        ("amenity", amenity_repository.create_indexes),
        ("personalization", personalization_repository.ensure_indexes),
        ("price_match", price_match_repository.ensure_indexes),
        ("hotel_media", hotel_media_service.ensure_indexes),
        ("inbound_dedupe", inbound_dedupe_repository.ensure_indexes),
        # Without this the claim insert never raises DuplicateKeyError, so two
        # warm instances can both message the same guest and a failing number
        # is retried forever instead of three times.
        ("task_notification", task_notification_service.ensure_indexes),
        # Same reason one collection over: the unique `ticket_id` index is the
        # only thing that stops a resolved escalation being messaged twice.
        ("ticket_notification", escalation_notification_service.ensure_indexes),
    ]


async def _async_background_init():
    """Run non-critical indexing, ticket housekeeping, and multi-hotel cache/RAG warmups in the background."""
    logger.info("⏳ Starting background database indexing, cache warming, and ChromaDB sync...")

    # 1. Create indexes (best-effort: a conflicting pre-existing index must
    #    never block container startup — log and continue instead).
    index_initializers = _index_initializers()
    for init_name, init_fn in index_initializers:
        try:
            await init_fn()
            logger.info(f"✅ Indexes ensured: {init_name}")
        except Exception as e:
            logger.warning(f"⚠️ Index init skipped ({init_name}): {e}")
    logger.info("✅ Database index initialization complete (best-effort)")

    # 2. Expire stale support tickets (>7 days old)
    try:
        expired = await ticket_repository.expire_stale_tickets()
        if expired:
            logger.info(f"✅ Expired {expired} stale support tickets")
    except Exception as e:
        logger.warning(f"⚠️ Ticket expiry check skipped: {e}")

    # 3. Warm Property Registry, Hotel Configs & Sync Knowledge Base
    try:
        from app.services.property_router import property_router
        await property_router.warm_registry()
        logger.info("✅ Property router registry warmed")
    except Exception as e:
        logger.warning(f"⚠️ Property router warming skipped: {e}")

    try:
        from app.db.mongodb import hotels_collection
        from app.services.hotel_config_service import warm_cache
        
        all_hotels = await hotels_collection().find({}).to_list(100)
        active_ids = set()
        for h in all_hotels:
            raw_id = h.get("hotel_id")
            if raw_id and str(raw_id).isdigit():
                active_ids.add(str(raw_id))
        from app.services.whatsapp_service import mapped_hotel_ids
        for m_id in mapped_hotel_ids():
            active_ids.add(str(m_id))
        active_ids.add(str(settings.HOTEL_ID))

        for hid in active_ids:
            try:
                await sync_knowledge_base_from_mongo(hid)
                await warm_cache(hid)
            except Exception as e:
                logger.warning(f"⚠️ Multi-hotel init skipped for hotel_id={hid}: {e}")
                
        logger.info(f"✅ Multi-hotel configs & ChromaDB synced for {len(active_ids)} hotels: {sorted(active_ids)}")
    except Exception as e:
        logger.warning(f"⚠️ Hotel warmup skipped: {e}")


# ── Lifespan (startup / shutdown) ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs once on startup and shutdown."""
    import asyncio
    import os

    # ── STARTUP ──────────────────────────────────────────────────
    logger.info("🚀 Starting StayInn Booking Bot...")

    # 1. Connect MongoDB
    await connect_mongo()
    logger.info("✅ MongoDB connected")

    # 2. Pre-compile LangGraph workflow (in-memory)
    try:
        get_compiled_graph()
        logger.info("✅ LangGraph workflow compiled")
    except Exception as e:
        logger.warning(f"⚠️ LangGraph compilation warning: {e}")

    import app.services.chat_runner as chat_runner
    chat_runner._bot_ready = True

    # 3. Start non-blocking background initializers (indexes, warmups, ChromaDB RAG sync)
    bg_init_task = asyncio.create_task(_async_background_init())

    # 4. Start Razorpay 5-min reconciliation background worker (Ticket C2 / Spec §10.2)
    from app.services.reconciliation_service import (
        start_hold_cleanup_background_task,
        start_reconciliation_background_task,
    )
    recon_task = asyncio.create_task(start_reconciliation_background_task())

    # 5. Start the expired-hold cleanup worker. The non-destructive replacement for the
    #    bookings TTL index, which was deleting live bookings rather than releasing holds.
    hold_task = asyncio.create_task(start_hold_cleanup_background_task())

    # 6. Start the task-completion notice sweep. Staff finish tasks in the CRM
    #    panel, which writes `hotel_tasks` directly instead of calling our API,
    #    so nothing else would ever tell the guest their payment was verified.
    from app.services.task_notification_service import (
        start_task_notification_background_task,
    )
    task_notify_task = asyncio.create_task(start_task_notification_background_task())

    # 7. Start the escalation-resolution notice sweep. Staff resolve tickets in
    #    the CRM panel too, so the API routes' notification never fires for them.
    from app.services.escalation_notification_service import (
        start_escalation_notification_background_task,
    )
    escalation_notify_task = asyncio.create_task(
        start_escalation_notification_background_task()
    )

    active_port = os.getenv("PORT", str(settings.APP_PORT))
    logger.info(f"🏨 Bot ready for hotel ID: {settings.HOTEL_ID}")
    logger.info(f"🌐 Serving on 0.0.0.0:{active_port}")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────
    bg_init_task.cancel()
    recon_task.cancel()
    hold_task.cancel()
    task_notify_task.cancel()
    escalation_notify_task.cancel()
    await close_mongo()
    logger.info("🛑 StayInn Bot shutdown complete")


# ── App factory ──────────────────────────────────────────────────
app = FastAPI(
    title="StayInn AI Booking Bot",
    description="Production-grade AI Hotel Booking Chatbot powered by LangGraph + Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Trace + latency middleware ───────────────────────────────────
@app.middleware("http")
async def record_http_metrics(request, call_next):
    from app.observability.failure_store import record_failure
    from app.observability.metrics import metrics
    from app.observability.tracing import bind_context, new_trace_id, reset_context

    path = request.url.path
    route = path if path.startswith(("/api/", "/health", "/docs", "/openapi")) else "other"
    # Honor upstream trace if present (CRM / gateway), else mint one
    inbound = (request.headers.get("x-trace-id") or request.headers.get("x-request-id") or "").strip()
    trace_id = inbound[:32] if inbound else new_trace_id()
    channel = "whatsapp" if "/webhooks/whatsapp" in path else ("api" if path.startswith("/api/") else "http")
    tokens = bind_context(trace_id=trace_id, channel=channel)

    try:
        with metrics.timer("http.request_latency_ms", method=request.method, route=route):
            response = await call_next(request)
        metrics.incr(
            "http.requests_total",
            method=request.method,
            route=route,
            status=response.status_code,
        )
        if response.status_code >= 500:
            record_failure(
                component="http",
                error=f"{request.method} {route} → {response.status_code}",
                severity="error",
                details={"method": request.method, "route": route, "status": response.status_code},
            )
        response.headers["X-Trace-Id"] = trace_id
        return response
    except Exception as e:
        record_failure(
            component="http",
            error=e,
            severity="error",
            details={"method": request.method, "route": route},
            exc=e,
        )
        raise
    finally:
        reset_context(tokens)


# ── Routes ───────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(whatsapp_router)
app.include_router(admin_router)
app.include_router(amenities_router, prefix="/api/amenities", tags=["amenities"])
app.include_router(payment_webhook_router)
app.include_router(smart_reply_router)
app.include_router(tasks_router)
app.include_router(escalations_router)
app.include_router(tickets_router)
app.include_router(email_router)


# ── Root endpoints ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "StayInn AI Booking Bot",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/")
@app.post("/webhook")
async def root_post(request: Request):
    """
    Handle incoming POST requests to root / or /webhook.
    Gracefully routes to either WhatsApp webhook or Chat endpoint if body matches,
    preventing 405 Method Not Allowed errors when upstream callers use the root URL.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    logger.info(
        "ROOT_POST_RECEIVED | keys=%s size=%d",
        list(data.keys()) if isinstance(data, dict) else type(data),
        len(str(data)),
    )

    # Case 1: WhatsApp Webhook payload
    if isinstance(data, dict) and data.get("object") == "whatsapp_business_account" and "entry" in data:
        from app.api.routes.whatsapp import WebhookPayload, receive_whatsapp_message
        payload = WebhookPayload(**data)
        return await receive_whatsapp_message(payload)

    # Case 2: Chat API Request
    if isinstance(data, dict) and "message" in data:
        from app.api.routes.chat import chat
        from app.models.schemas import ChatRequest
        req = ChatRequest(**data)
        return await chat(req)

    # Fallback: return 200 OK with endpoints directory instead of 405
    return {
        "status": "ok",
        "service": "StayInn AI Booking Bot",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "whatsapp_webhook": "/api/webhooks/whatsapp",
            "razorpay_webhook": "/api/webhooks/razorpay",
            "health": "/health",
            "docs": "/docs",
        },
        "received_keys": list(data.keys()) if isinstance(data, dict) else [],
    }
