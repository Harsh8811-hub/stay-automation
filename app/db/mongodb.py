"""
Async MongoDB connection manager using Motor.

Provides a singleton client and typed collection accessors for the canonical
migrated database schema (bot_migration / stayinn_web). The bot reads ONLY
this database — no master/legacy fallbacks.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config.settings import get_settings

# ── Singleton client & db ────────────────────────────────────────
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_mongo() -> None:
    """Initialise the Motor client. Called once at app startup."""
    global _client, _db
    settings = get_settings()
    _client = AsyncIOMotorClient(
        settings.MONGODB_URI,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
    )
    _db = _client[settings.MONGODB_DATABASE]

    # Connectivity check with retries for transient SSL network glitches
    import asyncio
    for attempt in range(3):
        try:
            await _client.admin.command("ping")
            break
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1)


async def close_mongo() -> None:
    """Graceful shutdown. Called once at app shutdown."""
    global _client, _db
    if _client:
        _client.close()
    _client = None
    _db = None


def get_database() -> AsyncIOMotorDatabase:
    """Return the active primary database handle (bot_migration / stayinn_web)."""
    global _client, _db
    running_loop = None
    try:
        import asyncio
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    client_loop = getattr(_client, "io_loop", None)
    need_new_client = (
        _client is None
        or (client_loop and client_loop.is_closed())
        or (running_loop and client_loop and client_loop != running_loop)
    )

    if need_new_client:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
        _db = None
    if _db is None:
        settings = get_settings()
        _db = _client[settings.MONGODB_DATABASE]
    return _db


# ── Canonical Migrated Collections (bot_migration / stayinn_web) ──

def guests_collection():
    """Canonical guest identity collection — R/W."""
    return get_database()["guests"]


def conversations_collection():
    """Conversation lifecycle and state persistence — R/W."""
    return get_database()["conversations"]


def messages_collection():
    """Append-only conversation message transcript — R/W."""
    return get_database()["messages"]


def bookings_collection():
    """Confirmed and pending reservations — R/W."""
    return get_database()["bookings"]


def leads_collection():
    """Consolidated booking inquiries and leads — R/W."""
    return get_database()["leads"]


def payments_collection():
    """Consolidated payment links and OCR receipts — R/W."""
    return get_database()["payments"]


def price_matches_collection():
    """Price match claims against OTAs — R/W."""
    return get_database()["price_matches"]


def support_tickets_collection():
    """Support escalation tickets for human takeover — R/W."""
    return get_database()["support_tickets"]


def hotel_tasks_collection():
    """Hotel departmental work orders (Housekeeping, etc.) — R/W."""
    return get_database()["hotel_tasks"]


def activity_logs_collection():
    """Append-only system activity log timeline."""
    return get_database()["bot_activity_logs"]


# ── Normalized Master Data Accessors (bot_migration) ─────────

def hotels_collection():
    """Hotels profile and basic metadata."""
    return get_database()["hotels"]


def hotel_details_collection():
    """Detailed hotel contact, location, and address metadata."""
    return get_database()["hotel_details"]


def hotel_policies_collection():
    """Check-in, check-out, and house rules."""
    return get_database()["hotel_policies"]


def room_types_collection():
    """Individual room catalog and amenities."""
    return get_database()["room_types"]


def rate_plans_collection():
    """Rate plans associated with room types."""
    return get_database()["rate_plans"]


def daily_rates_collection():
    """Calendar rate overrides."""
    return get_database()["daily_rates"]


def daily_inventory_collection():
    """Inventory allotments."""
    return get_database()["daily_inventory"]


def hotel_amenities_collection():
    """Hotel-wide and room-specific amenities."""
    return get_database()["hotel_amenities"]


def hotel_settings_collection():
    """Hotel business settings and tax configuration."""
    return get_database()["hotel_settings"]


def upsell_items_collection():
    """Upsell add-on items catalog."""
    return get_database()["upsell_items"]


def upsell_events_collection():
    """Tracks upsell impression and response events (offered, accepted, declined)."""
    return get_database()["upsell_events"]


def upsell_inventory_collection():
    """Tracks daily availability/capacity for limited inventory upsell items."""
    return get_database()["upsell_inventory"]


def payment_gateways_collection():
    """Payment Gateways & UPI Direct settings (VPA / QR)."""
    return get_database()["payment_gateways"]


def payment_receipts_collection():
    """Parsed payment receipt storage — R/W (merged into `payments` as source="receipt")."""
    return get_database()["payments"]


def knowledge_base_collection():
    """Hotel Knowledge Base (Articles & Long-form Policies)."""
    return get_database()["knowledge_base_articles"]


def hotel_kb_values_collection():
    """Hotel KB Values (Short key-value Policies)."""
    return get_database()["knowledge_base_articles"]


def coupons_collection():
    """Coupons & Promotions collection."""
    return get_database()["promotions"]


def knowledge_requests_collection():
    """Knowledge gap detection tickets for hotel staff — R/W."""
    return get_database()["knowledge_gaps"]


def human_interventions_collection():
    """Staff takeover sessions and chat history — R/W."""
    return get_database()["human_interventions"]


def hotel_customization_collection():
    """Assistant personality branding."""
    return get_database()["hotel_customization"]


def hotel_personality_modes_collection():
    """Personality mode prompt overlays."""
    return get_database()["hotel_personality_modes"]


def hotel_rules_collection():
    """Guardrail DO/DONT rules."""
    return get_database()["hotel_rules"]


def room_service_carts_collection():
    """Room service carts — R/W."""
    return get_database()["room_service_carts"]


def room_service_menu_collection():
    """Room service menus."""
    return get_database()["room_service_menus"]


def payment_policies_collection():
    """Global payment policies for cancellations."""
    return get_database()["payment_policies"]


def hotel_media_collection():
    """Hotel media metadata."""
    return get_database()["hotel_media"]


def pending_context_collection():
    """Pending staff context summaries stored before session creation — R/W."""
    return get_database()["pending_context"]


def processed_messages_collection():
    """
    Inbound webhook message ids already handled — R/W, append-only, TTL-cleaned.

    `_id` is the Meta `wamid`, so uniqueness is enforced by the always-present `_id` index
    and needs no custom index. See app/repositories/inbound_dedupe_repository.py.
    """
    return get_database()["processed_messages"]
