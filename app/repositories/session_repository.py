"""
Session Repository — CRUD for conversation state and messages.

Backs conversation state using the canonical `conversations` and `messages`
collections in bot_migration / stayinn_web. There is no legacy fallback: the
old "Booking_Sessions" accessor pointed at `conversations` itself, and no
document in that collection carries a top-level `session_id` or a `legacy`
sub-document, so every fallback query was dead code that could never match.
"""

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from app.db.mongodb import (
    conversations_collection,
    get_database,
    messages_collection,
)
from app.models.schemas import BookingSession
from app.repositories import guest_repository
from app.utils.phone import phone_lookup_variants

logger = logging.getLogger(__name__)


async def ensure_indexes() -> None:
    """Create indexes for conversations, messages, and session locks. Idempotent."""
    # 1. Conversations
    conv_col = conversations_collection()
    await conv_col.create_index("conversation_id", unique=True)
    await conv_col.create_index([("hotel_id", 1), ("guest_id", 1), ("status", 1)])
    await conv_col.create_index([("hotel_id", 1), ("last_message_at", -1)])
    await conv_col.create_index([("hotel_id", 1), ("updated_at", -1)])
    try:
        await conv_col.create_index(
            "workflow_state_expires_at",
            expireAfterSeconds=0,
            sparse=True,
        )
    except Exception:
        pass

    # 2. Messages
    msg_col = messages_collection()
    await msg_col.create_index("message_id", unique=True)
    await msg_col.create_index([("conversation_id", 1), ("created_at", -1)])
    await msg_col.create_index([("hotel_id", 1), ("created_at", -1)])

    # 3. Session Locks
    lock_coll = get_database()["session_locks"]
    await lock_coll.create_index("session_id", unique=True)
    await lock_coll.create_index("acquired_at", expireAfterSeconds=30)
    logger.info("conversations, messages, and session_locks indexes ensured")


@asynccontextmanager
async def acquire_session_lock(session_id: str, timeout_seconds: float = 5.0):
    """Distributed per-session lock backed by MongoDB to prevent race conditions."""
    lock_coll = get_database()["session_locks"]
    acquired = False
    # Monotonic, not datetime.now: this is a *duration*, and a wall clock that
    # steps backwards (NTP, or a frozen test clock) would make the elapsed time
    # read as zero forever, so the timeout below would never fire and the guest
    # would get no reply at all. Measured 2026-10-25.
    start = asyncio.get_running_loop().time()
    while not acquired:
        try:
            await lock_coll.insert_one({
                "session_id": session_id,
                "acquired_at": datetime.now(timezone.utc)
            })
            acquired = True
        except Exception:
            await asyncio.sleep(0.05)
            if asyncio.get_running_loop().time() - start >= timeout_seconds:
                # Force release stale lock after timeout
                await lock_coll.delete_one({"session_id": session_id})

    try:
        yield
    finally:
        if acquired:
            try:
                await lock_coll.delete_one({"session_id": session_id})
            except Exception:
                pass


def _map_role_to_sender(role: str) -> str:
    """Map internal role names to canonical sender enum (guest, bot, staff, system)."""
    r = (role or "").lower()
    if r in ("user", "guest", "customer"):
        return "guest"
    elif r in ("assistant", "bot", "ai"):
        return "bot"
    elif r in ("agent", "staff", "receptionist"):
        return "staff"
    return "system"


async def get_session(session_id: str, hotel_id: int | str | None = None) -> BookingSession | None:
    """Load a session by its ID.
    If hotel_id is provided, strictly enforces that the session belongs to that hotel.
    """
    if not session_id:
        return None

    conv_col = conversations_collection()
    query: dict[str, Any] = {"conversation_id": session_id}
    if hotel_id is not None:
        str_hid = str(hotel_id).strip()
        query["hotel_id"] = {"$in": [str_hid, int(str_hid)] if str_hid.isdigit() else [str_hid]}

    doc = await conv_col.find_one(query)

    if doc and doc.get("workflow_state"):
        state_dict = doc["workflow_state"]
        if isinstance(state_dict, dict):
            state_dict.setdefault("session_id", session_id)
            state_dict.setdefault("hotel_id", int(doc.get("hotel_id") or 111111) if str(doc.get("hotel_id", "")).isdigit() else 111111)
            try:
                session = BookingSession(**state_dict)
                if hotel_id is not None and str(session.hotel_id) != str(hotel_id):
                    logger.warning(
                        "CROSS_TENANT_SESSION_MISMATCH | session=%s sess_hotel=%s req_hotel=%s",
                        session_id, session.hotel_id, hotel_id,
                    )
                    return None
                return session
            except Exception as e:
                logger.warning(f"Error parsing workflow_state for {session_id}: {e}")

    return None


def _message_timestamp(m: Any, *, fallback: datetime) -> datetime:
    """The moment a message says it was sent, normalised to an aware UTC datetime.

    The transcript on `BookingSession` normally carries a real timestamp, but
    three older shapes reach here and each needs a safe answer rather than an
    exception: a dict without `timestamp`, an ISO string (what a round-trip
    through `model_dump(mode="json")` produces), and a naive datetime.
    """
    if isinstance(m, dict):
        raw = m.get("timestamp")
    else:
        raw = getattr(m, "timestamp", None)

    if isinstance(raw, str):
        try:
            raw = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    return fallback


def _message_id(session_id: str, sender: str, content: str, sent_at: datetime) -> str:
    """A message id that is a function of the message, not of when it was saved.

    This is what makes the write idempotent. The previous id embedded the save
    clock, so every re-save of the same transcript minted fresh ids and
    re-inserted the same three messages — measured live: identical rows 1 second
    apart under different ids. Including the message's own timestamp keeps a
    genuine repeat ("ok" sent twice) as two rows while a re-save is a no-op.

    The timestamp is truncated to milliseconds first, because BSON stores
    datetimes at millisecond precision. Hashing the microseconds meant a message
    that had been written and then read back off the document hashed differently
    from the in-memory one that wrote it — so any later save that carried it in
    its last-three window inserted a second row. Measured on the `ac_words`
    replay: one duplicated card, whose two rows differ only below the millisecond.
    """
    sent_ms = sent_at.replace(microsecond=(sent_at.microsecond // 1000) * 1000)
    digest = hashlib.sha1(
        repr((sent_ms.isoformat(), sender, content)).encode("utf-8")
    ).hexdigest()[:20]
    return f"MSG-{session_id}-{digest}"


async def upsert_session(session: BookingSession) -> None:
    """
    Insert or update a session document in `conversations`, sync guest identity in `guests`,
    and append messages to `messages`.
    """
    str_hotel_id = str(session.hotel_id)
    now = datetime.now(timezone.utc)
    session.updated_at = now

    # 1. Ensure canonical guest record
    guest = await guest_repository.get_or_create_guest(
        hotel_id=str_hotel_id,
        phone=session.customer_identifier,
        name=session.guest_name,
        email=session.guest_email,
    )
    guest_id = guest["guest_id"]

    # 2. Serialize full workflow state
    state_data = session.model_dump(mode="json")

    # 3. Upsert conversation doc
    conv_col = conversations_collection()
    await conv_col.update_one(
        {"conversation_id": session.session_id},
        {
            "$set": {
                "conversation_id": session.session_id,
                "hotel_id": str_hotel_id,
                "guest_id": guest_id,
                "channel": "whatsapp",
                "status": "active" if session.booking_stage not in ("confirmed", "cancelled") else "resolved",
                "workflow_state": state_data,
                "booking_id": session.booking_id,
                "last_message_at": now,
                "updated_at": now,
            },
            "$setOnInsert": {
                "started_at": now,
                "created_at": now,
                "metadata": {},
            }
        },
        upsert=True,
    )

    # 4. Ingest latest messages to messages collection
    if session.messages:
        msg_col = messages_collection()
        recent_msgs = session.messages[-3:] if len(session.messages) > 3 else session.messages
        for m in recent_msgs:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
            if not content:
                continue
            sender = _map_role_to_sender(role)
            sent_at = _message_timestamp(m, fallback=now)
            msg_id = _message_id(session.session_id, sender, content, sent_at)
            try:
                await msg_col.update_one(
                    {"message_id": msg_id},
                    {
                        "$setOnInsert": {
                            "message_id": msg_id,
                            "conversation_id": session.session_id,
                            "hotel_id": str_hotel_id,
                            "sender": sender,
                            "content": content,
                            "media": None,
                            "translation": None,
                            "created_at": sent_at,
                        }
                    },
                    upsert=True,
                )
            except Exception:
                pass


async def find_active_session(hotel_id: int | str, customer_identifier: str) -> BookingSession | None:
    """
    Find the most recent active session for a customer at a hotel.
    Used to resume conversations across WhatsApp turns and after server restarts.
    """
    if not hotel_id or not customer_identifier:
        return None

    str_hotel_id = str(hotel_id).strip()
    hotel_filter = {"$in": [str_hotel_id, int(str_hotel_id)] if str_hotel_id.isdigit() else [str_hotel_id]}
    variants = phone_lookup_variants(customer_identifier) or [customer_identifier.strip()]
    
    digits = "".join(ch for ch in str(customer_identifier) if ch.isdigit())
    clean_digits = digits[-10:] if len(digits) >= 10 else digits

    conv_col = conversations_collection()

    # 1. Direct query across conversations collection by phone variants, guest_id, and workflow_state
    or_clauses = [
        {"workflow_state.customer_identifier": {"$in": variants}},
        {"workflow_state.guest_phone": {"$in": variants}},
        {"metadata.phone": {"$in": variants}},
        {"conversation_id": {"$in": variants}},
    ]
    if clean_digits:
        or_clauses.extend([
            {"guest_id": f"GUEST-{str_hotel_id}-{clean_digits}"},
            {"guest_id": {"$regex": clean_digits}},
            {"workflow_state.customer_identifier": {"$regex": clean_digits}},
        ])

    doc = await conv_col.find_one(
        {
            "hotel_id": hotel_filter,
            "$or": or_clauses,
            "status": {"$ne": "cancelled"},
            "workflow_state": {"$ne": None},
        },
        sort=[("updated_at", -1), ("last_message_at", -1)],
    )

    if doc and doc.get("workflow_state"):
        try:
            state_dict = doc["workflow_state"]
            state_dict.setdefault("session_id", doc.get("conversation_id"))
            state_dict.setdefault("hotel_id", int(str_hotel_id) if str_hotel_id.isdigit() else 111111)
            return BookingSession(**state_dict)
        except Exception as e:
            logger.warning(f"Error parsing active workflow_state from conversations: {e}")

    # 2. Lookup by canonical guest record
    guest = await guest_repository.get_guest_by_phone(str_hotel_id, customer_identifier)
    if guest:
        doc = await conv_col.find_one(
            {
                "hotel_id": hotel_filter,
                "guest_id": guest["guest_id"],
                "status": {"$ne": "cancelled"},
                "workflow_state": {"$ne": None},
            },
            sort=[("updated_at", -1), ("last_message_at", -1)],
        )
        if doc and doc.get("workflow_state"):
            try:
                state_dict = doc["workflow_state"]
                state_dict.setdefault("session_id", doc.get("conversation_id"))
                state_dict.setdefault("hotel_id", int(str_hotel_id) if str_hotel_id.isdigit() else 111111)
                return BookingSession(**state_dict)
            except Exception as e:
                logger.warning(f"Error parsing active workflow_state: {e}")

    return None


async def delete_session(session_id: str) -> None:
    """Hard-delete a session (for testing/cleanup)."""
    conv_col = conversations_collection()
    await conv_col.delete_one({"conversation_id": session_id})


async def update_personality_mode(session_id: str, mode_id: str) -> None:
    """Update the active personality mode for a session."""
    now = datetime.now(timezone.utc)
    conv_col = conversations_collection()
    await conv_col.update_one(
        {"conversation_id": session_id},
        {
            "$set": {
                "workflow_state.active_personality_mode": mode_id,
                "workflow_state.personality_mode_switched_at": now.isoformat(),
                "updated_at": now,
            }
        }
    )
