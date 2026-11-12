"""
Staff message detector — CRM backend messages API.

Verified CRM endpoint:
  GET {CRM}/api/backend/messages/?Secretkey=&hotelID=&phone_number=

Intended CRM sender taxonomy (source of truth):
  - sender="user"  → hotel staff (CRM user)
  - sender="bot"   → AI bot
  - sender="guest" → customer

Mongo fallback reads the normalized `messages` collection (bot_migration):
  - sender="staff"  → hotel staff
  - sender="guest"  → customer
  - sender="bot"    → AI bot
  - sender="system" → system/bot-side note (not staff)
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Any
from urllib.parse import urljoin

import httpx

from app.config.settings import get_settings
from app.db.mongodb import messages_collection

logger = logging.getLogger(__name__)

# CRM HTTP API
_CRM_STAFF_SENDERS = frozenset({"user", "staff", "agent"})
_CRM_NON_STAFF_SENDERS = frozenset({"guest", "bot", "assistant", "ai", "customer"})

# Flat messages collection senders (guest/bot/system are NOT staff)
_MONGO_STAFF_ROLES = frozenset({"agent", "staff"})


def crm_messages_list_url() -> str:
    """Derive list URL from CRM_BACKEND_URL (.../messages/send/ → .../messages/)."""
    settings = get_settings()
    send_url = (settings.CRM_BACKEND_URL or "").rstrip("/") + "/"
    if send_url.endswith("/send/"):
        return send_url[: -len("send/")]
    if "/messages/send" in send_url:
        return send_url.split("/messages/send", 1)[0] + "/messages/"
    return urljoin(send_url, "../")


def is_crm_staff_message(msg: dict[str, Any]) -> bool:
    """CRM API: staff == sender user (not bot, not guest)."""
    if not isinstance(msg, dict):
        return False
    sender = str(msg.get("sender") or msg.get("Sender") or "").strip().lower()
    if sender in _CRM_NON_STAFF_SENDERS:
        return False
    return sender in _CRM_STAFF_SENDERS


def is_mongo_staff_message(msg: dict[str, Any]) -> bool:
    """Flat messages: staff == sender "staff" (guest/bot/system are not staff).
    Legacy Chat_Sessions docs: staff == role agent/staff (role=user is the guest)."""
    if not isinstance(msg, dict):
        return False
    sender = str(msg.get("sender") or "").strip().lower()
    if sender:
        return sender == "staff"
    role = str(msg.get("role") or msg.get("Role") or "").strip().lower()
    return role in _MONGO_STAFF_ROLES


def is_staff_phone(phone: str) -> bool:
    """Checks if a phone number matches configured staff phone numbers in GCP/VM settings."""
    if not phone:
        return False
    settings = get_settings()
    configured = getattr(settings, "STAFF_PHONE_NUMBERS", "") or ""
    if not configured:
        return False
    clean_target = phone.replace("+", "").strip()
    staff_list = [p.replace("+", "").strip() for p in configured.split(",") if p.strip()]
    return clean_target in staff_list


def parse_staff_command(text: str, session_id: str = "") -> dict[str, Any] | None:
    """
    Deterministic regex parser for hotel staff quick commands.
    Supported commands:
    - price_override: "2000 ok", "rate 2500", "give for 1800", "2000 me dedo"
    - discount: "10% discount", "give 500 off"
    - skip_payment: "skip payment", "no advance", "confirm without advance"
    - room_upgrade: "upgrade to deluxe", "give premier room"
    - pause / takeover: "pause bot", "hold bot", "stop bot"
    - resume: "resume bot", "start bot"
    - force_confirm: "confirm booking", "force confirm"
    """
    if not text or not isinstance(text, str):
        return None

    clean = text.strip()
    clean_lower = clean.lower()

    import re

    # 1. Price Override
    # e.g. "2000 ok", "2000 rate", "2000 me dedo", "rate 2000", "give for 2500", "offer 1800"
    m_price_prefix = re.search(r'\b(?:rate|price|offer|give\s+for|give\s+at|special\s+rate|give\s+me|dedo|at)\s*[:=]?\s*₹?\s*(\d+(?:\.\d+)?)\b', clean_lower)
    m_price_suffix = re.search(r'\b₹?\s*(\d+(?:\.\d+)?)\s*(?:ok|okay|rate|done|me\s+dedo|final|chalega|fix)\b', clean_lower)
    if m_price_prefix:
        val = float(m_price_prefix.group(1))
        logger.info("STAFF_CMD_DETECTED | command=price_override | value=%s | sess=%s", val, session_id or "-")
        return {"command": "price_override", "value": val}
    if m_price_suffix:
        val = float(m_price_suffix.group(1))
        logger.info("STAFF_CMD_DETECTED | command=price_override | value=%s | sess=%s", val, session_id or "-")
        return {"command": "price_override", "value": val}

    # 2. Discount
    m_disc_pct = re.search(r'\b(\d+(?:\.\d+)?)\s*%\s*(?:discount|off)\b', clean_lower)
    m_disc_fixed = re.search(r'\b(?:discount|off|give)\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:off|discount|less|rupees|rs)?\b', clean_lower)
    if m_disc_pct:
        val = {"type": "percent", "value": float(m_disc_pct.group(1))}
        logger.info("STAFF_CMD_DETECTED | command=discount | value=%s | sess=%s", val, session_id or "-")
        return {"command": "discount", "value": val}
    if m_disc_fixed:
        val = {"type": "fixed", "value": float(m_disc_fixed.group(1))}
        logger.info("STAFF_CMD_DETECTED | command=discount | value=%s | sess=%s", val, session_id or "-")
        return {"command": "discount", "value": val}

    # 3. Skip Payment
    if re.search(r'\b(?:skip\s*payment|no\s*advance|skip\s*advance|allow\s*pay\s*at\s*hotel|pay\s*at\s*checkin|payment\s*bypass|zero\s*advance)\b', clean_lower):
        logger.info("STAFF_CMD_DETECTED | command=skip_payment | value=True | sess=%s", session_id or "-")
        return {"command": "skip_payment", "value": True}

    # 4. Room Upgrade
    m_upg = re.search(r'\b(?:upgrade\s*(?:room\s*)?to|give)\s+([a-zA-Z\s]+?)(?:\s*room|\s*category)?\b', clean_lower)
    if m_upg:
        target_room = m_upg.group(1).strip().title()
        if target_room.lower() not in ("me", "him", "her", "them", "us", "discount", "price", "rate"):
            logger.info("STAFF_CMD_DETECTED | command=room_upgrade | value=%s | sess=%s", target_room, session_id or "-")
            return {"command": "room_upgrade", "value": target_room}

    # 5. Pause / Takeover
    if re.search(r'\b(?:pause\s*bot|stop\s*bot|hold\s*bot|pause|takeover|i\s*will\s*handle)\b', clean_lower):
        logger.info("STAFF_CMD_DETECTED | command=pause | value=True | sess=%s", session_id or "-")
        return {"command": "pause", "value": True}

    # 6. Resume
    if re.search(r'\b(?:resume\s*bot|start\s*bot|resume|unpause\s*bot|bot\s*continue)\b', clean_lower):
        logger.info("STAFF_CMD_DETECTED | command=resume | value=True | sess=%s", session_id or "-")
        return {"command": "resume", "value": True}

    # 7. Force Confirm
    if re.search(r'\b(?:force\s*confirm|confirm\s*booking|staff\s*confirm|book\s*now\s*staff|confirm\s*without\s*payment)\b', clean_lower):
        logger.info("STAFF_CMD_DETECTED | command=force_confirm | value=True | sess=%s", session_id or "-")
        return {"command": "force_confirm", "value": True}

    return None


def is_staff_message(msg: dict[str, Any]) -> bool:
    """
    Generic classifier: CRM sender taxonomy first, then flat-messages sender / legacy role,
    then staff phone whitelist.
    """
    if is_crm_staff_message(msg):
        return True
    # Flat messages use sender="staff"; legacy Chat_Sessions docs use role="agent"/"staff"
    if "sender" in msg or "role" in msg or "Role" in msg:
        return is_mongo_staff_message(msg)
    phone = str(msg.get("sender_phone") or msg.get("phone") or msg.get("from") or "")
    if phone and is_staff_phone(phone):
        return True
    return False


def _message_id(msg: dict[str, Any]) -> str | None:
    mid = msg.get("message_id") or msg.get("messageId") or msg.get("id") or msg.get("_id")
    return str(mid) if mid is not None else None


def _message_timestamp(msg: dict[str, Any]) -> datetime | None:
    raw = msg.get("timestamp") or msg.get("created_at") or msg.get("createdAt") or msg.get("time")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=UTC)
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _message_text(msg: dict[str, Any]) -> str:
    for key in ("text_body", "text", "content", "message", "body"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _staff_id_from_msg(msg: dict[str, Any]) -> int | str | None:
    for key in ("staff_id", "staffId", "agent_id", "agentId", "user_id", "userId", "author_id"):
        if msg.get(key) is not None:
            return msg[key]
    return None


def _normalize(msg: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_id": _message_id(msg),
        "text": _message_text(msg),
        "timestamp": _message_timestamp(msg) or datetime.now(UTC),
        "staff_id": _staff_id_from_msg(msg),
        "raw": msg,
    }


def normalize_with_sender(msg: dict[str, Any]) -> dict[str, Any]:
    """
    `_normalize` plus the sender, for callers that need the staff/guest split.

    `_normalize` drops the sender because every row it handles is already known
    to be staff. A whole-window transcript has to keep it, or every row reads as
    an unknown sender and the guard in `_label_messages` that stops a guest's
    own claim from becoming a staff decision has nothing to work with.
    """
    norm = _normalize(msg)
    norm["sender"] = str(msg.get("sender") or msg.get("Sender") or "").strip()
    return norm


async def fetch_crm_messages(hotel_id: int, customer_phone: str) -> list[dict[str, Any]]:
    """GET /api/backend/messages/?Secretkey=&hotelID=&phone_number="""
    settings = get_settings()
    phone = (customer_phone or "").strip()
    if not hotel_id or not phone:
        return []

    url = crm_messages_list_url()
    params = {
        "Secretkey": settings.BACKEND_API_SECRET_KEY,
        "hotelID": str(hotel_id),
        "phone_number": phone,
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url, params=params)
            if response.status_code != 200:
                logger.warning(
                    "CRM_MESSAGES_HTTP_%s | body=%s",
                    response.status_code,
                    (response.text or "")[:200],
                )
                return []
            data = response.json()
    except Exception:
        logger.warning("CRM_MESSAGES_FETCH_FAILED | hotel_id=%s", hotel_id, exc_info=True)
        return []

    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if isinstance(data, dict):
        msgs = data.get("messages") or data.get("data") or []
        return [m for m in msgs if isinstance(m, dict)]
    return []


async def _load_staff_messages(hotel_id: int | str) -> list[dict[str, Any]]:
    """Load recent staff messages from the normalized `messages` collection (sender="staff")."""
    str_hid = str(hotel_id)
    try:
        msg_col = messages_collection()
        # Find messages for this hotel with sender="staff"
        cursor = msg_col.find({
            "hotel_id": str_hid,
            "sender": "staff",
        }).sort("created_at", -1).limit(50)
        canon_msgs = []
        async for m in cursor:
            canon_msgs.append({
                "role": "agent",
                "text": m.get("content", ""),
                "timestamp": m.get("created_at"),
                "message_id": m.get("message_id"),
            })
        return canon_msgs
    except Exception as e:
        logger.debug(f"Canonical messages fetch notice: {e}")
        return []


async def fetch_staff_messages(
    hotel_id: int,
    customer_phone: str,
    *,
    since: datetime | None = None,
    exclude_message_ids: set[str] | None = None,
    window_out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Staff messages for Human Takeover, oldest → newest.

    Primary: CRM API sender="user"
    Fallback: Mongo `messages` sender="staff" (if CRM empty/unavailable)

    `window_out`, when given, is filled with the *whole* CRM conversation window
    — guest and bot turns included — instead of only the staff rows. The resume
    path needs the guest turns too: a staff line only means something next to
    the request it answers, and re-fetching the same URL to get them would put
    a second HTTP round trip on every guest message.
    """
    if not hotel_id or not customer_phone:
        return []

    exclude = exclude_message_ids or set()
    found: dict[str, dict[str, Any]] = {}

    crm_msgs = await fetch_crm_messages(hotel_id, customer_phone)
    if window_out is not None:
        window_out.extend(normalize_with_sender(m) for m in crm_msgs)
    staff_from_crm = 0
    for msg in crm_msgs:
        if not is_crm_staff_message(msg):
            continue
        staff_from_crm += 1
        norm = _normalize(msg)
        key = norm["message_id"] or f"crm:{norm['text'].lower()}:{norm['timestamp'].isoformat()}"
        found[key] = norm

    # Fallback / supplement from Mongo sender="staff" when CRM returned nothing useful
    if staff_from_crm == 0:
        for msg in await _load_staff_messages(hotel_id):
            if not is_mongo_staff_message(msg):
                continue
            norm = _normalize(msg)
            key = norm["message_id"] or f"mongo:{norm['text'].lower()}:{norm['timestamp'].isoformat()}"
            found.setdefault(key, norm)

    if crm_msgs:
        logger.info(
            "CRM_MESSAGES_SYNC | hotel_id=%s | total=%s | staff_user=%s",
            hotel_id,
            len(crm_msgs),
            staff_from_crm,
        )

    results: list[dict[str, Any]] = []
    for norm in found.values():
        mid = norm.get("message_id")
        if mid and mid in exclude:
            continue
        ts = norm.get("timestamp")
        if since is not None and isinstance(ts, datetime):
            since_aware = since if since.tzinfo else since.replace(tzinfo=UTC)
            ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
            if ts_aware <= since_aware:
                continue
        results.append(norm)

    results.sort(key=lambda m: m["timestamp"])
    return results
