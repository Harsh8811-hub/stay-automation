import logging
from typing import Optional, Dict, Any
import httpx

from app.config.settings import get_settings

from datetime import datetime, UTC

logger = logging.getLogger(__name__)
settings = get_settings()

def _parse_datetime(val: Any) -> datetime | None:
    """Parse various timestamp formats (datetime, float/int unix, ISO strings) into UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=UTC)
    if isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            return None
        try:
            dt = datetime.fromisoformat(val_clean.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(val_clean, fmt)
                return dt.replace(tzinfo=UTC)
            except ValueError:
                pass
    return None


def _history_identity(m: Dict[str, Any]) -> tuple:
    """What makes two history rows the same turn.

    Speaker, text and timestamp *together*. Text alone is not enough: "ok" from
    the guest and "ok" from the bot one minute later are two different turns, and
    collapsing them would silently shorten the transcript. The three field-name
    spellings are because the two sources were written by different teams —
    the bot's `messages` uses content/created_at, the CRM uses text/timestamp.
    """
    text = m.get("content") or m.get("text") or m.get("text_body") or m.get("message") or ""
    who = m.get("sender") or m.get("role") or m.get("direction") or m.get("sender_type") or ""
    when = m.get("created_at") or m.get("timestamp") or m.get("time") or ""
    return (str(who).strip().lower(), str(text).strip(), str(when))


def _history_sort_key(m: Dict[str, Any]):
    """Oldest first. Rows whose timestamp cannot be read sort to the front."""
    when = _parse_datetime(
        m.get("created_at") or m.get("timestamp") or m.get("time")
    )
    return when or datetime.min.replace(tzinfo=UTC)


def _merge_history(*sources: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Concatenate history sources, drop exact duplicates, order oldest first.

    Duplicates are expected, not exceptional: the CRM mirrors turns the bot also
    stores, so a merged transcript contains each of them twice unless they are
    collapsed here. The sort is stable, so rows sharing a timestamp keep the
    order their source returned them in.
    """
    seen: set[tuple] = set()
    merged: list[Dict[str, Any]] = []
    for source in sources:
        for m in source or []:
            if not isinstance(m, dict):
                continue
            key = _history_identity(m)
            if key in seen:
                continue
            seen.add(key)
            merged.append(m)
    merged.sort(key=_history_sort_key)
    return merged


class CRMService:
    """Service to handle interactions with the CRM backend (like sending media)."""

    def __init__(self):
        self.url = settings.CRM_BACKEND_URL
        self.secret_key = settings.BACKEND_API_SECRET_KEY

    async def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Base method to send the request to the CRM backend."""
        from app.observability.failure_store import record_failure
        from app.observability.metrics import metrics, record_outbound_send
        from app.observability.tracing import mask_phone

        # Ensure the secret key is always in the payload
        payload["Secretkey"] = self.secret_key
        msg_type = payload.get("message_type") or "unknown"
        hotel_id = payload.get("hotelID")
        phone = payload.get("phone_number")

        try:
            async with httpx.AsyncClient() as client:
                logger.info(
                    "CRM_SEND_START | url=%s hotel=%s type=%s to=%s",
                    self.url,
                    hotel_id,
                    msg_type,
                    mask_phone(str(phone) if phone else None),
                )
                with metrics.timer("crm.send_latency_ms", message_type=msg_type, hotel_id=hotel_id):
                    response = await client.post(self.url, json=payload, timeout=10.0)
                body_preview = (response.text or "")[:400]
                if response.status_code >= 400:
                    logger.error(
                        "CRM_SEND_HTTP_ERROR | status=%s type=%s body=%s",
                        response.status_code,
                        msg_type,
                        body_preview,
                    )
                    record_outbound_send(
                        channel="crm",
                        message_type=str(msg_type),
                        outcome="error",
                        hotel_id=hotel_id,
                    )
                    record_failure(
                        component="crm.send",
                        error=f"http_{response.status_code}",
                        hotel_id=hotel_id,
                        phone=str(phone) if phone else None,
                        details={"message_type": msg_type, "body": body_preview},
                    )
                    response.raise_for_status()

                data: Dict[str, Any]
                try:
                    parsed = response.json()
                    data = parsed if isinstance(parsed, dict) else {"raw": parsed}
                except Exception:
                    data = {"raw": body_preview}

                raw_nested = data.get("data")
                nested = raw_nested if isinstance(raw_nested, dict) else {}
                delivery_status = (
                    data.get("status")
                    or data.get("delivery_status")
                    or nested.get("status")
                )
                err_msg = data.get("error") or data.get("message") or nested.get("error")

                outcome = "ok"
                if isinstance(delivery_status, str) and delivery_status.lower() in (
                    "failed",
                    "error",
                    "undelivered",
                ):
                    outcome = "error"
                    record_failure(
                        component="crm.send",
                        error=f"delivery_{delivery_status}",
                        hotel_id=hotel_id,
                        phone=str(phone) if phone else None,
                        details={"message_type": msg_type, "response": str(data)[:400]},
                    )
                    logger.error(
                        "CRM_SEND_DELIVERY_FAIL | hotel=%s type=%s status=%s error=%s",
                        hotel_id,
                        msg_type,
                        delivery_status,
                        err_msg,
                    )
                else:
                    logger.info(
                        "CRM_SEND_OK | hotel=%s type=%s status=%s error_field=%s",
                        hotel_id,
                        msg_type,
                        delivery_status or response.status_code,
                        err_msg if err_msg and str(err_msg).lower() != "unknown error" else None,
                    )

                record_outbound_send(
                    channel="crm",
                    message_type=str(msg_type),
                    outcome=outcome,
                    hotel_id=hotel_id,
                )
                return data if isinstance(data, dict) else {"raw": data}
        except httpx.HTTPStatusError as e:
            # Already recorded above when status >= 400
            if e.response is None or e.response.status_code < 400:
                record_outbound_send(
                    channel="crm",
                    message_type=str(msg_type),
                    outcome="error",
                    hotel_id=hotel_id,
                )
                record_failure(
                    component="crm.send",
                    error=e,
                    hotel_id=hotel_id,
                    phone=str(phone) if phone else None,
                    details={"message_type": msg_type},
                    exc=e,
                )
            logger.error(
                "HTTP error occurred while calling CRM backend: %s",
                e.response.text if e.response is not None else e,
            )
            raise
        except Exception as e:
            record_outbound_send(
                channel="crm",
                message_type=str(msg_type),
                outcome="error",
                hotel_id=hotel_id,
            )
            record_failure(
                component="crm.send",
                error=e,
                hotel_id=hotel_id,
                phone=str(phone) if phone else None,
                details={"message_type": msg_type},
                exc=e,
            )
            logger.error(f"Unexpected error calling CRM backend: {str(e)}", exc_info=True)
            raise

    async def send_audio(self, phone_number: str, audio_link: str, hotel_id: str) -> Dict[str, Any]:
        """Send an audio file via a public URL."""
        payload = {
            "hotelID": str(hotel_id),
            "phone_number": phone_number,
            "message_type": "audio",
            "media_metadata": {
                "link": audio_link
            }
        }
        return await self._send_request(payload)

    async def send_image(self, phone_number: str, image_link: str, hotel_id: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """Send an image via a public URL."""
        payload = {
            "hotelID": str(hotel_id),
            "phone_number": phone_number,
            "message_type": "image",
            "media_metadata": {
                "link": image_link
            }
        }
        if caption:
            payload["media_metadata"]["caption"] = caption
            
        return await self._send_request(payload)

    async def send_video(self, phone_number: str, video_link: str, hotel_id: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """Send a video via a public URL."""
        payload = {
            "hotelID": str(hotel_id),
            "phone_number": phone_number,
            "message_type": "video",
            "media_metadata": {
                "link": video_link
            }
        }
        if caption:
            payload["media_metadata"]["caption"] = caption
            
        return await self._send_request(payload)

    async def send_document_url(self, phone_number: str, doc_link: str, filename: str, hotel_id: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """Send a document/PDF via a public URL."""
        payload = {
            "hotelID": str(hotel_id),
            "phone_number": phone_number,
            "message_type": "document",
            "media_metadata": {
                "link": doc_link,
                "filename": filename
            }
        }
        if caption:
            payload["media_metadata"]["caption"] = caption
            
        return await self._send_request(payload)

    async def send_document_base64(self, phone_number: str, base64_data: str, filename: str, hotel_id: str) -> Dict[str, Any]:
        """Send a document/PDF via Base64 (No public URL required)."""
        payload = {
            "hotelID": str(hotel_id),
            "phone_number": phone_number,
            "message_type": "document",
            "media_base64": base64_data,
            "media_metadata": {
                "filename": filename
            }
        }
        return await self._send_request(payload)

    async def get_chats_in_timeframe_from_db(
        self,
        hotel_id: str | int,
        phone_number: str,
        start_time: str | datetime,
        end_time: str | datetime,
    ) -> list[Dict[str, Any]]:
        """
        Retrieves all messages in a timeframe from the normalized `messages` collection.

        Flow (bot_migration): guests (phone) -> conversations (guest_id) -> messages (conversation_id),
        filtered by created_at within [start_time, end_time]. Returns plain dicts with keys
        message_id / sender / content / created_at / role (sender is already guest|bot|staff|system).
        """
        from app.utils.phone import phone_lookup_variants
        from app.db.mongodb import (
            guests_collection,
            conversations_collection,
            messages_collection,
        )

        dt_start = _parse_datetime(start_time)
        dt_end = _parse_datetime(end_time)

        phone_clean = str(phone_number).strip()
        variants = phone_lookup_variants(phone_clean) or [phone_clean]
        str_hotel_id = str(hotel_id)

        try:
            # 1. Resolve phone -> guest (no guest for this hotel/phone => no history)
            guest = await guests_collection().find_one({
                "hotel_id": str_hotel_id,
                "phone": {"$in": variants},
            })
            if not guest:
                return []

            # 2. Resolve guest -> conversation_ids (conversations join on guest_id string)
            guest_id = guest.get("guest_id")
            conv_cursor = conversations_collection().find(
                {"hotel_id": str_hotel_id, "guest_id": guest_id},
                projection={"_id": 0, "conversation_id": 1},
            )
            conv_ids = [doc.get("conversation_id") for doc in await conv_cursor.to_list(length=None)]
            conv_ids = [cid for cid in conv_ids if cid is not None]
            if not conv_ids:
                return []

            # 3. Query flat messages within the timeframe, oldest -> newest
            query: Dict[str, Any] = {"conversation_id": {"$in": conv_ids}}
            if dt_start is not None and dt_end is not None:
                query["created_at"] = {"$gte": dt_start, "$lte": dt_end}
            elif dt_start is not None:
                query["created_at"] = {"$gte": dt_start}
            elif dt_end is not None:
                query["created_at"] = {"$lte": dt_end}

            cursor = messages_collection().find(query).sort("created_at", 1)
            messages = []
            async for m in cursor:
                sender = m.get("sender")
                messages.append({
                    "message_id": m.get("message_id"),
                    "sender": sender,
                    "content": m.get("content"),
                    "created_at": m.get("created_at"),
                    "role": sender,
                })

            logger.info(
                "CRM_DB_FETCH_HISTORY_OK | col=messages hotel=%s phone=%s found=%s",
                hotel_id,
                phone_number,
                len(messages),
            )
            return messages
        except Exception as e:
            logger.warning(f"Failed to fetch chat history from messages collection: {e}")
            return []

    async def fetch_chat_history(
        self,
        hotel_id: str,
        phone_number: str,
        start_time: str,
        end_time: str
    ) -> list[Dict[str, Any]]:
        """
        Fetch chat history for context refresh.

        BOTH sources are queried and the results merged — they are complementary,
        not alternatives, which is what the old "priority / fallback" wording got
        wrong. The bot's own `messages` collection holds the guest and bot turns;
        a staff member's turns during a takeover exist *only* in the CRM. Because
        the Mongo query returns rows for any guest who has ever spoken, treating
        the CRM as a fallback meant it was never called for exactly the guests a
        context refresh exists to summarise, and every staff message was
        invisible. Measured on production: the refresh logged
        `CONTEXT_REFRESH_QUEUED` and then died, with no
        `CRM_FETCH_HISTORY_HTTP` line, for a guest with history.
        """
        db_messages: list[Dict[str, Any]] = []
        try:
            db_messages = await self.get_chats_in_timeframe_from_db(
                hotel_id=hotel_id,
                phone_number=phone_number,
                start_time=start_time,
                end_time=end_time,
            ) or []
        except Exception as e:
            logger.warning("CRM_DB_FETCH_EXCEPTION | error=%s", e)

        crm_messages = await self._fetch_history_from_crm(
            hotel_id=hotel_id,
            phone_number=phone_number,
            start_time=start_time,
            end_time=end_time,
        )

        return _merge_history(db_messages, crm_messages)

    async def _fetch_history_from_crm(
        self,
        hotel_id: str,
        phone_number: str,
        start_time: str,
        end_time: str,
    ) -> list[Dict[str, Any]]:
        """The CRM side of `fetch_chat_history`.

        Calls the endpoint the Human Takeover path already proves works —
        `GET /api/backend/messages/`, via `fetch_crm_messages` — and NOT
        `settings.CRM_CHAT_HISTORY_URL`. That setting points at
        `.../messages/history/`, which the CRM does not implement: measured
        2026-10-17 it answers **404 for every guest**, with and without the time
        params, so the context refresh had no staff turns to summarise no matter
        how the merge was written. The same host's `/messages/` answers 200 and
        returned 414 rows (157 guest / 255 bot / 2 staff) for one guest.

        The endpoint ignores the time window and returns the whole conversation,
        so the window is applied here. Rows whose timestamp cannot be read are
        kept rather than dropped — losing the one staff turn in the window is a
        worse failure than summarising a turn from outside it.

        Returns `[]` on every failure path: a CRM that is down or slow must
        degrade to "the bot's own history alone", never take the refresh with it.
        """
        from app.services.staff_message_detector import fetch_crm_messages

        try:
            rows = await fetch_crm_messages(
                int(hotel_id) if str(hotel_id).isdigit() else hotel_id,
                phone_number,
            )
        except Exception as e:
            logger.warning(
                "CRM_HISTORY_FETCH_EXCEPTION | error=%s | hotel=%s | phone=%s",
                e, hotel_id, phone_number,
            )
            return []

        dt_start = _parse_datetime(start_time)
        dt_end = _parse_datetime(end_time)
        if dt_start is None and dt_end is None:
            return rows

        windowed: list[Dict[str, Any]] = []
        for m in rows:
            if not isinstance(m, dict):
                continue
            when = _parse_datetime(m.get("created_at") or m.get("timestamp") or m.get("time"))
            if when is not None:
                if dt_start is not None and when < dt_start:
                    continue
                if dt_end is not None and when > dt_end:
                    continue
            windowed.append(m)

        logger.info(
            "CRM_HISTORY_WINDOWED | hotel=%s | to=%s | fetched=%s | in_window=%s",
            hotel_id, phone_number, len(rows), len(windowed),
        )
        return windowed


# Singleton instance
crm_service = CRMService()

