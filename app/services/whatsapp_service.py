import os
import logging
import requests
from typing import Optional, Dict, Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GRAPH_API_VERSION = os.getenv("WHATSAPP_GRAPH_API_VERSION", "v19.0").strip() or "v19.0"


class WhatsAppService:
    """
    Native WhatsApp sending service.
    Handles routing messages via ComBot/Meta and Twilio using environment credentials.
    """

    def send_message(
        self,
        to_number: str,
        message_body: str,
        sender_id: Optional[str] = None,
        hotel_id: Optional[str | int] = None,
    ) -> bool:
        """
        Smart router: Send via Twilio OR Meta based on which number received the message
        """
        from app.observability.metrics import record_outbound_send
        from app.observability.tracing import mask_phone

        receiver = str(to_number).replace("whatsapp:", "").replace("+", "").strip()
        sender = (
            str(sender_id).replace("whatsapp:", "").replace("+", "").strip()
            if sender_id
            else None
        )

        twilio_num = (
            os.getenv("TWILIO_WHATSAPP_NUMBER", "")
            .replace("whatsapp:", "")
            .replace("+", "")
            .strip()
        )

        # If it was received on the Twilio number, reply via Twilio
        if sender and twilio_num and sender == twilio_num:
            ok = self._send_via_twilio(receiver, message_body, twilio_num)
            record_outbound_send(
                channel="twilio",
                message_type="text",
                outcome="ok" if ok else "error",
                hotel_id=hotel_id,
            )
            return ok

        # Default to Meta/ComBot
        if not hotel_id:
            hotel_id = str(settings.HOTEL_ID)
        else:
            hotel_id = str(hotel_id)

        # ── T0 tenant guard: never send one hotel's content from another's number ──
        if sender:
            owner = self.resolve_hotel_id_strict(sender)
            if owner is None or str(owner) != hotel_id:
                from app.observability.failure_store import record_failure
                logger.error(
                    "TENANT_MISMATCH_BLOCKED | content_hotel=%s sender=%s owner=%s to=%s",
                    hotel_id, sender, owner, mask_phone(receiver),
                )
                record_failure(
                    component="whatsapp.tenant_guard",
                    error="tenant_mismatch_blocked",
                    hotel_id=hotel_id,
                    phone=receiver,
                    details={"content_hotel": hotel_id, "sender_id": sender, "sender_owner": owner},
                )
                record_outbound_send(
                    channel="meta_combot",
                    message_type="text",
                    outcome="blocked_tenant_mismatch",
                    hotel_id=hotel_id,
                )
                return False

        # WhatsApp Cloud API hard limit
        if len(message_body) > 4096:
            message_body = message_body[:4090] + "…"

        logger.info(
            "WA_SEND_TEXT_START | hotel=%s to=%s chars=%s sender=%s",
            hotel_id,
            mask_phone(receiver),
            len(message_body),
            sender or "default",
        )
        ok = self._send_via_meta(receiver, message_body, hotel_id=hotel_id, sender_id=sender)
        record_outbound_send(
            channel="meta_combot",
            message_type="text",
            outcome="ok" if ok else "error",
            hotel_id=hotel_id,
        )
        return ok

    def resolve_hotel_id_strict(
        self,
        phone_number_id: str | None = None,
    ) -> int | None:
        """
        Map phone_number_id to hotel_id strictly. Returns None when
        phone_number_id matches no configured hotel, instead of silently
        assuming the default hotel (T0 — cross-tenant leak).
        """
        pid = (phone_number_id or "").strip()
        if not pid:
            return None

        # Check environment variables first (as configured on VM)
        default_pid = os.getenv("META_COM_PHONE_NUMBER_ID", "").strip()
        if default_pid and default_pid == pid:
            try:
                return int(settings.HOTEL_ID)
            except ValueError:
                return None

        prefix = "META_COM_PHONE_NUMBER_ID_"
        for key, val in os.environ.items():
            if key.startswith(prefix) and val.strip() == pid:
                suffix = key[len(prefix):]
                try:
                    return int(suffix)
                except ValueError:
                    continue

        from app.services.property_router import property_router
        resolved = property_router.resolve_hotel_id(
            phone_number_id=pid,
            allow_default_fallback=False,
        )
        if resolved is not None:
            try:
                return int(resolved)
            except ValueError:
                return None

        return None

    def resolve_hotel_id(
        self,
        phone_number_id: str | None = None,
        display_phone_number: str | None = None,
        waba_id: str | None = None,
        property_code: str | None = None,
        allow_default_fallback: bool = True,
    ) -> str:
        """
        Map inbound WhatsApp metadata -> hotel_id using dynamic PropertyRouter.
        """
        from app.services.property_router import property_router
        resolved = property_router.resolve_hotel_id(
            phone_number_id=phone_number_id,
            display_phone_number=display_phone_number,
            waba_id=waba_id,
            property_code=property_code,
            allow_default_fallback=allow_default_fallback,
        )
        if resolved is not None:
            return resolved
        return str(settings.HOTEL_ID)

    def _access_token_for_hotel(self, hotel_id: str | int) -> str:
        hotel_id_str = str(hotel_id) if hotel_id else ""
        if hotel_id_str == "37292":
            hotel_id_str = "376892"

        suffix = f"_{hotel_id_str}" if hotel_id_str else ""
        token = os.getenv(f"COMBOT_API_TOKEN{suffix}", "").strip()
        if token:
            return token

        for key in (
            "COMBOT_API_TOKEN",
            "WHATSAPP_ACCESS_TOKEN",
            "META_ACCESS_TOKEN",
            "META_COM_ACCESS_TOKEN",
        ):
            tok = os.getenv(key, "").strip()
            if tok:
                return tok
        return ""

    def download_media(
        self,
        media_id: str,
        hotel_id: str | int | None = None,
    ) -> tuple[bytes, str, str] | None:
        """
        Download inbound WhatsApp media via Graph API.
        Returns (bytes, mime_type, filename) or None on failure.
        """
        from app.observability.failure_store import record_failure

        if not media_id:
            return None

        hotel_id = hotel_id or settings.HOTEL_ID
        token = self._access_token_for_hotel(hotel_id)
        if not token:
            logger.error("WhatsApp media download: missing access token hotel=%s", hotel_id)
            record_failure(
                component="whatsapp.media_download",
                error="missing_access_token",
                hotel_id=hotel_id,
                details={"media_id": media_id[:24]},
            )
            return None

        headers = {"Authorization": f"Bearer {token}"}
        meta_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"

        try:
            meta_resp = requests.get(meta_url, headers=headers, timeout=20)
            if meta_resp.status_code != 200:
                logger.error(
                    "WA_MEDIA_META_FAIL | status=%s body=%s",
                    meta_resp.status_code,
                    meta_resp.text[:300],
                )
                record_failure(
                    component="whatsapp.media_download",
                    error=f"meta_status_{meta_resp.status_code}",
                    hotel_id=hotel_id,
                    details={"body": meta_resp.text[:300], "media_id": media_id[:24]},
                )
                return None

            meta = meta_resp.json()
            media_url = meta.get("url")
            mime = meta.get("mime_type") or "application/octet-stream"
            if not media_url:
                logger.error(f"WhatsApp media meta missing url: {meta}")
                return None

            bin_resp = requests.get(media_url, headers=headers, timeout=60)
            if bin_resp.status_code != 200:
                logger.error(
                    f"WhatsApp media binary failed: {bin_resp.status_code}"
                )
                record_failure(
                    component="whatsapp.media_download",
                    error=f"binary_status_{bin_resp.status_code}",
                    hotel_id=hotel_id,
                )
                return None

            ext = _ext_from_mime(mime)
            if not ext:
                from app.services.image_vision import _detect_mime_type
                sniffed_mime = _detect_mime_type(bin_resp.content)
                ext = _ext_from_mime(sniffed_mime) or ".jpg"
                mime = sniffed_mime or mime

            filename = f"whatsapp_media_{media_id[:12]}{ext}"
            logger.info(
                "WA_MEDIA_DOWNLOAD_OK | hotel=%s mime=%s ext=%s bytes=%s",
                hotel_id,
                mime,
                ext,
                len(bin_resp.content),
            )
            return bin_resp.content, mime, filename
        except Exception as e:
            record_failure(
                component="whatsapp.media_download",
                error=e,
                hotel_id=hotel_id,
                exc=e,
            )
            logger.error(f"WhatsApp media download exception: {e}", exc_info=True)
            return None

    def send_image_bytes(
        self,
        *,
        to_number: str,
        image_bytes: bytes,
        mime_type: str,
        filename: str,
        caption: str = "",
        sender_id: Optional[str] = None,
        hotel_id: Optional[str | int] = None,
    ) -> bool:
        """Upload an image to Meta and send it by media id, never by object-storage URL."""
        from app.observability.failure_store import record_failure
        from app.observability.metrics import record_outbound_send
        from app.observability.tracing import mask_phone

        receiver = str(to_number).replace("whatsapp:", "").replace("+", "").strip()
        resolved_hotel_id = str(hotel_id or settings.HOTEL_ID)
        _, configured_phone_id, token = self._meta_combot_creds_for_hotel(resolved_hotel_id)
        phone_id = (
            str(sender_id).replace("whatsapp:", "").replace("+", "").strip()
            if sender_id
            else configured_phone_id
        )
        # ── T0 tenant guard: never send one hotel's content from another's number ──
        if sender_id:
            sender_norm = str(sender_id).replace("whatsapp:", "").replace("+", "").strip()
            owner = self.resolve_hotel_id_strict(sender_norm)
            if owner is None or str(owner) != resolved_hotel_id:
                logger.error(
                    "TENANT_MISMATCH_BLOCKED | content_hotel=%s sender=%s owner=%s to=%s",
                    resolved_hotel_id, sender_norm, owner, mask_phone(receiver),
                )
                record_failure(
                    component="whatsapp.tenant_guard",
                    error="tenant_mismatch_blocked",
                    hotel_id=resolved_hotel_id,
                    phone=receiver,
                    details={"content_hotel": resolved_hotel_id, "sender_id": sender_norm, "sender_owner": owner},
                )
                record_outbound_send(
                    channel="graph",
                    message_type="image",
                    outcome="blocked_tenant_mismatch",
                    hotel_id=resolved_hotel_id,
                )
                return False

        if not receiver or not phone_id or not token:
            logger.error(
                "WA_IMAGE_CREDS_MISSING | hotel=%s has_phone_id=%s has_token=%s to=%s",
                resolved_hotel_id,
                bool(phone_id),
                bool(token),
                mask_phone(receiver),
            )
            record_outbound_send(
                channel="graph",
                message_type="image",
                outcome="missing_creds",
                hotel_id=resolved_hotel_id,
            )
            record_failure(
                component="whatsapp.image_send",
                error="missing_credentials",
                hotel_id=resolved_hotel_id,
                phone=receiver,
                details={"has_phone_id": bool(phone_id), "has_token": bool(token)},
            )
            return False
        if mime_type not in {"image/jpeg", "image/png"}:
            logger.warning("WhatsApp image send: unsupported mime_type=%s", mime_type)
            record_outbound_send(
                channel="graph",
                message_type="image",
                outcome="skipped",
                hotel_id=resolved_hotel_id,
            )
            return False
        if not image_bytes:
            logger.warning("WhatsApp image send: empty image payload")
            return False

        headers = {"Authorization": f"Bearer {token}"}
        graph_base = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_id}"

        try:
            logger.info(
                "WA_IMAGE_UPLOAD_START | hotel=%s phone_id=%s…%s mime=%s bytes=%s to=%s",
                resolved_hotel_id,
                phone_id[:4] if len(phone_id) > 4 else phone_id,
                phone_id[-4:] if len(phone_id) > 4 else "",
                mime_type,
                len(image_bytes),
                mask_phone(receiver),
            )
            upload = requests.post(
                f"{graph_base}/media",
                headers=headers,
                data={"messaging_product": "whatsapp"},
                files={"file": (filename or "hotel-image", image_bytes, mime_type)},
                timeout=10,
            )
            if upload.status_code not in (200, 201):
                body = upload.text[:400]
                logger.error(
                    "WA_IMAGE_UPLOAD_FAIL | status=%s body=%s",
                    upload.status_code,
                    body,
                )
                record_outbound_send(
                    channel="graph",
                    message_type="image",
                    outcome="error",
                    hotel_id=resolved_hotel_id,
                )
                record_failure(
                    component="whatsapp.image_upload",
                    error=f"upload_status_{upload.status_code}",
                    hotel_id=resolved_hotel_id,
                    phone=receiver,
                    details={"body": body, "phone_id_suffix": phone_id[-6:]},
                )
                return False

            media_id = (upload.json() or {}).get("id")
            if not media_id:
                logger.error("WhatsApp image upload response had no media id")
                record_outbound_send(
                    channel="graph",
                    message_type="image",
                    outcome="error",
                    hotel_id=resolved_hotel_id,
                )
                return False

            image_payload: Dict[str, Any] = {"id": media_id}
            if caption:
                image_payload["caption"] = caption[:1024]
            sent = requests.post(
                f"{graph_base}/messages",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "messaging_product": "whatsapp",
                    "to": receiver,
                    "type": "image",
                    "image": image_payload,
                },
                timeout=10,
            )
            if sent.status_code in (200, 201):
                wamid = None
                try:
                    wamid = ((sent.json() or {}).get("messages") or [{}])[0].get("id")
                except Exception:
                    pass
                logger.info(
                    "WA_IMAGE_SENT | to=%s hotel=%s wamid=%s",
                    mask_phone(receiver),
                    resolved_hotel_id,
                    wamid or "-",
                )
                record_outbound_send(
                    channel="graph",
                    message_type="image",
                    outcome="ok",
                    hotel_id=resolved_hotel_id,
                )
                return True
            body = sent.text[:400]
            logger.error("WA_IMAGE_SEND_FAIL | status=%s body=%s", sent.status_code, body)
            record_outbound_send(
                channel="graph",
                message_type="image",
                outcome="error",
                hotel_id=resolved_hotel_id,
            )
            record_failure(
                component="whatsapp.image_send",
                error=f"send_status_{sent.status_code}",
                hotel_id=resolved_hotel_id,
                phone=receiver,
                details={"body": body},
            )
            return False
        except (requests.RequestException, ValueError) as e:
            record_outbound_send(
                channel="graph",
                message_type="image",
                outcome="error",
                hotel_id=resolved_hotel_id,
            )
            record_failure(
                component="whatsapp.image_send",
                error=e,
                hotel_id=resolved_hotel_id,
                phone=receiver,
                exc=e,
            )
            logger.exception("WhatsApp image send exception")
            return False

    def _resolve_meta_com_api_url(self) -> str:
        primary = os.getenv("META_COM_API_URL", "").strip()
        if primary:
            return primary.rstrip("/")
        crm = os.getenv("COMBOT_CRM_API_URL", "").strip().rstrip("/")
        if not crm:
            return ""
        if crm.endswith("/meta") or "/meta/" in (crm + "/"):
            return crm
        if "/v1/" in crm or crm.endswith("/messages"):
            return crm
        return f"{crm}/meta"

    def _meta_combot_creds_for_hotel(self, hotel_id_str: str) -> tuple[str, str, str]:
        hotel_id_str = str(hotel_id_str) if hotel_id_str else ""
        if hotel_id_str == "37292":
            hotel_id_str = "376892"

        suffix = f"_{hotel_id_str}" if hotel_id_str else ""

        api_url = self._resolve_meta_com_api_url()
        phone_id = os.getenv(f"META_COM_PHONE_NUMBER_ID{suffix}", "").strip()
        token = os.getenv(f"COMBOT_API_TOKEN{suffix}", "").strip()

        if not phone_id and hotel_id_str == "111111":
            phone_id = os.getenv("META_COM_PHONE_NUMBER_ID", "").strip()

        if not token:
            token = self._access_token_for_hotel(hotel_id_str)

        return api_url, phone_id, token

    def _send_via_twilio(self, to_number: str, message_body: str, twilio_number: str) -> bool:
        from app.observability.failure_store import record_failure
        from app.observability.tracing import mask_phone

        try:
            from twilio.rest import Client

            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")

            if not account_sid or not auth_token or not twilio_number:
                logger.error("Twilio credentials missing")
                record_failure(component="whatsapp.twilio", error="missing_credentials")
                return False

            client = Client(account_sid, auth_token)
            message = client.messages.create(
                from_=f"whatsapp:+{twilio_number}",
                body=message_body,
                to=f"whatsapp:+{to_number}",
            )
            logger.info("WA_TWILIO_SENT | to=%s sid=%s", mask_phone(to_number), message.sid)
            return True
        except Exception as e:
            record_failure(component="whatsapp.twilio", error=e, phone=to_number, exc=e)
            logger.error(f"Twilio error: {e}", exc_info=True)
            return False

    def _send_via_meta(
        self,
        to_number: str,
        message_body: str,
        hotel_id: str,
        sender_id: Optional[str] = None,
    ) -> bool:
        from app.observability.failure_store import record_failure
        from app.observability.tracing import mask_phone

        api_url, configured_phone_id, token = self._meta_combot_creds_for_hotel(hotel_id)
        phone_id = (
            str(sender_id).replace("whatsapp:", "").replace("+", "").strip()
            if sender_id
            else configured_phone_id
        )

        if not all([api_url, phone_id, token]):
            logger.error(
                "WA_META_CREDS_MISSING | hotel=%s has_url=%s has_phone_id=%s has_token=%s",
                hotel_id,
                bool(api_url),
                bool(phone_id),
                bool(token),
            )
            record_failure(
                component="whatsapp.meta_send",
                error="missing_credentials",
                hotel_id=hotel_id,
                phone=to_number,
                details={
                    "has_url": bool(api_url),
                    "has_phone_id": bool(phone_id),
                    "has_token": bool(token),
                },
            )
            return False

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {
            "from_phone_number_id": phone_id,
            "to": to_number,
            "type": "text",
            "text": {"body": message_body},
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            if response.status_code in [200, 201]:
                msg_id = None
                try:
                    data = response.json() or {}
                    msg_id = data.get("id") or data.get("message_id") or (
                        (data.get("messages") or [{}])[0].get("id")
                    )
                except Exception:
                    pass
                logger.info(
                    "WA_META_SENT | hotel=%s to=%s status=%s msg_id=%s",
                    hotel_id,
                    mask_phone(to_number),
                    response.status_code,
                    msg_id or "-",
                )
                return True

            body = response.text[:400]
            logger.error(
                "WA_META_FAIL | hotel=%s to=%s status=%s body=%s",
                hotel_id,
                mask_phone(to_number),
                response.status_code,
                body,
            )
            record_failure(
                component="whatsapp.meta_send",
                error=f"status_{response.status_code}",
                hotel_id=hotel_id,
                phone=to_number,
                details={"body": body, "api_host": api_url.split("/")[2] if "://" in api_url else api_url},
            )
            return False
        except Exception as e:
            record_failure(
                component="whatsapp.meta_send",
                error=e,
                hotel_id=hotel_id,
                phone=to_number,
                exc=e,
            )
            logger.error(f"Meta send exception: {e}", exc_info=True)
            return False


def _ext_from_mime(mime: str) -> str:
    mime = (mime or "").lower().split(";")[0].strip()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/pjpeg": ".jpg",
        "image/png": ".png",
        "image/x-png": ".png",
        "image/webp": ".webp",
        "image/heic": ".jpg",
        "application/pdf": ".pdf",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/aac": ".aac",
        "audio/webm": ".webm",
        "video/mp4": ".mp4",
    }
    return mapping.get(mime, "")


def mapped_hotel_ids() -> list[int]:
    """Every hotel id this instance is configured to serve, from the env map."""
    ids: set[int] = set()
    if os.getenv("META_COM_PHONE_NUMBER_ID", "").strip():
        try:
            ids.add(int(settings.HOTEL_ID))
        except ValueError:
            pass
    prefix = "META_COM_PHONE_NUMBER_ID_"
    for key in os.environ:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix.isdigit():
                ids.add(int(suffix))
    return sorted(ids)


# Singleton instance
whatsapp_service = WhatsAppService()
