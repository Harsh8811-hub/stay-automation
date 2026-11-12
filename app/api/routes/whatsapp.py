"""
WhatsApp Cloud API webhook — inbound messages → LangGraph → outbound reply.

Handles:
- Text, interactive (button/list), and media (image/document/audio/video)
- Stable session continuity by hotel + phone
- hotel_id resolution from WhatsApp phone_number_id
- Card formatting (quote, room_options, options, media URLs)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app.repositories import inbound_dedupe_repository, session_repository
from app.services.chat_runner import run_chat_turn, run_media_turn
from app.services.hotel_media_service import has_media_delivery_request, hotel_media_service
from app.services.whatsapp_service import whatsapp_service
from app.utils.phone import digits_only

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webhooks", tags=["WhatsApp Webhooks"])


class WebhookPayload(BaseModel):
    object: str
    entry: list[dict[str, Any]]


# ── Webhook verification ──────────────────────────────────────────

@router.get("/whatsapp")
async def verify_webhook(request: Request):
    """
    Standard Meta Webhook Verification endpoint.
    Meta sends a GET request with hub.mode, hub.challenge, and hub.verify_token.
    """
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == verify_token:
            logger.info("WhatsApp Webhook verified successfully!")
            return Response(content=challenge, media_type="text/plain", status_code=200)
        raise HTTPException(status_code=403, detail="Verification token mismatch")

    raise HTTPException(status_code=400, detail="Missing required query parameters")


# ── Inbound messages ──────────────────────────────────────────────

@router.post("/whatsapp")
async def receive_whatsapp_message(payload: WebhookPayload):
    """
    Receives inbound WhatsApp messages, runs the bot, replies via WhatsApp.
    """
    from app.observability.failure_store import record_failure
    from app.observability.metrics import record_webhook_event
    from app.observability.tracing import bind_context, get_trace_id, mask_phone

    try:
        if payload.object != "whatsapp_business_account":
            return {"status": "ignored", "reason": "not a whatsapp business account event"}

        for entry in payload.entry:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                # Delivery / read receipts — log for correlation, do not run bot
                statuses = value.get("statuses") or []
                for st in statuses:
                    logger.info(
                        "WA_STATUS | id=%s status=%s recipient=%s errors=%s",
                        st.get("id"),
                        st.get("status"),
                        mask_phone(str(st.get("recipient_id") or "")),
                        st.get("errors"),
                    )
                    if st.get("status") in ("failed", "undelivered"):
                        record_failure(
                            component="whatsapp.delivery_status",
                            error=f"status_{st.get('status')}",
                            details={
                                "wamid": st.get("id"),
                                "errors": st.get("errors"),
                                "timestamp": st.get("timestamp"),
                            },
                        )
                        record_webhook_event(
                            event="status",
                            outcome="failed",
                        )

                if "messages" not in value:
                    continue

                waba_id = entry.get("id")
                metadata = value.get("metadata") or {}
                phone_number_id = metadata.get("phone_number_id")
                display_phone_number = metadata.get("display_phone_number")

                from app.services.property_router import property_router
                hotel_id = property_router.resolve_hotel_id(
                    phone_number_id=phone_number_id,
                    display_phone_number=display_phone_number,
                    waba_id=waba_id,
                    allow_default_fallback=False,
                )

                if hotel_id is None:
                    logger.warning(
                        "UNMAPPED_TENANT_WEBHOOK | waba_id=%s phone_id=%s dpn=%s | Ignored to prevent cross-contamination",
                        waba_id,
                        phone_number_id,
                        display_phone_number,
                    )
                    record_webhook_event(
                        event="unmapped_tenant",
                        outcome="ignored",
                    )
                    continue

                business_phone_id = metadata.get("phone_number_id")
                bind_context(hotel_id=hotel_id, channel="whatsapp")

                for message in value["messages"]:
                    logger.info(
                        "WA_INBOUND | hotel=%s type=%s from=%s msg_id=%s trace=%s",
                        hotel_id,
                        message.get("type"),
                        mask_phone(message.get("from")),
                        message.get("id"),
                        get_trace_id(),
                    )
                    # B3a — idempotency. Meta retries when a turn is slow, and a turn here
                    # can take ~52s, so the retry is routine. The claim is taken before the
                    # work, not after: the duplicate arrives while the first attempt is
                    # still running. See app/repositories/inbound_dedupe_repository.py.
                    claim = await inbound_dedupe_repository.claim_message(
                        message.get("id"),
                        hotel_id=hotel_id,
                        phone=digits_only(message.get("from") or ""),
                    )
                    if not claim.should_process:
                        logger.info(
                            "WA_DUPLICATE_SUPPRESSED | hotel=%s type=%s msg_id=%s reason=%s",
                            hotel_id,
                            message.get("type"),
                            claim.wamid,
                            claim.reason,
                        )
                        record_webhook_event(
                            event="message",
                            outcome="duplicate",
                            hotel_id=hotel_id,
                        )
                        continue

                    record_webhook_event(
                        event="message",
                        outcome="received",
                        hotel_id=hotel_id,
                    )
                    await _handle_inbound_message(
                        message=message,
                        hotel_id=hotel_id,
                        business_phone_id=business_phone_id,
                    )

        return {"status": "success", "trace_id": get_trace_id()}

    except Exception as e:
        record_failure(component="whatsapp.webhook", error=e, exc=e)
        record_webhook_event(event="webhook", outcome="error")
        logger.error(f"Error processing WhatsApp webhook: {e}", exc_info=True)
        # Always 200 to Meta so it does not retry forever for app bugs
        return {"status": "error", "trace_id": get_trace_id()}


async def _handle_inbound_message(
    message: dict[str, Any],
    hotel_id: int,
    business_phone_id: str | None,
) -> None:
    from app.observability.failure_store import record_failure
    from app.observability.tracing import bind_context, mask_phone

    sender_phone = digits_only(message.get("from") or "")
    if not sender_phone:
        logger.warning("WhatsApp message missing 'from'")
        return

    message_type = message.get("type") or ""
    session_id = await _resolve_session_id(hotel_id, sender_phone)
    bind_context(session_id=session_id or "-", hotel_id=hotel_id, channel="whatsapp")

    result: dict[str, Any] | None = None

    try:
        if message_type == "text":
            user_text = (message.get("text") or {}).get("body", "") or ""
            if not user_text.strip():
                return
            result = await run_chat_turn(
                message=user_text,
                session_id=session_id,
                hotel_id=hotel_id,
                customer_phone=sender_phone,
            )

        elif message_type == "interactive":
            user_text = _extract_interactive_text(message.get("interactive") or {})
            if not user_text.strip():
                return
            result = await run_chat_turn(
                message=user_text,
                session_id=session_id,
                hotel_id=hotel_id,
                customer_phone=sender_phone,
            )

        elif message_type in ("image", "document", "audio", "video", "sticker"):
            result = await _handle_media_message(
                message=message,
                message_type=message_type,
                hotel_id=hotel_id,
                session_id=session_id,
                sender_phone=sender_phone,
            )

        else:
            logger.info(f"Ignoring unsupported WhatsApp message type={message_type}")
            return
    except Exception as e:
        record_failure(
            component="whatsapp.handle_inbound",
            error=e,
            hotel_id=hotel_id,
            session_id=session_id,
            phone=sender_phone,
            details={"message_type": message_type},
            exc=e,
        )
        logger.error(
            "WA_HANDLE_FAIL | hotel=%s type=%s phone=%s: %s",
            hotel_id,
            message_type,
            mask_phone(sender_phone),
            e,
            exc_info=True,
        )
        return

    if not result:
        return

    bind_context(session_id=result.get("session_id") or session_id or "-")
    cards = result.get("response") or []
    final_text = format_cards_for_whatsapp(cards)
    logger.info(
        "WA_OUTBOUND_PREPARE | session=%s stage=%s cards=%s text_chars=%s",
        result.get("session_id"),
        result.get("booking_stage"),
        [c.get("type") for c in cards if isinstance(c, dict)],
        len(final_text or ""),
    )
    if final_text:
        ok = whatsapp_service.send_message(
            to_number=sender_phone,
            message_body=final_text,
            sender_id=business_phone_id,
            hotel_id=hotel_id,
        )
        if not ok:
            record_failure(
                component="whatsapp.outbound_text",
                error="send_returned_false",
                hotel_id=hotel_id,
                session_id=result.get("session_id"),
                phone=sender_phone,
                details={"chars": len(final_text)},
            )

    delivery = await hotel_media_service.deliver_card_requests(
        cards=cards,
        to_number=sender_phone,
        sender_id=business_phone_id,
        hotel_id=hotel_id,
    )
    if has_media_delivery_request(cards):
        logger.info(
            "WA_MEDIA_DELIVERY | sent=%s skipped=%s requested=1",
            delivery.sent_count,
            delivery.skipped_count,
        )
    if has_media_delivery_request(cards) and delivery.sent_count == 0:
        record_failure(
            component="whatsapp.media_delivery",
            error="zero_images_sent",
            hotel_id=hotel_id,
            session_id=result.get("session_id"),
            phone=sender_phone,
            details={"skipped": delivery.skipped_count},
            severity="warning",
        )
        fallback_msg = _build_photo_unavailable_message(cards)
        whatsapp_service.send_message(
            to_number=sender_phone,
            message_body=fallback_msg,
            sender_id=business_phone_id,
            hotel_id=hotel_id,
        )


async def _resolve_session_id(hotel_id: int, phone: str) -> str | None:
    """Reuse active booking session for this guest at this hotel."""
    try:
        existing = await session_repository.find_active_session(hotel_id, phone)
        if existing:
            logger.info(
                f"WA_SESSION_RESUME | hotel={hotel_id} phone=…{phone[-4:]} "
                f"session={existing.session_id} stage={existing.booking_stage}"
            )
            return existing.session_id
    except Exception as e:
        logger.warning(f"WhatsApp session lookup failed: {e}")
    return None


def _extract_interactive_text(interactive: dict[str, Any]) -> str:
    itype = interactive.get("type")
    if itype == "button_reply":
        br = interactive.get("button_reply") or {}
        val = br.get("id") or br.get("title") or ""
        return str(val).strip()
    if itype == "list_reply":
        lr = interactive.get("list_reply") or {}
        val = lr.get("id") or lr.get("title") or ""
        return str(val).strip()
    return ""


async def _handle_media_message(
    message: dict[str, Any],
    message_type: str,
    hotel_id: int,
    session_id: str | None,
    sender_phone: str,
) -> dict[str, Any] | None:
    media_obj = message.get(message_type) or {}
    # stickers use image-like payload under "sticker"
    if message_type == "sticker":
        media_obj = message.get("sticker") or {}

    media_id = media_obj.get("id")
    caption = (media_obj.get("caption") or "").strip() or None
    mime = media_obj.get("mime_type") or "application/octet-stream"
    filename_hint = media_obj.get("filename")

    if not media_id:
        return await run_chat_turn(
            message=caption or "I sent a file but it could not be read.",
            session_id=session_id,
            hotel_id=hotel_id,
            customer_phone=sender_phone,
        )

    downloaded = whatsapp_service.download_media(media_id, hotel_id=hotel_id)
    if not downloaded:
        return {
            "session_id": session_id or "",
            "response": [{
                "type": "text",
                "text": (
                    "Sorry, I couldn't download your file from WhatsApp. "
                    "Please try sending it again, or describe your request in text."
                ),
            }],
            "booking_stage": "new",
            "booking_summary": None,
        }

    file_bytes, dl_mime, filename = downloaded
    content_type = dl_mime or mime
    if filename_hint:
        import os
        _, ext = os.path.splitext(filename_hint)
        if not ext:
            _, orig_ext = os.path.splitext(filename)
            filename = f"{filename_hint}{orig_ext}"
        else:
            filename = filename_hint

    # ── Strict MIME & Size Validation Gate ────────────────────────
    from app.utils.file_validation import validate_whatsapp_media_bytes
    is_valid, err_msg, detected_mime, category = validate_whatsapp_media_bytes(
        file_bytes=file_bytes,
        filename=filename,
        declared_mime=content_type,
    )
    size_mb = round(len(file_bytes) / (1024 * 1024), 2)
    logger.info(
        "MEDIA_VALIDATE | mime=%s | size_mb=%s | valid=%s | category=%s | sess=%s",
        detected_mime,
        size_mb,
        is_valid,
        category,
        session_id or "-",
    )
    if not is_valid:
        return {
            "session_id": session_id or "",
            "response": [{
                "type": "text",
                "text": err_msg or "Unsupported media format. Please send your document or photo as JPEG, PNG, or PDF.",
            }],
            "booking_stage": "new",
            "booking_summary": None,
        }

    return await run_media_turn(
        file_bytes=file_bytes,
        filename=filename,
        content_type=detected_mime or content_type,
        session_id=session_id,
        message=caption,
        hotel_id=hotel_id,
        customer_phone=sender_phone,
    )


# ── Outbound card → WhatsApp text ─────────────────────────────────

def _build_photo_unavailable_message(cards: list[Any]) -> str:
    """Build a specific 'photos not available' message from media_delivery card metadata."""
    subjects: list[str] = []
    for card in cards or []:
        if not isinstance(card, dict) or card.get("type") != "media_delivery":
            continue
        data = card.get("data") or {}

        # Prefer the human-readable caption (e.g. "Photos of STANDARD NON AC ROOM")
        caption = (data.get("caption") or "").strip()
        if caption:
            # Take only the first line (room_searcher captions are multi-line with price/occupancy)
            caption = caption.splitlines()[0].strip()
            # Strip leading "Photos of " / "Here are some photos of our " prefixes
            for prefix in ("Photos of ", "Here are some photos of our ", "Here are some photos from our "):
                if caption.lower().startswith(prefix.lower()):
                    caption = caption[len(prefix):].strip()
                    break
            # Strip trailing punctuation
            caption = caption.rstrip(".,!?").strip()
            # Skip generic "hotel" label — use the generic fallback message instead
            if caption and caption.lower() != "hotel" and caption not in subjects:
                subjects.append(caption.title())
            continue

        # Fall back to the first tag
        tags = data.get("tags") or []
        if tags:
            label = str(tags[0]).strip().title()
            if label and label not in subjects:
                subjects.append(label)
            continue

        # Fall back to entity_id
        eid = (data.get("entity_id") or "").strip().title()
        if eid and eid not in subjects:
            subjects.append(eid)

    if subjects:
        joined = " and ".join(subjects)
        return f"I'm sorry, photos of *{joined}* are not available at the moment."
    return "I'm sorry, photos are not available at the moment."


def format_cards_for_whatsapp(cards: list[Any]) -> str:
    """Convert bot UICards into a single WhatsApp-friendly text message."""
    final_text = ""

    for card in cards or []:
        if not isinstance(card, dict):
            continue

        ctype = card.get("type")

        if ctype == "text":
            final_text += (card.get("text") or "") + "\n\n"

        elif ctype == "options":
            final_text += (card.get("text") or "") + "\n"
            options = card.get("options") or (card.get("data") or {}).get("options") or []
            for i, opt in enumerate(options, 1):
                if isinstance(opt, dict):
                    label = opt.get("label") or opt.get("title") or opt.get("text") or str(opt)
                else:
                    label = str(opt)
                final_text += f"{i}. {label}\n"
            final_text += "\n"
            final_text = _with_buttons(final_text, card)

        elif ctype == "quote":
            if card.get("text"):
                final_text += str(card.get("text")).strip() + "\n\n"
                final_text = _with_buttons(final_text, card)
            else:
                data = card.get("data") or {}
                total = data.get("total", 0) or 0
                final_text += f"*Total: ₹{_fmt_money(total)}*\n\n_Breakdown:_\n"
                for rm in data.get("rooms") or []:
                    r_name = rm.get("room_name", "Room")
                    qty = rm.get("quantity", 1)
                    ppn = rm.get("price_per_night", 0) or 0
                    nights = rm.get("num_nights", 1) or 1
                    td = rm.get("tax_details") or {}
                    final_amt = td.get("final_amount", (ppn * nights * qty))
                    
                    room_prefix = f"{qty}× " if qty > 1 else ""
                    final_text += (
                        f"🏨 *{room_prefix}{r_name}*\n"
                        f"  ₹{_fmt_money(ppn)}/night × {nights} night(s) = ₹{_fmt_money(final_amt)}\n"
                    )
                    extra_a = rm.get("extra_adult_charge", 0) or 0
                    extra_c = rm.get("extra_child_charge", 0) or 0
                    if extra_a > 0:
                        # A mattress or an unseated adult — the same rupees, a
                        # different sentence. `None` keeps the old wording, which
                        # is what a breakdown without the field was built for.
                        _bed_ct = rm.get("extra_bed_count")
                        _what = (
                            "extra adult bed charge"
                            if (_bed_ct is None or int(_bed_ct) > 0)
                            else "extra person charge"
                        )
                        final_text += f"  (Includes {_what}: ₹{_fmt_money(extra_a)}/night)\n"
                    if extra_c > 0:
                        final_text += f"  (Includes extra child charge: ₹{_fmt_money(extra_c)}/night)\n"
                    final_text += "\n"

                for up in data.get("upsells") or []:
                    up_name = up.get("name") or up.get("item_name") or "Add-on"
                    up_final = up.get("final_amount") or up.get("amount", 0)
                    final_text += f"✨ *{up_name}*: ₹{_fmt_money(up_final)}\n"
                if data.get("upsells"):
                    final_text += "\n"

                if data.get("discount_amount"):
                    reason = data.get("discount_reason") or "Discount"
                    final_text += f"Discount ({reason}): -₹{_fmt_money(data.get('discount_amount', 0))}\n"
                
                subtotal = data.get('subtotal', 0) or 0
                tax_amt = data.get('tax_amount', 0) or 0
                tax_pct = data.get('tax_percent', 0) or 0
                final_text += f"Base Subtotal: ₹{_fmt_money(subtotal)}\n"
                final_text += f"GST ({tax_pct}%): ₹{_fmt_money(tax_amt)}\n"
                final_text += f"*Grand Total (incl. GST): ₹{_fmt_money(total)}*\n"
                
                advance = round((total * 0.30) / 10) * 10
                final_text += f"💳 *Advance Payment (30%): ₹{_fmt_money(advance)}*\n\n"
                
                final_text = _with_buttons(final_text, card)

        elif ctype == "room_options":
            text = str(card.get("text") or "").strip()
            if text:
                final_text += text + "\n\n"
                final_text = _with_buttons(final_text, card)
            else:
                data = card.get("data") or {}
                options = data.get("options") or []
                num_nights = data.get("num_nights")

                lines = []
                lines.append("🏨 *Rooms Available for You*\n")

                if options:
                    logger.info("CARD_SERIALIZED | channel=whatsapp | card_type=room_options | output_type=text | options_count=%s", len(options))
                    keycaps = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
                    max_rooms = 1
                    
                    for i, opt in enumerate(options, 1):
                        if not isinstance(opt, dict):
                            continue
                        
                        price_pn = float(opt.get("total_price_per_night_inclusive", opt.get("total_price_per_night", 0)) or 0)
                        stay_total = float(opt.get("total_price_for_stay", price_pn) or price_pn)
                        
                        rms = opt.get("rooms") or []
                        max_rooms = max(max_rooms, len(rms))
                        counts: dict[str, int] = {}
                        bed_info = None
                        for r in rms:
                            name = str((r or {}).get("room_name") or "ROOM").strip().title()
                            rp_name = str((r or {}).get("rate_plan_name") or "").lower()
                            if "foc" in rp_name or "free" in rp_name:
                                name = f"{name} (FOC)"
                            counts[name] = counts.get(name, 0) + 1
                            if not bed_info:
                                bed_info = (r or {}).get("bed_type") or (r or {}).get("beds") or (r or {}).get("room_type")

                        if len(rms) == 1 and rms:
                            room_str = list(counts.keys())[0] if counts else "Room"
                        else:
                            room_str = ", ".join(f"{cnt}x {n}" if cnt > 1 else n for n, cnt in counts.items())

                        labels = opt.get("recommendation_labels") or []
                        fmt_labels = []
                        for lbl in labels:
                            if lbl.lower() == "recommended":
                                fmt_labels.append("⭐ Recommended")
                            elif lbl.lower() == "most comfortable":
                                fmt_labels.append("✨ Most Comfortable")
                            elif "includes" in lbl.lower() or "no extra" in lbl.lower():
                                continue
                            else:
                                fmt_labels.append(lbl)
                        
                        num_icon = keycaps[i-1] if i <= len(keycaps) else f"*{i}*"
                        header_line = f"{num_icon} *{room_str}*"
                        if fmt_labels:
                            header_line += f" — {fmt_labels[0]}"

                        req_adults = data.get("adults")
                        req_children = data.get("children")
                        # How many mattresses, not how many extra adults — this
                        # is rendered below as "N Extra Bed". A CP / MAP guest
                        # charged for an adult above base occupancy has a person
                        # charge and no bed, and must not be told otherwise.
                        # `None` keeps the old charge-based count.
                        extra_adults = sum(
                            1
                            for r in rms
                            if (
                                bool(r.get("extra_bed_count"))
                                if r.get("extra_bed_count") is not None
                                else (r.get("extra_adult_charge") or 0) > 0
                            )
                        )

                        occ_parts = []
                        if req_adults is not None:
                            occ_parts.append(f"{req_adults} Adult{'s' if req_adults != 1 else ''}")
                            if req_children:
                                occ_parts.append(f"{req_children} Child{'ren' if req_children != 1 else ''}")
                        else:
                            total_adults = sum((r.get("assigned_adults") or 2) for r in rms)
                            total_children = sum((r.get("assigned_children") or 0) for r in rms)
                            if total_adults:
                                occ_parts.append(f"{total_adults} Adult{'s' if total_adults != 1 else ''}")
                            if total_children:
                                occ_parts.append(f"{total_children} Child{'ren' if total_children != 1 else ''}")
                        
                        if extra_adults > 0:
                            occ_parts.append(f"{extra_adults} Extra Bed")
                        
                        occ_str = "👥 " + ", ".join(occ_parts)
                        if bed_info:
                            if isinstance(bed_info, list) and bed_info:
                                first_bed = bed_info[0]
                                b_str = first_bed.get("bedType") or first_bed.get("type") if isinstance(first_bed, dict) else str(first_bed)
                                if b_str:
                                    occ_str += f" | 🛏️ {b_str.title()}"
                            elif isinstance(bed_info, str) and bed_info.strip():
                                occ_str += f" | 🛏️ {bed_info.strip().title()}"

                        price_pn_rounded = int(round(price_pn / 10) * 10)
                        if len(rms) > 1 and len(counts) == 1:
                            num_rms = len(rms)
                            single_room_rate = int(round((price_pn_rounded / num_rms) / 10) * 10)
                            price_str = f"💰 ₹{single_room_rate:,} x {num_rms} rooms = ₹{price_pn_rounded:,}/night"
                        elif len(rms) > 1 and len(counts) > 1:
                            room_breakdowns = []
                            for r in rms:
                                r_name = str((r or {}).get("room_name") or "Room").strip().title()
                                r_price = float((r or {}).get("price_per_night") or 0) + float((r or {}).get("extra_adult_charge") or 0)
                                r_price_rounded = int(round(r_price / 10) * 10)
                                room_breakdowns.append(f"{r_name} (₹{r_price_rounded:,})")
                            price_str = f"💰 {' + '.join(room_breakdowns)} = ₹{price_pn_rounded:,}/night"
                        else:
                            price_str = f"💰 ₹{price_pn_rounded:,}/night"
                        
                        lines.append(header_line)
                        lines.append(occ_str)
                        lines.append(price_str + "\n")

                    num_opts = len(options)
                    nights = max(int(num_nights or 1), 1)
                    room_text = f"{max_rooms} room" if max_rooms == 1 else f"{max_rooms} rooms"
                    night_text = f"{nights} night" if nights == 1 else f"{nights} nights"
                    lines.append(f"Prices are for {room_text} for {night_text}.\n")
                    lines.append("Meal plan upgrades (breakfast/dinner) available — ask me for details.\n")

                    if num_opts == 1:
                        opt_range = "1"
                    elif num_opts == 2:
                        opt_range = "1 or 2"
                    elif num_opts <= 6:
                        digits = [str(n) for n in range(1, num_opts + 1)]
                        opt_range = ", ".join(digits[:-1]) + f", or {digits[-1]}"
                    else:
                        opt_range = f"1 to {num_opts}"
                    lines.append(f"👉 Reply {opt_range} to continue with your booking.")
                    final_text += "\n".join(lines) + "\n\n"

        elif ctype == "image":
            # Images and their captions are delivered directly via hotel_media_service.
            # Do NOT append caption to final_text to prevent duplicate text messages on WhatsApp.
            pass

        elif ctype in ("video", "audio", "document"):
            url = card.get("url") or ""
            caption = card.get("caption") or card.get("text") or "Attached file"
            if url:
                final_text += f"📎 {caption}:\n{url}\n\n"
            elif caption:
                final_text += f"{caption}\n\n"

        elif ctype == "payment":
            data = card.get("data") or {}
            text = card.get("text") or ""
            if text:
                final_text += text + "\n\n"
            link = data.get("payment_link") or data.get("url") or card.get("url")
            amount = data.get("amount") or data.get("payment_link_amount")
            if amount is not None:
                final_text += f"Amount: ₹{_fmt_money(amount)}\n"
            if link:
                final_text += f"Pay here: {link}\n\n"
            final_text = _with_buttons(final_text, card)

        elif ctype in ("upsell_options", "addon_catalog"):
            text = card.get("text") or ""
            if text:
                final_text += text + "\n\n"
            items = card.get("data") or []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or "Add-on Service"
                    price = item.get("price", 0)
                    desc = item.get("description")
                    custom_pricing = item.get("custom_pricing", False)

                    line = f"• *{name}*"
                    if not custom_pricing:
                        try:
                            if float(price) > 0:
                                line += f" (₹{_fmt_money(price)})"
                        except (TypeError, ValueError):
                            pass
                    final_text += line + "\n"
                    if desc:
                        final_text += f"   _{desc}_\n"
                final_text += "\n"
            final_text = _with_buttons(final_text, card)

        elif ctype in ("button_group", "list_message"):
            # The authored button vocabulary. On the CRM/WhatsApp path these are translated
            # into a Meta `interactive` payload before they reach here (see
            # `chat.py:_collapse_for_crm_whatsapp`); this branch covers the bot's own webhook
            # and the gate-off path, so a set of choices is never reduced to a bare sentence
            # the guest has nothing to reply to.
            final_text += (card.get("text") or "").strip() + "\n"
            final_text += _reply_token_lines(_card_button_labels(card))
            footer = card.get("footer")
            if footer:
                final_text += f"\n_{footer}_\n"
            final_text += "\n"

        else:
            # Unknown card — best-effort text
            if card.get("text"):
                final_text += card.get("text") + "\n\n"

    return final_text.strip()


def _card_button_labels(card: dict[str, Any]) -> list[str]:
    """
    The guest-facing labels of an authored button/list card, in the order they were offered.

    `button_group` keeps its buttons flat; `list_message` nests them under sections. Both
    end up as one ordered list here so the text fallback reads the same either way.
    """
    if card.get("type") == "list_message":
        items = [
            row
            for section in (card.get("sections") or [])
            for row in (section.get("rows") or [])
        ]
    else:
        items = card.get("buttons") or []

    labels = []
    for item in items:
        if isinstance(item, dict):
            label = item.get("title") or item.get("label") or item.get("text") or item.get("id")
        else:
            label = item
        if label:
            labels.append(str(label))
    return labels


def _reply_token_lines(labels: list[str]) -> str:
    """
    The options, phrased as something a guest can send back.

    The label itself is the token. A tap and a typed reply then resolve through the same
    declaration (`app.utils.button_registry`): the guest reads "Room types & rates", sends
    "Room types & rates", and the router matches the words it was handed.

    This used to print `1.`, `2.`, `3.` instead, and the numbers did not mean what the lines
    beside them said. Measured 2026-10-15 against the menu reader at
    `understand.py:1137-1160`: `1` answered "make a new booking" while the line above it
    promised "Room types & rates", and `3` answered room prices while its line said "Talk to
    reception" — two of the three wrong. Only `2` matched, and by luck.

    Numbering cannot be made right from here: a number's meaning is positional and the reader
    is one global table, so it would have to be renumbered at every card. The label has no
    such problem — it is the same string on both sides by construction.
    """
    if not labels:
        return ""
    lines = "Reply with one of these:\n"
    for label in labels:
        lines += f"*{label}*\n"
    return lines


def _with_buttons(text: str, card: dict[str, Any]) -> str:
    buttons = card.get("buttons") or []
    if not buttons:
        return text
    labels = []
    for btn in buttons:
        if isinstance(btn, dict):
            label = btn.get("title") or btn.get("label") or btn.get("value") or "Option"
        else:
            label = str(btn)
        labels.append(str(label))
    sep = "\n\n" if text and not text.endswith("\n\n") else ""
    text += f"{sep}" + _reply_token_lines(labels)
    footer = card.get("footer")
    if footer:
        text += f"\n_{footer}_\n"
    return text + "\n"


def _fmt_money(value: Any) -> str:
    try:
        if isinstance(value, str):
            value = value.replace('"', '').replace("'", '').replace(',', '').strip()
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "0"
