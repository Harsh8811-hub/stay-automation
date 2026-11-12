"""
Shared chat execution — used by FastAPI routes and Streamlit UI (demo mode).
"""

from __future__ import annotations

import logging
import re
from io import BytesIO

from starlette.datastructures import UploadFile as StarletteUploadFile

from app.config.settings import get_settings
from app.db.mongodb import connect_mongo
from app.prompts.templates import injection_detected_response
from app.rag.chroma_service import seed_sample_knowledge
from app.repositories import booking_repository, payment_receipt_repository, session_repository
from app.services.file_processor import process_upload
from app.services.takeover_engine import evaluate_takeover_for_guest_turn
from app.utils.guardrails import check_injection, sanitize_input
from app.utils.helpers import generate_session_id
from app.utils.human_request import asks_for_a_person
from app.workflows.graph import get_compiled_graph
from app.services.personalization_service import personalization_service

logger = logging.getLogger(__name__)

_bot_ready = False


async def ensure_bot_ready() -> None:
    """One-time startup: MongoDB, indexes, ChromaDB seed, LangGraph compile."""
    global _bot_ready
    if _bot_ready:
        return

    settings = get_settings()
    await connect_mongo()
    await session_repository.ensure_indexes()
    await booking_repository.ensure_indexes()
    await payment_receipt_repository.ensure_indexes()
    from app.services.hotel_media_service import hotel_media_service
    await hotel_media_service.ensure_indexes()

    if getattr(settings, "ENVIRONMENT", "production") in ("development", "test"):
        try:
            seed_sample_knowledge(settings.HOTEL_ID)
        except Exception as e:
            logger.warning("ChromaDB seeding skipped: %s", e)

    get_compiled_graph()
    _bot_ready = True
    logger.info("Bot ready for hotel ID: %s", settings.HOTEL_ID)


def build_summary(state: dict) -> dict | None:
    if state.get("booking_stage") == "new":
        return None

    summary: dict = {}
    for key, out_key in (
        ("adults", "adults"),
        ("children", "children"),
        ("num_rooms", "rooms"),
        ("check_in", "check_in"),
        ("check_out", "check_out"),
        ("room_type_preference", "room_preference"),
        ("total_price", "total_price"),
        ("booking_id", "booking_id"),
        ("booking_stage", "stage"),
    ):
        if state.get(key) is not None:
            summary[out_key] = state[key]
    return summary or None


def _should_prepend_greeting(greeting: str, card_text: str) -> bool:
    """Check if signature greeting should be prepended without causing duplicate greetings."""
    if not greeting or not greeting.strip():
        return False
    greeting_clean = greeting.strip().lower()
    text_clean = (card_text or "").strip().lower()
    if not text_clean:
        return True
    if greeting_clean in text_clean:
        return False
    first_chunk = text_clean[:60]
    common_greetings = (
        "hello", "hi ", "hi!", "hi,", "hey", "namaste", "namaskar",
        "swagat", "welcome", "good morning", "good afternoon", "good evening",
    )
    return not any(g in first_chunk for g in common_greetings)


def _record_booking_note(session, note: dict) -> None:
    """Write a photographed booking slip into the session, as words.

    The slip used to be read into five fields, turned into a synthetic sentence
    and re-run through a fresh booking turn — and then forgotten. The guest's
    name, the children, and the amount written on the paper were extracted by
    the vision prompt and never read by anything. So when the guest came back
    with "confirm karein, main 25% advance bhej dunga", the session held the
    bot's own catalogue quote and nothing at all about the slip.

    What is recorded, and what is deliberately not:

    * Dates, room count, guest name and the party are written down, because
      they are the guest's own facts and the bot will otherwise ask for them
      again. `num_rooms`/`guest_name` are only filled when empty — a value the
      guest typed themselves outranks a reading of their handwriting.
    * The amount written on the slip is recorded **as text, and labelled as not
      applied**. It is not written to `total_price` and it does not change any
      quote. A photograph of a paper slip is not authority to charge below the
      catalogue; pricing from it is a staff decision, so this hands staff both
      numbers in the transcript instead of choosing one silently.
    * It goes into `session.messages`, not a new field, so it rides into
      `messages_history` on every later turn and survives a resume — which is
      the mechanism that already works.
    """
    from app.models.schemas import MessageRecord

    if not isinstance(note, dict):
        return

    parts = []
    check_in, check_out = note.get("check_in"), note.get("check_out")
    if check_in or check_out:
        parts.append(f"dates {check_in or '?'} to {check_out or '?'}")
    if note.get("num_rooms"):
        parts.append(f"{note['num_rooms']} room(s)")
    if note.get("adults"):
        parts.append(f"{note['adults']} adult(s)")
    _children = note.get("children")
    if isinstance(_children, (list, tuple)):
        _children = len(_children)
    if _children:
        parts.append(f"{_children} child(ren)")
    if note.get("room_type_preference"):
        parts.append(f"prefers {note['room_type_preference']}")
    if note.get("guest_name"):
        parts.append(f"name {note['guest_name']}")

    quoted = note.get("budget")
    quoted_phrase = ""
    if quoted not in (None, "", [], {}):
        quoted_phrase = (
            f" The slip also shows an amount of *{quoted}* — recorded but NOT "
            f"applied; quote the catalogue rate and let reception decide."
        )

    if not parts and not quoted_phrase:
        return

    session.messages.append(MessageRecord(
        role="system",
        sender_type="system",
        content=(
            "SYSTEM NOTE: The guest photographed a booking slip. It says: "
            + ", ".join(parts) + "." + quoted_phrase
        ),
    ))

    if note.get("num_rooms") and not getattr(session, "num_rooms", None):
        session.num_rooms = note["num_rooms"]
    if note.get("guest_name") and not getattr(session, "guest_name", None):
        session.guest_name = note["guest_name"]


async def _record_stated_fare(session, message: str) -> None:
    """Write a fare the guest typed into the session, as words.

    The photographed-slip equivalent is `_record_booking_note` above; this is
    the same boundary for a message that was typed rather than photographed.
    2026-10-29, hotel Sample Property: the guest pasted "Fare:- Rs. 2000/- x 3 Rooms
    x 3 Nights = Rs. 18000/-" and the bot's reply carried no trace of it —
    `budget` was None, `total_price` was None, and the number was gone by the
    next turn.

    Recorded, not applied: `total_price` is left exactly as the catalogue set
    it. See `app/utils/stated_fare.py` for why the reading is conservative.
    """
    from app.models.schemas import MessageRecord
    from app.utils.stated_fare import extract_stated_fare

    fare = extract_stated_fare(message)
    if fare is None:
        return

    # Same turn twice must not stack two notes — a guest who repeats the brief
    # is repeating themselves, not adding a second fare.
    marker = f"wrote a total fare of *{fare:,.0f}*"
    if any(marker in (getattr(m, "content", "") or "") for m in session.messages):
        return

    session.messages.append(MessageRecord(
        role="system", sender_type="system",
        content=(
            f"SYSTEM NOTE: The guest wrote a total fare of *{fare:,.0f}* for this "
            f"stay. Recorded but NOT applied — quote the catalogue rate; any "
            f"discount is reception's decision, not the bot's."
        ),
    ))


async def run_chat_turn(
    message: str,
    session_id: str | None = None,
    hotel_id: int | None = None,
    customer_phone: str = "",
    phone: str = "",
) -> dict:
    if not customer_phone and phone:
        customer_phone = phone
    await ensure_bot_ready()
    settings = get_settings()

    cleaned = sanitize_input(message, max_length=settings.MAX_MESSAGE_LENGTH)
    if not cleaned:
        logger.warning("Empty message after sanitization.")
        return {
            "session_id": session_id or generate_session_id(),
            "response": [{"type": "text", "text": "I didn't quite catch that. Could you please send some text?"}],
            "booking_stage": "new",
        }

    from app.utils.phone import digits_only
    customer_phone = digits_only(customer_phone) or (customer_phone or "")

    # ── Resolve Hotel ID with strict scoping ──────────────────
    # Priority: 1. Request payload -> 2. Existing session -> 3. Environment fallback
    hotel_source = "env_fallback"
    if hotel_id is not None and str(hotel_id).strip():
        resolved_hotel_id = str(hotel_id)
        hotel_source = "payload"
    else:
        resolved_hotel_id = str(settings.HOTEL_ID)

    sid = session_id or generate_session_id(resolved_hotel_id, customer_phone)

    from app.observability.tracing import bind_context, mask_phone
    bind_context(session_id=sid, hotel_id=resolved_hotel_id)

    # ── 1. Human Takeover gate (session SSOT + backend staff messages) ──
    # Staff is highest authority. Sync Chat_Sessions staff messages, pause AI
    # while takeover is active, auto-resume only after 10m inactivity.
    session = None
    try:
        if session_id:
            session = await session_repository.get_session(session_id, hotel_id=resolved_hotel_id)
        if not session and customer_phone:
            session = await session_repository.find_active_session(
                resolved_hotel_id, customer_phone
            )
        if not session:
            from app.models.schemas import BookingSession
            session = BookingSession(
                session_id=sid,
                hotel_id=resolved_hotel_id,
                customer_identifier=customer_phone,
            )
        else:
            # ── Check Session Staleness (Fix 7) ───────────────────────
            from datetime import datetime, timezone
            stale_threshold_hours = getattr(settings, "SESSION_STALE_AFTER_HOURS", getattr(settings, "SESSION_TTL_HOURS", 24))
            
            last_ts = getattr(session, "updated_at", None) or getattr(session, "last_message_at", None) or getattr(session, "created_at", None)
            if last_ts is not None:
                if isinstance(last_ts, str):
                    try:
                        last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    except Exception:
                        last_ts = None
                elif isinstance(last_ts, datetime) and last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
            
            now_utc = datetime.now(timezone.utc)
            if last_ts:
                age_seconds = (now_utc - last_ts).total_seconds()
                age_hours = age_seconds / 3600.0
                if age_hours > stale_threshold_hours:
                    from app.observability.tracing import mask_phone
                    logger.info(
                        "STALE_SESSION_RESET | phone=%s | old_session=%s | age_hours=%.2f | stage=%s",
                        mask_phone(customer_phone or session.customer_identifier),
                        session.session_id,
                        age_hours,
                        session.booking_stage,
                    )
                    committed_stages = ("payment_pending", "id_verification", "awaiting_payment", "ready_for_payment", "confirmed", "checked_in")
                    has_active_payment = bool(session.payment_link or session.payment_link_id or (session.total_price and session.selected_rooms) or session.booking_id)
                    
                    if session.booking_stage in ("confirmed", "checked_in"):
                        # Confirmed reservations are never wiped by staleness
                        pass
                    elif session.booking_stage in committed_stages and has_active_payment:
                        # Soft-confirm via lifecycle reset/resume flow
                        session.lifecycle_action_pending = "reset"
                    else:
                        # Early stages (new, greeting, collecting, rooms_shown) -> auto fresh start
                        from app.models.schemas import BookingSession
                        # Keep the SAME session_id; discard only the state.
                        #
                        # Minting a new id here (generate_session_id()) orphaned the record the
                        # caller keeps addressing. The CRM sends the guest's phone number AS the
                        # session_id, so after a reset the next message looked up this same stale
                        # document again — same id, still old updated_at — reset again, and the
                        # guest could never hold a conversation: every turn began blank, so a
                        # bare "4" answering "how many guests?" was read as menu option 4.
                        #
                        # Reusing the key means save_state overwrites this document with a current
                        # timestamp, so the next turn finds a live session and continues it. This
                        # also matches the committed-payment branch above, which keeps the session
                        # and only flags a reset.
                        #
                        # ── But the transcript is not state ──────────────────
                        # Blanking `messages` too made "waking up" literally true:
                        # a guest back after 24 hours started from nothing, and the
                        # bot re-asked for the dates, the party and the room it had
                        # been discussing hours earlier. `STALE_SESSION_KEEP_
                        # RECENT_MESSAGES` keeps the last few turns as *context to
                        # read*, while every booking field — dates, rooms, prices,
                        # stage — is still discarded. The bot can therefore follow
                        # the conversation without resurrecting a dead quote.
                        kept_messages = []
                        if getattr(settings, "STALE_SESSION_KEEP_RECENT_MESSAGES", 0):
                            kept_messages = list(
                                getattr(session, "messages", None)
                                or []
                            )[-(int(settings.STALE_SESSION_KEEP_RECENT_MESSAGES)):]
                        session = BookingSession(
                            session_id=session.session_id,
                            hotel_id=resolved_hotel_id,
                            customer_identifier=customer_phone,
                            messages=kept_messages,
                        )

            sid = session.session_id
            # If request didn't supply hotel_id, inherit from existing session
            if hotel_source == "env_fallback" and session.hotel_id:
                resolved_hotel_id = session.hotel_id
                hotel_source = "session"
            elif hotel_source == "payload" and session.hotel_id != resolved_hotel_id:
                session.hotel_id = resolved_hotel_id
                
            bind_context(session_id=sid, hotel_id=resolved_hotel_id)
            if customer_phone and not session.customer_identifier:
                session.customer_identifier = customer_phone

        logger.info(
            "REQUEST_HOTEL_ID | hotel_id=%s | source=%s | sess=%s",
            resolved_hotel_id,
            hotel_source,
            sid,
        )

        session, takeover_action = await evaluate_takeover_for_guest_turn(
            session,
            customer_phone=customer_phone or session.customer_identifier or "",
            guest_message=cleaned,
        )
        if takeover_action == "pause":
            from app.observability.metrics import record_chat_turn
            record_chat_turn("human_takeover", hotel_id=resolved_hotel_id)
            logger.info(
                "CHAT_TAKEOVER_PAUSE | phone=%s session=%s",
                mask_phone(customer_phone or session.customer_identifier),
                sid,
            )
            return {
                "session_id": sid,
                "response": [],
                "booking_stage": "human_intervention",
                "booking_summary": None,
            }
        # "resume" or "continue" → fall through to AI
        if takeover_action != "continue":
            logger.info("CHAT_TAKEOVER_ACTION | action=%s session=%s", takeover_action, sid)
    except Exception as e:
        from app.observability.failure_store import record_failure
        record_failure(
            component="takeover.evaluate",
            error=e,
            hotel_id=resolved_hotel_id,
            session_id=sid,
            phone=customer_phone,
            severity="warning",
            exc=e,
        )
        logger.warning(f"Takeover check failed (continuing with AI): {e}", exc_info=True)

    if check_injection(cleaned):
        logger.warning("Injection attempt blocked: %s...", cleaned[:50])
        from app.observability.metrics import record_chat_turn
        record_chat_turn("injection_blocked", hotel_id=resolved_hotel_id)
        return {
            "session_id": sid,
            "response": injection_detected_response(),
            "booking_stage": "new",
            "booking_summary": None,
        }

    # ── Check if this is the first turn ────────────────────────
    session = await session_repository.get_session(sid)
    is_first_turn = False
    if not session or not any(m.role == "assistant" for m in session.messages):
        is_first_turn = True

    # ── Pre-graph AI gate check (Python-enforced) ────────────────
    # If AI is disabled for this hotel, reject immediately without invoking the graph
    try:
        from app.services.hotel_config_service import get_hotel_config
        from app.services.feature_gate import FeatureGate
        config = await get_hotel_config(resolved_hotel_id)
        gate = FeatureGate(config, session_id=sid)
        ai_check = gate.check_ai_enabled()
        if not ai_check.allowed:
            logger.info(f"AI disabled for hotel_id={resolved_hotel_id}. Returning gate response.")
            from app.observability.metrics import record_chat_turn
            record_chat_turn("ai_disabled", hotel_id=resolved_hotel_id)
            return {
                "session_id": sid,
                "response": [{"type": "text", "text": ai_check.response}],
                "booking_stage": "new",
                "booking_summary": None,
            }
    except Exception as e:
        logger.warning(f"Hotel config check failed (proceeding anyway): {e}")

    # ── Mode Switch Detection ────────────────
    # Skipped outright when the guest is asking for a person. The detector asks
    # an LLM whether the message is a request to change the bot's behaviour, and
    # "I need to speak to a human agent" reads to it as "talk like a human" — the
    # turn then ends with the Human-Like mode's acknowledgement and no hand-off
    # at all. A human request has its own destination further down the graph, so
    # the precedence belongs here rather than in the prompt. See
    # `app/utils/human_request.py`.
    switched_mode = None
    if not asks_for_a_person(cleaned):
        switched_mode = await personalization_service.detect_mode_switch(
            cleaned, str(resolved_hotel_id)
        )
    else:
        logger.info(
            "PERSONALITY_SWITCH_SKIPPED | session_id=%s | the guest asked for a person",
            sid,
        )
    if switched_mode:
        logger.info(f"PERSONALITY_SWITCH | session_id={sid} | mode={switched_mode.mode_name}")
        
        # Ensure session document exists in DB before updating personality mode to avoid orphan updates
        from app.models.schemas import BookingSession
        if not session:
            session = BookingSession(
                session_id=sid,
                hotel_id=resolved_hotel_id,
                customer_identifier=customer_phone,
            )
            await session_repository.upsert_session(session)
            
        await session_repository.update_personality_mode(sid, switched_mode.mode_id)
        bot_response = [{"type": "text", "text": switched_mode.switch_acknowledgment}]
                
        return {
            "session_id": sid,
            "response": bot_response,
            "booking_stage": "new", # We don't advance the stage
            "booking_summary": None,
        }

    initial_state = {
        "session_id": sid,
        "hotel_id": resolved_hotel_id,
        "customer_identifier": customer_phone,
        "user_message": cleaned,
    }

    from app.observability.failure_store import record_failure
    from app.observability.metrics import metrics, record_chat_turn, record_node_error
    try:
        async with session_repository.acquire_session_lock(sid):
            with metrics.timer("chat.turn_latency_ms", hotel_id=resolved_hotel_id):
                final_state = await get_compiled_graph().ainvoke(initial_state)
    except Exception as e:
        record_failure(
            component="chat.graph_invoke",
            error=e,
            hotel_id=resolved_hotel_id,
            session_id=sid,
            phone=customer_phone,
            exc=e,
        )
        logger.error(f"Graph invoke failed session={sid}: {e}", exc_info=True)
        record_node_error("graph", hotel_id=resolved_hotel_id)
        record_chat_turn("error", hotel_id=resolved_hotel_id)
        
        fallback_text = "I'm sorry, I encountered a brief issue processing that. Could you please reply with your option number or restate your request?"

        # Second-order crash persistence: persist failure counter and escalate if crashes repeat
        try:
            from datetime import datetime, timezone
            from app.models.schemas import MessageRecord
            session = await session_repository.get_session(sid, hotel_id=resolved_hotel_id)
            if not session and customer_phone:
                session = await session_repository.find_active_session(resolved_hotel_id, customer_phone)

            if session:
                current_failures = int(getattr(session, "escalation_failure_count", 0) or 0)
                new_failures = current_failures + 1
                session.escalation_failure_count = new_failures

                # Emergency escalation if crashes repeat >= 3 times
                if new_failures >= 3:
                    logger.warning(
                        "Repeated crash threshold reached for session %s (failures=%s). Triggering emergency escalation.",
                        sid,
                        new_failures,
                    )
                    try:
                        from app.services.ticket_service import create_or_update_ticket, get_formatted_contact_info
                        await create_or_update_ticket(
                            hotel_id=resolved_hotel_id,
                            session_id=sid,
                            customer_phone=customer_phone,
                            issue_type="TECHNICAL_ERROR",
                            description=f"Automated escalation: System encountered repeated processing errors ({new_failures} consecutive crashes). Last error: {str(e)[:200]}",
                            priority="HIGH",
                        )
                        contact_info = await get_formatted_contact_info(resolved_hotel_id)
                        fallback_text = (
                            "I apologize, but I am having trouble processing your request right now. "
                            "I have notified our team to assist you directly."
                        )
                        if contact_info:
                            fallback_text += f"\n\nYou can also reach us at:\n\n{contact_info}"
                    except Exception as ticket_err:
                        logger.error(f"Failed to create emergency ticket on crash: {ticket_err}", exc_info=True)

                if cleaned:
                    session.messages.append(MessageRecord(
                        role="user",
                        content=cleaned,
                        sender_type="guest",
                        timestamp=datetime.now(timezone.utc),
                    ))
                session.messages.append(MessageRecord(
                    role="assistant",
                    content=fallback_text,
                    sender_type="bot",
                    timestamp=datetime.now(timezone.utc),
                ))
                session.messages = session.messages[-20:]
                await session_repository.upsert_session(session)
                logger.info("Persisted crash failure state to session %s: escalation_failure_count=%s", sid, new_failures)
        except Exception as persist_err:
            logger.error(f"Failed to persist failure state on graph crash: {persist_err}", exc_info=True)

        return {
            "session_id": sid,
            "response": [{"type": "text", "text": fallback_text}],
            "booking_stage": "collecting",
            "booking_summary": None,
        }

    bot_response = final_state.get(
        "bot_response",
        [{"type": "text", "text": "Something went wrong. Please try again."}],
    )
    
    if is_first_turn and not bot_response:
        context = await personalization_service.load_context(str(resolved_hotel_id))
        greeting = context.customization.bot_persona.signature_greeting.replace("{name}", context.customization.bot_persona.name).strip()
        if greeting:
            bot_response = [{"type": "text", "text": greeting}]

    stage = final_state.get("booking_stage", "new")
    intent = final_state.get("intent")
    cards = final_state.get("bot_response") or []
    record_chat_turn(stage if stage else "ok", hotel_id=resolved_hotel_id)
    logger.info(
        "CHAT_GRAPH_DONE | intent=%s stage=%s cards=%s phone=%s",
        intent,
        stage,
        [c.get("type") for c in cards if isinstance(c, dict)],
        mask_phone(customer_phone),
    )
    # A fare the guest typed is written down after the turn has been answered,
    # so the note is in history for the NEXT turn and for staff reading the
    # transcript — the same place the photographed slip's note lands. Doing it
    # here (rather than in a graph node) keeps it to one write, on the rare
    # turn that carries an amount, and reuses the session the graph just saved.
    try:
        from app.utils.stated_fare import extract_stated_fare
        if extract_stated_fare(cleaned):
            fare_session = await session_repository.get_session(
                final_state.get("session_id", sid)
            )
            if fare_session:
                await _record_stated_fare(fare_session, cleaned)
                await session_repository.upsert_session(fare_session)
    except Exception as e:
        # Losing the note is a lost detail, not a lost turn.
        logger.warning(
            f"STATED_FARE_RECORD_FAILED | session={sid} | {e}", exc_info=True
        )

    return {
        "session_id": final_state.get("session_id", sid),
        "response": bot_response,
        "booking_stage": stage,
        "booking_summary": build_summary(final_state),
    }


async def run_media_turn(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    session_id: str | None = None,
    message: str | None = None,
    hotel_id: int | None = None,
    customer_phone: str = "",
) -> dict:
    await ensure_bot_ready()
    settings = get_settings()

    try:
        from app.services.hotel_config_service import get_hotel_config
        from app.services.feature_gate import FeatureGate
        config = await get_hotel_config(hotel_id or settings.HOTEL_ID)
        gate = FeatureGate(config, session_id=session_id)
        
        doc_check = gate.check_document_understanding()
        if not doc_check.allowed:
            return {
                "session_id": session_id or generate_session_id(),
                "response": [{"type": "text", "text": doc_check.response}],
                "booking_stage": "new",
                "booking_summary": None,
            }
    except Exception as e:
        logger.warning(f"Hotel config check failed (proceeding anyway): {e}")

    is_image = (
        (content_type and content_type.startswith("image/"))
        or any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp", ".gif"))
        or file_bytes.startswith(b'\x89PNG\r\n\x1a\n')
        or file_bytes.startswith(b'\xff\xd8')
        or (file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[8:12])
    )

    if is_image:
        from app.repositories import session_repository
        from app.models.schemas import BookingSession, MessageRecord
        from app.services.vision.image_handler import process_image_pipeline

        # Load session for grounded context
        session = await session_repository.get_session(session_id) if session_id else None
        if not session and customer_phone:
            session = await session_repository.find_active_session(hotel_id or settings.HOTEL_ID, customer_phone)
            
        resolved_hid = hotel_id or (session.hotel_id if session else None) or settings.HOTEL_ID
        session_dict = session.model_dump() if session else {}

        vision_res = await process_image_pipeline(
            file_bytes=file_bytes,
            filename=filename,
            session_state=session_dict,
            hotel_id=resolved_hid,
            user_message=message,
        )

        if not session:
            session = BookingSession(
                session_id=session_id or generate_session_id(),
                hotel_id=resolved_hid,
                customer_identifier=customer_phone,
            )

        # Apply DB updates
        if vision_res.action.db_updates:
            for k, v in vision_res.action.db_updates.items():
                if hasattr(session, k):
                    if k == "booking_stage" and isinstance(v, str):
                        try:
                            from app.models.schemas import BookingStage
                            v = BookingStage(v)
                        except ValueError:
                            pass
                    setattr(session, k, v)
                    
        # Update messages history
        if message:
            session.messages.append(MessageRecord(
                role="user", content=message, sender_type="guest",
            ))
        reply_text = vision_res.action.message
        session.messages.append(MessageRecord(
            role="assistant", content=reply_text, sender_type="bot",
        ))
        
        # Save session
        await session_repository.upsert_session(session)
        
        stage = session.booking_stage or "new"
        cards: list[dict[str, Any]] = [{"type": "text", "text": reply_text}]

        # Attach real room photos if room_reference matched a catalog room
        if vision_res.image_type == "room_reference":
            matched_tag = (vision_res.action.db_updates or {}).get("photo_tag_preference") or (vision_res.validation.metadata or {}).get("matched_room_name")
            matched_room_id = (vision_res.action.db_updates or {}).get("preferred_room_id") or (vision_res.validation.metadata or {}).get("matched_room_id")
            if matched_tag:
                from app.services.hotel_media_service import media_delivery_card
                cards.append(media_delivery_card(
                    entity_type="room",
                    entity_id=matched_room_id,
                    tags=[matched_tag.lower()],
                    caption=f"📸 *{matched_tag}*",
                    limit=1,
                ))

        # Special handling for ID Card verification
        if vision_res.image_type == "id_card":
            import datetime
            guest_name = (vision_res.action.db_updates or {}).get("guest_name") or session.guest_name or "Guest"
            caption_text = (message or "").strip()
            caption_lower = caption_text.lower()
            
            # Check if caption specifies a room number for immediate check-in
            # e.g. "Room 204", "check in room 102", "305", "for room 201"
            room_match = (
                re.search(r'\b(?:room|room\s*no\.?|check\s*in|checking\s*in|for\s*room|kamra|kamre)\s*(?:is\s*|no\.?\s*|number\s*)?([0-9]{1,4}[a-zA-Z]?)\b', caption_lower)
                or re.search(r'^\s*([0-9]{1,4}[a-zA-Z]?)\s*[.,!]?\s*$', caption_lower)
            )
            has_room_kw = bool(re.search(r'\b(room|kamra|kamre|check\s*in|checking\s*in|checkin|for\s*room)\b', caption_lower))
            is_booking_query = bool(re.search(r'\b(?:adults?|children|child|sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug|night|nights|book|booking|reservation|rooms|room\s*only|with\s*breakfast|people|person|pax)\b', caption_lower))

            if caption_text and not is_booking_query and room_match and (has_room_kw or (room_match.group(1).isdigit() and len(room_match.group(1)) <= 4 and room_match.group(1) not in ("1", "2"))):
                room_num = room_match.group(1).strip().upper()
                formatted_room = f"Room {room_num}" if not room_num.lower().startswith("room") else room_num
                now_dt = datetime.datetime.now()
                now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                date_display = now_dt.strftime("%d %b %Y")

                from app.services.hotel_config_service import get_hotel_config
                cfg = await get_hotel_config(resolved_hid)
                hotel_name = cfg.hotel_name if cfg else "Sample Hotel"

                checkin_msg = (
                    f"Your ID verification has been completed successfully for {guest_name}. "
                    f"Check-in Recorded for {formatted_room}. Enjoy your stay at {hotel_name}! 🏨"
                )
                from app.models.schemas import BookingStage
                session.room_number = formatted_room
                session.id_received_date = now_str
                session.booking_stage = BookingStage.CHECKED_IN
                session.next_action = None
                if session.messages and session.messages[-1].role == "assistant":
                    session.messages[-1] = MessageRecord(
                        role="assistant", content=checkin_msg, sender_type="bot",
                    )
                else:
                    session.messages.append(MessageRecord(
                        role="assistant", content=checkin_msg, sender_type="bot",
                    ))
                await session_repository.upsert_session(session)
                return {
                    "session_id": session.session_id,
                    "response": [{"type": "text", "text": checkin_msg}],
                    "booking_stage": "checked_in",
                    "booking_summary": None,
                }

            elif caption_text and is_booking_query:
                # User provided booking criteria in the caption
                prefix = f"Your ID verification has been completed successfully for {guest_name}!\n\n"
                chat_res = await run_chat_turn(
                    message=caption_text,
                    session_id=session.session_id,
                    hotel_id=resolved_hid,
                    customer_phone=customer_phone,
                )
                if chat_res.get("response") and chat_res["response"][0].get("type") == "text":
                    chat_res["response"][0]["text"] = prefix + chat_res["response"][0]["text"]
                return chat_res

            else:
                # No caption or generic caption ("my id", "aadhaar", "id card")
                # Return the verified ID action message directly to preserve the prompt
                return {
                    "session_id": session.session_id,
                    "response": cards,
                    "booking_stage": stage,
                    "booking_summary": None,
                }

        # Special handling for room reference photos
        if vision_res.image_type == "room_reference":
            caption_text = (message or "").strip()
            caption_lower = caption_text.lower()
            # Check if caption contains explicit booking criteria (dates, adult/child counts)
            is_booking_query = bool(re.search(
                r'\b(?:adults?|children|child|sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug|night|nights|today|tomorrow|from\s+\d|to\s+\d|\d{1,2}[/-]\d{1,2})\b',
                caption_lower,
            ))
            if not caption_text or not is_booking_query:
                # User is simply asking for/about this room (e.g. "show me this kind of room in your hotel")
                # Return vision cards directly to preserve the rich visual explanation & matched room photo
                return {
                    "session_id": session.session_id,
                    "response": cards,
                    "booking_stage": stage,
                    "booking_summary": None,
                }
            else:
                # User provided explicit booking dates/guests in caption (e.g. "show me this room for tomorrow 2 adults")
                pref_name = (vision_res.action.db_updates or {}).get("photo_tag_preference") or "the matched room"
                prefix = f"I see you're interested in our *{pref_name}*! Let me check availability for your dates.\n\n"
                chat_res = await run_chat_turn(
                    message=caption_text,
                    session_id=session.session_id,
                    hotel_id=resolved_hid,
                    customer_phone=customer_phone,
                )
                if any(c.get("type") == "media_delivery" for c in cards):
                    media_cards = [c for c in cards if c.get("type") == "media_delivery"]
                    chat_res["response"] = media_cards + (chat_res.get("response") or [])
                return chat_res

        # Special handling for handwritten booking notes / paper slips -> Lead Creation & Live Search
        if vision_res.image_type == "booking_note":
            extracted_note = vision_res.extracted or {}
            if isinstance(extracted_note, dict):
                check_in = extracted_note.get("check_in")
                check_out = extracted_note.get("check_out")
                adults = extracted_note.get("adults") or 2
                rooms = extracted_note.get("num_rooms") or 1
                pref_room = extracted_note.get("room_type_preference") or ""
                children = extracted_note.get("children")
                note_dict = extracted_note
            else:
                check_in = getattr(extracted_note, "check_in", None)
                check_out = getattr(extracted_note, "check_out", None)
                adults = getattr(extracted_note, "adults", None) or 2
                rooms = getattr(extracted_note, "num_rooms", None) or 1
                pref_room = getattr(extracted_note, "room_type_preference", None) or ""
                children = getattr(extracted_note, "children", None)
                note_dict = (
                    extracted_note.model_dump()
                    if hasattr(extracted_note, "model_dump")
                    else {}
                )

            # Write the slip down before the search runs, so the guest's own
            # facts survive into the next turn and into a resumed session.
            # Persisted (not just held in memory) because the recursive call
            # below reloads the session from the database.
            try:
                _record_booking_note(session, note_dict)
                await session_repository.upsert_session(session)
            except Exception as e:
                # A slip that cannot be recorded must not cost the guest their
                # turn — the search below is the part they are waiting for.
                logger.warning(f"BOOKING_NOTE_RECORD_FAILED | session={session.session_id} | {e}", exc_info=True)

            # If note has stay dates, execute the room search so available room options are presented immediately
            if check_in and check_out:
                # The children were extracted and then dropped here: the search
                # query only ever carried `adults`, so a slip saying "2 child"
                # reached the room search as an adults-only party and the
                # pricing card was built for the wrong party.
                #
                # `children` arrives as a count, a string, or a list of ages
                # depending on how the slip was written, so it is normalised to
                # a count before it is spoken back — "[2, 5] children" would be
                # read as a quantity by the extractor and lost.
                if isinstance(children, (list, tuple)):
                    children = len(children)
                elif isinstance(children, str) and children.strip().isdigit():
                    children = int(children.strip())
                elif isinstance(children, str):
                    children = None
                _party = f"{adults} adults"
                if children:
                    _party += f" and {children} children"
                search_query = f"Book {rooms} {pref_room or 'room'} for {_party} from {check_in} to {check_out}".strip()
                chat_res = await run_chat_turn(
                    message=search_query,
                    session_id=session.session_id,
                    hotel_id=resolved_hid,
                    customer_phone=customer_phone,
                )
                # Prepend the Lead confirmation card before room cards
                chat_res["response"] = cards + (chat_res.get("response") or [])
                return chat_res
            else:
                # Dates missing on the handwritten slip: Lead is recorded in CRM, ask user for dates
                return {
                    "session_id": session.session_id,
                    "response": cards,
                    "booking_stage": "collecting",
                    "booking_summary": None,
                }

        # Special handling for payment receipts / proof
        if vision_res.image_type == "payment_proof":
            caption_text = (message or "").strip()
            caption_lower = caption_text.lower()
            # Check if there is an explicit side query (e.g. asking about location, wifi, checkin time)
            has_side_query = bool(re.search(
                r"\b(?:wifi|wi[- ]?fi|password|location|direction|directions|where|address|map|rasta|check\s*in\s*time|checkout|dinner|food|breakfast|lunch|restaurant|cab|taxi|station|airport)\b",
                caption_lower,
            ))

            if not caption_text or not has_side_query:
                # Direct confirmation output — preserves the verified payment message without degradation
                return {
                    "session_id": session.session_id,
                    "response": cards,
                    "booking_stage": stage,
                    "booking_summary": None,
                }
            else:
                # User asked a secondary query alongside receipt (e.g. "payment done, what is wifi password?")
                chat_res = await run_chat_turn(
                    message=caption_text,
                    session_id=session.session_id,
                    hotel_id=resolved_hid,
                    customer_phone=customer_phone,
                )
                # Prepend the booking confirmation card so confirmation is never lost
                chat_res["response"] = cards + (chat_res.get("response") or [])
                chat_res["booking_stage"] = stage
                return chat_res

        # If guest provided an accompanying text/caption with the image, process in dialogue graph
        if message and message.strip():
            analysis_note = (
                f"[IMAGE_ANALYSIS]\n"
                f"Type: Payment Screenshot\n" if vision_res.image_type == "payment_proof" else f"[Image Analysis: {vision_res.image_type.replace('_', ' ').title()}]\n"
                f"Details: {reply_text}\n"
                f"[/IMAGE_ANALYSIS]"
            )
            combined_message = f"{message.strip()}\n\n{analysis_note}"
            chat_res = await run_chat_turn(
                message=combined_message,
                session_id=session.session_id,
                hotel_id=resolved_hid,
                customer_phone=customer_phone,
            )
            # Ensure any media delivery cards (such as room catalog photos for the matched room) are preserved
            if any(c.get("type") == "media_delivery" for c in cards):
                media_cards = [c for c in cards if c.get("type") == "media_delivery"]
                chat_res["response"] = media_cards + (chat_res.get("response") or [])
            return chat_res

        return {
            "session_id": session.session_id,
            "response": cards,
            "booking_stage": stage,
            "booking_summary": None,
        }

    upload = StarletteUploadFile(
        filename=filename,
        file=BytesIO(file_bytes),
        headers={"content-type": content_type},
    )

    try:
        extracted_text = await process_upload(upload, message)
    except ValueError as e:
        return {
            "session_id": session_id or generate_session_id(),
            "response": [{"type": "text", "text": str(e)}],
            "booking_stage": "new",
            "booking_summary": None,
        }

    if message:
        sanitized = sanitize_input(message, max_length=settings.MAX_MESSAGE_LENGTH)
        combined = f"{sanitized}\n\n[Analysis from {filename}]:\n{extracted_text}"
    else:
        combined = f"[Analysis from {filename}]:\n{extracted_text}"

    if len(combined) > 50000:
        combined = combined[:50000] + "\n...[truncated]"

    return await run_chat_turn(
        message=combined,
        session_id=session_id,
        hotel_id=hotel_id,
        customer_phone=customer_phone,
    )
