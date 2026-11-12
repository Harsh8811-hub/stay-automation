"""Make the bot *speak* when a human takeover ends.

`POST /api/admin/context-refresh` has always summarised the staff conversation
and written it into the session — and then stopped. It saved memory and sent
nothing, so a guest sitting in a paused chat heard silence until they happened to
write again, and the bot's first real turn after a takeover began from scratch as
far as the guest could tell.

The CRM does its half: `run_override_scheduler` calls this endpoint the moment
its `ManualOverride` expires. This module is the other half — one message, from
the bot, continuing the conversation with whatever the colleague settled.

Deliberately *not* a graph turn. There is no guest message to answer, and driving
`run_chat_turn` would mean inventing one and writing it into history as though the
guest had said it. The staff decisions are already applied to the session by
`save_context_to_session` (which runs first), so the pricing and policy paths
honour them on the guest's *next* real turn regardless of what this sends; this
is the hand-back, not the transaction.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# A resume message is a conversational hand-back, so it must not be long. The
# composer is told once; this is the backstop for when it is ignored.
MAX_RESUME_CHARS = 600


def _parse_iso(value: Any) -> datetime | None:
    """Parse the CRM's `bot_off_time`.

    The CRM sends `override.bot_off_time.isoformat() + "Z"`, and mongoengine
    stores that field naive-UTC, so the usual shape is
    `2026-10-08T13:22:03.123456Z`. Tolerate an offset-bearing value too rather
    than guess: a misread timestamp silently disables the guard below.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        # Try in order: as sent; `Z` read as UTC; `Z` simply dropped. That last
        # one covers the CRM appending `Z` to an isoformat() that already ended
        # in "+00:00" — malformed, but real, and the parse must not silently
        # fail into the fallback cutoff.
        candidates = [text]
        if text.endswith("Z"):
            candidates += [text[:-1] + "+00:00", text[:-1]]
        dt = None
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate)
                break
            except ValueError:
                continue
        if dt is None:
            logger.warning(f"TAKEOVER_RESUME_BAD_TIMESTAMP | value={text[:40]}")
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _sender_of(m) -> str:
    """Classify one stored message as `guest`, `bot`, `staff` or `""`.

    `sender_type` is the authority and `role` is the fallback, because
    `sender_type` was added later with a default of "guest": rows written before
    it exist with no sender type at all, and reading those by `role` alone is
    how a bot turn gets mistaken for a guest turn. System notes are `""` — they
    are bookkeeping, not conversation.
    """
    sender = str(getattr(m, "sender_type", None) or "").strip().lower()
    role = str(getattr(m, "role", None) or "").strip().lower()
    if sender == "staff":
        return "staff"
    if sender == "bot" or role == "assistant":
        return "bot"
    if sender == "guest" or role == "user":
        return "guest"
    return ""


def _conversation(session) -> list[tuple[Any, str, str]]:
    """`(when, who, text)` for the session's own messages, oldest first.

    Only the bot side of the story: `session.messages` holds guest and bot turns
    and a single system note. Staff turns are not here — see `_rows_from_history`.
    """
    out: list[tuple[Any, str, str]] = []
    for m in (getattr(session, "messages", None) or []):
        who = _sender_of(m)
        if who:
            out.append((getattr(m, "timestamp", None), who, str(getattr(m, "content", "") or "").strip()))
    return out


def _rows_from_history(chats) -> list[tuple[Any, str, str]]:
    """`(when, who, text)` from the merged CRM history, oldest first.

    This is the only source that contains what the colleague typed. The bot
    never sees a staff message through its own graph — during a takeover the CRM
    answers the guest and the bot is paused — so `session.messages` alone reads
    every takeover as guest-terminated. The refresh already fetches this history
    to write the summary; deciding from it costs nothing extra.

    `crm_service._merge_history` has already de-duplicated and ordered it, and
    `staff_message_detector` is the authority on who is staff: the CRM spells a
    colleague `sender="user"`, the bot's own store spells it `sender="staff"`.
    """
    from app.services.crm_service import _parse_datetime
    from app.services.staff_message_detector import (
        is_crm_staff_message,
        is_mongo_staff_message,
    )

    out: list[tuple[Any, str, str]] = []
    for m in (chats or []):
        if not isinstance(m, dict):
            continue
        if is_crm_staff_message(m) or is_mongo_staff_message(m):
            who = "staff"
        else:
            sender = str(
                m.get("sender") or m.get("role") or m.get("direction") or ""
            ).strip().lower()
            if sender in ("bot", "assistant", "ai"):
                who = "bot"
            elif sender in ("guest", "customer"):
                who = "guest"
            else:
                # A system note, or a row nobody can attribute. It is not a turn.
                continue
        text = str(
            m.get("content") or m.get("text") or m.get("text_body") or m.get("message") or ""
        ).strip()
        out.append((_parse_datetime(m.get("created_at") or m.get("timestamp") or m.get("time")), who, text))
    return out


def _bot_spoke_since(session, cutoff: datetime) -> bool:
    """True if the bot has already said something since the takeover began.

    A bot message means the hand-back happened on its own and a second message
    now would talk over it. This reads the session, not the history, because the
    session is where our own sends are recorded — the CRM copy lags.

    The system note `save_context_to_session` appends is excluded — it is
    `role="system"`, written seconds before this runs.

    Guest messages deliberately do *not* count. They used to: the old guard
    skipped whenever either party spoke, so a guest who wrote while staff were
    mid-conversation and never got an answer was handed silence — the one case
    where the bot is most obviously needed.
    """
    for m in (getattr(session, "messages", None) or []):
        if _sender_of(m) != "bot":
            continue
        ts = getattr(m, "timestamp", None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts > cutoff:
            return True
    return False


# A guest turn with no text is not silence — it is a room-card tap, a photo of a
# competitor's board, or a shared screenshot. Dropping it would let a guest who
# only ever sent a picture fall through to "nothing to answer", so it is put to
# the composer as what it is.
NO_TEXT_PLACEHOLDER = "[the guest sent this with no text — an image or a button tap]"


def _resume_fingerprint(unanswered: list[str]) -> str | None:
    """A short, stable id for one run of unanswered guest messages.

    Content only, deliberately: timestamps move between refreshes (the CRM rebuilds
    its window each call) while the words the guest sent do not. Whitespace and case
    are normalised so a re-fetch that pads a string cannot produce a second send.
    """
    import hashlib

    if not unanswered:
        return None
    joined = "␟".join(" ".join(str(t).split()).lower() for t in unanswered)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _unanswered_guest_messages(rows) -> list[str]:
    """Guest messages that came after the last thing staff or the bot said.

    That trailing run is the guest talking to nobody: staff left without
    replying, and the bot was paused. These are the messages the waking bot is
    supposed to answer.
    """
    tail: list[str] = []
    for _, who, text in reversed(rows):
        if who != "guest":
            break
        tail.append(text or NO_TEXT_PLACEHOLDER)
    tail.reverse()
    return tail


def decide_resume_action(session, cutoff: datetime, history=None) -> tuple[str, object]:
    """What the waking bot should do, decided before any model call.

    Returns one of:

      ("skip", reason)      nothing is sent
      ("answer", [texts])   reply to these unanswered guest messages

    The rule is the guest's turn order, not a timer. Staff with the last word
    means the guest is already being served — a bot message there is the
    unprompted chatter that got this feature switched off on 2026-10-27. A guest
    with the last word is the opposite: somebody is waiting for an answer, so
    the bot answers instead of announcing itself.

    `history` is the merged CRM transcript from the refresh that called us. It is
    preferred over the session because it is the only one carrying staff turns;
    the session is the fallback for callers that have none.
    """
    if _bot_spoke_since(session, cutoff):
        return ("skip", "bot_already_spoke")

    rows = _rows_from_history(history) or _conversation(session)
    if not rows:
        return ("skip", "no_messages")

    last = rows[-1][1]
    if last == "staff":
        return ("skip", "staff_had_last_word")
    if last == "bot":
        return ("skip", "bot_had_last_word")
    if last != "guest":
        return ("skip", "unclassifiable_tail")

    unanswered = _unanswered_guest_messages(rows)
    if not unanswered:
        return ("skip", "no_unanswered")
    return ("answer", unanswered)


def resume_should_send(session, cutoff: datetime) -> bool:
    """The decision as a yes/no, for callers that only need the answer."""
    return decide_resume_action(session, cutoff)[0] == "answer"


async def resume_after_takeover(
    *,
    hotel_id: int | str,
    phone: str,
    started_at: Any = None,
    history=None,
) -> bool:
    """Send one continuation message if the bot has been silent since `started_at`.

    `history` is the merged CRM transcript the context refresh already fetched.
    Pass it: without it the decision cannot see what the colleague said, and
    "staff had the last word" is unanswerable.

    Returns True only when a message was actually sent. Every failure path
    returns False and logs: this runs inside the CRM's background request, and a
    raise here would take the context write down with it.
    """
    settings = get_settings()
    if not getattr(settings, "TAKEOVER_RESUME_ENABLED", False):
        # Logged, not silent. While this flag was off, production had no
        # `TAKEOVER_RESUME_*` line at all for eight days, so "the bot never wakes
        # up" was indistinguishable in the logs from "nothing needed answering".
        logger.info("TAKEOVER_RESUME_DISABLED | phone=%s | flag=off", phone)
        return False

    from app.repositories import session_repository
    from app.utils.phone import digits_only

    digits = digits_only(phone) or str(phone).strip()
    if not digits:
        return False

    try:
        hid = int(hotel_id) if str(hotel_id).isdigit() else hotel_id
    except (TypeError, ValueError):
        hid = hotel_id

    # `started_at` is when the colleague took over. Without it, fall back to one
    # timeout window ago — that is the moment the CRM decided the takeover ended.
    cutoff = _parse_iso(started_at) or (
        datetime.now(UTC) - timedelta(seconds=int(settings.TAKEOVER_TIMEOUT_SECONDS))
    )

    try:
        session = await session_repository.find_active_session(hid, digits)
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_SESSION_LOOKUP_FAILED | phone={digits} | error={e}")
        return False

    if session is None:
        # No live conversation to continue. The summary is already parked in
        # `pending_context` and will apply on the guest's next turn.
        logger.info(f"TAKEOVER_RESUME_SKIP | phone={digits} | reason=no_active_session")
        return False

    action, detail = decide_resume_action(session, cutoff, history=history)
    if action == "skip":
        logger.info(f"TAKEOVER_RESUME_SKIP | phone={digits} | reason={detail}")
        return False

    # `answer` carries the guest messages nobody replied to; `greet` carries
    # nothing and keeps the original hand-back behaviour.
    unanswered: list[str] = list(detail) if action == "answer" else []
    logger.info(
        f"TAKEOVER_RESUME_ACTION | phone={digits} | action={action} | "
        f"unanswered={len(unanswered)}"
    )

    # The same unanswered guest turn must not be answered twice. The refresh runs
    # many times per conversation (the CRM's override scheduler calls it minutes
    # apart), and `_bot_spoke_since` cannot be relied on for this alone: it compares
    # against the window's own start time, which moves every time the CRM starts a
    # new window, so the guard lapses and the guest gets the same message again.
    fingerprint = _resume_fingerprint(unanswered)
    if fingerprint and getattr(session, "takeover_resume_fingerprint", None) == fingerprint:
        logger.info(
            f"TAKEOVER_RESUME_SKIP | phone={digits} | reason=already_answered "
            f"| at={getattr(session, 'takeover_resume_at', None)}"
        )
        return False

    # The composer needs the same transcript the decision used, or it is asked
    # not to repeat "what the colleague already said" without being shown it.
    # `sender_type` is what `history_label` reads: staff and bot both render as
    # "Us", the guest as "Guest".
    rows = _rows_from_history(history) or _conversation(session)
    transcript = [
        {"sender_type": who, "content": text}
        for _ts, who, text in rows
        if text
    ]

    from app.services.gemini_service import generate_takeover_resume_message

    summary = _latest_staff_summary(session)
    decisions = _staff_decisions(session)
    hotel_name = "our hotel"
    try:
        from app.repositories import hotel_repository

        details = await hotel_repository.get_hotel_details(hid) or {}
        hotel_name = (
            (details.get("property") or {}).get("name")
            or details.get("hotel_name")
            or details.get("name")
            or hotel_name
        )
    except Exception as e:
        logger.warning(f"TAKEOVER_RESUME_HOTEL_LOOKUP_FAILED | hotel={hid} | error={e}")

    # The composer gets its own try. `6228e39` dedented this block into the
    # hotel-lookup `except` above, so `body` was only assigned when the lookup
    # *failed*, and on the normal path the `.strip()` below raised
    # `UnboundLocalError` (prod, 2026-10-22 13:46 — the only resume attempt
    # after that commit before the next deploy). Same commit left the two
    # guards below as siblings of the lookup's try, where they could never see
    # a composer failure. One `try` for one call restores both.
    from app.services.llm_errors import LLMUnavailableError

    try:
        body = await generate_takeover_resume_message(
            hotel_name=hotel_name,
            summary=summary,
            decisions=decisions,
            recent_messages=transcript or [
                m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m)
                for m in (getattr(session, "messages", None) or [])
            ],
            unanswered=unanswered,
        )
    except LLMUnavailableError as e:
        logger.warning(f"TAKEOVER_RESUME_DEGRADED | phone={digits} | reason=llm_unavailable | error={e}")
        session.context_refresh_status = "degraded_llm_unavailable"
        await session_repository.upsert_session(session)
        return False
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_COMPOSE_FAILED | phone={digits} | error={e}")
        return False

    body = (body or "").strip()
    if not body:
        # The composer returns "" on a Gemini failure, and a blank WhatsApp send
        # is worse than silence.
        logger.info(f"TAKEOVER_RESUME_SKIP | phone={digits} | reason=empty_message")
        return False
    if len(body) > MAX_RESUME_CHARS:
        body = body[: MAX_RESUME_CHARS - 1].rstrip() + "…"

    from app.services.whatsapp_service import whatsapp_service

    try:
        sent = whatsapp_service.send_message(
            to_number=digits,
            message_body=body,
            hotel_id=hid,
        )
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_SEND_FAILED | phone={digits} | error={e}")
        return False

    if not sent:
        logger.warning(f"TAKEOVER_RESUME_SEND_REJECTED | phone={digits}")
        return False

    # Record it, so the guest's next turn sees what the bot said — and so the
    # `_bot_spoke_since` guard refuses a second resume for this takeover. The
    # fingerprint is written beside it, which is the guard that survives the CRM
    # starting a new window: the same unanswered turn cannot be answered again.
    try:
        from app.models.schemas import MessageRecord

        session.messages.append(
            MessageRecord(role="assistant", content=body, sender_type="bot")
        )
        if fingerprint:
            session.takeover_resume_fingerprint = fingerprint
            session.takeover_resume_at = datetime.now(UTC)
        await session_repository.upsert_session(session)
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_PERSIST_FAILED | phone={digits} | error={e}")

    logger.info(
        f"TAKEOVER_RESUME_SENT | hotel={hid} | phone={digits} | chars={len(body)}"
    )
    return True


def _latest_staff_summary(session) -> str:
    """The prose `save_context_to_session` just wrote, read back off the session.

    The refresh endpoint applies the summary through `apply_staff_decisions`
    (filing it in the audit trail and `add_note`) or appends it as a system
    message; this reads whichever landed rather than passing it twice through
    two call sites that could drift.
    """
    for m in reversed(getattr(session, "messages", None) or []):
        if getattr(m, "role", None) == "system" and "Summary:" in (getattr(m, "content", "") or ""):
            return str(m.content).split("Summary:", 1)[1].strip()
    return ""


def _staff_decisions(session) -> dict:
    """The typed decisions already applied to the session, read off the overrides.

    These are the real agreed values — `apply_staff_decisions` files them in the
    domain dicts. `notes` is deliberately not read: it holds the prose summary,
    which is already passed separately, and the two would just duplicate.
    """
    overrides = getattr(session, "conversation_overrides", None)
    if overrides is None:
        return {}

    agreed: dict[str, Any] = {}
    for domain in ("pricing", "payment", "amenities", "inventory", "booking"):
        values = getattr(overrides, domain, None)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if value not in (None, "", [], {}):
                agreed[f"{domain}.{key}"] = value
    return agreed
