"""Save State Node — persist conversation state to MongoDB (always last before END)."""

import logging
from datetime import datetime

from app.config.settings import get_settings
from app.models.schemas import MessageRecord
from app.repositories import session_repository
from app.services.repetition_guard import next_repeat_count
from app.services.reply_guard import strip_questions_about_known_data
from app.utils.guardrails import clean_whatsapp_formatting, sanitize_output
from app.workflows.session_mapper import state_to_session
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def apply_price_dispute_guard(cards: list[dict], state: ConversationState) -> list[dict]:
    """
    State the two numbers when the guest's total disagrees with ours.

    Never changes the price, never confirms anything: it removes the sentences
    that re-assert our figure and replaces them with an honest line plus one
    front-desk task. A failure here must not cost the guest their reply, so it is
    caught and the reply goes out as it was.
    """
    from app.services import price_dispute

    stored = state.get("price_override") or state.get("total_price")
    guest = price_dispute.rival_total(state.get("user_message") or "", stored)
    if guest is None:
        return cards

    try:
        await price_dispute.ensure_dispute_task(
            hotel_id=state.get("hotel_id", 0),
            session_id=state.get("session_id"),
            phone=state.get("customer_identifier"),
            booking_id=state.get("booking_id"),
            guest_total=guest,
            stored_total=float(stored),
        )
    except Exception as e:  # noqa: BLE001 - the reply matters more than the task
        logger.warning("PRICE_DISPUTE_TASK_FAILED | %s", e)

    from app.utils.booking_resume import is_hindi_or_hinglish

    text = price_dispute.notice(
        guest, float(stored),
        is_hi=is_hindi_or_hinglish(state, state.get("user_message") or ""),
    )
    scrubbed = price_dispute.scrub(cards, float(stored))
    logger.warning(
        "PRICE_DISCREPANCY | session=%s guest=%.0f stored=%.0f",
        state.get("session_id"), guest, float(stored),
    )
    if not scrubbed:
        return [{"type": "text", "text": text}]
    # Put the notice first: the guest's question is the reason for the turn, and
    # whatever follows is only useful once they know it is being looked at.
    first = scrubbed[0]
    if isinstance(first, dict) and first.get("type") == "text":
        return [{**first, "text": f"{text}\n\n{first['text']}"}] + scrubbed[1:]
    return [{"type": "text", "text": text}] + scrubbed


def flatten_response_cards(cards: list) -> list[dict]:
    """Flatten nested card lists (text_card() already returns a list)."""
    flat: list[dict] = []
    for item in cards:
        if isinstance(item, list):
            flat.extend(flatten_response_cards(item))
        elif isinstance(item, dict):
            flat.append(item)
    return flat


async def save_state(state: ConversationState) -> ConversationState:
    """
    Persist the conversation state back to MongoDB.
    This is always the LAST node before END.
    """
    bot_response = flatten_response_cards(state.get("bot_response", []))

    for card in bot_response:
        if isinstance(card, dict) and card.get("text"):
            card["text"] = clean_whatsapp_formatting(card["text"])

    # Drop questions whose answer is already in the state. Done here, before the
    # reply is fingerprinted and persisted, so the guard's output is what the
    # guest sees, what the transcript records, and what the repeat counter
    # measures — a stripped question must not count as a second send.
    if get_settings().REPLY_KNOWN_DATA_GUARD_ENABLED:
        bot_response = strip_questions_about_known_data(bot_response, state)

    # A guest who has priced the stay differently gets the disagreement stated,
    # not our figure repeated. Runs after the known-data guard so the sentence it
    # adds cannot be stripped as a redundant question.
    if get_settings().PRICE_DISPUTE_GUARD_ENABLED:
        bot_response = await apply_price_dispute_guard(bot_response, state)

    # While a claimed payment waits on a human, no reply may ask for money or
    # carry a link. Last of the guards because it is the cheapest place to
    # enforce an invariant over every producer at once — including a FAQ answer
    # retrieved from the knowledge base, which is how the 25% / razorpay.me
    # paragraph reached the guest in the first place.
    if get_settings().PENDING_PAYMENT_GUARD_ENABLED:
        from app.services.pending_payment_guard import enforce
        from app.utils.booking_resume import get_booking_resume_nudge

        bot_response = enforce(
            bot_response,
            state,
            get_booking_resume_nudge(state) or "",
        )

    # Count consecutive identical replies. This has to happen before state_to_session,
    # because that call is what writes to Mongo: anything computed after it is returned to END
    # but never persisted. The values go in as overrides for the same reason.
    last_hash, repeat_count = next_repeat_count(
        state.get("last_response_hash"),
        bot_response,
        state.get("repeat_response_count"),
    )
    if repeat_count > 1:
        logger.info(
            "Repeat reply detected (%s in a row) | session=%s",
            repeat_count,
            state.get("session_id"),
        )

    session = state_to_session(
        state,
        include_messages=True,
        last_response_hash=last_hash,
        repeat_response_count=repeat_count,
    )

    # Preserve conversation_overrides if this turn's state omitted them
    # (prevents wiping Human Takeover / future domain overrides).
    if not state.get("conversation_overrides"):
        existing = await session_repository.get_session(session.session_id)
        if existing and getattr(existing, "conversation_overrides", None):
            session.conversation_overrides = existing.conversation_overrides

    if state.get("user_message"):
        session.messages.append(MessageRecord(
            role="user",
            content=state["user_message"],
            sender_type="guest",
            timestamp=datetime.utcnow(),
        ))

    if bot_response:
        bot_text_parts = []
        for card in bot_response:
            if card.get("text"):
                bot_text_parts.append(card["text"])
            elif card.get("type") == "room_options":
                bot_text_parts.append("[Showing Room Options]")
            elif card.get("type") == "quote":
                bot_text_parts.append("[Showing Price Quote]")
            elif card.get("type") == "payment":
                bot_text_parts.append("[Showing Payment Link]")

        bot_text_combined = sanitize_output("\n".join(bot_text_parts))
        session.messages.append(MessageRecord(
            role="assistant",
            content=bot_text_combined,
            sender_type="bot",
            timestamp=datetime.utcnow(),
        ))

    session.messages = session.messages[-20:]

    await session_repository.upsert_session(session)
    logger.info(f"Session saved: {session.session_id} | Stage: {session.booking_stage.value}")

    return {
        **state,
        "bot_response": bot_response,
        "last_response_hash": last_hash,
        "repeat_response_count": repeat_count,
    }
