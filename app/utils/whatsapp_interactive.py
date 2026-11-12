"""
WhatsApp interactive messages — reply buttons and list messages.

Builds the Meta-shaped payload for a response card of type `interactive`. The CRM
forwards `{"type": "interactive", "data": {...}}` from the bot straight to Meta
(combot_crm_v2 `webhook_receiver/views.py:305-307`), so nothing here has to know how
the message is transported.

Two things about this channel shape the whole design:

1. **Meta rejects the entire message when a limit is exceeded** — it does not truncate.
   So every title is clamped here. Letting Meta reject it means the guest receives
   *nothing*, which is worse than a slightly shortened label.

2. **The button `id` never reaches the bot.** The CRM unwraps a tap into the button's
   *title* and sends that as plain text (`webhook_receiver/views.py:119`). We keep
   emitting stable ids anyway, so that if the CRM ever forwards `button_reply.id` the
   router can prefer it with no change here. Until then, routing matches on the title,
   which is sound only because we author it — see `app/utils/button_reply.py`.

Limits, verified against Meta's Node.js SDK type reference 2026-10-13:

    reply button title  20      list row title        24
    reply button id     256     list row description  72
    interactive body    1024    list opening button   20
    reply buttons       max 3   list rows             max 10
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BUTTON_TITLE_MAX = 20
LIST_ROW_TITLE_MAX = 24
LIST_ROW_DESC_MAX = 72
LIST_BUTTON_MAX = 20
BODY_MAX = 1024
FOOTER_MAX = 60
HEADER_MAX = 60

MAX_BUTTONS = 3
MAX_ROWS = 10

# The card type the CRM recognises and passes through untouched.
CARD_TYPE = "interactive"


def _clamp(text: object, limit: int) -> str:
    """
    Trim to `limit`, collapsing whitespace first so a clamp never lands mid-space.

    For single-line fields only — button and row titles, descriptions, ids. Use `_clip`
    for anything the guest reads as prose.
    """
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip()


def _clip(text: object, limit: int) -> str:
    """
    Trim to `limit` but keep the line breaks.

    The interactive body is the bot's own prose, and our prompts are written with blank
    lines between paragraphs. Collapsing those turns a four-paragraph welcome into one
    wall of text — and the whole reason the body exists is that the guest still has to be
    able to read it, since it is what they see if they never tap anything.
    """
    trimmed = str(text or "").strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[:limit].rstrip()


def normalize_reply_text(text: object) -> str:
    """
    The one normalisation used on both sides of a button — when we author the title,
    and when we match the guest's reply back to it.

    Casefold and whitespace only. Anything cleverer (stripping punctuation, dropping
    stop words) would start matching things the guest typed rather than things we
    offered, which is exactly the ambiguity buttons exist to remove.

    WhatsApp may hand back a title with trailing whitespace, and our own copy is
    authored with newlines in places, so both sides must agree on the same reduction.
    """
    return " ".join(str(text or "").split()).casefold()


def reply_buttons(
    body: str,
    buttons: list[dict],
    *,
    header: str | None = None,
    footer: str | None = None,
) -> dict | None:
    """
    A message with up to 3 tappable reply buttons.

    `buttons` is a list of `{"id": str, "title": str, "action": dict}`. `action` is the
    state update this button means — carried *on the button* rather than looked up from
    a table elsewhere, so a copy edit can never desynchronise the label from its effect.

    Returns `None` when there is nothing worth sending (no usable buttons), so callers
    fall back to text rather than sending an empty interactive message.
    """
    usable = [b for b in (buttons or []) if str(b.get("title") or "").strip()][:MAX_BUTTONS]

    if not usable:
        logger.warning("reply_buttons called with no usable buttons; caller must fall back")
        return None

    if len(buttons) > MAX_BUTTONS:
        # Dropping the tail silently would change what the guest can do, so say so.
        logger.warning(
            "reply_buttons: %d buttons offered, Meta allows %d — the rest are not sent",
            len(buttons), MAX_BUTTONS,
        )

    payload: dict = {"type": "button"}

    if header:
        payload["header"] = {"type": "text", "text": _clip(header, HEADER_MAX)}
    payload["body"] = {"text": _clip(body, BODY_MAX)}
    if footer:
        payload["footer"] = {"text": _clip(footer, FOOTER_MAX)}

    payload["action"] = {
        "buttons": [
            {
                "type": "reply",
                "reply": {
                    "id": _clamp(b.get("id") or b["title"], 256),
                    "title": _clamp(b["title"], BUTTON_TITLE_MAX),
                },
            }
            for b in usable
        ]
    }

    return {
        "type": CARD_TYPE,
        "data": payload,
        # What the router may match against, and what to do when it matches.
        "offered": [_offer(b, "button") for b in usable],
    }


def option_list(
    body: str,
    button_text: str,
    rows: list[dict],
    *,
    header: str | None = None,
    footer: str | None = None,
) -> dict | None:
    """
    A message with an opening button that reveals up to 10 tappable rows.

    The right shape whenever there are more than 3 choices, or when a label is longer
    than a button title allows — which, at 20 characters, includes most of our room
    names ("PREMIER ROOM KING BED" is 21).

    `rows` is a list of `{"id": str, "title": str, "description": str, "action": dict}`.
    """
    usable = [r for r in (rows or []) if str(r.get("title") or "").strip()][:MAX_ROWS]

    if not usable:
        logger.warning("option_list called with no usable rows; caller must fall back")
        return None

    if len(rows) > MAX_ROWS:
        logger.warning(
            "option_list: %d rows offered, Meta allows %d — the rest are not sent",
            len(rows), MAX_ROWS,
        )

    payload: dict = {"type": "list"}

    if header:
        payload["header"] = {"type": "text", "text": _clip(header, HEADER_MAX)}
    payload["body"] = {"text": _clip(body, BODY_MAX)}
    if footer:
        payload["footer"] = {"text": _clip(footer, FOOTER_MAX)}

    rendered_rows = []
    for r in usable:
        row = {
            "id": _clamp(r.get("id") or r["title"], 256),
            "title": _clamp(r["title"], LIST_ROW_TITLE_MAX),
        }
        description = _clamp(r.get("description"), LIST_ROW_DESC_MAX)
        if description:
            row["description"] = description
        rendered_rows.append(row)

    payload["action"] = {
        "button": _clamp(button_text, LIST_BUTTON_MAX),
        "sections": [{"title": _clamp(header or "Options", LIST_ROW_TITLE_MAX),
                      "rows": rendered_rows}],
    }

    return {
        "type": CARD_TYPE,
        "data": payload,
        "offered": [_offer(r, "row") for r in usable],
    }


def _offer(item: dict, kind: str) -> dict:
    """
    One entry in the card's `offered` list — everything the router needs to recognise a
    tap and act on it. The title is stored **raw**, not clamped, so the router compares
    against the same string that will appear in `offered`; the clamped form only ever
    exists inside the Meta payload.
    """
    return {
        "id": str(item.get("id") or item["title"]),
        "title": " ".join(str(item["title"]).split()),
        "kind": kind,
        "action": item.get("action") or {},
    }


# ── Translating the authored card vocabulary ─────────────────────────────────

# The card types `app.utils.template_renderers` authors. These are the vocabulary the
# prompts speak; they are not anything Meta has ever heard of.
AUTHORED_TYPES = ("button_group", "list_message")


def to_interactive(card: dict) -> dict | None:
    """
    Turn an authored card into the Meta `interactive` payload the CRM forwards verbatim.

    There are two card vocabularies in this codebase and only one of them can be sent.
    The templates author `button_group` / `list_message`; Meta — and therefore the CRM,
    which hands `data` straight to the WhatsApp Cloud API — accepts `interactive`. Nothing
    performed that translation, and `format_cards_for_whatsapp` has no branch for either
    authored type, so every button was silently flattened to its body text on the way out.
    A guest saw "Here is what I can help you with at Sample Hotel:" and nothing else.

    Returns `None` for a card we cannot translate, so the caller falls back to text
    rather than sending nothing at all.
    """
    if not isinstance(card, dict):
        return None

    ctype = card.get("type")

    if ctype == "button_group":
        return reply_buttons(
            card.get("text") or "",
            card.get("buttons") or [],
            header=card.get("header"),
            footer=card.get("footer"),
        )

    if ctype == "room_options":
        # The room card is a long body with the priced combinations in it, and
        # its buttons are the same three choices as tappable reply buttons.
        #
        # Meta *rejects* a body over 1024 characters rather than truncating it,
        # and `reply_buttons` would clip it — which on this card means losing
        # the footer and the reply line, or a price. So an oversized card is
        # not buttoned at all: returning None makes the caller send it as text,
        # exactly as it did before buttons existed. A big party's card is
        # readable and un-tappable, which beats unreadable.
        text = card.get("text") or ""
        buttons = card.get("buttons") or []
        if not buttons or len(text) > BODY_MAX:
            return None
        payload = reply_buttons(text, buttons)
        if payload is not None:
            # Keep the card's own prose on the payload. The CRM turns `data` into
            # the WhatsApp message and ignores this, but the chat UIs render `text`
            # and know nothing about `interactive` — without it they would show a
            # buttoned card with nothing above the buttons.
            payload["text"] = text
        return payload

    if ctype == "list_message":
        rows = [row for section in (card.get("sections") or []) for row in (section.get("rows") or [])]
        return option_list(
            card.get("text") or "",
            card.get("button_title") or "Options",
            rows,
            footer=card.get("footer"),
        )

    return None
