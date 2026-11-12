"""
Repetition as a failure signal (B3b).

Why the existing counter could not catch this
--------------------------------------------
`escalation_failure_count` already exists and is already persisted, and `routing.py:39` already
escalates at 3 — the machinery was all there and never fired. The reason is what it counts:
`understand.py:1021` increments it only when the resolved intent is `UNKNOWN`. So it measures
*"the model did not understand"*.

Every reported loop was well understood. `919876543210` got
"I received your image, but it doesn't appear to be a payment receipt…" nineteen times — a
confident, correct classification of a guest who was actually asking for room photos. The model
understood every turn; the conversation still never moved. Nothing counted that.

What repetition means, and when it is not a failure
--------------------------------------------------
The signal is: we have sent this exact reply before, and the guest is still here. Not "the reply
was short", not "the model was unsure" — the same reply, again. A guest legitimately paging
through options with `MORE` gets *different* text each time, which is why the fingerprint below
is taken from the cards' own `text` fields rather than from the flattened transcript. The
transcript collapses every room-options page to `[Showing Room Options]` (`save_state.py:60`), so
fingerprinting that would read three healthy `MORE` taps as a loop and escalate a guest who was
happily browsing.

Honest limitation: this counts consecutive identical replies, so a guest who sends three
different messages that each legitimately deserve the same clarifying reply ("what dates?",
"what dates?", "what dates?") will escalate. That is intended — three identical replies in a row
*is* the loop, whatever caused it — but it does mean this fires on a confused guest, not only on
a broken bot. The escalation is a handoff to a human, which is the right answer for both.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

# Three consecutive identical replies have gone out; routing escalates on the turn after.
# Matches the existing failure threshold in routing.py:39, so an operator reading logs sees one
# number, not two. The counting in `next_repeat_count` makes this literally true — see its
# docstring for why that required saying.
REPEAT_ESCALATION_THRESHOLD = 3


def _body_and_labels(card: dict) -> tuple[str | None, list[str]]:
    """
    What a card says and what its controls offer, read from either shape it can arrive in.

    Two card vocabularies exist and both are plausible here. The authored one —
    `button_group` / `list_message` from `app.utils.template_renderers` — keeps `text` and
    the buttons at the top level, and is what actually reaches `save_state` today, because
    the translation to Meta's `interactive` shape happens in the API layer after the graph
    (`chat.py:_collapse_for_crm_whatsapp`).

    The wire one buries both under `data`. Reading only the authored shape is therefore
    "correct today" in the way that fails silently: the day a wire card is stored instead,
    every one of them loses its text and its buttons, fingerprints as the bare constant
    `[interactive]`, and three *different* prompts in a row read as a loop.

    Both shapes cost the same to read, so both are read.
    """
    if not isinstance(card, dict):
        return None, []

    data = card.get("data") if isinstance(card.get("data"), dict) else {}

    # Wire shape: {"type": "interactive", "data": {"body": {"text": ...}, "action": ...}}
    body = data.get("body") if isinstance(data.get("body"), dict) else {}
    text = card.get("text") or card.get("caption") or body.get("text")
    if text in ("", None):
        text = None

    action = data.get("action") if isinstance(data.get("action"), dict) else {}

    labels: list[str] = []

    # Authored list cards nest rows under sections; authored button cards keep them flat.
    rows = [row for section in (card.get("sections") or [])
            for row in ((section or {}).get("rows") or [])]
    for item in [*(card.get("buttons") or []), *rows]:
        if isinstance(item, dict):
            label = item.get("title") or item.get("label") or item.get("text") or item.get("id")
        else:
            label = item
        if label:
            labels.append(str(label))

    # Wire shape: replies under action.buttons, rows under action.sections.
    for button in (action.get("buttons") or []):
        reply = button.get("reply") if isinstance(button, dict) else None
        if isinstance(reply, dict) and reply.get("title"):
            labels.append(str(reply["title"]))
    for section in (action.get("sections") or []):
        for row in ((section or {}).get("rows") or []):
            if isinstance(row, dict) and row.get("title"):
                labels.append(str(row["title"]))

    return text, labels


def response_fingerprint(bot_response: Iterable[Any] | None) -> str:
    """
    A stable fingerprint of what we actually sent.

    Built from each card's own `text` (falling back to `url` for a bare image, then the card
    type), NOT from the flattened transcript — see the module docstring for why that
    distinction is load-bearing rather than cosmetic.

    Returns "" for an empty response, which the caller treats as "nothing to compare".
    """
    parts: list[str] = []
    for card in bot_response or []:
        if not isinstance(card, dict):
            continue
        text, labels = _body_and_labels(card)
        if text or labels:
            # The labels belong in the fingerprint, not just the prose. Two cards can share
            # a body — "Choose an option" over different buttons — and fingerprinting only
            # the body would call them the same reply and count a run that never happened.
            # Added as separate parts rather than joined: a joining character is one more
            # thing that can appear inside a title and split it somewhere new.
            if text:
                parts.append(str(text))
            parts.extend(labels)
            continue
        url = card.get("url") or card.get("image_url") or card.get("media_url")
        if url:
            parts.append(f"[{card.get('type', 'media')}]{url}")
            continue
        parts.append(f"[{card.get('type', 'unknown')}]")

    if not parts:
        return ""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def next_repeat_count(
    previous_hash: str | None,
    bot_response: Iterable[Any] | None,
    previous_count: int | None = None,
) -> tuple[str, int]:
    """
    (fingerprint to persist, repeat count to persist) for the turn that just ended.

    The count is the **length of the trailing run of identical replies** — so the first reply of
    a session is a run of one, not zero, and a reply that differs from the last starts a fresh
    run of one. That definition is what makes `REPEAT_ESCALATION_THRESHOLD = 3` mean what it
    says: three identical replies have gone out, and routing escalates on the fourth turn. An
    earlier version of this function left the first reply at 0, which made the constant a lie by
    one — the bot sent the same message four times while the code claimed three.

    A break in the run is what stops a long, healthy conversation escalating because it happened
    to ask for dates three times over twenty turns: only *consecutive* repeats accumulate.

    An empty or unavailable fingerprint counts as no run at all (0) rather than as a repeat: a
    turn that produced no text is not evidence of a loop, and treating it as one would make the
    counter climb on silence.
    """
    fingerprint = response_fingerprint(bot_response)
    if not fingerprint:
        return "", 0

    if fingerprint == (previous_hash or ""):
        return fingerprint, int(previous_count or 0) + 1

    return fingerprint, 1


# ═══════════════════════════════════════════════════════════════════
# Guest Repetition & Progress Tracking (Issue 3 Root Cause C)
# ═══════════════════════════════════════════════════════════════════

import re


def normalize_text_for_repetition(text: str) -> str:
    """Normalize text for repetition comparison by removing punctuation and collapsing whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return " ".join(cleaned.split())


def is_user_repetition(msg1: str, msg2: str) -> bool:
    """
    Check if two user messages are substantially identical.
    Handles exact matches, minor punctuation/whitespace differences,
    and polite filler words (e.g. 'please', 'bhai', 'sir').
    """
    n1 = normalize_text_for_repetition(msg1)
    n2 = normalize_text_for_repetition(msg2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True

    # Strip conversational politeness fillers that don't change intent
    filler = {"please", "kripya", "bhai", "bhaiya", "sir", "madam", "ji", "can you", "plz", "bro"}
    words1 = set(n1.split()) - filler
    words2 = set(n2.split()) - filler
    if words1 and words2:
        if words1 == words2:
            return True
        jaccard = len(words1 & words2) / len(words1 | words2)
        if jaccard >= 0.8:
            return True
    return False


#: A bare menu answer: "1", "3", "B", "option 2". Every numbered prompt in the
#: booking flow draws from this same handful of tokens, so the same one answering
#: three *different* questions is ordinary traffic, not a guest in a loop.
_BARE_MENU_ANSWER = re.compile(r"^(?:(?:option|opt|choice)\s*)?(?:\d{1,2}|[a-d])$", re.IGNORECASE)


def is_bare_menu_answer(text: str) -> bool:
    """
    Is this reply nothing but a menu choice — "1", "B", "option 2"?

    Repetition is counted against the guest's *words*, and a menu token is not a
    word: it means nothing outside the prompt it answers. "1" under the AC
    question, "1" under a room card, and "1" under a second room card are three
    different statements that happen to look identical.

    `count_trailing_user_repeats` cannot tell those prompts apart, because it
    reads the flattened transcript — and `save_state` flattens every room card
    to the one string "[Showing Room Options]", so the intervening bot turns are
    not merely similar, they are byte-identical. Comparing them buys nothing.

    So the counter gives these tokens up. What it gives up is small: a menu that
    is genuinely stuck is caught on the bot's own side, by `response_fingerprint`
    / `next_repeat_count`, which compare the cards actually sent and are immune
    to the flattening. That is the guard that owns this failure.
    """
    return bool(_BARE_MENU_ANSWER.match(normalize_text_for_repetition(text)))


def count_trailing_user_repeats(current_msg: str, messages_history: list[dict] | None) -> int:
    """
    Count consecutive times the guest has sent this exact (or substantially identical)
    message, including the current turn.

    Returns:
        1 if this is the first occurrence.
        2 if the previous guest message was also identical.
        >= 3 if repeated consecutively 3+ times.

    A bare menu answer always returns 1 — see `is_bare_menu_answer`. It is the
    same token every time by construction, so counting it measures the prompt
    sequence, not the guest.
    """
    if not current_msg or not current_msg.strip():
        return 0
    if is_bare_menu_answer(current_msg):
        return 1
    count = 1
    if not messages_history:
        return count
    for msg in reversed(messages_history):
        role = msg.get("role") or msg.get("sender_type")
        if role in ("user", "guest"):
            prev_content = msg.get("content", "")
            if is_user_repetition(current_msg, prev_content):
                count += 1
            else:
                break
    return count

