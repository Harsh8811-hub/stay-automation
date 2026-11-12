"""
ConversationState ↔ BookingSession mapping, generated from the contract.

Both directions are driven by ``app/models/session_contract.py``:

  ``PERSISTED``  — every BookingSession field that is not REPO_MANAGED.
                   Copied out of state on save and back into state on load,
                   by iteration, so a new model field round-trips without
                   anyone remembering to add a line here.
  ``SPECIAL``    — the handful whose mapping is not a plain copy. Each one has
                   a handler in a direction-specific table below, and a test
                   asserts the tables cover the declared set.
  ``EPHEMERAL``  — declared in the graph, deliberately not persisted. The load
                   path does not carry these, and the save path ignores them.

Why this replaced two hand-written lists
----------------------------------------
``state_to_session`` used to name 86 fields by hand and
``session_fields_for_state`` named 82 the other way. They had already drifted
apart, and ``session_fields_for_state`` had no callers at all — the real load
path was a third dict in ``context_loader``. Because ``upsert_session``
replaces the whole ``workflow_state`` document and ``state_to_session`` builds
a fresh session each turn, a field missing from the save list was not left
alone: it was reset to its default, every turn, silently.

Those two hand-written lists are gone. Everything below is derived.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from app.models.schemas import (
    BookingSession,
    BookingStage,
    MessageRecord,
    PaymentDetails,
    PriceBreakdown,
    SelectedRoom,
)
from app.models.conversation_overrides import ConversationOverrides
from app.models.session_contract import (
    DECLARED,
    PERSISTED,
    REPO_MANAGED,
    SAVE_ONLY,
    SPECIAL,
)
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# BookingSession declares these without a default, so the save path must always
# supply something. Empty values match the "fresh session" construction that
# chat_runner uses for a brand new conversation.
_REQUIRED_DEFAULTS: dict[str, Any] = {"session_id": "", "hotel_id": 0}

# State holds dicts; the model holds these objects. Handled in a second pass
# rather than as kwargs.
_NESTED_FIELDS: frozenset[str] = frozenset({
    "selected_rooms",
    "price_breakdown",
    "payment_details",
    "conversation_overrides",
})

_UNSET = object()


# ── Save direction: state → BookingSession ───────────────────────────────────


def _coerce_stage(raw: Any) -> BookingStage:
    """A stage string from state as a BookingStage, with a substring fallback."""
    if isinstance(raw, BookingStage):
        return raw
    try:
        return BookingStage(raw or "new")
    except (ValueError, KeyError):
        stage_str = str(raw or "").lower()
        if "collecting" in stage_str or "custom" in stage_str:
            return BookingStage.COLLECTING
        if "room" in stage_str:
            return BookingStage.ROOMS_SHOWN
        if "quote" in stage_str:
            return BookingStage.QUOTE_SHOWN
        if "payment" in stage_str:
            return BookingStage.AWAITING_PAYMENT
        if "confirm" in stage_str:
            return BookingStage.CONFIRMED
        return BookingStage.NEW


def _save_special_requests(value: Any) -> Any:
    """The model stores one string; state may hold a list."""
    return ", ".join(str(x) for x in value) if isinstance(value, list) else value


def _save_last_prompt_action(state_value: Any, state: Mapping[str, Any]) -> Any:
    """The prompt pointer, from this turn.

    `load_context` resets `next_action` to None at the start of every turn, so
    `next_action` is never what a later turn reads back — this field is. That is
    what makes it the pointer a reader has to guard on, and it has to carry
    *this* turn's prompt.

    The order was backwards, and it cost the change menu. `state_value` is the
    value hydrated from the previous turn, so after the first prompt it is
    truthy on every turn, and the old `state_value or state.get("next_action")`
    kept the previous prompt while dropping the new one. Showing the change menu
    left the pointer on "rooms_shown" while its reader looked for
    "change_menu_shown", so the menu was a dead end: every row tap and every
    typed word on it fell through to the model and re-quoted the room card.
    """
    return state.get("next_action") or state_value


_SAVE_SPECIAL: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {
    "booking_stage": lambda v, s: _coerce_stage(v),
    "special_requests": lambda v, s: _save_special_requests(v),
    "room_option_offset": lambda v, s: int(v or 0),
    "repeat_response_count": lambda v, s: int(v or 0),
    "last_prompt_action": _save_last_prompt_action,
}


def state_to_session(
    state: ConversationState | dict[str, Any],
    *,
    include_messages: bool = False,
    **overrides: Any,
) -> BookingSession:
    """
    Build a BookingSession from ConversationState.

    Pass keyword overrides to force specific fields (e.g. payment_status when
    creating a pending hold). Overrides win over state. Nested objects
    (selected_rooms, price_breakdown, payment_details, conversation_overrides)
    are rebuilt from dicts when present.
    """
    kwargs: dict[str, Any] = {}

    # Required-without-default fields first, so the model can always construct.
    for name in _REQUIRED_DEFAULTS:
        if name in overrides:
            kwargs[name] = overrides.pop(name)
        elif name in state:
            kwargs[name] = state[name]
        else:
            kwargs[name] = _REQUIRED_DEFAULTS[name]

    # Nested models are built after construction, not passed as kwargs.
    nested: dict[str, Any] = {}
    for name in _NESTED_FIELDS:
        value = overrides.pop(name, _UNSET)
        if value is not _UNSET:
            nested[name] = value

    # Everything the model declares and the repository does not own, plus the
    # SAVE_ONLY columns (written to the document, deliberately not hydrated).
    for name in PERSISTED | frozenset(SAVE_ONLY):
        # `_NESTED_FIELDS` are built by the pass below, which reads the override
        # first and falls back to `state` — copying them here too would duplicate
        # them, and for the one list-shaped field (`selected_rooms`) that
        # duplication compounds on every save.
        if name in kwargs or name in nested or name in _NESTED_FIELDS:
            continue
        if name in _SAVE_SPECIAL:
            continue
        if name in overrides:
            kwargs[name] = overrides.pop(name)
        elif name in state:
            kwargs[name] = state[name]
        # otherwise leave it out and let the model's own default apply

    # Special-cased scalars, using the same override-wins-precedence.
    for name, handler in _SAVE_SPECIAL.items():
        if name in overrides:
            kwargs[name] = overrides.pop(name)
        elif name in state:
            kwargs[name] = handler(state[name], state)
        # absent from both: an absent booking_stage still needs a value, because
        # the model has no default for it.
        elif name == "booking_stage":
            kwargs[name] = handler(None, state)

    session = BookingSession(**kwargs, **overrides)

    # ── Nested models, from either the override or state ──
    def _source(name: str) -> Any:
        if name in nested:
            return nested[name]
        return state.get(name)

    rooms_src = _source("selected_rooms")
    for r in rooms_src or []:
        if isinstance(r, SelectedRoom):
            session.selected_rooms.append(r)
        elif isinstance(r, dict):
            session.selected_rooms.append(SelectedRoom(**r))

    pb = _source("price_breakdown")
    if isinstance(pb, PriceBreakdown):
        session.price_breakdown = pb
    elif isinstance(pb, dict):
        session.price_breakdown = PriceBreakdown(**pb)

    pd = _source("payment_details")
    if isinstance(pd, PaymentDetails):
        session.payment_details = pd
    elif isinstance(pd, dict):
        session.payment_details = PaymentDetails(**pd)

    cov_raw = _source("conversation_overrides")
    if isinstance(cov_raw, ConversationOverrides):
        session.conversation_overrides = cov_raw
    elif isinstance(cov_raw, dict):
        session.conversation_overrides = ConversationOverrides(**cov_raw)

    # Message history (optional — save_state appends the current turn separately)
    if include_messages:
        for msg in state.get("messages_history", []) or []:
            if isinstance(msg, MessageRecord):
                session.messages.append(msg)
            elif isinstance(msg, dict):
                session.messages.append(MessageRecord(**msg))

    return session


# ── Load direction: BookingSession → state ───────────────────────────────────


def _load_conversation_overrides(session: BookingSession) -> Any:
    cov = getattr(session, "conversation_overrides", None)
    return cov.model_dump(mode="json") if cov else None


_LOAD_SPECIAL: dict[str, Callable[[BookingSession], Any]] = {
    "booking_stage": lambda s: s.booking_stage.value,
    "selected_rooms": lambda s: [r.model_dump() for r in (s.selected_rooms or [])],
    "price_breakdown": lambda s: s.price_breakdown.model_dump() if s.price_breakdown else None,
    "payment_details": lambda s: (
        s.payment_details.model_dump(exclude_none=True) if s.payment_details else None
    ),
    "conversation_overrides": _load_conversation_overrides,
}


def session_to_state(
    session: BookingSession,
    incoming: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Hydrate a graph state dict from a loaded BookingSession.

    ``incoming`` is the state the turn arrived with; a session value wins when
    it is present (not None), otherwise the incoming value is kept. This is the
    counterpart of state_to_session and is driven by the same PERSISTED set, so
    the two cannot drift.

    SAVE_ONLY columns are deliberately not hydrated — see session_contract.

    Note the deliberate difference from the dict this replaced: that one used
    ``value or fallback`` for about a dozen fields, so a legitimate ``0``,
    ``False`` or ``[]`` on the session fell through to the incoming state.
    Here only ``None`` falls through — a stored ``0`` child-age count stays
    ``0``.
    """
    incoming = incoming or {}
    loaded: dict[str, Any] = {}
    for name in PERSISTED:
        if name in _LOAD_SPECIAL:
            continue
        value = getattr(session, name, None)
        loaded[name] = incoming.get(name) if value is None else value

    for name, handler in _LOAD_SPECIAL.items():
        loaded[name] = handler(session)

    return loaded


def contract_coverage() -> dict[str, Any]:
    """
    Names the two special tables handle, for the contract test.

    ``SPECIAL`` describes what needs special handling; these tables are what
    actually implements it. The test asserts they agree, so a name cannot be
    listed as special without a handler in at least one direction.
    """
    return {
        "declared": DECLARED,
        "persisted": PERSISTED,
        "repo_managed": REPO_MANAGED,
        "save_special": frozenset(_SAVE_SPECIAL),
        "load_special": frozenset(_LOAD_SPECIAL),
        "declared_special": frozenset(SPECIAL),
    }
