"""Hold a settled guest's stage for the turn.

Why this exists
---------------
`stage_resolver` decides the booking stage once, at the top of every turn, from
the booking record. Its own docstring says the LLM must never decide or override
the stage. Nothing enforced that.

Between the resolver and the router sit `context_loader` → `understand` →
`feature_gate`. `understand` is a 3,400-line node with many exits, three of
which rebuild the stage from scratch:

* the fresh-start path writes `booking_stage: "new"` / `lifecycle_stage: "lead"`
* two "ghost booking" guards write `lead` whenever `guest_is_confirmed()` says
  no — and that predicate is False when the guest's id is a placeholder or when
  a CRM-confirmed stay has no id of ours at all

Each of those is a `return {...}` that never reaches the lifecycle invariant at
the foot of the function, which is the only place that used to hold the stage.
So a guest who had paid and been confirmed could be put back to `lead`
mid-conversation and sold a second booking. That is conversation 919739843511
on 2026-10-26, and it is what the product's "Confirm so confirm" is about.

The rule
--------
Only one direction is dangerous: moving a *settled* guest backwards, because
that turns a confirmed stay into a sales lead. Moving forward (lead → confirmed)
is what the resolver is for and is never touched here.

So this module answers one question — "did a later node try to move a settled
guest back to a selling stage?" — and if so, puts the resolver's answer back.
Nothing else. It cannot advance a stage, cannot cancel one, and cannot invent a
stage the resolver did not already return this turn.

Where the settled stage comes from
----------------------------------
`context_loader` copies the resolver's verdict into `resolved_booking_stage`
before `understand` runs. That is the whole point: it is a record of what the
database said, captured before anything could rewrite it. Without it the lock
would be reading a field that the node it is defending against has already
changed.

Releases
--------
A demotion is only refused when it was *not* asked for. Two things release it:

* `guest_reset_at` — the `/start` reset, which makes the resolver ignore every
  booking created before that timestamp. The resolver returns `lead` on its own
  in that case, so the lock never sees a settled stage to defend.
* `abandoned_draft` — the guest explicitly declined to continue a draft.
* `separate_booking_active` — the guest asked for a second booking beside the
  one they hold. Its draft lives at `lead`, so holding this one back would make
  the draft impossible to fill.

All three are checked from the state being locked, so a node that sets one
still gets its way.
"""

from __future__ import annotations

import logging

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

#: Every stage name the graph actually routes on, with how far along it is. This
#: is deliberately wider than `stage_resolver.BookingStage`: `booking_stage`
#: carries draft spellings too (`new`, `collecting`, `quote_shown`, …), and a
#: demotion into one of those is exactly what the reported bug did. The number
#: is only used to decide "is this lower", never to route.
STAGE_RANK: dict[str, int] = {
    # pre-sale drafts
    "lead": 0,
    "new": 0,
    "collecting": 0,
    "rooms_shown": 1,
    "quote_shown": 1,
    "awaiting_payment": 1,
    "payment_pending": 1,
    "payment_check": 1,
    "id_verification": 1,
    # the stay is sold
    "confirmed": 2,
    "in_house": 3,
    "checked_out": 4,
}

#: Stages that are an ending, not a demotion. `cancelled` must never be held
#: back — cancelling is the guest (or staff) ending the stay, and the lock
#: refusing it would strand them in a stage for a booking that no longer exists.
TERMINAL_STAGES: frozenset[str] = frozenset({"cancelled"})

#: Stages the lock defends. `payment_check` is deliberately excluded: a guest
#: who has sent proof of payment has no booking of ours to protect yet, and
#: `stage_payment_check`'s own Mongo guard is what stops *it* demoting a settled
#: stay. Adding it here would trap guest turns that the flow legitimately moves
#: between lead and payment_check.
SETTLED_STAGES: frozenset[str] = frozenset({"confirmed", "in_house", "checked_out"})

#: The field `context_loader` writes with the resolver's verdict.
RESOLVED_STAGE_FIELD = "resolved_booking_stage"


def resolved_stage_of(state: dict) -> str | None:
    """The stage the database said this turn, before any node could rewrite it."""
    value = state.get(RESOLVED_STAGE_FIELD)
    value = str(value).strip().lower() if value is not None else ""
    return value or None


def _declared_stage(state: dict) -> str | None:
    """What the turn is currently about to act on: `lifecycle_stage`, else `booking_stage`."""
    for field in ("lifecycle_stage", "booking_stage"):
        value = state.get(field)
        value = str(value).strip().lower() if value is not None else ""
        if value:
            return value
    return None


def _demotion_was_asked_for(state: dict) -> bool:
    """Whether the guest themselves ended the stay this turn."""
    if state.get("guest_reset_at") or state.get("abandoned_draft"):
        return True
    # A guest opening a *second* booking beside the one they hold is not a
    # demotion the lock should refuse: the draft sits at `lead` for as long as
    # it lives, and the latch below is what tells the resolver to keep the
    # confirmed stay aside. Measured 2026-11-05
    # (`week-sahu-confirmed-guest-new-booking`): the guest answered "is this a
    # separate booking?" with "6 rooms needed, 28 november to 5 decemeber";
    # `understand` opened the draft and returned `lead`, and this lock put
    # `confirmed` straight back, so the turn went to the confirmed handler and
    # the dates were answered as a general question.
    #
    # Narrow on purpose: `separate_booking_active` is only ever written by the
    # two places that open that draft (the yes branch in `stage_confirmed` and
    # the details-answer branch in `confirmed_booking`), and never without the
    # guest having asked for it.
    return bool(state.get("separate_booking_active"))


def lock_reason(state: dict) -> str | None:
    """Why the stage needs holding, or None when it does not.

    Split out from `apply_stage_lock` so a caller can log the decision without
    having to re-derive it, and so the tests assert on the reason rather than a
    boolean that is true for several unrelated situations.
    """
    if not get_settings().STAGE_LOCK_ENFORCED:
        return None

    resolved = resolved_stage_of(state)
    if resolved not in SETTLED_STAGES:
        return None

    if _demotion_was_asked_for(state):
        return None

    declared = _declared_stage(state)
    if declared is None:
        return None
    if declared in TERMINAL_STAGES:
        return None

    # A stage this module has never heard of is not something to defend against:
    # `stage_resolver` owns the vocabulary, and guessing at an unknown value
    # would be the same class of bug this exists to stop.
    if declared not in STAGE_RANK:
        return None

    # Only a move *backwards* is a demotion. Anything at or above the resolved
    # stage is either the resolver's own answer or a node moving the guest
    # forward, which is not this module's business.
    if STAGE_RANK[declared] >= STAGE_RANK[resolved]:
        return None

    return f"{resolved} -> {declared}"


def apply_stage_lock(state: dict) -> dict:
    """Return `state` with a settled stage restored if a node moved it backwards.

    Returns the same object when there is nothing to do, so callers can compare
    identity if they want to know whether a repair happened (they should use
    `lock_reason` for that instead — it says why).
    """
    reason = lock_reason(state)
    if reason is None:
        return state

    resolved = resolved_stage_of(state)
    logger.warning(
        "STAGE_LOCK_HELD | session=%s phone=%s resolved=%s moved_to=%s",
        state.get("session_id"),
        state.get("customer_identifier"),
        resolved,
        _declared_stage(state),
    )
    return {
        **state,
        "lifecycle_stage": resolved,
        "booking_stage": resolved,
    }
