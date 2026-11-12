"""The one place that decides whether the second-booking latch is on.

S57 asks a confirmed guest whether a new booking is separate, and on "yes" the
guest may build a second booking while the first still stands. That is a
behaviour change in stage resolution, so it sits behind two gates and both must
be on:

  1. ``SECOND_BOOKING_ALONGSIDE_ENABLED`` — the deployment kill switch. It is an
     env flag, and per the repo's history it cannot be flipped from this account
     (Cloud Run env changes need ``iam.serviceaccounts.actAs``), so it is the
     coarse switch, not the rollback.

  2. ``business_config.features.second_booking_alongside_enabled`` — the hotel's
     own flag, parsed from its Hotel-Settings document (in Mongo that is
     ``behavior.features``; ``business_config.features`` is the parsed path).
     This is the OPERABLE off-switch: the DB team can set it false for one hotel
     and the latch stops within the 5-minute config cache, with no deploy.

Both the reader (``context_loader``, which decides whether to pass the latch to
the resolver) and the writer (``stage_confirmed``, which sets it) ask this
function, so a gate-off deployment cannot leave a latch behind that a later
gate-on deployment inherits.
"""

from __future__ import annotations

from app.config.settings import get_settings


def second_booking_gate_enabled(state: dict, settings=None) -> bool:
    """True when both the env flag and the hotel's own flag are on.

    Missing hotel config counts as on: the hotel flag defaults to True, and a
    document that does not mention it should behave like one that sets it true.
    """
    settings = settings or get_settings()
    if not getattr(settings, "SECOND_BOOKING_ALONGSIDE_ENABLED", False):
        return False
    features = (
        ((state.get("hotel_config") or {}).get("business_config") or {}).get("features")
        or {}
    )
    if not isinstance(features, dict):
        return True
    return bool(features.get("second_booking_alongside_enabled", True))
