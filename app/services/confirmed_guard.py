"""
The last line of defence before a confirmed guest is sold to again.

Why this exists
---------------
`CRM_CONFIRMED_STAGE_ENABLED` promotes a guest whose stay staff confirmed into
the `confirmed` stage, and `route_by_intent` sends every turn in that stage to
`stage_confirmed`, which is specified as "ZERO SELLING". That chain is correct
and tested — but every link in it trusts one thing: that this turn's stage
resolution *succeeded*.

It can fail to. Three ways, all seen on the prod clone:

  * the CRM lookup returns nothing because it timed out — `lookup_crm_confirmation`
    is documented to fail closed, so the guest demotes to lead;
  * the guest talks to a revision older than the CRM fix;
  * the session's own `booking_stage` was written before the guest was confirmed
    and the resolver has no row of its own to override it with.

When that happens the turn falls through to the ordinary selling branches and
the room card goes out. Measured 2026-10-21 against the prod clone: of the 196
CRM contacts classified `CONFIRMED_BOOKING`, 10 received a room card *after* the
CRM's own `classified_at` timestamp, and five of those were after the CRM fix
deployed. Guest 919433543471 wrote "Kindly confirm our booking from 21-23 Oct
2026" and got the full ROOMS card twice, while the CRM held exactly that stay at
0.95 confidence.

So this module answers one question, at the three points a card or a date-ask is
built, from state that is already in hand:

    Do we already know this guest is confirmed?

If yes, the selling branch is replaced with a short concierge reply. Nothing
else about the turn changes — this can only ever REMOVE selling, never add a
stage, never confirm a booking, never release a room.

Why `guest_is_confirmed` and not a fifth predicate
--------------------------------------------------
`app/utils/helpers.guest_is_confirmed` is the single place four ghost guards
already agree on what "confirmed" means: a real `BK-…` id of ours, or the CRM's
staff signal (`crm_confirmed_stay`, recomputed every turn by `context_loader`).
Re-deriving the answer here is how five guards come to disagree, so this module
does not.

The one addition is a payment this session already verified (`payment_status ==
"paid"` with a non-zero `amount_paid`). That is the exact shape of the original
complaint — "the bot already had the payment and still acted as a booking agent"
— and it covers the window where the payment is verified but the booking row has
not been written yet. It is not a claim the guest made: `understand` sets both
fields only after the screenshot's own status and amount were checked.
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.prompts.templates import text_card
from app.utils.helpers import guest_is_confirmed

#: Language tags the reply below has a Hinglish variant for, matching the rest of
#: the stage handlers.
_HINDI_TAGS = {"hinglish", "hindi", "hi", "hin", "en-in-hi"}


def guest_should_not_be_sold_to(state: dict) -> bool:
    """Whether we already know this guest's stay, so this turn must not sell.

    `False` whenever the flag is off, which is the byte-identical old behaviour.

    **A checked-out guest is not "already booked".** Their stay ended; booking
    the next one is exactly what they are here to do. The resolver keeps their
    id during the feedback window so the bot can say "how was your stay?", and
    `guest_is_confirmed` reads any real id as a live booking — so once routing
    started sending a date-bearing enquiry on to the booking flow (see
    `routing.route_by_intent`), this guard answered a returning guest with
    "your booking is already confirmed, so I won't start a new room search".
    Measured 2026-10-28 on `checked-out-enquiry-dismisses`.

    Only `checked_out` is excused. `lead` is deliberately not: the whole point
    of this guard is the turn whose stage resolution says lead while we know
    the guest holds a stay.
    """
    if not get_settings().CONFIRMED_GUEST_NEVER_SOLD_ENABLED:
        return False
    stage = str(state.get("lifecycle_stage") or state.get("booking_stage") or "").strip().lower()
    if stage == "checked_out":
        return False
    # A guest who asked for a *second* booking beside the one they hold is
    # being sold to on purpose, and this guard is what stopped that draft from
    # being filled even after `understand` and the stage lock had both let it
    # through. Measured 2026-11-05 (`week-sahu-confirmed-guest-new-booking`):
    # the guest answered "is that a separate booking?" with "6 rooms needed,
    # 28 november to 5 decemeber", the draft opened at `lead`, and this guard
    # sent the turn to `stage_confirmed` anyway, where the dates became a
    # general question and were never quoted.
    #
    # The same narrow field the stage lock releases on — written only by the
    # two places that open that draft, and expired by `context_loader`.
    if state.get("separate_booking_active"):
        return False
    if guest_is_confirmed(state):
        return True
    return (
        str(state.get("payment_status") or "").strip().lower() == "paid"
        and float(state.get("amount_paid") or 0.0) > 0.0
    )


def confirmed_guest_reply(state: dict) -> list[dict]:
    """The reply a confirmed guest gets instead of a room card or a date-ask.

    Names the booking only when we hold a real id of ours — a CRM-confirmed guest
    has none, and printing "(None)" at them was a live defect once already (see
    `stage_confirmed`'s comment on `booking_ref`).
    """
    booking_id = str(state.get("booking_id") or state.get("active_booking_id") or "").strip()
    ref_en = f" **{booking_id}**" if booking_id else ""
    ref_hi = f" **{booking_id}**" if booking_id else ""

    lang = str(state.get("preferred_language") or "English").strip().lower()
    if lang in _HINDI_TAGS:
        text = (
            f"🙏 Aapki booking{ref_hi} already confirmed hai, isliye main naya room "
            f"search nahi karunga.\n\n"
            f"Agar aap apni stay ke liye kuch chahte hain — extra room, dates change, "
            f"ya airport pickup — bas bata dijiye, humari front desk team aapki madad karegi."
        )
    else:
        text = (
            f"🙏 Your booking{ref_en} is already confirmed with us, so I won't start a "
            f"new room search.\n\n"
            f"If you need anything for your stay — an extra room, a date change, or "
            f"airport pickup — just tell me and our front desk will help."
        )
    return text_card(text)
