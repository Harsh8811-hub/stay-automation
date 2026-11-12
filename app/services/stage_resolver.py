"""
Deterministic Booking Stage Resolver — StayInn Architecture.

CORE PRINCIPLE:
The bot's behaviour is decided by the guest's booking stage, not by the last message.
The stage resolver MUST run before the LLM.
The LLM MUST NEVER decide or override the booking stage.

Allowed Stages:
- lead (Stage 1): Inquiry / sales
- payment_check (Stage 1b): Payment under verification (reassured & pending)
- confirmed (Stage 2): Pre-arrival guidance (ZERO SELLING)
- in_house (Stage 3): In-stay concierge
- checked_out (Stage 4): Post-stay feedback & retention
- cancelled: Cancelled booking

Timezone: Property timezone (Asia/Kolkata for Sample Hotel & Sample Property).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from app.config.settings import get_settings
from app.db.mongodb import bookings_collection
from app.repositories import booking_repository

logger = logging.getLogger(__name__)

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_CHECK_IN_TIME = time(12, 0)   # 12:00 PM
DEFAULT_CHECK_OUT_TIME = time(11, 0)  # 11:00 AM


def _as_kolkata_datetime(value: Any) -> datetime | None:
    """Coerce a stored timestamp to an aware Asia/Kolkata datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KOLKATA_TZ)
    return dt.astimezone(KOLKATA_TZ)


def _stay_end(booking: dict[str, Any]) -> datetime | None:
    """When the stay actually ended: `checked_out_at` if stamped, else check-out.

    Returns None when the row carries neither — a hand-set `stage: checked_out`
    with no dates. Such a row cannot be aged, so the caller keeps it checked_out
    rather than guessing.
    """
    stamped = _as_kolkata_datetime(booking.get("checked_out_at"))
    if stamped is not None:
        return stamped
    co_date = _parse_iso_date(booking.get("check_out") or booking.get("check_out_date"))
    if co_date is not None:
        return datetime.combine(co_date, DEFAULT_CHECK_OUT_TIME, tzinfo=KOLKATA_TZ)
    return None


def _checked_out_still_speaks(booking: dict[str, Any], now_kolkata: datetime) -> bool:
    """Whether a completed stay may still claim the guest's turn as `checked_out`.

    `checked_out` is a *reply* stage — it asks for a rating and handles lost &
    found and invoices — not an ownership claim on the guest. It used to be
    permanent, so a stay that ended months ago outranked every new enquiry and
    the guest could not start a conversation again. See
    `CHECKED_OUT_FEEDBACK_WINDOW_HOURS`.
    """
    ended = _stay_end(booking)
    if ended is None:
        return True
    window = timedelta(hours=get_settings().CHECKED_OUT_FEEDBACK_WINDOW_HOURS)
    return now_kolkata - ended <= window


class BookingStage(str, Enum):
    LEAD = "lead"                    # Stage 1
    PAYMENT_CHECK = "payment_check"  # Stage 1b
    CONFIRMED = "confirmed"          # Stage 2
    IN_HOUSE = "in_house"            # Stage 3
    CHECKED_OUT = "checked_out"      # Stage 4
    CANCELLED = "cancelled"


STAGE_ALLOWED_TOOLS: dict[str, list[str]] = {
    BookingStage.LEAD.value: [
        "search_rooms",
        "show_room_options",
        "create_payment_link",
        "get_faq",
        "check_promotions",
    ],
    BookingStage.PAYMENT_CHECK.value: [
        "check_payment_status",
        "raise_owner_task",
        "get_faq",
    ],
    BookingStage.CONFIRMED.value: [
        "get_booking_details",
        "get_hotel_directions",
        "raise_owner_task",
        "get_faq",
    ],
    BookingStage.IN_HOUSE.value: [
        "get_booking_details",
        "raise_service_task",
        "quote_stay_extension",
        "get_local_guide",
        "get_hotel_amenities",
        "get_faq",
    ],
    BookingStage.CHECKED_OUT.value: [
        "submit_feedback",
        "get_review_link",
        "raise_owner_task",
        "get_faq",
        "start_new_inquiry",
    ],
    BookingStage.CANCELLED.value: [
        "get_faq",
        "start_new_inquiry",
    ],
}


@dataclass
class GuestStageContext:
    """Deterministic context computed before LLM execution."""
    stage: BookingStage
    stage_value: str
    active_booking: dict[str, Any] | None = None
    active_booking_id: str | None = None
    guest_name: str | None = None
    check_in_date: str | None = None
    check_out_date: str | None = None
    room_number: str | None = None
    amount_paid: float = 0.0
    total_amount: float = 0.0
    allowed_tools: list[str] = field(default_factory=list)
    is_separate_booking_inquiry: bool = False
    # The confirmed stay this guest still holds while a second booking is being
    # built. Only set alongside a `lead` stage, and only when the guest said yes
    # to S57's "is this a separate booking?" — see SECOND_BOOKING_ALONGSIDE_ENABLED.
    standing_booking: dict[str, Any] | None = None
    fail_safe_triggered: bool = False
    # True only when the stage came from the CRM's staff-confirmation signal rather
    # than from a booking row of our own. `understand` reads this: its ghost guard
    # demotes any "confirmed" stage that has no `booking_id`, which is every
    # CRM-confirmed guest, so without it the promotion below is undone one node later.
    crm_confirmed: bool = False
    # The CRM's own name for this guest's room. Only ever set alongside
    # `crm_confirmed` — a guest with a booking row of ours reads the room off that row.
    crm_room_type: str | None = None
    # Everything else the CRM's classifier read off the conversation for this
    # stay: guests, rooms, nights, rate plan, amount. Carried so a confirmed
    # guest who asks about the booking they already hold is answered from the
    # record rather than asked to repeat it. Never used to price or to confirm.
    crm_details: dict[str, Any] | None = None


def _parse_iso_date(val: Any) -> date | None:
    if not val:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    val_str = str(val).strip()
    try:
        return datetime.strptime(val_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def compute_booking_stage_from_record(
    booking: dict[str, Any],
    now_kolkata: datetime,
) -> tuple[BookingStage, bool]:
    """
    Computes booking stage from a booking document and current clock in Asia/Kolkata.
    Returns (BookingStage, fail_safe_triggered).
    """
    raw_status = str(booking.get("booking_status") or "").lower().strip()
    raw_stage = str(booking.get("stage") or "").lower().strip()
    b_id = str(booking.get("booking_id") or "").strip()

    if raw_stage == BookingStage.PAYMENT_CHECK.value or booking.get("payment_check_pending"):
        return BookingStage.PAYMENT_CHECK, False

    if raw_status == "cancelled" or raw_stage == BookingStage.CANCELLED.value:
        return BookingStage.CANCELLED, False

    if booking.get("checked_out_at") or raw_status == "checked_out" or raw_stage == BookingStage.CHECKED_OUT.value:
        if _checked_out_still_speaks(booking, now_kolkata):
            return BookingStage.CHECKED_OUT, False
        return BookingStage.LEAD, False

    ci_date = _parse_iso_date(booking.get("check_in") or booking.get("check_in_date"))
    co_date = _parse_iso_date(booking.get("check_out") or booking.get("check_out_date"))

    if booking.get("checked_in_at") or raw_status == "checked_in" or raw_stage == BookingStage.IN_HOUSE.value:
        if co_date:
            check_out_dt = datetime.combine(co_date, DEFAULT_CHECK_OUT_TIME, tzinfo=KOLKATA_TZ)
            if now_kolkata >= check_out_dt:
                if _checked_out_still_speaks(booking, now_kolkata):
                    return BookingStage.CHECKED_OUT, False
                return BookingStage.LEAD, False
        return BookingStage.IN_HOUSE, False

    # A row that says it is confirmed is confirmed, whatever its id looks like.
    #
    # `PAY-` is the id a booking is born with when it is opened at the payment
    # step, and until now that prefix alone decided the stage: every `PAY-` row
    # was read as a provisional hold, so a row whose `booking_status` said
    # `confirmed` and whose payment had been verified still came back as
    # PAYMENT_CHECK on every turn. Measured 2026-11-06, guest 917738662309:
    # Razorpay recorded Rs.1,110 paid, the booking row was `confirmed` with
    # `confirmed_at` set and `confirmation_source=payment_verified_system`, and
    # the bot told the guest "PENDING VERIFICATION" for three days because the
    # resolver kept returning PAYMENT_CHECK. The session's own stage is not the
    # only casualty — the guest is never told the stay is confirmed.
    #
    # The prefix still marks a draft when nothing else has been said, and the
    # status list below still catches every unpaid spelling. What changes is
    # only the case where the row itself states the outcome.
    stated_confirmed = (
        raw_status == "confirmed" or raw_stage == BookingStage.CONFIRMED.value
    )
    is_provisional_hold = (not stated_confirmed) and (
        b_id.startswith("PAY-") or raw_status in (
            "pending_payment", "payment_pending", "draft", "new", "pending", "awaiting_payment", "hold", "unpaid"
        )
    )

    if is_provisional_hold or (
        raw_status != "confirmed"
        and raw_stage != BookingStage.CONFIRMED.value
        and float(booking.get("amount_paid") or 0.0) <= 0
        and str(booking.get("payment_status") or "").lower() != "paid"
    ):
        if booking.get("payment_proof_submitted") or booking.get("claimed_paid"):
            return BookingStage.PAYMENT_CHECK, False
        return BookingStage.LEAD, False

    if not ci_date or not co_date:
        if raw_status != "confirmed" and raw_stage != BookingStage.CONFIRMED.value:
            return BookingStage.LEAD, False
        logger.warning(
            "STAGE_RESOLVER_FAILSAFE | booking_id=%s missing valid dates; failing safe to confirmed",
            booking.get("booking_id"),
        )
        return BookingStage.CONFIRMED, True

    check_in_dt = datetime.combine(ci_date, DEFAULT_CHECK_IN_TIME, tzinfo=KOLKATA_TZ)
    check_out_dt = datetime.combine(co_date, DEFAULT_CHECK_OUT_TIME, tzinfo=KOLKATA_TZ)

    if now_kolkata < check_in_dt:
        return BookingStage.CONFIRMED, False
    elif check_in_dt <= now_kolkata < check_out_dt:
        return BookingStage.IN_HOUSE, False
    else:
        if _checked_out_still_speaks(booking, now_kolkata):
            return BookingStage.CHECKED_OUT, False
        return BookingStage.LEAD, False


def _same_stay_check_out(dismissed_check_out: str | None, found_check_out: str | None) -> bool:
    """Whether a dismissed confirmation names the same stay as the one we just found.

    Compared as dates when both parse, because the CRM and the bot disagree about
    the spelling ("2027-01-18" vs "2027-01-18T00:00:00") and a string compare would
    read those as two different stays and quietly un-dismiss. Falls back to a
    trimmed string compare so a shape neither parser knows still behaves sanely —
    equal spellings still match, which is the conservative direction: it keeps the
    dismissal the guest asked for rather than silently reviving a sale.
    """
    if not dismissed_check_out:
        return False
    dismissed = _parse_iso_date(dismissed_check_out)
    found = _parse_iso_date(found_check_out)
    if dismissed is not None and found is not None:
        return dismissed == found
    return str(dismissed_check_out).strip() == str(found_check_out).strip()


async def _crm_confirmed_stage_context(
    hotel_id: int | str,
    customer_phone: str,
    now_kolkata: datetime,
    dismissed_check_out: str | None = None,
) -> GuestStageContext | None:
    """The guest's stage as the CRM's staff confirmation implies it, if it should decide.

    A stay confirmed by staff in the Control Centre never lands in the bot's own
    `bookings` collection. `resolve_guest_stage` therefore reads "no bookings" and
    demotes the guest to a lead — the bot then sells, and asks for money, from
    someone who has already paid and been confirmed.

    product rule, 2026-10-13: a confirmed stay whose check-out date has NOT passed
    keeps the guest in the confirmed stage (the FAQ/concierge path, zero selling);
    once that date has passed the confirmation is spent and the guest is treated
    normally. No check-out date we can parse means we cannot say the stay has not
    ended, which is the same answer as "passed".

    Returns `None` — "no opinion" — for every one of those cases, so the caller
    falls back to the behaviour it already had. This can only ever add a stage,
    never remove one.

    `dismissed_check_out` is the one input that lets it *remove* a stage, and it
    is the guest's own doing: an explicit "start fresh" while a confirmation
    stands writes that stay's check-out onto the session, and this function then
    reports no opinion about that same stay for as long as the session lives.
    Scoped to the stay, not to the guest — a confirmation naming a different
    check-out is a different booking and still applies.
    """
    from app.config.settings import get_settings

    if not getattr(get_settings(), "CRM_CONFIRMED_STAGE_ENABLED", False):
        return None

    from app.repositories.crm_confirmation_repository import lookup_crm_confirmation

    found = await lookup_crm_confirmation(hotel_id, customer_phone)
    if found is None or not found.confirmed:
        return None

    if _same_stay_check_out(dismissed_check_out, found.check_out):
        logger.info(
            "STAGE_CRM_CONFIRMATION_DISMISSED | hotel=%s phone=%s check_out=%s — guest "
            "asked to start fresh, so this confirmation does not set the stage",
            str(hotel_id),
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
            found.check_out,
        )
        return None

    check_out = _parse_iso_date(found.check_out)
    if check_out is not None:
        # A date we can read is a date we can compare the clock against.
        if now_kolkata.date() > check_out:
            return None
    else:
        # The CRM classified this guest as confirmed but named no dates. Measured
        # 2026-11-10: 105 of its 219 confirmed contacts are like this, and the old
        # rule dropped all of them to `lead` — so `handle_stage_confirmed` told a
        # paid guest "please tell me your check-in dates and number of guests",
        # which is the reported "asking the customer to send it again".
        #
        # With no dates there is nothing to compare against, so the question
        # becomes "did the CRM judge this recently?". `confirmed_at` is that
        # judgement's own timestamp. Deliberately bounded, because the alternative
        # to a window is treating a spent confirmation as current forever.
        window_days = int(getattr(get_settings(), "CRM_CONFIRMATION_NO_DATE_WINDOW_DAYS", 0) or 0)
        anchor = found.confirmed_at
        if window_days <= 0 or anchor is None:
            return None
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        age_days = (now_kolkata.astimezone(timezone.utc) - anchor).total_seconds() / 86400.0
        if age_days > window_days:
            logger.info(
                "STAGE_CRM_NO_DATE_EXPIRED | hotel=%s phone=%s age_days=%.1f window=%d — "
                "confirmation names no dates and is older than the window",
                str(hotel_id),
                customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                age_days,
                window_days,
            )
            return None
        logger.info(
            "STAGE_RESOLVED_FROM_CRM | hotel=%s phone=%s check_out=None age_days=%.1f — "
            "confirmed stay with no dates, inside the %d-day window",
            str(hotel_id),
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
            age_days,
            window_days,
        )
    if check_out is not None:
        logger.info(
            "STAGE_RESOLVED_FROM_CRM | hotel=%s phone=%s check_out=%s — confirmed stay, "
            "no booking row of ours",
            str(hotel_id),
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
            check_out.isoformat(),
        )
    return GuestStageContext(
        stage=BookingStage.CONFIRMED,
        stage_value=BookingStage.CONFIRMED.value,
        check_in_date=found.check_in or None,
        check_out_date=found.check_out or None,
        allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.CONFIRMED.value, []),
        crm_confirmed=True,
        crm_room_type=found.room_type or None,
        crm_details=found.details or None,
    )


def _is_past_stay_row(booking: dict[str, Any], now_kolkata: datetime) -> bool:
    """A real stay that ended outside the feedback window.

    Draft rows are excluded by the same preamble `compute_booking_stage_from_record`
    uses: a `pending_payment` row keeps its booking id on purpose (the provisional
    hold and the payment-proof flow both read it), and its check-out date being in
    the past is not the same thing as having stayed.
    """
    raw_status = str(booking.get("booking_status") or "").lower().strip()
    raw_stage = str(booking.get("stage") or "").lower().strip()
    if raw_stage == BookingStage.PAYMENT_CHECK.value or booking.get("payment_check_pending"):
        return False
    if raw_status in ("pending_payment", "payment_pending", "draft", "new", "cancelled"):
        return False
    if raw_stage == BookingStage.CANCELLED.value:
        return False
    return _stay_end(booking) is not None and not _checked_out_still_speaks(
        booking, now_kolkata
    )


#: Stays that own a guest's stage, and that a second booking has to be able to
#: stand beside. `payment_check` is deliberately absent: routing sends that guest
#: to the payment handler, so a second booking cannot be started from there, and
#: treating a hold as "standing" would strand the payment flow.
_STANDING_STAGES = (
    BookingStage.IN_HOUSE,
    BookingStage.CONFIRMED,
    BookingStage.CHECKED_OUT,
)


def _standing_booking_for_second(
    bookings: list[dict[str, Any]], now_kolkata: datetime
) -> tuple[BookingStage, dict[str, Any]] | None:
    """The stay a `separate_booking_active` guest still holds, if any.

    Ranked by date, not by list order. `get_recent_bookings` returns newest
    first, which is the wrong question here: a guest with three confirmed stays
    wants the one they are living in, then the one happening soonest — not the
    row created most recently. Order: in-house, then the confirmed stay with the
    earliest check-in, then the checked-out stay that ended most recently.

    A lookup failure is skipped rather than raised: this runs on every turn for
    every guest, and one malformed row must not cost the conversation its stage.
    """
    ranked: list[tuple[int, int, date, BookingStage, dict[str, Any]]] = []
    for booking in bookings:
        try:
            stage, _ = compute_booking_stage_from_record(booking, now_kolkata)
        except Exception:
            continue
        if stage not in _STANDING_STAGES:
            continue
        check_in = _parse_iso_date(booking.get("check_in") or booking.get("check_in_date"))
        check_out = _parse_iso_date(booking.get("check_out") or booking.get("check_out_date"))
        if stage is BookingStage.IN_HOUSE:
            ranked.append((0, 0, check_in or date.max, stage, booking))
        elif stage is BookingStage.CONFIRMED:
            ranked.append((1, 0, check_in or date.max, stage, booking))
        else:  # checked_out: the latest one first, so the date is negated
            ended = check_out or check_in or date.min
            ranked.append((2, -ended.toordinal(), date.min, stage, booking))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:3])
    _, _, _, stage, booking = ranked[0]
    return stage, booking


def second_booking_latch_expired(
    started_at: str | None, now: datetime | None = None
) -> bool:
    """Whether S57's "yes" has been standing too long to keep the stay aside.

    The latch stops `context_loader` from passing the guest's booking id, which
    is what makes the confirmed, in-house, checked-out, cancel and modify paths
    reachable — so an unbounded latch is a guest whose own paid stay the bot can
    no longer see. `SECOND_BOOKING_LATCH_HOURS` bounds it.

    A missing or unparseable timestamp counts as expired. There is no bound to
    enforce without one, and the only writer sets it in the same dict as the
    flag, so a latch without a start is a stale or hand-written one.
    """
    if not started_at:
        return True
    try:
        started = datetime.fromisoformat(str(started_at))
    except (TypeError, ValueError):
        return True
    if started.tzinfo is None:
        # Written by stage_confirmed as an aware UTC value; a naive one came from
        # somewhere else, and comparing it to an aware clock would raise.
        started = started.replace(tzinfo=KOLKATA_TZ)
    now_dt = now or datetime.now(KOLKATA_TZ)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=KOLKATA_TZ)
    try:
        hours = int(get_settings().SECOND_BOOKING_LATCH_HOURS)
    except Exception:
        hours = 24
    return (now_dt - started).total_seconds() > hours * 3600


def _as_utc(value: Any) -> datetime | None:
    """A comparison key for two timestamps that may not share a tzinfo.

    The reset is written as aware UTC; Mongo hands `created_at` back naive (it
    stores UTC, and pymongo returns it without a zone). Subtracting one from the
    other raises, and a raise here would cost the guest their turn — this runs
    from `context_loader` on every message. Anything unreadable returns None, and
    the caller treats that as "cannot be shown to postdate the reset".
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def resolve_guest_stage(
    hotel_id: int | str,
    customer_phone: str,
    *,
    now_dt: datetime | None = None,
    explicit_stage: str | None = None,
    session_stage: str | None = None,
    dismissed_crm_check_out: str | None = None,
    dismissed_booking_id: str | None = None,
    guest_reset_at: str | None = None,
    separate_booking_active: bool = False,
) -> GuestStageContext:
    """
    Resolve the guest's booking stage before LLM execution.
    Multi-tenant safe: strictly scoped by hotel_id and normalized customer phone.
    """
    now_kolkata = now_dt.astimezone(KOLKATA_TZ) if now_dt else datetime.now(KOLKATA_TZ)
    str_hid = str(hotel_id)

    if explicit_stage:
        norm_stage = explicit_stage.lower().strip()
        if norm_stage in ("payment_pending", "payment_check"):
            return GuestStageContext(
                stage=BookingStage.PAYMENT_CHECK,
                stage_value=norm_stage,
                allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.PAYMENT_CHECK.value, []),
            )
        if norm_stage == "id_verification":
            return GuestStageContext(
                stage=BookingStage.LEAD,
                stage_value="id_verification",
                allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.LEAD.value, []),
            )
        try:
            st_enum = BookingStage(norm_stage)
            return GuestStageContext(
                stage=st_enum,
                stage_value=st_enum.value,
                allowed_tools=STAGE_ALLOWED_TOOLS.get(st_enum.value, []),
            )
        except Exception:
            pass

    if not customer_phone:
        return GuestStageContext(
            stage=BookingStage.LEAD,
            stage_value=BookingStage.LEAD.value,
            allowed_tools=STAGE_ALLOWED_TOOLS[BookingStage.LEAD.value],
        )

    # This runs on EVERY turn, for EVERY guest, from context_loader — the entry
    # node. An unhandled failure here does not degrade one feature, it returns
    # no reply at all. So a lookup that fails is treated exactly like a lookup
    # that found nothing: the guest is a lead, and the turn proceeds. The
    # alternative — letting it propagate — turns a Mongo blip into an outage of
    # the whole conversation, including for guests who have no booking at all.
    try:
        bookings = await booking_repository.get_recent_bookings(
            hotel_id=str_hid,
            customer_phone=customer_phone,
            limit=5,
        )
    except Exception:
        logger.exception(
            "STAGE_LOOKUP_FAILED | hotel=%s phone=%s — degrading to lead",
            str_hid,
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
        )
        bookings = []

    # A guest who asked to start fresh has dismissed the stay they were on. Drop
    # that row here, at the single point where the rows are read, so everything
    # downstream — the confirmed/in-house branches, the payment_check preservation,
    # the CRM fallback, the candidates loop — sees the guest as they now are
    # instead of each having to remember the dismissal. If it was their only row
    # the list empties and the `if not bookings:` branch below answers.
    if dismissed_booking_id:
        kept = [
            b for b in bookings
            if str(b.get("booking_id") or "") != str(dismissed_booking_id)
        ]
        if len(kept) != len(bookings):
            logger.info(
                "STAGE_BOOKING_ROW_DISMISSED | hotel=%s phone=%s dismissed=%s rows=%d→%d — "
                "guest asked to start fresh, so this booking no longer sets the stage",
                str_hid,
                customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                dismissed_booking_id,
                len(bookings),
                len(kept),
            )
        bookings = kept

    # An explicit "/start" means the guest wants this number to read as new. The
    # id-shaped dismissal above only names the row that happened to set the stage
    # on that turn, so a phone holding two rows — or a session that had lost its
    # state — came back `confirmed` on the next message anyway. Measured
    # 2026-10-23 on the bossman's account: a row from June with no dates, no
    # rooms and a Rs 50,000 total kept reciting itself after the reset.
    #
    # A timestamp drops all of them at once, and only those that already existed:
    # a booking created after the reset has a later `created_at` and resolves
    # normally, so the reset cannot blind the bot to a genuinely new stay. A row
    # with no readable `created_at` is dropped too — it cannot be shown to
    # postdate the reset, and leaving it would defeat the reset entirely.
    if guest_reset_at:
        reset_moment = _as_utc(guest_reset_at)
        if reset_moment is not None:
            kept = [
                b for b in bookings
                if (created := _as_utc(b.get("created_at"))) is not None
                and created > reset_moment
            ]
            if len(kept) != len(bookings):
                logger.info(
                    "STAGE_GUEST_RESET | hotel=%s phone=%s reset_at=%s rows=%d→%d — "
                    "guest asked to start over, so bookings from before it no longer "
                    "name them",
                    str_hid,
                    customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                    guest_reset_at,
                    len(bookings),
                    len(kept),
                )
            bookings = kept

    # ── A second booking, with the first one still standing ──────────────────
    #
    # S57 asks a confirmed guest whether a new booking is separate, and on "yes"
    # sets `booking_stage="lead"`. That used to last exactly one turn: this
    # function re-reads the rows on every load and a confirmed row always beats a
    # lead draft, so the guest was dragged back to `confirmed` before they could
    # give a date. Measured on prod 2026-10-23 (phone 919616422222): three
    # attempts at "start fresh new booking" and every following turn was still
    # `STAGE_2_CONFIRMED_DISPATCH`.
    #
    # So while the latch is on, the standing stay stops owning the stage but is
    # not dismissed — it comes back as `standing_booking`, and the guest resolves
    # as a lead so the draft can advance. The difference from
    # `dismissed_booking_id` is the whole point: that one forgets the stay, this
    # one remembers it.
    #
    # Nothing has to clear the latch: `context_loader` stops passing it once the
    # session reaches `confirmed` again, which is the new booking completing.
    if separate_booking_active and get_settings().SECOND_BOOKING_ALONGSIDE_ENABLED:
        standing = _standing_booking_for_second(bookings, now_kolkata)
        if standing is not None:
            standing_stage, standing_row = standing
            # The draft's own payment outranks the latch. A guest who has said
            # "yes, new booking" and then paid for it has no row of its own yet —
            # the hold lives on the session, not on a booking document — so this
            # branch would answer `lead`, routing's stage short-circuit would skip
            # `stage_payment_check`, and the guest who just said "I paid" would be
            # asked for dates again. The standing stay is still carried, so the
            # first booking does not disappear from the turn.
            if session_stage == BookingStage.PAYMENT_CHECK.value:
                logger.info(
                    "STAGE_SECOND_BOOKING_PAYMENT_CHECK | hotel=%s phone=%s "
                    "standing_booking_id=%s — the new booking is awaiting payment "
                    "verification, so the latch waits",
                    str_hid,
                    customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                    standing_row.get("booking_id"),
                )
                return GuestStageContext(
                    stage=BookingStage.PAYMENT_CHECK,
                    stage_value=BookingStage.PAYMENT_CHECK.value,
                    allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.PAYMENT_CHECK.value, []),
                    guest_name=(
                        standing_row.get("guest_name")
                        or standing_row.get("customer_name")
                    ),
                    standing_booking=standing_row,
                )
            logger.info(
                "STAGE_SECOND_BOOKING_STANDING | hotel=%s phone=%s stage=%s "
                "standing_booking_id=%s — guest is booking again; the confirmed "
                "stay is carried, not dismissed",
                str_hid,
                customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                standing_stage.value,
                standing_row.get("booking_id"),
            )
            return GuestStageContext(
                stage=BookingStage.LEAD,
                stage_value=BookingStage.LEAD.value,
                allowed_tools=STAGE_ALLOWED_TOOLS[BookingStage.LEAD.value],
                guest_name=(
                    standing_row.get("guest_name")
                    or standing_row.get("customer_name")
                ),
                standing_booking=standing_row,
            )

    if not bookings:
        # If there are no bookings in DB, a guest CANNOT have a confirmed, in-house,
        # or checked-out stay. Preserving a ghost confirmed stage causes the bot to
        # send 'BK-CONFIRMED' slips to unconfirmed guests. Only pending payment checks
        # are preserved without a finalized booking record.
        if session_stage in (BookingStage.PAYMENT_CHECK.value, "payment_pending", "id_verification"):
            return GuestStageContext(
                stage=BookingStage.PAYMENT_CHECK if session_stage != "id_verification" else BookingStage.LEAD,
                stage_value=session_stage,
                allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.PAYMENT_CHECK.value if session_stage != "id_verification" else BookingStage.LEAD.value, []),
            )

        # A stay confirmed by staff in the Control Centre also has no row here, so it
        # lands in this branch too. Ask the CRM before demoting the guest to a lead —
        # but only after the payment_check case above, which must keep its own answer.
        crm_ctx = await _crm_confirmed_stage_context(
            hotel_id, customer_phone, now_kolkata, dismissed_crm_check_out
        )
        if crm_ctx is not None:
            # A staff-confirmed stay has no row to stand aside, so it is carried
            # in the same slot a row would occupy. Only its dates and room are
            # real — there is no booking id, because the CRM's confirmation does
            # not mint one of ours.
            if separate_booking_active and get_settings().SECOND_BOOKING_ALONGSIDE_ENABLED:
                logger.info(
                    "STAGE_SECOND_BOOKING_STANDING | hotel=%s phone=%s stage=%s "
                    "standing_booking_id=None — second booking beside a CRM stay",
                    str_hid,
                    customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                    crm_ctx.stage_value,
                )
                return GuestStageContext(
                    stage=BookingStage.LEAD,
                    stage_value=BookingStage.LEAD.value,
                    allowed_tools=STAGE_ALLOWED_TOOLS[BookingStage.LEAD.value],
                    guest_name=crm_ctx.guest_name,
                    standing_booking={
                        "source": "crm",
                        "check_in": crm_ctx.check_in_date,
                        "check_out": crm_ctx.check_out_date,
                        "room_type": crm_ctx.crm_room_type,
                    },
                )
            return crm_ctx

        return GuestStageContext(
            stage=BookingStage.LEAD,
            stage_value=BookingStage.LEAD.value,
            allowed_tools=STAGE_ALLOWED_TOOLS[BookingStage.LEAD.value],
        )

    candidates: list[tuple[int, BookingStage, dict[str, Any], bool]] = []

    for b in bookings:
        st, fail_safe = compute_booking_stage_from_record(b, now_kolkata)
        priority = 99
        if st == BookingStage.IN_HOUSE:
            priority = 1
        elif st == BookingStage.CONFIRMED:
            priority = 2
        elif st == BookingStage.PAYMENT_CHECK:
            priority = 3
        elif st == BookingStage.CHECKED_OUT:
            priority = 4
        elif st == BookingStage.LEAD:
            priority = 5
        elif st == BookingStage.CANCELLED:
            priority = 6

        candidates.append((priority, st, b, fail_safe))

    candidates.sort(key=lambda x: x[0])
    chosen_priority, chosen_stage, chosen_booking, fail_safe_used = candidates[0]

    single_cancelled = chosen_stage == BookingStage.CANCELLED and len(candidates) == 1
    if single_cancelled:
        chosen_stage = BookingStage.LEAD

    if chosen_stage == BookingStage.LEAD and session_stage == BookingStage.PAYMENT_CHECK.value:
        chosen_stage = BookingStage.PAYMENT_CHECK

    # A stay that ended outside the feedback window resolves to `lead` (above).
    # It must not leave its booking identity behind: `context_loader` copies
    # `active_booking_id` into the state for every stage, and
    # `helpers.guest_is_confirmed` treats any real `BK-…` id as "this guest is
    # already booked" — so the confirmed-guest guard would answer a returning
    # guest who asks for rooms with "your booking is already confirmed" instead
    # of selling. The name is kept, because greeting a returning guest by name
    # is the whole point of knowing them.
    expired_stay = (
        chosen_stage is BookingStage.LEAD and _is_past_stay_row(chosen_booking, now_kolkata)
    )
    if expired_stay:
        logger.info(
            "STAGE_PAST_STAY_RETIRED | hotel=%s phone=%s booking_id=%s — ended outside "
            "the %sh feedback window, so it no longer names the guest",
            str_hid,
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
            chosen_booking.get("booking_id"),
            get_settings().CHECKED_OUT_FEEDBACK_WINDOW_HOURS,
        )

    # A cancelled stay resolves to `lead` (above) but used to keep its id, and
    # `helpers.has_real_booking_id` is a shape test with no status reading — so
    # `confirmed_guard.guest_should_not_be_sold_to` answered a guest whose only
    # booking was cancelled with "your booking is already confirmed". Measured
    # 2026-10-28: stage resolved `lead`, then the turn ended `confirmed`, and a
    # brand-new enquiry printed the cancelled dates as if the stay were live.
    # Retrieving the id is the same operation as retiring an expired stay, so it
    # is the same predicate: a stay the guest no longer holds names nothing.
    cancelled_stay = chosen_stage is BookingStage.LEAD and single_cancelled
    if cancelled_stay:
        logger.info(
            "STAGE_CANCELLED_STAY_RETIRED | hotel=%s phone=%s booking_id=%s — "
            "cancelled, so it no longer names the guest",
            str_hid,
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
            chosen_booking.get("booking_id"),
        )
    retired_stay = expired_stay or cancelled_stay

    b_id = None if retired_stay else chosen_booking.get("booking_id")
    # A provisional hold (`PAY-…`) is a row, not a booking — see
    # BOOKING_ID_ON_PAYMENT_PROOF_ENABLED. Handing its reference to the graph as
    # `booking_id` would print it as a booking number and make the real id look
    # already minted. The equality is what distinguishes the two states: a hold
    # is renamed in place, so only an *unrenamed* row still has
    # booking_id == payment_reference.
    if b_id and (b_id == chosen_booking.get("payment_reference") or str(b_id).startswith("PAY-")):
        b_id = None
    g_name = chosen_booking.get("guest_name") or chosen_booking.get("customer_name")
    if retired_stay:
        ci = co = room_num = None
        paid = total = 0.0
    else:
        ci = chosen_booking.get("check_in") or chosen_booking.get("check_in_date")
        co = chosen_booking.get("check_out") or chosen_booking.get("check_out_date")
        room_num = chosen_booking.get("room_number")
        paid = float(chosen_booking.get("amount_paid") or 0.0)
        total = float(chosen_booking.get("total_price") or chosen_booking.get("amount") or 0.0)

    logger.info(
        "STAGE_RESOLVED | hotel=%s phone=%s stage=%s booking_id=%s fail_safe=%s",
        str_hid,
        customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
        chosen_stage.value,
        b_id,
        fail_safe_used,
    )

    return GuestStageContext(
        stage=chosen_stage,
        stage_value=chosen_stage.value,
        active_booking=None if retired_stay else chosen_booking,
        active_booking_id=b_id,
        guest_name=g_name,
        check_in_date=str(ci) if ci else None,
        check_out_date=str(co) if co else None,
        room_number=room_num,
        amount_paid=paid,
        total_amount=total,
        allowed_tools=STAGE_ALLOWED_TOOLS.get(chosen_stage.value, []),
        fail_safe_triggered=fail_safe_used,
    )


async def set_booking_stage_for_test(
    booking_id: str,
    stage: str,
    *,
    hotel_id: int | str = "111111",
    customer_phone: str = "919999999999",
    guest_name: str = "Test Guest",
    check_in: str | None = None,
    check_out: str | None = None,
    room_name: str = "Deluxe Room King Bed",
    room_number: str | None = None,
    amount_paid: float = 3000.0,
    total_amount: float = 3000.0,
    payment_proof_submitted: bool = False,
    claimed_paid: bool = False,
) -> dict[str, Any]:
    """
    Test hook: Directly upserts a booking record into MongoDB with specified stage and dates.
    Allows automated acceptance tests to deterministically verify S51–S64.
    """
    coll = bookings_collection()
    str_hid = str(hotel_id)
    now = datetime.now(KOLKATA_TZ)

    status_map = {
        BookingStage.LEAD.value: "pending_payment",
        BookingStage.PAYMENT_CHECK.value: "pending_payment",
        BookingStage.CONFIRMED.value: "confirmed",
        BookingStage.IN_HOUSE.value: "confirmed",
        BookingStage.CHECKED_OUT.value: "confirmed",
        BookingStage.CANCELLED.value: "cancelled",
    }
    booking_status = status_map.get(stage, "confirmed")

    doc = {
        "booking_id": booking_id,
        "hotel_id": str_hid,
        "customer_phone": customer_phone,
        "customer_identifier": customer_phone,
        "guest_name": guest_name,
        "stage": stage,
        "booking_status": booking_status,
        "check_in": check_in,
        "check_out": check_out,
        "check_in_date": check_in,
        "check_out_date": check_out,
        "room_name": room_name,
        "room_type": room_name,
        "room_number": room_number,
        "amount_paid": amount_paid,
        "total_price": total_amount,
        "payment_status": "paid" if amount_paid >= total_amount else "partial",
        "payment_proof_submitted": payment_proof_submitted or (stage == BookingStage.PAYMENT_CHECK.value),
        "claimed_paid": claimed_paid or (stage == BookingStage.PAYMENT_CHECK.value),
        "created_at": now,
        "updated_at": now,
    }

    if stage == BookingStage.IN_HOUSE.value:
        doc["checked_in_at"] = now
    elif stage == BookingStage.CHECKED_OUT.value:
        doc["checked_in_at"] = now
        doc["checked_out_at"] = now

    # This hook upserts straight into `bookings`, bypassing `create_booking` —
    # so it has to carry the dashboard aliases itself, or a row it plants shows
    # up on the panel with zero guests and no status
    # (see `app/repositories/booking_panel_fields.py`).
    from app.repositories.booking_panel_fields import apply_aliases

    await coll.replace_one({"booking_id": booking_id}, apply_aliases(doc), upsert=True)
    logger.info("TEST_HOOK_BOOKING_UPSERTED | id=%s stage=%s hotel=%s", booking_id, stage, str_hid)
    return doc
