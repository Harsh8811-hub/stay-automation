"""
Utility helpers — date parsing, ID generation, misc.
"""

import re
import uuid
from datetime import datetime, timedelta, timezone


def generate_session_id(
    hotel_id: int | str | None = None,
    phone: str | None = None,
) -> str:
    """
    Tenant-scoped session id: sc:{hotel}:{phone_tail}:{random}.

    Tenant first, so a session can never be read as another hotel's — including
    in logs, which is why only the last 4 digits of the phone are used.
    """
    tenant = str(hotel_id) if hotel_id not in (None, "") else "0"
    digits = "".join(ch for ch in str(phone or "") if ch.isdigit())
    tail = digits[-4:] if len(digits) >= 4 else "anon"
    return f"sc:{tenant}:{tail}:{uuid.uuid4().hex[:8]}"


def today_str() -> str:
    """Return today's date as YYYY-MM-DD in IST."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d")


# Placeholder values that have been written into `booking_id` over the years and
# mean "no booking", not "a booking called BK-CONFIRMED".
_PLACEHOLDER_BOOKING_IDS = ("BK-CONFIRMED", "None", "")


def has_real_booking_id(state: dict, booking_id: object = None) -> bool:
    """Whether the guest's stay is backed by a booking row of our own.

    `booking_id` lets a caller pass an id it resolved from somewhere else (a nested
    `active_booking` document, say) instead of having this re-derive one.
    """
    raw = booking_id if booking_id is not None else (
        state.get("booking_id") or state.get("active_booking_id")
    )
    raw_str = str(raw or "").strip()
    if not raw_str or raw_str in _PLACEHOLDER_BOOKING_IDS or raw_str.startswith("PAY-"):
        return False
    return True


#: Row statuses that mean the stay is settled, whatever the id looks like.
_CONFIRMED_ROW_STATUSES = {"confirmed", "checked_in", "in_house", "checked_out"}


def row_states_confirmed(state: dict) -> bool:
    """Whether the booking row the session is holding says the stay is settled.

    `PAY-` is the id a booking is born with at the payment step, so it says how
    the row *started*, not where it ended. A `PAY-` row can be confirmed and
    paid; reading its prefix as "still a hold" is what told guest 917738662309
    (2026-11-06) that a paid, confirmed stay was pending payment, three days
    after Razorpay had recorded the transfer.

    Only the row's own status is read here, never the id's shape.
    """
    row = state.get("active_booking")
    if not isinstance(row, dict):
        return False
    status = str(row.get("booking_status") or row.get("status") or "").lower().strip()
    stage = str(row.get("stage") or "").lower().strip()
    return status in _CONFIRMED_ROW_STATUSES or stage in _CONFIRMED_ROW_STATUSES


def guest_is_confirmed(state: dict, booking_id: object = None) -> bool:
    """Whether the guest's stay is confirmed — by our row, or by the CRM's staff signal.

    Four separate ghost guards in the graph demote any `confirmed` stage that has no
    booking id of ours. A stay confirmed by staff in the Control Centre has none, so
    each of them would throw the guest back to a lead and start selling to someone who
    has already paid. `crm_confirmed_stay` is set by the stage resolver for exactly that
    guest; this predicate is the one place that decides, so a fifth guard cannot drift.
    """
    return (
        has_real_booking_id(state, booking_id)
        or bool(state.get("crm_confirmed_stay"))
        or row_states_confirmed(state)
    )


#: Party size at which a room count stops being an instruction and becomes a
#: preference. The number matters more than the spelling — see below.
MULTI_ROOM_PARTY_SIZE = 5


def is_multi_room_party(adults, children=0) -> bool:
    """
    Is this party large enough that a room count is a preference, not a cage?

    A guest who says "2 rooms" for eleven people has named a number that cannot
    seat them; the allocator answers with three rooms and the card renders those
    three. The guest's "2" is still shown and still honoured — as a *preference*.

    That distinction has to be shared. `room_service.search_rooms` drops the
    count for these parties, so the card the guest is looking at genuinely
    carries three rooms against their "2". Any later check that *enforces* the
    count without asking this question first will reject the card the guest is
    holding, re-search, and render the same card again — a loop with no exit.

    Two call sites still encode this rule inline and are the reason this exists:
    `room_service.search_rooms` (which drops the count) and `room_searcher`
    (`is_multi_room`, which decides the card's shape). This is the shared
    spelling; those two have not been migrated yet.

    Unreadable counts read as "not a multi-room party": the conservative
    direction, since it keeps today's behaviour rather than silently relaxing a
    check on malformed session state.
    """
    try:
        return (int(adults or 0) + int(children or 0)) >= MULTI_ROOM_PARTY_SIZE
    except (TypeError, ValueError):
        return False


_CHILD_NOUN = r"(?:child|children|kid|kids|bacha|bachcha|bachche|bacche|beta|beti|son|daughter)"
_AGE_MARKER = re.compile(r"\b(?:aged|age|of|is|are|years?|yrs?|saal|old)\b|:", re.IGNORECASE)
# A year unit *immediately* after the number ("child 4 years") is an age marker
# too. Kept as its own anchored pattern rather than reusing `_AGE_MARKER` on the
# trailing text, or "child 4 is coming" would read as a four-year-old.
_YEAR_UNIT_AFTER = re.compile(r"^\s*(?:years?|yrs?|saal|sal|old)\b", re.IGNORECASE)
_AGE_POLICY_WORD = re.compile(
    r"\b(?:below|under|above|over|free|less\s+than|more\s+than|at\s+most|at\s+least)\b",
    re.IGNORECASE,
)


def extract_child_ages_from_text(text: str) -> list[int]:
    """
    Extract child ages from phrases like 'child 4 years', '4 year old', 'kids 6, 10', 'bacha 4 saal ka'.

    Two shapes, and the guards on each are the point.

    **Pattern 1 — "N years old".** The trailing qualifier ("old", "ka", "ki")
    is now *required*. Without it, "we have been visiting for 5 years" read as a
    5-year-old, and that age was merged into `state["child_ages"]` on every
    turn, where the three-band child policy turns it into money.

    **Pattern 2 — a child noun followed by ages.** This is what reads
    "kids 6, 10" — the previous pattern's own docstring example, which it did
    not actually match. A *single* number after the noun needs an age marker
    ("aged 5", "child is 7", "child 4 years"), because "children 2" is a head
    count, not a two-year-old, and that shape is common in this traffic. A run
    of two or more numbers ("kids 6, 10", "aged 5 and 8") is unambiguous and
    needs no marker. Numbers introduced by a policy word ("children below 5
    stay free") are skipped outright — that is a rule about children in general,
    not about this guest's.
    """
    if not text:
        return []
    ages: list[int] = []

    for m in re.finditer(r"\b(\d{1,2})\s*(?:years?|yrs?|saal|sal)\s*(?:old|ka|ki|ke)\b", text, re.IGNORECASE):
        age = int(m.group(1))
        if 0 <= age <= 17 and age not in ages:
            ages.append(age)

    for m in re.finditer(
        rf"\b{_CHILD_NOUN}\b(?P<between>[^\d\n]{{0,24}}?)(?P<numbers>\d{{1,2}}(?:\s*(?:,|and|&|/)\s*\d{{1,2}})*)(?P<after>[^\d\n]{{0,10}})",
        text,
        re.IGNORECASE,
    ):
        between = m.group("between") or ""
        if _AGE_POLICY_WORD.search(between):
            continue
        numbers = re.findall(r"\d{1,2}", m.group("numbers"))
        if len(numbers) < 2 and not (
            _AGE_MARKER.search(between) or _YEAR_UNIT_AFTER.match(m.group("after") or "")
        ):
            continue
        for raw in numbers:
            age = int(raw)
            if 0 <= age <= 17 and age not in ages:
                ages.append(age)
    return ages


def normalize_date(date_str: str | None) -> str | None:
    """Normalize DD/MM/YYYY, DD-MM-YYYY, or YYYY-MM-DD to YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        return None
    import re

    raw = date_str.strip()
    # The LLM extractor sometimes hands back a datetime rather than a date
    # ("2026-10-09 14:00:00"), usually when the guest names a clock time in the
    # same breath as the date. Take the date half, exactly as
    # format_display_date does, so the value is repaired here instead of
    # falling through to the caller and being echoed to the guest verbatim by
    # validate_proposed_date_range.
    #
    # Only when that half really is shaped like a date: a free-text value such
    # as "next tuesday" must still come back whole rather than truncated to
    # "next", which is what an unconditional split would do.
    head = raw.split("T")[0].split(" ")[0]
    if head != raw and re.match(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})$", head):
        raw = head
    if is_valid_date(raw):
        return raw

    # Try DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", raw)
    if m:
        day, month, year = m.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try YYYY/MM/DD
    m2 = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw)
    if m2:
        year, month, day = m2.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return raw


def format_display_date(date_str: str | None) -> str:
    """Format YYYY-MM-DD to DD/MM/YYYY for guest-facing messages."""
    if not date_str or not isinstance(date_str, str):
        return ""
    raw = date_str.strip().split("T")[0].split(" ")[0]
    import re
    m = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw)
    if m:
        y, month, day = m.groups()
        return f"{int(day):02d}/{int(month):02d}/{y}"
    m2 = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", raw)
    if m2:
        day, month, year = m2.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"
        
    # Fallback to datetime parsing
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(date_str.strip().replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y")
    except Exception:
        pass
        
    return raw


def format_quote_date_range(check_in: str | None, check_out: str | None) -> str:
    """
    Format check-in and check-out dates cleanly for WhatsApp quotes.
    Examples:
        '2026-08-22', '2026-08-23' -> '22-23 Aug'
        '2026-08-28', '2026-09-02' -> '28 Aug - 02 Sep'
    """
    if not check_in or not check_out:
        return ""
    from datetime import datetime
    try:
        ci_str = str(check_in).strip().split("T")[0].split(" ")[0]
        co_str = str(check_out).strip().split("T")[0].split(" ")[0]
        ci = datetime.strptime(ci_str, "%Y-%m-%d")
        co = datetime.strptime(co_str, "%Y-%m-%d")
        if ci.year == co.year and ci.month == co.month:
            return f"{ci.day}-{co.day} {ci.strftime('%b')}"
        else:
            return f"{ci.day} {ci.strftime('%b')} - {co.day} {co.strftime('%b')}"
    except Exception:
        ci_fmt = format_display_date(check_in) or str(check_in)
        co_fmt = format_display_date(check_out) or str(check_out)
        return f"{ci_fmt} - {co_fmt}"


def is_valid_date(date_str: str) -> bool:
    """Check if a string is a valid YYYY-MM-DD date."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def is_future_date(date_str: str) -> bool:
    """Check if a date is today or in the future."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        ist = timezone(timedelta(hours=5, minutes=30))
        return dt.date() >= datetime.now(ist).date()
    except (ValueError, TypeError):
        return False


def check_in_before_check_out(check_in: str, check_out: str) -> bool:
    """Validate that check-in is before check-out."""
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d")
        co = datetime.strptime(check_out, "%Y-%m-%d")
        return ci < co
    except (ValueError, TypeError):
        return False


def validate_proposed_date_range(check_in: str | None, check_out: str | None) -> tuple[bool, str | None]:
    """
    Validate proposed check-in and check-out dates.
    Returns (is_valid, error_message).
    """
    if check_in:
        if not is_valid_date(check_in):
            return False, f"Invalid check-in date format ({check_in}). Please specify dates in YYYY-MM-DD format."
        if not is_future_date(check_in):
            return False, f"Check-in date ({check_in}) cannot be in the past. Please select today or a future date."
    if check_out:
        if not is_valid_date(check_out):
            return False, f"Invalid check-out date format ({check_out}). Please specify dates in YYYY-MM-DD format."
    if check_in and check_out:
        if not check_in_before_check_out(check_in, check_out):
            return False, f"Check-out date ({check_out}) must be after your check-in date ({check_in}). Please provide a valid check-out date after {check_in}, or specify both new check-in and check-out dates."
    return True, None



def phone_lookup_variants(phone: str | None) -> list[str]:
    """
    Normalize a phone into common storage variants for Mongo lookups.
    Handles +91 / 91 / 10-digit local forms without inventing numbers.
    """
    import re

    raw = (phone or "").strip()
    if not raw:
        return []

    digits = re.sub(r"\D", "", raw)
    if not digits:
        return []

    variants: set[str] = {digits, f"+{digits}", raw}
    if digits.startswith("91") and len(digits) >= 12:
        local = digits[-10:]
        variants.update({local, f"91{local}", f"+91{local}"})
    elif len(digits) == 10:
        variants.update({f"91{digits}", f"+91{digits}"})

    return [v for v in variants if v]
