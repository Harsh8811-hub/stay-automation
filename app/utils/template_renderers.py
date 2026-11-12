"""
Template variable renderers and card builders conforming to StayInn WhatsApp Template Catalog v2.0.
Provides Indian currency formatting, date phrases, guest lines, label maps, and interactive card structures.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.utils.button_registry import BUTTONS_BY_ID, normalize_reply_text

IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger(__name__)


def render_currency(value: Any) -> str:
    """Format numeric values as Indian rupee representation (e.g. 5,250 or 1,25,000) with no decimals."""
    if value is None or value == "":
        return "0"
    try:
        if isinstance(value, str):
            clean = re.sub(r"[^\d.]", "", value)
            if not clean:
                return "0"
            f = float(clean)
        else:
            f = float(value)
        if math.isnan(f) or math.isinf(f):
            return "0"
        num = int(round(f))
    except (ValueError, TypeError, OverflowError):
        return "0"

    if num < 0:
        return f"-{render_currency(-num)}"

    s = str(num)
    if len(s) <= 3:
        return s

    # Indian numbering grouping: last 3 digits, then groups of 2 digits
    last_three = s[-3:]
    remaining = s[:-3]
    chunks = []
    while len(remaining) > 2:
        chunks.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        chunks.append(remaining)
    chunks.reverse()
    return ",".join(chunks) + "," + last_three


def render_nights_phrase(nights: int | str | None, lang: str = "English") -> str:
    """Singular/plural nights phrase (e.g., '1 night', '2 nights', '1 raat', '2 raat')."""
    try:
        if isinstance(nights, (int, float)):
            if math.isnan(nights) or math.isinf(nights):
                n = 1
            else:
                n = max(1, int(nights))
        else:
            n = max(1, int(nights or 1))
    except (ValueError, TypeError, OverflowError):
        n = 1

    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    if is_hi:
        return f"{n} raat"
    return f"{n} night" if n == 1 else f"{n} nights"


def render_dates_line(check_in: str | None, check_out: str | None, lang: str = "English") -> str:
    """
    Format standard stay date line with weekday, date and nights:
    e.g. 'Fri 22 Aug → Sun 24 Aug · 2 nights' / 'Fri 22 Aug → Sun 24 Aug · 2 raat'.
    """
    if not check_in:
        return ""

    cin_dt = _parse_date(check_in)
    cout_dt = _parse_date(check_out) if check_out else None

    if not cin_dt:
        return str(check_in) + (f" → {check_out}" if check_out else "")

    if not cout_dt:
        # Default to 1 night
        cout_dt = cin_dt + timedelta(days=1)

    now_year = datetime.now(IST).year
    show_year_in = f" {cin_dt.year}" if cin_dt.year != now_year else ""
    show_year_out = f" {cout_dt.year}" if cout_dt.year != now_year else ""

    cin_str = f"{cin_dt.strftime('%a')} {cin_dt.day} {cin_dt.strftime('%b')}{show_year_in}"
    cout_str = f"{cout_dt.strftime('%a')} {cout_dt.day} {cout_dt.strftime('%b')}{show_year_out}"

    nights = max(1, (cout_dt - cin_dt).days)
    n_str = render_nights_phrase(nights, lang=lang)

    return f"{cin_str} → {cout_str} · {n_str}"


def render_date_single(dt_str: str | None) -> str:
    """Format single date as e.g. 'Fri 22 Aug'."""
    if not dt_str:
        return ""
    dt = _parse_date(dt_str)
    if not dt:
        return str(dt_str)
    now_year = datetime.now(IST).year
    year_part = f" {dt.year}" if dt.year != now_year else ""
    return f"{dt.strftime('%a')} {dt.day} {dt.strftime('%b')}{year_part}"


def render_example_date(days_ahead: int = 7) -> str:
    """Dynamic example date (today + 7 days) as e.g. '18 Oct'."""
    now = datetime.now(IST) + timedelta(days=days_ahead)
    return f"{now.day} {now.strftime('%b')}"


def render_guests_line(
    adults: int | str | None,
    children: int | str | None = 0,
    child_ages: list[int] | None = None,
    lang: str = "English",
) -> str:
    """
    Format standard guests line:
    e.g. '2 adults + 1 child (8 yrs)' / '2 adults + 1 bachcha (8 saal)' / '1 adult'.
    """
    try:
        if isinstance(adults, (int, float)):
            if math.isnan(adults) or math.isinf(adults):
                a = 1
            else:
                a = max(1, int(adults))
        else:
            a = max(1, int(adults or 1))
    except (ValueError, TypeError, OverflowError):
        a = 1

    try:
        if isinstance(children, (int, float)):
            if math.isnan(children) or math.isinf(children):
                c = 0
            else:
                c = max(0, int(children))
        else:
            c = max(0, int(children or 0))
    except (ValueError, TypeError, OverflowError):
        c = 0

    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")
    a_str = f"{a} adult" if a == 1 else f"{a} adults"

    if c == 0:
        return a_str

    if is_hi:
        c_label = "1 bachcha" if c == 1 else f"{c} bachche"
    else:
        c_label = "1 child" if c == 1 else f"{c} children"

    if child_ages and len(child_ages) > 0:
        age_str = ", ".join(str(age) for age in child_ages)
        unit = "saal" if is_hi else ("yr" if len(child_ages) == 1 and child_ages[0] == 1 else "yrs")
        c_label += f" ({age_str} {unit})"

    return f"{a_str} + {c_label}"


def render_label(internal_value: str | None, lang: str = "English") -> str:
    """Map raw internal system keys and status enums to guest-facing text."""
    if not internal_value:
        return ""
    val = str(internal_value).strip().lower()
    is_hi = str(lang).lower() in ("hinglish", "hindi", "hi")

    LABEL_MAP = {
        "confirmed": ("Confirmed ✅", "Confirmed ✅"),
        "pending_payment": ("Waiting for payment ⏳", "Payment baaki hai ⏳"),
        "verifying": ("Payment being verified ⏳", "Payment verify ho raha hai ⏳"),
        "cancelled": ("Cancelled", "Cancel ho chuki hai"),
        "checked_in": ("Checked in", "Check-in ho chuka hai"),
        "ac": ("an AC room", "AC room"),
        "non_ac": ("a non-AC room", "non-AC room"),
        "ep": ("room only (no meals)", "sirf room (khana shamil nahi)"),
        "cp": ("with breakfast", "breakfast ke saath"),
        "map": ("with breakfast and dinner", "breakfast aur dinner ke saath"),
        "check_in": ("Check-in", "Check-in"),
        "check_out": ("Check-out", "Check-out"),
        "guests": ("Guests", "Guests"),
        "dates": ("Dates", "Dates"),
        "meal_plan": ("Meal plan", "Meal plan"),
        "room_type": ("Room type", "Room type"),
        "guest_name": ("Name on booking", "Booking ka naam"),
    }

    if val in LABEL_MAP:
        return LABEL_MAP[val][1] if is_hi else LABEL_MAP[val][0]

    return internal_value.replace("_", " ").title()


def _parse_date(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    s = str(dt_str).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d %b"):
        try:
            if fmt == "%d %b":
                curr_year = datetime.now(IST).year
                parsed = datetime.strptime(f"{s} {curr_year}", "%d %b %Y")
                return parsed
            parsed = datetime.strptime(s, fmt)
            return parsed
        except ValueError:
            continue
    return None


# ── Card Builders ────────────────────────────────────────────────────────────

def text_card(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text.strip()}]


def _check_button_is_routable(button_id: str, title: str) -> None:
    """
    Warn when a button is authored with wording the router cannot recognise.

    Only the title comes back from a tap, so a title the registry does not declare is a
    control that looks wired and does nothing — which is exactly how four buttons had
    silently died before the registry existed. This is a warning rather than a raise on
    purpose: raising here would turn one bad label into a 500 on a guest's turn, and the
    test guard (`tests/test_button_registry.py`) already fails loudly at build time.

    The id is checked too, so a template cannot quietly invent a new button that no one
    declared an action for.
    """
    declared = BUTTONS_BY_ID.get(button_id)
    if declared is None:
        logger.error(
            "Button id %r is not declared in app.utils.button_registry, so no tap on it can "
            "be routed. Declare it there (and give it an action) or use an existing id.",
            button_id,
        )
        return

    if normalize_reply_text(title) not in {normalize_reply_text(t) for t in declared.titles}:
        logger.error(
            "Button %r is authored with title %r, which the registry does not declare for it "
            "(known: %s). A tap sends the title back, so this button would do nothing. Add the "
            "title to the registry or use one of the declared ones.",
            button_id, title, list(declared.titles),
        )


def button_card(
    text: str,
    buttons: list[dict[str, str] | str],
    footer: str | None = None,
) -> list[dict[str, Any]]:
    """
    Construct a card carrying interactive reply buttons.
    Each button item is either a dict {'id': ..., 'title': ...} or a string label.
    """
    btn_objs = []
    for btn in buttons[:3]:  # WhatsApp hard cap: max 3 reply buttons
        if isinstance(btn, dict):
            b_id = btn.get("id") or btn.get("title") or "btn"
            b_title = btn.get("title") or btn.get("label") or btn.get("id") or "Option"
        else:
            # A bare label. The id is derived from the words, which means it is declared
            # nowhere and the title was never agreed — checked below like any other button,
            # so this form reports itself instead of quietly escaping the registry.
            b_title = str(btn)[:20]
            b_id = b_title.lower().replace(" ", "_")
        _check_button_is_routable(str(b_id), str(b_title))
        btn_objs.append({"id": str(b_id)[:256], "title": str(b_title)[:20]})

    card: dict[str, Any] = {
        "type": "button_group",
        "text": text.strip(),
        "buttons": btn_objs,
    }
    if footer:
        card["footer"] = str(footer)[:60]
    return [card]


def list_card(
    text: str,
    button_title: str,
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Construct an interactive list message card (up to 10 rows).

    Each row is checked against the registry exactly as a button is, and for the same
    reason: the CRM forwards a tap as the *title*, so a row whose title nothing recognises
    is a choice that silently does nothing — the failure mode that killed five buttons
    before the registry existed. A list is the only way to offer more than three options
    (`button_card` caps at three), so this is where the four-option menus land and the
    check belongs here rather than at each call site: a check written at the call site is
    one a future list can simply not have.
    """
    for section in sections or []:
        for row in (section or {}).get("rows") or []:
            if not isinstance(row, dict):
                continue
            _check_button_is_routable(
                str(row.get("id") or row.get("title") or "row"),
                str(row.get("title") or row.get("label") or row.get("id") or "Option"),
            )
    return [{
        "type": "list_message",
        "text": text.strip(),
        "button_title": str(button_title)[:20],
        "sections": sections,
    }]
