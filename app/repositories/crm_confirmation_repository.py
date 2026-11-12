"""
Read-only reader for the CRM's staff-confirmation signal.

Why this exists
---------------
A stay can be confirmed by staff in the Control Centre. That confirmation is written
into the CRM's own database and stops there — nothing carries it into the bot's
`bookings` collection. The bot's stage resolver therefore still reads `pending_payment`
and re-enters the payment flow, so a guest who has already paid is asked for the
advance again. Measured on hotel 111111, 2026-10-11: the guest paid, staff confirmed
and sent the voucher at 15:55, and at 06:37 the next morning the bot minted a *new*
booking and demanded ₹8,040.

This module answers exactly one question: **does the CRM believe this guest's stay is
already confirmed?**

It is deliberately NOT a booking lookup. The CRM holds no payment record, no amount
paid and no booking id — `bookingAmount` is present on only 56% of its confirmed
contacts, and there is no field anywhere naming what was actually received. So it can
never be the source of truth for money, and the caller must use this only to
**suppress** a payment demand, never to confirm a booking or release a room.

Two properties are load-bearing:

1. **It never raises.** A CRM outage, a timeout, a missing collection or a malformed
   document must all read as "unknown", because the alternative is losing the guest's
   reply. Every failure returns `None`.
2. **It is ON by default.** `CRM_CONFIRMATION_LOOKUP_ENABLED` and
   `CRM_CONFIRMED_STAGE_ENABLED` both default to `True`, so merging code that calls
   this changes live behaviour immediately — there is no "off until someone flips a
   flag" safety net here. The rollback is a revert commit.

Schema notes, all measured 2026-10-12 against the CRM's live data:
  - the hotel is `hotelID` (capital ID) at both the top level and inside
    `classification`; the snake-case `hotel_id` is empty on every document;
  - `phone_number` is `91` + 10 digits on 162 of 167, but 11 digits on 5 — so match on
    the last 10 digits rather than assuming a length;
  - `confidence` sits beside the intent, and 3 of 167 confirmed contacts are below 0.9
    (all three also carry no dates and no amount);
  - the intent is phone-scoped, not booking-scoped. There is exactly one `contact` per
    phone today, so a returning guest's next enquiry inherits the previous booking's
    confirmation. That is tolerable *only* because the caller suppresses rather than
    confirms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

CONFIRMED_INTENT = "CONFIRMED_BOOKING"

# How many trailing digits of a phone number identify the guest. The CRM stores
# `91` + 10 digits, but 5 of 167 documents carry 11 digits, so comparing whole
# strings would silently miss them.
_PHONE_TAIL = 10

_NOT_DIGIT = re.compile(r"\D")

# Separate cluster from the bot's own — a lazily created singleton, mirroring the
# loop-guard pattern in app/db/mongodb.py.
_crm_client: AsyncIOMotorClient | None = None


@dataclass(frozen=True)
class CrmConfirmation:
    """What the CRM says about a guest's stay, and how strongly."""

    confirmed: bool
    confidence: float | None = None
    check_in: str | None = None
    check_out: str | None = None
    hotel_id: str | None = None
    # The room the CRM recorded for this stay, e.g. "DELUXE DOUBLE KING SIZE BED".
    # Descriptive only — see the module docstring. The bot has no row of its own for a
    # CRM-confirmed guest, so this is the only name for their room that exists anywhere;
    # the alternative is a hardcoded "Deluxe Room" shown to a guest who booked something
    # else. Money is NOT read here for the same reason the docstring gives.
    room_type: str | None = None
    # The rest of what the CRM's classifier read off the conversation. None of it is
    # authoritative for money (see the docstring), but a confirmed guest who asks "how
    # many guests did I book for?" deserves their own answer rather than ours. Each is
    # optional because the classifier fills what it can: measured 2026-11-10 on the
    # 219 confirmed contacts, `adults` and `rooms` appear on 207, `bookingAmount` and
    # `ratePlan` on 185, `payment_claimed` on 209.
    #
    # `bookingAmount` and `payment_claimed` are deliberately NOT among them, even
    # though the CRM carries both: they are money, and this module's contract is
    # that the CRM never answers a money question. `test_no_amount_is_read_from_
    # the_crm` pins the field set to keep it that way.
    adults: int | None = None
    children: int | None = None
    rooms: int | None = None
    nights: int | None = None
    rate_plan: str | None = None
    # When the CRM last judged this contact. The only clock available for a
    # confirmation that names no dates — see `CRM_CONFIRMATION_NO_DATE_WINDOW_DAYS`.
    confirmed_at: datetime | None = None

    @property
    def details(self) -> dict:
        """The stay as a plain dict, for carrying through graph state.

        Only the fields that actually carry a value, so a reader can tell "the CRM
        did not say" from "the CRM said 2".
        """
        raw = {
            "room_type": self.room_type,
            "check_in": self.check_in,
            "check_out": self.check_out,
            "adults": self.adults,
            "children": self.children,
            "rooms": self.rooms,
            "nights": self.nights,
            "rate_plan": self.rate_plan,
        }
        return {k: v for k, v in raw.items() if v not in (None, "", False)}

    @property
    def has_dates(self) -> bool:
        """Whether the confirmation names a stay we could match against a booking."""
        return bool(self.check_in and self.check_out)


def normalize_phone(phone: object) -> str | None:
    """Last 10 digits of a phone number, or `None` if there are too few.

    The CRM and the bot disagree on formatting — the CRM stores `919415812132`, the
    bot's session key is built from whatever the CRM sent. Comparing raw strings
    across the two would be a silent miss.
    """
    digits = _NOT_DIGIT.sub("", str(phone or ""))
    return digits[-_PHONE_TAIL:] if len(digits) >= _PHONE_TAIL else None


def _crm_collection() -> AsyncIOMotorCollection | None:
    """The CRM's `contact` collection, or `None` if lookup is disabled/misconfigured."""
    global _crm_client

    settings = get_settings()
    uri = (settings.CRM_MONGO_URI or "").strip()
    database = (settings.CRM_CONFIRMATION_DATABASE or "").strip()
    if not uri or not database:
        logger.warning(
            "CRM confirmation lookup skipped: CRM_MONGO_URI or CRM_CONFIRMATION_DATABASE unset"
        )
        return None

    import asyncio

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    client_loop = getattr(_crm_client, "io_loop", None)
    if (
        _crm_client is None
        or (client_loop and client_loop.is_closed())
        or (running_loop and client_loop and client_loop != running_loop)
    ):
        # Short timeouts on purpose: this sits in the guest's turn, and a slow CRM must
        # cost a suppressed answer, not a stalled reply.
        _crm_client = AsyncIOMotorClient(
            uri,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=4000,
        )

    return _crm_client[database]["contact"]


async def lookup_crm_confirmation(
    hotel_id: object,
    phone: object,
    *,
    min_confidence: float | None = None,
) -> CrmConfirmation | None:
    """Does the CRM say this guest's stay is already confirmed?

    Returns `None` — "we don't know" — whenever the lookup is disabled, the phone or
    hotel cannot be normalised, no contact matches, or anything at all goes wrong. A
    `CrmConfirmation` is returned only for a real, matching document; its `confirmed`
    is then `True` only when the intent matches *and* clears the confidence floor.

    Never raises.
    """
    settings = get_settings()
    if not getattr(settings, "CRM_CONFIRMATION_LOOKUP_ENABLED", False):
        return None

    phone_tail = normalize_phone(phone)
    hotel = str(hotel_id or "").strip()
    if not phone_tail or not hotel:
        return None

    floor = (
        settings.CRM_CONFIRMATION_MIN_CONFIDENCE
        if min_confidence is None
        else min_confidence
    )

    try:
        collection = _crm_collection()
        if collection is None:
            return None

        # `hotelID` is not stored consistently: measured 2026-11-11, the CRM
        # holds `'111111'` (string) and `988273` (int) for the same kind of row.
        # Querying only `str(hotel_id)` silently missed 35 confirmed guests at
        # hotel 988273 — they were answered as if they had never booked. Both
        # shapes must be asked for; neither is "the wrong one" to fix up.
        hotel_forms: list = [hotel]
        if hotel.isdigit():
            hotel_forms.append(int(hotel))

        doc = await collection.find_one(
            {"hotelID": {"$in": hotel_forms}, "phone_number": {"$regex": f"{phone_tail}$"}},
            {
                "_id": 0,
                "classification.primary_intent": 1,
                "classification.confidence": 1,
                "classification.extracted_facts": 1,
                "classification.hotelID": 1,
                "classification.classified_at": 1,
                "last_interaction": 1,
            },
        )
        if not doc:
            return None

        classification = doc.get("classification") or {}
        intent = str(classification.get("primary_intent") or "").strip().upper()
        confidence = classification.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None

        facts = classification.get("extracted_facts") or {}
        confirmed = intent == CONFIRMED_INTENT and (
            confidence is None or confidence >= float(floor)
        )

        return CrmConfirmation(
            confirmed=confirmed,
            confidence=confidence,
            check_in=_as_date(facts.get("checkInDate")),
            check_out=_as_date(facts.get("checkOutDate")),
            hotel_id=str(classification.get("hotelID") or hotel),
            room_type=_as_date(facts.get("roomType")),
            adults=_as_int(facts.get("adults")),
            children=_as_int(facts.get("children")),
            rooms=_as_int(facts.get("rooms")),
            nights=_as_int(facts.get("numberOfNights")),
            rate_plan=_as_text(facts.get("ratePlan")),
            # `classified_at` is an ISO *string* on every one of the 219 confirmed
            # contacts (measured 2026-11-10), not a BSON date, so it needs parsing;
            # `last_interaction` beside it is a real datetime and is the fallback.
            confirmed_at=_as_datetime(classification.get("classified_at"))
            or _as_datetime(doc.get("last_interaction")),
        )
    except Exception:  # noqa: BLE001 — an unknown CRM must never cost the guest a reply
        logger.warning(
            "CRM confirmation lookup failed (hotel=%s, phone=…%s) — treating as unknown",
            hotel,
            phone_tail[-4:],
            exc_info=True,
        )
        return None


def _as_date(value: object) -> str | None:
    """The CRM's dates are sometimes absent and sometimes empty strings."""
    text = str(value or "").strip()
    return text or None


def _as_text(value: object) -> str | None:
    """Same as `_as_date`, for a field that is not a date. The CRM writes the
    string "None" into some fact slots, which must read as absent."""
    text = str(value or "").strip()
    return None if text in ("", "None", "null") else text


def _as_int(value: object) -> int | None:
    """The facts arrive as strings ("4", "2") as often as numbers.

    `bool` is an `int` in Python, so it is rejected explicitly: `True` would
    otherwise read as "1 adult".
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = _as_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _as_datetime(value: object) -> datetime | None:
    """A BSON date or an ISO string → aware UTC, or None."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = _as_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def reset_crm_client() -> None:
    """Drop the cached client. For tests, and for a loop that has gone away."""
    global _crm_client
    _crm_client = None
