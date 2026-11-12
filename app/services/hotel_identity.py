"""
Hotel Identity — one reader for the facts a guest is told about a property.

Why this exists
---------------
The stage handlers (in-house, confirmed, checked-out) each carried their own
private copy of the hotel's name, front-desk number and map link:

    phone = "+918141459509" if is_sanidhyam else "+919453222931"

That ternary is wrong for every property it was not written for. Hotel Paradise
(376891) and Hotel Maharaja (968225) are not Sample Property, so they fell through to
**Sample Hotel's number** — the R4 defect in the 3 Oct 2026 test report: one
hotel's front desk being handed to another hotel's guest.

The data was never missing. `hotel_details` already carries it for every
tenant, and `app/repositories/hotel_repository.py` already exposes canonical
readers for it (`get_hotel_contact_info`, `get_hotel_location`,
`get_hotel_by_id`). This module is the single place that decides how those
records become the strings a guest sees, so the three handlers cannot drift
apart again.

Fail-loud, not fail-to-Sahu
---------------------------
`property_router`'s own docstring states the principle — "NEVER silently fall
back to Sample Hotel (111111) for unmapped tenants" — and this follows it. When a
property has no phone number on file, the copy degrades to "at reception"
rather than borrowing a number. When it has no known wifi credentials, the
wifi block is omitted rather than printed with another hotel's password. A
missing fact is a smaller harm than a wrong one.

What is still not in the database
---------------------------------
`hotel_details` holds exactly `property`, `contact`, `location` and
`transit_directions`. There is **no wifi field and no review-link field in any
collection**. `_KNOWN_PROPERTY_EXTRAS` below isolates those two facts, keyed by
hotel, so they are at least in one place and visible rather than buried in
control flow. They should move to `hotel_settings` — until then, a property
absent from that table simply gets no wifi block and no review link.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.repositories import hotel_repository

logger = logging.getLogger(__name__)

# Hotel 111111 is Sample Hotel; 988273 (and its aliases) is Sample Property.
_SAHU = "111111"
_SANIDHYAM = {"988273", "108959", "710235"}

# ── The two facts with no database home ──────────────────────────────────────
#
# Deliberately sparse. A hotel not listed here gets NO wifi block and NO review
# link rather than another property's credentials — a wrong wifi password is
# worse than an absent one, and the pre-2026-10 code handed Sahu's password to
# every non-Sample Property tenant.
#
# TODO(data): move to `hotel_settings` (`wifi_name`, `wifi_password`,
# `review_link`) and delete this table.
_KNOWN_PROPERTY_EXTRAS: dict[str, dict[str, str]] = {
    _SAHU: {
        "wifi_name": "HotelSahu_Guest",
        "wifi_password": "KashiVishwanath2026",
        "review_link": "https://g.page/r/HotelSahuRiverside City/review",
    },
    "988273": {
        "wifi_name": "Sample Property_Guest",
        "wifi_password": "Sample Property2026",
        "review_link": "https://g.page/r/Sample PropertyRiverside City/review",
    },
}

# Used only when neither the session nor the database names the property.
# Never a real hotel name — an unnamed property must not be given another
# property's identity.
_UNNAMED_PROPERTY = "our hotel"

# Map-link field spellings seen across tenants; mirrors the probe list in
# `hotel_repository.format_location` so the two agree on what counts.
_MAP_KEYS = (
    "google_map_link",
    "googleMapsUrl",
    "mapsUrl",
    "mapLink",
    "google_maps_url",
)


@dataclass(frozen=True)
class HotelIdentity:
    """The guest-facing facts about one property. Any string may be empty."""

    hotel_id: str
    name: str
    phone: str
    alt_phone: str
    map_link: str
    address: str
    wifi_name: str
    wifi_password: str
    review_link: str

    @property
    def has_wifi(self) -> bool:
        return bool(self.wifi_name or self.wifi_password)

    @property
    def has_review_link(self) -> bool:
        return bool(self.review_link)

    def front_desk_clause(self, label: str = "Front Desk") -> str:
        """
        A "Front Desk: <number>" fragment that stays readable when no number is
        on file. Callers embed this in prose, so returning an empty string
        would leave a dangling colon in the guest's message.
        """
        if self.phone:
            return f"{label} / Reception: {self.phone}"
        return f"{label} / Reception: available at the desk"

    def contact_sentence(self) -> str:
        """
        Trailing "call us on <number>" sentence that reads correctly when the
        number is missing, instead of printing "call us on .".
        """
        if self.phone:
            return f"You can call the front desk on {self.phone}."
        return "Please ask at reception for anything urgent."


def _first_str(source: dict | None, keys: tuple[str, ...]) -> str:
    """First non-empty value among `keys`, coerced to a stripped string."""
    if not isinstance(source, dict):
        return ""
    for key in keys:
        val = source.get(key)
        if isinstance(val, (list, tuple)):
            val = next((v for v in val if v), None)
        if val not in (None, ""):
            return str(val).strip()
    return ""


def _compose_address(location: dict | None) -> str:
    """
    A single-line address from whichever shape the tenant stores.

    `_first_present` in `hotel_repository` handles the flat field; the nested
    `address` dict is the shape `format_location` also composes from, and is
    reproduced here because the stage handlers want one line, not a card.
    """
    if not isinstance(location, dict):
        return ""
    flat = _first_str(location, ("property_address", "propertyAddress"))
    if flat:
        return flat
    nested = location.get("address")
    if isinstance(nested, str):
        return nested.strip()
    if isinstance(nested, dict):
        parts = [
            str(nested[k]).strip()
            for k in ("line1", "line2", "street", "area", "city", "state", "pincode")
            if nested.get(k)
        ]
        return ", ".join(parts)
    return ""


async def _safe(
    label: str,
    hotel_id: str,
    call: Callable[[str], Awaitable[Any]],
) -> Any:
    """
    Run one repository read, converting any failure into `None`.

    A Mongo blip must degrade this to empty fields, never raise into a
    guest-facing handler — the message would be lost entirely, which is far
    worse than a missing phone number.
    """
    try:
        return await call(hotel_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"HOTEL_IDENTITY_{label}_LOOKUP_FAILED hotel={hotel_id}: {exc}")
        return None


async def resolve_hotel_identity(
    hotel_id: int | str,
    state: dict | None = None,
) -> HotelIdentity:
    """
    Resolve the guest-facing identity of a property.

    Precedence, per field:

      name      session `hotel_name` -> `hotels.name` -> "our hotel"
      phone     `hotel_details.contact` -> WhatsApp number -> ""
      map_link  `hotel_details.location` -> ""
      address   `hotel_details.location` -> ""

    The session name is kept first because a caller may already have resolved
    it (and because `hotel_name` is stamped on the state early in a session);
    it is the only field the session is allowed to win, since a wrong *name* is
    visibly wrong while a wrong *number* silently misroutes a guest.

    Every repository read is individually guarded and the two that are always
    needed run concurrently: this is called from handlers that fire on *every*
    message a guest sends, so the three lookups used to be three sequential
    round-trips on the critical path of every reply.
    """
    state = state or {}
    key = str(hotel_id).strip()

    name = str(state.get("hotel_name") or "").strip()
    phone = alt_phone = map_link = address = ""

    contact, location = await asyncio.gather(
        _safe("CONTACT", key, hotel_repository.get_hotel_contact_info),
        _safe("LOCATION", key, hotel_repository.get_hotel_location),
    )

    if isinstance(contact, dict):
        phone = str(contact.get("phone") or "").strip()
        alt_phone = str(contact.get("alt_phone") or "").strip()
        if not phone:
            # A WhatsApp-only tenant still has a reachable number.
            phone = str(contact.get("whatsapp") or "").strip()

    if isinstance(location, dict):
        map_link = _first_str(location, _MAP_KEYS)
        address = _compose_address(location)

    if not name:
        hotel_doc = await _safe("PROFILE", key, hotel_repository.get_hotel_by_id)
        if isinstance(hotel_doc, dict):
            name = str(hotel_doc.get("name") or "").strip()

    if not name:
        # NOT a fallback to a real hotel — see the module docstring.
        name = _UNNAMED_PROPERTY
        logger.warning(f"HOTEL_IDENTITY_UNNAMED hotel={key} — no name in session or DB")

    extras = _KNOWN_PROPERTY_EXTRAS.get(key, {})
    if not extras and key in _SANIDHYAM:
        extras = _KNOWN_PROPERTY_EXTRAS["988273"]

    return HotelIdentity(
        hotel_id=key,
        name=name,
        phone=phone,
        alt_phone=alt_phone,
        map_link=map_link,
        address=address,
        wifi_name=extras.get("wifi_name", ""),
        wifi_password=extras.get("wifi_password", ""),
        review_link=extras.get("review_link", ""),
    )
