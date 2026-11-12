"""
Payment Gateway Repository — manages payment gateway & UPI Direct configurations.

Fetches VPA UPI ID and QR code from MongoDB (`payment_gateways` / `hotel_settings`).
"""

import logging
import urllib.parse
from typing import Any

from app.db.mongodb import payment_gateways_collection, hotel_settings_collection

logger = logging.getLogger(__name__)


async def get_hotel_payment_gateways(hotel_id: int | str) -> dict[str, Any] | None:
    """Fetch gateway configuration document for a hotel from payment_gateways."""
    coll = payment_gateways_collection()
    str_hid = str(hotel_id)
    doc = await coll.find_one({"hotel_id": str_hid})
    if not doc and str_hid in ("988273", "108959"):
        doc = await coll.find_one({"hotel_id": "111111"})
    return doc


def _invented_vpa_allowed() -> bool:
    """Whether the two name-derived fallbacks below may still run.

    See `PAYMENT_UPI_REQUIRES_CONFIGURED_VPA` in `app/config/settings.py` for
    why they are off. Read lazily so this repository module never imports the
    settings object at import time (it is imported by the payment graph, which
    the settings module itself is imported from).
    """
    try:
        from app.config.settings import get_settings
        return not getattr(get_settings(), "PAYMENT_UPI_REQUIRES_CONFIGURED_VPA", True)
    except Exception:  # configuration unavailable — never invent an address
        return False


async def get_hotel_upi_details(
    hotel_id: int | str,
    hotel_name: str = "Hotel",
    amount: float = 0.0,
    booking_id: str = "",
) -> dict[str, Any]:
    """
    Retrieve UPI Direct (VPA & QR Code) details for a hotel.

    Only a VPA **stored** on the hotel is returned: `payment_gateways` first,
    then `hotel_settings`. Nothing is derived from the hotel's name or code —
    a guessed `{name}@ybl` is a real address belonging to whoever registered
    it, and the QR makes it one tap away from a guest's money.

    Returns:
        {
            "connected": bool,
            "vpa": str | None,
            "qr_code_url": str | None,
            "upi_uri": str | None,
        }
    """
    str_id = str(hotel_id)
    vpa = None
    qr_code_url = None
    connected = False
    #: The hotel's own switch for UPI Direct. `payment_gateways` is the only
    #: place that stores one, so only that document can turn UPI off.
    upi_disabled = False

    # 0. Handles that are *known*, not guessed. Applied below, after the two
    # stored sources, so a hotel that configures its own VPA overrides this.

    # 1. Look up in payment_gateways
    try:
        gw_doc = await get_hotel_payment_gateways(hotel_id)
        if gw_doc and isinstance(gw_doc.get("gateways"), dict):
            upi_cfg = gw_doc["gateways"].get("upi") or {}
            if isinstance(upi_cfg, dict):
                connected = bool(upi_cfg.get("connected", True))
                # `connected: false` is the hotel saying "UPI Direct is off".
                # It used to be overwritten by the fallbacks below, so a hotel
                # that had deliberately disabled it still got a VPA and a QR.
                upi_disabled = not connected
                # Check VPA fields. Assigned only when found, so a gateway row
                # that exists but carries no handle cannot wipe the known-handle
                # bridge above.
                found_vpa = (
                    upi_cfg.get("vpa") or upi_cfg.get("vpa_id")
                    or upi_cfg.get("upi_id") or upi_cfg.get("merchant_id")
                )
                # Also check direct apiKey / api_key if it looks like a VPA (contains @)
                raw_key = upi_cfg.get("apiKey") or upi_cfg.get("api_key") or ""
                if not found_vpa and raw_key and "@" in str(raw_key) and not str(raw_key).startswith("gAAAAA"):
                    found_vpa = str(raw_key).strip()
                if found_vpa:
                    vpa = str(found_vpa).strip()
                    connected = True
                qr_code_url = upi_cfg.get("qr_code_url") or upi_cfg.get("qr_code") or upi_cfg.get("qrUrl")
    except Exception as e:
        logger.error(f"Error fetching PaymentGateways for hotel {hotel_id}: {e}")

    # 2. Fallback check in hotel_settings — also stored configuration, so it
    #    counts as "configured". Not read when the gateway itself says UPI is
    #    off: that switch is the hotel's, and a second collection carrying a
    #    stale handle must not override it.
    if not vpa and not upi_disabled:
        try:
            settings_doc = await hotel_settings_collection().find_one({"hotel_id": str_id})
            if settings_doc:
                # Direct snake_case schema
                bank = settings_doc.get("bank") or {}
                if isinstance(bank, dict) and bank.get("upi_id"):
                    vpa = bank["upi_id"].strip()
                    connected = True
                upi_sec = settings_doc.get("upi") or {}
                if isinstance(upi_sec, dict) and upi_sec.get("vpa"):
                    vpa = upi_sec["vpa"].strip()
                    connected = True
                
                # Nested settings_data schema
                if not vpa and isinstance(settings_doc.get("settings_data"), dict):
                    sdata = settings_doc["settings_data"]
                    s_bank = sdata.get("bank") or {}
                    if isinstance(s_bank, dict) and s_bank.get("upi_id"):
                        vpa = s_bank["upi_id"].strip()
                        connected = True
                    s_upi = sdata.get("upi") or {}
                    if isinstance(s_upi, dict) and s_upi.get("vpa"):
                        vpa = s_upi["vpa"].strip()
                        connected = True
        except Exception as e:
            logger.debug(f"Error checking hotel_settings for UPI: {e}")

    # 2b. Handles that are *known*, not guessed.
    #
    # These hotels store no VPA anywhere — measured 2026-10-27, every
    # `payment_gateways` and `hotel_settings` row for them is empty — but
    # `hotelsahu@ybl` has been given to guests in 418 messages, so it is the
    # address the hotel actually receives money at. Before `1771bf3` it arrived
    # through the name-derived fallback by coincidence (`property_code`
    # HOTELSAHU lowercased to "hotelsahu"). Turning the guessing off without
    # this list would have silently taken the UPI option away from the one hotel
    # that has traffic.
    #
    # A bridge, not the answer: each hotel should store its handle in
    # `payment_gateways.gateways.upi.vpa`. When they do, delete this block —
    # both stored sources are read first and win over it.
    _KNOWN_VPA_BY_HOTEL = {
        "111111": "hotelsahu@ybl",
        "988273": "hotelsahu@ybl",
        "108959": "hotelsahu@ybl",
    }
    if not vpa and not upi_disabled and str_id in _KNOWN_VPA_BY_HOTEL:
        vpa = _KNOWN_VPA_BY_HOTEL[str_id]
        connected = True

    # 3 + 4. INVENTED addresses — only under the rollback flag.
    #
    # Neither of these reads a stored VPA. #3 turns the hotel's own `name` or
    # `property_code` into an address; #4 turns the `hotel_name` the *caller*
    # passed into one, and never fails, so `connected` was always true. Both are
    # guesses, and a `upi://` URI built from a guess is a QR a guest can pay.
    # Measured 2026-10-27: hotel 999999 (exists nowhere) → sometestresort@ybl,
    # connected=True, QR yes.
    if _invented_vpa_allowed() and not upi_disabled:
        # 3. Fallback check from hotels / property_code or hotel_name
        if not vpa:
            try:
                from app.db.mongodb import hotels_collection
                h_doc = await hotels_collection().find_one({"hotel_id": str_id})
                if h_doc:
                    pcode = h_doc.get("property_code") or h_doc.get("name")
                    if pcode:
                        clean_code = "".join(c for c in pcode.lower() if c.isalnum())
                        if str_id in ("988273", "108959") or "sanidhyam" in clean_code:
                            vpa = "hotelsahu@ybl"
                        else:
                            vpa = f"{clean_code}@ybl"
                        connected = True
            except Exception as e:
                logger.debug(f"Error checking hotels collection for UPI: {e}")

        # 4. Default fallback VPA
        if not vpa:
            if str_id in ("988273", "108959") or "sanidhyam" in hotel_name.lower():
                vpa = "hotelsahu@ybl"
            else:
                clean_name = "".join(c for c in hotel_name.lower() if c.isalnum())
                vpa = f"{clean_name or 'hotel'}@ybl"
            connected = True

    if upi_disabled:
        # No half-state: a disabled hotel gets no handle and no QR, whatever a
        # stale field elsewhere in the document happens to contain.
        logger.info(
            "UPI_DIRECT_DISABLED | hotel=%s — payment_gateways says connected:false; "
            "returning no VPA", str_id,
        )
        vpa = None
        qr_code_url = None

    upi_uri = None
    if vpa:
        payee_encoded = urllib.parse.quote(hotel_name or "Hotel")
        amt_str = f"{amount:.2f}" if amount > 0 else ""
        note = f"Booking_{booking_id}" if booking_id else "Hotel_Booking"
        upi_uri = f"upi://pay?pa={vpa}&pn={payee_encoded}"
        if amt_str:
            upi_uri += f"&am={amt_str}"
        upi_uri += f"&cu=INR&tn={note}"

        if not qr_code_url:
            qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=350x350&data={urllib.parse.quote(upi_uri)}"

    return {
        # "Connected" means "there is a handle we can ask the guest to pay".
        # It used to be `bool(vpa or connected)`, which reported True for a
        # hotel with a QR configured and no VPA at all.
        "connected": bool(vpa) and connected,
        "vpa": vpa,
        "qr_code_url": qr_code_url,
        "upi_uri": upi_uri,
    }
