"""
Bank Repository — manages hotel bank account details for direct NEFT/RTGS/IMPS transfers.
Source of truth: hotel_settings in bot_migration.
"""

import logging
from typing import Any

from app.db.mongodb import hotel_settings_collection, hotels_collection, hotel_details_collection

logger = logging.getLogger(__name__)


def _get_hotel_variants(hotel_id: int | str) -> list[int | str]:
    """Build list of tenant ID representations including known cross-system aliases."""
    str_hotel_id = str(hotel_id)
    if str_hotel_id in ("988273", "108959"):
        return ["108959", "988273", 108959, 988273]
    variants: list[int | str] = [str_hotel_id]
    if str_hotel_id.isdigit():
        variants.append(int(hotel_id))
    return list(dict.fromkeys(variants))


async def get_hotel_bank_details(hotel_id: int | str) -> dict[str, Any]:
    """
    Retrieve bank account details for a hotel from hotel_settings.
    
    Returns a normalized dict:
    {
        "is_configured": bool,
        "account_name": str,
        "bank_name": str,
        "account_number": str,
        "ifsc_code": str,
        "branch": str,
        "swift_code": str,
        "hotel_name": str,
        "contact_phone": str,
        "contact_email": str,
    }
    """
    variants = _get_hotel_variants(hotel_id)

    account_name = ""
    bank_name = ""
    account_number = ""
    ifsc_code = ""
    branch = ""
    swift_code = ""
    hotel_name = ""
    contact_phone = ""
    contact_email = ""

    # 1. Fetch from hotel_settings
    try:
        settings_doc = await hotel_settings_collection().find_one({"hotel_id": {"$in": variants}})
        if settings_doc:
            # Business info fallback
            business = settings_doc.get("business") or {}
            if isinstance(business, dict):
                hotel_name = business.get("name") or ""
                contact_phone = business.get("phone") or ""
                contact_email = business.get("email") or ""

            # Check direct bank dict
            bank = settings_doc.get("bank") or {}
            if not isinstance(bank, dict) and isinstance(settings_doc.get("settings_data"), dict):
                bank = settings_doc["settings_data"].get("bank") or {}

            if isinstance(bank, dict):
                account_number = str(bank.get("account") or bank.get("account_number") or "").strip()
                ifsc_code = str(bank.get("ifsc") or bank.get("ifsc_code") or "").strip().upper()
                branch = str(bank.get("branch") or "").strip()
                swift_code = str(bank.get("swift") or bank.get("swift_code") or "").strip().upper()
                
                # Bank name vs Account name resolution
                raw_name = str(bank.get("name") or "").strip()
                raw_bank_name = str(bank.get("bank_name") or bank.get("bank") or "").strip()

                if raw_bank_name:
                    bank_name = raw_bank_name
                    account_name = raw_name or hotel_name
                elif raw_name:
                    # If raw_name looks like a bank name (e.g. "HDFC Bank", "SBI")
                    if any(w in raw_name.lower() for w in ("bank", "sbi", "hdfc", "icici", "pnb", "axis")):
                        bank_name = raw_name
                        account_name = hotel_name
                    else:
                        account_name = raw_name
                        bank_name = "Bank Account"
    except Exception as e:
        logger.error(f"Error fetching bank details from hotel_settings for {hotel_id}: {e}", exc_info=True)

    # 2. Enrich hotel name / contact details if missing
    if not hotel_name or not contact_phone:
        try:
            h_doc = await hotels_collection().find_one({"hotel_id": {"$in": variants}})
            if h_doc:
                hotel_name = hotel_name or h_doc.get("name") or ""
                contact_phone = contact_phone or h_doc.get("phone") or ""
                contact_email = contact_email or h_doc.get("email") or ""

            if not contact_phone:
                hd_doc = await hotel_details_collection().find_one({"hotel_id": {"$in": variants}})
                if hd_doc and isinstance(hd_doc.get("contact"), dict):
                    c = hd_doc["contact"]
                    contact_phone = contact_phone or c.get("phone") or c.get("mobile") or ""
                    contact_email = contact_email or c.get("email") or ""
        except Exception as e:
            logger.debug(f"Error checking hotels/hotel_details for contact: {e}")

    account_name = account_name or hotel_name or "Hotel"
    is_configured = bool(account_number and ifsc_code)

    return {
        "is_configured": is_configured,
        "account_name": account_name,
        "bank_name": bank_name or ("Bank Account" if is_configured else ""),
        "account_number": account_number,
        "ifsc_code": ifsc_code,
        "branch": branch,
        "swift_code": swift_code,
        "hotel_name": hotel_name or "Hotel",
        "contact_phone": contact_phone,
        "contact_email": contact_email,
    }
