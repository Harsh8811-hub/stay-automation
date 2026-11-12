"""Phone number normalization helpers for booking lookups."""

from __future__ import annotations

import re


def digits_only(phone: str | None) -> str:
    return re.sub(r"\D", "", phone or "")


def phone_lookup_variants(phone: str | None) -> list[str]:
    """
    Return common storage forms for the same Indian/WhatsApp phone number.
    Used so DB lookups work across +91XXXXXXXXXX / 91XXXXXXXXXX / XXXXXXXXXX.
    """
    digits = digits_only(phone)
    if not digits:
        return []

    variants: set[str] = {digits, f"+{digits}"}

    if digits.startswith("91") and len(digits) >= 12:
        local = digits[2:]
        variants.update({local, f"91{local}", f"+91{local}"})
    elif len(digits) == 10:
        variants.update({f"91{digits}", f"+91{digits}"})

    raw = (phone or "").strip()
    if raw:
        variants.add(raw)

    return [v for v in variants if v]


def normalize_phone(phone: str | None) -> str:
    """Normalize phone number to standard E.164-style string or clean digits."""
    digits = digits_only(phone)
    if not digits:
        return (phone or "").strip()
    if len(digits) == 10:
        return f"+91{digits}"
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    return f"+{digits}" if not (phone or "").startswith("+") else (phone or "").strip()
