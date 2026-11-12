"""
Privacy Scrubber — Compliance with DPDP Act 2023.

Scrubs Aadhaar numbers and Passport numbers from incoming user messages
before persisting to logs, transcripts, or database sessions.
"""

import re

# 12-digit Aadhaar pattern (e.g. "1234 5678 9012" or "123456789012" or "1234-5678-9012")
_AADHAAR_REGEX = re.compile(r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b")

# Indian Passport: 1 letter (except Q, X, Z) followed by 7 digits
# International Passport general pattern: 1-2 letters followed by 6-9 digits
_PASSPORT_REGEX = re.compile(r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[0-9]\b|\b[A-Z]{1,2}[0-9]{6,9}\b", re.IGNORECASE)


def contains_id_number(text: str | None) -> bool:
    """Check if text contains an Aadhaar or Passport number."""
    if not text:
        return False
    return bool(_AADHAAR_REGEX.search(text) or _PASSPORT_REGEX.search(text))


def scrub_id_numbers(text: str | None) -> str:
    """Replace sensitive ID numbers with redacted placeholder."""
    if not text:
        return ""
    scrubbed = _AADHAAR_REGEX.sub("[REDACTED_AADHAAR]", text)
    scrubbed = _PASSPORT_REGEX.sub("[REDACTED_PASSPORT]", scrubbed)
    return scrubbed
