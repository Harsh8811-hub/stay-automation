"""
When a message may overwrite a payment receipt, and when it may only fill gaps.

Measured 2026-10-27, guest 919999000111. Staff's receipt said Rs.4,400 with
transaction id T2610261142497376769073. Two days later the guest typed his own
arithmetic — "Total payable: 17500/-" — into a message about the price. The
model tagged it as payment details, and `understand` replaced the whole object:
amount 4400 -> 17500, transaction_id -> None. From then on every reply quoted
"your uploaded payment of Rs.17,500" and demanded the Rs.380 difference.

The rule is not "trust receipts, mistrust text" — a guest typing a UTR number is
how half of them report a payment. It is narrower: a message that is not a
receipt may only *add* information, never contradict what the receipt said. A
real upload may still overwrite, because that is a guest correcting us with a
document.

Kept pure and separate so it can be tested without driving a conversation
through the graph.
"""

from __future__ import annotations

from typing import Any


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def merge_payment_details(
    existing: Any,
    extracted: dict[str, Any],
    is_receipt: bool,
) -> dict[str, Any]:
    """
    Combine the stored payment details with what this turn extracted.

    - `is_receipt=True` (a screenshot or a payment proof image): the new values
      win. The guest re-uploaded a document; it is more current than what we had.
    - `is_receipt=False` (prose): the new values win **only where we have
      nothing**. A number we already hold from a receipt is not overwritten by
      a number the guest typed while arguing about the price.
    """
    if not isinstance(existing, dict):
        existing = {}
    new = {k: v for k, v in (extracted or {}).items() if not _is_blank(v)}

    if is_receipt:
        return {**existing, **new}

    known = {k: v for k, v in existing.items() if not _is_blank(v)}
    return {**new, **known}


def kept_over_prose(existing: Any, extracted: dict[str, Any]) -> dict[str, Any]:
    """Fields the stored details kept against this turn — for the log line."""
    if not isinstance(existing, dict):
        return {}
    known = {k: v for k, v in existing.items() if not _is_blank(v)}
    return {
        k: known[k]
        for k, v in (extracted or {}).items()
        if k in known and not _is_blank(v) and v != known[k]
    }
