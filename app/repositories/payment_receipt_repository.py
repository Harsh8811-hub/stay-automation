"""
Payment Receipt Repository — CRUD for OCR/vision payment receipts.

Receipts are stored in the merged `payments` collection as flat documents
with `source="receipt"` (the legacy `payment_receipts` collection was dropped
and its documents merged into `payments`).
"""

from datetime import datetime, timezone

from app.db.mongodb import payment_receipts_collection


import logging

logger = logging.getLogger(__name__)


async def ensure_indexes() -> None:
    """Create indexes on the merged `payments` collection. Idempotent."""
    coll = payment_receipts_collection()
    try:
        # Unique partial index on the flat receipt transaction id to prevent duplicates
        # while ignoring null/missing fields for non-receipt payment rows.
        await coll.create_index(
            "receipt_transaction_id",
            unique=True,
            partialFilterExpression={"receipt_transaction_id": {"$type": "string"}},
        )
    except Exception as e:
        if getattr(e, "code", None) == 86 or "IndexKeySpecsConflict" in str(e):
            logger.info("Dropping legacy conflicting receipt_transaction_id index and recreating with partialFilterExpression...")
            try:
                await coll.drop_index("receipt_transaction_id_1")
                await coll.create_index(
                    "receipt_transaction_id",
                    unique=True,
                    partialFilterExpression={"receipt_transaction_id": {"$type": "string"}},
                )
            except Exception as drop_err:
                logger.warning(f"Failed to recreate receipt_transaction_id index after drop: {drop_err}")
        else:
            logger.warning(f"Failed to create receipt_transaction_id index on payments: {e}")

    try:
        await coll.create_index("session_id")
        await coll.create_index("hotel_id")
    except Exception as e:
        logger.warning(f"Failed to create session_id/hotel_id index on payments: {e}")


async def get_receipt_by_transaction_id(transaction_id: str) -> dict | None:
    """Fetch a parsed payment receipt by transaction_id."""
    if not transaction_id:
        return None
    coll = payment_receipts_collection()
    doc = await coll.find_one({"receipt_transaction_id": transaction_id})
    if doc:
        doc.pop("_id", None)
    return doc


async def check_duplicate_transaction(transaction_id: str) -> bool:
    """Check if a payment with this transaction_id already exists."""
    doc = await get_receipt_by_transaction_id(transaction_id)
    return doc is not None


async def save_payment_receipt(receipt_data: dict) -> str:
    """
    Save a payment receipt as a flat document in the merged `payments`
    collection with `source="receipt"`. Upserts on `receipt_transaction_id`.
    """
    import uuid
    coll = payment_receipts_collection()
    payment_details = receipt_data.get("payment_details", {}) or {}
    txn_id = (
        receipt_data.get("receipt_transaction_id")
        or receipt_data.get("transaction_id")
        or (payment_details.get("transaction_id") if isinstance(payment_details, dict) else None)
    )

    now = datetime.now(timezone.utc)
    payment_id = receipt_data.get("payment_id") or f"pay_rcpt_{txn_id or uuid.uuid4().hex[:12]}"

    # Build a flat receipt document.
    doc = {
        "payment_id": payment_id,
        "source": "receipt",
        "status": "paid",
        "receipt_amount": receipt_data.get("amount") or (payment_details.get("amount") if isinstance(payment_details, dict) else None),
        "receipt_transaction_id": txn_id,
        "receipt_status": (payment_details.get("status") if isinstance(payment_details, dict) else None) or receipt_data.get("status") or "verified",
        "receipt_method": (payment_details.get("payment_app") if isinstance(payment_details, dict) else None) or receipt_data.get("method"),
        "session_id": receipt_data.get("session_id"),
        "hotel_id": str(receipt_data.get("hotel_id")) if receipt_data.get("hotel_id") is not None else None,
        "is_approved": receipt_data.get("is_approved", True),
        "created_at": receipt_data.get("created_at", now),
        "updated_at": now,
    }

    # Preserve any extra fields under metadata.
    extras = {k: v for k, v in receipt_data.items() if k not in ("amount", "status", "method", "payment_details", "payment_id", "session_id", "hotel_id", "is_approved", "created_at", "updated_at", "receipt_transaction_id", "transaction_id")}
    if extras:
        doc["metadata"] = extras

    if txn_id:
        await coll.update_one(
            {"receipt_transaction_id": txn_id},
            {"$set": doc},
            upsert=True,
        )
        return txn_id
    else:
        res = await coll.insert_one(doc)
        return str(res.inserted_id)


async def get_latest_receipt_by_session(session_id: str) -> dict | None:
    """Fetch the most recent payment receipt for a session."""
    if not session_id:
        return None
    coll = payment_receipts_collection()
    doc = await coll.find_one(
        {"session_id": session_id, "source": "receipt"},
        sort=[("updated_at", -1)],
    )
    if doc:
        doc.pop("_id", None)
    return doc