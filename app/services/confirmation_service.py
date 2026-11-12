"""
Confirmation Service — centralized booking confirmation notification engine.

Dispatches rich, professional WhatsApp booking confirmation messages to guests
when payment is verified via:
1. Razorpay Webhooks (payment.captured / payment_link.paid / order.paid)
2. Background Gateway Reconciliation Loop
3. In-chat payment verification
4. Offline receipt manual approval
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.db.mongodb import (
    bookings_collection,
    conversations_collection,
    messages_collection,
    payments_collection,
)
from app.models.activity_schemas import ActivityLog
from app.repositories import booking_repository, hotel_repository
from app.repositories.activity_repository import log_activity
from app.services.whatsapp_service import whatsapp_service
from app.utils.helpers import format_quote_date_range

logger = logging.getLogger(__name__)


def _format_date(dt_val: Any) -> str:
    """Format datetime or string YYYY-MM-DD into DD/MM/YYYY."""
    if not dt_val:
        return ""
    if isinstance(dt_val, datetime):
        return dt_val.strftime("%d/%m/%Y")
    try:
        s = str(dt_val).strip()
        if "T" in s:
            s = s.split("T")[0]
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return str(dt_val)


def build_confirmation_message(
    booking: dict[str, Any],
    hotel_info: dict[str, Any] | None = None,
    hotel_policies: dict[str, Any] | None = None,
) -> str:
    """
    Build a comprehensive, premium booking confirmation message for WhatsApp.
    """
    booking_id = booking.get("booking_id") or "UNKNOWN"
    guest_name = booking.get("guest_name") or booking.get("customer_name") or "Valued Guest"
    check_in_raw = booking.get("check_in")
    check_out_raw = booking.get("check_out")
    
    check_in_str = _format_date(check_in_raw)
    check_out_str = _format_date(check_out_raw)
    
    num_nights = booking.get("num_nights") or 1
    night_label = f"{num_nights} night" if num_nights == 1 else f"{num_nights} nights"
    
    adults = booking.get("adults") or 2
    children = booking.get("children") or 0
    guest_str = f"{adults} Adult{'s' if adults != 1 else ''}"
    if children and int(children) > 0:
        guest_str += f", {children} Child{'ren' if int(children) != 1 else ''}"

    # Room details
    rooms = booking.get("rooms") or []
    room_lines = []
    if rooms:
        # Group identical rooms
        from collections import defaultdict
        counts: dict[str, int] = defaultdict(int)
        for r in rooms:
            r_name = str(r.get("room_name") or "Room").strip()
            rate_plan = r.get("rate_plan") or r.get("rate_plan_name") or ""
            key = f"{r_name} ({rate_plan})" if rate_plan else r_name
            counts[key] += 1
        for k, cnt in counts.items():
            prefix = f"{cnt}x " if cnt > 1 else ""
            room_lines.append(f"• {prefix}*{k}*")
    else:
        room_lines.append("• *Standard Room*")

    rooms_text = "\n".join(room_lines)

    # Payment details.
    #
    # `amount_paid` used to default to `total_price`, so a booking row that never
    # recorded a payment printed "Amount Paid: <full total>" and "Balance Due: 0
    # (Fully Paid)". Guest 917738662309 paid Rs 1,110 of Rs 3,700 and got exactly
    # that. Missing is now unknown, and unknown is never printed as paid:
    # `send_booking_confirmation_notification` resolves the figure from the
    # payments ledger before this runs, and if it still cannot, the line says what
    # is due rather than inventing a payment.
    total_price = float(booking.get("total_price") or 0.0)
    raw_paid = booking.get("amount_paid")
    if raw_paid is not None:
        amount_paid = float(raw_paid or 0.0)
        # Recompute rather than trust the stored due: every writer of
        # `amount_due` uses `total - paid`, and rows confirmed before this fix
        # carry a due that was never reduced (PAY-20261025-1C463331: paid
        # Rs 8,260, stored due still the full Rs 27,540).
        amount_due = (
            max(0.0, total_price - amount_paid)
            if total_price > 0
            else float(booking.get("amount_due") or 0.0)
        )
    elif get_settings().CONFIRMATION_SLIP_VERIFIES_PAID_AMOUNT:
        amount_paid = 0.0
        amount_due = float(booking.get("amount_due") or total_price)
    else:
        # Pre-2026-11 behaviour, kept one release as the rollback.
        amount_paid = total_price
        amount_due = float(booking.get("amount_due") or 0.0)

    payment_lines = [f"• Total Amount: ₹{total_price:,.0f}"]
    if amount_paid > 0:
        payment_lines.append(f"• Amount Paid: ₹{amount_paid:,.0f} ✅")
    if amount_due > 0:
        # "Remaining" implies something was paid; only say it when it was.
        label = "Remaining Due at Check-in" if amount_paid > 0 else "Amount to Pay"
        payment_lines.append(f"• {label}: ₹{amount_due:,.0f} ⏳")
    elif amount_paid > 0:
        payment_lines.append("• Balance Due: ₹0 (Fully Paid) ✅")

    payment_text = "\n".join(payment_lines)

    # Hotel & Timing details
    hotel_info = hotel_info or {}
    hotel_policies = hotel_policies or {}
    
    property_data = hotel_info.get("property") or {}
    contact_data = hotel_info.get("contact") or {}
    location_data = hotel_info.get("location") or {}
    
    hotel_name = property_data.get("display_name") or property_data.get("property_name") or hotel_info.get("name") or "Our Hotel"
    address = location_data.get("property_address") or location_data.get("address") or hotel_info.get("address") or ""
    phone = contact_data.get("primary_contact_number") or contact_data.get("phone") or ""

    check_in_time = hotel_policies.get("check_in_time") or property_data.get("check_in") or "01:00 PM"
    check_out_time = hotel_policies.get("check_out_time") or property_data.get("check_out") or "11:00 AM"

    # Assemble Message
    msg = (
        f"🎉 *Booking Confirmed!*\n\n"
        f"Dear *{guest_name}*,\n"
        f"Thank you for choosing *{hotel_name}*! Your reservation has been successfully confirmed.\n\n"
        f"📋 *Reservation Details:*\n"
        f"• *Booking ID:* `{booking_id}`\n"
        f"• *Dates:* {check_in_str} → {check_out_str} ({night_label})\n"
        f"• *Guests:* {guest_str}\n\n"
        f"🏨 *Rooms Booked:*\n"
        f"{rooms_text}\n\n"
        f"💳 *Payment Summary:*\n"
        f"{payment_text}\n\n"
        f"⏰ *Timings:*\n"
        f"• Check-in: {check_in_time}\n"
        f"• Check-out: {check_out_time}\n"
    )

    if address:
        msg += f"\n📍 *Location:*\n{address}\n"
    if phone:
        msg += f"📞 *Front Desk / Support:* {phone}\n"

    msg += "\nWe look forward to welcoming you! If you have any special requests or questions, simply reply to this message. Safe travels! 🌟"

    return msg


async def _ledger_amount_paid(booking_id: str) -> float | None:
    """Sum the payments ledger for a booking, or None if it holds no paid row.

    `booking.amount_paid` is only written by some of the confirmation paths; the
    payment-first drafts (`PAY-...`) never get it, so the slip had nothing real to
    print and defaulted to "fully paid". The ledger is where the gateway records
    the money, and `payment_webhook` reads the same collection.
    """
    try:
        cursor = payments_collection().find(
            {"booking_id": booking_id, "status": {"$in": ["paid", "captured"]}},
            {"amount": 1},
        )
        total = 0.0
        found = False
        async for row in cursor:
            try:
                total += float(row.get("amount") or 0.0)
                found = True
            except (TypeError, ValueError):
                continue
        return total if found else None
    except Exception as exc:  # a slip is not worth failing the confirmation over
        logger.warning(
            "CONFIRMATION_LEDGER_LOOKUP_FAILED | booking_id=%s error=%s",
            booking_id,
            exc,
        )
        return None


async def send_booking_confirmation_notification(
    booking_id: str,
    payment_id: str | None = None,
    amount_paid: float | None = None,
    force_resend: bool = False,
) -> bool:
    """
    Send an automated WhatsApp booking confirmation message to the customer.
    Idempotent: Avoids duplicate messages for the same booking unless force_resend is True.
    """
    if not booking_id:
        logger.warning("CONFIRMATION_NOTIFY_SKIPPED | reason=missing_booking_id")
        return False

    booking = await booking_repository.get_booking(booking_id)
    if not booking:
        logger.warning(f"CONFIRMATION_NOTIFY_SKIPPED | booking_id={booking_id} not found in DB")
        return False

    customer_phone = booking.get("customer_phone") or booking.get("customer_identifier")
    if not customer_phone:
        logger.warning(f"CONFIRMATION_NOTIFY_SKIPPED | booking_id={booking_id} has no customer_phone")
        return False

    hotel_id = booking.get("hotel_id") or 111111

    # Atomic Idempotency Lock: Avoid duplicate sends under concurrent webhook race conditions
    b_coll = bookings_collection()
    if not force_resend:
        now = datetime.now(timezone.utc)
        claim_update = await b_coll.update_one(
            {
                "booking_id": booking_id,
                "$or": [
                    {"confirmation_sent": {"$ne": True}},
                    {"confirmation_sent": False},
                    {"confirmation_sent": None},
                ]
            },
            {
                "$set": {
                    "confirmation_sent": True,
                    "confirmation_sent_at": now,
                    "last_confirmed_payment_id": payment_id,
                }
            }
        )
        if claim_update.modified_count == 0:
            logger.info(f"CONFIRMATION_NOTIFY_IDEMPOTENT | Another process already claimed confirmation for {booking_id}")
            return True

    # If amount_paid is provided, update booking object
    if amount_paid is not None:
        booking["amount_paid"] = amount_paid
        total = float(booking.get("total_price") or 0.0)
        booking["amount_due"] = max(0.0, total - amount_paid)
    elif (
        booking.get("amount_paid") is None
        and get_settings().CONFIRMATION_SLIP_VERIFIES_PAID_AMOUNT
    ):
        # The caller did not pass an amount and the row never stored one — the
        # payment-first drafts. Read the ledger so the slip can print the real
        # figure instead of defaulting to "fully paid".
        ledger_paid = await _ledger_amount_paid(booking_id)
        if ledger_paid is not None:
            booking["amount_paid"] = ledger_paid
            total = float(booking.get("total_price") or 0.0)
            booking["amount_due"] = max(0.0, total - ledger_paid)
            logger.info(
                "CONFIRMATION_PAID_FROM_LEDGER | booking_id=%s paid=%s due=%s",
                booking_id,
                ledger_paid,
                booking["amount_due"],
            )

    # 1. Fetch hotel details & policies for rich message
    hotel_info = None
    hotel_policies = None
    try:
        # The FULL hotel_details document, not get_hotel_contact() — that returns only the
        # `contact` sub-document, so build_confirmation_message's `hotel_info.get("property")`
        # was always None and the confirmation named "Our Hotel" instead of the real property.
        hotel_info = await hotel_repository.get_hotel_details(hotel_id) or {}
        location_doc = await hotel_repository.get_hotel_location(hotel_id) or {}
        if location_doc:
            hotel_info["location"] = location_doc
        hotel_policies = await hotel_repository.get_hotel_policies(hotel_id) or {}
    except Exception as e:
        logger.warning(f"Failed to fetch hotel details for confirmation message: {e}")

    # 2. Format the message
    confirmation_text = build_confirmation_message(
        booking=booking,
        hotel_info=hotel_info,
        hotel_policies=hotel_policies,
    )

    # 3. Send WhatsApp message
    send_ok = whatsapp_service.send_message(
        to_number=customer_phone,
        message_body=confirmation_text,
        hotel_id=hotel_id,
    )

    now = datetime.now(timezone.utc)

    if send_ok:
        logger.info(f"CONFIRMATION_NOTIFY_SUCCESS | booking_id={booking_id} phone={customer_phone[:4]}***")
        
        # 4. Mark confirmation_sent in bookings collection
        b_coll = bookings_collection()
        update_fields: dict[str, Any] = {
            "confirmation_sent": True,
            "confirmation_sent_at": now,
            "updated_at": now,
        }
        if payment_id:
            update_fields["last_confirmed_payment_id"] = payment_id
        if amount_paid is not None:
            update_fields["amount_paid"] = amount_paid

        await b_coll.update_one(
            {"booking_id": booking_id},
            {"$set": update_fields},
        )

        # 5. Sync active conversation session (so bot knows booking is confirmed)
        conv_id = booking.get("conversation_id")
        conv_coll = conversations_collection()
        
        try:
            conv_docs = []
            if conv_id:
                doc = await conv_coll.find_one({"conversation_id": conv_id})
                if doc:
                    conv_docs.append(doc)
            
            if not conv_docs:
                b_match = await conv_coll.find_one({
                    "$or": [
                        {"booking_id": booking_id},
                        {"workflow_state.booking_id": booking_id},
                    ]
                })
                if b_match:
                    conv_docs.append(b_match)

            if not conv_docs:
                str_hid = str(hotel_id)
                variants = [str_hid, int(str_hid)] if str_hid.isdigit() else [str_hid]
                cursor = conv_coll.find({
                    "hotel_id": {"$in": variants},
                    "$or": [
                        {"guest_id": booking.get("guest_id")},
                        {"customer_identifier": customer_phone},
                        {"customer_phone": customer_phone},
                        {"workflow_state.customer_identifier": customer_phone},
                    ],
                }).sort("last_message_at", -1).limit(3)
                conv_docs = await cursor.to_list(length=3)

            for c_doc in conv_docs:
                curr_ws = c_doc.get("workflow_state")
                if not isinstance(curr_ws, dict):
                    curr_ws = {}
                curr_ws["booking_stage"] = "confirmed"
                curr_ws["payment_status"] = "paid"
                curr_ws["amount_paid"] = booking.get("amount_paid")
                curr_ws["amount_due"] = booking.get("amount_due", 0.0)
                curr_ws["booking_id"] = booking_id

                await conv_coll.update_one(
                    {"_id": c_doc["_id"]},
                    {"$set": {
                        "status": "resolved",
                        "booking_id": booking_id,
                        "workflow_state": curr_ws,
                        "updated_at": now,
                    }}
                )
        except Exception as e:
            logger.warning(f"Error updating conversation state after confirmation: {e}")

        # 6. Append message to messages transcript for CRM / Admin visibility
        try:
            msg_coll = messages_collection()
            resolved_conv_id = conv_id or f"CONV-{booking_id}"
            await msg_coll.insert_one({
                "message_id": f"MSG-CONFIRM-{booking_id}-{int(now.timestamp())}",
                "conversation_id": resolved_conv_id,
                "hotel_id": str(hotel_id),
                "sender": "bot",
                "content": confirmation_text,
                "media": None,
                "created_at": now,
            })
        except Exception as e:
            logger.warning(f"Error logging confirmation message transcript: {e}")

        # 7. Log system activity
        await log_activity(ActivityLog(
            hotel_id=str(hotel_id),
            activity_type="BOOKING_CONFIRMED",
            reference_id=booking_id,
            booking_id=booking_id,
            customer_phone=customer_phone,
            customer_name=booking.get("guest_name"),
            description=f"Booking confirmation message automatically dispatched to {customer_phone}."
        ))

        return True
    else:
        logger.error(f"CONFIRMATION_NOTIFY_FAILED | whatsapp_service.send_message returned False for {booking_id}")
        b_coll = bookings_collection()
        await b_coll.update_one(
            {"booking_id": booking_id},
            {"$set": {"confirmation_sent": False, "confirmation_failed_at": now}},
        )
        return False
