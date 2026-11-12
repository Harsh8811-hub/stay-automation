"""Inventory hold + payment link generation."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import cast

from app.config.hotel_config import HotelConfig
from app.config.settings import get_settings
from app.prompts.templates import text_card
from app.repositories.booking_repository import create_booking, get_booking
from app.repositories.payment_gateway_repository import get_hotel_upi_details
from app.services.booking_engine import (
    build_booking_document,
    generate_booking_id,
    generate_payment_reference,
)
from app.services.feature_gate import FeatureGate
from app.services.inventory_service import get_available_rooms
from app.services.payment_service import create_payment_link
from app.services import qr_payment_service
from app.workflows.nodes.payment.constants import MIN_ADVANCE_PERCENT
from app.workflows.session_mapper import state_to_session
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IMAGE_URL = re.compile(r"\.(?:png|jpe?g|webp|gif)(?:\?|$)", re.IGNORECASE)


def _looks_like_an_image(url: str | None) -> bool:
    """Whether `url` can be sent as a WhatsApp image.

    Razorpay's `createqr_guest` returns the same `rzp.io/rzp/...` link for both
    `image_url` and `qr_link`. That is a web page, not a picture, and a chat
    client handed a web page as an image renders a broken attachment. Send the
    picture when the URL is one; otherwise send the link as text.
    """
    return bool(url) and bool(_IMAGE_URL.search(url or ""))


def _iso_date(value: object) -> str | None:
    """
    A real `YYYY-MM-DD` date, or `None`.

    `normalize_date` deliberately returns free text untouched — it refuses to truncate
    "next tuesday" to "next" — so it answers with a *string* for a value it could not
    parse. Comparing those two strings would read "next tuesday" against "2026-11-21" as
    two different stays and refuse to suppress, which is the very bug this guards against.
    Only a genuine date counts as evidence of a different stay.
    """
    from app.utils.helpers import normalize_date

    text = normalize_date(value) if isinstance(value, str) else None
    return text if text and _ISO_DATE.match(text) else None


async def _already_confirmed_reply(
    state: ConversationState, *, hotel_id: int
) -> dict | None:
    """
    What to say *instead of* asking for money, when the CRM says staff already confirmed.

    `None` means "carry on and ask for payment" — the lookup is off, the CRM holds nothing
    on this guest, or its confirmation names a different stay than the one being priced.

    Two rules this has to get right:

    1. **Suppress, never confirm.** The CRM holds no payment record, no amount paid and no
       booking id, so a wrong label must cost the guest a wait — never a room. We do not
       touch `payment_status` or mark anything paid here.
    2. **Match the stay.** The confirmation is keyed on the phone number, not on a
       booking, and it is sticky (measured: 70 of 167 unrefreshed, oldest 11 days). A
       returning guest's *new* enquiry therefore inherits their old confirmation. When both
       sides name a check-in date and they disagree, this is a different stay and the guest
       does owe for it — so we do not suppress.

    Never raises: the repository swallows its own failures and answers `None`.
    """
    from app.repositories.crm_confirmation_repository import lookup_crm_confirmation
    phone = state.get("customer_identifier") or state.get("guest_phone")
    found = await lookup_crm_confirmation(hotel_id, phone)
    if found is None or not found.confirmed:
        return None

    session_in = _iso_date(state.get("check_in"))
    crm_in = _iso_date(found.check_in)
    if session_in and crm_in and session_in != crm_in:
        logger.info(
            "CRM says confirmed but for a different stay (hotel=%s, crm=%s, session=%s) — "
            "asking for payment; the confirmation belongs to an earlier booking",
            hotel_id,
            crm_in,
            session_in,
        )
        return None

    if found.has_dates:
        return {
            "next_action": "payment_staff_confirmed",
            "bot_response": text_card(
                "Our team has already confirmed this booking, so there's nothing to pay "
                "right now — you're all set. I won't send a payment link unless you ask."
            ),
        }

    # Staff confirmed *something* for this guest, but the CRM carries no dates, so we
    # cannot tie it to the stay being priced. Still suppress the demand — dunning a guest
    # who has paid is the reported bug — and put a human on it rather than going quiet.
    from app.services.availability_guard import (
        AvailabilityVerdict,
        escalation_updates,
        raise_availability_ticket,
    )

    verdict = AvailabilityVerdict(
        "crm_confirmed_unmatched",
        "CRM reports CONFIRMED_BOOKING for this guest but carries no dates to match "
        "against the stay being priced — needs a human to confirm which booking it is",
    )
    # In this turn: routing only reaches handle_escalation on the next dispatch, so a guest
    # who never replies would otherwise get neither a ticket nor an answer.
    await raise_availability_ticket(state, verdict)
    return {
        "next_action": "payment_staff_confirmed",
        "bot_response": text_card(
            "Our team already has a confirmation on file for you, so I won't ask you to "
            "pay again. I've asked them to check it against this stay and they'll confirm "
            "with you shortly."
        ),
        **escalation_updates(verdict),
    }


async def generate_payment_link_flow(
    state: ConversationState,
    *,
    hotel_id: int,
    total: float,
    payment_status: str | None,
    existing_link: str | None,
    invalid_payment_msg: str | None,
) -> ConversationState:
    """
    Generate a payment link, then create a pending inventory hold.

    Order matters: never lock inventory if link creation fails.
    """
    from app.services.override_resolver import resolve_effective_total, skip_payment_ask

    # Staff conversation truth for payable amount
    total = float(resolve_effective_total(state, total) or total or 0)

    if skip_payment_ask(state):
        return cast(ConversationState, {
            **state,
            "total_price": total,
            "next_action": "payment_staff_confirmed",
            "bot_response": text_card(
                "Our team has already confirmed payment arrangements for this stay. "
                "I won't generate a new payment link unless you ask."
            ),
        })

    # The CRM's own view of this guest, read before any side effect. Staff confirm in the
    # Control Centre and that confirmation stops there — nothing carries it into the bot's
    # `bookings`, so without this read a guest who has already paid is asked for the
    # advance again (measured on hotel 111111, 2026-10-11). Off unless
    # CRM_CONFIRMATION_LOOKUP_ENABLED is set.
    crm_reply = await _already_confirmed_reply(state, hotel_id=hotel_id)
    if crm_reply is not None:
        return cast(ConversationState, {**state, "total_price": total, **crm_reply})

    is_requesting_new_amount = state.get("partial_payment_amount") is not None
    if existing_link and payment_status != "paid" and not is_requesting_new_amount:
        from app.prompts.templates import payment_pending
        return cast(ConversationState, {
            **state,
            "next_action": "payment_pending",
            "bot_response": payment_pending(existing_link),
        })

    if total <= 0:
        return cast(ConversationState, {
            **state,
            "next_action": "need_quote",
            "bot_response": text_card("Let me calculate the price first..."),
        })

    selected_rooms = state.get("selected_rooms", [])
    if not selected_rooms:
        return cast(ConversationState, {
            **state,
            "next_action": "need_quote",
            "bot_response": text_card("No rooms selected."),
        })

    if not state.get("guest_name"):
        state["guest_name"] = state.get("customer_name") or "Guest"

    # Feature gate before any side effects
    config_dict = state.get("hotel_config")
    config = HotelConfig(**config_dict) if config_dict else HotelConfig(hotel_id=hotel_id)
    gate_for_link = FeatureGate(config, session_id=state.get("session_id"))

    if not gate_for_link.can_direct_book():
        return cast(ConversationState, {
            **state,
            "next_action": "payment_pending",
            "bot_response": text_card(
                "Online payments are currently disabled. Our team will contact you shortly "
                "to complete your booking securely."
            ),
        })

    # Inventory pre-check (B2). Classify before believing a shortfall: an empty
    # availability answer and an int/str id split both used to read as "sold out" and
    # cost real bookings. See app/services/availability_guard.py for the three modes.
    from app.services.availability_guard import (
        BLOCKED_GUEST_COPY,
        classify_availability,
        escalation_updates,
        raise_availability_ticket,
    )

    available_rooms = await get_available_rooms(
        hotel_id, state.get("check_in"), state.get("check_out")
    )
    verdict = classify_availability(selected_rooms, available_rooms)

    if not verdict.ok:
        logger.warning(
            "Payment blocked before link creation: %s — %s "
            "(hotel=%s, selection=%s, booking=%s)",
            verdict.status,
            verdict.reason,
            hotel_id,
            selected_rooms,
            state.get("booking_id"),
        )
        # Raise the ticket in this turn — routing only reaches handle_escalation on the
        # next dispatch, so a guest who never replies would otherwise never get one.
        await raise_availability_ticket(state, verdict)
        return cast(ConversationState, {
            **state,
            "booking_stage": "collecting",
            "payment_link": None,
            "payment_link_id": None,
            "payment_link_amount": None,
            # selected_rooms is deliberately NOT cleared: the quote is still valid, the
            # selection is the guest's, and wiping it is half the harm of the old branch.
            "bot_response": text_card(BLOCKED_GUEST_COPY),
            "next_action": "rooms_unavailable",
            **escalation_updates(verdict),
        })

    booking_id = state.get("booking_id")
    if get_settings().BOOKING_ID_ON_PAYMENT_PROOF_ENABLED:
        # Nothing has been paid yet, so there is no booking to name. A `PAY-…`
        # reference keys the gateway order and the inventory hold; the `BK-…`
        # id is minted when the payment screenshot arrives (see
        # `payment/receipt.py`) or, for a gateway-only payment, in
        # `confirm_booking`. The guest is never shown the reference.
        payment_reference = state.get("payment_reference") or generate_payment_reference()
        hold_reference = payment_reference
    else:
        booking_id = booking_id or generate_booking_id()
        payment_reference = None
        hold_reference = booking_id

    # Partial / advance amount
    partial_amount = state.get("partial_payment_amount")
    advance_amount = round((total * MIN_ADVANCE_PERCENT) / 10) * 10

    if partial_amount is not None:
        if partial_amount < advance_amount:
            return cast(ConversationState, {
                **state,
                "booking_id": booking_id,
                "payment_reference": payment_reference,
                "partial_payment_amount": None,
                "bot_response": text_card(
                    f"⚠️ The minimum advance payment is **30%** of the total amount.\n\n"
                    f"💰 Total: ₹{total:,.0f}\n"
                    f"📌 Minimum advance: ₹{advance_amount:,.0f}\n\n"
                    f"You mentioned ₹{partial_amount:,.0f}, which is below the minimum.\n\n"
                    f"Please pay at least **₹{advance_amount:,.0f}** as advance, or type **'confirm'** "
                    f"to proceed with the standard 30% advance."
                ),
                "next_action": "payment_pending",
                "booking_stage": "quote_shown",
            })
        if partial_amount >= total:
            pay_amount = total
        else:
            pay_amount = partial_amount
    else:
        # Default: 30% advance payment link
        pay_amount = advance_amount

    upi_info = await get_hotel_upi_details(
        hotel_id=hotel_id,
        hotel_name=state.get("hotel_name", "Hotel"),
        amount=pay_amount,
        booking_id=hold_reference,
    )

    # 1) Create payment link FIRST (no hold yet)
    create_result = await create_payment_link(hotel_id, hold_reference, pay_amount)
    if not create_result.ok:
        # If the external gateway is offline/erroring, but hotel has active UPI Direct,
        # fallback to UPI Direct instead of stranding the guest.
        if upi_info.get("connected") and upi_info.get("vpa"):
            link = None
            payment_link_id = f"upi_{hold_reference}"
            logger.warning(
                f"PAYMENT_GATEWAY_OFFLINE | hotel={hotel_id} booking={hold_reference} "
                f"err={create_result.error}; gracefully falling back to connected UPI ({upi_info.get('vpa')})"
            )
        else:
            logger.error(
                f"PAYMENT_LINK_FAILED | hotel={hotel_id} booking={hold_reference} "
                f"err={create_result.error} — no UPI fallback available"
            )
            return cast(ConversationState, {
                **state,
                "booking_id": booking_id,
                "payment_reference": payment_reference,
                "bot_response": text_card(
                    "I'm sorry, I couldn't generate a payment link at the moment. Please try again later."
                ),
                "next_action": "payment_error",
            })
    else:
        link = create_result.url
        payment_link_id = create_result.payment_link_id

    # A Razorpay locked-amount UPI QR, alongside the link. Unlike the static
    # `upi://` picture this replaces, the QR is an order: it carries our
    # booking reference in its notes, so Razorpay's `qr_code.credited` webhook
    # can name the hold and confirm the booking with nobody touching a screen.
    # Failure here is not fatal — the payment link still works — so it is
    # logged and the turn continues.
    qr_result = None
    if qr_payment_service.enabled():
        qr_result = await qr_payment_service.create_guest_qr(
            hotel_id,
            amount=pay_amount,
            booking_id=hold_reference,
            customer_name=state.get("guest_name") or state.get("customer_name"),
            customer_phone=state.get("customer_identifier") or state.get("guest_phone"),
        )
        if not qr_result.ok:
            logger.warning(
                "QR_CREATE_FAILED | hotel=%s booking=%s err=%s — continuing with "
                "the payment link only",
                hotel_id, hold_reference, qr_result.error,
            )

    # 2) Create pending hold ONLY after link (or UPI fallback) succeeds
    from app.repositories.booking_repository import create_booking, get_booking

    existing_booking = await get_booking(hold_reference)
    if not existing_booking:
        temp_session = state_to_session(
            state,
            hotel_id=hotel_id,
            booking_id=hold_reference,
            payment_status="pending",
            payment_link=link,
            payment_link_id=payment_link_id,
            payment_link_amount=pay_amount,
            amount_paid=0.0,
            amount_due=total,
            total_price=total,
            selected_rooms=selected_rooms,
            booking_stage="awaiting_payment",
        )
        hold_doc = build_booking_document(temp_session)
        hold_doc["booking_status"] = "pending_payment"
        hold_doc["payment_link"] = link
        hold_doc["payment_link_id"] = payment_link_id
        hold_doc["payment_link_amount"] = pay_amount
        if qr_result is not None and qr_result.ok:
            # Persisted on the row, not only in state: the webhook and the
            # reconciliation sweep both read the booking document, and neither
            # runs inside this guest's conversation.
            hold_doc["qr_order_id"] = qr_result.order_id
            hold_doc["qr_id"] = qr_result.qr_id
            hold_doc["qr_link"] = qr_result.qr_link or qr_result.image_url
        hold_doc["hold_expires_at"] = datetime.utcnow() + timedelta(minutes=30)
        if payment_reference:
            # Marks the row as provisional. `reassign_booking_id` renames it to
            # the real id once the guest shows proof of payment, and the marker
            # makes that row recognisable in between.
            hold_doc["payment_reference"] = payment_reference
        await create_booking(hold_doc)
        logger.info(
            f"Created pending hold for booking {hold_reference} "
            f"(plink={payment_link_id}, expires in 30m)"
        )

    room_summary_lines = []
    if selected_rooms:
        counts: dict[str, int] = {}
        for r in selected_rooms:
            name = str(r.get("room_name") or "Room").strip().upper()
            counts[name] = counts.get(name, 0) + 1
        room_summary_lines.append("🏨 **Booked Room(s):** " + ", ".join(f"{cnt}x {n}" if cnt > 1 else n for n, cnt in counts.items()))

    room_names_str = ", ".join(room_summary_lines) if room_summary_lines else "Your Room Selection"

    is_partial = pay_amount < total
    advance_pct = int(round((pay_amount / total) * 100)) if total > 0 else 30
    remaining = max(0.0, total - pay_amount)

    from app.utils.template_renderers import button_card

    # Under BOOKING_ID_ON_PAYMENT_PROOF_ENABLED there is no booking yet, so the
    # message carries no number. Printing the `PAY-…` reference instead would
    # be worse than printing nothing: the guest would quote it to staff as a
    # booking id.
    booking_id_line = f"*Booking ID:* {booking_id}\n" if booking_id else ""

    upi_text_part = ""
    if upi_info.get("connected") and upi_info.get("vpa"):
        vpa_id = upi_info["vpa"]
        upi_text_part = (
            f"\n\n📱 *Or Pay directly via UPI (GPay / PhonePe / Paytm / BHIM):*\n"
            f"• *UPI ID / VPA:* `{vpa_id}`\n"
            f"• *Scan QR Code:* Scan the QR code below in any UPI app to pay ₹{pay_amount:,.0f}."
        )

    if link:
        payment_instructions = (
            f"💳 *Secure Payment Link (Cards, UPI, Netbanking):*\n"
            f"{link}"
            f"{upi_text_part}\n\n"
            f"Your booking will be confirmed automatically once payment completes.\n\n"
            f"_This link is valid for 15 minutes._"
        )
        link_text = (
            f"{booking_id_line}"
            f"*Room:* {room_names_str}\n"
            f"*Total:* Rs {total:,.0f}\n"
            f"*Advance ({advance_pct}%):* Rs {pay_amount:,.0f}\n"
            f"*Due at check-in:* Rs {remaining:,.0f}\n\n"
            f"{payment_instructions}"
        )
        link_response = button_card(link_text, [{"id": "payment_paid", "title": "I've paid"}])
    else:
        vpa_id = upi_info.get("vpa", "hotelsahu@ybl")
        payment_instructions = (
            f"📱 *Pay directly via UPI (GPay / PhonePe / Paytm / BHIM):*\n"
            f"• *UPI ID / VPA:* `{vpa_id}`\n"
            f"• *Advance to Pay:* Rs {pay_amount:,.0f}\n\n"
            f"⚠️ *Important:* Please complete payment and share your screenshot or transaction ID here.\n"
            f"Our front desk will verify and confirm your reservation shortly! 🙏\n\n"
            f"_Your room hold is valid for 30 minutes._"
        )
        link_text = (
            f"{booking_id_line}"
            f"*Room:* {room_names_str}\n"
            f"*Total:* Rs {total:,.0f}\n"
            f"*Advance ({advance_pct}%):* Rs {pay_amount:,.0f}\n"
            f"*Due at check-in:* Rs {remaining:,.0f}\n\n"
            f"{payment_instructions}"
        )
        link_response = button_card(link_text, [{"id": "payment_paid", "title": "I've paid"}])

    # Attach the Razorpay QR. It is the only payment route that confirms
    # itself, so it goes out on both the gateway-link path and the UPI-fallback
    # path. When Razorpay hands back a page rather than a picture, we send the
    # link as a line of text instead of a broken image attachment; the guest
    # still scans by opening it.
    if qr_result is not None and qr_result.ok:
        qr_target = qr_result.image_url or qr_result.qr_link
        if _looks_like_an_image(qr_target):
            link_response.append({
                "type": "image",
                "url": qr_target,
                "image_url": qr_target,
                "media_url": qr_target,
                "caption": f"Scan to pay advance ₹{pay_amount:,.0f} via any UPI App",
            })
        elif qr_result.qr_link:
            link_response += text_card(
                f"📲 *Or pay by UPI QR:* {qr_result.qr_link}\n"
                f"Amount is locked to ₹{pay_amount:,.0f}. Your booking confirms "
                f"automatically once the payment is received."
            )
    # Static `upi://` QR — the old manual route. Only when Razorpay gave us no
    # QR at all, so a guest is never left without a way to pay.
    elif upi_info.get("connected") and upi_info.get("qr_code_url"):
        link_response.append({
            "type": "image",
            "url": upi_info["qr_code_url"],
            "image_url": upi_info["qr_code_url"],
            "media_url": upi_info["qr_code_url"],
            "caption": f"Scan to pay advance ₹{pay_amount:,.0f} via any UPI App (VPA: {upi_info.get('vpa', '')})",
        })

    if invalid_payment_msg:
        link_response = text_card(invalid_payment_msg) + link_response

    return cast(ConversationState, {
        **state,
        "booking_id": booking_id,
        "payment_reference": payment_reference,
        "payment_link": link,
        "payment_link_id": payment_link_id,
        "payment_link_amount": pay_amount,
        "qr_order_id": (qr_result.order_id if qr_result and qr_result.ok else None),
        "qr_id": (qr_result.qr_id if qr_result and qr_result.ok else None),
        "qr_link": (
            (qr_result.qr_link or qr_result.image_url)
            if qr_result and qr_result.ok
            else None
        ),
        "payment_status": "pending",
        "booking_stage": "awaiting_payment",
        "amount_due": max(0.0, total - float(state.get("amount_paid") or 0.0)),
        "partial_payment_amount": None,
        "bot_response": link_response,
        "next_action": "payment_sent",
    })
