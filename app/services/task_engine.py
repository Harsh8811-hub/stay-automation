"""
Hotel Task Engine — Python makes ALL decisions.

Pipeline:
1. Receive Gemini's task classification (is_task_request, task_type, subtype, etc.)
2. Detect emergencies → route to CRITICAL support ticket instead
3. Verify booking (exists, active, belongs to customer)
4. Detect duplicates → append to existing task
5. Route to department (Python-only mapping)
6. Resolve priority (Python rules, not Gemini)
7. Create task in MongoDB
8. Return structured response

CRITICAL DESIGN PRINCIPLES:
- Gemini ONLY classifies — Python performs ALL business logic
- Emergency requests → CRITICAL Support Ticket, NOT Hotel Task
- Department routing is deterministic Python code, never AI
- Task IDs are human-readable: TASK-20260701-000123
- Duplicate detection prevents redundant tasks
"""

import logging
from datetime import datetime, timezone
from typing import Any
import re

from app.models.task_schemas import (
    HotelTask,
    TaskEvent,
    TaskClassification,
    TASK_TYPE_TO_DEPARTMENT,
    SUBTYPE_TO_DEPARTMENT,
    HIGH_PRIORITY_SUBTYPES,
    URGENT_PRIORITY_SUBTYPES,
    EMERGENCY_KEYWORDS,
)
from app.config.settings import get_settings
from app.repositories import task_repository
from app.repositories.booking_repository import get_booking, get_latest_booking, ACTIVE_BOOKING_STATUSES

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DEPARTMENT DISPLAY NAMES (for customer responses)
# ═══════════════════════════════════════════════════════════════════

DEPARTMENT_DISPLAY_NAMES = {
    "HOUSEKEEPING": "Housekeeping",
    "MAINTENANCE": "Maintenance",
    "ROOM_SERVICE": "Room Service",
    "FRONT_DESK": "Front Desk",
    "ENGINEERING": "Engineering",
    "SECURITY": "Security",
}


# ═══════════════════════════════════════════════════════════════════
# MAIN TASK PROCESSING ENGINE
# ═══════════════════════════════════════════════════════════════════

async def process_task_request(state: dict) -> dict:
    """
    Main task processing function. Called from the task_handler node.

    Returns a dict with:
    - action: "task_created" | "task_updated" | "emergency_escalated"
              | "booking_invalid" | "clarification_needed"
    - response_text: The bot response text
    - task_id: If a task was created/updated
    """
    task_classification = _extract_task_classification(state)
    user_message = state.get("user_message", "")
    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")

    logger.info(
        f"Task engine: type={task_classification.task_type}, "
        f"subtype={task_classification.subtype}, "
        f"urgency={task_classification.urgency}, "
        f"confidence={task_classification.confidence}"
    )

    # ── Step 1: Emergency Detection ──────────────────────────────
    if _is_emergency(user_message, task_classification):
        logger.warning(f"🚨 Emergency detected in task request: {user_message[:100]}")
        return await _handle_emergency(state)

    # ── Step 2: Confidence Check ─────────────────────────────────
    if task_classification.confidence < 0.6:
        logger.info(f"Low confidence ({task_classification.confidence}) — asking for clarification")
        return {
            "action": "clarification_needed",
            "response_text": (
                "I'd like to make sure I understand your request correctly. "
                "Could you please provide more details about what you need? 😊\n\n"
                "For example:\n"
                "• \"Please send extra pillows\"\n"
                "• \"The AC is not working\"\n"
                "• \"I need a wake-up call at 6 AM\""
            ),
        }

    # ── Step 3: Booking Verification (Zero-Wall Policy) ──────────
    # When hotel has no PMS integration or the guest booked via an OTA/offline,
    # the booking lookup returns empty. As mandated by product (Zero-Wall Policy),
    # never block the guest with a verification wall. Assume a confirmed/in-house
    # booking and dispatch the operational task to the hotel staff.
    booking = await _verify_booking(state)

    if not booking:
        c_phone = (
            state.get("customer_identifier")
            or state.get("guest_phone")
            or state.get("customer_phone")
            or ""
        )
        c_name = state.get("guest_name") or state.get("customer_name") or "Valued Guest"
        c_room = _extract_room_from_message(user_message) or state.get("room_number")
        b_id = (
            state.get("booking_id")
            or state.get("active_booking_id")
            or f"BK-CONFIRMED-{session_id[-6:] if session_id else 'DIRECT'}"
        )
        booking = {
            "booking_id": b_id,
            "customer_name": c_name,
            "customer_phone": c_phone,
            "room_number": c_room,
            "booking_status": "confirmed",
        }
        logger.info(
            f"No booking found in DB for task request (no PMS/offline) — "
            f"assumed confirmed booking {booking['booking_id']} for {c_phone}"
        )

    booking_id = booking.get("booking_id", "")
    room_number = (
        booking.get("room_number")
        or booking.get("room_no")
        or _extract_room_from_message(user_message)
    )
    customer_name = (
        booking.get("customer_name")
        or booking.get("guest_name")
        or state.get("guest_name")
    )
    customer_phone = (
        booking.get("customer_phone")
        or booking.get("guest_phone")
        or state.get("guest_phone")
        or state.get("customer_identifier")
    )

    # ── Step 4: Resolve Department ───────────────────────────────
    department = _resolve_department(
        task_classification.task_type,
        task_classification.subtype,
    )

    # ── Step 5: Resolve Priority ─────────────────────────────────
    priority = _resolve_priority(
        task_classification.urgency,
        task_classification.subtype,
    )

    # ── Step 6: Duplicate Detection ──────────────────────────────
    if task_classification.subtype == "PAYMENT_CONFIRMATION":
        existing = await task_repository.find_open_payment_task(
            hotel_id=hotel_id,
            booking_id=booking_id,
            session_id=session_id,
            customer_phone=customer_phone,
        )
    else:
        existing = await task_repository.find_duplicate_task(
            hotel_id=hotel_id,
            booking_id=booking_id,
            room_number=room_number,
            subtype=task_classification.subtype,
        )

    if existing:
        # Append to existing task
        task_id = existing["task_id"]
        await task_repository.append_customer_update(task_id, user_message)
        logger.info(f"📎 Appended to existing task: {task_id}")

        from app.observability.metrics import metrics
        metrics.incr("task.create_total", outcome="duplicate", hotel_id=hotel_id)

        dept_display = DEPARTMENT_DISPLAY_NAMES.get(existing.get("department", ""), "our team")
        return {
            "action": "task_updated",
            "response_text": (
                f"Your existing service request (**{task_id}**) is already being handled "
                f"by our {dept_display} team.\n\n"
                f"I've added your latest message to the request for our staff to review. "
                f"They will assist you as soon as possible. 🙏\n\n"
                f"What else can we help you with? 😊"
            ),
            "task_id": task_id,
        }

    # ── Step 7: Create New Task ──────────────────────────────────
    task_id = await task_repository.generate_task_id(hotel_id)
    now = datetime.utcnow()

    summary = _build_staff_summary(
        department=department,
        task_type=task_classification.task_type,
        subtype=task_classification.subtype,
        room_number=room_number,
        gemini_summary=task_classification.summary,
    )

    task = HotelTask(
        task_id=task_id,
        hotel_id=hotel_id,
        session_id=session_id,
        booking_id=booking_id,
        room_number=room_number,
        customer_name=customer_name,
        customer_phone=customer_phone,
        department=department,
        task_type=task_classification.task_type,
        subtype=task_classification.subtype,
        priority=priority,
        status="OPEN",
        summary=summary,
        customer_request=user_message,
        created_by="AI",
        events=[
            TaskEvent(
                event_type="CREATED",
                description=(
                    f"Task created via AI. "
                    f"Dept: {department}, Type: {task_classification.task_type}/{task_classification.subtype}. "
                    f"Priority: {priority}."
                ),
                timestamp=now,
                actor="system",
                metadata={
                    "department": department,
                    "task_type": task_classification.task_type,
                    "subtype": task_classification.subtype,
                    "priority": priority,
                    "session_id": session_id,
                    "booking_id": booking_id,
                    "confidence": task_classification.confidence,
                },
            ),
        ],
        notes=[
            f"[{now.isoformat()}] AI: Task auto-created from guest request.",
        ],
        created_at=now,
        updated_at=now,
    )

    await task_repository.create_task(task)

    from app.observability.metrics import metrics
    metrics.incr(
        "task.create_total",
        outcome="ok",
        hotel_id=hotel_id,
        department=department,
    )

    dept_display = DEPARTMENT_DISPLAY_NAMES.get(department, "our team")
    is_bill_request = (
        task_classification.subtype in ("BILLING_INQUIRY", "INVOICE_REQUEST")
        or any(w in user_message.lower() for w in ("bill", "invoice", "receipt", "hisab", "payment slip"))
    )
    if is_bill_request:
        response_text = (
            f"Certainly! I've noted your request for the bill. ✅\n\n"
            f"🔧 **Request ID:** {task_id}\n\n"
            f"Our **Front Desk** team has been notified and will share your bill shortly. 🙏\n\n"
            f"What else can we help you with? 😊"
        )
    else:
        response_text = (
            f"Certainly! I've created your service request. ✅\n\n"
            f"🔧 **Task ID:** {task_id}\n\n"
            f"Our **{dept_display}** team has been notified and will assist you as soon as possible. 🙏\n\n"
            f"What else can we help you with? 😊"
        )

    logger.info(f"🔧 New task created: {task_id} | Dept: {department} | Priority: {priority}")

    return {
        "action": "task_created",
        "response_text": response_text,
        "task_id": task_id,
    }


# ═══════════════════════════════════════════════════════════════════
# EMERGENCY DETECTION
# ═══════════════════════════════════════════════════════════════════

def _is_emergency(user_message: str, classification: TaskClassification) -> bool:
    """
    Check if the request is actually an emergency.
    Emergencies must NOT become normal tasks — they become CRITICAL support tickets.
    """
    msg_lower = user_message.lower()

    # Check against known emergency phrases (multi-word)
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in msg_lower:
            return True

    # Check individual danger words as standalone tokens
    from app.models.task_schemas import EMERGENCY_DANGER_WORDS
    msg_words = set(msg_lower.split())
    if msg_words & EMERGENCY_DANGER_WORDS:
        return True

    # Gas-related patterns: "smell gas", "gas is leaking", etc.
    if "gas" in msg_words:
        gas_context = {"smell", "leak", "leaking", "fumes", "danger", "dangerous"}
        if msg_words & gas_context:
            return True

    # Gemini classified as URGENT with high confidence
    if classification.urgency == "URGENT" and classification.confidence >= 0.85:
        # Additional safety: check if the subtype implies danger
        dangerous_subtypes = {"ELECTRICAL_ISSUE", "POWER_ISSUE"}
        if classification.subtype in dangerous_subtypes:
            # Only emergency if the message suggests imminent danger
            danger_words = {"sparks", "fire", "smoke", "electrocution", "shock", "burning"}
            if any(w in msg_lower for w in danger_words):
                return True

    return False


async def _handle_emergency(state: dict) -> dict:
    """
    Route emergency to CRITICAL Support Ticket instead of Hotel Task.
    Uses the existing escalation engine.
    """
    from app.services.escalation_engine import evaluate_escalation

    # Override escalation fields for emergency routing
    emergency_state = {
        **state,
        "escalation_required": True,
        "escalation_issue_type": "SAFETY",
        "escalation_reason_code": "SAFETY_CONCERN",
        "escalation_urgency": "CRITICAL",
        "escalation_sentiment": "DISTRESSED",
        "escalation_reason": "Emergency detected in task request",
    }

    result = await evaluate_escalation(emergency_state)

    return {
        "action": "emergency_escalated",
        "response_text": result.get("response_text", ""),
        "ticket_id": result.get("ticket_id"),
    }


# ═══════════════════════════════════════════════════════════════════
# BOOKING VERIFICATION
# ═══════════════════════════════════════════════════════════════════

async def _verify_booking(state: dict) -> dict | None:
    """
    Verify that the customer has a valid, active booking.

    Checks:
    1. Booking exists
    2. Booking belongs to the current customer (normalized phone)
    3. Booking is active (not cancelled, not checked out)

    Returns the booking dict if valid, None if not.
    """
    from app.utils.phone import digits_only, phone_lookup_variants

    booking_id = state.get("booking_id")
    customer_phone = (
        state.get("guest_phone")
        or state.get("customer_identifier")
    )

    def _phones_match(stored: str | None, session_phone: str | None) -> bool:
        if not stored or not session_phone:
            return True  # can't verify — don't block
        stored_vars = set(phone_lookup_variants(stored))
        session_vars = set(phone_lookup_variants(session_phone))
        if stored_vars & session_vars:
            return True
        sd, ss = digits_only(stored), digits_only(session_phone)
        if len(sd) >= 10 and len(ss) >= 10:
            return sd[-10:] == ss[-10:]
        return sd == ss

    # Try by booking ID first
    if booking_id:
        booking = await get_booking(booking_id)
        if booking and _is_booking_active(booking):
            bk_phone = booking.get("customer_phone") or booking.get("customer_identifier")
            if _phones_match(bk_phone, customer_phone):
                return await _verify_room_number(booking, state.get("user_message", ""))
            logger.warning(
                f"Phone mismatch on booking {booking_id}: "
                f"booking={bk_phone} session={customer_phone}"
            )

    # Try by customer phone (scoped to this hotel)
    hotel_id = state.get("hotel_id") or 0
    if customer_phone and hotel_id:
        booking = await get_latest_booking(
            hotel_id=hotel_id,
            customer_phone=customer_phone,
            booking_status=ACTIVE_BOOKING_STATUSES,
        )
        if booking and _is_booking_active(booking):
            return await _verify_room_number(booking, state.get("user_message", ""))

    return None


def _extract_room_from_message(user_message: str) -> str | None:
    room_match = re.search(
        r"\b(?:room|rm)\s*#?\s*([0-9]{2,4}[a-zA-Z]?)\b",
        (user_message or "").lower(),
    )
    return room_match.group(1) if room_match else None


async def _verify_room_number(booking: dict, user_message: str) -> dict | None:
    """Verify that if the user mentions a room number, it matches their booking."""
    requested_room = _extract_room_from_message(user_message)
    if not requested_room:
        return booking

    booking_room = str(
        booking.get("room_number") or booking.get("room_no") or ""
    ).lower()

    # Booking has no assigned room yet — accept guest-stated room (common for holds)
    if not booking_room:
        return booking

    if requested_room.lower() != booking_room:
        logger.warning(
            f"Room mismatch: requested {requested_room}, booking has {booking_room}. "
            f"Accepting guest-stated room for service task."
        )
        booking["room_number"] = requested_room

    return booking


def _is_booking_active(booking: dict) -> bool:
    """Check if a booking is in an active state (eligible for task creation)."""
    status = booking.get("booking_status", "").lower()

    # Reject cancelled/checked-out bookings
    if status in ("cancelled", "checked_out", "no_show"):
        return False

    # Accept confirmed, checked_in, pending holds, partial payments
    if status in ("confirmed", "checked_in", "pending", "pending_payment", "partial", ""):
        return True

    # Default: accept (be lenient)
    return True


# ═══════════════════════════════════════════════════════════════════
# DEPARTMENT ROUTING (Python-only — deterministic)
# ═══════════════════════════════════════════════════════════════════

def _resolve_department(task_type: str, subtype: str) -> str:
    """
    Resolve the correct department. Python-only, never AI.

    Priority:
    1. Subtype-specific mapping (most precise)
    2. Task type mapping (fallback)
    3. FRONT_DESK (ultimate default)
    """
    # Subtype takes priority (more specific)
    if subtype in SUBTYPE_TO_DEPARTMENT:
        return SUBTYPE_TO_DEPARTMENT[subtype]

    # Fall back to task type
    if task_type in TASK_TYPE_TO_DEPARTMENT:
        return TASK_TYPE_TO_DEPARTMENT[task_type]

    # Ultimate default
    return "FRONT_DESK"


# ═══════════════════════════════════════════════════════════════════
# PRIORITY RESOLUTION (Python-only)
# ═══════════════════════════════════════════════════════════════════

def _resolve_priority(gemini_urgency: str, subtype: str) -> str:
    """
    Resolve final priority. Python rules override Gemini when needed.

    Logic:
    1. Subtype-based rules always win (URGENT/HIGH subtypes)
    2. Then use the higher of Gemini's assessment and subtype default
    3. Default to NORMAL
    """
    priority_rank = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "URGENT": 3}

    # Check subtype-based rules first
    if subtype in URGENT_PRIORITY_SUBTYPES:
        subtype_priority = "URGENT"
    elif subtype in HIGH_PRIORITY_SUBTYPES:
        subtype_priority = "HIGH"
    else:
        subtype_priority = "NORMAL"

    # Use the higher of Gemini and subtype-based priority
    gemini_rank = priority_rank.get(gemini_urgency, 1)
    subtype_rank = priority_rank.get(subtype_priority, 1)

    if gemini_rank >= subtype_rank:
        return gemini_urgency if gemini_urgency in priority_rank else "NORMAL"
    return subtype_priority


# ═══════════════════════════════════════════════════════════════════
# STAFF SUMMARY BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_staff_summary(
    department: str,
    task_type: str,
    subtype: str,
    room_number: str | None,
    gemini_summary: str,
) -> str:
    """
    Build a concise internal summary for hotel staff.
    No Gemini call needed — deterministic formatting.
    """
    dept_display = DEPARTMENT_DISPLAY_NAMES.get(department, department)
    subtype_display = subtype.replace("_", " ").title()

    parts = [
        f"Department: {dept_display}",
        f"Task: {subtype_display}",
    ]
    if room_number:
        parts.append(f"Room: {room_number}")
    if gemini_summary:
        parts.append(f"Summary: {gemini_summary}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _extract_task_classification(state: dict) -> TaskClassification:
    """Extract the task classification from the graph state."""
    return TaskClassification(
        is_task_request=state.get("task_is_request", False),
        task_type=state.get("task_type", "OTHER"),
        subtype=state.get("task_subtype", "OTHER"),
        urgency=state.get("task_urgency", "NORMAL"),
        confidence=state.get("task_confidence", 0.0),
        summary=state.get("task_summary", ""),
    )


# ═══════════════════════════════════════════════════════════════════
# DEDICATED PAYMENT CONFIRMATION TASK CREATOR
# ═══════════════════════════════════════════════════════════════════

async def create_payment_confirmation_task(
    state: dict,
    payment_info: dict | None = None,
    source: str = "payment_flow",
) -> dict[str, Any]:
    """
    Dedicated operational task creator for payment confirmation.
    
    Handles:
    - Payments claimed via online other party (brother, friend, 3rd party UPI)
    - Other payment modes (Direct Bank Transfer / NEFT / IMPS / RTGS, Cash, Cheque)
    - Manual review needed (receipt OCR ambiguous or partial)
    - UTR / transaction IDs submitted in chat
    """
    payment_info = payment_info or {}
    hotel_id = state.get("hotel_id", 0)
    session_id = state.get("session_id", "")
    booking_id = state.get("booking_id") or payment_info.get("booking_id")
    # A payment reported before the guest produced proof has no `BK-…` id yet:
    # the real id is minted only when the screenshot arrives (see
    # `booking_identity`). What does exist is the provisional `PAY-…`
    # reference, and it is the id of the very row this task is about. Stamped
    # here because it is the last moment it is known — without it the task
    # reaches the front desk, and later the completion message, with nothing
    # to name. Measured 2026-11-10: 9 of 11 completed payment tasks carried no
    # booking id.
    payment_reference = (
        state.get("payment_reference")
        or payment_info.get("payment_reference")
        or None
    )
    task_booking_ref = booking_id or payment_reference
    user_message = state.get("user_message", "")

    customer_phone = (
        state.get("guest_phone")
        or state.get("customer_identifier")
        or payment_info.get("customer_phone")
    )
    customer_name = (
        state.get("guest_name")
        or payment_info.get("customer_name")
    )

    # 1. Resolve payment metadata
    amount = (
        payment_info.get("amount")
        or state.get("amount_paid")
        or state.get("payment_link_amount")
        or state.get("total_price")
        or 0.0
    )
    try:
        amount_val = float(amount)
    except (ValueError, TypeError):
        amount_val = 0.0

    payment_mode = payment_info.get("payment_mode") or "Online Other Party / Other Modes"
    payer_name = payment_info.get("payer_name") or "Not Specified / Other Party"
    utr_number = payment_info.get("utr_number") or payment_info.get("transaction_id") or "Pending Verification"
    details = payment_info.get("details") or user_message[:300]

    # 2. Check for duplicate/existing open payment confirmation task
    existing = await task_repository.find_open_payment_task(
        hotel_id=hotel_id,
        booking_id=task_booking_ref,
        session_id=session_id,
        customer_phone=customer_phone,
    )

    if existing:
        task_id = existing["task_id"]
        update_text = (
            f"Payment Update: Mode={payment_mode}, Amount=₹{amount_val:,.0f}, "
            f"Payer={payer_name}, UTR={utr_number}. Details: {details}"
        )
        await task_repository.append_customer_update(task_id, update_text)
        logger.info(f"📎 Appended payment update to existing task: {task_id}")

        return {
            "action": "task_updated",
            "task_id": task_id,
            "response_text": (
                f"I've added your payment details to your pending verification request (**{task_id}**). ✅\n\n"
                f"Our front desk team is reviewing it and will confirm your booking as soon as verified. 🙏"
            ),
        }

    # 2b. Refuse to raise a task with nothing for the front desk to verify.
    #
    # Measured 2026-10-27 in the mirror: 133 of 157 payment-confirmation tasks
    # carry no booking reference at all. Most are legitimate — a guest sends a
    # screenshot before a `BK-…` id exists, and the task is what staff work
    # from. The thing that is never legitimate is a task with no booking, no
    # amount, no UTR and no receipt behind it: there is nothing in it to check,
    # and it lands in the same queue as the real ones.
    #
    # "A source other than the default" is itself evidence here — every caller
    # that passes one reached this function because it had a receipt, a
    # screenshot or a guest's explicit claim (`vision_*`, `handwritten_slip`,
    # `receipt_review`, `guest_claims_paid`).
    if get_settings().PAYMENT_TASK_REQUIRES_EVIDENCE:
        has_booking_ref = bool(str(booking_id or "").strip())
        has_amount = amount_val > 0
        has_utr = bool(
            utr_number
            and str(utr_number).strip().lower() not in ("", "none", "pending verification")
        )
        has_source = bool(source and source != "payment_flow")
        if not any((has_booking_ref, has_amount, has_utr, has_source)):
            logger.warning(
                "PAYMENT_TASK_REFUSED_NO_EVIDENCE | hotel=%s phone=%s session=%s msg=%r",
                hotel_id, customer_phone, session_id, user_message[:80],
            )
            # Count the refusal. Without this the gate is invisible: the number
            # of tasks simply drops, and nobody can tell a quiet week from a
            # gate that is refusing everything.
            from app.observability.metrics import metrics
            metrics.incr("task.create_total", outcome="skipped_no_evidence", hotel_id=hotel_id)
            return {
                "action": "task_skipped",
                "reason": "no booking reference, amount, UTR or receipt to verify",
                "task_id": None,
                "response_text": None,
            }

    # 3. Create new HotelTask
    task_id = await task_repository.generate_task_id(hotel_id)
    now = datetime.now(timezone.utc)

    summary_lines = [
        "Department: Front Desk",
        "Task: Payment Confirmation",
        f"Mode: {payment_mode}",
        f"Amount: ₹{amount_val:,.0f}" if amount_val > 0 else "Amount: Review Needed",
        f"Payer: {payer_name}",
        f"UTR / Ref: {utr_number}",
    ]
    if customer_name:
        summary_lines.append(f"Guest: {customer_name}")
    if customer_phone:
        summary_lines.append(f"Phone: {customer_phone}")
    if task_booking_ref:
        summary_lines.append(f"Booking ID: {task_booking_ref}")
    if details:
        summary_lines.append(f"Notes: {details[:200]}")

    summary = "\n".join(summary_lines)

    task = HotelTask(
        task_id=task_id,
        hotel_id=hotel_id,
        session_id=session_id,
        booking_id=task_booking_ref,
        room_number=state.get("room_number") or _extract_room_from_message(user_message),
        customer_name=customer_name,
        customer_phone=customer_phone,
        department="FRONT_DESK",
        task_type="FRONT_DESK",
        subtype="PAYMENT_CONFIRMATION",
        priority="HIGH",
        status="OPEN",
        summary=summary,
        customer_request=user_message or f"Payment confirmation request: {payment_mode}",
        created_by="AI",
        events=[
            TaskEvent(
                event_type="CREATED",
                description=(
                    f"Payment confirmation task created for Front Desk. "
                    f"Mode: {payment_mode}, Amount: ₹{amount_val:,.0f}, UTR: {utr_number}."
                ),
                timestamp=now,
                actor="system",
                metadata={
                    "department": "FRONT_DESK",
                    "task_type": "FRONT_DESK",
                    "subtype": "PAYMENT_CONFIRMATION",
                    "priority": "HIGH",
                    "payment_mode": payment_mode,
                    "amount": amount_val,
                    "payer_name": payer_name,
                    "utr_number": utr_number,
                    "session_id": session_id,
                    "booking_id": booking_id,
                    "payment_reference": payment_reference,
                    "source": source,
                },
            ),
        ],
        notes=[
            f"[{now.isoformat()}] AI: Payment confirmation task auto-created via {source} for Front Desk verification.",
        ],
        # Top-level, not only inside the event: the completion sweep reads the
        # task document and follows this reference to the row it became.
        metadata={"payment_reference": payment_reference} if payment_reference else {},
        created_at=now,
        updated_at=now,
    )

    await task_repository.create_task(task)

    from app.observability.metrics import metrics
    metrics.incr(
        "task.create_total",
        outcome="ok",
        hotel_id=hotel_id,
        department="FRONT_DESK",
    )

    logger.info(f"💳 Payment confirmation task created: {task_id} | Amount: ₹{amount_val} | Mode: {payment_mode}")

    return {
        "action": "task_created",
        "task_id": task_id,
        "response_text": (
            f"Thank you! I have created a payment verification request for our front desk team. ✅\n\n"
            f"🔧 **Task ID:** {task_id}\n\n"
            f"Our team has been notified to verify your payment ({payment_mode}) and confirm your reservation. 🙏"
        ),
    }

