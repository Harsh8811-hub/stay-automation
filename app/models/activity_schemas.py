"""
Activity Log Schemas.

A unified timeline tracking every significant event across the system:
- Bookings
- Payments
- Tasks
- Tickets
- Lifecycle events

This enables a single-view chronological dashboard for hotel staff.
"""

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

class ActivityType(str, enum.Enum):
    # Booking
    BOOKING_CREATED = "BOOKING_CREATED"
    BOOKING_CONFIRMED = "BOOKING_CONFIRMED"
    BOOKING_CANCELLED = "BOOKING_CANCELLED"
    BOOKING_MODIFIED = "BOOKING_MODIFIED"
    BOOKING_DELETED = "BOOKING_DELETED"      # Row hard-deleted (backend/testing only)
    
    # Payment
    PAYMENT_RECEIVED = "PAYMENT_RECEIVED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    REFUND_ISSUED = "REFUND_ISSUED"
    
    # Tasks
    TASK_CREATED = "TASK_CREATED"
    TASK_UPDATED = "TASK_UPDATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_CANCELLED = "TASK_CANCELLED"
    
    # Support Tickets (Escalations)
    TICKET_CREATED = "TICKET_CREATED"
    TICKET_UPDATED = "TICKET_UPDATED"
    TICKET_RESOLVED = "TICKET_RESOLVED"

    # Leads
    LEAD_CREATED = "LEAD_CREATED"

    # ── Lifecycle Events ──────────────────────────────────────────────
    BOOKING_STARTED = "BOOKING_STARTED"
    BOOKING_RESET = "BOOKING_RESET"                        # Guest confirmed start-fresh
    BOOKING_RESET_ABORTED = "BOOKING_RESET_ABORTED"        # Guest declined start-fresh
    BOOKING_CANCEL_REQUESTED = "BOOKING_CANCEL_REQUESTED"  # Guest asked to cancel
    CANCELLATION_ABORTED = "CANCELLATION_ABORTED"          # Guest backed out of cancel
    MODIFICATION_CONFIRMED = "MODIFICATION_CONFIRMED"      # Modification applied
    MODIFICATION_DISCARDED = "MODIFICATION_DISCARDED"      # Modification discarded
    STAY_EXTENDED = "STAY_EXTENDED"                        # Checked-in guest extended checkout
    EARLY_CHECKOUT_REQUESTED = "EARLY_CHECKOUT_REQUESTED"  # Checked-in guest shortened stay
    LIFECYCLE_BLOCKED = "LIFECYCLE_BLOCKED"                # Blocked due to human takeover


class ActivityLog(BaseModel):
    """Unified log entry for the activity_logs collection."""
    hotel_id: str | int
    activity_type: str            # ActivityType value
    reference_id: str             # e.g., BK-123, TASK-001, SUP-001, TXN-001
    booking_id: str | None = None # Often present for tasks/tickets as well
    session_id: str | None = None # Conversation session for lifecycle tracing
    queue_id: str | None = None   # Multi-booking queue ID if applicable
    customer_phone: str | None = None
    customer_name: str | None = None
    
    # The actual event text for display (e.g. "Guest requested extra pillows")
    description: str
    
    # Additional context
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # When it happened
    timestamp: datetime = Field(default_factory=datetime.utcnow)
