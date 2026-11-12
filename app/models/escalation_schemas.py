"""
Escalation & Support Ticket Schemas.

Covers:
- Gemini escalation classification output
- Support ticket MongoDB document model
- Ticket event audit log
- Reason codes for granular analytics
"""

import enum
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class EscalationCategory(str, enum.Enum):
    """High-level issue category for routing."""
    NORMAL = "NORMAL"
    EXPLICIT_HUMAN_REQUEST = "EXPLICIT_HUMAN_REQUEST"
    BOOKING_DISPUTE = "BOOKING_DISPUTE"
    PAYMENT_PROBLEM = "PAYMENT_PROBLEM"
    COMPLAINT = "COMPLAINT"
    EMOTIONAL_CUSTOMER = "EMOTIONAL_CUSTOMER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    SECURITY = "SECURITY"
    LEGAL = "LEGAL"
    MEDICAL = "MEDICAL"
    SAFETY = "SAFETY"
    UNKNOWN = "UNKNOWN"


class ReasonCode(str, enum.Enum):
    """Granular reason codes for analytics and dashboard filtering."""
    # Payment
    PAYMENT_DUPLICATE = "PAYMENT_DUPLICATE"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    REFUND_REQUEST = "REFUND_REQUEST"
    # Booking
    BOOKING_NOT_FOUND = "BOOKING_NOT_FOUND"
    BOOKING_MODIFICATION = "BOOKING_MODIFICATION"
    BOOKING_MISMATCH = "BOOKING_MISMATCH"
    # Cancellation
    CANCELLATION_REQUEST = "CANCELLATION_REQUEST"
    # Complaints
    GENERAL_COMPLAINT = "GENERAL_COMPLAINT"
    SERVICE_COMPLAINT = "SERVICE_COMPLAINT"
    # Security
    SECURITY_CONCERN = "SECURITY_CONCERN"
    FRAUD_REPORT = "FRAUD_REPORT"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    # Out of scope
    JOB_ENQUIRY = "JOB_ENQUIRY"
    PARTNERSHIP_ENQUIRY = "PARTNERSHIP_ENQUIRY"
    MEDIA_ENQUIRY = "MEDIA_ENQUIRY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    # Human request
    EXPLICIT_HUMAN_REQUEST = "EXPLICIT_HUMAN_REQUEST"
    # Safety
    MEDICAL_EMERGENCY = "MEDICAL_EMERGENCY"
    SAFETY_CONCERN = "SAFETY_CONCERN"
    LEGAL_MATTER = "LEGAL_MATTER"
    # Technical
    TECHNICAL_ISSUE = "TECHNICAL_ISSUE"
    # Catch-all
    OTHER = "OTHER"


class TicketPriority(str, enum.Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TicketStatus(str, enum.Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class Sentiment(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    ANGRY = "ANGRY"
    DISTRESSED = "DISTRESSED"


# ═══════════════════════════════════════════════════════════════════
# GEMINI ESCALATION CLASSIFICATION (embedded in understand_message)
# ═══════════════════════════════════════════════════════════════════

class EscalationClassification(BaseModel):
    """
    Structured output from Gemini — classification only, no routing.
    Gemini fills this; Python makes all decisions based on it.
    """
    escalation_required: bool = False
    reason: str = "normal_query"
    issue_type: str = "NORMAL"       # Maps to EscalationCategory
    reason_code: str = "OTHER"       # Maps to ReasonCode — granular
    urgency: str = "LOW"             # Maps to TicketPriority
    sentiment: str = "NEUTRAL"       # Maps to Sentiment
    confidence: float = 0.9


# ═══════════════════════════════════════════════════════════════════
# TICKET EVENT (append-only audit log entry)
# ═══════════════════════════════════════════════════════════════════

class TicketEvent(BaseModel):
    """Single audit log entry. Never overwritten, always appended."""
    event_type: str          # "CREATED", "STATUS_CHANGED", "CUSTOMER_REPLIED", etc.
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor: str = "system"    # "system", "customer", "staff"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# CUSTOMER MESSAGE (visible to customer + staff)
# ═══════════════════════════════════════════════════════════════════

class TicketMessage(BaseModel):
    """A message in the ticket conversation visible to both customer and staff."""
    role: str                # "customer" or "bot"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════
# STAFF NOTE (internal only — never visible to customers)
# ═══════════════════════════════════════════════════════════════════

class StaffNote(BaseModel):
    """Internal note visible only to hotel staff. Never exposed to customers."""
    author: str = "system"   # staff name or "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════
# SUPPORT TICKET (MongoDB document model)
# ═══════════════════════════════════════════════════════════════════

class SupportTicket(BaseModel):
    """
    Full support ticket stored in MongoDB.
    
    Design:
    - `conversation` holds customer-visible messages
    - `staff_notes` holds internal-only notes (never shown to customer)
    - `events` holds append-only audit trail
    """
    ticket_id: str                           # "SUP-20260701-000123"
    hotel_id: str | int
    session_id: str
    booking_id: str | None = None

    # Customer info
    customer_name: str | None = None
    customer_phone: str | None = None
    customer_email: str | None = None

    # Issue classification
    issue_type: str = "UNKNOWN"              # EscalationCategory value
    reason_code: str = "OTHER"               # ReasonCode — granular
    priority: str = "NORMAL"                 # TicketPriority value
    status: str = "OPEN"                     # TicketStatus value

    # Content
    summary: str = ""                        # Gemini-generated staff summary
    last_user_message: str = ""
    booking_state: dict[str, Any] = Field(default_factory=dict)
    smart_replies: list[str] = Field(default_factory=list)  # Suggested quick replies for hotel staff

    # Conversation (customer-visible)
    conversation: list[TicketMessage] = Field(default_factory=list)

    # Internal staff notes (NEVER shown to customer)
    staff_notes: list[StaffNote] = Field(default_factory=list)

    # Assignment
    assigned_to: str | None = None
    created_by: str = "AI"                   # "AI", "Reception", "Manager", "Admin", "Website", "Phone"
    resolution: str | None = None

    # Audit trail (append-only)
    events: list[TicketEvent] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=7)
    )

    # Extensible metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
