"""
Hotel Task Management Schemas.

Completely separate from Support Tickets.
Support Tickets = customer support & human escalation.
Hotel Tasks = operational work performed by hotel departments.

Covers:
- Task type / subtype enums
- Department mapping (Python-only — Gemini never invents departments)
- Priority levels
- Task status lifecycle
- MongoDB document model
- Gemini task classification output schema
- Append-only event log
"""

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class TaskDepartment(str, enum.Enum):
    """Departments that handle operational tasks. Python-only mapping."""
    HOUSEKEEPING = "HOUSEKEEPING"
    MAINTENANCE = "MAINTENANCE"
    ROOM_SERVICE = "ROOM_SERVICE"
    FRONT_DESK = "FRONT_DESK"
    ENGINEERING = "ENGINEERING"
    SECURITY = "SECURITY"


class TaskType(str, enum.Enum):
    """High-level task categories classified by Gemini."""
    HOUSEKEEPING = "HOUSEKEEPING"
    MAINTENANCE = "MAINTENANCE"
    ROOM_SERVICE = "ROOM_SERVICE"
    FRONT_DESK = "FRONT_DESK"
    ENGINEERING = "ENGINEERING"
    SECURITY = "SECURITY"
    OTHER = "OTHER"


class TaskSubtype(str, enum.Enum):
    """Granular subtypes for analytics and routing."""
    # Housekeeping
    EXTRA_PILLOWS = "EXTRA_PILLOWS"
    EXTRA_TOWELS = "EXTRA_TOWELS"
    EXTRA_BLANKETS = "EXTRA_BLANKETS"
    ROOM_CLEANING = "ROOM_CLEANING"
    LINEN_CHANGE = "LINEN_CHANGE"
    TOILETRIES = "TOILETRIES"
    DRINKING_WATER = "DRINKING_WATER"
    TRASH_REMOVAL = "TRASH_REMOVAL"
    # Maintenance
    AC_REPAIR = "AC_REPAIR"
    TV_REPAIR = "TV_REPAIR"
    WIFI_ISSUE = "WIFI_ISSUE"
    PLUMBING_ISSUE = "PLUMBING_ISSUE"
    ELECTRICAL_ISSUE = "ELECTRICAL_ISSUE"
    FURNITURE_ISSUE = "FURNITURE_ISSUE"
    DOOR_LOCK_ISSUE = "DOOR_LOCK_ISSUE"
    HOT_WATER_ISSUE = "HOT_WATER_ISSUE"
    GENERAL_MAINTENANCE = "GENERAL_MAINTENANCE"
    # Room service
    FOOD_DELIVERY = "FOOD_DELIVERY"
    BEVERAGE_REQUEST = "BEVERAGE_REQUEST"
    MINIBAR_REFILL = "MINIBAR_REFILL"
    SPECIAL_DIETARY = "SPECIAL_DIETARY"
    # Front desk
    WAKE_UP_CALL = "WAKE_UP_CALL"
    LUGGAGE_ASSISTANCE = "LUGGAGE_ASSISTANCE"
    TAXI_REQUEST = "TAXI_REQUEST"
    CHECKOUT_REQUEST = "CHECKOUT_REQUEST"
    ROOM_CHANGE = "ROOM_CHANGE"
    # A confirmed booking asked to be changed, and a confirmed booking asked to
    # be cancelled. Distinct from ROOM_CHANGE, which is an in-stay request, and
    # from CHECKOUT_REQUEST, which is a request to *leave* — a cancellation was
    # filed under that until 2026-10-10, so an operator scanning subtypes could
    # not tell "guest wants to check out early" from "guest wants their money
    # back". These are pre-arrival, money-bearing, and need a human decision.
    BOOKING_AMENDMENT = "BOOKING_AMENDMENT"
    BOOKING_CANCELLATION = "BOOKING_CANCELLATION"
    LATE_CHECKOUT = "LATE_CHECKOUT"
    KEY_REPLACEMENT = "KEY_REPLACEMENT"
    INFORMATION_REQUEST = "INFORMATION_REQUEST"
    PAYMENT_CONFIRMATION = "PAYMENT_CONFIRMATION"
    BILLING_INQUIRY = "BILLING_INQUIRY"
    INVOICE_REQUEST = "INVOICE_REQUEST"
    # Engineering
    POWER_ISSUE = "POWER_ISSUE"
    ELEVATOR_ISSUE = "ELEVATOR_ISSUE"
    HVAC_ISSUE = "HVAC_ISSUE"
    # Security
    NOISE_COMPLAINT = "NOISE_COMPLAINT"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    LOST_AND_FOUND = "LOST_AND_FOUND"
    ACCESS_ISSUE = "ACCESS_ISSUE"
    # Catch-all
    OTHER = "OTHER"


class TaskPriority(str, enum.Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class TaskStatus(str, enum.Enum):
    OPEN = "OPEN"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


# ═══════════════════════════════════════════════════════════════════
# EMERGENCY KEYWORDS (bypass task creation → CRITICAL ticket)
# ═══════════════════════════════════════════════════════════════════

EMERGENCY_KEYWORDS = {
    "fire", "smoke", "gas smell", "gas leak", "medical emergency",
    "violence", "flooding", "flood", "electrical hazard", "electrocution",
    "heart attack", "unconscious", "not breathing", "ambulance",
    "security threat", "intruder", "theft", "robbery", "assault",
    "sparks", "burning", "explosion", "collapsed", "earthquake",
}

# Individual danger words that trigger emergency when found as standalone tokens
EMERGENCY_DANGER_WORDS = {
    "fire", "smoke", "flood", "flooding", "ambulance", "unconscious",
    "intruder", "robbery", "assault", "explosion", "earthquake",
    "electrocution", "sparks", "burning", "violence", "collapsed",
}


# ═══════════════════════════════════════════════════════════════════
# DEPARTMENT ROUTING MAP (Python-only — Gemini NEVER invents routing)
# ═══════════════════════════════════════════════════════════════════

TASK_TYPE_TO_DEPARTMENT: dict[str, str] = {
    "HOUSEKEEPING": "HOUSEKEEPING",
    "MAINTENANCE": "MAINTENANCE",
    "ROOM_SERVICE": "ROOM_SERVICE",
    "FRONT_DESK": "FRONT_DESK",
    "ENGINEERING": "ENGINEERING",
    "SECURITY": "SECURITY",
    "OTHER": "FRONT_DESK",  # Default to front desk for uncategorized
}

SUBTYPE_TO_DEPARTMENT: dict[str, str] = {
    # Housekeeping
    "EXTRA_PILLOWS": "HOUSEKEEPING",
    "EXTRA_TOWELS": "HOUSEKEEPING",
    "EXTRA_BLANKETS": "HOUSEKEEPING",
    "ROOM_CLEANING": "HOUSEKEEPING",
    "LINEN_CHANGE": "HOUSEKEEPING",
    "TOILETRIES": "HOUSEKEEPING",
    "DRINKING_WATER": "HOUSEKEEPING",
    "TRASH_REMOVAL": "HOUSEKEEPING",
    # Maintenance
    "AC_REPAIR": "MAINTENANCE",
    "TV_REPAIR": "MAINTENANCE",
    "WIFI_ISSUE": "MAINTENANCE",
    "PLUMBING_ISSUE": "MAINTENANCE",
    "ELECTRICAL_ISSUE": "MAINTENANCE",
    "FURNITURE_ISSUE": "MAINTENANCE",
    "DOOR_LOCK_ISSUE": "MAINTENANCE",
    "HOT_WATER_ISSUE": "MAINTENANCE",
    "GENERAL_MAINTENANCE": "MAINTENANCE",
    # Room service
    "FOOD_DELIVERY": "ROOM_SERVICE",
    "BEVERAGE_REQUEST": "ROOM_SERVICE",
    "MINIBAR_REFILL": "ROOM_SERVICE",
    "SPECIAL_DIETARY": "ROOM_SERVICE",
    # Front desk
    "WAKE_UP_CALL": "FRONT_DESK",
    "LUGGAGE_ASSISTANCE": "FRONT_DESK",
    "TAXI_REQUEST": "FRONT_DESK",
    "CHECKOUT_REQUEST": "FRONT_DESK",
    "ROOM_CHANGE": "FRONT_DESK",
    "BOOKING_AMENDMENT": "FRONT_DESK",
    "BOOKING_CANCELLATION": "FRONT_DESK",
    "LATE_CHECKOUT": "FRONT_DESK",
    "KEY_REPLACEMENT": "FRONT_DESK",
    "INFORMATION_REQUEST": "FRONT_DESK",
    "PAYMENT_CONFIRMATION": "FRONT_DESK",
    "BILLING_INQUIRY": "FRONT_DESK",
    "INVOICE_REQUEST": "FRONT_DESK",
    # Engineering
    "POWER_ISSUE": "ENGINEERING",
    "ELEVATOR_ISSUE": "ENGINEERING",
    "HVAC_ISSUE": "ENGINEERING",
    # Security
    "NOISE_COMPLAINT": "SECURITY",
    "SUSPICIOUS_ACTIVITY": "SECURITY",
    "LOST_AND_FOUND": "SECURITY",
    "ACCESS_ISSUE": "SECURITY",
    # Catch-all
    "OTHER": "FRONT_DESK",
}


# ═══════════════════════════════════════════════════════════════════
# PRIORITY RULES (Python-only)
# ═══════════════════════════════════════════════════════════════════

# Subtypes that default to HIGH priority
HIGH_PRIORITY_SUBTYPES = {
    "AC_REPAIR", "HOT_WATER_ISSUE", "DOOR_LOCK_ISSUE",
    "PLUMBING_ISSUE", "POWER_ISSUE", "PAYMENT_CONFIRMATION",
}

# Subtypes that default to URGENT priority
URGENT_PRIORITY_SUBTYPES = {
    "ELECTRICAL_ISSUE",  # Could be dangerous
}


# ═══════════════════════════════════════════════════════════════════
# GEMINI TASK CLASSIFICATION (output from understand_message)
# ═══════════════════════════════════════════════════════════════════

class TaskClassification(BaseModel):
    """
    Structured output from Gemini — classification only, no routing.
    Gemini fills this; Python makes all decisions based on it.
    """
    is_task_request: bool = False
    task_type: str = "OTHER"        # Maps to TaskType
    subtype: str = "OTHER"          # Maps to TaskSubtype
    urgency: str = "NORMAL"         # Maps to TaskPriority
    confidence: float = 0.0
    summary: str = ""               # Short description of the request


# ═══════════════════════════════════════════════════════════════════
# TASK EVENT (append-only audit log entry)
# ═══════════════════════════════════════════════════════════════════

class TaskEvent(BaseModel):
    """Single audit log entry. Never overwritten, always appended."""
    event_type: str          # "CREATED", "ASSIGNED", "STARTED", "CUSTOMER_UPDATED", "COMPLETED", "CANCELLED"
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor: str = "system"    # "system", "customer", "staff"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# HOTEL TASK (MongoDB document model)
# ═══════════════════════════════════════════════════════════════════

class HotelTask(BaseModel):
    """
    Full hotel task stored in MongoDB `hotel_tasks` collection.

    This is completely separate from SupportTicket.
    Support Tickets → customer support & human escalation.
    Hotel Tasks → operational work performed by hotel departments.
    """
    task_id: str                             # "TASK-20260701-000123"
    hotel_id: str | int
    session_id: str
    booking_id: str | None = None
    room_number: str | None = None

    # Customer info
    customer_name: str | None = None
    customer_phone: str | None = None

    # Classification (set by Python from Gemini output)
    department: str = "FRONT_DESK"           # TaskDepartment value
    task_type: str = "OTHER"                 # TaskType value
    subtype: str = "OTHER"                   # TaskSubtype value
    priority: str = "NORMAL"                 # TaskPriority value
    status: str = "OPEN"                     # TaskStatus value

    # Content
    summary: str = ""                        # Internal summary for staff
    customer_request: str = ""               # Original customer message
    customer_updates: list[dict[str, Any]] = Field(default_factory=list) # Append updates here

    # Assignment (future: staff dashboard integration)
    assigned_staff: str | None = None
    created_by: str = "AI"                   # "AI", "Reception", "Manager", "Admin", "Website", "Phone"

    # Audit trail (append-only)
    events: list[TaskEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    expires_at: datetime | None = None       # Optional — for time-bound tasks

    # Extensible metadata (future: SLA, notifications, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)
