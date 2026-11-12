from datetime import datetime, date
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field

class AccessType(str, Enum):
    FREE = "free"
    PAID = "paid"
    RESERVATION_REQUIRED = "reservation_required"
    MEMBERS_ONLY = "members_only"
    ROOM_TYPE_RESTRICTED = "room_type_restricted"

class DaySchedule(BaseModel):
    open_time: Optional[str] = Field(None, description="HH:MM format (24-hour)")
    close_time: Optional[str] = Field(None, description="HH:MM format (24-hour)")

class MaintenancePeriod(BaseModel):
    start_date: datetime
    end_date: datetime
    reason: Optional[str] = None

class AccessRules(BaseModel):
    access_type: AccessType = AccessType.FREE
    allowed_room_types: Optional[List[str]] = Field(default_factory=list)

class Pricing(BaseModel):
    price: Optional[float] = None
    currency: str = "INR"
    tax_inclusive: bool = True

class Scheduling(BaseModel):
    is_24_hours: bool = False
    default_schedule: Dict[str, DaySchedule] = Field(
        default_factory=dict, 
        description="Keys should be lowercase days: monday, tuesday, etc."
    )
    special_schedule: Dict[str, DaySchedule] = Field(
        default_factory=dict,
        description="Keys should be YYYY-MM-DD"
    )
    maintenance_periods: List[MaintenancePeriod] = Field(default_factory=list)
    holiday_overrides: Dict[str, DaySchedule] = Field(default_factory=dict)

class BookingRules(BaseModel):
    booking_required: bool = False
    slot_duration_minutes: Optional[int] = None
    minimum_notice_minutes: Optional[int] = None
    maximum_advance_days: Optional[int] = None
    maximum_capacity: Optional[int] = None
    maximum_booking_per_guest: Optional[int] = None

class OperationalInfo(BaseModel):
    location: Optional[str] = None
    contact_extension: Optional[str] = None
    notes: Optional[str] = None
    display_order: int = 0
    images: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AmenityBase(BaseModel):
    name: str
    category: str = "General"
    description: str = ""
    available: bool = True
    active: bool = True
    
    access_rules: AccessRules = Field(default_factory=AccessRules)
    pricing: Pricing = Field(default_factory=Pricing)
    scheduling: Scheduling = Field(default_factory=Scheduling)
    booking_rules: BookingRules = Field(default_factory=BookingRules)
    operational_info: OperationalInfo = Field(default_factory=OperationalInfo)

class AmenityCreate(AmenityBase):
    hotel_id: str | int
    amenity_id: str

class AmenityUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    available: Optional[bool] = None
    active: Optional[bool] = None
    
    access_rules: Optional[AccessRules] = None
    pricing: Optional[Pricing] = None
    scheduling: Optional[Scheduling] = None
    booking_rules: Optional[BookingRules] = None
    operational_info: Optional[OperationalInfo] = None

class AmenityInDB(AmenityBase):
    id: str = Field(alias="_id")
    hotel_id: str | int
    amenity_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
