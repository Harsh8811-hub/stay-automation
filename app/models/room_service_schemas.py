import enum
from datetime import datetime, UTC
from typing import Any

from pydantic import BaseModel, Field

class RoomServiceCartStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    CHECKED_OUT = "CHECKED_OUT"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

class RoomServiceCartItem(BaseModel):
    menu_item_id: str | None = None
    item_name: str
    quantity: int
    unit_price: float
    total_price: float
    notes: str | None = None
    category: str | None = None
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class RoomServiceCart(BaseModel):
    cart_id: str
    hotel_id: str | int
    booking_id: str | None = None
    session_id: str
    customer_phone: str | None = None
    room_number: str | None = None
    status: RoomServiceCartStatus = RoomServiceCartStatus.ACTIVE
    items: list[RoomServiceCartItem] = Field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    service_charge: float = 0.0
    total: float = 0.0
    special_instructions: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class RoomServiceMenuItem(BaseModel):
    menu_item_id: str
    hotel_id: str | int
    category: str
    item_name: str
    description: str | None = None
    price: float
    available: bool = True
    active: bool = True

class ExtractedCartItem(BaseModel):
    item_name: str
    quantity: int | None = None
    notes: str | None = None

class RoomServiceExtraction(BaseModel):
    action: str | None = None # ADD_ITEMS, REMOVE_ITEM, UPDATE_QUANTITY, CLEAR_CART, VIEW_CART, CHECKOUT, ADD_NOTES
    items: list[ExtractedCartItem] = Field(default_factory=list)
    notes: str | None = None
