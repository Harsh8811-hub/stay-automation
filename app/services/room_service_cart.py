import logging
from typing import Any
import uuid

from app.models.room_service_schemas import RoomServiceCart, RoomServiceCartItem, RoomServiceCartStatus
from app.repositories.room_service_repository import (
    get_active_cart, get_menu_items, save_cart, generate_cart_id, mark_cart_checked_out
)
from app.models.task_schemas import HotelTask
from app.repositories.task_repository import create_task

logger = logging.getLogger(__name__)


def calculate_cart_totals(cart: RoomServiceCart) -> None:
    subtotal = sum(item.total_price for item in cart.items)
    # Simple deterministic logic: 5% tax, 10% service charge for room service
    tax = subtotal * 0.05
    service_charge = subtotal * 0.10
    total = subtotal + tax + service_charge
    
    cart.subtotal = round(subtotal, 2)
    cart.tax = round(tax, 2)
    cart.service_charge = round(service_charge, 2)
    cart.total = round(total, 2)


async def process_cart_action(
    hotel_id: int, 
    session_id: str,
    action: str, 
    items: list[dict], 
    notes: str | None = None,
    booking_id: str | None = None,
    room_number: str | None = None,
    customer_phone: str | None = None
) -> dict:
    """
    Process a room service cart action.
    Returns a response dict to be sent back to the user.
    """
    cart = await get_active_cart(hotel_id, session_id)
    if not cart:
        cart_id = await generate_cart_id(hotel_id)
        cart = RoomServiceCart(
            cart_id=cart_id,
            hotel_id=hotel_id,
            session_id=session_id,
            booking_id=booking_id,
            room_number=room_number,
            customer_phone=customer_phone
        )
    
    if notes:
        if cart.special_instructions:
            cart.special_instructions += f" | {notes}"
        else:
            cart.special_instructions = notes

    menu = await get_menu_items(hotel_id)
    menu_map = {m.item_name.lower(): m for m in menu}
    
    response_messages = []
    
    if action == "ADD_ITEMS":
        for extracted_item in items:
            name = extracted_item.get("item_name", "").lower()
            quantity = extracted_item.get("quantity") or 1
            item_notes = extracted_item.get("notes")
            
            # Simple fuzzy matching could be added here
            menu_item = menu_map.get(name)
            if not menu_item:
                response_messages.append(f"Sorry, '{extracted_item.get('item_name')}' is not on the menu.")
                continue
                
            if not menu_item.available:
                response_messages.append(f"Sorry, '{menu_item.item_name}' is currently unavailable.")
                continue

            # Check if already in cart
            existing = next((i for i in cart.items if i.menu_item_id == menu_item.menu_item_id), None)
            if existing:
                existing.quantity += quantity
                existing.total_price = existing.quantity * existing.unit_price
                if item_notes:
                    existing.notes = item_notes
                response_messages.append(f"Added {quantity} more {menu_item.item_name}.")
            else:
                cart.items.append(RoomServiceCartItem(
                    menu_item_id=menu_item.menu_item_id,
                    item_name=menu_item.item_name,
                    quantity=quantity,
                    unit_price=menu_item.price,
                    total_price=quantity * menu_item.price,
                    notes=item_notes,
                    category=menu_item.category
                ))
                response_messages.append(f"Added {quantity} x {menu_item.item_name} to your cart.")
                
    elif action == "REMOVE_ITEM":
        for extracted_item in items:
            name = extracted_item.get("item_name", "").lower()
            original_len = len(cart.items)
            cart.items = [i for i in cart.items if i.item_name.lower() != name]
            if len(cart.items) < original_len:
                response_messages.append(f"Removed {extracted_item.get('item_name')} from your cart.")
            else:
                response_messages.append(f"Could not find {extracted_item.get('item_name')} in your cart.")
                
    elif action == "UPDATE_QUANTITY":
        for extracted_item in items:
            name = extracted_item.get("item_name", "").lower()
            quantity = extracted_item.get("quantity") or 1
            existing = next((i for i in cart.items if i.item_name.lower() == name), None)
            if existing:
                if quantity <= 0:
                    cart.items.remove(existing)
                    response_messages.append(f"Removed {existing.item_name}.")
                else:
                    existing.quantity = quantity
                    existing.total_price = quantity * existing.unit_price
                    response_messages.append(f"Updated {existing.item_name} quantity to {quantity}.")
            else:
                response_messages.append(f"Could not find {extracted_item.get('item_name')} in your cart.")
                
    elif action == "CLEAR_CART":
        cart.items = []
        cart.special_instructions = None
        response_messages.append("Your cart has been cleared.")
        
    elif action == "CHECKOUT":
        if not cart.items:
            return {"status": "error", "message": "Your cart is empty. Please add items before checking out."}
            
        calculate_cart_totals(cart)
        
        # Create a single task for the kitchen/room service
        item_details = [f"{i.quantity}x {i.item_name} (₹{i.total_price})" + (f" - Note: {i.notes}" if i.notes else "") for i in cart.items]
        summary = f"Room Service Order - Total: ₹{cart.total}\n" + "\n".join(item_details)
        if cart.special_instructions:
            summary += f"\nSpecial Instructions: {cart.special_instructions}"
            
        task = HotelTask(
            task_id=f"RS-{cart.cart_id}-{uuid.uuid4().hex[:6]}",
            hotel_id=hotel_id,
            session_id=session_id,
            booking_id=cart.booking_id,
            room_number=cart.room_number,
            customer_phone=cart.customer_phone,
            department="ROOM_SERVICE",
            task_type="ROOM_SERVICE",
            subtype="FOOD_DELIVERY",
            priority="HIGH",
            status="OPEN",
            summary=summary,
            customer_request="Guest placed a room service order.",
            created_by="AI"
        )
        task_id = await create_task(task)
        
        await mark_cart_checked_out(cart.cart_id)
        
        return {
            "status": "success", 
            "message": f"Your order has been placed successfully! Total amount is ₹{cart.total}. It will be delivered to your room shortly.",
            "cart": cart.model_dump(),
            "task_id": task_id
        }
        
    # Recalculate totals
    calculate_cart_totals(cart)
    await save_cart(cart)
    
    msg = "\n".join(response_messages) if response_messages else "Cart updated."
    
    if action == "VIEW_CART":
        if not cart.items:
            msg = "Your cart is currently empty."
        else:
            msg = "Here is your current cart:\n"
            for item in cart.items:
                msg += f"- {item.quantity}x {item.item_name} (₹{item.total_price})\n"
            msg += f"\nSubtotal: ₹{cart.subtotal}\nTaxes & Fees: ₹{round(cart.tax + cart.service_charge, 2)}\n**Total: ₹{cart.total}**\n\nWould you like to checkout?"
    
    return {
        "status": "updated",
        "message": msg,
        "cart": cart.model_dump()
    }
