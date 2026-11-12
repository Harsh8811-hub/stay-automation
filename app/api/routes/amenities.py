from typing import List
from fastapi import APIRouter, HTTPException, Depends, Query, status

from app.models.amenity_schemas import AmenityCreate, AmenityUpdate, AmenityInDB
from app.repositories import amenity_repository

router = APIRouter()

@router.post("/", response_model=AmenityInDB, status_code=status.HTTP_201_CREATED)
async def create_amenity(amenity: AmenityCreate):
    """Create a new amenity for a hotel."""
    # Check if exists
    existing = await amenity_repository.get_amenity(amenity.amenity_id, amenity.hotel_id)
    if existing:
        raise HTTPException(
            status_code=400, 
            detail=f"Amenity with ID {amenity.amenity_id} already exists for hotel {amenity.hotel_id}"
        )
    
    created = await amenity_repository.create_amenity(amenity)
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create amenity")
    return created

@router.get("/{hotel_id}", response_model=List[AmenityInDB])
async def list_amenities(
    hotel_id: int, 
    active_only: bool = Query(True, description="Filter only active amenities")
):
    """List all amenities for a given hotel."""
    return await amenity_repository.list_amenities(hotel_id, active_only)

@router.get("/{hotel_id}/{amenity_id}", response_model=AmenityInDB)
async def get_amenity(hotel_id: int, amenity_id: str):
    """Get details of a specific amenity."""
    amenity = await amenity_repository.get_amenity(amenity_id, hotel_id)
    if not amenity:
        raise HTTPException(status_code=404, detail="Amenity not found")
    return amenity

@router.patch("/{hotel_id}/{amenity_id}", response_model=AmenityInDB)
async def update_amenity(hotel_id: int, amenity_id: str, update_data: AmenityUpdate):
    """Update a specific amenity."""
    updated = await amenity_repository.update_amenity(amenity_id, hotel_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Amenity not found or failed to update")
    return updated

@router.delete("/{hotel_id}/{amenity_id}")
async def delete_amenity(
    hotel_id: int, 
    amenity_id: str, 
    hard_delete: bool = Query(False, description="Permanently delete the amenity instead of soft delete")
):
    """Soft delete (or hard delete) an amenity."""
    if hard_delete:
        success = await amenity_repository.hard_delete_amenity(amenity_id, hotel_id)
    else:
        success = await amenity_repository.delete_amenity(amenity_id, hotel_id)
        
    if not success:
        raise HTTPException(status_code=404, detail="Amenity not found or already deleted")
    return {"status": "success", "message": "Amenity deleted successfully"}
