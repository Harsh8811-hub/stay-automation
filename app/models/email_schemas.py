"""
Email API Schemas for guest email ingestion and context-aware reply generation.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class EmailMessage(BaseModel):
    """A single email turn in a thread."""
    sender: str = Field(description="'guest' or 'hotel'")
    body: str = Field(description="Body of the email message")
    timestamp: str | None = Field(default=None, description="ISO timestamp of the message if available")


class EmailReplyRequest(BaseModel):
    """Incoming request to ingest a guest email and generate a response."""
    hotel_id: int | str = Field(description="Hotel ID (e.g. 111111 or '111111')")
    from_email: str = Field(description="Sender's email address")
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(description="Email body text from the guest")
    guest_name: str | None = Field(default=None, description="Optional guest name if known")
    booking_id: str | None = Field(default=None, description="Optional booking ID reference")
    thread_history: list[EmailMessage] = Field(default_factory=list, description="Prior thread messages")
    preferred_language: str | None = Field(default=None, description="Preferred response language, default English")


class EmailReplyResponse(BaseModel):
    """Response containing the generated email text and context metadata."""
    success: bool = True
    subject: str = Field(description="Subject line for the reply")
    reply_body: str = Field(description="Full text of the generated email reply")
    guest_name: str | None = Field(default=None, description="Resolved or extracted guest name")
    detected_intent: str | None = Field(default=None, description="Classified intent (FAQ, RESERVATION_INQUIRY, EXISTING_BOOKING, etc.)")
    booking_reference: str | None = Field(default=None, description="Referenced booking ID if resolved")
    context_sources_used: list[str] = Field(default_factory=list, description="List of context sources retrieved and incorporated")
