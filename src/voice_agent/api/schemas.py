"""The shape of what the web endpoints send back."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str                    # "ok", or "degraded" when the database cannot be reached
    version: str
    database: bool
    models: dict[str, str]
    login_required: bool


class BookingEvent(BaseModel):
    """One booking event the (pretend) clinic booking system has received."""

    model_config = ConfigDict(extra="allow")       # created / rescheduled / cancelled events carry a few more fields

    booking_id: int
    reference: str
    caller_name: str
    reason: str
    preferred_time: str
    phone: str
    received_at: str
