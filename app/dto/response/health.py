"""
Health response DTOs.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
