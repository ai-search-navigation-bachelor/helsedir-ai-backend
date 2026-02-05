"""
Logging response DTOs.
"""

from pydantic import BaseModel


class LogResponse(BaseModel):
    """Response model for logging endpoint."""
    success: bool
