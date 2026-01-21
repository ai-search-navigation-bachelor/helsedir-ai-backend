"""
Click log entity representing user click events.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ClickLog(BaseModel):
    """Entity representing a click log entry."""
    id: Optional[int] = None
    search_id: str
    content_id: str
    position: Optional[int] = None
    query: Optional[str] = None
    role: Optional[str] = None
    timestamp: datetime
