"""
Content statistics entity.
"""

from pydantic import BaseModel
from typing import Optional


class ContentStats(BaseModel):
    """Entity representing content performance statistics."""
    content_id: str
    impressions: int = 0
    clicks: int = 0
    ctr: Optional[float] = None  # Click-through rate (virtual generated)
