"""
Search log entity representing search queries.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SearchLog(BaseModel):
    """Entity representing a search log entry."""
    id: Optional[int] = None
    search_id: str
    query: str
    role: Optional[str] = None
    results_count: int = 0
    timestamp: datetime
