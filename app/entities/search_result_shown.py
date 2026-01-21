"""
Search result shown entity for learning-to-rank data.
"""

from pydantic import BaseModel
from typing import Optional


class SearchResultShown(BaseModel):
    """Entity representing a search result that was shown to the user."""
    id: Optional[int] = None
    search_id: str
    content_id: str
    position: int
    score: float = 0.0
