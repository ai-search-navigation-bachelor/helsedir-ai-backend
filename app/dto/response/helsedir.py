"""
Helsedirektoratet response DTOs.
"""

from typing import List
from pydantic import BaseModel


class HelseDirectorateSearchResponse(BaseModel):
    """Response from Helsedirektoratet API search."""
    results: List[dict]
    total: int
    query: str
