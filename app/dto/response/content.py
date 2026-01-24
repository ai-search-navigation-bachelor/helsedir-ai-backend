"""
Content response DTOs.
"""

from pydantic import BaseModel
from typing import List


class ContentResponse(BaseModel):
    """Response model for content endpoint."""
    id: str
    title: str
    body: str
    url: str
    content_type: str
    target_groups: List[str] = []
