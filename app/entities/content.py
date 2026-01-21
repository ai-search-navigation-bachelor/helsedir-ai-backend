"""
Entity models.

Core business entities that represent the domain model.
"""

from pydantic import BaseModel
from typing import Optional, List


class ContentItem(BaseModel):
    """Content item entity."""
    id: str
    title: str
    body: str
    url: str
    content_type: str
    target_groups: List[str] = []
    tags: Optional[List[str]] = []
