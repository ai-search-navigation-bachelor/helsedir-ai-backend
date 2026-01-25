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
    content_type: str
    target_groups: List[str] = []

    @property
    def info_type(self) -> str:
        """Alias for content_type (used by ranking features)."""
        return self.content_type
