"""
Content response DTOs.
"""

from pydantic import BaseModel
from typing import List, Optional


class ContentLinkResponse(BaseModel):
    """Link to related content."""
    rel: str  # forelder, barn, root, publikasjon
    type: str  # kapittel, pakkeforlop-anbefaling, nasjonalt-forlop, etc.
    tittel: Optional[str] = None
    href: str
    strukturId: Optional[str] = None


class ContentResponse(BaseModel):
    """Response model for content endpoint."""
    id: str
    title: str
    body: str
    content_type: str
    target_groups: List[str] = []
    links: List[ContentLinkResponse] = []
