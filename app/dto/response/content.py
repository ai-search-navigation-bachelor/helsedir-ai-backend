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


class LinkedContentItem(BaseModel):
    """A single linked content item under a theme page."""
    id: str
    title: str
    info_type: str


class GroupedLinkedContent(BaseModel):
    """Linked content grouped by info_type."""
    info_type: str
    display_name: str
    items: List[LinkedContentItem]


class ContentResponse(BaseModel):
    """Response model for content endpoint."""
    id: str
    title: str
    body: str
    content_type: str
    target_groups: List[str] = []
    links: List[ContentLinkResponse] = []
    linked_content: Optional[List[GroupedLinkedContent]] = None  # For theme pages
