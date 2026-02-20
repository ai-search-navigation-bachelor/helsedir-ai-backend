"""
Content response DTOs.
"""

from pydantic import BaseModel, model_validator
from typing import List, Optional


class ContentLinkResponse(BaseModel):
    """Link to related content."""
    rel: str  # forelder, barn, root, publikasjon
    type: str  # kapittel, pakkeforlop-anbefaling, nasjonalt-forlop, etc.
    tittel: Optional[str] = None
    # For internal links (in our database): use id
    # For external links (not in database): use href
    id: Optional[str] = None
    href: Optional[str] = None

    @model_validator(mode='after')
    def validate_id_or_href(self):
        """Ensure exactly one of 'id' or 'href' is provided (and not empty/whitespace)."""
        # Treat empty/whitespace-only strings as missing
        has_id = self.id is not None and self.id.strip() != ""
        has_href = self.href is not None and self.href.strip() != ""

        if not has_id and not has_href:
            raise ValueError("ContentLinkResponse must have either 'id' or 'href'")
        if has_id and has_href:
            raise ValueError("ContentLinkResponse cannot have both 'id' and 'href'")

        return self


class LinkedContentItem(BaseModel):
    """A single linked content item under a theme page."""
    id: str
    title: str
    info_type: str
    path: Optional[str] = None


class GroupedLinkedContent(BaseModel):
    """Linked content grouped by info_type."""
    info_type: str
    display_name: str
    items: List[LinkedContentItem]


class AnbefalingFieldsResponse(BaseModel):
    """Anbefaling-specific fields."""
    praktisk: Optional[str] = None
    rasjonale: Optional[str] = None
    fordeler_ulemper: Optional[str] = None
    verdier_preferanser: Optional[str] = None
    kvalitet_dokumentasjon: Optional[str] = None
    ressurshensyn: Optional[str] = None
    styrke: Optional[str] = None


class ContentResponse(BaseModel):
    """Response model for content endpoint."""
    id: str
    title: str
    body: str
    content_type: str
    path: Optional[str] = None
    target_groups: List[str] = []
    links: List[ContentLinkResponse] = []
    linked_content: Optional[List[GroupedLinkedContent]] = None  # For theme pages

    # Info type-specific fields (extensible pattern for future types)
    anbefaling_fields: Optional[AnbefalingFieldsResponse] = None
