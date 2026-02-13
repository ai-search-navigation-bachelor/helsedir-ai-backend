"""
Entity models.

Core business entities that represent the domain model.
"""

from pydantic import BaseModel
from typing import Optional, List


class ContentLink(BaseModel):
    """Link to related content."""
    rel: str  # forelder, barn, root, publikasjon
    type: str  # kapittel, pakkeforlop-anbefaling, nasjonalt-forlop, etc.
    tittel: Optional[str] = None
    # For internal links: use id (references content in our database)
    # For external links: use href (points to external URL)
    id: Optional[str] = None
    href: Optional[str] = None
    # Legacy field - will be removed after migration
    strukturId: Optional[str] = None


class AnbefalingFields(BaseModel):
    """Anbefaling-specific fields from /innhold/anbefalinger/{id}."""
    praktisk: Optional[str] = None
    rasjonale: Optional[str] = None
    fordeler_ulemper: Optional[str] = None
    verdier_preferanser: Optional[str] = None
    kvalitet_dokumentasjon: Optional[str] = None
    ressurshensyn: Optional[str] = None
    styrke: Optional[str] = None


class ContentItem(BaseModel):
    """Content item entity."""
    id: str
    title: str
    body: str
    content_type: str
    target_groups: List[str] = []
    links: List[ContentLink] = []

    # Info type-specific fields (extensible pattern for future types)
    anbefaling_fields: Optional[AnbefalingFields] = None

    @property
    def info_type(self) -> str:
        """Alias for content_type (used by ranking features)."""
        return self.content_type
