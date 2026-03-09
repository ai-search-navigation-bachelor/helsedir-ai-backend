"""
Entity models.

Core business entities that represent the domain model.
"""

from datetime import datetime
from pydantic import BaseModel, model_validator
from typing import Optional, List


class ContentLink(BaseModel):
    """Link to related content."""
    rel: str  # forelder, barn, root, publikasjon, temaside
    type: str  # kapittel, pakkeforlop-anbefaling, nasjonalt-forlop, temaside, etc.
    tittel: Optional[str] = None
    # For internal links: use id (references content in our database)
    # For external links: use href (points to external URL)
    id: Optional[str] = None
    href: Optional[str] = None
    path: Optional[str] = None
    # Legacy field - will be removed after migration
    strukturId: Optional[str] = None

    @model_validator(mode='after')
    def validate_id_or_href(self):
        """Ensure exactly one of 'id' or 'href' is provided (and not empty/whitespace)."""
        # Treat empty/whitespace-only strings as missing
        has_id = self.id is not None and self.id.strip() != ""
        has_href = self.href is not None and self.href.strip() != ""

        if not has_id and not has_href:
            raise ValueError("ContentLink must have either 'id' or 'href'")
        if has_id and has_href:
            raise ValueError("ContentLink cannot have both 'id' and 'href'")

        return self


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
    path: Optional[str] = None
    forst_publisert: Optional[datetime] = None
    sist_faglig_oppdatert: Optional[datetime] = None
    role_tags: List[str] = []
    links: List[ContentLink] = []

    # Info type-specific fields (extensible pattern for future types)
    anbefaling_fields: Optional[AnbefalingFields] = None

    @property
    def info_type(self) -> str:
        """Alias for content_type (used by ranking features)."""
        return self.content_type
