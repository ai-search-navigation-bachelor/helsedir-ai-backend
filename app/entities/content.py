"""
Entity models.

Core business entities that represent the domain model.
"""

from typing import List, Optional, Self

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.services.data.document_metadata import has_visible_text, resolve_public_document_url


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
    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    body: str
    content_type: str
    path: Optional[str] = None
    role_tags: List[str] = []
    links: List[ContentLink] = []

    # Info type-specific fields (extensible pattern for future types)
    anbefaling_fields: Optional[AnbefalingFields] = None

    @model_validator(mode='after')
    def populate_content_metadata(self) -> Self:
        """Ensure text metadata is consistent for in-memory content items."""
        self.has_text_content = bool(has_visible_text(self.body))
        return self

    @property
    def info_type(self) -> str:
        """Alias for content_type (used by ranking features)."""
        return self.content_type

    @property
    def target_groups(self) -> List[str]:
        """Backward-compatible alias for role_tags."""
        return self.role_tags

    @property
    def is_pdf_only(self) -> bool:
        """True when content has no text body but points to a document."""
        return not self.has_text_content and bool(self.document_url)

    @property
    def public_document_url(self) -> Optional[str]:
        """Preferred frontend URL for PDF-only content."""
        if not self.is_pdf_only:
            return None
        return resolve_public_document_url(self.path, self.document_url)
