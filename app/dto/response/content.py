"""
Content response DTOs.
"""

from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class ContentLinkResponse(BaseModel):
    """Link to related content."""
    rel: str  # forelder, barn, root, publikasjon, temaside
    type: str  # kapittel, pakkeforlop-anbefaling, nasjonalt-forlop, temaside, etc.
    title: Optional[str] = None
    # For internal links (in our database): use id
    # For external links (not in database): use href
    id: Optional[str] = None
    href: Optional[str] = None
    path: Optional[str] = None
    last_reviewed_date: Optional[datetime] = None
    children: Optional[List["ContentLinkResponse"]] = None  # Populated for barn links (1st level only)
    tags: List[str] = Field(default_factory=list)

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


# Resolve the recursive forward reference for the children field.
ContentLinkResponse.model_rebuild()


class LinkedContentItem(BaseModel):
    """A single linked content item under a theme page."""
    id: str
    title: str
    short_title: Optional[str] = None
    display_title: str
    info_type: str
    path: Optional[str] = None
    has_text_content: bool = False
    document_url: Optional[str] = None
    is_pdf_only: bool = False
    tags: List[str] = Field(default_factory=list)


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


class ContentSummaryResponse(BaseModel):
    """Presentation-friendly summary for related content."""
    detail_level: str = "summary"
    id: Optional[str] = None
    title: str = ""
    short_title: Optional[str] = None
    display_title: str = ""
    content_type: str
    path: Optional[str] = None
    href: Optional[str] = None

    @model_validator(mode='after')
    def validate_id_or_href(self) -> "ContentSummaryResponse":
        """Ensure the summary can still identify external-only content."""
        has_id = self.id is not None and self.id.strip() != ""
        has_href = self.href is not None and self.href.strip() != ""

        if not has_id and not has_href:
            raise ValueError("ContentSummaryResponse must have either 'id' or 'href'")

        return self


class ChildGroupResponse(BaseModel):
    """Presentation-ready grouped child content."""
    key: str
    label: str
    items: List[ContentSummaryResponse]


class RelatedLinkResponse(BaseModel):
    """Related report/document link resolved from public Helsedir pages."""
    title: str
    url: str
    is_document: bool = False
    file_type: Optional[str] = None
    url_type: Optional[str] = None
    target: Optional[str] = None
    path: Optional[str] = None
    content_id: Optional[str] = None


class EhelsestandardAttachmentResponse(BaseModel):
    """Normalized attachment metadata for e-helsestandard content."""
    title: str
    url: str
    file_type: Optional[str] = None


class EhelsestandardFieldsResponse(BaseModel):
    """Frontend-friendly fields from Helsedir file payloads."""
    standard_id: Optional[str] = None
    standard_type: Optional[str] = None
    purpose_html: Optional[str] = None
    applies_to_html: Optional[str] = None
    attachments: List[EhelsestandardAttachmentResponse] = []


class ContentResponse(BaseModel):
    """Response model for content endpoint."""
    detail_level: str = "full"
    id: str
    title: str
    short_title: Optional[str] = None
    display_title: str
    body: str
    content_type: str
    path: Optional[str] = None
    first_published: Optional[datetime] = None
    last_reviewed_date: Optional[datetime] = None
    role_tags: List[str] = []
    has_text_content: bool = False
    document_url: Optional[str] = None
    is_pdf_only: bool = False
    url: Optional[str] = None
    links: List[ContentLinkResponse] = []
    parent: Optional[ContentSummaryResponse] = None
    root_publication: Optional[ContentSummaryResponse] = None
    chapters: List[ContentSummaryResponse] = []
    references: List[ContentSummaryResponse] = []
    related_content: List[ContentSummaryResponse] = []
    child_groups: List[ChildGroupResponse] = []
    linked_content: Optional[List[GroupedLinkedContent]] = None  # For theme pages
    related_links: Optional[List[RelatedLinkResponse]] = None

    # Info type-specific fields (extensible pattern for future types)
    anbefaling_fields: Optional[AnbefalingFieldsResponse] = None
    ehelsestandard_fields: Optional[EhelsestandardFieldsResponse] = None
