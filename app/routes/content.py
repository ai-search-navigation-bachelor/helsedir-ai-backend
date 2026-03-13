"""
Content route for retrieving content and logging clicks.
"""

import asyncio
import logging
import re
from fastapi import APIRouter, HTTPException, Query
from typing import Any, Mapping, Optional, Dict, List
from urllib.parse import urljoin, urlparse

from collections import defaultdict
from starlette.concurrency import run_in_threadpool
from app.dto.response.content import (
    ContentResponse,
    ContentLinkResponse,
    GroupedLinkedContent,
    LinkedContentItem,
    AnbefalingFieldsResponse,
    EhelsestandardAttachmentResponse,
    EhelsestandardFieldsResponse,
    ContentSummaryResponse,
    ChildGroupResponse,
    RelatedLinkResponse,
)
from app.entities.content import ContentItem, ContentLink
from app.services.data.content_service import content_service
from app.services.data.document_metadata import (
    build_content_metadata,
    has_visible_text,
    resolve_public_related_links,
)
from app.services.data.database_service import database_service
from app.services.repositories.content_repository import content_repository
from app.services.external.helsedir_api_service import helsedir_api_service
from app.constants import get_category_display_name, normalize_content_type
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content", tags=["content"])

_HELSEDIR_ID_RE = re.compile(r'/innhold/[^/]+/([0-9]{4}-[0-9]{4}[-a-f0-9]+)', re.IGNORECASE)
_CHILD_GROUP_LABELS = {
    "chapters": "Kapitler",
    "references": "Referanser",
    "related_content": "Relatert innhold",
}


def _id_from_href(href: str) -> Optional[str]:
    """Extract the content ID embedded in a Helsedir API href URL."""
    m = _HELSEDIR_ID_RE.search(href)
    return m.group(1) if m else None


def _public_path_from_href(href: str) -> Optional[str]:
    """Extract a public Helsedirektoratet path from a path or public URL."""
    if not href:
        return None

    value = href.strip()
    if not value:
        return None

    if value.startswith("/"):
        return value.rstrip("/") or "/"

    parsed = urlparse(value)
    public_base = urlparse(settings.helsedir_public_base_url)
    if parsed.scheme and parsed.netloc:
        if parsed.netloc != public_base.netloc:
            return None
        return (parsed.path or "").rstrip("/") or "/"

    return None


def _is_api_href(href: str) -> bool:
    """Return True when href points to the Helsedir API domain."""
    if not href:
        return False

    parsed = urlparse(href.strip())
    expected = urlparse(settings.helsedir_api_url)
    return bool(parsed.scheme and parsed.netloc) and (
        parsed.scheme == expected.scheme and parsed.netloc == expected.netloc
    )


def _children_from_content_links(links: List) -> List[ContentLinkResponse]:
    """Build ContentLinkResponse list from a content item's barn sub-links."""
    result = []
    for gl in links:
        if gl.rel != "barn":
            continue
        last_reviewed_date = None
        if gl.id:
                child = content_service.get_content_by_id(gl.id)
                if child:
                    last_reviewed_date = child.sist_faglig_oppdatert
        result.append(ContentLinkResponse(
            rel=gl.rel,
            type=gl.type,
            title=gl.tittel,
            id=gl.id,
            href=gl.href,
            last_reviewed_date=last_reviewed_date,
        ))
    return result


async def _build_links_with_children(links: List[ContentLink]) -> List[ContentLinkResponse]:
    """
    Build links response, populating children for barn links in parallel.

    For each barn link:
    - id-based → cache lookup (O(1))
    - href-based → async Helsedir API call

    All links are processed concurrently via asyncio.gather, preserving order.
    Non-barn links are returned with children=[].
    """
    async def _build_link(link: ContentLink) -> ContentLinkResponse:
        children: List[ContentLinkResponse] = []
        last_reviewed_date = None

        # Look up cached content for id-based links
        cached = None
        if link.id:
            cached = content_service.get_content_by_id(link.id)
        elif link.href:
            href_id = _id_from_href(link.href)
            if href_id:
                cached = content_service.get_content_by_id(href_id)

        if cached:
            last_reviewed_date = cached.sist_faglig_oppdatert

        if link.rel == "barn":
            if cached:
                children = _children_from_content_links(cached.links)
            elif link.href:
                # Try cache first by extracting the ID from the href URL
                child = content_service.get_content_by_id(_id_from_href(link.href) or "")
                if child:
                    children = _children_from_content_links(child.links)
                else:
                    public_path = _public_path_from_href(link.href)
                    if public_path:
                        child = content_service.get_content_by_path(public_path)
                        if child:
                            children = _children_from_content_links(child.links)
                    elif _is_api_href(link.href):
                        # Fallback: fetch from Helsedir API only for API URLs
                        try:
                            data = await helsedir_api_service.get_content_by_href_async(link.href)
                            children = [
                                ContentLinkResponse(
                                    rel=al.get("rel", "barn"),
                                    type=al.get("type") or al.get("infoType", ""),
                                    title=al.get("tittel"),
                                    id=al.get("id"),
                                    href=al.get("href"),
                                )
                                for al in (data.get("links") or [])
                                if al.get("rel") == "barn"
                                and (al.get("id") or al.get("href"))
                            ]
                        except Exception as exc:
                            logger.warning(
                                "Failed to fetch children for barn link %s: %s",
                                link.href,
                                exc,
                                exc_info=True,
                            )

        return ContentLinkResponse(
            rel=link.rel,
            type=link.type,
            title=link.tittel,
            id=link.id,
            href=link.href,
            path=link.path,
            last_reviewed_date=last_reviewed_date,
            children=children,
        )

    return list(await asyncio.gather(*[_build_link(link) for link in links]))


def _get_link_target_id(link: ContentLinkResponse) -> Optional[str]:
    """Resolve a link to its canonical content ID when possible."""
    if link.id:
        return link.id
    if link.href:
        return _id_from_href(link.href)
    return None


def _build_content_summary(link: ContentLinkResponse) -> ContentSummaryResponse:
    """Build a normalized summary object for a related content link."""
    target_id = _get_link_target_id(link)
    linked_content = content_service.get_content_by_id(target_id) if target_id else None

    return ContentSummaryResponse(
        id=linked_content.id if linked_content else target_id,
        title=linked_content.title if linked_content else (link.title or ""),
        content_type=normalize_content_type(
            linked_content.content_type if linked_content else link.type
        ),
        path=linked_content.path if linked_content else link.path,
        href=link.href,
    )


def _classify_child_summary(summary: ContentSummaryResponse) -> str:
    """Map child content into presentation-friendly groups."""
    if summary.content_type == "referanse":
        return "references"
    if summary.content_type == "kapittel":
        return "chapters"
    return "related_content"


def _append_unique_summary(
    items: List[ContentSummaryResponse],
    seen: set,
    summary: ContentSummaryResponse,
) -> None:
    """Append a summary once per ID/href/content_type combination."""
    dedupe_key = (summary.id or "", summary.href or "", summary.content_type)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    items.append(summary)


def _build_child_groups(
    chapters: List[ContentSummaryResponse],
    references: List[ContentSummaryResponse],
    related_content: List[ContentSummaryResponse],
) -> List[ChildGroupResponse]:
    """Build grouped child payloads for generic frontend rendering."""
    groups = []
    for key, items in (
        ("chapters", chapters),
        ("references", references),
        ("related_content", related_content),
    ):
        if not items:
            continue
        groups.append(
            ChildGroupResponse(
                key=key,
                label=_CHILD_GROUP_LABELS[key],
                items=items,
            )
        )
    return groups


def _normalize_relations(
    links: List[ContentLinkResponse],
) -> Dict[str, Optional[object]]:
    """Normalize raw links into explicit backend-owned relation fields."""
    parent = None
    root_publication = None
    chapters: List[ContentSummaryResponse] = []
    references: List[ContentSummaryResponse] = []
    related_content: List[ContentSummaryResponse] = []
    seen_groups = {
        "chapters": set(),
        "references": set(),
        "related_content": set(),
    }

    for link in links:
        summary = _build_content_summary(link)

        if link.rel == "forelder" and parent is None:
            parent = summary
            continue

        if link.rel in {"root", "publikasjon"} and root_publication is None:
            root_publication = summary
            continue

        if link.rel not in {"barn", "temaside"}:
            continue

        group_key = _classify_child_summary(summary)
        target = {
            "chapters": chapters,
            "references": references,
            "related_content": related_content,
        }[group_key]
        _append_unique_summary(target, seen_groups[group_key], summary)

    return {
        "parent": parent,
        "root_publication": root_publication,
        "chapters": chapters,
        "references": references,
        "related_content": related_content,
        "child_groups": _build_child_groups(chapters, references, related_content),
    }


def _get_theme_page_linked_content(theme_page_id: str) -> Optional[List[GroupedLinkedContent]]:
    """
    Fetch and group linked content for a theme page.

    Args:
        theme_page_id: ID of the theme page

    Returns:
        List of grouped linked content by info_type, or None if no content found
    """
    # Fetch linked content from database
    linked_content = content_repository.get_theme_page_content(theme_page_id)

    if not linked_content:
        return None

    # Group by info_type
    grouped: Dict[str, List[LinkedContentItem]] = defaultdict(list)
    for content_item in linked_content:
        info_type = content_item.get('info_type', '').lower()
        if not info_type:
            continue

        metadata = build_content_metadata(
            content_item.get('path'),
            content_item.get('has_text_content'),
            content_item.get('document_url'),
        )

        # Create LinkedContentItem
        linked_item = LinkedContentItem(
            id=content_item.get('id', ''),
            title=content_item.get('tittel', ''),
            info_type=info_type,
            path=content_item.get('path'),
            has_text_content=metadata["has_text_content"],
            document_url=metadata["document_url"],
            is_pdf_only=metadata["is_pdf_only"],
        )
        grouped[info_type].append(linked_item)

    # Convert to GroupedLinkedContent list
    result = []
    for info_type, items in sorted(grouped.items()):
        result.append(GroupedLinkedContent(
            info_type=info_type,
            display_name=get_category_display_name(info_type),
            items=items,
        ))

    return result if result else None


def _should_resolve_related_links(content: ContentItem) -> bool:
    """Return True for report pages that have no text and no primary document URL."""
    if content.content_type.lower() != "rapport":
        return False
    if content.has_text_content or content.document_url:
        return False

    useful_link_rels = {link.rel for link in content.links if link.rel}
    return useful_link_rels.issubset({"root", "publikasjon", "temaside"})


def _get_report_related_links(content: ContentItem) -> Optional[List[RelatedLinkResponse]]:
    """Resolve related report/document links from the public HTML page."""
    if not _should_resolve_related_links(content):
        return None

    resolved_links = resolve_public_related_links(content.path)
    if not resolved_links:
        return None

    normalized_links = []
    for link in resolved_links:
        raw_url = link["url"]
        internal_path = _public_path_from_href(raw_url)
        internal_content = (
            content_service.get_content_by_path(internal_path)
            if internal_path and not bool(link.get("is_document"))
            else None
        )
        normalized_links.append(
            RelatedLinkResponse(
                title=link["title"],
                url=internal_content.path if internal_content and internal_content.path else raw_url,
                is_document=bool(link.get("is_document")),
                file_type=link.get("file_type"),
                url_type=(
                    "internal"
                    if internal_content and internal_content.path
                    else link.get("url_type")
                ),
                target=link.get("target"),
                path=internal_content.path if internal_content else internal_path,
                content_id=internal_content.id if internal_content else None,
            )
        )

    return normalized_links



def _resolve_ehelsestandard_attachment_url(file_uri: Optional[str]) -> Optional[str]:
    """Return an absolute public attachment URL for e-helsestandard files."""
    if not file_uri or not file_uri.strip():
        return None

    normalized = file_uri.strip()
    parsed = urlparse(normalized)
    if parsed.scheme and parsed.netloc:
        return normalized

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return urljoin(settings.helsedir_public_base_url, normalized)


def _build_public_content_url(path: Optional[str]) -> Optional[str]:
    """Return the public Helsedirektoratet URL derived from a stored path."""
    if not path or not path.strip():
        return None
    normalized = path.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return urljoin(settings.helsedir_public_base_url, normalized)


def _normalize_ehelsestandard_attachments(
    payload: Mapping[str, Any],
) -> List[EhelsestandardAttachmentResponse]:
    """Map raw Helsedir attachments to a frontend-friendly shape."""
    raw_attachments = payload.get("attachments")
    if not isinstance(raw_attachments, list):
        return []

    normalized_attachments: List[EhelsestandardAttachmentResponse] = []
    for attachment in raw_attachments:
        if not isinstance(attachment, Mapping):
            continue

        attachment_url = _resolve_ehelsestandard_attachment_url(
            attachment.get("fileUri")
            if isinstance(attachment.get("fileUri"), str)
            else attachment.get("file_uri")
            if isinstance(attachment.get("file_uri"), str)
            else attachment.get("url")
            if isinstance(attachment.get("url"), str)
            else attachment.get("href")
            if isinstance(attachment.get("href"), str)
            else None
        )
        if not attachment_url:
            continue

        title = (
            attachment.get("title")
            if isinstance(attachment.get("title"), str)
            else attachment.get("tittel")
            if isinstance(attachment.get("tittel"), str)
            else attachment.get("name")
            if isinstance(attachment.get("name"), str)
            else attachment_url.rstrip("/").rsplit("/", 1)[-1]
        )
        file_type = (
            attachment.get("fileType")
            if isinstance(attachment.get("fileType"), str)
            else attachment.get("file_type")
            if isinstance(attachment.get("file_type"), str)
            else attachment.get("type")
            if isinstance(attachment.get("type"), str)
            else None
        )
        normalized_attachments.append(
            EhelsestandardAttachmentResponse(
                title=title.strip(),
                url=attachment_url,
                file_type=file_type,
            )
        )

    return normalized_attachments


def _build_ehelsestandard_response_overrides_from_content(
    content: ContentItem,
) -> Optional[Dict[str, Any]]:
    """Build overrides from stored e-helsestandard DB details when available."""
    if content.content_type.lower() != "ehelsestandard":
        return None

    body_has_visible_text = has_visible_text(content.body)
    stored_fields = content.ehelsestandard_fields
    has_stored_detail_data = stored_fields is not None

    if not has_stored_detail_data:
        return None

    response_attachments = [
        EhelsestandardAttachmentResponse(
            title=attachment.title,
            url=attachment.url,
            file_type=attachment.file_type,
        )
        for attachment in (stored_fields.attachments if stored_fields else [])
    ]
    purpose_html = stored_fields.purpose_html if stored_fields else None
    final_body = content.body if body_has_visible_text else (purpose_html or content.body)
    final_has_text_content = has_visible_text(final_body)
    first_attachment_url = response_attachments[0].url if response_attachments else None

    return {
        "body": final_body,
        "has_text_content": final_has_text_content,
        "document_url": first_attachment_url or content.document_url,
        "is_pdf_only": (not final_has_text_content) and bool(first_attachment_url),
        "first_published": content.forst_publisert,
        "last_reviewed_date": content.sist_faglig_oppdatert,
        "url": _build_public_content_url(content.path),
        "ehelsestandard_fields": (
            EhelsestandardFieldsResponse(
                standard_id=stored_fields.standard_id,
                standard_type=stored_fields.standard_type,
                purpose_html=final_body if final_has_text_content else None,
                applies_to_html=stored_fields.applies_to_html,
                attachments=response_attachments,
            )
            if stored_fields
            else None
        ),
    }


async def _get_ehelsestandard_response_overrides(
    content: ContentItem,
) -> Optional[Dict[str, Any]]:
    """Build request-time overrides for e-helsestandard detail responses."""
    if content.content_type.lower() != "ehelsestandard":
        return None

    db_overrides = _build_ehelsestandard_response_overrides_from_content(content)
    if db_overrides is not None:
        return db_overrides

    try:
        logger.debug("Using runtime HAPI fallback for ehelsestandard content_id=%s", content.id)
        payload = await helsedir_api_service.get_file_by_id_async(content.id)
    except Exception as exc:
        logger.warning(
            "Failed to enrich ehelsestandard content_id=%s: %s",
            content.id,
            exc,
            exc_info=True,
        )
        return None

    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    purpose_html = data.get("formalBruksomrade") if isinstance(data.get("formalBruksomrade"), str) else None
    applies_to_html = data.get("standardenGjelderFor") if isinstance(data.get("standardenGjelderFor"), str) else None
    body_has_visible_text = has_visible_text(content.body)
    final_body = content.body if body_has_visible_text else (purpose_html or content.body)
    final_has_text_content = has_visible_text(final_body)
    attachments = _normalize_ehelsestandard_attachments(payload)
    public_url = payload.get("url") if isinstance(payload.get("url"), str) else None

    ehelsestandard_fields = EhelsestandardFieldsResponse(
        standard_id=data.get("idStandard") if isinstance(data.get("idStandard"), str) else None,
        standard_type=data.get("typeStandard") if isinstance(data.get("typeStandard"), str) else None,
        purpose_html=purpose_html,
        applies_to_html=applies_to_html,
        attachments=attachments,
    )

    return {
        "body": final_body,
        "has_text_content": final_has_text_content,
        "document_url": attachments[0].url if attachments else None,
        "is_pdf_only": (not final_has_text_content) and bool(attachments),
        "first_published": payload.get("forstPublisert") or content.forst_publisert,
        "last_reviewed_date": payload.get("sistFagligOppdatert") or content.sist_faglig_oppdatert,
        "url": public_url or _build_public_content_url(content.path),
        "ehelsestandard_fields": ehelsestandard_fields,
    }


async def _build_content_response(content: ContentItem, search_id: Optional[str] = None) -> ContentResponse:
    """Build ContentResponse from a ContentItem, with parallel link enrichment and click logging."""
    coros = [_build_links_with_children(content.links), _get_ehelsestandard_response_overrides(content)]
    if search_id:
        coros.append(run_in_threadpool(
            database_service.log_click,
            search_id=search_id,
            content_id=content.id,
        ))
    should_resolve_related_links = _should_resolve_related_links(content)
    if should_resolve_related_links:
        coros.append(run_in_threadpool(_get_report_related_links, content))
    results = await asyncio.gather(*coros, return_exceptions=True)
    links_result = results[0]
    if isinstance(links_result, BaseException):
        raise links_result
    links_response = links_result
    overrides_result = results[1]
    if isinstance(overrides_result, BaseException):
        raise overrides_result
    response_overrides = overrides_result or {}
    next_result_index = 2
    if search_id:
        if isinstance(results[next_result_index], BaseException):
            logger.warning(
                "Failed to log click for search_id=%s: %s",
                search_id,
                results[next_result_index],
                exc_info=results[next_result_index],
            )
        next_result_index += 1

    linked_content_response = None
    if content.content_type.lower() == "temaside":
        linked_content_response = _get_theme_page_linked_content(content.id)

    related_links_response = None
    if should_resolve_related_links:
        related_links_result = results[next_result_index]
        if isinstance(related_links_result, BaseException):
            logger.warning(
                "Failed to resolve related links for content_id=%s: %s",
                content.id,
                related_links_result,
                exc_info=related_links_result,
            )
        else:
            related_links_response = related_links_result

    anbefaling_fields_response = None
    if content.anbefaling_fields:
        anbefaling_fields_response = AnbefalingFieldsResponse(
            praktisk=content.anbefaling_fields.praktisk,
            rasjonale=content.anbefaling_fields.rasjonale,
            fordeler_ulemper=content.anbefaling_fields.fordeler_ulemper,
            verdier_preferanser=content.anbefaling_fields.verdier_preferanser,
            kvalitet_dokumentasjon=content.anbefaling_fields.kvalitet_dokumentasjon,
            ressurshensyn=content.anbefaling_fields.ressurshensyn,
            styrke=content.anbefaling_fields.styrke,
        )

    normalized_relations = _normalize_relations(links_response)
    response_body = response_overrides.get("body", content.body)
    response_has_text_content = response_overrides.get("has_text_content", content.has_text_content)
    response_document_url = response_overrides.get("document_url", content.public_document_url)
    response_is_pdf_only = response_overrides.get("is_pdf_only", content.is_pdf_only)
    response_first_published = response_overrides.get("first_published", content.forst_publisert)
    response_last_reviewed_date = response_overrides.get("last_reviewed_date", content.sist_faglig_oppdatert)
    response_url = response_overrides.get("url")
    response_ehelsestandard_fields = response_overrides.get("ehelsestandard_fields")

    return ContentResponse(
        id=content.id,
        title=content.title,
        body=response_body,
        content_type=normalize_content_type(content.content_type),
        path=content.path,
        first_published=response_first_published,
        last_reviewed_date=response_last_reviewed_date,
        role_tags=content.role_tags,
        has_text_content=response_has_text_content,
        document_url=response_document_url,
        is_pdf_only=response_is_pdf_only,
        url=response_url,
        links=links_response,
        parent=normalized_relations["parent"],
        root_publication=normalized_relations["root_publication"],
        chapters=normalized_relations["chapters"],
        references=normalized_relations["references"],
        related_content=normalized_relations["related_content"],
        child_groups=normalized_relations["child_groups"],
        linked_content=linked_content_response,
        related_links=related_links_response,
        anbefaling_fields=anbefaling_fields_response,
        ehelsestandard_fields=response_ehelsestandard_fields,
    )


@router.get("/by-path", response_model=ContentResponse)
async def get_content_by_path(
    path: str = Query(..., description="Content path, e.g. /retningslinjer/adhd"),
    search_id: Optional[str] = Query(None, description="Search ID for click tracking"),
):
    """
    Get content by path and optionally log click.

    Used when the frontend navigates to a path-based URL (e.g. /retningslinjer/adhd)
    and needs to resolve the content.
    """
    content = content_service.get_content_by_path(path)

    if not content:
        raise HTTPException(status_code=404, detail=f"Content not found for path: {path}")

    return await _build_content_response(content, search_id)


@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(
    content_id: str,
    search_id: Optional[str] = Query(None, description="Search ID for click tracking"),
):
    """
    Get content by ID and optionally log click.

    If search_id is provided, logs the click for LTR training.
    The position is automatically looked up from search_results_shown.
    """
    content = content_service.get_content_by_id(content_id)

    if not content:
        raise HTTPException(status_code=404, detail=f"Content not found: {content_id}")

    return await _build_content_response(content, search_id)
