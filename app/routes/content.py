"""
Content route for retrieving content and logging clicks.
"""

import asyncio
import logging
import re
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, List

from collections import defaultdict
from starlette.concurrency import run_in_threadpool
from app.dto.response.content import (
    ContentResponse,
    ContentLinkResponse,
    GroupedLinkedContent,
    LinkedContentItem,
    AnbefalingFieldsResponse,
)
from app.entities.content import ContentItem, ContentLink
from app.services.data.content_service import content_service
from app.services.data.document_metadata import build_content_metadata
from app.services.data.database_service import database_service
from app.services.repositories.content_repository import content_repository
from app.services.external.helsedir_api_service import helsedir_api_service
from app.exceptions.helsedir import HelseDirectorateAPIError
from app.constants import get_category_display_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content", tags=["content"])

_HELSEDIR_ID_RE = re.compile(r'/innhold/[^/]+/([0-9]{4}-[0-9]{4}[-a-f0-9]+)', re.IGNORECASE)


def _id_from_href(href: str) -> Optional[str]:
    """Extract the content ID embedded in a Helsedir API href URL."""
    m = _HELSEDIR_ID_RE.search(href)
    return m.group(1) if m else None


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
                # Fallback: fetch from Helsedir API
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
                except HelseDirectorateAPIError as exc:
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


async def _build_content_response(content: ContentItem, search_id: Optional[str] = None) -> ContentResponse:
    """Build ContentResponse from a ContentItem, with parallel link enrichment and click logging."""
    coros = [_build_links_with_children(content.links)]
    if search_id:
        coros.append(run_in_threadpool(
            database_service.log_click,
            search_id=search_id,
            content_id=content.id,
        ))
    results = await asyncio.gather(*coros, return_exceptions=True)
    links_result = results[0]
    if isinstance(links_result, BaseException):
        raise links_result
    links_response = links_result
    if len(results) > 1 and isinstance(results[1], BaseException):
        logger.warning("Failed to log click for search_id=%s: %s", search_id, results[1], exc_info=results[1])

    linked_content_response = None
    if content.content_type.lower() == "temaside":
        linked_content_response = _get_theme_page_linked_content(content.id)

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

    return ContentResponse(
        id=content.id,
        title=content.title,
        body=content.body,
        content_type=content.content_type,
        path=content.path,
        first_published=content.forst_publisert,
        last_reviewed_date=content.sist_faglig_oppdatert,
        role_tags=content.role_tags,
        has_text_content=content.has_text_content,
        document_url=content.public_document_url,
        is_pdf_only=content.is_pdf_only,
        links=links_response,
        linked_content=linked_content_response,
        anbefaling_fields=anbefaling_fields_response,
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
