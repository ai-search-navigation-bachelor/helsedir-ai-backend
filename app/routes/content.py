"""
Content route for retrieving content and logging clicks.
"""

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
from app.services.data.content_service import content_service
from app.services.data.database_service import database_service
from app.services.repositories.content_repository import content_repository
from app.constants import get_category_display_name

router = APIRouter(prefix="/content", tags=["content"])


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

        # Create LinkedContentItem
        linked_item = LinkedContentItem(
            id=content_item.get('id', ''),
            title=content_item.get('tittel', ''),
            info_type=info_type,
            path=content_item.get('path'),
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


def _build_content_response(content) -> ContentResponse:
    """Build ContentResponse from a ContentItem."""
    links_response = [
        ContentLinkResponse(
            rel=link.rel,
            type=link.type,
            tittel=link.tittel,
            id=link.id,
            href=link.href,
            path=link.path,
        )
        for link in content.links
    ]

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
        target_groups=content.target_groups,
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

    if search_id:
        await run_in_threadpool(
            database_service.log_click,
            search_id=search_id,
            content_id=content.id,
        )

    return _build_content_response(content)


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

    if search_id:
        await run_in_threadpool(
            database_service.log_click,
            search_id=search_id,
            content_id=content_id,
        )

    return _build_content_response(content)
