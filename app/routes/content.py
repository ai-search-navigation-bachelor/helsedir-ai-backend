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


@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(
    content_id: str,
    search_id: Optional[str] = Query(None, description="Search ID for click tracking"),
):
    """
    Get content by ID and optionally log click.

    If search_id is provided, logs the click for LTR training.
    The position is automatically looked up from search_results_shown.

    Args:
        content_id: The content ID to retrieve
        search_id: Optional search_id to link this click to a search

    Returns:
        ContentResponse with full content details
    """
    # Get content
    content = content_service.get_content_by_id(content_id)

    if not content:
        raise HTTPException(status_code=404, detail=f"Content not found: {content_id}")

    # Log click if search_id is provided (run blocking DB call in threadpool)
    if search_id:
        await run_in_threadpool(
            database_service.log_click,
            search_id=search_id,
            content_id=content_id,
        )

    # Convert links to response format (already processed at import/migration)
    links_response = [
        ContentLinkResponse(
            rel=link.rel,
            type=link.type,
            tittel=link.tittel,
            id=link.id,
            href=link.href,
        )
        for link in content.links
    ]

    # Fetch linked content for theme pages
    linked_content_response = None
    if content.content_type.lower() == "temaside":
        linked_content_response = _get_theme_page_linked_content(content_id)

    # Map anbefaling-specific fields if present
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
        target_groups=content.target_groups,
        links=links_response,
        linked_content=linked_content_response,
        anbefaling_fields=anbefaling_fields_response,
    )
