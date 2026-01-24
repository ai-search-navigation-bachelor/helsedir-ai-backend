"""
Content route for retrieving content and logging clicks.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.dto.response.content import ContentResponse
from app.services.data.content_service import content_service
from app.services.data.database_service import database_service

router = APIRouter(prefix="/content", tags=["content"])


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

    # Log click if search_id is provided
    if search_id:
        database_service.log_click(
            search_id=search_id,
            content_id=content_id,
        )

    return ContentResponse(
        id=content.id,
        title=content.title,
        body=content.body,
        url=content.url,
        content_type=content.content_type,
        target_groups=content.target_groups,
    )
