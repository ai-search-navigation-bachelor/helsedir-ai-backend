from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.models.schemas import SearchResponse
from app.services.search_service import search_service
from app.services.logging_service import logging_service

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    role: Optional[str] = Query(None, description="User role for filtering"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    method: str = Query("hybrid", description="Search method: 'keyword', 'semantic', or 'hybrid'")
):
    """
    Search for content based on query.

    Args:
        query: Search query string
        role: Optional user role for filtering
        k: Number of results to return (1-100)
        method: Search method - 'keyword', 'semantic', or 'hybrid' (default)

    Returns:
        SearchResponse with list of results, query, and total count
    """
    try:
        # Select search method
        if method == "semantic":
            results = search_service.search_semantic(
                query=query, role=role, k=k
            )
        elif method == "keyword":
            results = search_service.search(
                query=query, role=role, k=k
            )
        else:  # hybrid (default)
            results = search_service.search_hybrid(
                query=query, role=role, k=k
            )

        # Log the search event
        logging_service.log_event(
            event_type="search", query=query, role=role
        )

        return SearchResponse(
            results=results, query=query, total=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
