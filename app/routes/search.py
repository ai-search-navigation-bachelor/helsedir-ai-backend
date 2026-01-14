from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchRequest, SearchResponse
from app.services.search_service import search_service
from app.services.logging_service import logging_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for content based on query.

    Args:
        request: Search request with query, optional role, and k (number of results)

    Returns:
        SearchResponse with list of results, query, and total count
    """
    try:
        # Perform search
        results = search_service.search(
            query=request.query, role=request.role, k=request.k
        )

        # Log the search event
        logging_service.log_event(
            event_type="search", query=request.query, role=request.role
        )

        return SearchResponse(
            results=results, query=request.query, total=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
