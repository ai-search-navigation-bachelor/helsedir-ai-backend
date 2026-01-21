from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.models.schemas import SearchResponse
from app.controllers.search_controller import search_controller

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
        return await search_controller.search(
            query=query,
            role=role,
            k=k,
            method=method
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
