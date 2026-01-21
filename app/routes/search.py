from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import SearchRequest
from app.dto.response.search import SearchResponse
from app.controllers.search_controller import search_controller

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(request: SearchRequest = Depends()):
    """
    Search for content based on query.

    Args:
        request: SearchRequest with query, role, k, and method

    Returns:
        SearchResponse with list of results, query, and total count
    """
    try:
        return await search_controller.search(
            query=request.query,
            role=request.role,
            k=request.k,
            method=request.method
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
