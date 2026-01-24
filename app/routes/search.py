from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import SearchRequest
from app.dto.response.search import SearchResponse
from app.controllers.search_controller import search_controller

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(request: SearchRequest = Depends()):
    """
    Search for content with pagination.

    Args:
        request: SearchRequest with query, role, method, offset, limit, search_id

    Returns:
        SearchResponse with paginated results and search_id for click tracking

    Query params:
        - query: Search text (required)
        - role: User role for filtering (optional)
        - method: 'keyword', 'semantic', or 'hybrid' (default: hybrid)
        - offset: Results to skip (default: 0)
        - limit: Results per page (default: 10, max: 50)
        - search_id: Existing search_id for pagination (optional)
    """
    try:
        return await search_controller.search(
            query=request.query,
            role=request.role,
            method=request.method,
            offset=request.offset,
            limit=request.limit,
            search_id=request.search_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
