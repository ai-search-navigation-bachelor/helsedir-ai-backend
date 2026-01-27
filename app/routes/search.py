from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import SearchRequest, CategorizedSearchRequest, CategorySearchRequest
from app.dto.response.search import SearchResponse, CategorizedSearchResponse
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


@router.get("/categorized", response_model=CategorizedSearchResponse)
async def search_categorized(request: CategorizedSearchRequest = Depends()):
    """
    Search for content and return results grouped by category.

    Priority categories (e.g., retningslinje) return all results.
    Other categories return count and top 5 preview.

    Query params:
        - query: Search text (required)
        - role: User role for filtering (optional)
        - method: 'keyword', 'semantic', or 'hybrid' (default: hybrid)

    Returns:
        CategorizedSearchResponse with:
        - priority_categories: Full results for retningslinjer etc.
        - other_categories: Count + top 5 preview for other types
    """
    try:
        return await search_controller.search_categorized(
            query=request.query,
            role=request.role,
            method=request.method,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorized search failed: {str(e)}")


@router.get("/category", response_model=SearchResponse)
async def search_category(request: CategorySearchRequest = Depends()):
    """
    Get all results for a specific category.

    Use this when user clicks on a category to see all results.

    Query params:
        - query: Search text (required)
        - category: Category (info_type) to filter by (required)
        - role: User role for filtering (optional)
        - method: 'keyword', 'semantic', or 'hybrid' (default: hybrid)
        - search_id: Existing search_id from categorized search (optional)

    Returns:
        SearchResponse with all results in the specified category
    """
    try:
        return await search_controller.search_category(
            query=request.query,
            category=request.category,
            role=request.role,
            method=request.method,
            search_id=request.search_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Category search failed: {str(e)}")
