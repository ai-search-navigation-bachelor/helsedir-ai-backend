from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import HelseDirectorateSearchRequest
from app.controllers.helsedir_controller import (
    helsedir_controller,
    HelseDirectorateSearchResponse,
)
from app.services.external.helsedir_api_service import HelseDirectorateAPIError

router = APIRouter(prefix="/helsedir", tags=["helsedirektoratet"])

# Maximum depth for recursive children fetching (increased for full content display)
MAX_DEPTH = 10


@router.get("/search", response_model=HelseDirectorateSearchResponse)
async def search_helsedirektoratet(request: HelseDirectorateSearchRequest = Depends()):
    """
    Perform a live search against the Helsedirektoratet API and return complete result content by default.
    
    Returns:
        Complete search results where each result includes full content (titles, full text, and nested infobits) suitable for rendering a full mock of the site.
    """
    try:
        return await helsedir_controller.search(
            query=request.query,
            filter_query=request.filter,
            search_mode=request.search_mode,
            query_type=request.query_type,
            get_full_infobits=request.get_full_infobits,
        )
    except HelseDirectorateAPIError as e:
        raise HTTPException(
            status_code=503, detail=f"Helsedirektoratet API unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/infobit/{infobit_id}")
async def get_infobit(infobit_id: str, include_children: bool = True, depth: int = 10):
    """
    Fetches a complete infobit and its nested content from the Helsedirektoratet API.
    
    Retrieves the specified infobit and, when enabled, recursively includes child content up to the requested depth to produce a full-content representation.
    
    Parameters:
        include_children (bool): If True, recursively fetch child content for the infobit. Defaults to True.
        depth (int): Maximum recursion depth for fetching nested children. Must be between 1 and MAX_DEPTH (default 10).
    
    Returns:
        dict: Complete infobit data including nested children up to `depth` levels.
    """
    # Validate depth parameter to prevent DoS attacks
    if depth < 1:
        raise HTTPException(
            status_code=400, 
            detail=f"Depth must be at least 1, got {depth}"
        )
    if depth > MAX_DEPTH:
        raise HTTPException(
            status_code=400,
            detail=f"Depth cannot exceed {MAX_DEPTH} to prevent excessive API calls, got {depth}"
        )
    
    try:
        return await helsedir_controller.get_infobit(infobit_id, include_children, depth)
    except HelseDirectorateAPIError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch infobit: {str(e)}")