from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import HelseDirectorateSearchRequest
from app.dto.response.helsedir import HelseDirectorateSearchResponse
from app.controllers.helsedir_controller import helsedir_controller
from app.exceptions.helsedir import HelseDirectorateAPIError

router = APIRouter(prefix="/helsedir", tags=["helsedirektoratet"])

# Maximum depth for recursive children fetching (increased for full content display)
MAX_DEPTH = 10


@router.get("/search", response_model=HelseDirectorateSearchResponse)
async def search_helsedirektoratet(request: HelseDirectorateSearchRequest = Depends()):
    """
    Search directly in Helsedirektoratet API with FULL content returned by default.

    This endpoint searches live data from Helsedirektoratet and returns complete 
    information for all results (no \"read more\" buttons needed).

    **Parameters:**
    - **QueryText** (required): Search query text
    - **Filter** (optional): OData filter expression
      - Example: `infoType eq 'retningslinje'`
    - **SearchMode** (optional): 'Any' (match any term) or 'All' (match all terms)
    - **QueryType** (optional): 'Simple' or 'Full' (Lucene syntax)
    - **getFullInfobits** (optional): true/false - Return full content (default: TRUE)

    **Examples:**

    Basic search (returns FULL content):
    ```
    GET /helsedir/search?QueryText=adhd
    ```

    Filter by type (returns FULL content):
    ```
    GET /helsedir/search?QueryText=adhd&Filter=infoType%20eq%20'retningslinje'
    ```

    Opt-out of full content (faster, less data):
    ```
    GET /helsedir/search?QueryText=diabetes&getFullInfobits=false
    ```

    Returns:
        Complete search results with all content (tittel, full tekst, etc.) - 
        ready to display as a full \"mock\" of the website without additional API calls.
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
    Get detailed information for a specific infobit with ALL nested content.

    This endpoint fetches the complete details of an infobit from the Helsedirektoratet API,
    including all nested children to create a full content "mock" of the website.

    **Parameters:**
    - **infobit_id** (required): The infobit ID (e.g., 0006-0014-70b46b52-eb30-4ee9-b8c8-ef5e238c419f)
    - **include_children** (optional): If true, fetches ALL child content recursively (default: True)
    - **depth** (optional): How many levels deep to fetch children (default: 10, max: 10)
      - depth=10: Fetches all nested content structure completely
      - Lower values can be used to limit depth if needed

    **Examples:**
    ```
    # Get complete infobit with all nested content (default behavior)
    GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e
    
    # Get infobit without children (opt-out if needed)
    GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e?include_children=false
    
    # Limit depth if needed
    GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e?depth=3
    ```

    Returns:
        Complete infobit details with ALL nested children structure (creates a full \"mock\" of the content):
        - Main infobit data (tittel, tekst, koder, lenker, etc.) - FULL CONTENT
        - children (recursively fetched to specified depth):
          - Kapitler with complete data
          - Pakkeforlop-anbefaling with complete data
          - All nested content types to full depth
          - No \"read more\" needed - all information is included
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
