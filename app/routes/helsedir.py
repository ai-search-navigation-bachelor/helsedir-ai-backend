from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import HelseDirectorateSearchRequest
from app.controllers.helsedir_controller import (
    helsedir_controller,
    HelseDirectorateSearchResponse,
)
from app.services.external.helsedir_api_service import HelseDirectorateAPIError

router = APIRouter(prefix="/helsedir", tags=["helsedirektoratet"])


@router.get("/search", response_model=HelseDirectorateSearchResponse)
async def search_helsedirektoratet(request: HelseDirectorateSearchRequest = Depends()):
    """
    Search directly in Helsedirektoratet API.

    This endpoint bypasses local cache and searches live data from Helsedirektoratet.

    **Parameters:**
    - **QueryText** (required): Search query text
    - **Filter** (optional): OData filter expression
      - Example: `infoType eq 'retningslinje'`
    - **SearchMode** (optional): 'Any' (match any term) or 'All' (match all terms)
    - **QueryType** (optional): 'Simple' or 'Full' (Lucene syntax)
    - **getFullInfobits** (optional): true/false - Return full content

    **Examples:**

    Basic search:
    ```
    GET /helsedir/search?QueryText=adhd
    ```

    Filter by type:
    ```
    GET helsedir/search/?QueryText=adhd&Filter=infoType%20eq%20'retningslinje'
    ```

    Full content:
    ```
    GET /helsedir/search?QueryText=diabetes&getFullInfobits=true
    ```

    Returns:
        Search results directly from Helsedirektoratet API with Norwegian field names
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
async def get_infobit(infobit_id: str, include_children: bool = False, depth: int = 1):
    """
    Get detailed information for a specific infobit.

    This endpoint fetches the full details of an infobit from the Helsedirektoratet API.

    **Parameters:**
    - **infobit_id** (required): The infobit ID (e.g., 0006-0014-70b46b52-eb30-4ee9-b8c8-ef5e238c419f)
    - **include_children** (optional): If true, fetches child content (kapitler, etc.)
    - **depth** (optional): How many levels deep to fetch children (default: 1)
      - depth=1: Direct children only (kapitler)
      - depth=2: Children and grandchildren (kapitler + pakkeforlop-anbefaling)
      - depth=3+: Continue recursively

    **Examples:**
    ```
    GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e
    GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e?include_children=true
    GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e?include_children=true&depth=2
    ```

    Returns:
        Full infobit details with nested children structure:
        - Main infobit data (tittel, tekst, koder, etc.)
        - children (if include_children=true):
          - Kapitler with their full data
          - children (if depth >= 2):
            - Pakkeforlop-anbefaling or other child types
            - And so on based on depth parameter
    """
    try:
        return await helsedir_controller.get_infobit(infobit_id, include_children, depth)
    except HelseDirectorateAPIError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch infobit: {str(e)}")
