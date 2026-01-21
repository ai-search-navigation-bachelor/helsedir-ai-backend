from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.controllers.helsedir_controller import (
    helsedir_controller,
    HelseDirectorateSearchResponse,
)
from app.services.external.helsedir_api_service import HelseDirectorateAPIError

router = APIRouter(prefix="/helsedir", tags=["helsedirektoratet"])


@router.get("/search", response_model=HelseDirectorateSearchResponse)
async def search_helsedirektoratet(
    query: str = Query(..., description="Search query text", min_length=1, alias="QueryText"),
    filter: Optional[str] = Query(None, description="OData filter expression (e.g., infoType eq 'retningslinje')", alias="Filter"),
    search_mode: Optional[str] = Query(None, description="Search mode: 'Any' or 'All'", alias="SearchMode"),
    query_type: Optional[str] = Query(None, description="Query type: 'Simple' or 'Full'", alias="QueryType"),
    get_full_infobits: bool = Query(False, description="Return full infobit content", alias="getFullInfobits"),
):
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
            query=query,
            filter_query=filter,
            search_mode=search_mode,
            query_type=query_type,
            get_full_infobits=get_full_infobits,
        )
    except HelseDirectorateAPIError as e:
        raise HTTPException(
            status_code=503, detail=f"Helsedirektoratet API unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
