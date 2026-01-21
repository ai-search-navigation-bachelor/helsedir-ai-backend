"""
Helsedirektoratet controller.

Handles business logic for Helsedirektoratet API operations.
"""

from typing import Optional, List
from pydantic import BaseModel
from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)


class HelseDirectorateSearchResponse(BaseModel):
    """Response from Helsedirektoratet API search."""
    results: List[dict]
    total: int
    query: str


class HelseDirectorateController:
    """Controller for Helsedirektoratet API operations."""

    def __init__(self):
        self.api_service = helsedir_api_service

    async def search(
        self,
        query: str,
        filter_query: Optional[str] = None,
        search_mode: Optional[str] = None,
        query_type: Optional[str] = None,
        get_full_infobits: bool = False,
    ) -> HelseDirectorateSearchResponse:
        """
        Search Helsedirektoratet API.

        Args:
            query: Search query text
            filter_query: OData filter expression
            search_mode: 'Any' or 'All'
            query_type: 'Simple' or 'Full'
            get_full_infobits: Return full content or not

        Returns:
            HelseDirectorateSearchResponse with results

        Raises:
            HelseDirectorateAPIError: If API is unavailable
            Exception: If search fails
        """
        # Search API with all parameters
        results = await self.api_service.search_infobits_async(
            query_text=query,
            filter_query=filter_query,
            search_mode=search_mode,
            query_type=query_type,
            get_full_infobits=get_full_infobits,
        )

        # Ensure results is a list
        if not isinstance(results, list):
            results = []

        return HelseDirectorateSearchResponse(
            results=results,
            total=len(results),
            query=query,
        )


# Global instance
helsedir_controller = HelseDirectorateController()
