"""
Helsedirektoratet controller.

Handles business logic for Helsedirektoratet API operations.
"""

from typing import Optional
from app.services.external.helsedir_api_service import helsedir_api_service
from app.exceptions.helsedir import HelseDirectorateAPIError
from app.dto.response.helsedir import HelseDirectorateSearchResponse


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

    async def get_infobit(
        self,
        infobit_id: str,
        include_children: bool = True,
        depth: int = 10,
    ) -> dict:
        """
        Get complete information for a specific infobit with all nested content.

        Args:
            infobit_id: The infobit ID
            include_children: Whether to fetch ALL child content recursively (default: True)
            depth: How many levels deep to fetch (default: 10 for complete data)

        Returns:
            Dictionary with COMPLETE infobit details and all nested children data.
            Creates a full \"mock\" of the website content - no \"read more\" needed.

        Raises:
            HelseDirectorateAPIError: If API is unavailable or infobit not found
        """
        # Fetch main infobit
        infobit = await self.api_service.get_infobit_by_id_async(infobit_id)

        # If children are requested, fetch them recursively
        if include_children and depth > 0:
            infobit["children"] = await self._fetch_children(
                infobit.get("links", []),
                depth - 1
            )

        return infobit

    async def _fetch_children(
        self,
        links: list,
        remaining_depth: int,
    ) -> list:
        """
        Recursively fetch child content from links.

        Args:
            links: List of link objects from parent content
            remaining_depth: How many more levels to fetch (0 = no more children)

        Returns:
            List of child content with their data
        """
        children = []
        
        for link in links:
            # Only fetch "barn" (child) links
            if link.get("rel") != "barn":
                continue

            href = link.get("href", "")
            if not href:
                continue

            try:
                # Fetch child content using generic method
                child_data = await self.api_service.get_content_by_href_async(href)
                
                child_item = {
                    "tittel": link.get("tittel"),
                    "type": link.get("type"),
                    "href": href,
                    "data": child_data
                }

                # Recursively fetch grandchildren if depth allows
                if remaining_depth > 0 and "links" in child_data:
                    child_item["children"] = await self._fetch_children(
                        child_data.get("links", []),
                        remaining_depth - 1
                    )

                children.append(child_item)
                
            except HelseDirectorateAPIError:
                # Skip if content cannot be fetched
                pass

        return children


# Global instance
helsedir_controller = HelseDirectorateController()
