"""
Search controller.

Handles business logic for search operations.
"""

from typing import Optional, List
from app.models.schemas import SearchResult, SearchResponse
from app.services.search.search_service import search_service
from app.services.analytics.logging_service import logging_service


class SearchController:
    """Controller for search operations."""

    def __init__(self):
        self.search_service = search_service
        self.logging_service = logging_service

    async def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
        method: str = "hybrid"
    ) -> SearchResponse:
        """
        Execute search with specified method.

        Args:
            query: Search query string
            role: Optional user role for filtering
            k: Number of results to return
            method: Search method ('keyword', 'semantic', or 'hybrid')

        Returns:
            SearchResponse with results

        Raises:
            ValueError: If invalid search method
            Exception: If search fails
        """
        # Validate method
        valid_methods = {"keyword", "semantic", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid search method: {method}. Must be one of {valid_methods}")

        # Select search method
        if method == "semantic":
            results = self.search_service.search_semantic(
                query=query, role=role, k=k
            )
        elif method == "keyword":
            results = self.search_service.search(
                query=query, role=role, k=k
            )
        else:  # hybrid (default)
            results = self.search_service.search_hybrid(
                query=query, role=role, k=k
            )

        # Log the search event
        self.logging_service.log_event(
            event_type="search", query=query, role=role
        )

        return SearchResponse(
            results=results, query=query, total=len(results)
        )


# Global instance
search_controller = SearchController()
