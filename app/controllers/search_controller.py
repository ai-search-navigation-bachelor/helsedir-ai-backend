"""
Search controller.

Handles business logic for search operations with pagination and ML feature logging.
"""

import uuid
import re
from typing import Optional, List

from app.dto.response.search import SearchResult, SearchResponse
from app.services.search.search_service import search_service
from app.services.search.feature_extractor import feature_extractor
from app.services.data.database_service import database_service
from app.services.data.content_service import content_service


class SearchController:
    """Controller for search operations."""

    def __init__(self):
        self.search_service = search_service

    async def search(
        self,
        query: str,
        role: Optional[str] = None,
        method: str = "hybrid",
        offset: int = 0,
        limit: int = 10,
        search_id: Optional[str] = None,
    ) -> SearchResponse:
        """
        Execute search with pagination and ML feature logging.

        Args:
            query: Search query string
            role: Optional user role for filtering
            method: Search method ('keyword', 'semantic', or 'hybrid')
            offset: Number of results to skip
            limit: Number of results per page
            search_id: Existing search_id for pagination (None = new search)

        Returns:
            SearchResponse with paginated results
        """
        # Validate method
        valid_methods = {"keyword", "semantic", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid search method: {method}. Must be one of {valid_methods}")

        # Execute search
        max_results = 100
        all_results = self._execute_search(query, role, method, max_results)
        total = len(all_results)

        # Apply pagination
        page_results = all_results[offset:offset + limit]

        # Handle search_id (new search vs pagination)
        search_id = self._handle_search_id(search_id, query, role)

        # Extract ML features and log results
        self._log_results(search_id, query, role, page_results, offset)

        return SearchResponse(
            results=page_results,
            query=query,
            total=total,
            search_id=search_id,
            offset=offset,
            limit=limit,
            has_next=offset + limit < total,
            has_prev=offset > 0,
        )

    def _execute_search(
        self,
        query: str,
        role: Optional[str],
        method: str,
        max_results: int
    ) -> List[SearchResult]:
        """Execute the appropriate search method."""
        if method == "semantic":
            return self.search_service.search_semantic(query=query, role=role, k=max_results)
        elif method == "keyword":
            return self.search_service.search(query=query, role=role, k=max_results)
        else:  # hybrid
            return self.search_service.search_hybrid(query=query, role=role, k=max_results)

    def _handle_search_id(
        self,
        search_id: Optional[str],
        query: str,
        role: Optional[str]
    ) -> str:
        """Generate new search_id or validate existing one."""
        if not search_id:
            # New search
            search_id = str(uuid.uuid4())
            database_service.log_search(search_id=search_id, query=query, role=role)
        else:
            # Validate existing search_id
            stored_search = database_service.get_search_by_id(search_id)
            if stored_search is None:
                raise ValueError(f"Invalid search_id: {search_id}")
            if stored_search["query"].strip().lower() != query.strip().lower():
                raise ValueError(
                    f"Query mismatch: expected '{stored_search['query']}', got '{query}'"
                )

        return search_id

    def _log_results(
        self,
        search_id: str,
        query: str,
        role: Optional[str],
        results: List[SearchResult],
        offset: int,
    ) -> None:
        """Extract ML features and log results to database."""
        query_keywords = set(re.findall(r'\w+', query.lower()))

        results_to_log = []
        for local_index, result in enumerate(results):
            position = offset + local_index + 1
            content_item = content_service.get_content_by_id(result.id)

            features = {}
            if content_item:
                features = feature_extractor.extract_features(
                    content_item, query, query_keywords, role
                )

            results_to_log.append({
                "content_id": result.id,
                "position": position,
                "score": result.score,
                "semantic_similarity": features.get("semantic_similarity"),
                "keyword_score_total": features.get("keyword_score_total"),
                "exact_title_proportion": features.get("exact_title_proportion"),
                "full_coverage_proportion": features.get("full_coverage_proportion"),
                "title_keyword_proportion": features.get("title_keyword_proportion"),
                "type_match": features.get("type_match"),
                "role_match": features.get("role_match"),
                "code_match_count": features.get("code_match_count", 0),
                "lis_match": features.get("lis_match", 0),
                "maalgruppe_match": features.get("maalgruppe_match", 0),
            })

        database_service.log_search_results(search_id, results_to_log)


# Global instance
search_controller = SearchController()
