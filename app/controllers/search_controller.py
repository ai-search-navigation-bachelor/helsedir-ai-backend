"""
Search controller.

Handles business logic for search operations with pagination and ML feature logging.
"""

import uuid
import re
import logging
from typing import Optional, List

from app.dto.response.search import SearchResult, SearchResponse
from app.services.search.search_service import search_service
from app.services.search.feature_extractor import feature_extractor
from app.services.data.database_service import database_service
from app.services.data.content_service import content_service

logger = logging.getLogger(__name__)


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

        # Validate and clamp offset and limit
        offset = max(0, offset) if isinstance(offset, int) else 0
        limit = max(1, limit) if isinstance(limit, int) else 10

        # Execute search
        max_results = 100
        all_results = self._execute_search(query, role, method, max_results)

        # Coerce None to empty list
        if all_results is None:
            all_results = []

        total = len(all_results)

        # Clamp offset to not exceed result length
        offset = min(offset, max(0, total))

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
            # Validate role matches
            stored_role = stored_search.get("role") or None
            incoming_role = role or None
            if stored_role != incoming_role:
                raise ValueError(
                    f"Role mismatch: expected '{stored_role}', got '{incoming_role}'"
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

        # Default feature values to avoid NULLs
        default_features = {
            "semantic_similarity": 0.0,
            "keyword_score_total": 0.0,
            "exact_title_proportion": 0.0,
            "full_coverage_proportion": 0.0,
            "title_keyword_proportion": 0.0,
            "type_match": 0.5,
            "role_match": 0.0,
            "code_match_count": 0,
            "lis_match": 0,
            "maalgruppe_match": 0,
        }

        results_to_log = []
        for local_index, result in enumerate(results):
            position = offset + local_index + 1
            content_item = content_service.get_content_by_id(result.id)

            features = default_features.copy()
            if content_item:
                try:
                    extracted = feature_extractor.extract_features(
                        content_item, query, query_keywords, role
                    )
                    if extracted:
                        # Merge extracted features with defaults
                        for key in default_features:
                            if key in extracted and extracted[key] is not None:
                                features[key] = extracted[key]
                except Exception as e:
                    # Log failure with context but keep default features
                    logger.exception(
                        "Feature extraction failed for content_id=%s, query=%s, role=%s: %s",
                        result.id, query, role, e
                    )

            results_to_log.append({
                "content_id": result.id,
                "position": position,
                "score": result.score,
                "semantic_similarity": features["semantic_similarity"],
                "keyword_score_total": features["keyword_score_total"],
                "exact_title_proportion": features["exact_title_proportion"],
                "full_coverage_proportion": features["full_coverage_proportion"],
                "title_keyword_proportion": features["title_keyword_proportion"],
                "type_match": features["type_match"],
                "role_match": features["role_match"],
                "code_match_count": features["code_match_count"],
                "lis_match": features["lis_match"],
                "maalgruppe_match": features["maalgruppe_match"],
            })

        database_service.log_search_results(search_id, results_to_log)


# Global instance
search_controller = SearchController()
