"""
Search controller.

Handles business logic for search operations with pagination and ML feature logging.
"""

import uuid
import re
import logging
from typing import Optional, List, Dict
from collections import defaultdict

from app.dto.response.search import (
    SearchResult,
    SearchResponse,
    CategoryResults,
    CategorizedSearchResponse,
)
from app.services.search.search_service import search_service
from app.services.search.feature_extractor import feature_extractor
from app.services.data.database_service import database_service
from app.services.data.content_service import content_service
from app.config import settings
from app.constants import (
    is_priority_category,
    get_category_display_name,
    PRIORITY_CATEGORIES,
)

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

    async def search_categorized(
        self,
        query: str,
        role: Optional[str] = None,
        method: str = "hybrid",
    ) -> CategorizedSearchResponse:
        """
        Execute search and return results grouped by category.

        - Priority categories (e.g., retningslinje) return ALL results
        - Other categories return count + top N preview

        Args:
            query: Search query string
            role: Optional user role for filtering
            method: Search method ('keyword', 'semantic', or 'hybrid')

        Returns:
            CategorizedSearchResponse with grouped results
        """
        # Validate method
        valid_methods = {"keyword", "semantic", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid search method: {method}. Must be one of {valid_methods}")

        # Execute search with high limit to get all potential results
        max_results = 500
        all_results = self._execute_search(query, role, method, max_results)

        if all_results is None:
            all_results = []

        # Filter by minimum score
        min_score = settings.search_min_score
        filtered_results = [r for r in all_results if r.score >= min_score]

        # Group results by info_type
        grouped: Dict[str, List[SearchResult]] = defaultdict(list)
        for result in filtered_results:
            category = result.info_type.lower() if result.info_type else "unknown"
            grouped[category].append(result)

        # Generate search_id for this search
        search_id = str(uuid.uuid4())
        database_service.log_search(search_id=search_id, query=query, role=role)

        # Build priority categories (show all results)
        priority_categories: List[CategoryResults] = []
        for category in PRIORITY_CATEGORIES:
            if category in grouped:
                results = grouped[category]
                priority_categories.append(CategoryResults(
                    category=category,
                    display_name=get_category_display_name(category),
                    count=len(results),
                    is_priority=True,
                    results=results,
                ))

        # Build other categories (show count + top N preview)
        preview_count = settings.search_category_preview_count
        other_categories: List[CategoryResults] = []

        # Sort other categories by count (descending)
        other_category_keys = [
            k for k in grouped.keys()
            if k not in PRIORITY_CATEGORIES
        ]
        other_category_keys.sort(key=lambda k: len(grouped[k]), reverse=True)

        for category in other_category_keys:
            results = grouped[category]
            other_categories.append(CategoryResults(
                category=category,
                display_name=get_category_display_name(category),
                count=len(results),
                is_priority=False,
                results=results[:preview_count],  # Only top N for preview
            ))

        # Log results for ML (flatten all shown results)
        all_shown_results = []
        for cat in priority_categories:
            all_shown_results.extend(cat.results)
        for cat in other_categories:
            all_shown_results.extend(cat.results)

        if all_shown_results:
            self._log_results(search_id, query, role, all_shown_results, offset=0)

        return CategorizedSearchResponse(
            query=query,
            total=len(filtered_results),
            min_score=min_score,
            search_id=search_id,
            priority_categories=priority_categories,
            other_categories=other_categories,
        )

    async def search_category(
        self,
        query: str,
        category: str,
        role: Optional[str] = None,
        method: str = "hybrid",
        search_id: Optional[str] = None,
    ) -> SearchResponse:
        """
        Get all results for a specific category.

        Used when user clicks on a category to see all results.

        Args:
            query: Search query string
            category: The info_type category to filter by
            role: Optional user role for filtering
            method: Search method
            search_id: Existing search_id (reuse from categorized search)

        Returns:
            SearchResponse with all results in the category
        """
        # Validate method
        valid_methods = {"keyword", "semantic", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid search method: {method}. Must be one of {valid_methods}")

        # Execute search
        max_results = 500
        all_results = self._execute_search(query, role, method, max_results)

        if all_results is None:
            all_results = []

        # Filter by minimum score and category
        min_score = settings.search_min_score
        category_lower = category.lower()
        filtered_results = [
            r for r in all_results
            if r.score >= min_score and r.info_type.lower() == category_lower
        ]

        # Handle search_id
        if search_id:
            # Validate existing search_id
            stored_search = database_service.get_search_by_id(search_id)
            if stored_search is None:
                # Create new search_id if invalid
                search_id = str(uuid.uuid4())
                database_service.log_search(search_id=search_id, query=query, role=role)
        else:
            search_id = str(uuid.uuid4())
            database_service.log_search(search_id=search_id, query=query, role=role)

        # Log results
        if filtered_results:
            self._log_results(search_id, query, role, filtered_results, offset=0)

        return SearchResponse(
            results=filtered_results,
            query=query,
            total=len(filtered_results),
            search_id=search_id,
            offset=0,
            limit=len(filtered_results),
            has_next=False,
            has_prev=False,
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
