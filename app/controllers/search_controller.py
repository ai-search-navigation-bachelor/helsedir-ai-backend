"""
Search controller.

Handles business logic for search operations with pagination and ML feature logging.
"""

import uuid
import re
import logging
import difflib
from typing import Optional, List, Dict
from collections import defaultdict
from fastapi import BackgroundTasks

from app.dto.response.search import (
    SearchResult,
    SearchResponse,
    CategoryResults,
    CategorizedSearchResponse,
    GroupedContent,
)
from app.services.search.search_service import search_service
from app.services.search.feature_extractor import feature_extractor
from app.services.data.database_service import database_service
from app.services.data.content_service import content_service
from app.services.repositories.content_repository import content_repository
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
        background_tasks: Optional[BackgroundTasks] = None,
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
            background_tasks: FastAPI background tasks for async logging

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

        # Populate theme page children AFTER pagination
        page_results = self._populate_theme_page_children(page_results)

        # Handle search_id (new search vs pagination)
        search_id = self._handle_search_id(search_id, query, role)

        # Extract ML features and log results (in background)
        if background_tasks:
            background_tasks.add_task(self._log_results, search_id, query, role, page_results, offset)
        else:
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
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> CategorizedSearchResponse:
        """
        Execute search and return results grouped by category.

        - Priority categories (e.g., retningslinje) return ALL results
        - Other categories return count + top N preview

        Args:
            query: Search query string
            role: Optional user role for filtering
            method: Search method ('keyword', 'semantic', or 'hybrid')
            background_tasks: FastAPI background tasks for async logging

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

        # Populate theme page children
        all_results = self._populate_theme_page_children(all_results)

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

        # Log only the FIRST result from each priority category (what user initially sees)
        # When user clicks "Vis flere", /search/category logs the rest
        initially_shown = []
        for cat in priority_categories:
            if cat.results:
                initially_shown.append(cat.results[0])  # Only first result

        if initially_shown:
            if background_tasks:
                background_tasks.add_task(self._log_results, search_id, query, role, initially_shown, 0)
            else:
                self._log_results(search_id, query, role, initially_shown, offset=0)

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
        search_id: str = "",
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> SearchResponse:
        """
        Get all results for a specific category.

        Used when user clicks on a category to see all results.

        Args:
            query: Search query string
            category: The info_type category to filter by
            role: Optional user role for filtering
            method: Search method
            search_id: search_id from categorized search (required)
            background_tasks: FastAPI background tasks for async logging

        Returns:
            SearchResponse with all results in the category
        """
        # Validate method
        valid_methods = {"keyword", "semantic", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid search method: {method}. Must be one of {valid_methods}")

        # Validate search_id exists
        stored_search = database_service.get_search_by_id(search_id)
        if stored_search is None:
            raise ValueError(f"Invalid search_id: {search_id}")

        # Execute search
        max_results = 500
        all_results = self._execute_search(query, role, method, max_results)

        if all_results is None:
            all_results = []

        # Populate theme page children
        all_results = self._populate_theme_page_children(all_results)

        # Filter by minimum score and category
        min_score = settings.search_min_score
        category_lower = category.lower()
        filtered_results = [
            r for r in all_results
            if r.score >= min_score and r.info_type and r.info_type.lower() == category_lower
        ]

        # Log results (in background)
        if filtered_results:
            if background_tasks:
                background_tasks.add_task(self._log_results, search_id, query, role, filtered_results, 0)
            else:
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

    def _search_theme_pages_fuzzy(self, query: str, fuzzy_threshold: float = 0.6) -> List[SearchResult]:
        """
        Search theme pages using fuzzy keyword matching.

        Uses difflib for fuzzy string matching to handle typos.

        Args:
            query: Search query
            fuzzy_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching theme pages
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        results = []

        # Get all theme pages
        all_content = content_service.get_all_content()
        theme_pages = [item for item in all_content if item.content_type.lower() == 'temaside']

        for theme_page in theme_pages:
            title = theme_page.title.lower()
            title_words = set(re.findall(r'\w+', title))

            # Calculate fuzzy match score
            max_score = 0.0

            # Method 1: Word-by-word fuzzy matching
            for query_word in query_words:
                for title_word in title_words:
                    # Use difflib's SequenceMatcher for fuzzy matching
                    similarity = difflib.SequenceMatcher(None, query_word, title_word).ratio()
                    max_score = max(max_score, similarity)

            # Method 2: Full title fuzzy matching
            full_title_similarity = difflib.SequenceMatcher(None, query_lower, title).ratio()
            max_score = max(max_score, full_title_similarity)

            # Method 3: Check if query is a substring of title (for partial matches)
            if query_lower in title:
                max_score = max(max_score, 0.9)

            # If score exceeds threshold, add to results
            if max_score >= fuzzy_threshold:
                results.append(SearchResult(
                    id=theme_page.id,
                    title=theme_page.title,
                    info_type='temaside',
                    score=max_score,
                    explanation=f"Fuzzy match: {int(max_score * 100)}%"
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _populate_theme_page_children(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Populate children for theme pages using batch query (performance optimized).

        For each theme page result, fetch its linked content from the junction table
        and group by info_type. Uses a single database query for all theme pages.

        Args:
            results: List of search results

        Returns:
            Same list with theme page children populated
        """
        # Collect all theme page IDs
        theme_page_ids = [r.id for r in results if r.info_type.lower() == "temaside"]

        if not theme_page_ids:
            return results  # No theme pages, nothing to do

        # Fetch all children in ONE database query (much faster!)
        all_children = content_repository.get_theme_pages_content_batch(theme_page_ids)

        # Populate each theme page with its children
        for result in results:
            if result.info_type.lower() != "temaside":
                continue

            linked_content = all_children.get(result.id, [])
            if not linked_content:
                continue

            # Group by info_type
            grouped: Dict[str, List[SearchResult]] = defaultdict(list)
            for content in linked_content:
                info_type = content.get('info_type', '').lower()
                if not info_type:
                    continue

                # Create SearchResult for child content
                child_result = SearchResult(
                    id=content.get('id', ''),
                    title=content.get('tittel', ''),
                    info_type=info_type,
                    score=1.0,  # Children inherit parent's relevance
                    explanation=f"Under {result.title}"
                )
                grouped[info_type].append(child_result)

            # Convert grouped dict to GroupedContent list
            children = []
            for info_type, items in sorted(grouped.items()):
                children.append(GroupedContent(
                    info_type=info_type,
                    display_name=get_category_display_name(info_type),
                    items=items
                ))

            result.children = children

        return results

    def _merge_theme_page_results(
        self,
        regular_results: List[SearchResult],
        theme_page_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Merge theme page results with regular results.

        Removes duplicates and sorts by score.

        Args:
            regular_results: Results from regular search
            theme_page_results: Results from fuzzy theme page search

        Returns:
            Merged and deduplicated results
        """
        # Create a dict to avoid duplicates (using id as key)
        merged = {}

        # Add regular results first
        for result in regular_results:
            merged[result.id] = result

        # Add theme page results (override if better score)
        for result in theme_page_results:
            if result.id not in merged or result.score > merged[result.id].score:
                merged[result.id] = result

        # Convert back to list and sort by score
        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x.score, reverse=True)

        return merged_list

    def _execute_search(
        self,
        query: str,
        role: Optional[str],
        method: str,
        max_results: int
    ) -> List[SearchResult]:
        """Execute the appropriate search method and include fuzzy theme page matching."""
        # Execute regular search
        if method == "semantic":
            regular_results = self.search_service.search_semantic(query=query, role=role, k=max_results)
        elif method == "keyword":
            regular_results = self.search_service.search(query=query, role=role, k=max_results)
        else:  # hybrid
            regular_results = self.search_service.search_hybrid(query=query, role=role, k=max_results)

        # Coerce None to empty list
        if regular_results is None:
            regular_results = []

        # Search theme pages with fuzzy matching
        theme_page_results = self._search_theme_pages_fuzzy(query, fuzzy_threshold=0.6)

        # Merge results
        merged_results = self._merge_theme_page_results(regular_results, theme_page_results)

        return merged_results

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

    async def get_theme_pages(
        self,
        category: Optional[str] = None,
    ):
        """
        Get all theme pages, optionally filtered by category.

        Args:
            category: Optional category slug (e.g., 'forebygging-diagnose-og-behandling')

        Returns:
            ThemePageResponse with all theme pages
        """
        from app.dto.response.search import ThemePageResponse, ThemePageResult

        # Validate category if provided
        if category:
            from app.constants import is_valid_theme_category
            if not is_valid_theme_category(category):
                raise ValueError(f"Invalid theme category: {category}")

        # Get theme pages from repository
        theme_pages = content_repository.get_theme_pages(category=category)

        # Convert to ThemePageResult objects
        results = []
        for item in theme_pages:
            results.append(ThemePageResult(
                id=item.get('id', ''),
                title=item.get('tittel', ''),
                info_type='temaside',
                path=item.get('path', ''),
            ))

        return ThemePageResponse(
            results=results,
            total=len(results),
        )

    def _log_results(
        self,
        search_id: str,
        query: str,
        role: Optional[str],
        results: List[SearchResult],
        offset: int,
    ) -> None:
        """Extract ML features and log results to database."""
        # Get already logged content_ids to avoid duplicates
        already_logged = database_service.get_logged_content_ids_for_search(search_id)

        # Filter out already logged results
        new_results = [r for r in results if r.id not in already_logged]

        if not new_results:
            return  # Nothing new to log

        # Get max position to continue from
        max_position = database_service.get_max_position_for_search(search_id)

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
        for local_index, result in enumerate(new_results):
            position = max_position + local_index + 1
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
