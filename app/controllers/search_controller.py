"""
Search controller.

Handles business logic for search operations with pagination and ML feature logging.
"""

import uuid
import re
import logging
import time
import difflib
import hashlib
import json
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from fastapi import BackgroundTasks

from app.dto.response.search import (
    SearchResult,
    SearchResponse,
    CategoryResults,
    CategorizedSearchResponse,
    GroupedContent,
    ThemePageResponse,
    Suggestion,
    SuggestionResponse,
)
from app.services.search.search_service import search_service
from app.services.data.database_service import database_service
from app.services.data.content_service import content_service
from app.services.data.document_metadata import resolve_public_document_url
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

    # Cache search results for 30 seconds so concurrent category-filtered
    # requests (same query) don't re-run the full pipeline.
    _CACHE_TTL_SECONDS = 30.0
    _SEARCH_ID_SIGNATURE_MARKER = "c0de"
    _SEARCH_ID_SIGNATURE_HEX_LENGTH = 8

    def __init__(self):
        self.search_service = search_service
        self._search_cache: Dict[
            Tuple[object, ...],
            Tuple[float, List["SearchResult"]],
        ] = {}

    async def search(
        self,
        query: str,
        role: Optional[str] = None,
        method: str = "hybrid",
        offset: int = 0,
        limit: int = 10,
        search_id: Optional[str] = None,
        category: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
        log: bool = True,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        temaside_boost: Optional[float] = None,
        retningslinje_boost: Optional[float] = None,
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
            category: Optional category (info_type) to filter results by
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

        # Fetch a fixed pool so that total and category_counts are stable across
        # all pagination requests for the same query.
        max_results = 500
        all_results = self._execute_search(
            query,
            role,
            method,
            max_results,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
            temaside_boost=temaside_boost,
            retningslinje_boost=retningslinje_boost,
        )

        # Coerce None to empty list
        if all_results is None:
            all_results = []

        # Filter low-relevance results after RRF normalization.
        min_score = settings.search_min_score
        all_results = [r for r in all_results if r.score >= min_score]

        total = len(all_results)

        # Compute category distribution from all relevant results (before filtering)
        category_counts: Dict[str, int] = defaultdict(int)
        for r in all_results:
            cat = r.info_type.lower() if r.info_type else "unknown"
            category_counts[cat] += 1

        # Filter by category if specified (after computing counts so tabs stay accurate)
        # Supports comma-separated values, e.g. "retningslinje,nasjonalt-forlop"
        if category:
            category_set = {c.strip().lower() for c in category.split(",") if c.strip()}
            all_results = [r for r in all_results if r.info_type and r.info_type.lower() in category_set]
            total = len(all_results)

        # Clamp offset to not exceed result length
        offset = min(offset, max(0, total))

        # Apply pagination
        page_results = all_results[offset:offset + limit]

        # Populate theme page children AFTER pagination
        page_results = self._populate_theme_page_children(page_results)

        # Handle search_id (new search vs pagination)
        search_id = self._handle_search_id(
            search_id=search_id,
            query=query,
            role=role,
            method=method,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
            temaside_boost=temaside_boost,
            retningslinje_boost=retningslinje_boost,
        )

        # Extract ML features and log results (skip for prefetch requests)
        if log:
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
            category_counts=dict(category_counts),
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

    def _search_theme_pages_fuzzy(
        self,
        query: str,
        fuzzy_threshold: float = 0.72,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search theme pages using fuzzy keyword matching.

        Uses difflib for typo-tolerant fallback matching.

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
                    path=theme_page.path,
                    has_text_content=theme_page.has_text_content,
                    document_url=theme_page.public_document_url,
                    is_pdf_only=theme_page.is_pdf_only,
                    score=max_score,
                    explanation=f"Theme fallback (fuzzy): {int(max_score * 100)}%"
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        if max_results is None:
            return results
        return results[:max(1, int(max_results))]

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
                    path=content.get('path'),
                    has_text_content=bool(content.get('has_text_content')),
                    document_url=resolve_public_document_url(
                        content.get('path'),
                        content.get('document_url'),
                    ),
                    is_pdf_only=not bool(content.get('has_text_content')) and bool(content.get('document_url')),
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

    def _needs_theme_fuzzy_fallback(
        self,
        regular_results: List[SearchResult],
    ) -> bool:
        """
        Decide whether theme-page fuzzy fallback should run.

        BM25/semantic results are primary. Fallback is used only when:
        - no theme pages are returned, or
        - best theme page score is below configured minimum score.
        """
        theme_results = [
            r for r in regular_results
            if r.info_type and r.info_type.lower() == "temaside"
        ]
        if not theme_results:
            return True

        top_theme_score = max(r.score for r in theme_results)
        return top_theme_score < settings.search_min_score

    def _merge_theme_fallback_results(
        self,
        regular_results: List[SearchResult],
        fallback_theme_results: List[SearchResult],
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Merge fallback theme-page results into regular results.

        Existing BM25/semantic results are never overridden by fallback hits.
        """
        if not fallback_theme_results:
            return regular_results

        merged = {result.id: result for result in regular_results}

        for result in fallback_theme_results:
            if result.id not in merged:
                merged[result.id] = result

        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x.score, reverse=True)
        if max_results is None:
            return merged_list
        return merged_list[:max(1, int(max_results))]

    def _execute_search(
        self,
        query: str,
        role: Optional[str],
        method: str,
        max_results: int,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        temaside_boost: Optional[float] = None,
        retningslinje_boost: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Execute the appropriate search method with short-lived caching.

        The frontend fires multiple concurrent requests for the same query
        (one per category tab). This cache avoids re-running the full
        BM25 + semantic + RRF pipeline for each one.

        Theme pages are BM25/semantic-first with controlled fuzzy fallback.
        """
        cache_key = (
            query.strip().lower(),
            role,
            method,
            max_results,
            bm25_weight,
            semantic_weight,
            rrf_k,
            temaside_boost,
            retningslinje_boost,
        )
        now = time.monotonic()

        # Check cache
        if cache_key in self._search_cache:
            cached_time, cached_results = self._search_cache[cache_key]
            if now - cached_time < self._CACHE_TTL_SECONDS:
                return cached_results

        # Evict stale entries (keep cache bounded)
        stale_keys = [
            k for k, (t, _) in self._search_cache.items()
            if now - t >= self._CACHE_TTL_SECONDS
        ]
        for k in stale_keys:
            del self._search_cache[k]

        # Execute regular search
        if method == "semantic":
            regular_results = self.search_service.search_semantic(query=query, role=role, k=max_results)
        elif method == "keyword":
            regular_results = self.search_service.search(query=query, role=role, k=max_results)
        else:  # hybrid
            regular_results = self.search_service.search_hybrid(
                query=query, role=role, k=max_results,
                bm25_weight=bm25_weight, semantic_weight=semantic_weight,
                rrf_k=max(1, int(rrf_k if rrf_k is not None else settings.search_rrf_k)),
                temaside_boost=temaside_boost,
                retningslinje_boost=retningslinje_boost,
            )

        # Coerce None to empty list
        if regular_results is None:
            regular_results = []
        else:
            regular_results = regular_results[:max(1, int(max_results))]

        if self._needs_theme_fuzzy_fallback(regular_results):
            # Controlled fallback for typo tolerance on theme pages only.
            theme_page_results = self._search_theme_pages_fuzzy(
                query=query,
                fuzzy_threshold=0.72,
                max_results=max_results,
            )
            regular_results = self._merge_theme_fallback_results(
                regular_results,
                theme_page_results,
                max_results=max_results,
            )

        # Store in cache
        self._search_cache[cache_key] = (now, regular_results)

        return regular_results

    def _handle_search_id(
        self,
        search_id: Optional[str],
        query: str,
        role: Optional[str],
        method: str,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        temaside_boost: Optional[float] = None,
        retningslinje_boost: Optional[float] = None,
    ) -> str:
        """Generate new search_id or validate existing one."""
        expected_signature = self._build_search_signature(
            method=method,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
            temaside_boost=temaside_boost,
            retningslinje_boost=retningslinje_boost,
        )

        if not search_id:
            # New search
            search_id = self._build_signed_search_id(expected_signature)
            database_service.log_search(search_id=search_id, query=query, role=role)
        else:
            # Validate existing search_id
            stored_search = database_service.get_search_by_id(search_id)
            if stored_search is None:
                raise ValueError(f"Invalid search_id: {search_id}")

            embedded_signature = self._extract_search_signature(search_id)
            if embedded_signature is None:
                raise ValueError(
                    "Invalid search_id format for pagination. "
                    "Please start a new search."
                )
            if embedded_signature != expected_signature:
                raise ValueError(
                    "search_id does not match the current ranking configuration. "
                    "Please start a new search."
                )

            stored_query = (stored_search.get("query") or "").strip().lower()
            stored_role = stored_search.get("role") or None
            incoming_role = role or None

            if stored_query != query.strip().lower() or stored_role != incoming_role:
                # Query or role changed — start a fresh search
                search_id = self._build_signed_search_id(expected_signature)
                database_service.log_search(search_id=search_id, query=query, role=role)

        return search_id

    @classmethod
    def _build_search_signature(
        cls,
        method: str,
        bm25_weight: Optional[float],
        semantic_weight: Optional[float],
        rrf_k: Optional[int],
        temaside_boost: Optional[float],
        retningslinje_boost: Optional[float],
    ) -> str:
        """Build a stable signature for the ranking configuration."""
        effective_tunables = {
            "method": method.strip().lower(),
            "bm25_weight": round(
                float(
                    bm25_weight
                    if bm25_weight is not None
                    else settings.search_rrf_weight_bm25
                ),
                6,
            ),
            "semantic_weight": round(
                float(
                    semantic_weight
                    if semantic_weight is not None
                    else settings.search_rrf_weight_semantic
                ),
                6,
            ),
            "rrf_k": max(1, int(rrf_k if rrf_k is not None else settings.search_rrf_k)),
            "temaside_boost": round(
                float(
                    temaside_boost
                    if temaside_boost is not None
                    else settings.search_boost_temaside
                ),
                6,
            ),
            "retningslinje_boost": round(
                float(
                    retningslinje_boost
                    if retningslinje_boost is not None
                    else settings.search_boost_retningslinje
                ),
                6,
            ),
        }
        canonical = json.dumps(effective_tunables, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[: cls._SEARCH_ID_SIGNATURE_HEX_LENGTH]

    @classmethod
    def _build_signed_search_id(cls, signature: str) -> str:
        """
        Encode a short configuration signature into the UUID tail.

        Layout: <20 random hex><marker 4 hex><signature 8 hex>.
        """
        random_hex = uuid.uuid4().hex
        signed_tail = f"{cls._SEARCH_ID_SIGNATURE_MARKER}{signature}"
        signed_hex = f"{random_hex[:-12]}{signed_tail}"
        return str(uuid.UUID(signed_hex))

    @classmethod
    def _extract_search_signature(cls, search_id: str) -> Optional[str]:
        """Extract configuration signature from a signed UUID search_id."""
        try:
            search_hex = uuid.UUID(search_id).hex
        except ValueError:
            return None

        signed_tail = search_hex[-12:]
        if not signed_tail.startswith(cls._SEARCH_ID_SIGNATURE_MARKER):
            return None
        return signed_tail[len(cls._SEARCH_ID_SIGNATURE_MARKER):]

    async def get_theme_pages(
        self,
        category: Optional[str] = None,
    ) -> ThemePageResponse:
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

    def get_suggestions(self, query: str) -> SuggestionResponse:
        """
        Get autocomplete suggestions from theme pages.

        Matches theme page titles against query using prefix and substring matching.
        Returns up to 5 suggestions, with prefix matches ranked first.

        Args:
            query: Partial search text from user input

        Returns:
            SuggestionResponse with up to 5 matching theme pages
        """
        max_suggestions = 5
        query_lower = query.lower().strip()

        if not query_lower:
            return SuggestionResponse(suggestions=[])

        all_content = content_service.get_all_content()
        theme_pages = [item for item in all_content if item.content_type.lower() == 'temaside']

        prefix_matches = []
        substring_matches = []

        for page in theme_pages:
            title_lower = page.title.lower()
            if title_lower.startswith(query_lower):
                prefix_matches.append(page)
            elif query_lower in title_lower:
                substring_matches.append(page)

        # Sort each group by title length (shorter = more specific)
        prefix_matches.sort(key=lambda p: len(p.title))
        substring_matches.sort(key=lambda p: len(p.title))

        # Prefix matches first, then substring matches
        matched = prefix_matches + substring_matches
        top = matched[:max_suggestions]

        suggestions = [Suggestion(id=page.id, title=page.title) for page in top]
        return SuggestionResponse(suggestions=suggestions)

    @staticmethod
    def _compute_type_match(info_type: Optional[str]) -> float:
        """Content type authority score."""
        content_type_map = {
            "retningslinje": 0.9,
            "veileder": 0.8,
            "fagprosedyre": 0.75,
            "faktaark": 0.6,
            "artikkel": 0.5,
        }
        return content_type_map.get(
            info_type.lower() if info_type else None,
            0.5,
        )

    @staticmethod
    def _compute_role_match(role: Optional[str], role_tags: Optional[List[str]]) -> float:
        """Role match score."""
        role_tags = role_tags or []
        if role and role_tags:
            if role in role_tags:
                return 1.0 / len(role_tags)
        elif not role and not role_tags:
            return 0.5
        elif not role_tags:
            return 0.3
        return 0.0

    def _log_results(
        self,
        search_id: str,
        query: str,
        role: Optional[str],
        results: List[SearchResult],
        offset: int,
    ) -> None:
        """Log results with pipeline scores to database."""
        # Get already logged content_ids to avoid duplicates
        already_logged = database_service.get_logged_content_ids_for_search(search_id)

        # Filter out already logged results
        new_results = [r for r in results if r.id not in already_logged]

        if not new_results:
            return  # Nothing new to log

        # Get max position to continue from
        max_position = database_service.get_max_position_for_search(search_id)

        results_to_log = []
        for local_index, result in enumerate(new_results):
            position = max_position + local_index + 1
            content_item = content_service.get_content_by_id(result.id)

            # Metadata features from content item (lightweight)
            type_match = 0.5
            role_match = 0.0
            maalgruppe_match = 0
            if content_item:
                type_match = self._compute_type_match(content_item.info_type)
                role_match = self._compute_role_match(role, content_item.role_tags)
                maalgruppe_match = 1 if role and role in (content_item.role_tags or []) else 0

            results_to_log.append({
                "content_id": result.id,
                "position": position,
                "score": result.score,
                "semantic_score": result.semantic_score or 0.0,
                "bm25_score": result.bm25_score or 0.0,
                "rrf_score": result.rrf_score or 0.0,
                "type_match": type_match,
                "role_match": role_match,
                "maalgruppe_match": maalgruppe_match,
            })

        database_service.log_search_results(search_id, results_to_log)


# Global instance
search_controller = SearchController()
