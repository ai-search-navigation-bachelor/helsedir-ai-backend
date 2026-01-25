"""
Search controller.

Handles business logic for search operations with pagination and ML feature logging.
"""

import uuid
import re
from typing import Optional, List
from app.dto.response.search import SearchResult, SearchResponse
from app.services.search.search_service import search_service
from app.services.data.database_service import database_service
from app.services.data.content_service import content_service
from app.config import settings


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

        Raises:
            ValueError: If invalid search method
        """
        # Validate method
        valid_methods = {"keyword", "semantic", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid search method: {method}. Must be one of {valid_methods}")

        # Get all results up to offset + limit for this page
        # We fetch more to know the total count
        max_results = 100  # Max results to consider

        if method == "semantic":
            all_results = self.search_service.search_semantic(
                query=query, role=role, k=max_results
            )
        elif method == "keyword":
            all_results = self.search_service.search(
                query=query, role=role, k=max_results
            )
        else:  # hybrid
            all_results = self.search_service.search_hybrid(
                query=query, role=role, k=max_results
            )

        total = len(all_results)

        # Apply pagination
        page_results = all_results[offset:offset + limit]

        # Generate or reuse search_id
        is_new_search = search_id is None
        if is_new_search:
            search_id = str(uuid.uuid4())
            # Log new search to search_logs
            database_service.log_search(
                search_id=search_id,
                query=query,
                role=role,
            )
        else:
            # Validate that query matches the original search
            stored_search = database_service.get_search_by_id(search_id)
            if stored_search is None:
                raise ValueError(f"Invalid search_id: {search_id}")
            if stored_search["query"] != query:
                raise ValueError("Query does not match the original search for this search_id")

        # Extract ML features and log results shown
        results_with_features = self._extract_and_log_results(
            search_id=search_id,
            query=query,
            role=role,
            results=page_results,
            offset=offset,
        )

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

    def _extract_and_log_results(
        self,
        search_id: str,
        query: str,
        role: Optional[str],
        results: List[SearchResult],
        offset: int,
    ) -> List[dict]:
        """
        Extract ML features for each result and log to database.

        Args:
            search_id: The search ID
            query: Search query
            role: User role
            results: List of SearchResult objects
            offset: Current offset for calculating actual position

        Returns:
            List of result dicts with features
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        results_to_log = []

        for local_index, result in enumerate(results):
            # Actual position in full result set (1-indexed)
            position = offset + local_index + 1

            # Get content item for feature extraction
            content_item = content_service.get_content_by_id(result.id)

            features = {}
            if content_item:
                features = self._calculate_features(
                    content_item, query, query_keywords, role
                )

            results_to_log.append({
                "content_id": result.id,
                "position": position,
                "semantic_similarity": features.get("semantic_similarity"),
                "keyword_score_total": features.get("keyword_score_total"),
                "exact_title_proportion": features.get("exact_title_proportion"),
                "full_coverage_proportion": features.get("full_coverage_proportion"),
                "title_keyword_proportion": features.get("title_keyword_proportion"),
                "body_keyword_proportion": features.get("body_keyword_proportion"),
                "exact_body_proportion": features.get("exact_body_proportion"),
                "type_match": features.get("type_match"),
                "role_match": features.get("role_match"),
                "code_match_count": features.get("code_match_count", 0),
                "lis_match": features.get("lis_match", 0),
                "maalgruppe_match": features.get("maalgruppe_match", 0),
            })

        # Log results to database
        database_service.log_search_results(search_id, results_to_log)

        return results_to_log

    def _calculate_features(
        self,
        item,
        query: str,
        query_keywords: set,
        role: Optional[str],
    ) -> dict:
        """Calculate ML features for a content item."""
        query_lower = query.lower()
        title_lower = item.title.lower() if item.title else ""
        body_lower = item.body.lower() if item.body else ""

        # Calculate semantic similarity
        semantic_similarity = self.search_service.get_semantic_similarity(query, item.id)

        # Calculate keyword score components
        exact_title_score = 0.0
        full_coverage_score = 0.0
        title_keyword_score = 0.0
        body_keyword_score = 0.0
        exact_body_score = 0.0

        # Exact phrase in title
        if query_lower in title_lower:
            exact_title_score = settings.search_exact_phrase_title_weight

        # Full title coverage
        title_keywords = set(re.findall(r'\w+', title_lower))
        if title_keywords and title_keywords.issubset(query_keywords):
            full_coverage_score = settings.search_full_title_coverage_weight

        # Title keyword matches
        title_matches = query_keywords & title_keywords
        if title_matches:
            title_keyword_score = len(title_matches) * settings.search_keyword_title_weight

        # Body keyword matches
        body_keywords = set(re.findall(r'\w+', body_lower))
        body_matches = query_keywords & body_keywords
        if body_matches:
            body_keyword_score = len(body_matches) * settings.search_keyword_body_weight

        # Exact phrase in body
        if query_lower in body_lower:
            exact_body_score = settings.search_exact_phrase_body_weight

        # Total keyword score
        total_keyword_score = (
            exact_title_score + full_coverage_score +
            title_keyword_score + body_keyword_score + exact_body_score
        )

        # Calculate proportions
        if total_keyword_score > 0:
            exact_title_prop = exact_title_score / total_keyword_score
            full_coverage_prop = full_coverage_score / total_keyword_score
            title_keyword_prop = title_keyword_score / total_keyword_score
            body_keyword_prop = body_keyword_score / total_keyword_score
            exact_body_prop = exact_body_score / total_keyword_score
        else:
            exact_title_prop = 0.0
            full_coverage_prop = 0.0
            title_keyword_prop = 0.0
            body_keyword_prop = 0.0
            exact_body_prop = 0.0

        # Type match (content authority)
        content_type_map = {
            "retningslinje": 0.9,
            "veileder": 0.8,
            "fagprosedyre": 0.75,
            "faktaark": 0.6,
            "artikkel": 0.5,
        }
        type_match = content_type_map.get(
            item.info_type.lower() if item.info_type else None,
            0.5
        )

        # Role match
        role_match = 0.0
        target_groups = item.target_groups or []
        if role and target_groups:
            if role in target_groups:
                role_match = 1.0 / len(target_groups)
        elif not role and not target_groups:
            role_match = 0.5
        elif not target_groups:
            role_match = 0.3

        # Maalgruppe match
        maalgruppe_match = 1 if role and role in target_groups else 0

        return {
            "semantic_similarity": semantic_similarity,
            "keyword_score_total": total_keyword_score,
            "exact_title_proportion": exact_title_prop,
            "full_coverage_proportion": full_coverage_prop,
            "title_keyword_proportion": title_keyword_prop,
            "body_keyword_proportion": body_keyword_prop,
            "exact_body_proportion": exact_body_prop,
            "type_match": type_match,
            "role_match": role_match,
            "code_match_count": 0,  # TODO: implement code matching
            "lis_match": 0,  # TODO: implement LIS matching
            "maalgruppe_match": maalgruppe_match,
        }


# Global instance
search_controller = SearchController()
