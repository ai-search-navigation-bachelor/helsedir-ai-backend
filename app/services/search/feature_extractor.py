"""
ML feature extraction for search results.
"""

import re
from typing import Optional, Dict, Any

from app.services.search.search_service import search_service
from app.config import settings


class FeatureExtractor:
    """Extracts ML features for content items."""

    def extract_features(
        self,
        item,
        query: str,
        query_keywords: set,
        role: Optional[str],
    ) -> Dict[str, Any]:
        """
        Calculate ML features for a content item.

        Features extracted:
        - semantic_similarity: Cosine similarity from embedding (-1 to 1)
        - keyword_score_total: Total keyword score
        - exact_title_proportion: Proportion from exact title match
        - full_coverage_proportion: Proportion from full title coverage
        - title_keyword_proportion: Proportion from title keyword matches
        - type_match: Content type authority level (0-1)
        - role_match: User role match with target groups (0-1)
        - code_match_count: Number of matched codes
        - lis_match: LIS code match (0-1)
        - maalgruppe_match: Target group match (0-1)
        """
        query_lower = query.lower()
        title_lower = item.title.lower() if item.title else ""

        # Calculate semantic similarity (normalize None to 0.0)
        semantic_similarity = search_service.get_semantic_similarity(query, item.id)
        if semantic_similarity is None:
            semantic_similarity = 0.0

        # Calculate keyword score components (title-only)
        exact_title_score = 0.0
        full_coverage_score = 0.0
        title_keyword_score = 0.0

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

        # Total keyword score
        total_keyword_score = exact_title_score + full_coverage_score + title_keyword_score

        # Calculate proportions
        if total_keyword_score > 0:
            exact_title_prop = exact_title_score / total_keyword_score
            full_coverage_prop = full_coverage_score / total_keyword_score
            title_keyword_prop = title_keyword_score / total_keyword_score
        else:
            exact_title_prop = 0.0
            full_coverage_prop = 0.0
            title_keyword_prop = 0.0

        # Type match (content authority)
        type_match = self._calculate_type_match(item.info_type)

        # Role match
        role_match = self._calculate_role_match(role, item.target_groups)

        # Maalgruppe match
        target_groups = item.target_groups or []
        maalgruppe_match = 1 if role and role in target_groups else 0

        return {
            "semantic_similarity": semantic_similarity,
            "keyword_score_total": total_keyword_score,
            "exact_title_proportion": exact_title_prop,
            "full_coverage_proportion": full_coverage_prop,
            "title_keyword_proportion": title_keyword_prop,
            "type_match": type_match,
            "role_match": role_match,
            "code_match_count": 0,  # TODO: implement code matching
            "lis_match": 0,  # TODO: implement LIS matching
            "maalgruppe_match": maalgruppe_match,
        }

    def _calculate_type_match(self, info_type: Optional[str]) -> float:
        """Calculate content type authority score."""
        content_type_map = {
            "retningslinje": 0.9,
            "veileder": 0.8,
            "fagprosedyre": 0.75,
            "faktaark": 0.6,
            "artikkel": 0.5,
        }
        return content_type_map.get(
            info_type.lower() if info_type else None,
            0.5
        )

    def _calculate_role_match(
        self,
        role: Optional[str],
        target_groups: Optional[list]
    ) -> float:
        """Calculate role match score."""
        target_groups = target_groups or []

        if role and target_groups:
            if role in target_groups:
                return 1.0 / len(target_groups)
        elif not role and not target_groups:
            return 0.5
        elif not target_groups:
            return 0.3

        return 0.0


# Global instance
feature_extractor = FeatureExtractor()
