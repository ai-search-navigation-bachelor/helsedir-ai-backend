"""
Keyword-based search functionality.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from app.dto.response.search import SearchResult
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.config import settings


class KeywordSearch:
    """Keyword-based search implementation."""

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Perform baseline keyword search.

        Scores based on:
        - Exact matches in title (high weight)
        - Keyword matches in title (medium weight)
        - Full title coverage (all title words in query)
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        results = []
        for item in content_service.get_all_content():
            if item.content_type not in content_service.searchable_types:
                continue

            if role and role not in item.role_tags:
                continue

            score, breakdown = self.calculate_score_with_breakdown(
                item, query_lower, query_keywords
            )

            if score > 0:
                results.append((item, score, breakdown))

        # Sort by score (descending) and ensure consistent ordering
        results.sort(key=lambda x: (-x[1], x[0].id))

        # Normalize scores (max score = 1.0)
        if results:
            max_score = results[0][1]
            normalized = []
            for item, raw_score, breakdown in results[:k]:
                norm_score = raw_score / max_score if max_score > 0 else 0
                explanation = self._create_explanation(breakdown, role)
                explanation += f" | Score: {raw_score:.1f} → {norm_score:.2f}"
                normalized.append(
                    SearchResult(
                        id=item.id,
                        title=item.title,
                        info_type=item.content_type,
                        path=item.path,
                        score=round(norm_score, 3),
                        explanation=explanation,
                    )
                )
            return normalized

        return []

    def calculate_score(
        self,
        item: ContentItem,
        query_lower: str,
        query_keywords: set
    ) -> float:
        """Calculate relevance score for a content item."""
        score, _ = self.calculate_score_with_breakdown(item, query_lower, query_keywords)
        return score

    def calculate_score_with_breakdown(
        self,
        item: ContentItem,
        query_lower: str,
        query_keywords: set
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate relevance score with detailed breakdown (title-only)."""
        score = 0.0
        breakdown = {}

        title_lower = item.title.lower()

        # Exact phrase match in title (highest weight)
        if query_lower in title_lower:
            points = settings.search_exact_phrase_title_weight
            score += points
            breakdown['exact_title'] = points

        # Keyword matches in title
        title_keywords = set(re.findall(r'\w+', title_lower))
        title_matches = query_keywords & title_keywords
        if title_matches:
            points = len(title_matches) * settings.search_keyword_title_weight
            score += points
            breakdown['title_keywords'] = {
                'count': len(title_matches),
                'matches': list(title_matches),
                'points': points
            }

        # Full title coverage: all title words present in query
        if title_keywords and title_keywords.issubset(query_keywords):
            points = settings.search_full_title_coverage_weight
            score += points
            breakdown['full_title_coverage'] = points

        return score, breakdown

    def _create_explanation(
        self,
        breakdown: Dict[str, Any],
        role: Optional[str]
    ) -> str:
        """Create detailed explanation from score breakdown."""
        parts = []

        if 'exact_title' in breakdown:
            parts.append(f"Exact title match (+{breakdown['exact_title']:.1f})")

        if 'full_title_coverage' in breakdown:
            parts.append(f"Full title coverage (+{breakdown['full_title_coverage']:.1f})")

        if 'title_keywords' in breakdown:
            kw = breakdown['title_keywords']
            parts.append(f"Title words: {', '.join(kw['matches'][:3])} (+{kw['points']:.1f})")

        if role:
            parts.append(f"Role: {role}")

        return " | ".join(parts) if parts else "Match"


# Global instance
keyword_search = KeywordSearch()
