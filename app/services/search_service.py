from typing import List, Optional
from app.models.schemas import SearchResult, ContentItem
from app.services.content_service import content_service
import re


class SearchService:
    """Service for searching content (baseline implementation)."""

    def __init__(self):
        self.content_service = content_service

    def search(self, query: str, role: Optional[str] = None, k: int = 10) -> List[SearchResult]:
        """
        Perform baseline keyword search.

        This is a simple implementation that scores based on:
        - Exact matches in title (high weight)
        - Keyword matches in title (medium weight)
        - Keyword matches in body (low weight)
        - Role filtering if specified
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        results = []
        for item in self.content_service.get_all_content():
            # Filter by role if specified
            if role and role not in item.target_groups:
                continue

            score = self._calculate_score(item, query_lower, query_keywords)

            if score > 0:
                snippet = self._create_snippet(item.body, query_keywords)
                explanation = self._create_explanation(item, query_keywords, role)

                results.append(
                    SearchResult(
                        id=item.id,
                        title=item.title,
                        url=item.url,
                        snippet=snippet,
                        score=score,
                        explanation=explanation,
                    )
                )

        # Sort by score (descending) and ensure consistent ordering
        results.sort(key=lambda x: (-x.score, x.id))

        return results[:k]

    def _calculate_score(self, item: ContentItem, query_lower: str, query_keywords: set) -> float:
        """Calculate relevance score for a content item."""
        score = 0.0

        title_lower = item.title.lower()
        body_lower = item.body.lower()

        # Exact phrase match in title (highest weight)
        if query_lower in title_lower:
            score += 10.0

        # Keyword matches in title
        title_keywords = set(re.findall(r'\w+', title_lower))
        title_matches = query_keywords & title_keywords
        score += len(title_matches) * 3.0

        # Keyword matches in body
        body_keywords = set(re.findall(r'\w+', body_lower))
        body_matches = query_keywords & body_keywords
        score += len(body_matches) * 1.0

        # Exact phrase match in body
        if query_lower in body_lower:
            score += 2.0

        # Tag matches
        if item.tags:
            tag_text = " ".join(item.tags).lower()
            tag_keywords = set(re.findall(r'\w+', tag_text))
            tag_matches = query_keywords & tag_keywords
            score += len(tag_matches) * 2.0

        return score

    def _create_snippet(self, body: str, query_keywords: set, max_length: int = 200) -> str:
        """Create a snippet highlighting relevant parts of the body."""
        # Find the first occurrence of any query keyword
        body_lower = body.lower()
        best_pos = len(body)

        for keyword in query_keywords:
            pos = body_lower.find(keyword)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        # Extract snippet around the keyword
        if best_pos < len(body):
            start = max(0, best_pos - 50)
            end = min(len(body), best_pos + max_length - 50)

            snippet = body[start:end].strip()

            if start > 0:
                snippet = "..." + snippet
            if end < len(body):
                snippet = snippet + "..."

            return snippet
        else:
            # No keyword found, return beginning
            return body[:max_length].strip() + ("..." if len(body) > max_length else "")

    def _create_explanation(self, item: ContentItem, query_keywords: set, role: Optional[str]) -> str:
        """Create a short explanation of why this result matches."""
        explanations = []

        title_keywords = set(re.findall(r'\w+', item.title.lower()))
        title_matches = query_keywords & title_keywords

        if title_matches:
            explanations.append(f"matches in title")

        if role and role in item.target_groups:
            explanations.append(f"relevant for {role}")

        if item.tags:
            tag_keywords = set(re.findall(r'\w+', " ".join(item.tags).lower()))
            tag_matches = query_keywords & tag_keywords
            if tag_matches:
                explanations.append(f"related tags: {', '.join(list(tag_matches)[:3])}")

        if not explanations:
            explanations.append("matches query terms")

        return "Relevant: " + "; ".join(explanations)


# Global instance
search_service = SearchService()
