"""
Search service facade.

Provides unified interface for all search methods.
"""

from typing import List, Optional

from app.dto.response.search import SearchResult
from app.services.search.keyword_search import keyword_search
from app.services.search.semantic_search import semantic_search
from app.services.search.hybrid_search import hybrid_search


class SearchService:
    """
    Unified search service facade.

    Delegates to specialized search implementations.
    """

    def __init__(self):
        self._keyword = keyword_search
        self._semantic = semantic_search
        self._hybrid = hybrid_search

    # ==================== Search Methods ====================

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """Perform keyword-based search."""
        result = self._keyword.search(query, role, k)
        return result if result is not None else []

    def search_semantic(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        return self._semantic.search(query, role, k)

    def search_hybrid(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        temaside_boost: Optional[float] = None,
        retningslinje_boost: Optional[float] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search combining BM25 and semantic retrieval (RRF fusion)."""
        return self._hybrid.search(
            query,
            role,
            k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
            temaside_boost=temaside_boost,
            retningslinje_boost=retningslinje_boost,
        )

    # ==================== Utility Methods ====================

    def get_semantic_similarity(self, query: str, content_id: str) -> Optional[float]:
        """Calculate semantic similarity between a query and content item."""
        return self._semantic.get_similarity(query, content_id)

    # ==================== Internal Access (for backward compatibility) ====================

    def _load_embedding_model(self) -> bool:
        """Load the embedding model."""
        return self._semantic.load_embedding_model()

    def _load_content_embeddings(self) -> bool:
        """Load content embeddings from database."""
        return self._semantic.load_content_embeddings()

    @property
    def embedding_model(self):
        """Access to embedding model (for backward compatibility)."""
        return self._semantic.embedding_model

    @property
    def content_embeddings(self):
        """Access to content embeddings cache (for backward compatibility)."""
        return self._semantic.content_embeddings

    def _calculate_score(self, item, query_lower: str, query_keywords: set) -> float:
        """Calculate keyword score (for backward compatibility)."""
        return self._keyword.calculate_score(item, query_lower, query_keywords)


# Global instance
search_service = SearchService()
