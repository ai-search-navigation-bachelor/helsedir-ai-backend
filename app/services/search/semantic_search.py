"""
Semantic search functionality using embeddings.
"""

from typing import List, Optional, Dict
from pathlib import Path

import numpy as np

from app.dto.response.search import SearchResult
from app.services.data.content_service import content_service
from app.constants import is_allowed_info_type


class SemanticSearch:
    """Semantic search using E5 embeddings."""

    def __init__(self):
        self.embedding_model = None
        self.content_embeddings: Dict[str, np.ndarray] = {}
        self._embeddings_loaded = False
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

    def load_embedding_model(self) -> bool:
        """Load the E5 embedding model if available."""
        if self.embedding_model is not None:
            return True

        try:
            from app.ml.embedding_model import HealthContentEmbedding

            from app.config import settings
            self.embedding_model = HealthContentEmbedding(model_name=settings.ml_embedding_model)
            return True
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            return False

    def load_content_embeddings(self) -> bool:
        """Load embeddings from database into memory cache."""
        if self._embeddings_loaded:
            return True

        from app.services.repositories.base import db_pool

        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM content WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()

            self.content_embeddings = {}
            for content_id, embedding_bytes in rows:
                if embedding_bytes:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    self.content_embeddings[content_id] = embedding

            # Only mark as loaded if at least one embedding was found
            if len(self.content_embeddings) > 0:
                self._embeddings_loaded = True
                print(f"Loaded {len(self.content_embeddings)} embeddings into cache")
                return True
            else:
                print("No embeddings found in database")
                return False

        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def is_available(self) -> bool:
        """Check if semantic search is available."""
        return self.load_embedding_model() and self.load_content_embeddings()

    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query (cached)."""
        if not self.load_embedding_model():
            return None

        if query not in self._query_embedding_cache:
            self._query_embedding_cache = {query: self.embedding_model.encode_query(query)}

        return self._query_embedding_cache[query]

    def get_similarity(self, query: str, content_id: str) -> Optional[float]:
        """
        Calculate semantic similarity between a query and a content item.

        Returns:
            Cosine similarity score, or None if embeddings unavailable
        """
        if not self.is_available():
            return None

        if content_id not in self.content_embeddings:
            return None

        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return None

        doc_embedding = self.content_embeddings[content_id]

        # Cosine similarity (embeddings are L2-normalized)
        return float(np.dot(query_embedding, doc_embedding))

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Perform semantic search using E5 embeddings (vectorized).

        Returns empty list if embeddings not available.
        """
        if not self.is_available():
            return []

        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []

        # Pre-filter content items and prepare for vectorized operations
        valid_items = []
        valid_embeddings = []

        for item in content_service.get_all_content():
            # Filter by info_type
            if not is_allowed_info_type(item.content_type):
                continue

            # Filter by role
            if role and role not in item.target_groups:
                continue

            # Check if embedding exists
            if item.id in self.content_embeddings:
                valid_items.append(item)
                valid_embeddings.append(self.content_embeddings[item.id])

        if not valid_embeddings:
            return []

        # Vectorized similarity calculation (much faster than loop)
        embeddings_matrix = np.vstack(valid_embeddings)  # Shape: (n_docs, 768)
        similarities = np.dot(embeddings_matrix, query_embedding)  # Shape: (n_docs,)

        # Get top-k indices (argsort returns ascending, so reverse)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Build results
        results = []
        for idx in top_k_indices:
            item = valid_items[idx]
            sem_score = float(similarities[idx])
            norm_score = (sem_score + 1) / 2
            results.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    info_type=item.content_type,
                    score=round(norm_score, 3),
                    explanation=f"Semantic: {sem_score:.3f} → {norm_score:.2f}",
                )
            )

        return results


# Global instance
semantic_search = SemanticSearch()
