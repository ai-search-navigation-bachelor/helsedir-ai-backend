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

        # Cached embeddings matrix for fast vectorized search
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._content_ids: List[str] = []  # Maps matrix row index to content_id

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
        """Load embeddings from database into memory cache and build matrix."""
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
            embeddings_list = []
            content_ids_list = []

            for content_id, embedding_bytes in rows:
                if embedding_bytes:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    self.content_embeddings[content_id] = embedding
                    embeddings_list.append(embedding)
                    content_ids_list.append(content_id)

            # Only mark as loaded if at least one embedding was found
            if len(self.content_embeddings) > 0:
                # Build cached matrix for fast vectorized search
                self._embeddings_matrix = np.vstack(embeddings_list)
                self._content_ids = content_ids_list

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
        Perform semantic search using E5 embeddings (optimized with cached matrix).

        Returns empty list if embeddings not available.
        """
        if not self.is_available() or self._embeddings_matrix is None:
            return []

        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []

        # Vectorized similarity calculation using cached matrix (fast!)
        similarities = np.dot(self._embeddings_matrix, query_embedding)  # Shape: (n_docs,)

        # Build id -> similarity mapping for filtering
        id_to_similarity = {
            content_id: float(sim)
            for content_id, sim in zip(self._content_ids, similarities)
        }

        # Filter content items by role and info_type
        valid_items = []
        for item in content_service.get_all_content():
            # Filter by info_type
            if not is_allowed_info_type(item.content_type):
                continue

            # Filter by role
            if role and role not in item.target_groups:
                continue

            # Check if embedding exists
            if item.id in id_to_similarity:
                valid_items.append((item, id_to_similarity[item.id]))

        if not valid_items:
            return []

        # Sort by similarity and take top-k
        valid_items.sort(key=lambda x: x[1], reverse=True)
        top_items = valid_items[:k]

        # Build results
        results = []
        for item, sem_score in top_items:
            norm_score = (sem_score + 1) / 2
            results.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    info_type=item.content_type,
                    path=item.path,
                    score=round(norm_score, 3),
                    explanation=f"Semantic: {sem_score:.3f} → {norm_score:.2f}",
                )
            )

        return results


# Global instance
semantic_search = SemanticSearch()
