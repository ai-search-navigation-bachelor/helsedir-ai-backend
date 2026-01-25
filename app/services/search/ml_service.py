"""
ML Service for coordinating machine learning models.

This service manages loading and using ML models for:
- Semantic search via embeddings
- Learning-to-rank for result scoring
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.services.data.content_service import content_service
from app.ml.embedding_model import HealthContentEmbedding
from app.ml.ranking_model import HealthContentRanker


class MLService:
    """
    Service for coordinating ML models.

    Handles loading, caching, and inference for:
    - Embedding model for semantic similarity
    - Ranking model for learning-to-rank
    """

    def __init__(self):
        self.embedding_model: Optional[HealthContentEmbedding] = None
        self.ranking_model: Optional[HealthContentRanker] = None
        self.content_embeddings: Optional[np.ndarray] = None
        self.content_ids: List[str] = []
        self._embedding_loaded = False
        self._ranking_loaded = False

    def load_embedding_model(self) -> bool:
        """
        Load the embedding model from disk.

        Returns:
            True if model was loaded successfully
        """
        if not settings.ml_embedding_enabled:
            return False

        model_path = Path(settings.ml_models_dir) / "embedding" / "model"
        keras_path = Path(f"{model_path}.keras")

        if not keras_path.exists():
            print(f"Embedding model not found at {keras_path}")
            return False

        try:
            self.embedding_model = HealthContentEmbedding.load(str(model_path))
            self._precompute_embeddings()
            self._embedding_loaded = True
            print("Embedding model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return False

    def load_ranking_model(self) -> bool:
        """
        Load the ranking model from disk.

        Returns:
            True if model was loaded successfully
        """
        if not settings.ml_ranking_enabled:
            return False

        model_path = Path(settings.ml_models_dir) / "ranking" / "model"
        keras_path = Path(f"{model_path}.keras")

        if not keras_path.exists():
            print(f"Ranking model not found at {keras_path}")
            return False

        try:
            self.ranking_model = HealthContentRanker.load(str(model_path))
            self._ranking_loaded = True
            print("Ranking model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading ranking model: {e}")
            return False

    def _precompute_embeddings(self) -> None:
        """Precompute embeddings for all content items."""
        if self.embedding_model is None:
            return

        content_items = content_service.get_all_content()
        if not content_items:
            print("No content items to embed")
            return

        # Combine title and body for embedding
        texts = [f"{item.title} {item.body}" for item in content_items]
        self.content_ids = [item.id for item in content_items]

        try:
            self.content_embeddings = self.embedding_model.encode(texts)
            print(f"Precomputed embeddings for {len(texts)} content items")
        except Exception as e:
            print(f"Error precomputing embeddings: {e}")
            self.content_embeddings = None

    def get_semantic_scores(
        self, query: str, content_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get semantic similarity scores for content items.

        Args:
            query: Search query
            content_ids: Optional list of content IDs to score (default: all)

        Returns:
            Dictionary mapping content_id to semantic similarity score
        """
        if self.embedding_model is None or self.content_embeddings is None:
            return {}

        try:
            # Encode query (with "query: " prefix for E5)
            query_embedding = self.embedding_model.encode_query(query)

            # Compute similarities
            similarities = self.embedding_model.compute_similarity(
                query_embedding, self.content_embeddings
            )

            # Build result dict
            scores = {}
            for i, content_id in enumerate(self.content_ids):
                if content_ids is None or content_id in content_ids:
                    scores[content_id] = float(similarities[i])

            return scores
        except Exception as e:
            print(f"Error computing semantic scores: {e}")
            return {}

    def get_ranking_scores(
        self, features: List[Dict[str, float]]
    ) -> List[float]:
        """
        Get ranking scores from the learning-to-rank model.

        Args:
            features: List of feature dictionaries for each result

        Returns:
            List of ranking scores
        """
        if self.ranking_model is None:
            return [0.0] * len(features)

        try:
            return self.ranking_model.predict(features)
        except Exception as e:
            print(f"Error computing ranking scores: {e}")
            return [0.0] * len(features)

    def is_embedding_available(self) -> bool:
        """Check if embedding model is loaded and available."""
        return self._embedding_loaded and self.content_embeddings is not None

    def is_ranking_available(self) -> bool:
        """Check if ranking model is loaded and available."""
        return self._ranking_loaded

    def reload_models(self) -> Dict[str, bool]:
        """
        Reload all ML models.

        Returns:
            Dictionary with load status for each model
        """
        return {
            "embedding": self.load_embedding_model(),
            "ranking": self.load_ranking_model(),
        }

    def refresh_embeddings(self) -> bool:
        """
        Refresh content embeddings (e.g., after content update).

        Returns:
            True if embeddings were refreshed successfully
        """
        if not self._embedding_loaded:
            return False

        try:
            self._precompute_embeddings()
            return self.content_embeddings is not None
        except Exception as e:
            print(f"Error refreshing embeddings: {e}")
            return False


# Global instance
ml_service = MLService()
