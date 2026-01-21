"""
TensorFlow learning-to-rank model for health content search.

This module provides a ranking model that combines multiple features
to predict the relevance of search results.
"""

from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Feature names used by the ranking model
RANKING_FEATURES = [
    "title_keyword_score",
    "body_keyword_score",
    "tag_match_score",
    "exact_phrase_title",
    "exact_phrase_body",
    "semantic_similarity",
    "content_type_encoded",
    "days_since_published",
    "historical_ctr",
    "role_match",
]


class HealthContentRanker:
    """
    Learning-to-rank model that combines multiple features.

    Features include:
    - Keyword match scores (from baseline search)
    - Semantic similarity (from embedding model)
    - Content metadata (type, age, etc.)
    - Historical CTR (click-through rate)

    Architecture:
    - Multi-layer feedforward neural network
    - Sigmoid output for relevance probability
    """

    def __init__(self, num_features: int = 10, hidden_units: List[int] = None):
        """
        Initialize the ranking model.

        Args:
            num_features: Number of input features
            hidden_units: List of hidden layer sizes (default: [64, 32])
        """
        self.num_features = num_features
        self.hidden_units = hidden_units or [64, 32]
        self.feature_names = RANKING_FEATURES[:num_features]

        self.model = self._build_model()

    def _build_model(self) -> "keras.Model":
        """Build the ranking model architecture."""
        # Input layer
        features_input = keras.Input(
            shape=(self.num_features,), name="features_input"
        )

        x = features_input

        # Hidden layers with batch normalization and dropout
        for i, units in enumerate(self.hidden_units):
            x = keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(1e-4),
                name=f"dense_{i}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = keras.layers.Dropout(0.2, name=f"dropout_{i}")(x)

        # Output layer - relevance score
        output = keras.layers.Dense(
            1, activation="sigmoid", name="relevance_score"
        )(x)

        model = keras.Model(
            inputs=features_input, outputs=output, name="health_ranker"
        )
        return model

    def compile(
        self,
        learning_rate: float = 1e-3,
        loss: str = "binary_crossentropy",
    ) -> None:
        """
        Compile the model for training.

        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function (default: binary_crossentropy for click/no-click)
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
    ) -> "keras.callbacks.History":
        """
        Train the ranking model.

        Args:
            features: Training features of shape (n_samples, num_features)
            labels: Training labels (1 = clicked, 0 = not clicked)
            validation_split: Fraction of data for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=early_stopping_patience,
                mode="max",
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
        ]

        return self.model.fit(
            features,
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    def predict(self, features: List[Dict[str, float]]) -> List[float]:
        """
        Predict relevance scores for search results.

        Args:
            features: List of feature dictionaries

        Returns:
            List of relevance scores (0-1)
        """
        # Convert feature dicts to numpy array
        feature_array = self._features_to_array(features)
        predictions = self.model.predict(feature_array, verbose=0)
        return predictions.flatten().tolist()

    def _features_to_array(
        self, features: List[Dict[str, float]]
    ) -> np.ndarray:
        """Convert list of feature dicts to numpy array."""
        result = np.zeros((len(features), self.num_features))

        for i, feature_dict in enumerate(features):
            for j, name in enumerate(self.feature_names):
                result[i, j] = feature_dict.get(name, 0.0)

        return result

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model (without extension)
        """
        self.model.save(f"{path}.keras")

    @classmethod
    def load(cls, path: str) -> "HealthContentRanker":
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model (without extension)

        Returns:
            Loaded HealthContentRanker instance
        """
        instance = cls.__new__(cls)
        instance.model = keras.models.load_model(f"{path}.keras")
        instance.num_features = instance.model.input_shape[1]
        instance.feature_names = RANKING_FEATURES[: instance.num_features]
        return instance


def extract_features(
    query: str,
    result: dict,
    semantic_score: float = 0.0,
    historical_ctr: float = 0.0,
    content_type_map: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Extract ranking features for a search result.

    Args:
        query: Search query
        result: Search result dict with score breakdown
        semantic_score: Semantic similarity score (0-1)
        historical_ctr: Historical click-through rate for this content
        content_type_map: Mapping from content type to encoded value

    Returns:
        Feature dictionary
    """
    content_type_map = content_type_map or {
        "retningslinje": 0,
        "veileder": 1,
        "informasjon": 2,
    }

    # Parse days since published if available
    days_since_published = 0.0
    if "published_at" in result:
        from datetime import datetime

        try:
            pub_date = datetime.fromisoformat(
                result["published_at"].replace("Z", "+00:00")
            )
            days_since_published = (datetime.now(pub_date.tzinfo) - pub_date).days
        except (ValueError, AttributeError):
            pass

    # Normalize days to 0-1 range (cap at 5 years)
    days_normalized = min(days_since_published / 1825.0, 1.0)

    return {
        "title_keyword_score": result.get("title_keyword_score", 0.0),
        "body_keyword_score": result.get("body_keyword_score", 0.0),
        "tag_match_score": result.get("tag_match_score", 0.0),
        "exact_phrase_title": float(result.get("exact_phrase_title", False)),
        "exact_phrase_body": float(result.get("exact_phrase_body", False)),
        "semantic_similarity": semantic_score,
        "content_type_encoded": content_type_map.get(
            result.get("content_type", "").lower(), 2
        )
        / 2.0,  # Normalize to 0-1
        "days_since_published": days_normalized,
        "historical_ctr": historical_ctr,
        "role_match": float(result.get("role_match", False)),
    }


def prepare_training_data_from_logs(logs: List[dict]) -> tuple:
    """
    Prepare training data from log entries.

    For each search event that has at least one click:
    - Positive example for clicked results (label=1)
    - Negative example for shown but not clicked (label=0)

    Searches without any clicks are ignored to avoid polluting training data
    with potentially irrelevant negative examples (e.g., user refined query).

    Uses search_id to correctly link searches with their clicks.

    Args:
        logs: List of log entries

    Returns:
        Tuple of (features_array, labels_array)
    """
    # Group logs by search sessions
    search_events = [log for log in logs if log.get("event_type") == "search"]
    click_events = [log for log in logs if log.get("event_type") == "click"]

    # Build click lookup by search_id
    # Key: search_id -> Set of clicked content_ids
    clicks_by_search_id = {}
    for click in click_events:
        search_id = click.get("search_id")
        content_id = click.get("content_id", "")
        if search_id and content_id:
            if search_id not in clicks_by_search_id:
                clicks_by_search_id[search_id] = set()
            clicks_by_search_id[search_id].add(content_id)

    features_list = []
    labels_list = []

    for search in search_events:
        search_id = search.get("search_id")
        results_shown = search.get("results_shown", [])

        # Skip searches without search_id (can't link to clicks)
        if not search_id:
            continue

        # Skip searches that have no clicks (avoid polluting with bad negatives)
        if search_id not in clicks_by_search_id:
            continue

        clicked_content_ids = clicks_by_search_id[search_id]

        for result in results_shown:
            content_id = result.get("content_id", "")
            position = result.get("position", 0)
            score = result.get("score", 0.0)

            # Check if this result was clicked
            was_clicked = content_id in clicked_content_ids

            # Create feature dict (simplified for now)
            feature_dict = {
                "title_keyword_score": score * 0.3,  # Approximation
                "body_keyword_score": score * 0.1,
                "tag_match_score": score * 0.2,
                "exact_phrase_title": 0.0,
                "exact_phrase_body": 0.0,
                "semantic_similarity": 0.0,
                "content_type_encoded": 0.5,
                "days_since_published": 0.0,
                "historical_ctr": 0.0,
                "role_match": 0.0,
            }

            features_list.append(list(feature_dict.values()))
            labels_list.append(1.0 if was_clicked else 0.0)

    return np.array(features_list), np.array(labels_list)
