"""
TensorFlow embedding model for Norwegian health content semantic search.

This module provides a custom embedding model that can be trained on
health content to learn domain-specific representations.
"""

from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras


class HealthContentEmbedding:
    """
    Custom TensorFlow embedding model for Norwegian health content.

    Architecture:
    - TextVectorization for tokenization
    - Embedding layer (learned)
    - Bidirectional LSTM encoder
    - Dense projection to embedding space
    - L2 normalization for cosine similarity

    Usage:
        model = HealthContentEmbedding()
        model.adapt(corpus)  # Fit vectorizer on corpus
        embeddings = model.encode(texts)
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        output_dim: int = 256,
        max_sequence_length: int = 512,
    ):
        """
        Initialize the embedding model.

        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of token embeddings
            output_dim: Dimension of output sentence embeddings
            max_sequence_length: Maximum sequence length for input texts
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length

        self.model = self._build_model()
        self._is_adapted = False

    def _build_model(self) -> "keras.Model":
        """Build the embedding model architecture."""
        # Input layer
        text_input = keras.Input(shape=(1,), dtype=tf.string, name="text_input")

        # Text vectorization layer
        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_sequence_length=self.max_sequence_length,
            standardize="lower_and_strip_punctuation",
            name="vectorizer",
        )

        # Apply vectorization
        x = self.vectorizer(text_input)

        # Embedding layer
        x = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="token_embedding",
        )(x)

        # Bidirectional LSTM encoder
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=False, name="lstm"),
            name="bidirectional",
        )(x)

        # Dense projection
        x = keras.layers.Dense(self.output_dim, activation=None, name="projection")(x)

        # L2 normalization for cosine similarity
        output = keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1), name="normalize"
        )(x)

        model = keras.Model(inputs=text_input, outputs=output, name="health_embedding")
        return model

    def adapt(self, corpus: List[str]) -> None:
        """
        Adapt the text vectorizer to a corpus.

        Args:
            corpus: List of texts to build vocabulary from
        """
        # Convert to tensor dataset for vectorizer
        text_ds = tf.data.Dataset.from_tensor_slices(corpus)
        self.vectorizer.adapt(text_ds)
        self._is_adapted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            NumPy array of shape (len(texts), output_dim)
        """
        if not self._is_adapted:
            raise ValueError(
                "Model must be adapted to a corpus first. Call adapt(corpus)."
            )

        # Convert to TensorFlow tensor to avoid numpy dtype issues with long strings
        texts_tensor = tf.constant([[t] for t in texts], dtype=tf.string)
        return self.model.predict(texts_tensor, verbose=0)

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding of shape (1, output_dim)
            doc_embeddings: Document embeddings of shape (n_docs, output_dim)

        Returns:
            Similarity scores of shape (n_docs,)
        """
        # Since embeddings are L2-normalized, dot product equals cosine similarity
        similarities = np.matmul(query_embedding, doc_embeddings.T)
        return similarities.flatten()

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model (without extension)
        """
        import os
        os.environ["PYTHONUTF8"] = "1"

        # Save weights separately (more robust for special characters)
        self.model.save_weights(f"{path}_weights.weights.h5")

        # Save vocabulary separately with UTF-8 encoding
        vocab = self.vectorizer.get_vocabulary()
        vocab_path = f"{path}_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            import json
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "output_dim": self.output_dim,
            "max_sequence_length": self.max_sequence_length,
        }
        config_path = f"{path}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Saved: weights, vocabulary ({len(vocab)} words), config")

    @classmethod
    def load(cls, path: str) -> "HealthContentEmbedding":
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model (without extension)

        Returns:
            Loaded HealthContentEmbedding instance
        """
        import json

        # Load config
        config_path = f"{path}_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create instance with same config
        instance = cls(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            output_dim=config["output_dim"],
            max_sequence_length=config["max_sequence_length"],
        )

        # Load vocabulary
        vocab_path = f"{path}_vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Set vocabulary on vectorizer
        instance.vectorizer.set_vocabulary(vocab)
        instance._is_adapted = True

        # Load weights
        instance.model.load_weights(f"{path}_weights.weights.h5")

        print(f"Loaded: weights, vocabulary ({len(vocab)} words)")
        return instance

    def compile_for_training(
        self,
        learning_rate: float = 1e-4,
        loss: str = "cosine_similarity",
    ) -> None:
        """
        Compile the model for training with contrastive learning.

        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function to use
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
        )


def create_contrastive_pairs(
    texts: List[str],
    labels: List[List[int]],
    pairs_per_doc: int = 3,
    weight_by_overlap: bool = True,
) -> tuple:
    """
    Create contrastive learning pairs from texts with multi-label support.

    Documents sharing ANY tag are positive pairs. Documents sharing MORE tags
    get higher weights (stronger positive signal).

    Args:
        texts: List of texts
        labels: List of label lists (each doc has multiple labels/tags)
        pairs_per_doc: Number of pairs to create per document
        weight_by_overlap: If True, weight pairs by number of shared tags

    Returns:
        Tuple of (anchors, positives, negatives, weights)
        - weights: Float weights based on tag overlap (1.0, 2.0, 3.0, etc.)

    Example:
        Doc A: [1, 2, 3] (diabetes, behandling, barn)
        Doc B: [1, 4, 5] (diabetes, eldre, kosthold)
        Doc C: [1, 2, 6] (diabetes, behandling, voksen)

        A-C share 2 tags → weight = 2.0
        A-B share 1 tag  → weight = 1.0
    """
    import random

    n_docs = len(texts)

    # Build index: tag -> list of doc indices
    tag_to_docs = {}
    for doc_idx, doc_labels in enumerate(labels):
        for tag in doc_labels:
            if tag not in tag_to_docs:
                tag_to_docs[tag] = []
            tag_to_docs[tag].append(doc_idx)

    # Convert labels to sets for faster overlap calculation
    label_sets = [set(doc_labels) for doc_labels in labels]

    anchors = []
    positives = []
    negatives = []
    weights = []

    for anchor_idx in range(n_docs):
        anchor_tags = label_sets[anchor_idx]

        if not anchor_tags:
            continue

        # Find all docs that share at least one tag (potential positives)
        positive_candidates = set()
        for tag in anchor_tags:
            for doc_idx in tag_to_docs.get(tag, []):
                if doc_idx != anchor_idx:
                    positive_candidates.add(doc_idx)

        # Find docs that share NO tags (negatives)
        negative_candidates = [
            idx for idx in range(n_docs)
            if idx != anchor_idx and idx not in positive_candidates
        ]

        if not positive_candidates or not negative_candidates:
            continue

        # Create multiple pairs per document
        positive_list = list(positive_candidates)

        for _ in range(min(pairs_per_doc, len(positive_list))):
            pos_idx = random.choice(positive_list)

            # Calculate weight based on tag overlap (squared for stronger effect)
            if weight_by_overlap:
                overlap = len(anchor_tags & label_sets[pos_idx])
                weight = float(overlap ** 2)  # 1→1, 2→4, 3→9, 4→16, 5→25
            else:
                weight = 1.0

            neg_idx = random.choice(negative_candidates)

            anchors.append(texts[anchor_idx])
            positives.append(texts[pos_idx])
            negatives.append(texts[neg_idx])
            weights.append(weight)

    return anchors, positives, negatives, weights


def triplet_loss(anchor_emb, positive_emb, negative_emb, margin: float = 0.5, weights=None):
    """
    Compute triplet loss with optional sample weights.

    Args:
        anchor_emb: Anchor embeddings (batch_size, embedding_dim)
        positive_emb: Positive embeddings (batch_size, embedding_dim)
        negative_emb: Negative embeddings (batch_size, embedding_dim)
        margin: Margin for triplet loss
        weights: Optional sample weights (batch_size,)

    Returns:
        Scalar loss value
    """
    # Compute similarities (L2-normalized embeddings → dot product = cosine similarity)
    pos_similarity = tf.reduce_sum(anchor_emb * positive_emb, axis=1)
    neg_similarity = tf.reduce_sum(anchor_emb * negative_emb, axis=1)

    # Triplet loss: want pos_similarity > neg_similarity + margin
    loss = tf.maximum(0.0, neg_similarity - pos_similarity + margin)

    if weights is not None:
        loss = loss * weights

    return tf.reduce_mean(loss)
