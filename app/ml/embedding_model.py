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

        # Ensure texts are in the right format
        texts_array = np.array(texts).reshape(-1, 1)
        return self.model.predict(texts_array, verbose=0)

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
        self.model.save(f"{path}.keras")

    @classmethod
    def load(cls, path: str) -> "HealthContentEmbedding":
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model (without extension)

        Returns:
            Loaded HealthContentEmbedding instance
        """
        instance = cls.__new__(cls)
        instance.model = keras.models.load_model(f"{path}.keras")

        # Extract layers from loaded model
        instance.vectorizer = instance.model.get_layer("vectorizer")

        # Get config from model
        instance._is_adapted = True

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
    texts: List[str], labels: Optional[List[int]] = None
) -> tuple:
    """
    Create contrastive learning pairs from texts.

    If labels are provided, texts with the same label are positive pairs,
    otherwise augmented versions of the same text are positive pairs.

    Args:
        texts: List of texts
        labels: Optional list of labels for supervised contrastive learning

    Returns:
        Tuple of (anchor_texts, positive_texts, negative_texts)
    """
    anchors = []
    positives = []
    negatives = []

    if labels is not None:
        # Supervised contrastive learning
        label_to_texts = {}
        for text, label in zip(texts, labels):
            if label not in label_to_texts:
                label_to_texts[label] = []
            label_to_texts[label].append(text)

        for text, label in zip(texts, labels):
            same_label = [t for t in label_to_texts[label] if t != text]
            diff_label = [
                t for l, ts in label_to_texts.items() if l != label for t in ts
            ]

            if same_label and diff_label:
                anchors.append(text)
                positives.append(np.random.choice(same_label))
                negatives.append(np.random.choice(diff_label))
    else:
        # Self-supervised: use text augmentation (simple version)
        for i, text in enumerate(texts):
            # Use the same text as positive (in practice, would use augmentation)
            other_texts = [t for j, t in enumerate(texts) if j != i]
            if other_texts:
                anchors.append(text)
                positives.append(text)  # Would be augmented in practice
                negatives.append(np.random.choice(other_texts))

    return anchors, positives, negatives
