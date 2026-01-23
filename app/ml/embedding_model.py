"""
Multilingual E5 embedding model for Norwegian health content semantic search.

This module provides a wrapper around the intfloat/multilingual-e5-base model
optimized for health content with structured passage formatting.
"""

import re
from typing import List, Optional, Dict, Any

import numpy as np


class HealthContentEmbedding:
    """
    Multilingual E5 embedding model wrapper for Norwegian health content.

    Uses intfloat/multilingual-e5-base pre-trained transformer model.
    Formats content items as structured passages with metadata fields.

    Architecture:
    - Pre-trained multilingual E5 model (768-dim embeddings)
    - Passage formatting with health-specific fields
    - Query/passage prefixes for optimal retrieval

    Usage:
        model = HealthContentEmbedding()
        # No adaptation needed - pre-trained model
        embeddings = model.encode(texts)
        query_emb = model.encode_query("diabetes behandling")
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier (default: intfloat/multilingual-e5-base)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._is_loaded = False

    def _load_model(self):
        """Load the sentence transformer model."""
        if self._is_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )

        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._is_loaded = True
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    @staticmethod
    def strip_html_tags(text: str) -> str:
        """
        Remove HTML tags from text.

        Args:
            text: Text possibly containing HTML

        Returns:
            Clean text without HTML tags
        """
        if not text:
            return ""
        # Replace HTML tags with space to preserve word boundaries
        clean = re.sub(r'<[^>]+>', ' ', text)
        # Replace multiple whitespace with single space
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    @staticmethod
    def format_passage(content_item: Dict[str, Any]) -> str:
        """
        Format a content item as a structured passage.

        Creates a text representation with all relevant metadata fields:
        - tittel (title)
        - type (content type)
        - icd-10 (diagnose kode)
        - icpc-2 (primærhelsetjeneste kode)
        - snomed-ct (medisinsk terminologi)
        - lis-spesialitet (spesialist område)
        - lis-læringsmål (læringsmål)
        - innhold (clean body text without HTML)

        Args:
            content_item: Dict with content fields

        Returns:
            Formatted passage string
        """
        parts = []

        # Title
        title = content_item.get("title") or content_item.get("tittel")
        if title:
            parts.append(f"Tittel: {title}")

        # Content type
        content_type = content_item.get("content_type") or content_item.get("type")
        if content_type:
            parts.append(f"Type: {content_type}")

        # ICD-10 code
        icd10 = content_item.get("icd10") or content_item.get("icd_10")
        if icd10:
            parts.append(f"ICD-10: {icd10}")

        # ICPC-2 code
        icpc2 = content_item.get("icpc2") or content_item.get("icpc_2")
        if icpc2:
            parts.append(f"ICPC-2: {icpc2}")

        # SNOMED CT
        snomed = content_item.get("snomed_ct") or content_item.get("snomed")
        if snomed:
            parts.append(f"SNOMED-CT: {snomed}")

        # LIS spesialitet
        lis_spec = content_item.get("lis_spesialitet") or content_item.get("lis_specialty")
        if lis_spec:
            parts.append(f"LIS-spesialitet: {lis_spec}")

        # LIS læringsmål
        lis_goal = content_item.get("lis_læringsmål") or content_item.get("lis_learning_goal")
        if lis_goal:
            parts.append(f"LIS-læringsmål: {lis_goal}")

        # Body content (strip HTML)
        body = content_item.get("body") or content_item.get("tekst") or content_item.get("innhold")
        if body:
            clean_body = HealthContentEmbedding.strip_html_tags(body)
            if clean_body:
                parts.append(f"Innhold: {clean_body}")

        return " ".join(parts)

    def encode(
        self, 
        texts: List[str], 
        is_query: bool = False,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors.

        Args:
            texts: List of texts to encode
            is_query: If True, adds "query: " prefix (for search queries)
                     If False, adds "passage: " prefix (for documents)
            show_progress_bar: Show encoding progress

        Returns:
            NumPy array of shape (len(texts), 768)
        """
        if not self._is_loaded:
            self._load_model()

        # Add E5 prefix for better retrieval performance
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + text for text in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True  # L2 normalization
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding of shape (768,)
        """
        return self.encode([query], is_query=True)[0]

    def encode_passages(
        self, 
        content_items: List[Dict[str, Any]],
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Encode content items as passages.

        Formats each content item with structured metadata before encoding.

        Args:
            content_items: List of content item dicts
            show_progress_bar: Show encoding progress

        Returns:
            NumPy array of shape (len(content_items), 768)
        """
        passages = [self.format_passage(item) for item in content_items]
        return self.encode(passages, is_query=False, show_progress_bar=show_progress_bar)

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding of shape (768,) or (1, 768)
            doc_embeddings: Document embeddings of shape (n_docs, 768)

        Returns:
            Similarity scores of shape (n_docs,)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Since embeddings are L2-normalized, dot product equals cosine similarity
        similarities = np.matmul(query_embedding, doc_embeddings.T)
        return similarities.flatten()

    def save(self, path: str) -> None:
        """
        Save model configuration.

        Note: The pre-trained model itself doesn't need to be saved,
        only the configuration for reloading.

        Args:
            path: Path to save the config (without extension)
        """
        import json
        import os

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        config = {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": 768,  # E5-base dimension
        }

        config_path = f"{path}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Saved config to: {config_path}")
        print(f"Model: {self.model_name}")

    @classmethod
    def load(cls, path: str) -> "HealthContentEmbedding":
        """
        Load model from saved configuration.

        Args:
            path: Path to the saved config (without extension)

        Returns:
            Loaded HealthContentEmbedding instance
        """
        import json

        config_path = f"{path}_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create instance with saved config
        instance = cls(
            model_name=config["model_name"],
            device=config.get("device"),
        )

        # Model will be lazy-loaded on first encode()
        print(f"Loaded config: {config['model_name']}")

        return instance

