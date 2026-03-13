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

        print(f"Loading embedding model: {self.model_name} (device={self.device})...")
        # Use ONNX backend for CPU (faster inference), default backend for CUDA
        if self.device and self.device.startswith("cuda"):
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = SentenceTransformer(self.model_name, backend="onnx", device=self.device)
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
        Format a content item as a natural language passage for embedding.

        Uses natural language format optimized for embedding models:
        - No special separators or structured formatting
        - Important information (title, content) comes first
        - Metadata woven in naturally
        - Related content appended as natural text

        Args:
            content_item: Dict with content fields. Can include 'linked_content'
                         which is a list of dicts with 'tittel' and 'tekst' from
                         fetched linked documents.

        Returns:
            Natural language passage string
        """
        sentences = []

        # Title - the most important part, comes first
        title = content_item.get("title") or content_item.get("tittel") or ""

        # Content type
        content_type = content_item.get("content_type") or content_item.get("info_type") or content_item.get("type") or ""

        # Build opening sentence with title and type
        is_temaside = (content_type or "").lower() == "temaside"
        if title:
            if is_temaside:
                # Repeat title in a natural sentence to boost the title signal.
                # Temasider often have empty body text, so without this the title
                # is a tiny fraction of the passage and gets drowned out.
                sentences.append(f"{title}. Dette er en temaside om {title.lower()}.")
            elif content_type:
                sentences.append(f"{title}. Dette er en {content_type}.")
            else:
                sentences.append(f"{title}.")

        # Role tags - who is this for?
        role_tags = content_item.get("role_tags")
        if role_tags:
            if isinstance(role_tags, str):
                try:
                    import json
                    role_tags = json.loads(role_tags)
                except (json.JSONDecodeError, TypeError):
                    role_tags = []
            if isinstance(role_tags, list) and role_tags:
                sentences.append(f"Målgruppe: {', '.join(str(t) for t in role_tags)}.")

        # Medical codes - include naturally
        koder = content_item.get("koder")
        if koder:
            if isinstance(koder, str):
                try:
                    import json
                    koder = json.loads(koder)
                except (json.JSONDecodeError, TypeError):
                    koder = {}

            if isinstance(koder, dict):
                kode_parts = []
                for key in ["ICD10", "icd10", "ICPC2", "icpc2", "SNOMED", "snomed", "LIS"]:
                    val = koder.get(key)
                    if val:
                        if isinstance(val, list):
                            kode_parts.extend(val)
                        else:
                            kode_parts.append(str(val))
                if kode_parts:
                    sentences.append(f"Relevante koder: {', '.join(kode_parts)}.")

        # Main body content - the core of the passage
        body = content_item.get("body") or content_item.get("tekst") or content_item.get("innhold")
        if body:
            clean_body = HealthContentEmbedding.strip_html_tags(body)
            if clean_body:
                sentences.append(clean_body)

        # Linked/related content - append as additional context.
        # E5-base has 512 token limit (~2000 chars Norwegian), so we truncate
        # linked text and cap total passage length to stay within the window.
        linked_content = content_item.get("linked_content", [])

        if linked_content:
            current_length = sum(len(s) for s in sentences)
            max_passage_length = 1800  # Leave room for E5 prefix + tokenizer overhead

            if is_temaside:
                # Temasider: use only child titles (no body text) to keep the
                # passage focused on the temaside's own identity. Including
                # body snippets from children dilutes the title signal and
                # causes poor semantic match when searching for the temaside name.
                child_titles = []
                for linked in linked_content:
                    if current_length >= max_passage_length:
                        break
                    linked_title = linked.get("tittel", "")
                    if linked_title:
                        child_titles.append(linked_title)
                        current_length += len(linked_title) + 2  # +2 for ", "

                if child_titles:
                    sentences.append("Undertema: " + ", ".join(child_titles) + ".")
            else:
                related_parts = []
                for linked in linked_content:
                    if current_length >= max_passage_length:
                        break
                    linked_title = linked.get("tittel", "")
                    linked_text = linked.get("tekst", "")
                    linked_type = linked.get("type", "")
                    if linked_title:
                        clean_text = HealthContentEmbedding.strip_html_tags(linked_text) if linked_text else ""
                        type_part = f" ({linked_type})" if linked_type else ""
                        if clean_text:
                            clean_text = clean_text[:200].rstrip('.').rstrip()
                            part = f"{linked_title}{type_part}: {clean_text}"
                        else:
                            part = f"{linked_title}{type_part}"
                        related_parts.append(part)
                        current_length += len(part)

                if related_parts:
                    sentences.append("Se også: " + "; ".join(related_parts) + ".")

        return " ".join(sentences)

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

