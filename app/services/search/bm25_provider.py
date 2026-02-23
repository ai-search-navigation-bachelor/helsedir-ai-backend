"""
BM25 provider for in-process lexical retrieval.

This project intentionally uses the local BM25 backend only.
"""

from typing import List, Optional

from app.services.search.bm25_search import BM25Hit, bm25_search


class BM25Provider:
    """Thin wrapper around local BM25 retrieval."""

    def __init__(self):
        self.local = bm25_search

    def search(self, query: str, role: Optional[str] = None, k: int = 100) -> List[BM25Hit]:
        """Execute BM25 retrieval using the local in-process index."""
        return self.local.search(query=query, role=role, k=k)


# Global instance
bm25_provider = BM25Provider()
