"""BM25 lexical retrieval over cached content."""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.services.search.bm25_hierarchy import BM25HierarchyConfig, BM25HierarchyIndex

logger = logging.getLogger(__name__)


@dataclass
class BM25Hit:
    """Single BM25 retrieval hit."""

    item: ContentItem
    score: float
    rank: int
    direct_score: float = 0.0
    inherited_score: float = 0.0


class BM25Search:
    """
    BM25 retriever using in-memory index over `content_service` data.

    Title tokens are up-weighted by repeating title text during indexing.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        title_weight: Optional[int] = None,
    ):
        self.k1 = k1
        self.b = b
        effective_title_weight = settings.search_bm25_title_weight if title_weight is None else title_weight
        self.title_weight = max(1, int(effective_title_weight))

        hierarchy_config = BM25HierarchyConfig(
            enabled=bool(settings.search_bm25_hierarchy_enabled),
            max_depth=max(1, int(settings.search_bm25_hierarchy_max_depth)),
            decay=max(0.0, min(1.0, float(settings.search_bm25_hierarchy_decay))),
            source_top_k=max(1, int(settings.search_bm25_hierarchy_source_top_k)),
            top_children=max(1, int(settings.search_bm25_hierarchy_top_children)),
            tail_weight=max(0.0, float(settings.search_bm25_hierarchy_tail_weight)),
            weight=max(0.0, float(settings.search_bm25_hierarchy_weight)),
            min_contribution=max(0.0, float(settings.search_bm25_hierarchy_min_contribution)),
        )
        self._hierarchy = BM25HierarchyIndex(hierarchy_config)
        self._index_signature: Optional[FrozenSet[Tuple[str, str]]] = None
        self._items: List[ContentItem] = []
        self._doc_lengths: np.ndarray = np.array([], dtype=np.float32)
        self._avg_doc_length: float = 1.0
        self._idf: Dict[str, float] = {}
        self._postings: Dict[str, List[Tuple[int, int]]] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _needs_rebuild(self) -> bool:
        current = content_service.get_all_content()
        return BM25HierarchyIndex.compute_signature(current) != self._index_signature

    def _build_index(self) -> None:
        items = content_service.get_all_content()
        self._items = list(items)
        self._index_signature = BM25HierarchyIndex.compute_signature(self._items)

        n_docs = len(self._items)
        if n_docs == 0:
            self._doc_lengths = np.array([], dtype=np.float32)
            self._avg_doc_length = 1.0
            self._idf = {}
            self._postings = {}
            return

        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        doc_lengths = np.zeros(n_docs, dtype=np.float32)
        doc_freq: Dict[str, int] = defaultdict(int)

        for idx, item in enumerate(self._items):
            title = item.title or ""
            body = item.body or ""

            # Up-weight title matches in lexical retrieval.
            weighted_text = (" ".join([title] * self.title_weight) + " " + body).strip()
            terms = self._tokenize(weighted_text)
            doc_lengths[idx] = float(len(terms))

            if not terms:
                continue

            term_counts = Counter(terms)
            for term, tf in term_counts.items():
                postings[term].append((idx, int(tf)))
                doc_freq[term] += 1

        avg_dl = float(doc_lengths.mean()) if len(doc_lengths) else 1.0
        if avg_dl <= 0.0:
            avg_dl = 1.0

        idf = {}
        for term, df in doc_freq.items():
            # Robertson-Sparck Jones style IDF.
            idf[term] = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

        self._doc_lengths = doc_lengths
        self._avg_doc_length = avg_dl
        self._idf = idf
        self._postings = dict(postings)
        self._hierarchy.build(self._items)

    def _ensure_index(self) -> None:
        if self._needs_rebuild():
            self._build_index()

    def prebuild(self) -> int:
        """Build the BM25 index eagerly. Returns number of indexed items."""
        self._ensure_index()
        return len(self._items)

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 100,
    ) -> List[BM25Hit]:
        """
        Retrieve top-k content items ranked by BM25 score.
        """
        self._ensure_index()

        if not self._items:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        query_tf = Counter(query_terms)
        direct_scores = np.zeros(len(self._items), dtype=np.float32)

        for term, q_weight in query_tf.items():
            postings = self._postings.get(term)
            if not postings:
                continue

            term_idf = self._idf.get(term, 0.0)
            if term_idf <= 0.0:
                continue

            for doc_idx, tf in postings:
                dl = float(self._doc_lengths[doc_idx]) if doc_idx < len(self._doc_lengths) else 0.0
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avg_doc_length))
                if denom <= 0.0:
                    continue
                direct_scores[doc_idx] += (
                    float(q_weight) * term_idf * ((tf * (self.k1 + 1.0)) / denom)
                )

        if not np.any(direct_scores > 0):
            return []

        inherited_scores = self._hierarchy.propagate(direct_scores)

        final_scores = direct_scores + inherited_scores

        hits: List[Tuple[ContentItem, float]] = []
        hit_details: Dict[str, Tuple[float, float]] = {}
        for idx, score in enumerate(final_scores):
            if score <= 0.0:
                continue

            item = self._items[idx]
            if item.content_type not in content_service.searchable_types:
                continue

            hits.append((item, float(score)))
            hit_details[item.id] = (
                float(direct_scores[idx]),
                float(inherited_scores[idx]),
            )

        if not hits:
            return []

        hits.sort(key=lambda x: (-x[1], x[0].id))
        top_hits = hits[: max(1, int(k))]
        out = []
        for i, (item, score) in enumerate(top_hits, start=1):
            direct, inherited = hit_details.get(item.id, (0.0, 0.0))
            out.append(
                BM25Hit(
                    item=item,
                    score=score,
                    rank=i,
                    direct_score=direct,
                    inherited_score=inherited,
                )
            )

        if logger.isEnabledFor(logging.INFO):
            inherited_only = sum(
                1 for hit in out if hit.direct_score <= 0.0 and hit.inherited_score > 0.0
            )
            logger.info(
                "BM25 hits=%d inherited_only=%d hierarchy=%s",
                len(out),
                inherited_only,
                self._hierarchy.config.enabled,
            )
        return out


# Global instance
bm25_search = BM25Search()
