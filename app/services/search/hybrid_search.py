"""
Hybrid search combining keyword and semantic search.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from app.dto.response.search import SearchResult
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.services.data.database_service import database_service
from app.services.search.bm25_search import bm25_search
from app.services.search.keyword_search import keyword_search
from app.services.search.rrf_fusion import fuse_ranked_lists
from app.services.search.semantic_search import semantic_search
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridCandidate:
    """Intermediate search candidate carrying scores from each retrieval stage."""

    item: ContentItem
    combined_score: float
    keyword_raw: float
    semantic_raw: float
    keyword_norm: float
    semantic_norm: float


class HybridSearch:
    """Hybrid search combining BM25 and semantic retrieval with RRF fusion."""

    def __init__(
        self,
        candidate_multiplier: int = 3,
        min_candidate_pool: int = 100,
        max_candidate_pool: int = 1000,
    ):
        self.rrf_k = max(1, settings.search_rrf_k)
        self.candidate_multiplier = max(2, int(candidate_multiplier))
        self.min_candidate_pool = max(20, int(min_candidate_pool))
        self.max_candidate_pool = max(100, int(max_candidate_pool))

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
    ) -> List[SearchResult]:
        """Perform hybrid search combining BM25 and semantic retrieval via RRF."""
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        try:
            candidates = self._score_with_rrf(query, role, query_lower, query_keywords, k)
        except Exception:
            logger.exception("Error in RRF fusion")
            candidates = []

        if not candidates:
            return []

        if settings.ml_ranking_enabled:
            candidates = self._apply_ranking_model(candidates, role)

        candidates = self._normalize_combined_scores(candidates)

        return self._build_results(candidates, k)

    def _score_with_rrf(
        self,
        query: str,
        role: Optional[str],
        query_lower: str,
        query_keywords: set,
        k: int,
    ) -> List[HybridCandidate]:
        """Retrieve with BM25 + dense and fuse with RRF."""
        t_start = time.perf_counter()

        candidate_pool = min(
            max(k * self.candidate_multiplier, self.min_candidate_pool),
            self.max_candidate_pool,
        )

        t0 = time.perf_counter()
        bm25_hits = bm25_search.search(query, role, k=candidate_pool)
        t_bm25 = time.perf_counter() - t0

        semantic_hits = []
        semantic_available = semantic_search.is_available()
        t0 = time.perf_counter()
        if semantic_available:
            semantic_hits = semantic_search.search(query=query, role=role, k=candidate_pool)
        t_semantic = time.perf_counter() - t0

        if not bm25_hits and not semantic_hits:
            return []

        t0 = time.perf_counter()
        ranked_lists: Dict[str, List[str]] = {}
        if bm25_hits:
            ranked_lists["bm25"] = [hit.item.id for hit in bm25_hits]
        if semantic_hits:
            ranked_lists["semantic"] = [result.id for result in semantic_hits]

        rrf_weights = {
            "bm25": settings.search_rrf_weight_bm25,
            "semantic": settings.search_rrf_weight_semantic,
        }
        fused = fuse_ranked_lists(ranked_lists, rrf_k=self.rrf_k, weights=rrf_weights)
        t_rrf = time.perf_counter() - t0
        if not fused:
            return []

        bm25_item_by_id = {hit.item.id: hit.item for hit in bm25_hits}
        bm25_score_by_id = {hit.item.id: hit.score for hit in bm25_hits}

        # Semantic search returns scores in [0,1]. Keep as-is for display (sem_norm).
        # For LTR features we need [-1,1] range (sem_raw).
        semantic_norm_by_id = {result.id: float(result.score) for result in semantic_hits}

        query_embedding = None
        if semantic_available:
            query_embedding = semantic_search.get_query_embedding(query)

        t0 = time.perf_counter()
        candidates: List[HybridCandidate] = []
        for fused_result in fused[:candidate_pool]:
            content_id = fused_result.content_id
            item = bm25_item_by_id.get(content_id) or content_service.get_content_by_id(content_id)
            if not item:
                continue

            if item.content_type not in content_service.searchable_types:
                continue
            if role and role not in item.target_groups:
                continue

            # Keep old keyword feature semantics for reranker compatibility.
            keyword_raw = keyword_search.calculate_score(item, query_lower, query_keywords)

            sem_norm = semantic_norm_by_id.get(content_id, 0.0)
            if sem_norm <= 0.0 and query_embedding is not None and content_id in semantic_search.content_embeddings:
                dot = float(np.dot(query_embedding, semantic_search.content_embeddings[content_id]))
                sem_norm = max(0.0, min(1.0, (dot + 1.0) / 2.0))

            candidates.append(HybridCandidate(
                item=item,
                combined_score=fused_result.score,
                keyword_raw=keyword_raw,
                semantic_raw=(sem_norm * 2.0) - 1.0,
                keyword_norm=0.0,
                semantic_norm=sem_norm,
            ))
        t_candidates = time.perf_counter() - t0

        if not candidates:
            return []

        # Normalize BM25 scores for display.
        bm25_values = [bm25_score_by_id.get(c.item.id, 0.0) for c in candidates]
        bm25_min, bm25_max = min(bm25_values), max(bm25_values)

        for c in candidates:
            bm25_raw = bm25_score_by_id.get(c.item.id, 0.0)
            if bm25_max > bm25_min:
                c.keyword_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min)
            else:
                c.keyword_norm = 1.0 if bm25_raw > 0 else 0.0

        # Apply content type boosts
        type_boosts = {
            "temaside": settings.search_boost_temaside,
            "retningslinje": settings.search_boost_retningslinje,
        }
        for c in candidates:
            boost = type_boosts.get(c.item.content_type.lower(), 1.0)
            if boost != 1.0:
                c.combined_score *= boost

        candidates.sort(key=lambda c: -c.combined_score)

        t_total = time.perf_counter() - t_start
        logger.info(
            "Hybrid search timings: BM25=%.0fms  Semantic=%.0fms  RRF=%.0fms  "
            "Candidates(%d)=%.0fms  Total=%.0fms",
            t_bm25 * 1000, t_semantic * 1000, t_rrf * 1000,
            len(candidates), t_candidates * 1000, t_total * 1000,
        )

        return candidates

    @staticmethod
    def _normalize_combined_scores(candidates: List[HybridCandidate]) -> List[HybridCandidate]:
        """Normalize combined score field to [0,1]."""
        if not candidates:
            return []

        scores = [c.combined_score for c in candidates]
        s_min, s_max = min(scores), max(scores)

        for c in candidates:
            if s_max > s_min:
                c.combined_score = (c.combined_score - s_min) / (s_max - s_min)
            else:
                c.combined_score = 1.0 if c.combined_score > 0 else 0.0

        return candidates

    @staticmethod
    def _build_results(candidates: List[HybridCandidate], k: int) -> List[SearchResult]:
        """Build API search results from candidates."""
        results = []
        for c in candidates[:k]:
            explanation = (
                f"BM25={c.keyword_norm:.2f} | Semantic={c.semantic_norm:.2f} | "
                f"RRF final={c.combined_score:.2f}"
            )

            results.append(
                SearchResult(
                    id=c.item.id,
                    title=c.item.title,
                    info_type=c.item.content_type,
                    score=round(c.combined_score, 3),
                    explanation=explanation,
                    bm25_score=c.keyword_norm,
                    semantic_score=c.semantic_norm,
                    rrf_score=c.combined_score,
                )
            )
        return results

    def _apply_ranking_model(
        self,
        candidates: List[HybridCandidate],
        role: Optional[str],
    ) -> List[HybridCandidate]:
        """Apply ranking model to re-rank results."""
        try:
            from app.services.search.ml_service import ml_service

            if not ml_service.is_ranking_available():
                ml_service.load_ranking_model()
                if not ml_service.is_ranking_available():
                    return candidates

            # Use windowed CTR (30 days) to match training data
            ctr_data = database_service.get_content_ctr_windowed(days=30)

            # Extract features for each candidate
            features_list = []
            for c in candidates:
                features = self._extract_ranking_features(
                    c, role, ctr_data.get(c.item.id, 0.0)
                )
                features_list.append(features)

            # Get ranking scores from model
            ranking_scores = ml_service.get_ranking_scores(features_list)

            # Replace combined score with ranking score and re-sort
            for i, c in enumerate(candidates):
                c.combined_score = ranking_scores[i]

            candidates.sort(key=lambda c: -c.combined_score)
            return candidates

        except Exception:
            logger.exception("Error applying ranking model")
            return candidates

    def _extract_ranking_features(
        self,
        candidate: HybridCandidate,
        role: Optional[str],
        ctr: float,
    ) -> Dict[str, float]:
        """Extract features for ranking model from a HybridCandidate."""
        item = candidate.item

        # Content type encoding
        content_type_map = {
            "retningslinje": 0.9,
            "veileder": 0.8,
            "fagprosedyre": 0.75,
            "faktaark": 0.6,
            "artikkel": 0.5,
        }
        type_match = content_type_map.get(
            item.info_type.lower() if item.info_type else None,
            0.5
        )

        # Role match
        role_match = 0.0
        if role and item.target_groups:
            if role in item.target_groups:
                role_match = 1.0 / len(item.target_groups)
        elif not role and not item.target_groups:
            role_match = 0.5
        elif not item.target_groups:
            role_match = 0.3

        # Maalgruppe match
        target_groups = item.target_groups or []
        maalgruppe_match = 1.0 if role and role in target_groups else 0.0

        return {
            "semantic_score": candidate.semantic_norm,
            "bm25_score": candidate.keyword_norm,
            "rrf_score": candidate.combined_score,
            "type_match": type_match,
            "role_match": role_match,
            "maalgruppe_match": maalgruppe_match,
            "smoothed_ctr": ctr,
            "position": 0.0,
        }


# Global instance
hybrid_search = HybridSearch()
