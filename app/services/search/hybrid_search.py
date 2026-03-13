"""
Hybrid search combining keyword and semantic search.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np

from app.dto.response.search import SearchResult
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.services.search.bm25_search import bm25_search
from app.services.search.keyword_search import keyword_search, _normalize_query_keywords
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
    rrf_raw: float = 0.0  # Original RRF score, preserved when ML ranking overwrites combined_score
    role_boost: float = 1.0  # Role boost/penalty multiplier applied (1.0 = neutral)
    pre_rerank_position: int = 0  # Position before ML reranking (1-indexed, 0 = not set)
    rerank_score: Optional[float] = None  # Raw ML model score (None = model not applied)
    rerank_contributions: Optional[Dict[str, float]] = None  # Per-feature SHAP contributions


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
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        temaside_boost: Optional[float] = None,
        retningslinje_boost: Optional[float] = None,
        role_boost: Optional[float] = None,
        role_penalty: Optional[float] = None,
        rerank: Optional[bool] = None,
        explain: bool = False,
    ) -> List[SearchResult]:
        """Perform hybrid search combining BM25 and semantic retrieval via RRF."""
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        try:
            candidates = self._score_with_rrf(
                query, role, query_lower, query_keywords, k,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                rrf_k=rrf_k,
                temaside_boost=temaside_boost,
                retningslinje_boost=retningslinje_boost,
                role_boost=role_boost,
                role_penalty=role_penalty,
            )
        except Exception:
            logger.exception("Error in RRF fusion")
            candidates = []

        if not candidates:
            return []

        # rerank=None means use global setting, True/False overrides per-request
        use_rerank = rerank if rerank is not None else settings.ml_ranking_enabled
        if use_rerank:
            # Only rerank top candidates — items ranked low by RRF are unlikely to surface
            rerank_cutoff = min(max(k * 10, 100), len(candidates))
            candidates = self._apply_ranking_model(
                candidates[:rerank_cutoff], role, query_lower, query_keywords,
                explain=explain,
            )

        # Trim to reasonable size before applying boosts and normalization
        if not use_rerank:
            candidates = candidates[:k]

        # Apply content type boosts and role boost/penalty AFTER ML reranking
        # (ML replaces combined_score, so multiplicative boosts must come after)
        type_boosts = {
            "temaside": temaside_boost if temaside_boost is not None else settings.search_boost_temaside,
            "retningslinje": (
                retningslinje_boost
                if retningslinje_boost is not None
                else settings.search_boost_retningslinje
            ),
        }
        for c in candidates:
            boost = type_boosts.get(c.item.content_type.lower(), 1.0)
            if boost != 1.0:
                c.combined_score *= boost
            c.combined_score *= c.role_boost
        candidates.sort(key=lambda c: -c.combined_score)

        candidates = self._normalize_combined_scores(candidates)

        return self._build_results(candidates, k)

    def _score_with_rrf(
        self,
        query: str,
        role: Optional[str],
        query_lower: str,
        query_keywords: set,
        k: int,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        temaside_boost: Optional[float] = None,
        retningslinje_boost: Optional[float] = None,
        role_boost: Optional[float] = None,
        role_penalty: Optional[float] = None,
    ) -> List[HybridCandidate]:
        """Retrieve with BM25 + dense and fuse with RRF."""
        t_start = time.perf_counter()

        candidate_pool = min(
            max(k * self.candidate_multiplier, self.min_candidate_pool),
            self.max_candidate_pool,
        )

        semantic_available = semantic_search.is_available()

        t0 = time.perf_counter()
        if semantic_available:
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(bm25_search.search, query, role, candidate_pool)
                sem_future = executor.submit(semantic_search.search, query=query, role=role, k=candidate_pool)

                try:
                    bm25_hits = bm25_future.result()
                except Exception:
                    logger.exception("BM25 search failed in parallel execution")
                    bm25_hits = []

                try:
                    semantic_hits = sem_future.result()
                except Exception:
                    logger.exception("Semantic search failed in parallel execution")
                    semantic_hits = []

            t_bm25 = time.perf_counter() - t0
            t_semantic = t_bm25  # parallel, so same wall time
        else:
            bm25_hits = bm25_search.search(query, role, k=candidate_pool)
            t_bm25 = time.perf_counter() - t0
            semantic_hits = []
            t_semantic = 0.0

        if not bm25_hits and not semantic_hits:
            return []

        t0 = time.perf_counter()
        ranked_lists: Dict[str, List[str]] = {}
        if bm25_hits:
            ranked_lists["bm25"] = [hit.item.id for hit in bm25_hits]
        if semantic_hits:
            ranked_lists["semantic"] = [result.id for result in semantic_hits]

        rrf_weights = {
            "bm25": bm25_weight if bm25_weight is not None else settings.search_rrf_weight_bm25,
            "semantic": semantic_weight if semantic_weight is not None else settings.search_rrf_weight_semantic,
        }
        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k
        fused = fuse_ranked_lists(ranked_lists, rrf_k=effective_rrf_k, weights=rrf_weights)
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

        # Pre-normalize query keywords once (stemming + synonym expansion)
        normalized_keywords = _normalize_query_keywords(query_keywords)

        t0 = time.perf_counter()
        candidates: List[HybridCandidate] = []
        for fused_result in fused[:candidate_pool]:
            content_id = fused_result.content_id
            item = bm25_item_by_id.get(content_id) or content_service.get_content_by_id(content_id)
            if not item:
                continue

            if item.content_type not in content_service.searchable_types:
                continue

            # Keep old keyword feature semantics for reranker compatibility.
            keyword_raw = keyword_search.calculate_score(item, query_lower, normalized_keywords, _pre_normalized=True)

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
                rrf_raw=fused_result.score,
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

        # Store boosts on each candidate for application after ML reranking in search().
        # If ML is disabled, these are still applied in search() before final sort.
        if role:
            effective_role_boost = role_boost if role_boost is not None else settings.search_role_match_boost
            effective_role_penalty = role_penalty if role_penalty is not None else settings.search_role_mismatch_penalty
            for c in candidates:
                if role in c.item.role_tags:
                    c.role_boost = effective_role_boost
                elif c.item.role_tags:  # has tags but no match
                    c.role_boost = effective_role_penalty
                # No tags = neutral (1.0, default)

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
        from app.dto.response.search import PipelineScores, RerankInfo

        results = []
        for position, c in enumerate(candidates[:k], start=1):
            # Build rerank info if model was applied
            rerank_info = None
            if c.rerank_score is not None and c.pre_rerank_position > 0:
                rerank_info = RerankInfo(
                    score=round(c.rerank_score, 4),
                    rank_change=c.pre_rerank_position - position,
                    contributions=c.rerank_contributions or {},
                )

            pipeline = PipelineScores(
                bm25=round(c.keyword_norm, 4),
                semantic=round(c.semantic_norm, 4),
                rrf=round(c.rrf_raw, 6),
                role_boost=round(c.role_boost, 4),
                rerank=rerank_info,
            )

            results.append(
                SearchResult(
                    id=c.item.id,
                    title=c.item.title,
                    info_type=c.item.content_type,
                    path=c.item.path,
                    has_text_content=c.item.has_text_content,
                    document_url=c.item.public_document_url,
                    is_pdf_only=c.item.is_pdf_only,
                    score=round(c.combined_score, 3),
                    pipeline=pipeline,
                )
            )
        return results

    def _apply_ranking_model(
        self,
        candidates: List[HybridCandidate],
        role: Optional[str],
        query_lower: str = "",
        query_keywords: Optional[set] = None,
        explain: bool = False,
    ) -> List[HybridCandidate]:
        """Apply ranking model to re-rank results."""
        t_start = time.perf_counter()
        try:
            from app.services.search.ml_service import ml_service

            if not ml_service.is_ranking_available():
                ml_service.load_ranking_model(force=True)
                if not ml_service.is_ranking_available():
                    return candidates

            # Save pre-rerank positions (1-indexed)
            for i, c in enumerate(candidates):
                c.pre_rerank_position = i + 1

            # Fetch cached CTR map for popularity signal
            t0 = time.perf_counter()
            ctr_map = ml_service.get_ctr_map()
            t_ctr = time.perf_counter() - t0

            # Extract 7 features for each candidate
            t0 = time.perf_counter()
            features_list = []
            for c in candidates:
                features = self._extract_ranking_features(c, role, ctr_map, query_lower, query_keywords)
                features_list.append(features)
            t_features = time.perf_counter() - t0

            t0 = time.perf_counter()
            if explain:
                # Slow path: include per-feature SHAP contributions
                results_with_contribs = ml_service.get_ranking_scores_with_contributions(features_list)
                for i, c in enumerate(candidates):
                    score, contribs = results_with_contribs[i]
                    c.rerank_score = score
                    c.rerank_contributions = contribs
                    c.combined_score = score
            else:
                # Fast path: scores only
                scores = ml_service.get_ranking_scores(features_list)
                for i, c in enumerate(candidates):
                    c.rerank_score = scores[i]
                    c.combined_score = scores[i]
            t_predict = time.perf_counter() - t0

            candidates.sort(key=lambda c: -c.combined_score)

            t_total = time.perf_counter() - t_start
            logger.info(
                "Reranking timings: CTR=%.0fms  Features=%.0fms  Predict=%.0fms  Total=%.0fms",
                t_ctr * 1000, t_features * 1000, t_predict * 1000, t_total * 1000,
            )
            return candidates

        except Exception:
            logger.exception("Error applying ranking model")
            return candidates

    @staticmethod
    def _extract_ranking_features(
        candidate: HybridCandidate,
        role: Optional[str],
        ctr_map: Optional[Dict[str, float]] = None,
        query_lower: str = "",
        query_keywords: Optional[set] = None,
    ) -> Dict[str, float]:
        """Extract 7 features for ranking model from a HybridCandidate."""
        item = candidate.item

        # Role match
        role_match = 0.0
        if role and item.role_tags:
            if role in item.role_tags:
                role_match = 1.0 / len(item.role_tags)
        elif not role and not item.role_tags:
            role_match = 0.5
        elif not item.role_tags:
            role_match = 0.3

        # Smoothed CTR (Bayesian prior for unseen content)
        default_ctr = 1.0 / 21.0
        smoothed_ctr = (ctr_map or {}).get(item.id, default_ctr)

        # Query length
        q_terms = query_keywords or set()
        query_length = float(len(q_terms))

        # Title-query Jaccard overlap
        title_terms = set(re.findall(r'\w+', item.title.lower())) if item.title else set()
        if q_terms or title_terms:
            title_query_overlap = len(q_terms & title_terms) / max(len(q_terms | title_terms), 1)
        else:
            title_query_overlap = 0.0

        # Content freshness
        content_freshness = 0.5
        if item.sist_faglig_oppdatert:
            try:
                dt = item.sist_faglig_oppdatert
                if not isinstance(dt, datetime):
                    dt = datetime.fromisoformat(str(dt))
                days = max(0, (datetime.now() - dt).days)
                content_freshness = 1.0 / (1.0 + days / 365.0)
            except Exception:
                pass

        return {
            "semantic_score": candidate.semantic_norm,
            "bm25_score": candidate.keyword_norm,
            "smoothed_ctr": smoothed_ctr,
            "role_match": role_match,
            "query_length": query_length,
            "title_query_overlap": title_query_overlap,
            "content_freshness": content_freshness,
        }


# Global instance
hybrid_search = HybridSearch()
