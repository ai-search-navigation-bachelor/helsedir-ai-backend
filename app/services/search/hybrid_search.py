"""
Hybrid search combining keyword and semantic search.
"""

import re
from typing import List, Optional, Dict, Any

import numpy as np

from app.dto.response.search import SearchResult
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.services.data.database_service import database_service
from app.services.search.bm25_provider import bm25_provider
from app.services.search.keyword_search import keyword_search
from app.services.search.rrf_fusion import fuse_ranked_lists
from app.services.search.semantic_search import semantic_search
from app.config import settings


class HybridSearch:
    """Hybrid search combining BM25 and semantic retrieval with RRF fusion."""

    def __init__(
        self,
        rrf_k: int = 60,
        candidate_multiplier: int = 8,
        min_candidate_pool: int = 100,
    ):
        self.rrf_k = max(1, int(rrf_k))
        self.candidate_multiplier = max(2, int(candidate_multiplier))
        self.min_candidate_pool = max(20, int(min_candidate_pool))

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and semantic retrieval via RRF.

        `keyword_weight` and `semantic_weight` are kept for backward compatibility
        with existing call sites, but are no longer used.
        """
        _ = keyword_weight
        _ = semantic_weight

        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        try:
            normalized_items = self._score_with_rrf(query, role, query_lower, query_keywords, k)
        except Exception as e:
            print(f"Error in RRF fusion: {e}")
            normalized_items = []

        if not normalized_items:
            return []

        # Apply ranking model if enabled
        if settings.ml_ranking_enabled:
            normalized_items = self._apply_ranking_model(
                normalized_items, query, query_keywords, role
            )

        # RRF scores are small absolute values, so normalize to [0,1] for API response.
        normalized_items = self._normalize_combined_scores(normalized_items)

        return self._build_results(normalized_items, k, use_rrf=True)

    def _score_with_rrf(
        self,
        query: str,
        role: Optional[str],
        query_lower: str,
        query_keywords: set,
        k: int,
    ) -> List[tuple]:
        """Retrieve with BM25 + dense and fuse with RRF."""
        candidate_pool = max(k * self.candidate_multiplier, self.min_candidate_pool)

        bm25_hits = bm25_provider.search(query, role, k=candidate_pool)

        semantic_hits = []
        semantic_available = semantic_search.is_available()
        if semantic_available:
            semantic_hits = semantic_search.search(query=query, role=role, k=candidate_pool)

        if not bm25_hits and not semantic_hits:
            return []

        ranked_lists: Dict[str, List[str]] = {}
        if bm25_hits:
            ranked_lists["bm25"] = [hit.item.id for hit in bm25_hits]
        if semantic_hits:
            ranked_lists["semantic"] = [result.id for result in semantic_hits]

        fused = fuse_ranked_lists(ranked_lists, rrf_k=self.rrf_k)
        if not fused:
            return []

        bm25_item_by_id = {hit.item.id: hit.item for hit in bm25_hits}
        bm25_score_by_id = {hit.item.id: hit.score for hit in bm25_hits}

        semantic_raw_by_id = {
            result.id: max(-1.0, min(1.0, (float(result.score) * 2.0) - 1.0))
            for result in semantic_hits
        }

        query_embedding = None
        if semantic_available:
            query_embedding = semantic_search.get_query_embedding(query)

        fused_items: List[tuple] = []
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
            keyword_feature_raw = keyword_search.calculate_score(item, query_lower, query_keywords)

            sem_raw = semantic_raw_by_id.get(content_id, -1.0)
            if sem_raw <= -1.0 and query_embedding is not None and content_id in semantic_search.content_embeddings:
                sem_raw = float(np.dot(query_embedding, semantic_search.content_embeddings[content_id]))

            sem_norm = max(0.0, min(1.0, (sem_raw + 1.0) / 2.0))
            fused_items.append(
                (item, fused_result.score, keyword_feature_raw, sem_raw, 0.0, sem_norm)
            )

        if not fused_items:
            return []

        # Use BM25 strength for displayed lexical component (kw_norm).
        bm25_values = [bm25_score_by_id.get(item.id, 0.0) for item, *_ in fused_items]
        bm25_min, bm25_max = min(bm25_values), max(bm25_values)

        normalized_items = []
        for item, fused_score, kw_raw, sem_raw, _, sem_norm in fused_items:
            bm25_raw = bm25_score_by_id.get(item.id, 0.0)
            if bm25_max > bm25_min:
                kw_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min)
            else:
                kw_norm = 1.0 if bm25_raw > 0 else 0.0

            normalized_items.append((item, fused_score, kw_raw, sem_raw, kw_norm, sem_norm))

        normalized_items.sort(key=lambda x: -x[1])
        return normalized_items

    @staticmethod
    def _normalize_combined_scores(items: List[tuple]) -> List[tuple]:
        """Normalize combined score field (index 1) to [0,1]."""
        if not items:
            return []

        scores = [float(x[1]) for x in items]
        s_min, s_max = min(scores), max(scores)

        normalized = []
        for item, combined, kw_raw, sem_raw, kw_norm, sem_norm in items:
            if s_max > s_min:
                score_norm = (float(combined) - s_min) / (s_max - s_min)
            else:
                score_norm = 1.0 if float(combined) > 0 else 0.0
            normalized.append((item, score_norm, kw_raw, sem_raw, kw_norm, sem_norm))
        return normalized

    @staticmethod
    def _build_results(items: List[tuple], k: int, use_rrf: bool = False) -> List[SearchResult]:
        """Build API search results from normalized tuples."""
        results = []
        for item, combined, kw_raw, sem_raw, kw_norm, sem_norm in items[:k]:
            bm25_text = f"{kw_norm:.2f}" if kw_norm > 0 else "0.00"
            semantic_text = f"{sem_norm:.2f}" if sem_norm > 0 else "0.00"
            fusion_label = "RRF" if use_rrf else "weighted-sum"
            explanation = (
                f"BM25={bm25_text} | Semantic={semantic_text} | "
                f"{fusion_label} final={combined:.2f}"
            )

            results.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    info_type=item.content_type,
                    score=round(float(combined), 3),
                    explanation=explanation,
                )
            )
        return results

    def _apply_ranking_model(
        self,
        items: List[tuple],
        query: str,
        query_keywords: set,
        role: Optional[str]
    ) -> List[tuple]:
        """Apply ranking model to re-rank results."""
        try:
            from app.services.search.ml_service import ml_service

            if not ml_service.is_ranking_available():
                ml_service.load_ranking_model()
                if not ml_service.is_ranking_available():
                    return items

            # Use windowed CTR (30 days) to match training data
            ctr_data = database_service.get_content_ctr_windowed(days=30)

            # Extract RAW features for each item
            features_list = []
            for item, combined, kw_raw, sem_raw, kw_norm, sem_norm in items:
                features = self._extract_ranking_features(
                    item, query, query_keywords, role,
                    kw_raw, sem_raw, ctr_data.get(item.id, 0.0)
                )
                features_list.append(features)

            # Normalize keyword_score_total
            max_kw_score = max(
                (f["keyword_score_total"] for f in features_list),
                default=1.0
            )
            if max_kw_score > 0:
                for features in features_list:
                    features["keyword_score_total"] = features["keyword_score_total"] / max_kw_score

            # Get ranking scores from model
            ranking_scores = ml_service.get_ranking_scores(features_list)

            # Replace combined score with ranking score and re-sort
            re_ranked = []
            for i, (item, _, kw_raw, sem_raw, kw_norm, sem_norm) in enumerate(items):
                re_ranked.append((item, ranking_scores[i], kw_raw, sem_raw, kw_norm, sem_norm))

            re_ranked.sort(key=lambda x: -x[1])
            return re_ranked

        except Exception as e:
            print(f"Error applying ranking model: {e}")
            return items

    def _extract_ranking_features(
        self,
        item: ContentItem,
        query: str,
        query_keywords: set,
        role: Optional[str],
        keyword_score: float,
        semantic_score: float,
        ctr: float
    ) -> Dict[str, float]:
        """Extract features for ranking model."""
        query_lower = query.lower()
        title_lower = item.title.lower()

        # Calculate individual keyword score components for proportions
        exact_title_score = 0.0
        full_coverage_score = 0.0
        title_keyword_score = 0.0

        if query_lower in title_lower:
            exact_title_score = settings.search_exact_phrase_title_weight

        title_keywords = set(re.findall(r'\w+', title_lower))
        if title_keywords and title_keywords.issubset(query_keywords):
            full_coverage_score = settings.search_full_title_coverage_weight

        title_matches = query_keywords & title_keywords
        if title_matches:
            title_keyword_score = len(title_matches) * settings.search_keyword_title_weight

        total_keyword_score = exact_title_score + full_coverage_score + title_keyword_score

        # Calculate proportions
        if total_keyword_score > 0:
            exact_title_prop = exact_title_score / total_keyword_score
            full_coverage_prop = full_coverage_score / total_keyword_score
            title_keyword_prop = title_keyword_score / total_keyword_score
        else:
            exact_title_prop = 0.0
            full_coverage_prop = 0.0
            title_keyword_prop = 0.0

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
            "semantic_similarity": semantic_score,
            "keyword_score_total": keyword_score,
            "exact_title_proportion": exact_title_prop,
            "full_coverage_proportion": full_coverage_prop,
            "title_keyword_proportion": title_keyword_prop,
            "type_match": type_match,
            "role_match": role_match,
            "code_match_count": 0.0,  # TODO: implement code matching
            "lis_match": 0.0,  # TODO: implement LIS matching
            "maalgruppe_match": maalgruppe_match,
            "smoothed_ctr": ctr,
            "position": 0.0,  # Position unknown during re-ranking
        }


# Global instance
hybrid_search = HybridSearch()
