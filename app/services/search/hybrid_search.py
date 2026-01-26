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
from app.services.search.keyword_search import keyword_search
from app.services.search.semantic_search import semantic_search
from app.config import settings
from app.constants import is_allowed_info_type


class HybridSearch:
    """Hybrid search combining keyword and semantic scores."""

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining keyword and semantic scores.

        Uses min-max normalization to scale both scores to [0, 1].
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        # Check if semantic search is available
        if not semantic_search.is_available():
            return keyword_search.search(query, role, k)

        query_embedding = semantic_search.get_query_embedding(query)
        if query_embedding is None:
            return keyword_search.search(query, role, k)

        # Score all content (raw scores)
        scored_items = []

        for item in content_service.get_all_content():
            if not is_allowed_info_type(item.content_type):
                continue

            if role and role not in item.target_groups:
                continue

            # Keyword score (raw)
            kw_score = keyword_search.calculate_score(item, query_lower, query_keywords)

            # Semantic score (raw, -1 to 1)
            if item.id in semantic_search.content_embeddings:
                doc_embedding = semantic_search.content_embeddings[item.id]
                sem_score = float(np.dot(query_embedding, doc_embedding))
            else:
                sem_score = -1.0

            if kw_score > 0 or sem_score > -1.0:
                scored_items.append((item, kw_score, sem_score))

        if not scored_items:
            return []

        # Normalize weights to ensure they sum to 1 and stay in [0,1]
        total_weight = keyword_weight + semantic_weight
        if total_weight > 0:
            norm_keyword_weight = max(0.0, min(1.0, keyword_weight / total_weight))
            norm_semantic_weight = max(0.0, min(1.0, semantic_weight / total_weight))
        else:
            norm_keyword_weight = 0.3
            norm_semantic_weight = 0.7

        # Min-max normalization for keyword scores
        keyword_scores = [kw for _, kw, _ in scored_items]
        kw_min, kw_max = min(keyword_scores), max(keyword_scores)

        # Normalize and combine
        normalized_items = []
        for item, kw_raw, sem_raw in scored_items:
            if kw_max > kw_min:
                kw_norm = (kw_raw - kw_min) / (kw_max - kw_min)
            else:
                kw_norm = 1.0 if kw_raw > 0 else 0.0

            # Semantic score: normalize from [-1,1] to [0,1]
            sem_norm = max(0.0, min(1.0, (sem_raw + 1.0) / 2.0))

            # Combined score (using normalized weights)
            combined_score = norm_keyword_weight * kw_norm + norm_semantic_weight * sem_norm

            normalized_items.append((item, combined_score, kw_raw, sem_raw, kw_norm, sem_norm))

        # Sort by combined score
        normalized_items.sort(key=lambda x: -x[1])

        # Apply ranking model if enabled
        if settings.ml_ranking_enabled:
            normalized_items = self._apply_ranking_model(
                normalized_items, query, query_keywords, role
            )

        # Create results
        results = []
        for item, combined, kw_raw, sem_raw, kw_norm, sem_norm in normalized_items[:k]:
            parts = []
            if kw_norm > 0:
                parts.append(f"Keyword: {kw_norm:.2f}")
            if sem_norm > 0:
                parts.append(f"Semantic: {sem_norm:.2f}")
            explanation = " + ".join(parts) if parts else "No match"
            explanation += f" = {combined:.2f}"

            results.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    info_type=item.content_type,
                    score=round(combined, 3),
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
