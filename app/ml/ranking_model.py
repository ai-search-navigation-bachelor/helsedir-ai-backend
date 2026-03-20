"""
Professional Learning-to-Rank reranker for health content search (LambdaMART / XGBoost).

This module replaces the previous TensorFlow binary classifier approach.

Why this version is better:
- Trains a *ranking* model (LambdaMART) grouped by search_id (true LTR), not global click classification.
- Uses IPS (inverse propensity scoring) via sample_weight to reduce position bias.
- Works with your existing DB logging tables:
  - search_logs
  - search_results_shown (impressions + per-result features)
  - click_logs (click events)

Runtime usage:
1) Generate candidates (BM25 + semantic + RRF fusion), scores carried on SearchResult.
2) Call reranker.rerank(...) to sort candidates.
3) Log shown results + clicks (already in your system).

Training usage:
- Call reranker.train_from_database() periodically (offline job / manual).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "Missing dependency: xgboost. Install with: pip install xgboost"
    ) from e

from app.services.data.database_service import database_service


# ---------------------------------------------------------------------
# Feature schema (MUST be consistent between training and inference)
# ---------------------------------------------------------------------

RERANK_FEATURES: List[str] = [
    # Retrieval scores
    "semantic_score",               # normalized semantic similarity (0-1)
    "bm25_score",                   # normalized BM25 score (0-1)

    # Popularity signal
    "smoothed_ctr",                 # Bayesian-smoothed CTR last 30 days (0-1)

    # Metadata / intent alignment
    "role_match",                   # role match score (0-1)

    # Query features
    "query_length",                 # number of terms in query
    "title_query_overlap",          # Jaccard overlap between query and title terms

    # Freshness
    "content_freshness",            # 1.0 / (1.0 + days_since_update / 365.0)
]


def _f(x, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_freshness(sist_faglig_oppdatert) -> float:
    """Compute freshness score from sist_faglig_oppdatert date.

    Returns 1.0 / (1.0 + days_since_update / 365.0), or 0.5 if date is missing.
    """
    if sist_faglig_oppdatert is None:
        return 0.5
    try:
        if isinstance(sist_faglig_oppdatert, datetime):
            dt = sist_faglig_oppdatert
        else:
            dt = datetime.fromisoformat(str(sist_faglig_oppdatert))
        # Strip timezone info for safe subtraction with naive datetime.now()
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        days = max(0, (datetime.now() - dt).days)
        return 1.0 / (1.0 + days / 365.0)
    except (ValueError, TypeError):
        return 0.5


def propensity_for_position(pos: int, db_propensities: Optional[Dict[int, float]] = None) -> float:
    """
    Get propensity (P(click | position)) used for IPS weighting.

    Uses DB propensities if available, with fallback to last known value
    for positions beyond what's stored. Falls back to hardcoded values
    if no DB propensities exist.

    Args:
        pos: Position (1-indexed)
        db_propensities: Optional dict of position -> propensity from DB

    Returns:
        Propensity value for the position
    """
    # If we have DB propensities, use them with fallback for high positions
    if db_propensities:
        if pos in db_propensities:
            return db_propensities[pos]
        # Use last known propensity for positions beyond max
        max_pos = max(db_propensities.keys())
        return db_propensities[max_pos]

    # Hardcoded fallback if no DB propensities
    if pos <= 1:
        return 1.00
    if pos == 2:
        return 0.70
    if pos == 3:
        return 0.55
    if pos == 4:
        return 0.45
    if pos == 5:
        return 0.40
    if pos == 6:
        return 0.35
    if pos == 7:
        return 0.30
    if pos == 8:
        return 0.28
    if pos == 9:
        return 0.26
    if pos == 10:
        return 0.24
    return 0.20


# ---------------------------------------------------------------------
# Candidate dataclass used at inference time
# ---------------------------------------------------------------------

@dataclass
class RerankCandidate:
    content_id: str

    # Retrieval scores
    semantic_score: float = 0.0
    bm25_score: float = 0.0

    # Popularity signal
    smoothed_ctr: float = 0.0

    # Metadata alignment
    role_match: float = 0.0

    # Query features
    query_length: float = 0.0
    title_query_overlap: float = 0.0

    # Freshness
    content_freshness: float = 0.0


# ---------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------

class HealthContentReranker:
    """
    Professional reranker using LambdaMART (XGBoost XGBRanker).

    - train_from_database(): builds groupwise training data from logs.
    - rerank(): scores candidates and returns sorted list.
    - save()/load(): persists model.

    Notes:
    - This model is trained on clicks as weak labels.
    - Make sure search_results_shown logs per-result features.
    """

    def __init__(self) -> None:
        self.model: Optional[xgb.XGBRanker] = None
        self.feature_names: List[str] = list(RERANK_FEATURES)

    # -------------------------
    # Training
    # -------------------------

    def train_from_database(
        self,
        *,
        days_back: int = 180,
        min_group_size: int = 5,
        require_any_click: bool = True,
        use_db_propensity: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train a LambdaMART reranker from DB logs.

        Uses 7 features (retrieval scores + role match + query features + freshness).
        Position is used only for IPS sample weights (not as a feature).

        Expected database_service methods:
          - get_ltr_training_rows(days_back) -> list[dict]
          - (optional) get_position_propensities() -> dict[position] = propensity float

        Args:
            days_back: Days of training data to use
            min_group_size: Minimum results per search group
            require_any_click: Only use groups with at least one click
            use_db_propensity: Use position propensities from DB for IPS
            verbose: Show training progress
        """
        rows = database_service.get_ltr_training_rows(days_back=days_back)
        if not rows:
            return {"trained": 0.0, "groups": 0.0, "rows": 0.0}

        # Fetch smoothed CTR (last 30 days) as an orthogonal popularity signal
        ctr_map = database_service.get_content_ctr_map(days_back=30)
        default_ctr = 1.0 / 21.0  # prior: 0 clicks, 0 impressions

        # Group by search_id
        groups: Dict[str, List[dict]] = {}
        for r in rows:
            sid = r.get("search_id")
            if sid is None:
                continue
            groups.setdefault(str(sid), []).append(r)

        # Optional propensity table from DB for IPS weighting
        pos_prop: Dict[int, float] = {}
        if use_db_propensity:
            try:
                pos_prop = database_service.get_position_propensities()
            except Exception:
                pos_prop = {}

        X_all: List[List[float]] = []
        y_all: List[float] = []
        w_all: List[float] = []
        qid_all: List[int] = []

        used_groups = 0
        used_rows = 0

        for sid, items in groups.items():
            if len(items) < min_group_size:
                continue

            # Sort by position (stable)
            items_sorted = sorted(items, key=lambda x: int(x.get("position") or 10**9))

            # Get query for this group (same for all items in group)
            query_text = str(items_sorted[0].get("query", "")).lower()
            query_terms = set(re.findall(r'\w+', query_text))
            query_len = float(len(query_terms))

            # First pass: build feature dicts
            feat_dicts: List[Dict[str, float]] = []
            labels: List[float] = []
            ips_sample_weights: List[float] = []

            any_pos = False
            any_click = False

            for rr in items_sorted:
                pos = int(rr.get("position") or 0)
                if pos <= 0:
                    pos = 10  # fallback
                any_pos = True

                clicked = int(rr.get("clicked") or 0)
                if clicked == 1:
                    any_click = True

                content_id = rr.get("content_id", "")

                # Title-query Jaccard overlap
                title_text = str(rr.get("title", "")).lower()
                title_terms = set(re.findall(r'\w+', title_text))
                if query_terms or title_terms:
                    overlap = len(query_terms & title_terms) / max(len(query_terms | title_terms), 1)
                else:
                    overlap = 0.0

                # Content freshness
                freshness = _compute_freshness(rr.get("sist_faglig_oppdatert"))

                feat_dict = {
                    "semantic_score": _f(rr.get("semantic_score"), 0.0),
                    "bm25_score": _f(rr.get("bm25_score"), 0.0),
                    "smoothed_ctr": ctr_map.get(content_id, default_ctr),
                    "role_match": _f(rr.get("role_match"), 0.0),
                    "query_length": query_len,
                    "title_query_overlap": overlap,
                    "content_freshness": freshness,
                }

                feat_dicts.append(feat_dict)

                # Binary label + IPS sample weight for position bias correction
                labels.append(float(clicked))
                prop = propensity_for_position(pos, pos_prop if pos_prop else None)
                ips_sample_weights.append(1.0 / max(float(prop), 1e-6))

            if not any_pos:
                continue
            if require_any_click and not any_click:
                continue

            # Second pass: build feature vectors
            feats: List[List[float]] = []
            for fd in feat_dicts:
                feats.append([fd[n] for n in self.feature_names])

            X_all.extend(feats)
            y_all.extend(labels)
            w_all.extend(ips_sample_weights)
            qid_all.extend([used_groups] * len(labels))
            used_groups += 1
            used_rows += len(labels)

        if used_groups == 0:
            return {"trained": 0.0, "groups": 0.0, "rows": 0.0}

        X = np.asarray(X_all, dtype=np.float32)
        y = np.asarray(y_all, dtype=np.float32)
        w = np.asarray(w_all, dtype=np.float32)
        qid = np.asarray(qid_all, dtype=np.int32)

        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            learning_rate=0.08,
            max_depth=6,
            n_estimators=350,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )

        self.model.fit(X, y, qid=qid, sample_weight=w, verbose=verbose)
        self.model.get_booster().feature_names = self.feature_names

        return {"trained": 1.0, "groups": float(used_groups), "rows": float(used_rows)}

    def train_from_groups(
        self,
        *,
        groups: Dict[str, List[dict]],
        ctr_map: Optional[Dict[str, float]] = None,
        min_group_size: int = 5,
        require_any_click: bool = True,
        use_db_propensity: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train from pre-split groups (avoids data leakage in evaluation).

        Same logic as train_from_database but accepts groups directly.
        """
        if ctr_map is None:
            ctr_map = {}
        default_ctr = 1.0 / 21.0

        pos_prop: Dict[int, float] = {}
        if use_db_propensity:
            try:
                pos_prop = database_service.get_position_propensities()
            except Exception:
                pos_prop = {}

        X_all: List[List[float]] = []
        y_all: List[float] = []
        w_all: List[float] = []
        qid_all: List[int] = []
        used_groups = 0
        used_rows = 0

        for sid, items in groups.items():
            if len(items) < min_group_size:
                continue

            items_sorted = sorted(items, key=lambda x: int(x.get("position") or 10**9))
            query_text = str(items_sorted[0].get("query", "")).lower()
            query_terms = set(re.findall(r'\w+', query_text))
            query_len = float(len(query_terms))

            feat_dicts: List[Dict[str, float]] = []
            labels: List[float] = []
            ips_sample_weights: List[float] = []
            any_pos = False
            any_click = False

            for rr in items_sorted:
                pos = int(rr.get("position") or 0)
                if pos <= 0:
                    pos = 10
                any_pos = True

                clicked = int(rr.get("clicked") or 0)
                if clicked == 1:
                    any_click = True

                content_id = rr.get("content_id", "")
                title_text = str(rr.get("title", "")).lower()
                title_terms = set(re.findall(r'\w+', title_text))
                if query_terms or title_terms:
                    overlap = len(query_terms & title_terms) / max(len(query_terms | title_terms), 1)
                else:
                    overlap = 0.0

                freshness = _compute_freshness(rr.get("sist_faglig_oppdatert"))

                feat_dicts.append({
                    "semantic_score": _f(rr.get("semantic_score"), 0.0),
                    "bm25_score": _f(rr.get("bm25_score"), 0.0),
                    "smoothed_ctr": ctr_map.get(content_id, default_ctr),
                    "role_match": _f(rr.get("role_match"), 0.0),
                    "query_length": query_len,
                    "title_query_overlap": overlap,
                    "content_freshness": freshness,
                })

                # Binary label + IPS sample weight for position bias correction
                labels.append(float(clicked))
                prop = propensity_for_position(pos, pos_prop if pos_prop else None)
                ips_sample_weights.append(1.0 / max(float(prop), 1e-6))

            if not any_pos:
                continue
            if require_any_click and not any_click:
                continue

            feats = [[fd[n] for n in self.feature_names] for fd in feat_dicts]
            X_all.extend(feats)
            y_all.extend(labels)
            w_all.extend(ips_sample_weights)
            qid_all.extend([used_groups] * len(labels))
            used_groups += 1
            used_rows += len(labels)

        if used_groups == 0:
            return {"trained": 0.0, "groups": 0.0, "rows": 0.0}

        X = np.asarray(X_all, dtype=np.float32)
        y = np.asarray(y_all, dtype=np.float32)
        w = np.asarray(w_all, dtype=np.float32)
        qid = np.asarray(qid_all, dtype=np.int32)

        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            learning_rate=0.08,
            max_depth=6,
            n_estimators=350,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )
        self.model.fit(X, y, qid=qid, sample_weight=w, verbose=verbose)
        self.model.get_booster().feature_names = self.feature_names

        return {"trained": 1.0, "groups": float(used_groups), "rows": float(used_rows)}

    # -------------------------
    # Inference
    # -------------------------

    def rerank(
        self,
        query: str,
        role: str,
        candidates: Sequence[RerankCandidate],
    ) -> List[Tuple[RerankCandidate, float]]:
        """
        Rerank a list of candidates using 7 model features.

        Args:
            query: Search query
            role: User role
            candidates: List of candidates to rerank

        Returns: list of (candidate, score), sorted by score descending.
        """
        if not candidates:
            return []

        if self.model is None:
            # Safe fallback: blend semantic and BM25 scores
            _FALLBACK_ALPHA = 0.6
            scored = [
                (c, _FALLBACK_ALPHA * float(c.semantic_score)
                     + (1 - _FALLBACK_ALPHA) * float(c.bm25_score))
                for c in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        # Build feature matrix (7 features)
        X = []
        for c in candidates:
            feat_dict = {
                "semantic_score": _f(c.semantic_score),
                "bm25_score": _f(c.bm25_score),
                "smoothed_ctr": _f(c.smoothed_ctr),
                "role_match": _f(c.role_match),
                "query_length": _f(c.query_length),
                "title_query_overlap": _f(c.title_query_overlap),
                "content_freshness": _f(c.content_freshness),
            }
            X.append([feat_dict[n] for n in self.feature_names])

        X_np = np.asarray(X, dtype=np.float32)
        scores = list(self.model.predict(X_np).astype(float))

        out = list(zip(list(candidates), scores))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # -------------------------
    # Evaluation
    # -------------------------

    def evaluate(
        self,
        groups: Dict[str, List[dict]],
        ctr_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality on held-out data.

        Args:
            groups: Dict of search_id -> list of result dicts (with features + clicked).
            ctr_map: Optional CTR map for smoothed_ctr feature.

        Returns:
            Dict with NDCG@5, NDCG@10, MRR, Precision@1 metrics.
        """
        from sklearn.metrics import ndcg_score

        default_ctr = 1.0 / 21.0
        if ctr_map is None:
            ctr_map = {}

        ndcg5_scores = []
        ndcg10_scores = []
        mrr_scores = []
        p1_scores = []

        for sid, items in groups.items():
            if len(items) < 2:
                continue

            items_sorted = sorted(items, key=lambda x: int(x.get("position") or 10**9))

            query_text = str(items_sorted[0].get("query", "")).lower()
            query_terms = set(re.findall(r'\w+', query_text))
            query_len = float(len(query_terms))

            features_list = []
            relevance = []

            for rr in items_sorted:
                content_id = rr.get("content_id", "")
                title_text = str(rr.get("title", "")).lower()
                title_terms = set(re.findall(r'\w+', title_text))
                if query_terms or title_terms:
                    overlap = len(query_terms & title_terms) / max(len(query_terms | title_terms), 1)
                else:
                    overlap = 0.0

                freshness = _compute_freshness(rr.get("sist_faglig_oppdatert"))

                feat_dict = {
                    "semantic_score": _f(rr.get("semantic_score"), 0.0),
                    "bm25_score": _f(rr.get("bm25_score"), 0.0),
                    "smoothed_ctr": ctr_map.get(content_id, default_ctr),
                    "role_match": _f(rr.get("role_match"), 0.0),
                    "query_length": query_len,
                    "title_query_overlap": overlap,
                    "content_freshness": freshness,
                }
                features_list.append(feat_dict)
                relevance.append(float(rr.get("clicked", 0)))

            if sum(relevance) == 0:
                continue

            scores = self.predict(features_list)

            # Sort by predicted score descending
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked_rel = [relevance[i] for i in ranked]

            # NDCG@5, NDCG@10
            true_rel = [relevance]
            pred_scores = [scores]
            try:
                ndcg5_scores.append(ndcg_score(true_rel, pred_scores, k=5))
            except Exception:
                pass
            try:
                ndcg10_scores.append(ndcg_score(true_rel, pred_scores, k=10))
            except Exception:
                pass

            # MRR: reciprocal rank of first clicked result
            for rank, rel in enumerate(ranked_rel, start=1):
                if rel > 0:
                    mrr_scores.append(1.0 / rank)
                    break

            # Precision@1
            p1_scores.append(1.0 if ranked_rel[0] > 0 else 0.0)

        return {
            "ndcg@5": float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0,
            "ndcg@10": float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
            "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "precision@1": float(np.mean(p1_scores)) if p1_scores else 0.0,
            "num_queries": len(ndcg5_scores),
        }

    # -------------------------
    # Persistence
    # -------------------------

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No reranker model loaded/trained. Cannot save.")
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "HealthContentReranker":
        """Load model from disk and return a new instance."""
        instance = cls()
        m = xgb.XGBRanker()
        m.load_model(path)
        instance.model = m
        return instance

    def predict(self, features: List[Dict[str, float]]) -> List[float]:
        """
        Predict ranking scores from feature dictionaries.

        Args:
            features: List of feature dictionaries

        Returns:
            List of ranking scores
        """
        if self.model is None:
            return [0.0] * len(features)

        if not features:
            return []

        X_np = self._build_feature_matrix(features)
        scores = self.model.predict(X_np)
        return [float(s) for s in scores]

    def predict_with_contributions(
        self, features: List[Dict[str, float]]
    ) -> List[Tuple[float, Dict[str, float]]]:
        """
        Predict ranking scores with per-feature contribution explanations.

        Uses XGBoost's built-in SHAP-based feature contributions to explain
        how each feature influenced the final score.

        Args:
            features: List of feature dictionaries

        Returns:
            List of (score, contributions) where contributions maps
            feature name -> contribution value (positive = helped, negative = hurt)
        """
        if self.model is None:
            empty = {name: 0.0 for name in self.feature_names}
            return [(0.0, dict(empty)) for _ in features]

        if not features:
            return []

        X_np = self._build_feature_matrix(features)
        scores = self.model.predict(X_np)

        # Get SHAP-style feature contributions from XGBoost
        # Returns shape (n_samples, n_features + 1) where last column is bias
        booster = self.model.get_booster()
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X_np, feature_names=self.feature_names)
        contribs = booster.predict(dmatrix, pred_contribs=True)

        results = []
        for i in range(len(features)):
            contrib_dict = {}
            for j, name in enumerate(self.feature_names):
                contrib_dict[name] = round(float(contribs[i][j]), 4)
            contrib_dict["bias"] = round(float(contribs[i][-1]), 4)
            results.append((float(scores[i]), contrib_dict))

        return results

    def _build_feature_matrix(self, features: List[Dict[str, float]]) -> np.ndarray:
        """Build numpy feature matrix from list of feature dicts."""
        X = []
        for feat_dict in features:
            row = [_f(feat_dict.get(name, 0.0)) for name in self.feature_names]
            X.append(row)
        return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------
# Compatibility helpers (if old code expects feature dicts)
# ---------------------------------------------------------------------

def extract_features_for_candidate(
    *,
    semantic_score: float = 0.0,
    bm25_score: float = 0.0,
    smoothed_ctr: float = 0.0,
    role_match: float = 0.0,
    query_length: float = 0.0,
    title_query_overlap: float = 0.0,
    content_freshness: float = 0.0,
) -> Dict[str, float]:
    """
    Optional helper if your pipeline builds dict features first.
    (RerankCandidate is preferred.)
    """
    return {
        "semantic_score": float(semantic_score),
        "bm25_score": float(bm25_score),
        "smoothed_ctr": float(smoothed_ctr),
        "role_match": float(role_match),
        "query_length": float(query_length),
        "title_query_overlap": float(title_query_overlap),
        "content_freshness": float(content_freshness),
    }


# Alias for backward compatibility with ml_service.py
HealthContentRanker = HealthContentReranker