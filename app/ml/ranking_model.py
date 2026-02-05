"""
Professional Learning-to-Rank reranker for health content search (LambdaMART / XGBoost).

This module replaces the previous TensorFlow binary classifier approach.

Why this version is better:
- Trains a *ranking* model (LambdaMART) grouped by search_id (true LTR), not global click classification.
- Uses IPS (inverse propensity scoring) sample weights to reduce position bias.
- Uses smoothed CTR as a weak prior (avoids one-click wonders).
- Works with your existing DB logging tables:
  - search_logs
  - search_results_shown (impressions + per-result features)
  - click_logs (click + dwell)
  - content_stats (global clicks/impressions)

Runtime usage:
1) Generate candidates (vector search / hybrid), compute per-result features.
2) Call reranker.rerank(...) to sort candidates.
3) Log shown results + clicks (already in your system).

Training usage:
- Call reranker.train_from_database() periodically (offline job / manual).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    # Semantic signal
    "semantic_similarity",          # cosine similarity (-1 to 1, typically 0-1)

    # Keyword signals - absolute magnitude
    "keyword_score_total",          # total keyword score normalized (0-1)

    # Keyword signals - proportions (title-only)
    "exact_title_proportion",       # exact phrase in title / total
    "full_coverage_proportion",     # full title coverage / total
    "title_keyword_proportion",     # title keyword matches / total

    # Metadata / intent alignment
    "type_match",                   # 0/1  (info_type matches query intent)
    "role_match",                   # 0/1  (role matches allowed roles)
    "code_match_count",             # int  (# matched codes: ICD/ICPC/SNOMED/LIS)
    "lis_match",                    # 0/1
    "maalgruppe_match",             # 0/1

    # Popularity prior (windowed CTR - last 30 days)
    "smoothed_ctr",                 # smoothed CTR in [0..1]

    # Bias/context
    "position",                     # shown position (1..N)
]


def _f(x, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def smoothed_ctr(clicks: int, impressions: int, alpha: float = 1.0, beta: float = 20.0) -> float:
    """
    Smoothed CTR prevents "one-click wonders" from dominating.
    (clicks + alpha) / (impressions + alpha + beta)
    """
    clicks = max(0, int(clicks))
    impressions = max(0, int(impressions))
    return float((clicks + alpha) / (impressions + alpha + beta))


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
    position: int

    # Semantic signal
    semantic_similarity: float = 0.0

    # Keyword signals - absolute magnitude
    keyword_score_total: float = 0.0

    # Keyword signals - proportions (title-only)
    exact_title_proportion: float = 0.0
    full_coverage_proportion: float = 0.0
    title_keyword_proportion: float = 0.0

    # Metadata alignment
    type_match: float = 0.0
    role_match: float = 0.0
    code_match_count: float = 0.0
    lis_match: float = 0.0
    maalgruppe_match: float = 0.0

    # Popularity (windowed CTR)
    smoothed_ctr: float = 0.0


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
        ctr_window_days: int = 30,
        min_group_size: int = 5,
        require_any_click: bool = True,
        use_db_propensity: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train a LambdaMART reranker from DB logs.

        Expected database_service methods:
          - get_ltr_training_rows(days_back) -> list[dict]
            Each row should include:
              search_id, content_id, position, clicked,
              semantic_similarity, keyword_score_total,
              exact_title_proportion, full_coverage_proportion, title_keyword_proportion,
              type_match, role_match, code_match_count, lis_match, maalgruppe_match
          - get_content_ctr_windowed(days) -> dict[content_id] = smoothed_ctr
          - (optional) get_position_propensities() -> dict[position] = propensity float

        Args:
            days_back: Days of training data to use
            ctr_window_days: Days for windowed CTR calculation (default: 30)
        """
        rows = database_service.get_ltr_training_rows(days_back=days_back)
        if not rows:
            return {"trained": 0.0, "groups": 0.0, "rows": 0.0}

        # Group by search_id
        groups: Dict[str, List[dict]] = {}
        for r in rows:
            sid = r.get("search_id")
            if sid is None:
                continue
            groups.setdefault(str(sid), []).append(r)

        # Windowed CTR for recent popularity signal
        ctr_windowed = database_service.get_content_ctr_windowed(days=ctr_window_days)

        # Optional propensity table from DB
        pos_prop: Dict[int, float] = {}
        if use_db_propensity:
            try:
                pos_prop = database_service.get_position_propensities()
            except Exception:
                pos_prop = {}

        X_all: List[List[float]] = []
        y_all: List[float] = []
        w_all: List[float] = []
        group_sizes: List[int] = []

        used_groups = 0
        used_rows = 0

        for sid, items in groups.items():
            if len(items) < min_group_size:
                continue

            # Sort by position (stable)
            items_sorted = sorted(items, key=lambda x: int(x.get("position") or 10**9))

            # First pass: build feature dicts and find max keyword_score_total
            feat_dicts: List[Dict[str, float]] = []
            labels: List[int] = []
            weights: List[float] = []

            any_pos = False
            any_click = False

            for rr in items_sorted:
                cid = str(rr.get("content_id", ""))
                pos = int(rr.get("position") or 0)
                if pos <= 0:
                    pos = 10  # fallback
                any_pos = True

                clicked = int(rr.get("clicked") or 0)
                if clicked == 1:
                    any_click = True

                # popularity prior (using windowed CTR for recency)
                ctr = ctr_windowed.get(cid, 0.05)  # Default prior if no data

                # build feature dict from logged row (preferred) + priors
                feat_dict = {
                    "semantic_similarity": _f(rr.get("semantic_similarity"), _f(rr.get("candidate_score"), 0.0)),
                    "keyword_score_total": _f(rr.get("keyword_score_total"), 0.0),  # RAW - normalized below
                    "exact_title_proportion": _f(rr.get("exact_title_proportion"), 0.0),
                    "full_coverage_proportion": _f(rr.get("full_coverage_proportion"), 0.0),
                    "title_keyword_proportion": _f(rr.get("title_keyword_proportion"), 0.0),
                    "type_match": _f(rr.get("type_match"), 0.0),
                    "role_match": _f(rr.get("role_match"), 0.0),
                    "code_match_count": _f(rr.get("code_match_count"), 0.0),
                    "lis_match": _f(rr.get("lis_match"), 0.0),
                    "maalgruppe_match": _f(rr.get("maalgruppe_match"), 0.0),
                    "smoothed_ctr": ctr,
                    "position": float(pos),
                }

                feat_dicts.append(feat_dict)
                labels.append(float(clicked))

                # IPS weight to reduce position bias
                prop = propensity_for_position(pos, pos_prop if pos_prop else None)
                weights.append(1.0 / max(float(prop), 1e-6))

            if not any_pos:
                continue
            if require_any_click and not any_click:
                continue

            # Normalize keyword_score_total by max in this search group
            max_kw = max((fd["keyword_score_total"] for fd in feat_dicts), default=1.0)
            if max_kw > 0:
                for fd in feat_dicts:
                    fd["keyword_score_total"] = fd["keyword_score_total"] / max_kw

            # Second pass: build feature vectors
            feats: List[List[float]] = []
            for fd in feat_dicts:
                feats.append([fd[n] for n in self.feature_names])

            X_all.extend(feats)
            y_all.extend(labels)
            w_all.extend(weights)
            group_sizes.append(len(labels))
            used_groups += 1
            used_rows += len(labels)

        if used_groups == 0:
            return {"trained": 0.0, "groups": 0.0, "rows": 0.0}

        X = np.asarray(X_all, dtype=np.float32)
        y = np.asarray(y_all, dtype=np.float32)
        w = np.asarray(w_all, dtype=np.float32)

        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            learning_rate=0.08,
            max_depth=6,
            n_estimators=350,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )

        self.model.fit(X, y, sample_weight=w, group=group_sizes, verbose=verbose)

        return {"trained": 1.0, "groups": float(used_groups), "rows": float(used_rows)}

    # -------------------------
    # Inference
    # -------------------------

    def rerank(
        self,
        query: str,
        role: str,
        candidates: Sequence[RerankCandidate],
        *,
        ctr_window_days: int = 30,
        ctr_windowed: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[RerankCandidate, float]]:
        """
        Rerank a list of candidates.

        Provide candidates with their precomputed features (semantic/keyword/metadata).
        This method adds popularity prior (windowed CTR) automatically.

        Args:
            query: Search query
            role: User role
            candidates: List of candidates to rerank
            ctr_window_days: Days for windowed CTR (default: 30)
            ctr_windowed: Optional pre-fetched windowed CTR

        Returns: list of (candidate, score), sorted by score descending.
        """
        if not candidates:
            return []

        if self.model is None:
            # Safe fallback if model isn't available yet:
            scored = [(c, float(c.semantic_similarity)) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        # Fetch windowed CTR (recent popularity)
        ctr_data = ctr_windowed or database_service.get_content_ctr_windowed(days=ctr_window_days)

        # Build feature matrix
        X = []
        for c in candidates:
            c.smoothed_ctr = ctr_data.get(c.content_id, 0.05)  # Windowed CTR

            feat_dict = {
                "semantic_similarity": _f(c.semantic_similarity),
                "keyword_score_total": _f(c.keyword_score_total),
                "exact_title_proportion": _f(c.exact_title_proportion),
                "full_coverage_proportion": _f(c.full_coverage_proportion),
                "title_keyword_proportion": _f(c.title_keyword_proportion),
                "type_match": _f(c.type_match),
                "role_match": _f(c.role_match),
                "code_match_count": _f(c.code_match_count),
                "lis_match": _f(c.lis_match),
                "maalgruppe_match": _f(c.maalgruppe_match),
                "smoothed_ctr": _f(c.smoothed_ctr),
                "position": float(int(c.position) if c.position else 0),
            }
            X.append([feat_dict[n] for n in self.feature_names])

        X_np = np.asarray(X, dtype=np.float32)
        scores = self.model.predict(X_np)
        out = list(zip(list(candidates), [float(s) for s in scores]))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

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

        This method provides a simpler interface for ml_service.py,
        accepting feature dicts directly instead of RerankCandidate objects.

        Args:
            features: List of feature dictionaries

        Returns:
            List of ranking scores
        """
        if self.model is None:
            return [0.0] * len(features)

        if not features:
            return []

        # Build feature matrix from dicts
        X = []
        for feat_dict in features:
            row = [_f(feat_dict.get(name, 0.0)) for name in self.feature_names]
            X.append(row)

        X_np = np.asarray(X, dtype=np.float32)
        scores = self.model.predict(X_np)
        return [float(s) for s in scores]


# ---------------------------------------------------------------------
# Compatibility helpers (if old code expects feature dicts)
# ---------------------------------------------------------------------

def extract_features_for_candidate(
    *,
    semantic_similarity: float,
    keyword_score_total: float = 0.0,
    exact_title_proportion: float = 0.0,
    full_coverage_proportion: float = 0.0,
    title_keyword_proportion: float = 0.0,
    type_match: bool = False,
    role_match: bool = False,
    code_match_count: int = 0,
    lis_match: bool = False,
    maalgruppe_match: bool = False,
) -> Dict[str, float]:
    """
    Optional helper if your pipeline builds dict features first.
    (RerankCandidate is preferred.)
    """
    return {
        "semantic_similarity": float(semantic_similarity),
        "keyword_score_total": float(keyword_score_total),
        "exact_title_proportion": float(exact_title_proportion),
        "full_coverage_proportion": float(full_coverage_proportion),
        "title_keyword_proportion": float(title_keyword_proportion),
        "type_match": float(bool(type_match)),
        "role_match": float(bool(role_match)),
        "code_match_count": float(int(code_match_count)),
        "lis_match": float(bool(lis_match)),
        "maalgruppe_match": float(bool(maalgruppe_match)),
    }


# Alias for backward compatibility with ml_service.py
HealthContentRanker = HealthContentReranker
