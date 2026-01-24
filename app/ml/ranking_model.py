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

    # Keyword signals - proportions (where did the score come from?)
    "exact_title_proportion",       # exact phrase in title / total
    "full_coverage_proportion",     # full title coverage / total
    "title_keyword_proportion",     # title keyword matches / total
    "body_keyword_proportion",      # body keyword matches / total
    "exact_body_proportion",        # exact phrase in body / total

    # Metadata / intent alignment
    "type_match",                   # 0/1  (info_type matches query intent)
    "role_match",                   # 0/1  (role matches allowed roles)
    "code_match_count",             # int  (# matched codes: ICD/ICPC/SNOMED/LIS)
    "lis_match",                    # 0/1
    "maalgruppe_match",             # 0/1

    # Popularity priors (weak)
    "smoothed_ctr",                 # smoothed CTR in [0..1]
    "log_impressions",              # log(1 + impressions)

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


def log1p_int(x: int) -> float:
    return float(np.log1p(max(0, int(x))))


def propensity_for_position(pos: int) -> float:
    """
    Approximate propensity (P(click | position)) used for IPS weighting.
    Replace with DB-driven propensities if you created `position_propensity`.
    """
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

    # Keyword signals - proportions
    exact_title_proportion: float = 0.0
    full_coverage_proportion: float = 0.0
    title_keyword_proportion: float = 0.0
    body_keyword_proportion: float = 0.0
    exact_body_proportion: float = 0.0

    # Metadata alignment
    type_match: float = 0.0
    role_match: float = 0.0
    code_match_count: float = 0.0
    lis_match: float = 0.0
    maalgruppe_match: float = 0.0

    # Popularity (filled from content_stats)
    smoothed_ctr: float = 0.0
    log_impressions: float = 0.0


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

        Expected database_service methods:
          - get_ltr_training_rows(days_back) -> list[dict]
            Each row should include:
              search_id, content_id, position, clicked,
              semantic_similarity, keyword_score_total,
              exact_title_proportion, full_coverage_proportion,
              title_keyword_proportion, body_keyword_proportion, exact_body_proportion,
              type_match, role_match, code_match_count, lis_match, maalgruppe_match
          - get_content_stats_bulk() -> dict[content_id] = {"clicks": int, "impressions": int}
          - (optional) get_position_propensities() -> dict[position] = propensity float
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

        # Bulk content stats for priors
        stats = database_service.get_content_stats_bulk()

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

                # popularity priors
                st = stats.get(cid, {"clicks": 0, "impressions": 0})
                ctr = smoothed_ctr(st.get("clicks", 0), st.get("impressions", 0))
                log_imp = log1p_int(st.get("impressions", 0))

                # build feature dict from logged row (preferred) + priors
                feat_dict = {
                    "semantic_similarity": _f(rr.get("semantic_similarity"), _f(rr.get("candidate_score"), 0.0)),
                    "keyword_score_total": _f(rr.get("keyword_score_total"), 0.0),  # RAW - normalized below
                    "exact_title_proportion": _f(rr.get("exact_title_proportion"), 0.0),
                    "full_coverage_proportion": _f(rr.get("full_coverage_proportion"), 0.0),
                    "title_keyword_proportion": _f(rr.get("title_keyword_proportion"), 0.0),
                    "body_keyword_proportion": _f(rr.get("body_keyword_proportion"), 0.0),
                    "exact_body_proportion": _f(rr.get("exact_body_proportion"), 0.0),
                    "type_match": _f(rr.get("type_match"), 0.0),
                    "role_match": _f(rr.get("role_match"), 0.0),
                    "code_match_count": _f(rr.get("code_match_count"), 0.0),
                    "lis_match": _f(rr.get("lis_match"), 0.0),
                    "maalgruppe_match": _f(rr.get("maalgruppe_match"), 0.0),
                    "smoothed_ctr": ctr,
                    "log_impressions": log_imp,
                    "position": float(pos),
                }

                feat_dicts.append(feat_dict)
                labels.append(float(clicked))

                # IPS weight to reduce position bias
                prop = pos_prop.get(pos) if pos_prop else None
                if prop is None:
                    prop = propensity_for_position(pos)
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
        content_stats: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> List[Tuple[RerankCandidate, float]]:
        """
        Rerank a list of candidates.

        Provide candidates with their precomputed features (semantic/keyword/metadata).
        This method adds popularity priors (smoothed_ctr/log_impressions) automatically.

        Returns: list of (candidate, score), sorted by score descending.
        """
        if not candidates:
            return []

        if self.model is None:
            # Safe fallback if model isn't available yet:
            scored = [(c, float(c.semantic_similarity)) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        # Fetch stats once (bulk)
        stats = content_stats or database_service.get_content_stats_bulk()

        # Build feature matrix
        X = []
        for c in candidates:
            st = stats.get(c.content_id, {"clicks": 0, "impressions": 0})
            c.smoothed_ctr = smoothed_ctr(st.get("clicks", 0), st.get("impressions", 0))
            c.log_impressions = log1p_int(st.get("impressions", 0))

            feat_dict = {
                "semantic_similarity": _f(c.semantic_similarity),
                "keyword_score_total": _f(c.keyword_score_total),
                "exact_title_proportion": _f(c.exact_title_proportion),
                "full_coverage_proportion": _f(c.full_coverage_proportion),
                "title_keyword_proportion": _f(c.title_keyword_proportion),
                "body_keyword_proportion": _f(c.body_keyword_proportion),
                "exact_body_proportion": _f(c.exact_body_proportion),
                "type_match": _f(c.type_match),
                "role_match": _f(c.role_match),
                "code_match_count": _f(c.code_match_count),
                "lis_match": _f(c.lis_match),
                "maalgruppe_match": _f(c.maalgruppe_match),
                "smoothed_ctr": _f(c.smoothed_ctr),
                "log_impressions": _f(c.log_impressions),
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

    def load(self, path: str) -> None:
        """Load model from disk."""
        m = xgb.XGBRanker()
        m.load_model(path)
        self.model = m


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
    body_keyword_proportion: float = 0.0,
    exact_body_proportion: float = 0.0,
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
        "body_keyword_proportion": float(body_keyword_proportion),
        "exact_body_proportion": float(exact_body_proportion),
        "type_match": float(bool(type_match)),
        "role_match": float(bool(role_match)),
        "code_match_count": float(int(code_match_count)),
        "lis_match": float(bool(lis_match)),
        "maalgruppe_match": float(bool(maalgruppe_match)),
    }
