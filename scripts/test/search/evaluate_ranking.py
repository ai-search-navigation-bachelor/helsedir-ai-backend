#!/usr/bin/env python3
"""
Offline evaluation of ranking models: NDCG@5, NDCG@10, MRR, Precision@1.

Fetches historical searches from DB, splits by time (80% train / 20% test),
and compares:
  1. RRF-only baseline (no ML reranking)
  2. Current trained model (loaded from disk)
  3. Newly trained model (if --retrain flag is used)

Usage:
    python scripts/test/search/evaluate_ranking.py
    python scripts/test/search/evaluate_ranking.py --retrain
    python scripts/test/search/evaluate_ranking.py --days 90
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.data.database_service import database_service
from app.ml.ranking_model import HealthContentReranker, _f, _compute_freshness


def load_groups(days_back: int = 180) -> Dict[str, List[dict]]:
    """Load training rows from DB and group by search_id."""
    rows = database_service.get_ltr_training_rows(days_back=days_back)
    if not rows:
        print("No training data found in database.")
        return {}

    groups: Dict[str, List[dict]] = {}
    for r in rows:
        sid = r.get("search_id")
        if sid is None:
            continue
        groups.setdefault(str(sid), []).append(r)

    # Filter: require at least one click in the group
    filtered = {}
    for sid, items in groups.items():
        if any(int(item.get("clicked", 0)) == 1 for item in items):
            if len(items) >= 2:
                filtered[sid] = items

    return filtered


def time_split(groups: Dict[str, List[dict]], train_ratio: float = 0.8):
    """Split groups into train/test by time ordering (oldest first)."""
    # Sort search_ids by the earliest timestamp in each group
    def earliest_ts(items):
        timestamps = [item.get("timestamp") for item in items if item.get("timestamp")]
        return min(timestamps) if timestamps else ""

    sorted_sids = sorted(groups.keys(), key=lambda sid: earliest_ts(groups[sid]))

    split_idx = int(len(sorted_sids) * train_ratio)
    train_sids = sorted_sids[:split_idx]
    test_sids = sorted_sids[split_idx:]

    train_groups = {sid: groups[sid] for sid in train_sids}
    test_groups = {sid: groups[sid] for sid in test_sids}

    return train_groups, test_groups


def compute_ctr_from_groups(groups: Dict[str, List[dict]]) -> Dict[str, float]:
    """Compute smoothed CTR from a set of groups (avoids data leakage)."""
    impressions: Dict[str, int] = {}
    clicks: Dict[str, int] = {}
    for items in groups.values():
        for item in items:
            cid = item.get("content_id", "")
            impressions[cid] = impressions.get(cid, 0) + 1
            if int(item.get("clicked", 0)) == 1:
                clicks[cid] = clicks.get(cid, 0) + 1
    return {
        cid: (clicks.get(cid, 0) + 1) / (imp + 21)
        for cid, imp in impressions.items()
    }


def evaluate_rrf_baseline(groups: Dict[str, List[dict]]) -> Dict[str, float]:
    """Evaluate RRF-only baseline (rank by rrf_score, no ML)."""
    from sklearn.metrics import ndcg_score

    ndcg5_scores, ndcg10_scores, mrr_scores, p1_scores = [], [], [], []

    for sid, items in groups.items():
        if len(items) < 2:
            continue

        relevance = [float(item.get("clicked", 0)) for item in items]
        if sum(relevance) == 0:
            continue

        # Use RRF score as the ranking signal
        rrf_scores = [_f(item.get("rrf_score"), 0.0) for item in items]

        try:
            ndcg5_scores.append(ndcg_score([relevance], [rrf_scores], k=5))
        except Exception:
            logger.debug("NDCG@5 failed for group %s: rel=%s scores=%s", sid, relevance, rrf_scores)
        try:
            ndcg10_scores.append(ndcg_score([relevance], [rrf_scores], k=10))
        except Exception:
            logger.debug("NDCG@10 failed for group %s: rel=%s scores=%s", sid, relevance, rrf_scores)

        # MRR + P@1 based on RRF ranking
        ranked = sorted(range(len(rrf_scores)), key=lambda i: rrf_scores[i], reverse=True)
        ranked_rel = [relevance[i] for i in ranked]
        for rank, rel in enumerate(ranked_rel, start=1):
            if rel > 0:
                mrr_scores.append(1.0 / rank)
                break
        p1_scores.append(1.0 if ranked_rel[0] > 0 else 0.0)

    return {
        "ndcg@5": float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0,
        "ndcg@10": float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "precision@1": float(np.mean(p1_scores)) if p1_scores else 0.0,
        "num_queries": len(ndcg5_scores),
    }


def print_metrics(name: str, metrics: Dict[str, float]):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  NDCG@5:       {metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10:      {metrics['ndcg@10']:.4f}")
    print(f"  MRR:          {metrics['mrr']:.4f}")
    print(f"  Precision@1:  {metrics['precision@1']:.4f}")
    print(f"  Queries:      {int(metrics['num_queries'])}")


def print_comparison(baseline: Dict[str, float], model: Dict[str, float], label: str):
    """Print delta between baseline and model."""
    print(f"\n  Delta ({label} vs RRF baseline):")
    for metric in ["ndcg@5", "ndcg@10", "mrr", "precision@1"]:
        delta = model[metric] - baseline[metric]
        sign = "+" if delta >= 0 else ""
        print(f"    {metric:14s} {sign}{delta:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking models offline")
    parser.add_argument("--days", type=int, default=180, help="Days of data to use")
    parser.add_argument("--retrain", action="store_true", help="Train a new model on train split")
    parser.add_argument("--split", type=float, default=0.8, help="Train/test split ratio")
    args = parser.parse_args()

    print("Loading data from database...")
    groups = load_groups(days_back=args.days)
    if not groups:
        print("No data available for evaluation.")
        return

    print(f"Loaded {len(groups)} search groups with clicks")

    train_groups, test_groups = time_split(groups, train_ratio=args.split)
    print(f"Train: {len(train_groups)} groups, Test: {len(test_groups)} groups")

    if not test_groups:
        print("Not enough data for test split.")
        return

    # Compute CTR maps scoped to each partition to avoid data leakage
    train_ctr_map = compute_ctr_from_groups(train_groups)
    test_ctr_map = compute_ctr_from_groups(test_groups)

    # 1. RRF baseline
    rrf_metrics = evaluate_rrf_baseline(test_groups)
    print_metrics("RRF-Only Baseline", rrf_metrics)

    # 2. Current model from disk
    model_path = project_root / "models" / "ranking" / "reranker.json"
    if model_path.exists():
        print("\nLoading current model from disk...")
        current_model = HealthContentReranker.load(str(model_path))
        current_metrics = current_model.evaluate(test_groups, ctr_map=test_ctr_map)
        print_metrics("Current Model (from disk)", current_metrics)
        print_comparison(rrf_metrics, current_metrics, "Current model")
    else:
        print(f"\nNo model found at {model_path}, skipping current model evaluation.")

    # 3. Retrained model (optional) — trained strictly on train_groups to avoid data leakage
    if args.retrain:
        print("\nTraining new model on train split...")
        new_model = HealthContentReranker()
        train_result = new_model.train_from_groups(
            groups=train_groups,
            ctr_map=train_ctr_map,
            min_group_size=5,
            require_any_click=True,
            verbose=False,
        )
        print(f"Training: {int(train_result['groups'])} groups, {int(train_result['rows'])} rows")

        if train_result["trained"]:
            new_metrics = new_model.evaluate(test_groups, ctr_map=test_ctr_map)
            print_metrics("Newly Trained Model", new_metrics)
            print_comparison(rrf_metrics, new_metrics, "New model")
        else:
            print("Training failed - not enough data")

    print(f"\n{'=' * 50}")
    print("Evaluation complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
