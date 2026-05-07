#!/usr/bin/env python3
"""
Compare BM25, semantic, and hybrid search on the held-out GPL test triplets.

Metrics: NDCG@10, MRR@10, Recall@10

Usage:
    python scripts/test/search/evaluate_search_methods.py
    python scripts/test/search/evaluate_search_methods.py --k 10
    python scripts/test/search/evaluate_search_methods.py --test-triplets data/gpl_test_triplets.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if (rank is not None and rank <= k) else 0.0


def find_rank(results, positive_id: str) -> int | None:
    for i, result in enumerate(results, start=1):
        rid = str(getattr(result, "id", None) or getattr(result, "item", None) and result.item.id)
        if rid == positive_id:
            return i
    return None


def find_rank_bm25(results, positive_id: str) -> int | None:
    for i, hit in enumerate(results, start=1):
        if str(hit.item.id) == positive_id:
            return i
    return None


def evaluate_method(name: str, search_fn, triplets: list, k: int) -> dict:
    ndcg_scores, mrr_scores, recall_scores = [], [], []
    failed = 0

    for triplet in triplets:
        query = triplet["query"]
        positive_id = str(triplet["positive_id"])

        try:
            results = search_fn(query, k=k)
        except Exception:
            failed += 1
            ndcg_scores.append(0.0)
            mrr_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        if name == "BM25":
            rank = find_rank_bm25(results, positive_id)
        else:
            rank = find_rank(results, positive_id)

        ndcg_scores.append(ndcg_at_k(rank, k))
        mrr_scores.append(mrr_at_k(rank, k))
        recall_scores.append(recall_at_k(rank, k))

    n = len(triplets)
    return {
        "ndcg": sum(ndcg_scores) / n,
        "mrr": sum(mrr_scores) / n,
        "recall": sum(recall_scores) / n,
        "failed": failed,
    }


def print_results(methods: list[str], results: list[dict], k: int) -> None:
    col_width = 14
    header_width = 14

    print("\n" + "=" * (header_width + col_width * len(methods)))
    print(f"{'Metric':<{header_width}}" + "".join(f"{m:>{col_width}}" for m in methods))
    print("-" * (header_width + col_width * len(methods)))

    for metric_key, metric_label in [("ndcg", f"NDCG@{k}"), ("mrr", f"MRR@{k}"), ("recall", f"Recall@{k}")]:
        values = [r[metric_key] for r in results]
        best = max(values)
        row = f"{metric_label:<{header_width}}"
        for val in values:
            marker = "*" if val == best and len(methods) > 1 else " "
            row += f"{marker}{val:.4f}".rjust(col_width)
        print(row)

    print("=" * (header_width + col_width * len(methods)))
    if len(methods) > 1:
        print("  * = best")

    for method, result in zip(methods, results):
        if result["failed"] > 0:
            print(f"  {method}: {result['failed']} queries failed (counted as 0)")


def main():
    parser = argparse.ArgumentParser(description="Compare search methods on GPL test triplets")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for metrics (default: 10)")
    parser.add_argument(
        "--test-triplets",
        default="data/gpl_test_triplets.json",
        help="Path to test triplets from 2_finetune_gpl.py",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "semantic", "hybrid"],
        choices=["bm25", "semantic", "hybrid"],
        help="Search methods to evaluate",
    )
    args = parser.parse_args()

    # Load test triplets
    triplets_path = project_root / args.test_triplets
    if not triplets_path.exists():
        print(f"Error: test triplets not found at {triplets_path}")
        print("Run 2_finetune_gpl.py first to generate the test triplets.")
        sys.exit(1)

    with open(triplets_path, encoding="utf-8") as f:
        triplets = json.load(f)
    print(f"Loaded {len(triplets)} test triplets")

    # Initialize services
    print("Initializing search services...")
    from app.services.data.content_service import content_service
    from app.services.search.bm25_search import bm25_search
    from app.services.search.semantic_search import semantic_search
    from app.services.search.hybrid_search import hybrid_search
    from app.config import settings

    content_service.load_all_content()
    bm25_search._ensure_index()

    if "semantic" in args.methods or "hybrid" in args.methods:
        if not settings.ml_embedding_enabled:
            print("Warning: ML_EMBEDDING_ENABLED=false — semantic and hybrid will return empty results.")
        else:
            semantic_search.load_embedding_model()
            semantic_search.load_content_embeddings()

    # Build search functions
    method_map = {
        "bm25":     ("BM25",     lambda q, k: bm25_search.search(q, k=k)),
        "semantic": ("Semantic", lambda q, k: semantic_search.search(q, k=k)),
        "hybrid":   ("Hybrid",   lambda q, k: hybrid_search.search(q, k=k, rerank=False)),
    }

    method_names, search_fns = [], []
    for m in args.methods:
        name, fn = method_map[m]
        method_names.append(name)
        search_fns.append(fn)

    # Evaluate
    results = []
    for name, fn in zip(method_names, search_fns):
        print(f"\nEvaluating {name}...")
        result = evaluate_method(name, fn, triplets, args.k)
        results.append(result)
        print(f"  NDCG@{args.k}: {result['ndcg']:.4f}  MRR@{args.k}: {result['mrr']:.4f}  Recall@{args.k}: {result['recall']:.4f}")

    print_results(method_names, results, args.k)


if __name__ == "__main__":
    main()
