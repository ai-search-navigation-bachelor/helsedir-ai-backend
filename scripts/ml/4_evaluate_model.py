#!/usr/bin/env python3
"""
Evaluate and compare embedding models on the held-out test set.

Uses the test triplets saved by 2_finetune_gpl.py and evaluates against the
full document corpus — giving realistic IR metrics for reporting.

Typical use: compare base model vs fine-tuned model side by side.

Usage:
    # Compare base vs fine-tuned (default)
    python scripts/ml/4_evaluate_model.py

    # Evaluate a single model
    python scripts/ml/4_evaluate_model.py --models intfloat/multilingual-e5-base

    # Custom models
    python scripts/ml/4_evaluate_model.py \\
        --models intfloat/multilingual-e5-base models/finetuned-e5-gpl
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml.utils import enrich_temasider_with_children, enrich_with_child_content  # noqa: E402


def resolve_triplets_path(path_arg: str) -> Path:
    """Resolve test triplets from the common repo locations."""
    raw_path = Path(path_arg)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            project_root / raw_path,
            Path.cwd() / raw_path,
            project_root / "scripts" / "data" / raw_path.name,
            project_root / "data" / raw_path.name,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def load_content(db_service) -> list:
    conn = db_service._get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT c.id, c.tittel, c.tekst, c.info_type, c.role_tags, c.links
        FROM content c
        JOIN content_type_config ctc
          ON LOWER(c.info_type) = LOWER(ctc.info_type)
        WHERE ctc.searchable = 1
        """
    )
    items = cursor.fetchall()
    cursor.close()
    return items


def build_id_to_passage(content_items: list) -> dict:
    from app.ml.embedding_model import HealthContentEmbedding
    return {
        str(item["id"]): HealthContentEmbedding.format_passage(item)
        for item in content_items
    }


def evaluate_model(model_name: str, test_queries: dict, test_relevant_docs: dict, corpus: dict) -> dict:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    print(f"\n  Loading {model_name}...")
    model = SentenceTransformer(model_name)

    evaluator = InformationRetrievalEvaluator(
        test_queries,
        corpus,
        test_relevant_docs,
        name="ir-test",
        show_progress_bar=True,
    )

    result = evaluator(model)
    return result if isinstance(result, dict) else {}


def print_comparison(model_names: list, results: list[dict]) -> None:
    metrics = ["ndcg@10", "mrr@10", "recall@10", "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10"]

    col_width = max(len(m) for m in model_names) + 2
    header_width = 22

    print("\n" + "=" * (header_width + col_width * len(model_names)))
    print(f"{'Metric':<{header_width}}" + "".join(f"{m:>{col_width}}" for m in model_names))
    print("-" * (header_width + col_width * len(model_names)))

    for metric in metrics:
        row = f"{metric:<{header_width}}"
        values = []
        for result in results:
            val = None
            for key, v in result.items():
                if metric in key.lower():
                    val = v
                    break
            values.append(val)

        best_val = max((v for v in values if v is not None), default=None)
        for val in values:
            if val is None:
                row += f"{'N/A':>{col_width}}"
            elif val == best_val and len(model_names) > 1:
                row += f"{'*' + f'{val:.4f}':>{col_width}}"
            else:
                row += f"{val:>{col_width}.4f}"
        print(row)

    print("=" * (header_width + col_width * len(model_names)))
    if len(model_names) > 1:
        print("  * = best")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding models on held-out test set")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["intfloat/multilingual-e5-base", "models/finetuned-e5-gpl"],
        help="Model names or paths to evaluate",
    )
    parser.add_argument(
        "--test-triplets",
        default="data/gpl_test_triplets.json",
        help="Path to saved test triplets from 2_finetune_gpl.py",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        print("sentence-transformers available")
    except ImportError:
        print("Error: sentence-transformers not installed")
        sys.exit(1)

    from app.services.data.database_service import database_service
    print("Database pool initialized")

    # Load test triplets
    test_triplets_path = resolve_triplets_path(args.test_triplets)
    if not test_triplets_path.exists():
        print(f"Error: Test triplets not found at {test_triplets_path}")
        print("Also checked scripts/data/gpl_test_triplets.json and data/gpl_test_triplets.json.")
        print("Run 2_finetune_gpl.py first to generate and save the test triplets.")
        sys.exit(1)

    with open(test_triplets_path, encoding="utf-8") as f:
        test_triplets = json.load(f)
    print(f"Loaded {len(test_triplets)} test triplets from {test_triplets_path}")

    # Load content and build full corpus
    print("\nLoading content from database...")
    content_items = load_content(database_service)
    print(f"Found {len(content_items)} content items")

    print("\nEnriching temasider with linked content...")
    enrich_temasider_with_children(content_items)
    print("\nEnriching content with child content...")
    enrich_with_child_content(content_items)

    print("\nBuilding full corpus...")
    id_to_passage = build_id_to_passage(content_items)
    print(f"Corpus size: {len(id_to_passage)} documents")

    # Build evaluator inputs
    test_queries = {f"q{i}": t["query"] for i, t in enumerate(test_triplets)}
    test_relevant_docs = {f"q{i}": {t["positive_id"]} for i, t in enumerate(test_triplets)}
    print(f"Test queries: {len(test_queries)}")

    # Evaluate each model
    results = []
    for model_name in args.models:
        model_path = project_root / model_name if not Path(model_name).is_absolute() else Path(model_name)
        resolved = str(model_path) if model_path.exists() else model_name
        print(f"\nEvaluating: {model_name}")
        result = evaluate_model(resolved, test_queries, test_relevant_docs, id_to_passage)
        results.append(result)

    # Print comparison table
    short_names = [Path(m).name if "/" in m or "\\" in m else m for m in args.models]
    print_comparison(short_names, results)


if __name__ == "__main__":
    main()
