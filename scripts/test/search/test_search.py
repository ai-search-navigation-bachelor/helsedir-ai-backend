#!/usr/bin/env python3
"""
Test and compare different search methods.

Usage:
    python scripts/test_search.py
    python scripts/test_search.py --query "diabetes behandling"
    python scripts/test_search.py --query "hodepine" --k 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.search.search_service import search_service


def print_results(title: str, results: list, max_results: int = 5):
    """Print search results in a readable format."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    if not results:
        print("  (no results)")
        return

    for i, result in enumerate(results[:max_results], 1):
        print(f"\n  {i}. {result.title[:60]}...")
        print(f"     Score: {result.score:.4f}")
        print(f"     {result.explanation}")


def main():
    parser = argparse.ArgumentParser(description="Test search methods")
    parser.add_argument(
        "--query",
        type=str,
        default="diabetes behandling",
        help="Search query to test",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to show",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        help="Multiple queries to test",
    )
    args = parser.parse_args()

    # Test queries
    if args.queries:
        queries = args.queries
    else:
        queries = [args.query]

    # Default test queries if none specified
    if queries == ["diabetes behandling"]:
        queries = [
            "diabetes behandling",
            "hodepine",
            "barn med astma",
            "psykisk helse ungdom",
            "legemidler eldre",
        ]

    print("\n" + "="*60)
    print("  SEARCH METHOD COMPARISON")
    print("="*60)

    for query in queries:
        print(f"\n\n{'#'*60}")
        print(f"  QUERY: \"{query}\"")
        print(f"{'#'*60}")

        # 1. Keyword search (baseline)
        keyword_results = search_service.search(query, k=args.k)
        print_results("KEYWORD SEARCH (baseline)", keyword_results, args.k)

        # 2. Semantic search
        semantic_results = search_service.search_semantic(query, k=args.k)
        if semantic_results:
            print_results("SEMANTIC SEARCH", semantic_results, args.k)
        else:
            print("\n" + "="*60)
            print("  SEMANTIC SEARCH")
            print("="*60)
            print("  (not available - run train_embedding_model.py and generate_embeddings.py first)")

        # 3. Hybrid search
        hybrid_results = search_service.search_hybrid(query, k=args.k)
        if hybrid_results and semantic_results:
            print_results("HYBRID SEARCH (BM25 + semantic via RRF)", hybrid_results, args.k)

        # Compare rankings
        if keyword_results and semantic_results:
            print(f"\n{'-'*60}")
            print("  RANKING COMPARISON")
            print(f"{'-'*60}")

            keyword_ids = [r.id for r in keyword_results[:args.k]]
            semantic_ids = [r.id for r in semantic_results[:args.k]]
            hybrid_ids = [r.id for r in hybrid_results[:args.k]]

            # Find differences
            only_in_keyword = set(keyword_ids) - set(semantic_ids)
            only_in_semantic = set(semantic_ids) - set(keyword_ids)

            if only_in_keyword:
                print(f"  Only in keyword top-{args.k}: {len(only_in_keyword)} results")
            if only_in_semantic:
                print(f"  Only in semantic top-{args.k}: {len(only_in_semantic)} results")

            overlap = set(keyword_ids) & set(semantic_ids)
            print(f"  Overlap in top-{args.k}: {len(overlap)} results")

    print("\n")


if __name__ == "__main__":
    main()
