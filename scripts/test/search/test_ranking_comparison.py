"""
Compare search results with and without ranking model.

This script runs the same queries with ML ranking enabled and disabled
to show how the ranking model affects result ordering.

Usage:
    python scripts/test_ranking_comparison.py
    python scripts/test_ranking_comparison.py --query "diabetes behandling"
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from app.services.search.search_service import search_service
from app.config import settings


def format_result(result, rank: int) -> str:
    """Format a search result for display."""
    return (
        f"{rank}. [{result.score:.3f}] {result.title[:60]}\n"
        f"   ID: {result.id}"
    )


def compare_results(
    query: str,
    without_ranking: List,
    with_ranking: List,
    top_k: int = 10
) -> None:
    """Compare and display results from both methods."""
    
    print("\n" + "="*80)
    print(f"Query: '{query}'")
    print("="*80)
    
    # Without ranking
    print(f"\n📊 WITHOUT RANKING MODEL (Hybrid search only):")
    print("-" * 80)
    for i, result in enumerate(without_ranking[:top_k], 1):
        print(format_result(result, i))
        print()
    
    # With ranking
    print(f"\n🤖 WITH RANKING MODEL (ML re-ranked):")
    print("-" * 80)
    for i, result in enumerate(with_ranking[:top_k], 1):
        print(format_result(result, i))
        print()
    
    # Position changes analysis
    print("\n" + "="*80)
    print("📈 POSITION CHANGES:")
    print("="*80)
    
    # Create ID -> position mappings
    without_positions = {r.id: i for i, r in enumerate(without_ranking[:top_k], 1)}
    with_positions = {r.id: i for i, r in enumerate(with_ranking[:top_k], 1)}
    
    # Find changes
    changes = []
    for content_id, new_pos in with_positions.items():
        old_pos = without_positions.get(content_id, None)
        if old_pos is None:
            changes.append((content_id, None, new_pos, "NEW"))
        elif old_pos != new_pos:
            change = old_pos - new_pos  # Positive = moved up
            changes.append((content_id, old_pos, new_pos, change))
    
    # Check for items that dropped out
    for content_id, old_pos in without_positions.items():
        if content_id not in with_positions:
            changes.append((content_id, old_pos, None, "OUT"))
    
    if not changes:
        print("✓ No position changes - ranking model agrees with hybrid search")
    else:
        # Sort by absolute change (biggest movers first)
        changes.sort(key=lambda x: abs(x[3]) if isinstance(x[3], int) else 0, reverse=True)
        
        for content_id, old_pos, new_pos, change in changes:
            # Get result title
            result = None
            for r in with_ranking:
                if r.id == content_id:
                    result = r
                    break
            if result is None:
                for r in without_ranking:
                    if r.id == content_id:
                        result = r
                        break
            
            title = result.title[:50] if result else content_id
            
            if change == "NEW":
                print(f"  🆕 NEW in top-{top_k}: #{new_pos} - {title}")
            elif change == "OUT":
                print(f"  ❌ Dropped out: was #{old_pos} - {title}")
            elif change > 0:
                print(f"  ⬆️  Moved UP {change} positions: #{old_pos} → #{new_pos} - {title}")
            elif change < 0:
                print(f"  ⬇️  Moved DOWN {abs(change)} positions: #{old_pos} → #{new_pos} - {title}")
    
    # Score statistics
    print("\n" + "="*80)
    print("📊 SCORE STATISTICS:")
    print("="*80)
    
    if without_ranking:
        without_scores = [r.score for r in without_ranking[:top_k]]
        print(f"Without ranking:")
        print(f"  Range: {min(without_scores):.3f} - {max(without_scores):.3f}")
        print(f"  Average: {sum(without_scores)/len(without_scores):.3f}")
    
    if with_ranking:
        with_scores = [r.score for r in with_ranking[:top_k]]
        print(f"\nWith ranking:")
        print(f"  Range: {min(with_scores):.3f} - {max(with_scores):.3f}")
        print(f"  Average: {sum(with_scores)/len(with_scores):.3f}")


def run_comparison(query: str, role: str = None, k: int = 10) -> None:
    """Run comparison for a single query."""
    
    print("\n🔍 Running search without ranking model...")
    
    # Temporarily disable ranking
    original_setting = settings.ml_ranking_enabled
    settings.ml_ranking_enabled = False
    
    try:
        without_ranking = search_service.search_hybrid(
            query=query,
            role=role,
            k=k
        )
        
        print(f"✓ Found {len(without_ranking)} results")
        
        # Enable ranking
        settings.ml_ranking_enabled = True
        
        print("\n🤖 Running search with ranking model...")
        with_ranking = search_service.search_hybrid(
            query=query,
            role=role,
            k=k
        )
        
        print(f"✓ Found {len(with_ranking)} results")
        
        # Compare
        compare_results(query, without_ranking, with_ranking, k)
        
    finally:
        # Restore original setting
        settings.ml_ranking_enabled = original_setting


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare search results with and without ranking model"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query to test (if not provided, runs multiple test queries)",
    )
    parser.add_argument(
        "--role",
        type=str,
        help="User role (lege, sykepleier, pasient)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of results to compare (default: 10)",
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 RANKING MODEL COMPARISON TEST")
    print("="*80)
    
    # Check if ranking model is available
    from app.services.search.ml_service import ml_service
    
    model_path = Path("models/ranking/model.keras")
    if not model_path.exists():
        print("\n❌ Error: Ranking model not found!")
        print(f"   Expected at: {model_path}")
        print("\n   Train the model first:")
        print("   python scripts/train/train_ranking_model.py")
        sys.exit(1)
    
    # Load ranking model
    print("\n📥 Loading ranking model...")
    if ml_service.load_ranking_model():
        print("✓ Ranking model loaded successfully")
    else:
        print("❌ Failed to load ranking model")
        sys.exit(1)
    
    # Run comparison
    if args.query:
        # Single query
        run_comparison(args.query, args.role, args.k)
    else:
        # Multiple test queries
        test_queries = [
            ("diabetes behandling", None),
            ("astma retningslinjer", "lege"),
            ("covid-19 symptomer", "pasient"),
            ("sårbehandling", "sykepleier"),
        ]
        
        for i, (query, role) in enumerate(test_queries, 1):
            if i > 1:
                input("\n\nPress Enter to continue to next query...")
            
            run_comparison(query, role, args.k)
    
    print("\n" + "="*80)
    print("✓ Comparison complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
