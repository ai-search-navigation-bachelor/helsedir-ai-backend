#!/usr/bin/env python3
"""
Evaluate search quality against real page view data from helsedirektoratet.no.

For the most visited pages, derives search queries from titles and checks
whether our search system ranks them in the top results.

Metrics:
  - Recall@K: fraction of popular pages found in top K results
  - Weighted Recall@K: same, but weighted by views (missing a 13k-view page
    is worse than missing a 200-view page)
  - MRR (Mean Reciprocal Rank): average of 1/rank for found pages

Usage:
    python scripts/test/search/eval_popularity_recall.py
    python scripts/test/search/eval_popularity_recall.py --top 50 --k 5
    python scripts/test/search/eval_popularity_recall.py --method hybrid
"""

import argparse
import csv
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from app.services.search.search_service import search_service
from app.services.data.content_service import content_service


# ---------------------------------------------------------------------------
# CSV parsing (same as seed script)
# ---------------------------------------------------------------------------

def parse_page_view_csv(csv_path: str) -> List[Dict]:
    """Parse the helsedir page view CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if raw_line.startswith('"') and raw_line.endswith('"'):
                raw_line = raw_line[1:-1]
            raw_line = raw_line.replace('""', '"')

            reader = csv.reader(io.StringIO(raw_line))
            for fields in reader:
                if len(fields) < 5:
                    continue
                if line_no == 0 and fields[0].strip().lower() == "title":
                    continue

                try:
                    views = int(fields[4].strip())
                except (ValueError, IndexError):
                    continue

                title = fields[0].strip()
                url = fields[1].strip()

                if not url or "helsedirektoratet.no" not in url:
                    continue

                path = urlparse(url).path.rstrip("/") or "/"

                rows.append({
                    "title": title,
                    "url": url,
                    "views": views,
                    "path": path,
                })

    return rows


# ---------------------------------------------------------------------------
# Query derivation
# ---------------------------------------------------------------------------

# Words too generic to be useful as search queries
STOP_WORDS = {
    "helsedirektoratet", "og", "i", "for", "av", "til", "med", "om",
    "på", "en", "et", "er", "det", "de", "den", "som", "ved", "fra",
    "kan", "har", "skal", "vil", "bli", "ble", "blitt", "eller",
    "andre", "denne", "dette", "disse", "alle", "noen", "hva",
    "hvordan", "når", "hvor", "the", "and", "of", "in", "for", "to",
    "-", "–",
}


def derive_query(title: str) -> Optional[str]:
    """
    Derive a search query from a page title.

    Strips "- Helsedirektoratet" suffix and stop words.
    Returns None if the remaining query is too short.
    """
    # Remove "- Helsedirektoratet" suffix
    title = re.sub(r'\s*[-–]\s*Helsedirektoratet\s*$', '', title, flags=re.IGNORECASE)

    # Remove parenthetical content like (§§ 35–37)
    title = re.sub(r'\([^)]*\)', '', title)

    # Keep meaningful words
    words = [w for w in title.split() if w.lower() not in STOP_WORDS and len(w) > 1]

    if len(words) < 1:
        return None

    # Use first 4 words to avoid overly specific queries
    query = " ".join(words[:4])
    return query.strip()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_csv_to_content(csv_rows: List[Dict]) -> List[Dict]:
    """
    Match CSV rows to content in our database via path.

    Only returns rows that have a matching content item.
    """
    # Build path -> content_id lookup from content service
    all_content = content_service.get_all_content()
    path_to_id = {}
    for item in all_content:
        if item.path:
            path_to_id[item.path] = item.id

    matched = []
    for row in csv_rows:
        path = row["path"]
        if path in path_to_id:
            row["content_id"] = path_to_id[path]
            matched.append(row)

    return matched


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pages: List[Dict],
    method: str = "hybrid",
    k: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate search recall for popular pages.

    For each page, derive a query from the title, run it,
    and check if the page appears in the top K results.
    """
    found = 0
    total = len(pages)
    total_views = sum(p["views"] for p in pages)
    found_views = 0
    reciprocal_ranks = []
    results_detail = []

    for i, page in enumerate(pages):
        query = derive_query(page["title"])
        if not query:
            total -= 1
            total_views -= page["views"]
            continue

        content_id = page["content_id"]

        # Run search
        if method == "hybrid":
            results = search_service.search_hybrid(query=query, k=k)
        elif method == "semantic":
            results = search_service.search_semantic(query=query, k=k)
        else:
            results = search_service.search(query=query, k=k)

        # Check if target page is in results
        result_ids = [r.id for r in results]
        rank = None
        if content_id in result_ids:
            rank = result_ids.index(content_id) + 1
            found += 1
            found_views += page["views"]
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        status = f"rank={rank}" if rank else "MISS"
        results_detail.append({
            "title": page["title"],
            "query": query,
            "views": page["views"],
            "rank": rank,
            "status": status,
        })

        if verbose:
            icon = f"#{rank}" if rank else "MISS"
            print(f"  [{i+1:>3}/{len(pages)}] {icon:<6} {page['views']:>6,} views  q=\"{query}\"")

    # Calculate metrics
    recall_at_k = found / total if total > 0 else 0.0
    weighted_recall = found_views / total_views if total_views > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    metrics = {
        "recall@k": recall_at_k,
        "weighted_recall@k": weighted_recall,
        "mrr": mrr,
        "found": found,
        "total": total,
        "k": k,
    }

    return metrics, results_detail


def print_report(
    metrics: Dict,
    results_detail: List[Dict],
    method: str,
    top_n: int,
):
    """Print evaluation report."""
    k = metrics["k"]

    print("\n" + "=" * 60)
    print(f"  Popularity Recall Evaluation")
    print(f"  Method: {method} | Top {top_n} pages | K={k}")
    print("=" * 60)

    print(f"\n  Recall@{k}:          {metrics['recall@k']:.1%}  ({metrics['found']}/{metrics['total']} pages found)")
    print(f"  Weighted Recall@{k}: {metrics['weighted_recall@k']:.1%}  (weighted by views)")
    print(f"  MRR:                {metrics['mrr']:.3f}")

    # Show misses (most impactful first)
    misses = [r for r in results_detail if r["rank"] is None]
    misses.sort(key=lambda x: -x["views"])

    if misses:
        print(f"\n  Misses ({len(misses)} pages not found in top {k}):")
        for m in misses[:15]:
            title = re.sub(r'\s*[-–]\s*Helsedirektoratet\s*$', '', m["title"])[:55]
            print(f"    {m['views']:>6,} views  q=\"{m['query']}\"  {title}")
        if len(misses) > 15:
            print(f"    ... and {len(misses) - 15} more")

    # Show rank distribution
    ranks = [r["rank"] for r in results_detail if r["rank"] is not None]
    if ranks:
        print(f"\n  Rank distribution (found pages):")
        for pos in range(1, k + 1):
            count = sum(1 for r in ranks if r == pos)
            if count > 0:
                bar = "#" * count
                print(f"    #{pos:<3} {count:>3}  {bar}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate search recall against popular pages"
    )
    parser.add_argument(
        "--csv",
        default=str(project_root / "scripts" / "data" / "importing" / "helsedir_page_view.csv"),
        help="Path to page view CSV file",
    )
    parser.add_argument("--top", type=int, default=100, help="Evaluate top N most viewed pages")
    parser.add_argument("--k", type=int, default=10, help="Check if page appears in top K results")
    parser.add_argument("--method", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    args = parser.parse_args()

    print("=" * 60)
    print("  Popularity Recall Evaluation")
    print("=" * 60)

    # Parse CSV
    print(f"\nParsing CSV...")
    csv_rows = parse_page_view_csv(args.csv)
    print(f"  {len(csv_rows)} pages with views")

    # Match to content
    print("Matching to content database...")
    matched = match_csv_to_content(csv_rows)
    print(f"  {len(matched)} pages matched to content")

    if not matched:
        print("\nNo pages matched. Check that content is loaded.")
        return

    # Take top N by views
    matched.sort(key=lambda x: -x["views"])
    pages = matched[:args.top]
    print(f"\nEvaluating top {len(pages)} pages (views: {pages[0]['views']:,} - {pages[-1]['views']:,})")
    print()

    # Run evaluation
    metrics, details = evaluate(
        pages,
        method=args.method,
        k=args.k,
        verbose=not args.quiet,
    )

    print_report(metrics, details, args.method, len(pages))


if __name__ == "__main__":
    main()
