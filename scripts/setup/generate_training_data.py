"""
Generate training data for ranking model using real page view data and real search results.

Uses the helsedir page view CSV to:
1. Derive search queries from popular page titles
2. Run real hybrid searches to get results with real features
3. Simulate clicks weighted by page popularity (views)

This produces training data grounded in real popularity and real retrieval scores,
not random numbers.

Usage:
    python scripts/setup/generate_training_data.py
    python scripts/setup/generate_training_data.py --top 200 --k 10
    python scripts/setup/generate_training_data.py --clear
"""

import csv
import io
import random
import re
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mysql.connector
from app.config import settings
from app.services.search.search_service import search_service
from app.services.data.content_service import content_service


# ---------------------------------------------------------------------------
# CSV parsing & query derivation (shared with eval script)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "helsedirektoratet", "og", "i", "for", "av", "til", "med", "om",
    "på", "en", "et", "er", "det", "de", "den", "som", "ved", "fra",
    "kan", "har", "skal", "vil", "bli", "ble", "blitt", "eller",
    "andre", "denne", "dette", "disse", "alle", "noen", "hva",
    "hvordan", "når", "hvor", "the", "and", "of", "in", "to",
    "-", "–",
}


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


def derive_query(title: str) -> Optional[str]:
    """Derive a search query from a page title."""
    title = re.sub(r'\s*[-–]\s*Helsedirektoratet\s*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\([^)]*\)', '', title)
    words = [w for w in title.split() if w.lower() not in STOP_WORDS and len(w) > 1]

    if len(words) < 1:
        return None

    return " ".join(words[:4]).strip()


def match_csv_to_content(csv_rows: List[Dict]) -> List[Dict]:
    """Match CSV rows to content in our database via path."""
    all_content = content_service.get_all_content()
    path_to_id = {}
    for item in all_content:
        if item.path:
            path_to_id[item.path] = item.id

    matched = []
    for row in csv_rows:
        if row["path"] in path_to_id:
            row["content_id"] = path_to_id[row["path"]]
            matched.append(row)

    return matched


# ---------------------------------------------------------------------------
# Role match (same logic as search_controller)
# ---------------------------------------------------------------------------

def compute_role_match(role: Optional[str], role_tags: Optional[List[str]]) -> float:
    """Compute role match score."""
    role_tags = role_tags or []
    if role and role_tags:
        if role in role_tags:
            return 1.0 / len(role_tags)
    elif not role and not role_tags:
        return 0.5
    elif not role_tags:
        return 0.3
    return 0.0


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

ROLES = ["Lege", "Sykepleier", None]


def get_connection():
    """Create database connection."""
    try:
        return mysql.connector.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
        )
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def generate_training_data(conn, pages: List[Dict], k: int = 10):
    """
    Generate training data by running real searches for popular pages.

    For each page:
    1. Derive a query from the title
    2. Run hybrid search to get real results with real features
    3. Log the search and results
    4. Simulate clicks: the target page (if found) is clicked with high probability,
       other results are clicked with lower probability weighted by position bias
    """
    cursor = conn.cursor()

    # Build content_id -> role_tags lookup
    all_content = content_service.get_all_content()
    content_role_tags = {item.id: item.role_tags or [] for item in all_content}

    # Normalize views for click probability
    max_views = max(p["views"] for p in pages) if pages else 1

    searches_created = 0
    results_shown = 0
    clicks_created = 0
    skipped = 0

    start_date = datetime.now() - timedelta(days=30)

    for i, page in enumerate(pages):
        query = derive_query(page["title"])
        if not query:
            skipped += 1
            continue

        target_id = page["content_id"]
        view_weight = page["views"] / max_views  # 0-1, higher = more popular

        # Pick a random role for this search
        role = random.choice(ROLES)

        # Run real hybrid search
        try:
            results = search_service.search_hybrid(query=query, role=role, k=k, rerank=False)
        except Exception as e:
            print(f"  Search failed for '{query}': {e}")
            skipped += 1
            continue

        if not results:
            skipped += 1
            continue

        search_id = str(uuid.uuid4())
        timestamp = start_date + timedelta(seconds=random.randint(0, 30 * 24 * 60 * 60))

        # Insert search log
        cursor.execute(
            "INSERT INTO search_logs (search_id, query, role, timestamp) VALUES (%s, %s, %s, %s)",
            (search_id, query, role, timestamp),
        )
        searches_created += 1

        # Insert results shown with real features
        for position, result in enumerate(results, start=1):
            role_match = compute_role_match(role, content_role_tags.get(result.id, []))

            cursor.execute(
                """
                INSERT INTO search_results_shown
                    (search_id, content_id, position, score,
                     semantic_score, bm25_score, rrf_score, role_match)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    search_id, result.id, position, result.score,
                    result.pipeline.semantic if result.pipeline else 0.0,
                    result.pipeline.bm25 if result.pipeline else 0.0,
                    result.pipeline.rrf if result.pipeline else 0.0,
                    role_match,
                ),
            )
            results_shown += 1

            # Simulate clicks
            is_target = result.id == target_id
            position_bias = 1.0 / position
            info_type = (result.info_type or "").lower()

            if is_target:
                # Popular target page: high click probability scaled by views
                click_prob = 0.5 + 0.4 * view_weight  # 0.5 - 0.9
            elif info_type == "temaside":
                # Temasider are high-value overview pages users often click
                click_prob = 0.35 * position_bias
            elif info_type == "retningslinje":
                # Retningslinjer are authoritative and frequently clicked
                click_prob = 0.30 * position_bias
            else:
                # Other content: lower click probability
                click_prob = 0.15 * position_bias

            if random.random() < click_prob:
                cursor.execute(
                    "INSERT INTO click_logs (search_id, content_id, position, timestamp) VALUES (%s, %s, %s, %s)",
                    (search_id, result.id, position, timestamp),
                )
                clicks_created += 1

        # Commit every 20 searches
        if (i + 1) % 20 == 0:
            conn.commit()
            print(f"  [{i + 1}/{len(pages)}] {searches_created} searches, {clicks_created} clicks")

    conn.commit()
    cursor.close()

    print("\n  Generated:")
    print(f"    {searches_created} searches (skipped {skipped})")
    print(f"    {results_shown} results shown")
    print(f"    {clicks_created} clicks")


def verify_data(conn):
    """Verify generated data."""
    cursor = conn.cursor(dictionary=True)

    print("\n" + "=" * 60)
    print("  Training Data Verification")
    print("=" * 60)

    cursor.execute("SELECT COUNT(*) as count FROM search_logs")
    search_count = cursor.fetchone()["count"]
    print(f"  Searches: {search_count}")

    cursor.execute("SELECT COUNT(*) as count FROM search_results_shown")
    shown_count = cursor.fetchone()["count"]
    print(f"  Results shown: {shown_count}")

    cursor.execute("SELECT COUNT(*) as count FROM click_logs")
    click_count = cursor.fetchone()["count"]
    print(f"  Clicks: {click_count}")

    if shown_count > 0:
        ctr = (click_count / shown_count) * 100
        print(f"  Overall CTR: {ctr:.1f}%")

    # Count unique search groups with clicks (what LTR training needs)
    cursor.execute(
        """
        SELECT COUNT(DISTINCT sl.search_id) as count
        FROM search_logs sl
        INNER JOIN click_logs cl ON sl.search_id = cl.search_id
        INNER JOIN search_results_shown srs ON sl.search_id = srs.search_id
        GROUP BY sl.search_id
        HAVING COUNT(DISTINCT srs.id) >= 5
        """
    )
    groups = cursor.fetchall()
    print(f"  Training groups (searches with clicks + 5+ results): {len(groups)}")

    print("=" * 60)

    if len(groups) >= 20:
        print("\n  Sufficient data for ranking model training!")
        print("  Run: python scripts/train/train_ranking_model.py")
    else:
        print("\n  Need more data. Try increasing --top.")

    cursor.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training data from page view CSV + real searches"
    )
    parser.add_argument(
        "--csv",
        default=str(project_root / "scripts" / "data" / "importing" / "helsedir_page_view.csv"),
        help="Path to page view CSV file",
    )
    parser.add_argument("--top", type=int, default=200, help="Use top N most viewed pages (default: 200)")
    parser.add_argument("--k", type=int, default=10, help="Results per search (default: 10)")
    parser.add_argument("--clear", action="store_true", help="Clear existing logs first")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Generate Training Data for Ranking Model")
    print("=" * 60)

    # Parse CSV
    print("\nParsing CSV...")
    csv_rows = parse_page_view_csv(args.csv)
    print(f"  {len(csv_rows)} pages with views")

    # Match to content
    print("Matching to content database...")
    matched = match_csv_to_content(csv_rows)
    print(f"  {len(matched)} pages matched to content")

    if not matched:
        print("\nNo pages matched. Check that content is loaded.")
        return

    if args.top <= 0:
        print("\nError: --top must be a positive integer.")
        return

    # Take top N by views
    matched.sort(key=lambda x: -x["views"])
    pages = matched[:args.top]
    if not pages:
        print("\nNo pages after slicing. Check --top value.")
        return
    print(f"\nUsing top {len(pages)} pages (views: {pages[0]['views']:,} - {pages[-1]['views']:,})")

    print("\nConfiguration:")
    print(f"  Results per search: {args.k}")
    print(f"  Clear existing: {args.clear}")

    # Connect to database
    conn = get_connection()

    try:
        if args.clear:
            print("\nClearing existing logs...")
            cursor = conn.cursor()
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            cursor.execute("TRUNCATE TABLE click_logs")
            cursor.execute("TRUNCATE TABLE search_results_shown")
            cursor.execute("TRUNCATE TABLE search_logs")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            conn.commit()
            cursor.close()
            print("  Existing logs cleared")

        print("\nGenerating training data (running real searches)...")
        generate_training_data(conn, pages, k=args.k)
        verify_data(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
