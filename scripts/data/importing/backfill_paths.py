#!/usr/bin/env python3
"""
Backfill helsedirektoratet.no path for content items where path IS NULL.

Fetches the url field from the API for each item and updates path in DB.

Usage:
    python scripts/data/importing/backfill_paths.py           # Backfill all NULL paths
    python scripts/data/importing/backfill_paths.py --limit 100  # Test with 100 items
    python scripts/data/importing/backfill_paths.py --dry-run    # Show what would be updated
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.services.external.helsedir_api_service import helsedir_api_service
from app.services.repositories.base import db_pool


def get_content_with_null_path(limit: int = 0):
    """Fetch all content IDs where path IS NULL (excluding temasider)."""
    conn = db_pool.get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT id FROM content WHERE path IS NULL AND info_type != 'temaside'"
        if limit > 0:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        return [row['id'] for row in cursor.fetchall()]
    except Exception as e:
        print(f"ERROR fetching NULL-path content: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def batch_update_paths(updates: list) -> int:
    """
    Update paths for multiple content items in a single connection.

    Args:
        updates: List of (content_id, path) tuples

    Returns:
        Number of rows updated
    """
    if not updates:
        return 0
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not get DB connection for batch update")
        return 0
    try:
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE content SET path = %s WHERE id = %s",
            [(path, content_id) for content_id, path in updates]
        )
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        print(f"ERROR batch updating paths: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        conn.close()


def fetch_path(content_id: str):
    """
    Fetch helsedirektoratet.no path from API for a single content item.

    Returns:
        (content_id, path) where path is None if not found
    """
    detailed = helsedir_api_service.get_infobit_by_id(content_id, timeout=15.0)
    api_url = detailed.get("url", "")
    if api_url and "helsedirektoratet.no" in api_url:
        return content_id, urlparse(api_url).path.rstrip('/') or '/'
    return content_id, None


def main():
    parser = argparse.ArgumentParser(description="Backfill NULL paths from Helsedir API")
    parser.add_argument("--limit", type=int, default=0, help="Max items to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without writing")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default: 10)")
    args = parser.parse_args()

    print("=" * 50)
    print("BACKFILL PATHS")
    print("=" * 50)
    if args.dry_run:
        print("DRY RUN - no changes will be saved")

    print("\nFetching content with NULL path...", end=" ", flush=True)
    content_ids = get_content_with_null_path(args.limit)
    print(f"found {len(content_ids)} items")

    if not content_ids:
        print("Nothing to backfill.")
        return

    no_url = 0
    failed = 0
    updates = []  # Collect all (content_id, path) pairs before writing to DB

    print(f"\nFetching paths with {args.workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_path, cid): cid for cid in content_ids}
        for i, future in enumerate(as_completed(futures), 1):
            content_id = futures[future]
            try:
                cid, path = future.result()
                if path is None:
                    no_url += 1
                else:
                    updates.append((cid, path))
                    if args.dry_run:
                        print(f"  {cid} -> {path}")
            except Exception as e:
                failed += 1
                print(f"  FAILED {content_id}: {e}")

            if i % 100 == 0:
                print(f"  [{i}/{len(content_ids)}] Found paths: {len(updates)}, No URL: {no_url}, Failed: {failed}")

    print(f"\nFetched {len(updates)} paths from API")

    if not args.dry_run and updates:
        print(f"Saving {len(updates)} paths to database...", end=" ", flush=True)
        saved = batch_update_paths(updates)
        print(f"done ({saved} rows updated)")

    print(f"\n{'='*50}")
    print(f"Paths found:  {len(updates)}")
    print(f"No URL:       {no_url}  (API returned no helsedirektoratet.no URL)")
    print(f"Failed:       {failed}")


if __name__ == "__main__":
    main()
