#!/usr/bin/env python3
"""
Reduce content database to a target number of items while maintaining
proportional representation across all info types.

Usage:
    python scripts/data/reduce_content.py --target 400
    python scripts/data/reduce_content.py --target 400 --dry-run  # Preview only
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Reduce content database to target size with proportional type distribution"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=400,
        help="Target number of items to keep (default: 400)",
    )
    parser.add_argument(
        "--min-per-type",
        type=int,
        default=2,
        help="Minimum items to keep per type (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    from app.services.data.database_service import database_service

    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Load all content
    print("Loading content from database...")
    items = database_service.get_all_content()
    total = len(items)
    print(f"Found {total} items")

    if total <= args.target:
        print(f"Already at or below target ({args.target}). Nothing to do.")
        sys.exit(0)

    # Group by type
    by_type = {}
    for item in items:
        t = item.get('info_type', 'unknown')
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(item)

    print(f"Types: {len(by_type)}")

    # Calculate proportional distribution
    min_per_type = args.min_per_type
    remaining = args.target - (min_per_type * len(by_type))

    keep_counts = {}
    for t, items_of_type in by_type.items():
        proportion = len(items_of_type) / total
        extra = max(0, int(remaining * proportion))
        keep_counts[t] = min(len(items_of_type), min_per_type + extra)

    # Adjust to hit target exactly
    current_total = sum(keep_counts.values())
    diff = args.target - current_total
    sorted_types = sorted(by_type.keys(), key=lambda t: len(by_type[t]), reverse=True)

    for t in sorted_types:
        if diff == 0:
            break
        if diff > 0 and keep_counts[t] < len(by_type[t]):
            add = min(diff, len(by_type[t]) - keep_counts[t])
            keep_counts[t] += add
            diff -= add
        elif diff < 0 and keep_counts[t] > min_per_type:
            remove = min(-diff, keep_counts[t] - min_per_type)
            keep_counts[t] -= remove
            diff += remove

    # Score items by content richness (prefer items with more text and links)
    def score_item(item):
        tekst_len = len(item.get('tekst') or '')
        links_raw = item.get('links')
        if links_raw:
            links = json.loads(links_raw) if isinstance(links_raw, str) else links_raw
            links_count = len(links) if links else 0
        else:
            links_count = 0
        return tekst_len + (links_count * 100)

    # Select which items to keep (highest scores)
    keep_ids = set()
    delete_ids = set()

    print("\nSelection by type:")
    print("=" * 60)

    for t in sorted_types:
        items_of_type = by_type[t]
        keep_count = keep_counts[t]

        # Sort by score (highest first)
        sorted_items = sorted(items_of_type, key=score_item, reverse=True)

        # Keep the best ones
        for item in sorted_items[:keep_count]:
            keep_ids.add(item['id'])
        for item in sorted_items[keep_count:]:
            delete_ids.add(item['id'])

        print(f"{t}: keeping {keep_count}/{len(items_of_type)}")

    print("=" * 60)
    print(f"Total to keep: {len(keep_ids)}")
    print(f"Total to delete: {len(delete_ids)}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Confirm
    print("\n" + "!" * 60)
    response = input("Type 'DELETE' to confirm deletion: ")
    if response != 'DELETE':
        print("Aborted.")
        sys.exit(0)

    # Delete items
    print("\nDeleting items...")
    deleted = 0
    conn = database_service._get_connection()
    if not conn:
        print("Error: Could not get database connection")
        sys.exit(1)

    try:
        cursor = conn.cursor()
        for content_id in delete_ids:
            cursor.execute("DELETE FROM content WHERE id = %s", (content_id,))
            deleted += 1
            if deleted % 100 == 0:
                print(f"  Deleted {deleted}/{len(delete_ids)}...")
        conn.commit()
        print(f"\nDeleted {deleted} items.")
        print(f"Remaining: {len(keep_ids)} items.")
    except Exception as e:
        print(f"Error during deletion: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
