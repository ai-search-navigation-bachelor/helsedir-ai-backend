#!/usr/bin/env python3
"""
Backfill anbefaling_details table for existing anbefaling content.

Fetches full details from /innhold/anbefalinger/{id} API and populates
the anbefaling-specific fields (praktisk, rasjonale, etc.).

Usage:
    python scripts/data/backfill_anbefaling_details.py              # Backfill all anbefalinger
    python scripts/data/backfill_anbefaling_details.py --limit 10   # Test with 10 items
    python scripts/data/backfill_anbefaling_details.py --force      # Re-fetch all (ignore existing)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def fetch_anbefaling_details(api_key: str, content_id: str) -> Dict[str, Any]:
    """
    Fetch full anbefaling details from API.

    Args:
        api_key: Helsedir API key
        content_id: Content ID

    Returns:
        Full anbefaling data dict
    """
    import httpx

    try:
        response = httpx.get(
            f"https://api.helsedirektoratet.no/innhold/anbefalinger/{content_id}",
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=10.0,
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠️  API error for {content_id}: {response.status_code}")
            return {}

    except Exception as e:
        print(f"  ⚠️  Request error for {content_id}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Backfill anbefaling_details table for existing content"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of items to backfill (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all (ignore existing details)",
    )
    args = parser.parse_args()

    from app.config import settings
    from app.services.data.database_service import database_service

    # Check API key
    if not settings.helsedir_api_key:
        print("Error: HELSEDIR_API_KEY not configured in .env")
        sys.exit(1)

    # Check database
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    print("=" * 60)
    print("BACKFILL ANBEFALING DETAILS")
    print("=" * 60)

    # Get all anbefaling content
    print("\nFetching anbefaling content from database...")
    all_content = database_service.get_all_content()
    anbefalinger = [
        item for item in all_content
        if item.get("info_type") == "anbefaling"
    ]

    print(f"Found {len(anbefalinger)} anbefaling items in database")

    if not anbefalinger:
        print("No anbefalinger found to backfill")
        return

    # Filter out items that already have details (unless --force)
    if not args.force:
        items_to_backfill = [
            item for item in anbefalinger
            if not item.get("praktisk")  # If praktisk is NULL, needs backfill
        ]
        print(f"  {len(items_to_backfill)} need backfill (missing details)")
        print(f"  {len(anbefalinger) - len(items_to_backfill)} already have details")
    else:
        items_to_backfill = anbefalinger
        print(f"  --force specified, will re-fetch all {len(items_to_backfill)}")

    if not items_to_backfill:
        print("\n✓ All anbefalinger already have details!")
        print("  Use --force to re-fetch all")
        return

    # Apply limit if specified
    if args.limit > 0:
        items_to_backfill = items_to_backfill[:args.limit]
        print(f"\n--limit specified, processing {len(items_to_backfill)} items")

    print(f"\nBackfilling {len(items_to_backfill)} items...")
    print(f"Estimated time: ~{len(items_to_backfill) * 0.5:.1f} seconds")

    # Confirm
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(0)

    # Backfill
    print("\nProcessing...")
    print("-" * 60)

    success = 0
    failed = 0
    skipped = 0

    for i, item in enumerate(items_to_backfill, 1):
        content_id = item["id"]
        tittel = item.get("tittel", "")[:50]

        print(f"[{i}/{len(items_to_backfill)}] {tittel}...", end=" ")

        # Fetch full details from API
        full_data = fetch_anbefaling_details(settings.helsedir_api_key, content_id)

        if not full_data:
            print("FAILED (API error)")
            failed += 1
            continue

        # Build content dict with all fields
        content_with_details = {
            "id": content_id,
            "tittel": item.get("tittel"),
            "tekst": item.get("tekst"),
            "info_type": "anbefaling",
            "koder": item.get("koder"),
            "maalgruppe": item.get("maalgruppe"),
            "links": item.get("links"),
            "path": item.get("path"),
            "data": full_data.get("data", {}),
        }

        # Cache using repository (which now handles anbefaling_details)
        if database_service.cache_content(content_with_details):
            print("OK")
            success += 1
        else:
            print("FAILED (DB error)")
            failed += 1

        # Rate limiting (be nice to API)
        if i < len(items_to_backfill):
            time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(items_to_backfill)}")

    if success > 0:
        print(f"\n✓ Successfully backfilled {success} anbefaling items!")


if __name__ == "__main__":
    main()
