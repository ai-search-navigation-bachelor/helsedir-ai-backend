#!/usr/bin/env python3
"""
Test import from Helsedirektoratet API without saving to database.

Prints all fields to inspect what the API returns.

Usage:
    python scripts/data/test_import.py                    # Test 3 items
    python scripts/data/test_import.py --count 5          # Test 5 items
    python scripts/data/test_import.py --type retningslinje  # Test specific type
    python scripts/data/test_import.py --query diabetes   # Test specific query
"""

import argparse
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)


def print_item(item: dict, index: int, detailed: dict = None):
    """Print all fields of an item."""
    print(f"\n{'='*60}")
    print(f"ITEM {index}")
    print("="*60)

    print("\n--- FROM SEARCH (basic) ---")
    for key, value in sorted(item.items()):
        if key in ("tekst",) and value and len(str(value)) > 200:
            print(f"  {key}: {str(value)[:200]}... ({len(str(value))} chars)")
        elif isinstance(value, (list, dict)):
            print(f"  {key}: {json.dumps(value, ensure_ascii=False, indent=4)[:500]}")
        else:
            print(f"  {key}: {value}")

    if detailed:
        print("\n--- FROM DETAILED (get_infobit_by_id) ---")
        for key, value in sorted(detailed.items()):
            if key in item:
                continue  # Skip duplicates
            if key in ("tekst",) and value and len(str(value)) > 200:
                print(f"  {key}: {str(value)[:200]}... ({len(str(value))} chars)")
            elif isinstance(value, (list, dict)):
                print(f"  {key}: {json.dumps(value, ensure_ascii=False, indent=4)[:500]}")
            else:
                print(f"  {key}: {value}")

        # Highlight key fields
        print("\n--- KEY FIELDS ---")
        print(f"  koder: {detailed.get('koder')}")
        print(f"  maalgruppe: {detailed.get('maalgruppe')}")
        print(f"  links: {detailed.get('links')}")


def main():
    parser = argparse.ArgumentParser(description="Test import without saving to database")
    parser.add_argument("--count", type=int, default=3, help="Number of items to test (default: 3)")
    parser.add_argument("--type", type=str, help="Filter by info type (e.g., retningslinje)")
    parser.add_argument("--query", type=str, default="diabetes", help="Search query (default: diabetes)")
    parser.add_argument("--no-details", action="store_true", help="Skip detailed fetch")
    parser.add_argument("--with-koder", action="store_true", help="Only show items that have koder")

    args = parser.parse_args()

    # Handle conflicting flags
    if args.with_koder and args.no_details:
        print("WARNING: --with-koder requires details to be fetched. Ignoring --no-details.")
        args.no_details = False

    print("="*60)
    print("TEST IMPORT (no database save)")
    print("="*60)
    print(f"Query: {args.query}")
    print(f"Type filter: {args.type or 'None'}")
    print(f"Count: {args.count}")
    print(f"Fetch details: {not args.no_details}")
    print(f"Filter for koder: {args.with_koder}")

    try:
        # Build filter
        filter_query = None
        if args.type:
            filter_query = f"infoType eq '{args.type}'"
            print(f"Filter: {filter_query}")

        # Fetch basic info
        print("\n>>> Fetching basic info...")
        basic_results = helsedir_api_service.search_infobits(
            query_text=args.query,
            filter_query=filter_query,
            get_full_infobits=False,
            timeout=30.0,
        )
        print(f"    Got {len(basic_results)} results")

        # Fetch full info
        print(">>> Fetching full info...")
        full_results = helsedir_api_service.search_infobits(
            query_text=args.query,
            filter_query=filter_query,
            get_full_infobits=True,
            timeout=30.0,
        )
        print(f"    Got {len(full_results)} results")

        # Create lookup for basic info
        basic_by_id = {item.get("id"): item for item in basic_results}

        # Process items
        found_count = 0
        for item in full_results:
            if found_count >= args.count:
                break

            item_id = item.get("id")

            # Merge infoType from basic
            if item_id in basic_by_id:
                item["infoType"] = basic_by_id[item_id].get("infoType")

            # Fetch detailed info
            detailed = None
            if not args.no_details and item_id:
                try:
                    print(f"\n>>> Fetching details for {item_id}...")
                    detailed = helsedir_api_service.get_infobit_by_id(item_id, timeout=15.0)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

            # Skip if --with-koder and no koder
            if args.with_koder:
                koder = detailed.get("koder") if detailed else None
                if not koder:
                    print(f"    Skipping (no koder)")
                    continue

            found_count += 1
            print_item(item, found_count, detailed)

        print(f"\n{'='*60}")
        print(f"Done! Found {found_count} items" + (" with koder" if args.with_koder else ""))
        print("="*60)

    except HelseDirectorateAPIError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
