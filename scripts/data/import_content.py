#!/usr/bin/env python3
"""
Import content from Helsedirektoratet API to database.

Usage:
    python scripts/import_content.py
    python scripts/import_content.py --search-terms diabetes,kreft,adhd
    python scripts/import_content.py --alphabet  # Search using alphabet for broader coverage
"""

import argparse
import sys
import os

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)
from app.services.data.database_service import database_service


# Default search terms for broad medical coverage
DEFAULT_SEARCH_TERMS = [
    # Common conditions
    "diabetes",
    "kreft",
    "adhd",
    "depresjon",
    "angst",
    "astma",
    "kols",
    "demens",
    "epilepsi",
    "allergi",

    # Organ systems
    #"hjerte",
    #"lunge",
    #"nyre",
    #"lever",
    #"mage",
    #"tarm",
    #"hud",
    #"øye",
    #"øre",

    # Life stages / groups
    #"barn",
    #"eldre",
    #"gravid",
    #"ungdom",
    #"spedbarn",

    # Mental health
    #"psykisk",
    #"rus",
    #"avhengighet",
    #"selvmord",
    #"spiseforstyrrelser",

    # Common treatments
    #"legemiddel",
    #"antibiotika",
    #"vaksin",
    #"kirurgi",
    #"rehabilitering",

    # Healthcare settings
    #"fastlege",
    #"sykehus",
    #"legevakt",
    #"helsestasjon",
    #"sykehjem",

    # Other common topics
    #"smitte",
    #"infeksjon",
    #"kosthold",
    #"ernæring",
    #"fysisk aktivitet",
    #"søvn",
    #"smerter",
    #"blodtrykk",
    #"kolesterol",
]

# Alphabet for broader coverage
ALPHABET_SEARCH_TERMS = list("abcdefghijklmnopqrstuvwxyzæøå")


def fetch_content(search_terms: list, verbose: bool = True) -> dict:
    """
    Fetch content from Helsedir API using search terms.

    Args:
        search_terms: List of search terms to use
        verbose: Whether to print progress

    Returns:
        Dictionary of content items keyed by ID (deduped)
    """
    all_content = {}

    for i, term in enumerate(search_terms, 1):
        if verbose:
            print(f"[{i}/{len(search_terms)}] Searching for: {term}...", end=" ")

        try:
            # First get basic info (includes correct infoType)
            basic_results = helsedir_api_service.search_infobits(
                query_text=term,
                get_full_infobits=False,
                timeout=30.0,
            )

            # Then get full info (includes full text content)
            full_results = helsedir_api_service.search_infobits(
                query_text=term,
                get_full_infobits=True,
                timeout=30.0,
            )

            # Create lookup for basic info by id
            basic_by_id = {item.get("id"): item for item in basic_results}

            # Merge: use full results but add infoType from basic
            results = []
            for item in full_results:
                item_id = item.get("id")
                if item_id in basic_by_id:
                    item["infoType"] = basic_by_id[item_id].get("infoType")
                results.append(item)

            new_count = 0
            for item in results:
                content_id = item.get("id")
                if content_id and content_id not in all_content:
                    all_content[content_id] = item
                    new_count += 1

            if verbose:
                print(f"Found {len(results)} results ({new_count} new)")

        except HelseDirectorateAPIError as e:
            if verbose:
                print(f"Error: {e}")
            continue

    return all_content


def save_to_database(content_items: dict, verbose: bool = True) -> int:
    """
    Save content items to database.

    Args:
        content_items: Dictionary of content items keyed by ID
        verbose: Whether to print progress

    Returns:
        Number of items saved
    """
    if not content_items:
        return 0

    contents = list(content_items.values())

    if verbose:
        print(f"\nSaving {len(contents)} items to database...")

    saved = database_service.cache_content_batch(contents)

    if verbose:
        print(f"Saved {saved} items")

    return saved


def print_statistics(verbose: bool = True):
    """Print statistics about database content."""
    if not verbose:
        return

    print("\n" + "=" * 50)
    print("DATABASE STATISTICS")
    print("=" * 50)

    total = database_service.get_content_count()
    print(f"\nTotal content items: {total}")

    print("\nContent by type:")
    stats = database_service.get_content_stats_by_type()
    for stat in stats:
        info_type = stat.get("info_type") or "(no type)"
        count = stat.get("count", 0)
        print(f"  {info_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Import content from Helsedirektoratet API to database"
    )
    parser.add_argument(
        "--search-terms",
        type=str,
        help="Comma-separated list of search terms",
    )
    parser.add_argument(
        "--alphabet",
        action="store_true",
        help="Use alphabet search (a-z, æøå) for broader coverage",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Determine search terms to use
    if args.search_terms:
        search_terms = [t.strip() for t in args.search_terms.split(",")]
    elif args.alphabet:
        search_terms = ALPHABET_SEARCH_TERMS
    else:
        search_terms = DEFAULT_SEARCH_TERMS

    if verbose:
        print("=" * 50)
        print("HELSEDIR CONTENT IMPORT")
        print("=" * 50)
        print(f"\nUsing {len(search_terms)} search terms")

    # Check database connection
    if not database_service.is_connected():
        print("ERROR: Could not connect to database")
        print("Make sure MySQL is running and .env is configured correctly")
        sys.exit(1)

    # Fetch content
    content_items = fetch_content(search_terms, verbose=verbose)

    if verbose:
        print(f"\nTotal unique content items found: {len(content_items)}")

    # Save to database
    if content_items:
        save_to_database(content_items, verbose=verbose)
    else:
        if verbose:
            print("No content to save")

    # Print statistics
    print_statistics(verbose=verbose)


if __name__ == "__main__":
    main()
