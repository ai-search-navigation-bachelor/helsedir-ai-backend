#!/usr/bin/env python3
"""
Import content from Helsedirektoratet API to database.

Note: Does not import 'temaside' type (only exists locally, not in API).

Usage with --info-type (fetch specific content type):
    python scripts/data/import_content.py --info-type anbefaling --limit 2000   # Fetch 2000 anbefalinger
    python scripts/data/import_content.py --info-type anbefaling --limit 0      # Fetch ALL available anbefalinger
    python scripts/data/import_content.py --info-type retningslinje --limit 500 # Fetch 500 retningslinjer
    python scripts/data/import_content.py --info-type anbefaling                # Fetch 50 anbefalinger (default)

Usage with --by-type (fetches directly by info_type, NO search terms used):
    python scripts/data/import_content.py --by-type                # Fetch evenly distributed: 1000 total / 23 API types = ~43 per type
    python scripts/data/import_content.py --by-type --target 1200  # Fetch 1200 total, evenly distributed (~52 per type)
    python scripts/data/import_content.py --by-type --per-type 30  # Fetch 30 items per type (690 total)

Usage with search terms (iterates through search terms until target reached):
    python scripts/data/import_content.py                          # Default: 1000 items using 10 search terms
    python scripts/data/import_content.py --extended               # Use extended search terms (~120 terms)
    python scripts/data/import_content.py --target 2000            # Fetch up to 2000 items using search
    python scripts/data/import_content.py --alphabet               # Search using alphabet (a-z, æøå)

Other options:
    python scripts/data/import_content.py --no-links               # Skip fetching links (much faster)
    python scripts/data/import_content.py --skip-existing          # Skip detail fetch for existing items (saves API calls)
    python scripts/data/import_content.py --info-type anbefaling --limit 2000 --skip-existing  # Update existing anbefalinger

Performance tips:
    - Use --skip-existing on re-runs to avoid unnecessary API calls for items you already have
    - Use --no-links for fastest import (but you'll miss koder, maalgruppe, and links data)
    - Combine both for maximum speed: --skip-existing --no-links
"""

import argparse
import sys
import os
import re
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)
from app.services.data.database_service import database_service
from app.constants import ALLOWED_INFO_TYPES, CATEGORY_INFO


def extract_content_id_from_href(href: str) -> Optional[str]:
    """Extract content ID from Helsedir API URL."""
    if not href:
        return None
    pattern = r'/innhold/[^/]+/([0-9]{4}-[0-9]{4}-[a-f0-9\-]+)'
    match = re.search(pattern, href, re.IGNORECASE)
    return match.group(1) if match else None


def process_links_at_import(links: List[Dict], existing_ids: set) -> List[Dict]:
    """
    Process links during import to new format.

    For all link types (forelder, root, barn, publikasjon):
    - Extract ID from href
    - If exists in DB: use 'id' field
    - If not: use 'href' field
    - Remove 'strukturId'
    """
    if not links:
        return []

    processed = []
    for link in links:
        if not isinstance(link, dict):
            continue

        rel = link.get('rel', '')
        new_link = {
            'rel': rel,
            'type': link.get('type', ''),
        }

        if link.get('tittel'):
            new_link['tittel'] = link['tittel']

        # Resolve internal links for all types
        href = link.get('href', '')
        extracted_id = extract_content_id_from_href(href)

        if extracted_id and extracted_id in existing_ids:
            new_link['id'] = extracted_id  # Internal link
        else:
            new_link['href'] = href  # External link or no valid ID

        processed.append(new_link)

    return processed

# Info types to fetch from API (exclude temaside as it only exists locally)
API_INFO_TYPES = [t for t in ALLOWED_INFO_TYPES if t != "temaside"]


# Default search terms (quick import)
DEFAULT_SEARCH_TERMS = [
    "diabetes", "kreft", "adhd", "depresjon", "angst",
    "astma", "kols", "demens", "epilepsi", "allergi",
]

# Extended search terms for comprehensive import
EXTENDED_SEARCH_TERMS = [
    # Diseases and conditions
    "diabetes", "kreft", "hjerte", "lunge", "psykisk", "demens", "astma", "kols",
    "hypertensjon", "allergi", "infeksjon", "smerte", "depresjon", "angst", "rus",
    "alkohol", "overvekt", "ernæring", "søvn", "stress", "migrene", "hodepine",
    "rygg", "nakke", "ledd", "artritt", "osteoporose", "fibromyalgi", "ms",
    "parkinson", "alzheimer", "epilepsi", "cerebral parese",

    # Cancer types
    "brystkreft", "lungekreft", "prostatakreft", "tarmkreft", "hudkreft",
    "blodkreft", "leukemi", "lymfekreft", "hjernesvulst",

    # Cardiovascular
    "hjerteinfarkt", "hjerneslag", "hjertesvikt", "arytmi", "atrieflimmer",
    "blodpropp", "trombose", "angina",

    # Respiratory
    "pneumoni", "bronkitt", "tuberkulose", "covid", "influensa", "lungebetennelse",

    # Digestive
    "mage", "tarm", "lever", "gallestein", "pankreatitt", "hepatitt",
    "crohn", "ulcerøs kolitt", "ibs", "reflux", "magesår", "cøliaki",

    # Endocrine
    "skjoldbruskkjertel", "stoffskifte", "hormon", "insulin",

    # Kidney/Urinary
    "nyre", "nyresvikt", "dialyse", "urinveisinfeksjon", "inkontinens", "prostata",

    # Skin
    "hud", "eksem", "psoriasis", "akne", "melanom", "utslett", "sår",

    # Eyes/Ears
    "øye", "syn", "grå stær", "grønn stær", "øre", "hørsel", "tinnitus", "svimmelhet",

    # Mental health
    "adhd", "autisme", "bipolar", "schizofreni", "ptsd", "spiseforstyrrelser",
    "anoreksi", "selvskading", "selvmord", "psykose", "ocd", "panikk",

    # Women's health
    "graviditet", "fødsel", "amming", "menstruasjon", "overgangsalder",
    "endometriose", "pcos", "prevensjon", "svangerskapsforgiftning",

    # Children
    "barn", "spedbarn", "nyfødt", "prematur", "barnevaksinasjon", "vannkopper",
    "meslinger", "kikhoste", "dysleksi",

    # Elderly
    "eldre", "geriatri", "aldring", "fallforebygging", "hoftebrudd", "sykehjem",

    # Infections
    "bakterie", "virus", "sepsis", "mrsa", "smittevern", "hiv", "klamydia", "borreliose",

    # Medical specialties
    "pediatri", "kirurgi", "ortopedi", "nevrologi", "kardiologi", "onkologi",
    "gynekologi", "urologi", "dermatologi", "psykiatri", "akuttmedisin",

    # Treatments
    "behandling", "medisin", "terapi", "rehabilitering", "forebygging",
    "vaksinasjon", "operasjon", "strålebehandling", "cellegift", "fysioterapi",

    # Medications
    "legemiddel", "antibiotika", "smertestillende", "paracetamol", "morfin",
    "antidepressiva", "blodfortynnende", "insulin", "bivirkninger",

    # Healthcare processes
    "utredning", "diagnostikk", "oppfølging", "henvisning", "innleggelse",
    "poliklinikk", "screening", "mammografi", "koloskopi", "blodprøve",

    # Healthcare settings
    "fastlege", "sykehus", "legevakt", "helsestasjon", "apotek",

    # Lifestyle
    "røyking", "fysisk aktivitet", "trening", "kosthold", "vekt", "kolesterol",
    "blodtrykk", "blodsukker", "vitamin",

    # Guidelines
    "retningslinje", "veileder", "pakkeforløp", "pasientsikkerhet",
]

# Alphabet for broader coverage
ALPHABET_SEARCH_TERMS = list("abcdefghijklmnopqrstuvwxyzæøå")


def fetch_content_by_type(verbose: bool = True, fetch_links: bool = True, target_per_type: int = 50, specific_type: Optional[str] = None, skip_existing: bool = False, existing_ids: Optional[set] = None) -> dict:
    """
    Fetch content from Helsedir API by filtering on each info type.

    This ensures balanced coverage across all content types from the API.
    Note: Excludes 'temaside' as it only exists locally, not in the API.

    Args:
        verbose: Whether to print progress
        fetch_links: Whether to fetch detailed info including links
        target_per_type: Target number of items per info type (0 = all available)
        specific_type: If set, only fetch this specific info_type
        skip_existing: If True, skip fetching details for items already in database
        existing_ids: Set of content IDs that already exist in database

    Returns:
        Dictionary of content items keyed by ID (deduped)
    """
    all_content = {}
    type_counts = defaultdict(int)
    skipped_count = 0

    if existing_ids is None:
        existing_ids = set()

    # If specific type is requested, only fetch that one
    types_to_fetch = [specific_type] if specific_type else API_INFO_TYPES

    for i, info_type in enumerate(types_to_fetch, 1):
        display_name = CATEGORY_INFO.get(info_type, info_type)

        if verbose:
            print(f"\n[{i}/{len(types_to_fetch)}] Fetching: {display_name} ({info_type})...", flush=True)

        try:
            # Filter by info type
            filter_query = f"infoType eq '{info_type}'"

            # Get basic info first (for correct infoType)
            # Use empty query with filter - API requires either query or filter
            if verbose:
                print("    basic...", end=" ", flush=True)
            basic_results = helsedir_api_service.search_infobits(
                query_text=None,
                filter_query=filter_query,
                get_full_infobits=False,
                timeout=30.0,
            )

            # Get full info
            if verbose:
                print("full...", end=" ", flush=True)
            full_results = helsedir_api_service.search_infobits(
                query_text=None,
                filter_query=filter_query,
                get_full_infobits=True,
                timeout=30.0,
            )

            # Create lookup for basic info
            basic_by_id = {item.get("id"): item for item in basic_results}

            # Merge and limit to target (if target_per_type is 0, take all)
            new_count = 0
            new_items = []
            skipped_existing = 0

            items_to_process = full_results if target_per_type == 0 else full_results[:target_per_type]

            for item in items_to_process:
                item_id = item.get("id")
                if item_id and item_id not in all_content:
                    # Add infoType from basic
                    if item_id in basic_by_id:
                        item["infoType"] = basic_by_id[item_id].get("infoType")
                    all_content[item_id] = item

                    # Only add to new_items if not skipping existing
                    if skip_existing and item_id in existing_ids:
                        skipped_existing += 1
                        skipped_count += 1
                    else:
                        new_items.append((item_id, item))

                    new_count += 1
                    type_counts[info_type] += 1

            if verbose:
                limit_msg = "all" if target_per_type == 0 else f"first {target_per_type}"
                skip_msg = f", skipped {skipped_existing} existing" if skip_existing and skipped_existing > 0 else ""
                print(f"found {len(full_results)}, added {new_count} ({limit_msg}){skip_msg} | Total: {len(all_content)}", flush=True)

            # Fetch details for new items
            if fetch_links and new_items:
                if verbose:
                    print(f"    Fetching details for {len(new_items)} items...", end=" ", flush=True)

                failed_count = 0
                for content_id, item in new_items:
                    try:
                        detailed = helsedir_api_service.get_infobit_by_id(content_id, timeout=15.0)
                        item["links"] = detailed.get("links")
                        if detailed.get("koder") is not None:
                            item["koder"] = detailed.get("koder")
                        if detailed.get("maalgruppe") is not None:
                            item["maalgruppe"] = detailed.get("maalgruppe")
                        time.sleep(0.1)
                    except Exception as err:
                        failed_count += 1
                        if verbose:
                            print(f"\n      WARN: Failed to fetch details for {content_id}: {err}", flush=True)
                        continue

                if verbose:
                    status = "done" if failed_count == 0 else f"done ({failed_count} failed)"
                    print(status, flush=True)

            time.sleep(0.2)

        except HelseDirectorateAPIError as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    if verbose:
        print(f"\n\n{'='*50}")
        print("CONTENT BY TYPE:")
        print("="*50)
        types_to_show = types_to_fetch if specific_type else API_INFO_TYPES
        for info_type in types_to_show:
            count = type_counts.get(info_type, 0)
            display = CATEGORY_INFO.get(info_type, info_type)
            print(f"  {display}: {count}")
        print(f"\nTotal: {len(all_content)}")
        if skip_existing and skipped_count > 0:
            print(f"Skipped existing: {skipped_count}")

    return all_content


def fetch_content(search_terms: list, verbose: bool = True, fetch_links: bool = True, target: int = 0, skip_existing: bool = False, existing_ids: Optional[set] = None) -> dict:
    """
    Fetch content from Helsedir API using search terms.

    Args:
        search_terms: List of search terms to use
        verbose: Whether to print progress
        fetch_links: Whether to fetch detailed info including links for each item
        target: Target number of items (0 = no limit)
        skip_existing: If True, skip fetching details for items already in database
        existing_ids: Set of content IDs that already exist in database

    Returns:
        Dictionary of content items keyed by ID (deduped)
    """
    all_content = {}
    total_detail_fetches = 0
    total_skipped = 0

    if existing_ids is None:
        existing_ids = set()

    for i, term in enumerate(search_terms, 1):
        # Check if we've reached target
        if target > 0 and len(all_content) >= target:
            if verbose:
                print(f"\n\nReached target of {target} items!")
            break

        if verbose:
            progress = f" [{len(all_content)}/{target}]" if target > 0 else ""
            print(f"\n[{i}/{len(search_terms)}]{progress} Searching: '{term}'...", end=" ", flush=True)

        try:
            # First get basic info (includes correct infoType)
            if verbose:
                print("basic...", end=" ", flush=True)
            basic_results = helsedir_api_service.search_infobits(
                query_text=term,
                get_full_infobits=False,
                timeout=30.0,
            )

            # Then get full info (includes full text content)
            if verbose:
                print("full...", end=" ", flush=True)
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
            new_items_this_term = []

            for item in results:
                # Stop if we've reached target
                if target > 0 and len(all_content) >= target:
                    break

                content_id = item.get("id")
                if content_id and content_id not in all_content:
                    all_content[content_id] = item
                    new_items_this_term.append((content_id, item))
                    new_count += 1

            if verbose:
                print(f"found {len(results)} results ({new_count} new) | Total: {len(all_content)}/{target}", flush=True)

            # Fetch detailed info with links for new items
            if fetch_links and new_items_this_term:
                # Filter out existing items if skip_existing is enabled
                items_to_fetch = []
                skipped_this_term = 0

                for content_id, item in new_items_this_term:
                    if skip_existing and content_id in existing_ids:
                        skipped_this_term += 1
                        total_skipped += 1
                    else:
                        items_to_fetch.append((content_id, item))

                if verbose:
                    skip_msg = f" (skipped {skipped_this_term} existing)" if skipped_this_term > 0 else ""
                    print(f"    Fetching details for {len(items_to_fetch)} items{skip_msg}:", flush=True)

                for j, (content_id, item) in enumerate(items_to_fetch):
                    try:
                        if verbose:
                            print(f"      [{j+1}/{len(items_to_fetch)}] ID: {content_id}", flush=True)

                        detailed = helsedir_api_service.get_infobit_by_id(content_id, timeout=15.0)

                        if verbose:
                            print(f"        Response keys: {list(detailed.keys()) if detailed else 'None'}", flush=True)
                            print(f"        links type: {type(detailed.get('links'))}, value: {detailed.get('links')}", flush=True)

                        # Use detailed response as authoritative source
                        item["links"] = detailed.get("links")
                        links_count = len(detailed.get("links") or [])

                        if detailed.get("koder") is not None:
                            item["koder"] = detailed.get("koder")
                        if detailed.get("maalgruppe") is not None:
                            item["maalgruppe"] = detailed.get("maalgruppe")
                        total_detail_fetches += 1

                        if verbose:
                            print(f"        -> Saved with {links_count} links", flush=True)

                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        if verbose:
                            print(f"        FAILED: {type(e).__name__}: {e}", flush=True)

            # Check if we've reached target AFTER fetching details
            if target > 0 and len(all_content) >= target:
                if verbose:
                    print(f"\n>>> Reached target of {target} items! <<<")
                break

            time.sleep(0.2)  # Rate limiting between searches

        except HelseDirectorateAPIError as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    if verbose:
        skip_msg = f", Skipped existing: {total_skipped}" if total_skipped > 0 else ""
        print(f"\n\nTotal items: {len(all_content)}, Detail fetches: {total_detail_fetches}{skip_msg}")

    return all_content


def save_to_database(content_items: dict, verbose: bool = True) -> int:
    """Save content items to database with processed links."""
    if not content_items:
        return 0

    contents = list(content_items.values())

    # Get existing content IDs for link resolution
    if verbose:
        print(f"\nLoading existing content IDs for link resolution...")
    existing_ids = set(database_service.get_all_content_ids())
    if verbose:
        print(f"  Found {len(existing_ids)} existing content items")

    # Process links for each content item
    if verbose:
        print(f"Processing links for {len(contents)} items...")

    for content in contents:
        if 'links' in content and content['links']:
            content['links'] = process_links_at_import(content['links'], existing_ids)

    if verbose:
        print(f"Saving {len(contents)} items to database...")

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
        "--extended",
        action="store_true",
        help="Use extended search terms (~120 terms) for comprehensive import",
    )
    parser.add_argument(
        "--alphabet",
        action="store_true",
        help="Use alphabet search (a-z, æøå) for broader coverage",
    )
    parser.add_argument(
        "--no-links",
        action="store_true",
        help="Skip fetching detailed info/links (much faster)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target number of items total. With --by-type: distributed evenly across types. Without --by-type: total items from search (default: 1000, 0 = no limit)",
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        help="Fetch by info type to ensure balanced coverage. Use --target to set total (distributed evenly) or --per-type to set per-type count",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Items per type when using --by-type. If not set, --target is distributed evenly across all types",
    )
    parser.add_argument(
        "--info-type",
        type=str,
        default=None,
        help="Fetch only a specific info_type (e.g., 'anbefaling', 'retningslinje'). Use with --limit to control count (0 = all available)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of items to fetch for the specified --info-type (0 = all available). If not set, defaults to 50",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip fetching details for items already in database (faster re-runs, saves API calls)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    fetch_links = not args.no_links

    # Check for --info-type mode
    if args.info_type:
        # Validate info_type
        if args.info_type not in API_INFO_TYPES:
            print(f"ERROR: Invalid info_type '{args.info_type}'")
            print(f"\nAvailable info types:")
            for info_type in API_INFO_TYPES:
                display = CATEGORY_INFO.get(info_type, info_type)
                print(f"  - {info_type:30s} ({display})")
            sys.exit(1)

        # Specific info_type mode - use --limit or default to 50
        specific_type = args.info_type
        limit = args.limit if args.limit is not None else 50
        mode = "specific_type"
    else:
        specific_type = None
        limit = None
        mode = "by_type" if args.by_type else "search"

    # Determine search terms to use (for search mode)
    if args.search_terms:
        search_terms = [t.strip() for t in args.search_terms.split(",")]
    elif args.alphabet:
        search_terms = ALPHABET_SEARCH_TERMS
    elif args.extended:
        search_terms = EXTENDED_SEARCH_TERMS
    else:
        search_terms = DEFAULT_SEARCH_TERMS

    target = args.target

    # Calculate per-type target when using --by-type
    if args.by_type:
        if args.per_type is not None:
            # Explicit per-type count
            per_type_target = args.per_type
            total_target = per_type_target * len(API_INFO_TYPES)
        else:
            # Distribute --target evenly across types (excluding temaside)
            total_target = target
            per_type_target = max(1, target // len(API_INFO_TYPES))
    else:
        per_type_target = 50  # Not used in search mode
        total_target = target

    if verbose:
        print("=" * 50)
        print("HELSEDIR CONTENT IMPORT")
        print("=" * 50)
        if mode == "specific_type":
            display_name = CATEGORY_INFO.get(specific_type, specific_type)
            print(f"\nMode: Specific info type")
            print(f"Info type: {specific_type} ({display_name})")
            print(f"Limit: {'All available' if limit == 0 else limit}")
        elif args.by_type:
            print(f"\nMode: By info type (balanced coverage)")
            print(f"Info types from API: {len(API_INFO_TYPES)} (excludes 'temaside')")
            print(f"Target per type: {per_type_target}")
            print(f"Total target: ~{total_target}")
        else:
            print(f"\nMode: Search terms")
            print(f"Search terms: {len(search_terms)}")
            print(f"Target items: {target if target > 0 else 'No limit'}")
        print(f"Fetch links: {'Yes' if fetch_links else 'No (fast mode)'}")

    # Check database connection
    if not database_service.is_connected():
        print("\nERROR: Could not connect to database")
        print("Make sure MySQL is running and .env is configured correctly")
        sys.exit(1)

    # Show existing count and load IDs if skip-existing is enabled
    existing = database_service.get_content_count()
    existing_ids = set()

    if verbose:
        print(f"Existing items in database: {existing}")

    if args.skip_existing:
        if verbose:
            print(f"Loading existing IDs for skip-check...", end=" ", flush=True)
        existing_ids = set(database_service.get_all_content_ids())
        if verbose:
            print(f"loaded {len(existing_ids)} IDs")
            print(f"⚡ Optimization: Will skip detail fetch for existing items")

    # Fetch content
    if mode == "specific_type":
        content_items = fetch_content_by_type(
            verbose=verbose,
            fetch_links=fetch_links,
            target_per_type=limit,
            specific_type=specific_type,
            skip_existing=args.skip_existing,
            existing_ids=existing_ids
        )
    elif args.by_type:
        content_items = fetch_content_by_type(
            verbose=verbose,
            fetch_links=fetch_links,
            target_per_type=per_type_target,
            skip_existing=args.skip_existing,
            existing_ids=existing_ids
        )
    else:
        content_items = fetch_content(
            search_terms,
            verbose=verbose,
            fetch_links=fetch_links,
            target=target,
            skip_existing=args.skip_existing,
            existing_ids=existing_ids
        )

    # Save to database
    if content_items:
        save_to_database(content_items, verbose=verbose)
    else:
        if verbose:
            print("No new content to save")

    # Print statistics
    print_statistics(verbose=verbose)


if __name__ == "__main__":
    main()
