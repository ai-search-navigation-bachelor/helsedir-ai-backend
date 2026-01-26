#!/usr/bin/env python3
"""
Import content from Helsedirektoratet API to database.

Usage:
    python scripts/data/import_content.py                 # Default: 500 items, 10 search terms
    python scripts/data/import_content.py --extended      # Use extended search terms (~120 terms)
    python scripts/data/import_content.py --target 1000   # Fetch up to 1000 items
    python scripts/data/import_content.py --no-links      # Skip fetching links (much faster)
    python scripts/data/import_content.py --alphabet      # Search using alphabet (a-z, æøå)
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)
from app.services.data.database_service import database_service


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


def fetch_content(search_terms: list, verbose: bool = True, fetch_links: bool = True, target: int = 0) -> dict:
    """
    Fetch content from Helsedir API using search terms.

    Args:
        search_terms: List of search terms to use
        verbose: Whether to print progress
        fetch_links: Whether to fetch detailed info including links for each item
        target: Target number of items (0 = no limit)

    Returns:
        Dictionary of content items keyed by ID (deduped)
    """
    all_content = {}
    total_detail_fetches = 0

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
                if verbose:
                    print(f"    Fetching details for {len(new_items_this_term)} items:", flush=True)

                for j, (content_id, item) in enumerate(new_items_this_term):
                    try:
                        if verbose:
                            print(f"      [{j+1}/{len(new_items_this_term)}] ID: {content_id}", flush=True)

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
        print(f"\n\nTotal items: {len(all_content)}, Detail fetches: {total_detail_fetches}")

    return all_content


def save_to_database(content_items: dict, verbose: bool = True) -> int:
    """Save content items to database."""
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
        default=500,
        help="Target number of items to fetch (default: 500, 0 = no limit)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    fetch_links = not args.no_links

    # Determine search terms to use
    if args.search_terms:
        search_terms = [t.strip() for t in args.search_terms.split(",")]
    elif args.alphabet:
        search_terms = ALPHABET_SEARCH_TERMS
    elif args.extended:
        search_terms = EXTENDED_SEARCH_TERMS
    else:
        search_terms = DEFAULT_SEARCH_TERMS

    target = args.target

    if verbose:
        print("=" * 50)
        print("HELSEDIR CONTENT IMPORT")
        print("=" * 50)
        print(f"\nSearch terms: {len(search_terms)}")
        print(f"Target items: {target if target > 0 else 'No limit'}")
        print(f"Fetch links: {'Yes' if fetch_links else 'No (fast mode)'}")

    # Check database connection
    if not database_service.is_connected():
        print("\nERROR: Could not connect to database")
        print("Make sure MySQL is running and .env is configured correctly")
        sys.exit(1)

    # Show existing count
    existing = database_service.get_content_count()
    if verbose:
        print(f"Existing items in database: {existing}")

    # Fetch content
    content_items = fetch_content(search_terms, verbose=verbose, fetch_links=fetch_links, target=target)

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
