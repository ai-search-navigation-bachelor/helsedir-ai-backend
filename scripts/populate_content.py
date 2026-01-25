"""
Script to populate the content table with data from Helsedirektoratet API.

Fetches content using various search queries to get diverse health content.
Uses the same import pattern as scripts/data/import_content.py (two API calls
per query to get both infoType and full text content).

Usage:
    python scripts/populate_content.py
    python scripts/populate_content.py --target 3000
"""

import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external.helsedir_api_service import helsedir_api_service
from app.services.data.database_service import database_service
from app.exceptions.helsedir import HelseDirectorateAPIError


# Extensive search queries to get diverse content (targeting ~2000+ items)
SEARCH_QUERIES = [
    # === Diseases and conditions ===
    "diabetes",
    "kreft",
    "hjerte",
    "lunge",
    "psykisk",
    "demens",
    "astma",
    "kols",
    "hypertensjon",
    "allergi",
    "infeksjon",
    "smerte",
    "depresjon",
    "angst",
    "rus",
    "alkohol",
    "overvekt",
    "ernæring",
    "søvn",
    "stress",
    "migrene",
    "hodepine",
    "rygg",
    "nakke",
    "ledd",
    "artritt",
    "osteoporose",
    "fibromyalgi",
    "ms",
    "parkinson",
    "alzheimer",
    "epilepsi",
    "cerebral parese",

    # === Cancer types ===
    "brystkreft",
    "lungekreft",
    "prostatakreft",
    "tarmkreft",
    "hudkreft",
    "blodkreft",
    "leukemi",
    "lymfekreft",
    "hjernesvulst",
    "eggstokkreft",
    "livmorkreft",
    "blærekreft",
    "nyrekreft",
    "bukspyttkjertelkreft",
    "magekreft",
    "spiserørskreft",
    "strupekreft",
    "skjoldbruskkjertelkreft",

    # === Cardiovascular ===
    "hjerteinfarkt",
    "hjerneslag",
    "hjertesvikt",
    "arytmi",
    "atrieflimmer",
    "blodpropp",
    "trombose",
    "emboli",
    "aneurisme",
    "aterosklerose",
    "angina",
    "koronar",
    "bypass",
    "pacemaker",

    # === Respiratory ===
    "pneumoni",
    "bronkitt",
    "tuberkulose",
    "covid",
    "influensa",
    "forkjølelse",
    "bihulebetennelse",
    "lungebetennelse",
    "lungeemboli",
    "lungefibrose",
    "sarkoidose",

    # === Digestive ===
    "mage",
    "tarm",
    "lever",
    "gallestein",
    "pankreatitt",
    "hepatitt",
    "cirrhose",
    "crohn",
    "ulcerøs kolitt",
    "ibs",
    "irritabel",
    "reflux",
    "magesår",
    "obstipasjon",
    "forstoppelse",
    "diaré",
    "cøliaki",
    "glutenintoleranse",
    "laktoseintoleranse",

    # === Endocrine ===
    "skjoldbruskkjertel",
    "hypotyreose",
    "hypertyreose",
    "stoffskifte",
    "binyrer",
    "hypofyse",
    "hormon",
    "insulin",
    "metabolsk syndrom",

    # === Kidney/Urinary ===
    "nyre",
    "nyresvikt",
    "dialyse",
    "nyretransplantasjon",
    "urinveisinfeksjon",
    "blærebetennelse",
    "nyrestein",
    "inkontinens",
    "prostata",

    # === Skin ===
    "hud",
    "eksem",
    "psoriasis",
    "akne",
    "rosacea",
    "føflekk",
    "melanom",
    "hudsykdom",
    "utslett",
    "kløe",
    "sår",
    "brannskade",

    # === Eyes/Ears ===
    "øye",
    "syn",
    "grå stær",
    "grønn stær",
    "glaukom",
    "katarakt",
    "makuladegenerasjon",
    "øre",
    "hørsel",
    "tinnitus",
    "svimmelhet",
    "ørebetennelse",

    # === Mental health ===
    "adhd",
    "autisme",
    "bipolar",
    "schizofreni",
    "ptsd",
    "spiseforstyrrelser",
    "anoreksi",
    "bulimi",
    "selvskading",
    "selvmord",
    "suicid",
    "psykose",
    "personlighetsforstyrrelse",
    "ocd",
    "tvangstanker",
    "fobier",
    "sosial angst",
    "panikk",
    "utbrenthet",
    "sorg",
    "traume",

    # === Women's health ===
    "graviditet",
    "fødsel",
    "amming",
    "menstruasjon",
    "mensen",
    "overgangsalder",
    "klimakteriet",
    "endometriose",
    "pcos",
    "myomer",
    "livmor",
    "eggstokk",
    "prevensjon",
    "abort",
    "infertilitet",
    "ufrivillig barnløshet",
    "svangerskapsforgiftning",
    "keisersnitt",

    # === Men's health ===
    "erektil dysfunksjon",
    "testosteron",
    "testikkelkreft",

    # === Children/Pediatrics ===
    "barn",
    "spedbarn",
    "nyfødt",
    "prematur",
    "barnevaksinasjon",
    "feber barn",
    "ørebetennelse barn",
    "kolikk",
    "mage tarm virus",
    "hånd fot munn",
    "vannkopper",
    "meslinger",
    "kikhoste",
    "skarlagensfeber",
    "barneautisme",
    "utviklingsforsinkelse",
    "språkforsinkelse",
    "lærevansker",
    "dysleksi",

    # === Elderly ===
    "eldre",
    "geriatri",
    "aldring",
    "fallforebygging",
    "hoftebrudd",
    "pleie",
    "omsorg",
    "sykehjem",
    "hjemmesykepleie",
    "demensutredning",

    # === Infections ===
    "bakterie",
    "virus",
    "sopp",
    "parasitt",
    "sepsis",
    "mrsa",
    "resistente bakterier",
    "smittevern",
    "isolasjon",
    "hygiene",
    "hiv",
    "aids",
    "klamydia",
    "gonoré",
    "syfilis",
    "hepatitt b",
    "hepatitt c",
    "borreliose",
    "flåttbitt",
    "malaria",

    # === Medical specialties ===
    "pediatri",
    "geriatri",
    "kirurgi",
    "ortopedi",
    "nevrologi",
    "kardiologi",
    "onkologi",
    "gynekologi",
    "urologi",
    "dermatologi",
    "revmatologi",
    "gastroenterologi",
    "endokrinologi",
    "nefrologi",
    "hematologi",
    "immunologi",
    "infeksjonsmedisin",
    "psykiatri",
    "radiologi",
    "patologi",
    "anestesi",
    "intensiv",
    "akuttmedisin",
    "allmennmedisin",
    "indremedisin",

    # === Treatments ===
    "behandling",
    "medisin",
    "terapi",
    "rehabilitering",
    "forebygging",
    "vaksinasjon",
    "operasjon",
    "smertebehandling",
    "strålebehandling",
    "cellegift",
    "kjemoterapi",
    "immunterapi",
    "fysioterapi",
    "ergoterapi",
    "logopedi",
    "psykoterapi",
    "kognitiv terapi",
    "dialysebehandling",
    "transplantasjon",

    # === Medications ===
    "legemiddel",
    "antibiotika",
    "smertestillende",
    "paracetamol",
    "ibuprofen",
    "morfin",
    "opioider",
    "antidepressiva",
    "antipsykotika",
    "blodfortynnende",
    "warfarin",
    "betablokker",
    "acehemmer",
    "statin",
    "insulin",
    "kortison",
    "antihistamin",
    "sovemedisin",
    "beroligende",
    "bivirkninger",
    "interaksjoner",

    # === Healthcare processes ===
    "utredning",
    "diagnostikk",
    "oppfølging",
    "henvisning",
    "innleggelse",
    "poliklinikk",
    "akutt",
    "øyeblikkelig hjelp",
    "elektiv",
    "dagkirurgi",
    "kontroll",
    "screening",
    "mammografi",
    "koloskopi",
    "gastroskopi",
    "ct",
    "mr",
    "ultralyd",
    "røntgen",
    "blodprøve",
    "biopsi",

    # === Healthcare settings ===
    "fastlege",
    "sykehus",
    "legevakt",
    "helsestasjon",
    "skolehelsetjeneste",
    "bedriftshelsetjeneste",
    "tannlege",
    "apotek",
    "hjelpemiddel",
    "kommunehelsetjeneste",
    "spesialisthelsetjeneste",

    # === Lifestyle ===
    "røyking",
    "snus",
    "fysisk aktivitet",
    "trening",
    "kosthold",
    "vekt",
    "fedme",
    "bmi",
    "kolesterol",
    "blodtrykk",
    "blodsukker",
    "vitamin",
    "mineral",
    "kosttilskudd",
    "vegansk",
    "vegetar",

    # === Patient rights/system ===
    "pasientrettigheter",
    "samtykke",
    "journal",
    "kjernejournal",
    "eresept",
    "frikort",
    "egenandel",
    "sykemelding",
    "uføretrygd",
    "nav",
    "arbeidsavklaringspenger",
    "individuell plan",
    "koordinator",
    "brukermedvirkning",

    # === Guidelines/Documents ===
    "retningslinje",
    "veileder",
    "nasjonale faglige retningslinjer",
    "pakkeforløp",
    "forløp",
    "prosedyre",
    "kvalitet",
    "pasientsikkerhet",
    "avvik",
    "melding",

    # === Alphabet for broader coverage ===
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z", "æ", "ø", "å",
]


def fetch_and_cache_content(target_count: int = 2000, verbose: bool = True):
    """
    Fetch content from API and cache in database.
    Uses two API calls per query (basic + full) to get both infoType and text.

    Args:
        target_count: Target number of unique content items to fetch
        verbose: Whether to print progress
    """
    seen_ids = set()
    total_new = 0

    # First, get existing content IDs from database
    existing = database_service.get_all_content()
    for item in existing:
        seen_ids.add(item.get("id"))

    if verbose:
        print(f"Found {len(seen_ids)} existing items in database")

    for i, query in enumerate(SEARCH_QUERIES, 1):
        if len(seen_ids) >= target_count:
            if verbose:
                print(f"\nReached target of {target_count} items!")
            break

        if verbose:
            print(f"[{i}/{len(SEARCH_QUERIES)}] Searching for: '{query}'...", end=" ")

        try:
            # First get basic info (includes correct infoType)
            basic_results = helsedir_api_service.search_infobits(
                query_text=query,
                get_full_infobits=False,
                timeout=30.0,
            )

            # Then get full info (includes full text content)
            full_results = helsedir_api_service.search_infobits(
                query_text=query,
                get_full_infobits=True,
                timeout=30.0,
            )

            # Create lookup for basic info by id
            basic_by_id = {item.get("id"): item for item in basic_results}

            # Merge: use full results but add infoType from basic
            merged_results = []
            for item in full_results:
                item_id = item.get("id")
                if item_id in basic_by_id:
                    item["infoType"] = basic_by_id[item_id].get("infoType")
                merged_results.append(item)

            # Filter out already seen items
            new_items = []
            for item in merged_results:
                item_id = item.get("id")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    new_items.append(item)

            if new_items:
                # Cache in database
                cached = database_service.cache_content_batch(new_items)
                total_new += cached
                if verbose:
                    print(f"Found {len(merged_results)} results ({len(new_items)} new)")
            else:
                if verbose:
                    print(f"Found {len(merged_results)} results (all already cached)")

            # Rate limiting - be nice to the API
            time.sleep(0.3)

        except HelseDirectorateAPIError as e:
            if verbose:
                print(f"API error: {e}")
            continue
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue

    return total_new


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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Populate database with content from Helsedirektoratet API"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2000,
        help="Target number of content items (default: 2000)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("=" * 50)
        print("HELSEDIR CONTENT POPULATION")
        print("=" * 50)

    # Check API key
    if not helsedir_api_service.api_key:
        print("\nERROR: HELSEDIR_API_KEY not configured!")
        print("Add it to your .env file:")
        print("  HELSEDIR_API_KEY=your_key_here")
        sys.exit(1)

    # Check database connection
    if not database_service.is_connected():
        print("\nERROR: Cannot connect to database!")
        print("Check your MySQL settings in .env")
        sys.exit(1)

    if verbose:
        print(f"\nAPI URL: {helsedir_api_service.base_url}")
        print(f"Target: {args.target} items")
        print(f"Search queries: {len(SEARCH_QUERIES)}")

    # Fetch content
    new_items = fetch_and_cache_content(target_count=args.target, verbose=verbose)

    if verbose:
        print(f"\nNew items added: {new_items}")

    # Print statistics
    print_statistics(verbose=verbose)


if __name__ == "__main__":
    main()
