"""
Generate GPL queries for theme pages (temasider).

Since temasider have no body text, queries are derived from:
- The theme page title itself
- Child theme page titles (from the hierarchy)
- Path segments (which often contain descriptive terms)

Usage:
    python scripts/data/maintenance/generate_temaside_queries.py

    # Preview without writing
    python scripts/data/maintenance/generate_temaside_queries.py --dry-run

    # Custom output
    python scripts/data/maintenance/generate_temaside_queries.py --output data/gpl_queries_with_temasider.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.search.synonyms import SYNONYM_LOOKUP

MAX_QUERIES = 10


def clean_title(title: str) -> str:
    """Normalize title casing (theme pages often have odd casing)."""
    return title.strip()


def get_synonym_variants(text: str) -> List[str]:
    """
    Generate synonym variants for a text string.

    Checks both multi-word and single-word synonyms.
    Returns a list of variant strings with one term replaced.
    """
    text_lower = text.lower()
    variants: List[str] = []
    seen: Set[str] = set()

    # Multi-word synonyms first (longer matches take priority)
    multi_word_terms = sorted(
        [t for t in SYNONYM_LOOKUP if " " in t],
        key=len,
        reverse=True,
    )
    matched_ranges: List[tuple] = []

    for term in multi_word_terms:
        idx = text_lower.find(term)
        if idx >= 0:
            end = idx + len(term)
            if not any(s < end and e > idx for s, e in matched_ranges):
                matched_ranges.append((idx, end))
                for syn in SYNONYM_LOOKUP[term]:
                    if syn.lower() != term:
                        variant = text_lower[:idx] + syn + text_lower[end:]
                        if variant not in seen:
                            seen.add(variant)
                            variants.append(variant)

    # Single-word synonyms — use re.finditer for accurate positions
    for m in re.finditer(r"\w+", text_lower):
        token = m.group()
        if token in SYNONYM_LOOKUP:
            idx = m.start()
            end = m.end()
            if not any(s <= idx and e >= end for s, e in matched_ranges):
                matched_ranges.append((idx, end))
                for syn in SYNONYM_LOOKUP[token]:
                    if syn.lower() != token:
                        variant = text_lower[:idx] + syn + text_lower[end:]
                        if variant not in seen:
                            seen.add(variant)
                            variants.append(variant)

    return variants


def generate_queries_for_temaside(
    page: dict,
    children_titles: List[str],
) -> List[str]:
    """
    Generate up to MAX_QUERIES search queries for a theme page.

    Strategy:
    1. Title as-is
    2. Synonym variants of the title
    3. Child page titles
    4. Synonym variants of child titles
    5. Path-derived queries
    6. Parent context for deeper pages
    """
    title = clean_title(page["tittel"])
    path = page.get("path", "")
    queries: List[str] = []
    seen_lower: Set[str] = set()

    def add(q: str):
        q = q.strip()
        if not q or q.lower() in seen_lower:
            return
        if len(queries) >= MAX_QUERIES:
            return
        seen_lower.add(q.lower())
        queries.append(q)

    title_lower = title.lower()

    # 1. Title as-is
    add(title)

    # 2. Synonym variants of the title
    for variant in get_synonym_variants(title):
        add(variant)

    # 3. Natural query templates using the title
    templates = [
        f"hva er {title_lower}",
        f"behandling av {title_lower}",
        f"symptomer på {title_lower}",
        f"retningslinjer for {title_lower}",
        f"{title_lower} veileder",
    ]
    for t in templates:
        add(t)

    # 4. Child-based queries + their synonym variants
    for child_title in children_titles:
        child_clean = clean_title(child_title)
        add(child_clean)
        for variant in get_synonym_variants(child_clean):
            add(variant)

    return queries[:MAX_QUERIES]


def build_hierarchy(theme_pages: List[dict]) -> Dict[str, List[str]]:
    """
    Build a mapping from each theme page path to its children's titles.
    """
    path_to_title = {}
    path_to_children: Dict[str, List[str]] = {}

    for page in theme_pages:
        p = page["path"].rstrip("/")
        path_to_title[p] = page["tittel"]
        path_to_children.setdefault(p, [])

    for page in theme_pages:
        for link in page.get("links", []):
            if link.get("rel") == "barn" and link.get("type") == "temaside":
                child_path = link["href"].rstrip("/")
                parent_path = page["path"].rstrip("/")
                if child_path in path_to_title:
                    path_to_children.setdefault(parent_path, []).append(
                        path_to_title[child_path]
                    )

    return path_to_children


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPL queries for theme pages"
    )
    parser.add_argument(
        "--theme-pages",
        type=str,
        default=str(PROJECT_ROOT / "data" / "theme_pages.json"),
        help="Path to theme_pages.json",
    )
    parser.add_argument(
        "--gpl-queries",
        type=str,
        default=str(PROJECT_ROOT / "data" / "gpl_queries.json"),
        help="Path to existing gpl_queries.json to merge into",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrite gpl-queries file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing",
    )
    args = parser.parse_args()

    theme_pages_path = Path(args.theme_pages)
    gpl_queries_path = Path(args.gpl_queries)
    output_path = Path(args.output) if args.output else gpl_queries_path

    # Load theme pages
    print(f"Loading theme pages from {theme_pages_path}")
    with open(theme_pages_path, "r", encoding="utf-8") as f:
        theme_pages = json.load(f)
    print(f"Loaded {len(theme_pages)} theme pages")

    # Load existing GPL queries
    print(f"Loading existing GPL queries from {gpl_queries_path}")
    with open(gpl_queries_path, "r", encoding="utf-8") as f:
        gpl_queries = json.load(f)
    print(f"Loaded {len(gpl_queries)} document entries")

    # Build hierarchy
    children_by_path = build_hierarchy(theme_pages)

    # Generate queries for each theme page
    new_count = 0
    updated_count = 0
    total_queries = 0

    for page in theme_pages:
        doc_id = page["id"]
        path = page["path"].rstrip("/")
        children_titles = children_by_path.get(path, [])

        queries = generate_queries_for_temaside(page, children_titles)
        total_queries += len(queries)

        if doc_id in gpl_queries:
            updated_count += 1
        else:
            new_count += 1

        gpl_queries[doc_id] = queries

    print(f"\n=== Results ===")
    print(f"  Theme pages processed: {len(theme_pages)}")
    print(f"  New entries added: {new_count}")
    print(f"  Existing entries updated: {updated_count}")
    print(f"  Total queries generated: {total_queries}")
    print(f"  Avg queries per temaside: {total_queries / len(theme_pages):.1f}")

    # Show examples
    print(f"\n=== Examples ===")
    for page in theme_pages[:5]:
        doc_id = page["id"]
        print(f"\n{page['tittel']} ({page['path']})")
        for i, q in enumerate(gpl_queries[doc_id]):
            print(f"  [{i}] {q}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print(f"\nWriting to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gpl_queries, f, ensure_ascii=False, indent=2)
        print(
            f"Done! Total entries: {len(gpl_queries)} "
            f"({len(gpl_queries) - len(theme_pages)} content + {len(theme_pages)} temasider)"
        )


if __name__ == "__main__":
    main()
