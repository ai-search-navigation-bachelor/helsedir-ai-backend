"""
Generate theme pages JSON from paths.

This script takes a list of theme page paths and generates a JSON file
with structured theme page data including titles, parent relationships,
and links.
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

# Add project root to path for importing theme paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.theme_paths import ALL_THEME_PATHS


def path_to_title(path: str) -> str:
    """
    Convert a URL path to a human-readable title.

    Examples:
        /digitalisering-og-e-helse -> Digitalisering og e-helse
        /autorisasjon-og-spesialistutdanning -> Autorisasjon og spesialistutdanning
    """
    # Get last segment
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""

    last_segment = segments[-1]

    # URL decode (handles %C3%B8 -> ø, etc.)
    last_segment = unquote(last_segment)

    # Replace hyphens with spaces
    title = last_segment.replace("-", " ")

    # Capitalize first letter of each word, but preserve acronyms
    words = title.split()
    capitalized_words = []
    for word in words:
        # Keep acronyms like 'e-helse' lowercase
        if word in ["og", "for", "i", "av", "til", "med", "pa", "om"]:
            capitalized_words.append(word)
        else:
            # Capitalize first letter
            capitalized_words.append(
                word[0].upper() + word[1:] if len(word) > 1 else word.upper()
            )

    return " ".join(capitalized_words)


def get_parent_path(path: str) -> Optional[str]:
    """Get the parent path from a path."""
    segments = [s for s in path.split("/") if s]
    if len(segments) <= 1:
        return None
    return "/" + "/".join(segments[:-1])


def get_children_paths(path: str, all_paths: List[str]) -> List[str]:
    """Get all direct children of a path."""
    children = []
    path_depth = path.count("/")

    for p in all_paths:
        if p == path:
            continue
        if p.startswith(path + "/"):
            # Check if it's a direct child (one level deeper)
            if p.count("/") == path_depth + 1:
                children.append(p)

    return sorted(children)


# Root theme categories with fixed category codes
ROOT_CATEGORIES = {
    "forebygging-diagnose-og-behandling": "0001",
    "digitalisering-og-e-helse": "0002",
    "lov-og-forskrift": "0003",
    "helseberedskap": "0004",
    "autorisasjon-og-spesialistutdanning": "0005",
    "tilskudd-og-finansiering": "0006",
    "statistikk-registre-og-rapporter": "0007",
}


def get_category_code(path: str) -> str:
    """Get category code for a path based on root theme."""
    segments = [s for s in path.split("/") if s]
    if not segments:
        return "9999"  # Uncategorized

    root = segments[0]
    return ROOT_CATEGORIES.get(root, "9999")  # 9999 for uncategorized


def generate_theme_pages(paths: List[str]) -> List[Dict]:
    """Generate theme page data from paths."""
    # First, decode all paths and replace spaces with hyphens
    decoded_paths = [unquote(p).replace(" ", "-") for p in paths]

    theme_pages = []

    for path in sorted(decoded_paths):
        # Determine which root category this page belongs to
        category_code = get_category_code(path)

        # Generate ID in same format as Helsedir API content
        # Format: 9999-{category}-{uuid}
        # - 9999: indicates theme page
        # - category: 0001-0007 for main themes, 9999 for uncategorized
        # - uuid: unique identifier for this specific page
        page_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"https://helsedirektoratet.no{path}")
        )
        page_id = f"9999-{category_code}-{page_uuid}"

        # Generate title
        title = path_to_title(path)

        # Get parent
        parent_path = get_parent_path(path)

        # Get children
        children = get_children_paths(path, decoded_paths)

        # Build links array
        links = []

        # Add parent link if exists
        if parent_path:
            links.append({"rel": "forelder", "type": "temaside", "href": parent_path})

        # Add child links
        for child_path in children:
            links.append({"rel": "barn", "type": "temaside", "href": child_path})

        theme_page = {
            "id": page_id,
            "tittel": title,
            "tekst": "",  # Empty for now
            "info_type": "temaside",
            "path": path,
            "koder": None,
            "maalgruppe": None,
            "links": links,
        }

        theme_pages.append(theme_page)

    return theme_pages


if __name__ == "__main__":
    # Generate theme pages from imported paths
    theme_pages = generate_theme_pages(ALL_THEME_PATHS)

    # Write to JSON file
    output_path = "data/theme_pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(theme_pages, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(theme_pages)} theme pages")
    print(f"Saved to {output_path}")

    # Print root category mapping
    print(f"\nRoot categories ({len(ROOT_CATEGORIES)}):")
    for root, code in sorted(ROOT_CATEGORIES.items(), key=lambda x: x[1]):
        print(f"  9999-{code}: /{root}")
    print("  9999-9999: /... (uncategorized)")

    # Print some examples
    print("\nExample theme pages:")
    for page in theme_pages[:3]:
        print(f"\nID: {page['id']}")
        print(f"Path: {page['path']}")
        print(f"Title: {page['tittel']}")
        print(f"Links: {len(page['links'])}")
