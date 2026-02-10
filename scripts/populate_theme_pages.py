"""
Populate theme pages into the database.

This script reads theme_pages.json and inserts all theme pages
into the content table. Run this after clearing the content table
or when updating theme pages.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.repositories.content_repository import content_repository


def load_theme_pages(json_path: str = "data/theme_pages.json") -> list:
    """Load theme pages from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def populate_theme_pages():
    """Populate theme pages into the database."""
    print("Loading theme pages from JSON...")
    theme_pages = load_theme_pages()
    print(f"Found {len(theme_pages)} theme pages")

    print("\nPopulating database...")
    success_count = 0
    error_count = 0

    for page in theme_pages:
        try:
            # Use content_repository to cache the theme page
            success = content_repository.cache_content(page)
            if success:
                success_count += 1
                print(f"✓ {page['path']}")
            else:
                error_count += 1
                print(f"✗ Failed: {page['path']}")
        except Exception as e:
            error_count += 1
            print(f"✗ Error on {page['path']}: {e}")

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Successfully inserted: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(theme_pages)}")
    print(f"{'='*60}")

    return success_count == len(theme_pages)


if __name__ == "__main__":
    try:
        success = populate_theme_pages()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
