#!/usr/bin/env python3
"""
Backfill content.is_dead_end_theme_page for cached theme pages.

This script ensures the database column exists, reloads cached content,
recomputes navigable/dead-end theme pages, and persists the result.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.repositories.content_repository import content_repository
from app.services.data.content_service import content_service


def main() -> int:
    print("Ensuring content.is_dead_end_theme_page exists...")
    if not content_repository.ensure_dead_end_theme_page_column():
        print("ERROR: Could not ensure content.is_dead_end_theme_page column")
        return 1

    print("Reloading cached content...")
    content_service.reload_content()
    print("Recalculating and persisting dead-end theme pages...")
    content_service.refresh_dead_end_theme_page_flags(persist=True)

    theme_pages = [
        item for item in content_service.get_all_content()
        if item.content_type.lower() == "temaside"
    ]
    dead_end_count = sum(1 for item in theme_pages if item.is_dead_end_theme_page)

    print(f"Theme pages: {len(theme_pages)}")
    print(f"Dead-end theme pages: {dead_end_count}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
