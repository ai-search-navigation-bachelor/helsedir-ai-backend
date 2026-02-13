#!/usr/bin/env python3
"""
Migrate existing links in database to new format.

This script:
1. Reads all content with links
2. For each link (all types: forelder, root, barn, publikasjon):
   - Extracts content ID from href
   - Checks if that content exists in database
   - If yes: replaces href with id field
   - If no: keeps href as external link
3. Removes strukturId from all links
4. Updates the database

Usage:
    python scripts/data/migrate_links.py
    python scripts/data/migrate_links.py --dry-run  # Preview changes without saving
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.repositories.base import db_pool


def extract_content_id_from_href(href: str) -> Optional[str]:
    """
    Extract content ID from Helsedir API href URL.

    Extracts ID from URLs like:
    - https://api.helsedirektoratet.no/innhold/veiledere/0006-0008-2c686203-c2d2-4bd5-94eb-3e1366c57bfa
    - https://api.helsedirektoratet.no/innhold/kapitler/0006-0041-662959ab-1f9b-4438-8b98-9bbf9dcc1ac1
    """
    if not href:
        return None

    # Pattern: /innhold/{type}/{id}
    pattern = r'/innhold/[^/]+/([0-9]{4}-[0-9]{4}-[a-f0-9\-]+)'
    match = re.search(pattern, href, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def get_all_content_ids() -> set:
    """Get all content IDs from database."""
    conn = db_pool.get_connection()
    if not conn:
        return set()

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM content")
        results = cursor.fetchall()
        return {row[0] for row in results}
    finally:
        cursor.close()
        conn.close()


def transform_link(link: dict, existing_ids: set) -> dict:
    """
    Transform a single link to new format.

    Args:
        link: Original link dict with href and strukturId
        existing_ids: Set of content IDs that exist in database

    Returns:
        Transformed link dict
    """
    rel = link.get('rel', '')

    # Create new link without strukturId
    new_link = {
        'rel': rel,
        'type': link.get('type', ''),
    }

    # Add tittel if present
    if link.get('tittel'):
        new_link['tittel'] = link['tittel']

    # Process all link types - try to resolve to internal ID
    href = link.get('href', '')
    extracted_id = extract_content_id_from_href(href)

    if extracted_id and extracted_id in existing_ids:
        # Internal link - use id
        new_link['id'] = extracted_id
    else:
        # External link or no valid ID - keep href
        new_link['href'] = href

    return new_link


def migrate_content_links(dry_run: bool = False):
    """
    Migrate all content links to new format.

    Args:
        dry_run: If True, only preview changes without saving
    """
    print("=" * 60)
    print("MIGRATING CONTENT LINKS TO NEW FORMAT")
    print("=" * 60)

    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be saved\n")

    # Get all content IDs for lookup
    print("Loading all content IDs...")
    existing_ids = get_all_content_ids()
    print(f"  Found {len(existing_ids)} content items in database")

    # Get all content with links
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        return

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, links FROM content WHERE links IS NOT NULL")
        content_rows = cursor.fetchall()
        print(f"\nProcessing {len(content_rows)} content items with links...\n")

        updated_count = 0
        unchanged_count = 0
        error_count = 0
        processed_count = 0

        for row in content_rows:
            processed_count += 1
            content_id = row['id']
            links_json = row['links']

            try:
                # Parse links
                if isinstance(links_json, str):
                    try:
                        links = json.loads(links_json)
                    except json.JSONDecodeError:
                        print(f"  ⚠️  Skipping {content_id}: Invalid JSON")
                        error_count += 1
                        continue
                else:
                    links = links_json

                if not isinstance(links, list):
                    unchanged_count += 1
                    continue

                # Transform each link
                new_links = []
                has_changes = False

                for link in links:
                    if not isinstance(link, dict):
                        # Preserve non-dict entries unchanged
                        print(f"  ⚠️  Warning: Non-dict link in {content_id}: {link}")
                        new_links.append(link)
                        continue

                    new_link = transform_link(link, existing_ids)
                    new_links.append(new_link)

                    # Check if anything changed
                    if 'strukturId' in link:
                        has_changes = True
                    # Check if any link got converted to use 'id' instead of 'href'
                    if 'id' in new_link and 'id' not in link:
                        has_changes = True

                if has_changes:
                    updated_count += 1

                    # Show example of first 5 changes
                    if updated_count <= 5:
                        print(f"\n📝 {content_id}:")
                        print(f"  Before: {len(links)} links")
                        for i, (old, new) in enumerate(zip(links, new_links)):
                            if old != new:
                                print(f"    Link {i+1}:")
                                print(f"      Old: {old}")
                                print(f"      New: {new}")

                    # Update database
                    if not dry_run:
                        update_cursor = conn.cursor()
                        new_links_json = json.dumps(new_links, ensure_ascii=False)
                        update_cursor.execute(
                            "UPDATE content SET links = %s WHERE id = %s",
                            (new_links_json, content_id)
                        )
                        update_cursor.close()
                else:
                    unchanged_count += 1

                # Commit every 100 items to avoid losing progress if script crashes
                if processed_count % 100 == 0:
                    if not dry_run:
                        conn.commit()
                    print(f"  Progress: {processed_count}/{len(content_rows)} items processed ({updated_count} updated, {unchanged_count} unchanged, {error_count} errors)")

            except Exception as e:
                print(f"  ❌ Error processing {content_id}: {e}")
                error_count += 1
                continue

        # Final commit for any remaining changes
        if not dry_run:
            conn.commit()
            print(f"\n✅ All changes committed to database")

        # Summary
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"  Updated: {updated_count}")
        print(f"  Unchanged: {unchanged_count}")
        print(f"  Errors: {error_count}")
        print(f"  Total: {len(content_rows)}")

        if dry_run:
            print(f"\n⚠️  DRY RUN - No changes were saved")
            print(f"   Run without --dry-run to apply changes")
        else:
            print(f"\n✅ Migration complete!")

    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate content links to new format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving to database"
    )
    args = parser.parse_args()

    migrate_content_links(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
