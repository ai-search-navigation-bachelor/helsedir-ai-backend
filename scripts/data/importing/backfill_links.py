#!/usr/bin/env python3
"""
Backfill content for all barn and forelder links in the database.

For every content item in the database, this script:
1. Reads each barn/forelder link that still has href (not yet resolved to id)
2. Fetches the linked content from Helsedir API (in parallel)
3. Saves it to the database (as e.g. kapittel, anbefaling, etc.)
4. Re-runs link migration so links are updated: href → id

After this script, content_service will find all linked content in
the in-memory cache and never need a runtime API call to populate children.

Usage:
    python scripts/data/importing/backfill_links.py
    python scripts/data/importing/backfill_links.py --dry-run   # Preview only
    python scripts/data/importing/backfill_links.py --workers 20
"""

import argparse
import asyncio
import json
import sys
import os
from collections import Counter
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.services.repositories.base import db_pool
from app.services.external.helsedir_api_service import helsedir_api_service, HelseDirectorateAPIError
from scripts.data.importing.link_utils import extract_content_id_from_href, extract_helsedir_path


# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────

def get_all_content_ids() -> Set[str]:
    """Return all content IDs currently in the database."""
    conn = db_pool.get_connection()
    if not conn:
        raise RuntimeError("Database connection unavailable: db_pool.get_connection() returned None")
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM content")
        return {row[0] for row in cursor.fetchall()}
    finally:
        if cursor:
            cursor.close()
        conn.close()


def get_all_content_with_links() -> List[Dict]:
    """Return all content rows that have a non-null links field."""
    conn = db_pool.get_connection()
    if not conn:
        raise RuntimeError("Database connection unavailable: db_pool.get_connection() returned None")
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, tittel, info_type, links FROM content WHERE links IS NOT NULL")
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        conn.close()


def save_content_batch(items: List[Dict]) -> int:
    """
    Insert/update content items in the database.
    Uses the same ON DUPLICATE KEY UPDATE pattern as the main import.
    """
    if not items:
        return 0

    conn = db_pool.get_connection()
    if not conn:
        return 0

    cursor = None
    saved = 0
    try:
        cursor = conn.cursor()
        for item in items:
            info_type = (
                item.get("info_type")
                or item.get("infoType")
                or (item.get("tekniskeData") or {}).get("infoType")
                or ""
            )
            links_json = item.get("links")
            if isinstance(links_json, list):
                links_json = json.dumps(links_json, ensure_ascii=False)

            cursor.execute(
                """
                INSERT INTO content (id, tittel, tekst, info_type, koder, maalgruppe, links, path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    tittel     = VALUES(tittel),
                    tekst      = VALUES(tekst),
                    info_type  = VALUES(info_type),
                    links      = COALESCE(VALUES(links), links),
                    path       = COALESCE(VALUES(path), path)
                """,
                (
                    item.get("id"),
                    item.get("tittel") or "",
                    item.get("tekst") or "",
                    info_type,
                    json.dumps(item["koder"], ensure_ascii=False) if item.get("koder") else None,
                    json.dumps(item["maalgruppe"], ensure_ascii=False) if item.get("maalgruppe") else None,
                    links_json,
                    item.get("path") or extract_helsedir_path(item.get("url", "") or ""),
                ),
            )
            saved += 1

        conn.commit()
    except Exception as e:
        print(f"  ERROR saving batch: {e}")
        conn.rollback()
        saved = 0
    finally:
        if cursor:
            cursor.close()
        conn.close()

    return saved


def update_links_for_content(content_id: str, links: List[Dict]) -> None:
    """Overwrite the links JSON for a single content row."""
    conn = db_pool.get_connection()
    if not conn:
        return
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE content SET links = %s WHERE id = %s",
            (json.dumps(links, ensure_ascii=False), content_id),
        )
        conn.commit()
    finally:
        if cursor:
            cursor.close()
        conn.close()


# ─────────────────────────────────────────────
# Link processing (same logic as import_content)
# ─────────────────────────────────────────────

def resolve_links(links: List[Dict], existing_ids: Set[str]) -> List[Dict]:
    """
    Replace href with id for links whose content is now in the database.
    Mirrors process_links_at_import from import_content.py.
    """
    resolved = []
    for link in links:
        if not isinstance(link, dict):
            continue

        new_link = {
            "rel":  link.get("rel", ""),
            "type": link.get("type", ""),
        }
        if link.get("tittel"):
            new_link["tittel"] = link["tittel"]

        existing_id = link.get("id")
        href = link.get("href") or None
        if href in ("None", ""):
            href = None

        if existing_id:
            new_link["id"] = existing_id
        elif href:
            extracted_id = extract_content_id_from_href(href)
            if extracted_id and extracted_id in existing_ids:
                new_link["id"] = extracted_id
            else:
                new_link["href"] = href
        else:
            continue  # skip invalid

        resolved.append(new_link)
    return resolved


# ─────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────

_BACKFILL_RELS = {"barn", "forelder"}


def collect_missing_hrefs(all_content: List[Dict], existing_ids: Set[str]) -> Dict[str, str]:
    """
    Scan all content links and return a mapping of href → extracted_id
    for barn/forelder links that:
      - have href (not yet resolved to id), AND
      - the extracted ID is not already in the database.
    """
    missing: Dict[str, str] = {}  # href -> content_id

    for row in all_content:
        raw_links = row.get("links")
        if not raw_links:
            continue

        try:
            links = json.loads(raw_links) if isinstance(raw_links, str) else raw_links
        except (json.JSONDecodeError, TypeError):
            continue

        for link in links:
            if not isinstance(link, dict):
                continue
            if link.get("rel") not in _BACKFILL_RELS:
                continue
            if link.get("id"):
                continue  # already resolved

            href = link.get("href")
            if not href:
                continue

            content_id = extract_content_id_from_href(href)
            if content_id and content_id not in existing_ids:
                missing[href] = content_id

    return missing


async def fetch_hrefs_async(hrefs: List[str], workers: int) -> Dict[str, Dict]:
    """
    Fetch all hrefs in parallel (up to `workers` concurrent requests).
    Returns mapping of href → API response dict.
    Errors are printed and skipped.
    """
    semaphore = asyncio.Semaphore(workers)
    total = len(hrefs)
    done = 0
    ok = 0
    failed = 0

    async def fetch_one(href: str):
        nonlocal done, ok, failed
        async with semaphore:
            try:
                data = await helsedir_api_service.get_content_by_href_async(href)
                ok += 1
                return href, data
            except HelseDirectorateAPIError as e:
                failed += 1
                print(f"\n  WARN [{done+1}/{total}]: {e}")
                return href, None
            except Exception as e:
                failed += 1
                print(f"\n  ERROR [{done+1}/{total}]: {e}")
                return href, None
            finally:
                done += 1
                print(f"  [{done}/{total}] ok={ok} failed={failed}", end="\r", flush=True)

    results = await asyncio.gather(*[fetch_one(h) for h in hrefs])
    print()  # newline after progress line
    return {href: data for href, data in results if data is not None}


_MIGRATION_BATCH_SIZE = 100


def run_link_migration(all_content: List[Dict], existing_ids: Set[str], dry_run: bool) -> int:
    """
    Update links in the database: replace href → id for any link
    whose target content is now in existing_ids.
    Returns number of content rows updated.

    Uses a single shared connection and commits every _MIGRATION_BATCH_SIZE rows.
    Per-row errors are logged and skipped so one bad row doesn't abort the migration.
    """
    updated = 0

    if dry_run:
        for row in all_content:
            raw_links = row.get("links")
            if not raw_links:
                continue
            try:
                links = json.loads(raw_links) if isinstance(raw_links, str) else raw_links
            except (json.JSONDecodeError, TypeError):
                continue
            if resolve_links(links, existing_ids) != links:
                updated += 1
        return updated

    conn = db_pool.get_connection()
    if not conn:
        print("  ERROR: database connection unavailable, skipping link migration")
        return 0

    cursor = None
    try:
        cursor = conn.cursor()
        pending = 0
        for row in all_content:
            raw_links = row.get("links")
            if not raw_links:
                continue
            try:
                links = json.loads(raw_links) if isinstance(raw_links, str) else raw_links
            except (json.JSONDecodeError, TypeError):
                continue

            resolved = resolve_links(links, existing_ids)
            if resolved == links:
                continue

            try:
                cursor.execute(
                    "UPDATE content SET links = %s WHERE id = %s",
                    (json.dumps(resolved, ensure_ascii=False), row["id"]),
                )
                updated += 1
                pending += 1
                if pending >= _MIGRATION_BATCH_SIZE:
                    conn.commit()
                    pending = 0
            except Exception as e:
                print(f"  WARN: failed to update links for {row['id']}: {e}")

        if pending:
            conn.commit()
    except Exception as e:
        print(f"  ERROR during link migration: {e}")
        conn.rollback()
    finally:
        if cursor:
            cursor.close()
        conn.close()

    return updated


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill content for all barn and forelder links in the database"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be fetched/updated without writing to DB")
    parser.add_argument("--workers", type=int, default=10,
                        help="Max concurrent API requests (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("BACKFILL BARN & FORELDER LINKS")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN — no changes will be written\n")

    # 1. Load current state
    print("Loading content from database...")
    all_content = get_all_content_with_links()
    existing_ids = get_all_content_ids()
    print(f"  {len(all_content)} content rows with links")
    print(f"  {len(existing_ids)} unique content IDs in DB")

    # 2. Find missing barn/forelder content
    print("\nScanning barn and forelder links...")
    missing = collect_missing_hrefs(all_content, existing_ids)
    print(f"  {len(missing)} unique barn/forelder hrefs not yet in database")

    if not missing:
        print("\nNothing to fetch — all barn/forelder links are already in the database.")
    else:
        print(f"\nFetching {len(missing)} items from Helsedir API ({args.workers} concurrent)...")

        if args.dry_run:
            for href, content_id in list(missing.items())[:10]:
                print(f"  Would fetch: {content_id}  ({href[:70]}...)")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        else:
            fetched = asyncio.run(fetch_hrefs_async(list(missing.keys()), args.workers))
            print(f"  Fetched: {len(fetched)}/{len(missing)}")

            # Build items to save (map href → fetched data, set correct id)
            items_to_save = []
            for href, data in fetched.items():
                content_id = missing[href]
                data["id"] = content_id  # ensure correct ID
                items_to_save.append(data)

            # Process links on newly fetched items before saving
            # (so their own barn links also get id-resolved if possible)
            all_ids_with_new = existing_ids | {item["id"] for item in items_to_save}
            for item in items_to_save:
                raw = item.get("links") or []
                if isinstance(raw, list):
                    item["links"] = resolve_links(raw, all_ids_with_new)

            saved = save_content_batch(items_to_save)
            print(f"  Saved: {saved} items to database")

            # Print breakdown by info_type
            type_counts = Counter(
                item.get("info_type")
                or item.get("infoType")
                or (item.get("tekniskeData") or {}).get("infoType")
                or "(unknown)"
                for item in items_to_save
            )
            print("\n  By info_type:")
            for info_type, count in type_counts.most_common():
                print(f"    {info_type}: {count}")

            # Update existing_ids with newly saved content
            existing_ids = get_all_content_ids()

    # 3. Re-run link migration (update all links: href → id)
    print("\nUpdating links (href → id)...")
    # Reload content since links may have changed
    all_content = get_all_content_with_links()
    updated = run_link_migration(all_content, existing_ids, args.dry_run)
    action = "Would update" if args.dry_run else "Updated"
    print(f"  {action} links in {updated} content rows")

    print("\n" + "=" * 60)
    print("Done.")
    print()
    print("Next steps:")
    print("  1. Restart the backend to reload the in-memory content cache")
    print("  2. Linked content (barn/forelder) will now be served without API calls")


if __name__ == "__main__":
    main()
