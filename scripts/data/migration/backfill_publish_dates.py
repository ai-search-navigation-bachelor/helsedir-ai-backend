#!/usr/bin/env python3
"""
Backfill forstPublisert and sistFagligOppdatert for existing content.

Adds the two date columns if missing, then fetches dates from the
Helsedirektoratet API for all non-temaside content that is missing them.

Usage:
    python scripts/data/migration/backfill_publish_dates.py
    python scripts/data/migration/backfill_publish_dates.py --dry-run
    python scripts/data/migration/backfill_publish_dates.py --workers 5
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.repositories.base import db_pool
from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)


def ensure_columns_exist():
    """Add forst_publisert and sist_faglig_oppdatert columns if they don't exist."""
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()
        # Check which columns exist
        cursor.execute("SHOW COLUMNS FROM content LIKE 'forst_publisert'")
        has_forst = cursor.fetchone() is not None
        cursor.execute("SHOW COLUMNS FROM content LIKE 'sist_faglig_oppdatert'")
        has_sist = cursor.fetchone() is not None

        if not has_forst:
            print("Adding column forst_publisert...")
            cursor.execute("ALTER TABLE content ADD COLUMN forst_publisert DATETIME DEFAULT NULL AFTER links")
            conn.commit()
            print("  Done.")
        else:
            print("Column forst_publisert already exists.")

        if not has_sist:
            # Check if old column name exists (needs rename)
            cursor.execute("SHOW COLUMNS FROM content LIKE 'sist_oppdatert'")
            has_old = cursor.fetchone() is not None

            if has_old:
                print("Renaming column sist_oppdatert -> sist_faglig_oppdatert...")
                cursor.execute("ALTER TABLE content CHANGE sist_oppdatert sist_faglig_oppdatert DATETIME DEFAULT NULL")
                conn.commit()
                print("  Done.")
            else:
                print("Adding column sist_faglig_oppdatert...")
                cursor.execute("ALTER TABLE content ADD COLUMN sist_faglig_oppdatert DATETIME DEFAULT NULL AFTER forst_publisert")
                conn.commit()
                print("  Done.")
        else:
            print("Column sist_faglig_oppdatert already exists.")
    finally:
        if cursor:
            cursor.close()
        conn.close()


def get_content_to_backfill(refetch: bool = False):
    """Get IDs of content items that need date backfilling (excluding temaside).

    Args:
        refetch: If True, return ALL non-temaside content (for re-fetching after field correction).
                 If False, only return items missing both date fields.
    """
    conn = db_pool.get_connection()
    if not conn:
        return []

    cursor = None
    try:
        cursor = conn.cursor()
        if refetch:
            cursor.execute(
                """
                SELECT id FROM content
                WHERE info_type != 'temaside'
                """
            )
        else:
            cursor.execute(
                """
                SELECT id FROM content
                WHERE info_type != 'temaside'
                AND (forst_publisert IS NULL OR sist_faglig_oppdatert IS NULL)
                """
            )
        return [row[0] for row in cursor.fetchall()]
    finally:
        if cursor:
            cursor.close()
        conn.close()


def fetch_dates_from_api(content_id: str):
    """Fetch forstPublisert and sistFagligOppdatert from API for a single item."""
    detailed = helsedir_api_service.get_infobit_by_id(content_id, timeout=15.0)
    return {
        "forstPublisert": detailed.get("forstPublisert"),
        "sistFagligOppdatert": detailed.get("sistFagligOppdatert"),
    }


def save_dates_batch(updates):
    """Save date fields for a batch of items."""
    if not updates:
        return 0

    conn = db_pool.get_connection()
    if not conn:
        return 0

    cursor = None
    try:
        cursor = conn.cursor()
        for content_id, dates in updates:
            cursor.execute(
                """
                UPDATE content
                SET forst_publisert = COALESCE(%s, forst_publisert),
                    sist_faglig_oppdatert = COALESCE(%s, sist_faglig_oppdatert)
                WHERE id = %s
                """,
                (dates.get("forstPublisert") or None, dates.get("sistFagligOppdatert") or None, content_id),
            )
        conn.commit()
        return len(updates)
    except Exception as e:
        print(f"  ERROR saving batch: {e}")
        conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill publish dates for content")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--refetch", action="store_true", help="Clear sist_faglig_oppdatert and re-fetch all from API (use after field name correction)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel API workers (default: 10)")
    parser.add_argument("--batch-size", type=int, default=50, help="DB write batch size (default: 50)")
    args = parser.parse_args()

    print("=" * 60)
    print("BACKFILL PUBLISH DATES")
    print("=" * 60)

    # Step 1: Ensure columns exist
    print("\n[1/4] Ensuring database columns exist...")
    ensure_columns_exist()

    # Step 2: Clear sist_faglig_oppdatert if --refetch
    if args.refetch:
        print("\n[2/4] Clearing sist_faglig_oppdatert (--refetch)...")
        if not args.dry_run:
            conn = db_pool.get_connection()
            if conn:
                cursor = None
                try:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE content SET sist_faglig_oppdatert = NULL WHERE info_type != 'temaside'")
                    affected = cursor.rowcount
                    conn.commit()
                    print(f"  Cleared {affected} rows.")
                finally:
                    if cursor:
                        cursor.close()
                    conn.close()
        else:
            print("  [DRY RUN - skipped]")
    else:
        print("\n[2/4] Skipping clear (use --refetch to re-fetch all).")

    # Step 3: Find items to backfill
    print("\n[3/4] Finding content to backfill...")
    missing_ids = get_content_to_backfill(refetch=args.refetch)
    label = "all non-temaside items (--refetch)" if args.refetch else "items missing dates"
    print(f"  Found {len(missing_ids)} {label}")

    if not missing_ids:
        print("\nAll content already has dates. Nothing to do.")
        return

    # Step 4: Fetch from API and update (sequential with batched DB writes)
    print(f"\n[4/4] Fetching dates from API ({args.workers} workers)...")
    if args.dry_run:
        print("  [DRY RUN - no DB writes]")

    total = len(missing_ids)
    start_time = time.monotonic()
    success = 0
    errors = 0
    skipped = 0
    pending_updates = []
    i = 0

    # Process in chunks to avoid overwhelming the API
    for chunk_start in range(0, total, args.workers):
        chunk = missing_ids[chunk_start:chunk_start + args.workers]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_dates_from_api, cid): cid
                for cid in chunk
            }

            for future in as_completed(futures):
                i += 1
                content_id = futures[future]
                try:
                    dates = future.result()
                    has_dates = dates.get("forstPublisert") or dates.get("sistFagligOppdatert")

                    if has_dates:
                        if not args.dry_run:
                            pending_updates.append((content_id, dates))
                        success += 1
                        pub = dates.get("forstPublisert") or "-"
                        upd = dates.get("sistFagligOppdatert") or "-"
                        print(f"  [{i}/{total}] {content_id}  publisert={pub}  oppdatert={upd}")
                    else:
                        skipped += 1
                        print(f"  [{i}/{total}] {content_id}  (no dates in API)")

                except HelseDirectorateAPIError as e:
                    errors += 1
                    print(f"  [{i}/{total}] {content_id}  ERROR (API): {e}")
                except Exception as e:
                    errors += 1
                    print(f"  [{i}/{total}] {content_id}  ERROR: {type(e).__name__}: {e}")

        # Flush batch after each chunk
        if not args.dry_run and len(pending_updates) >= args.batch_size:
            saved = save_dates_batch(pending_updates)
            print(f"  --- Batch saved: {saved} items ---")
            if saved > 0:
                pending_updates.clear()
            else:
                print("  WARNING: Batch save returned 0, retaining items for retry")

        # Summary progress every 100 items
        if i % 100 == 0 and i > 0:
            elapsed = time.monotonic() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = total - i
            eta = remaining / rate if rate > 0 else 0
            print(
                f"\n  === Progress: {i}/{total} ({100*i/total:.0f}%) | "
                f"{rate:.1f} items/s | ETA: {eta/60:.1f} min | "
                f"OK: {success}, Skip: {skipped}, Err: {errors} ===\n"
            )

        # Rate limit between chunks
        time.sleep(0.3)

    # Flush remaining
    if not args.dry_run and pending_updates:
        saved = save_dates_batch(pending_updates)
        if saved > 0:
            print(f"  Final batch saved: {saved} items")
        else:
            print(f"  ERROR: Final batch save failed, {len(pending_updates)} items not saved")

    elapsed = time.monotonic() - start_time
    print()
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Total: {len(missing_ids)}")
    print(f"  Updated: {success}")
    print(f"  Skipped (no dates in API): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
