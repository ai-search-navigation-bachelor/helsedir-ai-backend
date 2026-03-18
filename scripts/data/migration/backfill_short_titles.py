#!/usr/bin/env python3
"""
Backfill kort_tittel for existing content rows.

Ensures the kort_tittel column exists, then fetches content details from the
Helsedirektoratet API for rows where kort_tittel is NULL or empty and stores it
when the API returns a non-empty kortTittel.

Usage:
    python scripts/data/migration/backfill_short_titles.py
    python scripts/data/migration/backfill_short_titles.py --dry-run
    python scripts/data/migration/backfill_short_titles.py --workers 10
    python scripts/data/migration/backfill_short_titles.py --limit 100
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.repositories.base import db_pool
from app.services.external.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError,
)

MAX_BATCH_SAVE_RETRIES = 3


def has_short_title_column() -> bool:
    """Return True when content.kort_tittel exists."""
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM content LIKE 'kort_tittel'")
        return cursor.fetchone() is not None
    finally:
        if cursor:
            cursor.close()
        conn.close()


def ensure_short_title_column(dry_run: bool = False) -> None:
    """Add kort_tittel column if it does not exist."""
    if has_short_title_column():
        print("Column kort_tittel already exists.")
        return

    if dry_run:
        print("Column kort_tittel is MISSING (would be added without --dry-run)")
        return

    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()
        print("Adding column kort_tittel...")
        cursor.execute("ALTER TABLE content ADD COLUMN kort_tittel TEXT NULL AFTER tittel")
        conn.commit()
        print("  Done.")
    finally:
        if cursor:
            cursor.close()
        conn.close()


def get_content_to_backfill(limit: int = 0):
    """Fetch content IDs with missing short titles, excluding local theme pages."""
    conn = db_pool.get_connection()
    if not conn:
        return []

    cursor = None
    try:
        cursor = conn.cursor()
        query = """
            SELECT id
            FROM content
            WHERE (info_type IS NULL OR info_type <> 'temaside')
            AND (kort_tittel IS NULL OR TRIM(kort_tittel) = '')
            ORDER BY id
        """
        params = ()
        if limit > 0:
            query += " LIMIT %s"
            params = (limit,)
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    finally:
        if cursor:
            cursor.close()
        conn.close()


def fetch_short_title(content_id: str):
    """Fetch kortTittel from the API for a single content item."""
    detailed = helsedir_api_service.get_infobit_by_id(content_id, timeout=15.0)
    short_title = detailed.get("kortTittel")
    if isinstance(short_title, str):
        short_title = short_title.strip()
    if not short_title:
        short_title = None
    return {
        "kortTittel": short_title,
        "tittel": detailed.get("tittel"),
    }


def save_short_titles_batch(updates) -> int:
    """Save a batch of short titles."""
    if not updates:
        return 0

    conn = db_pool.get_connection()
    if not conn:
        return 0

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """
            UPDATE content
            SET kort_tittel = %s
            WHERE id = %s
            """,
            [(payload["kortTittel"], content_id) for content_id, payload in updates],
        )
        conn.commit()
        return cursor.rowcount
    except Exception as exc:
        print(f"  ERROR saving batch: {exc}")
        conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        conn.close()


def flush_pending_updates(pending_updates, batch_size: int) -> int:
    """Persist pending updates in deterministic batches with bounded retries."""
    saved_total = 0

    while len(pending_updates) >= batch_size:
        batch = pending_updates[:batch_size]
        for attempt in range(1, MAX_BATCH_SAVE_RETRIES + 1):
            saved = save_short_titles_batch(batch)
            if saved > 0:
                saved_total += saved
                print(f"  Batch saved: {saved} items")
                del pending_updates[:batch_size]
                break

            if attempt == MAX_BATCH_SAVE_RETRIES:
                raise RuntimeError(
                    f"Batch save failed after {MAX_BATCH_SAVE_RETRIES} attempts for {len(batch)} items"
                )

            print(f"  WARNING: Batch save returned 0, retrying ({attempt}/{MAX_BATCH_SAVE_RETRIES})")

    return saved_total


def positive_int(value: str) -> int:
    """argparse helper for positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return ivalue


def _print_progress(
    processed: int,
    total: int,
    found: int,
    saved_total: int,
    skipped: int,
    errors: int,
    start_time: float,
    dry_run: bool,
) -> None:
    """Print a compact progress summary."""
    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    saved_label = "[DRY RUN]" if dry_run else str(saved_total)
    print(
        f"  Progress {processed}/{total} ({processed / total:.0%}) | "
        f"Found: {found} | Saved: {saved_label} | "
        f"No kortTittel: {skipped} | Errors: {errors} | "
        f"ETA: {eta_seconds / 60:.1f} min"
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill kort_tittel for content")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--workers", type=positive_int, default=10, help="Parallel API workers (default: 10)")
    parser.add_argument("--batch-size", type=positive_int, default=50, help="DB write batch size (default: 50)")
    parser.add_argument("--limit", type=int, default=0, help="Max items to process (0 = all)")
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=100,
        help="Print progress summary every N processed items (default: 100)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BACKFILL SHORT TITLES")
    print("=" * 60)

    print("\n[1/3] Ensuring database column exists...")
    ensure_short_title_column(dry_run=args.dry_run)

    if args.dry_run and not has_short_title_column():
        print("\nDry run stops before querying content because kort_tittel does not exist yet.")
        return

    print("\n[2/3] Finding content missing kort_tittel...")
    content_ids = get_content_to_backfill(limit=args.limit)
    print(f"  Found {len(content_ids)} items missing short title")

    if not content_ids:
        print("\nNothing to backfill.")
        return

    print(f"\n[3/3] Fetching short titles from API ({args.workers} workers)...")
    if args.dry_run:
        print("  [DRY RUN - no DB writes]")

    total = len(content_ids)
    start_time = time.monotonic()
    found = 0
    saved_total = 0
    skipped = 0
    errors = 0
    pending_updates = []

    for chunk_start in range(0, total, args.workers):
        chunk = content_ids[chunk_start:chunk_start + args.workers]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_short_title, content_id): content_id
                for content_id in chunk
            }

            for index, future in enumerate(as_completed(futures), start=chunk_start + 1):
                content_id = futures[future]
                try:
                    payload = future.result()
                    short_title = payload.get("kortTittel")
                    if short_title:
                        found += 1
                        if not args.dry_run:
                            pending_updates.append((content_id, payload))
                    else:
                        skipped += 1
                except HelseDirectorateAPIError as exc:
                    errors += 1
                    print(f"  [{index}/{total}] {content_id} ERROR (API): {exc}")
                except Exception as exc:
                    errors += 1
                    print(f"  [{index}/{total}] {content_id} ERROR: {type(exc).__name__}: {exc}")

                if index % args.progress_every == 0 or index == total:
                    _print_progress(
                        processed=index,
                        total=total,
                        found=found,
                        saved_total=saved_total,
                        skipped=skipped,
                        errors=errors,
                        start_time=start_time,
                        dry_run=args.dry_run,
                    )

        if not args.dry_run and len(pending_updates) >= args.batch_size:
            saved_total += flush_pending_updates(pending_updates, args.batch_size)

        time.sleep(0.3)

    if not args.dry_run and pending_updates:
        for attempt in range(1, MAX_BATCH_SAVE_RETRIES + 1):
            saved = save_short_titles_batch(pending_updates)
            if saved > 0:
                saved_total += saved
                print(f"  Final batch saved: {saved} items")
                pending_updates.clear()
                break

            if attempt == MAX_BATCH_SAVE_RETRIES:
                raise RuntimeError(
                    f"Final batch save failed after {MAX_BATCH_SAVE_RETRIES} attempts for {len(pending_updates)} items"
                )

            print(f"  WARNING: Final batch save returned 0, retrying ({attempt}/{MAX_BATCH_SAVE_RETRIES})")

    elapsed = time.monotonic() - start_time
    print()
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Total checked: {total}")
    print(f"  kortTittel found: {found}")
    print(f"  Saved to DB: {saved_total if not args.dry_run else '[DRY RUN]'}")
    print(f"  No kortTittel in API: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
