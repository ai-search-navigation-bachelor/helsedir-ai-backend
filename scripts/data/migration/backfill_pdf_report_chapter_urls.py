#!/usr/bin/env python3
"""
Backfill document_url for PDF report chapter pages.

This script targets chapter rows whose public pages expose a resolved PDF link,
typically paths ending with /pdf-av-rapporten. The PDF URL is scraped from the
public Helsedirektoratet HTML page and stored in content.document_url.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.services.repositories.base import db_pool  # noqa: E402
from app.services.data.document_metadata import compute_document_metadata_with_fallback  # noqa: E402


def _require_db_connection(operation: str):
    conn = db_pool.get_connection()
    if not conn:
        message = f"ERROR: Database connection unavailable during {operation}"
        print(message)
        raise RuntimeError(message)
    return conn


def _fetch_rows(limit: int = 0, force: bool = False) -> List[Dict]:
    conn = _require_db_connection("row fetch for PDF chapter backfill")

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT id, tittel, tekst, path, document_url, has_text_content
            FROM content
            WHERE info_type = 'kapittel'
              AND (
                    path LIKE %s
                 OR path LIKE %s
              )
        """
        params: List[object] = ["%/pdf-av-rapporten", "%/pdf-versjon-av-rapporten"]
        if not force:
            query += " AND document_url IS NULL"
        query += " ORDER BY id"
        if limit > 0:
            query += " LIMIT %s"
            params.append(limit)
        cursor.execute(query, tuple(params))
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        conn.close()


def _apply_updates(updates: List[Tuple[Optional[str], int, str]]) -> int:
    if not updates:
        return 0

    conn = _require_db_connection("content update for PDF chapter backfill")

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """
            UPDATE content
            SET document_url = %s,
                has_text_content = %s
            WHERE id = %s
            """,
            updates,
        )
        conn.commit()
        return cursor.rowcount
    finally:
        if cursor:
            cursor.close()
        conn.close()


def _resolve_row(row: Dict) -> Tuple[str, Optional[str], int]:
    meta = compute_document_metadata_with_fallback(row, timeout=20.0)
    return row["id"], meta["document_url"], int(meta["has_text_content"])


def main():
    parser = argparse.ArgumentParser(description="Backfill PDF URLs for PDF report chapter pages")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to the database")
    parser.add_argument("--force", action="store_true", help="Recompute rows even when document_url is already set")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel HTTP workers")
    parser.add_argument("--batch-size", type=int, default=50, help="DB update batch size")
    args = parser.parse_args()

    rows = _fetch_rows(limit=max(0, args.limit), force=args.force)
    if not rows:
        print("No PDF report chapter rows need backfill")
        return

    print(f"Loaded {len(rows)} PDF report chapter rows")
    if args.dry_run:
        print("Dry-run enabled: no database writes will be made")

    processed = 0
    resolved = 0
    warnings = 0
    written = 0
    pending_updates: List[Tuple[Optional[str], int, str]] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_resolve_row, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            processed += 1
            try:
                content_id, document_url, has_text_content = future.result()
                if document_url:
                    resolved += 1
                else:
                    warnings += 1
                    print(f"WARN {content_id}: no PDF URL found for path={row.get('path')}")

                pending_updates.append((document_url, has_text_content, content_id))
            except (httpx.HTTPError, RuntimeError, ValueError, TypeError) as exc:
                warnings += 1
                print(f"WARN {row['id']}: {type(exc).__name__}: {exc}")
                continue

            if len(pending_updates) >= max(1, args.batch_size):
                if not args.dry_run:
                    written += _apply_updates(pending_updates)
                pending_updates.clear()

            elapsed = max(time.perf_counter() - started, 0.001)
            rate = processed / elapsed
            print(
                f"Progress: {processed}/{len(rows)} ({processed / len(rows):.1%}), "
                f"resolved={resolved}, warnings={warnings}, rate={rate:.1f} rows/s"
            )

    if pending_updates and not args.dry_run:
        written += _apply_updates(pending_updates)

    print(
        f"Done. processed={processed}, resolved={resolved}, warnings={warnings}, "
        f"written={written if not args.dry_run else 0}"
    )


if __name__ == "__main__":
    main()
