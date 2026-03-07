#!/usr/bin/env python3
"""
Backfill has_text_content and document_url for existing content rows.

Strategy:
- Compute has_text_content locally from stored tekst/body.
- Only call Helsedirektoratet API for rows that have no text and no document_url.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.services.external.helsedir_api_service import (  # noqa: E402
    helsedir_api_service,
    HelseDirectorateAPIError,
)
from app.services.repositories.base import db_pool  # noqa: E402
from app.services.data.document_metadata import compute_document_metadata  # noqa: E402


def _fetch_rows(limit: int = 0, force: bool = False) -> List[Dict]:
    conn = db_pool.get_connection()
    if not conn:
        return []

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT id, tekst, has_text_content, document_url
            FROM content
        """
        if not force:
            query += """
            WHERE (has_text_content IS NULL OR has_text_content = 0)
              AND document_url IS NULL
            """
        query += " ORDER BY id"
        if limit > 0:
            query += " LIMIT %s"
            cursor.execute(query, (limit,))
        else:
            cursor.execute(query)
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        conn.close()


def _apply_updates(updates: List[Tuple[int, Optional[str], str]]) -> int:
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
            SET has_text_content = %s,
                document_url = %s
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


def _apply_updates_in_batches(
    updates: List[Tuple[int, Optional[str], str]],
    batch_size: int,
    label: str,
) -> int:
    """Apply updates in batches and print progress along the way."""
    if not updates:
        return 0

    total = len(updates)
    updated_total = 0
    started = time.perf_counter()

    for start in range(0, total, batch_size):
        batch = updates[start:start + batch_size]
        updated_total += _apply_updates(batch)
        done = min(start + len(batch), total)
        elapsed = max(time.perf_counter() - started, 0.001)
        rate = done / elapsed
        print(
            f"{label}: {done}/{total} "
            f"({done / total:.1%}) at {rate:.1f} rows/s"
        )

    return updated_total


def _fetch_document_metadata(content_id: str) -> Tuple[str, Optional[str]]:
    detailed = helsedir_api_service.get_infobit_by_id(content_id, timeout=15.0)
    meta = compute_document_metadata(detailed)
    return content_id, meta["document_url"]


def main():
    parser = argparse.ArgumentParser(description="Backfill content document metadata")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API workers")
    parser.add_argument("--batch-size", type=int, default=500, help="DB update batch size")
    parser.add_argument("--progress-every", type=int, default=100, help="Print API progress every N items")
    parser.add_argument("--force", action="store_true", help="Recompute all rows")
    args = parser.parse_args()

    rows = _fetch_rows(limit=args.limit, force=args.force)
    if not rows:
        print("No rows need backfill")
        return

    print(f"Loaded {len(rows)} rows")
    print("Computing local metadata...")

    local_updates: List[Tuple[int, Optional[str], str]] = []
    api_candidates: List[str] = []

    for index, row in enumerate(rows, start=1):
        meta = compute_document_metadata(row)
        has_text = int(meta["has_text_content"])
        document_url = row.get("document_url") or meta["document_url"]
        local_updates.append((has_text, document_url, row["id"]))

        if not has_text and not document_url:
            api_candidates.append(row["id"])

        if index % 5000 == 0 or index == len(rows):
            print(
                f"Prepared local metadata for {index}/{len(rows)} rows; "
                f"API lookups needed so far: {len(api_candidates)}"
            )

    updated = _apply_updates_in_batches(
        local_updates,
        batch_size=max(1, args.batch_size),
        label="Applied local updates",
    )
    print(f"Applied local updates to {updated} rows")

    if not api_candidates:
        print("No API lookups needed")
        return

    print(f"Fetching document_url from API for {len(api_candidates)} empty-text rows")
    api_updates: List[Tuple[int, Optional[str], str]] = []
    processed = 0
    found_document_url = 0
    total_updated = 0
    started = time.perf_counter()
    progress_every = max(1, args.progress_every)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_fetch_document_metadata, content_id): content_id
            for content_id in api_candidates
        }
        for future in as_completed(futures):
            content_id = futures[future]
            try:
                _, document_url = future.result()
                api_updates.append((0, document_url, content_id))
                if document_url:
                    found_document_url += 1
            except HelseDirectorateAPIError as exc:
                print(f"WARN {content_id}: {exc}")
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                print(f"WARN {content_id}: {type(exc).__name__}: {exc}")

            processed += 1

            if len(api_updates) >= max(1, args.batch_size):
                total_updated += _apply_updates(api_updates)
                api_updates.clear()

            if processed % progress_every == 0 or processed == len(api_candidates):
                elapsed = max(time.perf_counter() - started, 0.001)
                rate = processed / elapsed
                remaining = len(api_candidates) - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                print(
                    f"API progress: {processed}/{len(api_candidates)} "
                    f"({processed / len(api_candidates):.1%}), "
                    f"found document_url for {found_document_url}, "
                    f"ETA ~{eta_seconds / 60:.1f} min"
                )

    total_updated += _apply_updates(api_updates)
    print(f"Applied remaining API updates to {total_updated} rows")


if __name__ == "__main__":
    main()
