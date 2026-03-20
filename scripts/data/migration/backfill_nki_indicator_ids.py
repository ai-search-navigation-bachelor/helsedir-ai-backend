#!/usr/bin/env python3
"""
Backfill content.nki_indicator_id by matching NKI indicators to content pages.

Matching priority:
1. Public URL/path embedded in the indicator payload
2. Exact title
3. Normalized title

Ambiguous matches are skipped.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.exceptions.helsedir import HelseDirectorateAPIError
from app.services.external.helsedir_api_service import helsedir_api_service
from app.services.repositories.base import db_pool
from app.services.statistics.nki_matching import build_match_indexes, find_indicator_match_with_indexes


def has_nki_indicator_column() -> bool:
    """Return True when content.nki_indicator_id exists."""
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM content LIKE 'nki_indicator_id'")
        return cursor.fetchone() is not None
    finally:
        if cursor:
            cursor.close()
        conn.close()


def ensure_nki_indicator_column(dry_run: bool = False) -> None:
    """Add the nki_indicator_id column if it does not exist."""
    if has_nki_indicator_column():
        print("Column nki_indicator_id already exists.")
        return

    if dry_run:
        print("Column nki_indicator_id is MISSING (would be added without --dry-run)")
        return

    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()
        print("Adding column nki_indicator_id...")
        cursor.execute("ALTER TABLE content ADD COLUMN nki_indicator_id VARCHAR(32) NULL AFTER document_url")
        cursor.execute("CREATE INDEX idx_nki_indicator_id ON content (nki_indicator_id)")
        conn.commit()
        print("  Done.")
    finally:
        if cursor:
            cursor.close()
        conn.close()


def get_content_rows() -> List[Dict[str, Any]]:
    """Fetch candidate content rows for matching."""
    conn = db_pool.get_connection()
    if not conn:
        return []

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, tittel, kort_tittel, path, info_type, nki_indicator_id
            FROM content
            WHERE info_type IS NULL OR info_type <> 'temaside'
            ORDER BY id
            """
        )
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        conn.close()


def _normalize_indicator_list(payload: Any) -> List[Dict[str, Any]]:
    """Handle list and common wrapped-list payload formats."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("items", "value", "results", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def fetch_indicators(limit: int = 0) -> List[Dict[str, Any]]:
    """Fetch indicators from the Helsedir API."""
    payload = helsedir_api_service.list_nki_quality_indicators(timeout=30.0)
    indicators = _normalize_indicator_list(payload)
    if limit > 0:
        return indicators[:limit]
    return indicators


def save_matches_batch(updates: List[Dict[str, Any]]) -> int:
    """Save a batch of content -> nki_indicator_id assignments."""
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
            SET nki_indicator_id = %s
            WHERE id = %s
            """,
            [(update["indicator_id"], update["content_id"]) for update in updates],
        )
        conn.commit()
        return cursor.rowcount
    finally:
        if cursor:
            cursor.close()
        conn.close()


def _print_skip_summary(skipped: List[Dict[str, Any]], limit: int = 20) -> None:
    if not skipped:
        print("  No skipped indicators")
        return

    reason_counts = Counter(item.get("reason", "unknown") for item in skipped)
    print("  Skipped reasons:")
    for reason, count in reason_counts.most_common():
        print(f"    {reason}: {count}")

    print(f"  Showing first {min(limit, len(skipped))} skipped items:")
    for item in skipped[:limit]:
        print(f"    {item}")


def _print_progress(
    *,
    processed: int,
    total: int,
    matched: int,
    skipped: int,
    start_time: float,
) -> None:
    """Print a compact progress summary with ETA."""
    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    percent = (processed / total * 100.0) if total else 100.0
    print(
        f"  Progress {processed}/{total} ({percent:.0f}%) | "
        f"Planned: {matched} | Skipped: {skipped} | "
        f"Rate: {rate:.1f}/s | ETA: {eta_seconds:.1f}s"
    )


def _match_indicators_with_progress(
    indicators: List[Dict[str, Any]],
    content_rows: List[Dict[str, Any]],
    *,
    force: bool,
    progress_every: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match indicators while streaming progress to stdout."""
    indexes = build_match_indexes(content_rows)
    rows_by_id = {row["id"]: row for row in content_rows if row.get("id")}
    updates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    preliminary_matches: Dict[str, List[Dict[str, Any]]] = {}
    total = len(indicators)
    start_time = time.monotonic()

    for index, indicator in enumerate(indicators, start=1):
        indicator_id = indicator.get("id")
        result = find_indicator_match_with_indexes(indicator, indexes)
        if result["match"] is None:
            skipped.append(
                {
                    "indicator_id": indicator_id,
                    "title": indicator.get("tittel") or indicator.get("title"),
                    "reason": result["reason"],
                }
            )
        else:
            content_row = result["match"]
            preliminary_matches.setdefault(content_row["id"], []).append(
                {
                    "content_id": content_row["id"],
                    "indicator_id": indicator_id,
                    "strategy": result["strategy"],
                    "title": indicator.get("tittel") or indicator.get("title"),
                }
            )

        if index % progress_every == 0 or index == total:
            matched_count = sum(len(items) for items in preliminary_matches.values())
            _print_progress(
                processed=index,
                total=total,
                matched=matched_count,
                skipped=len(skipped),
                start_time=start_time,
            )

    for content_id, matches in preliminary_matches.items():
        if len(matches) > 1:
            skipped.append(
                {
                    "content_id": content_id,
                    "indicator_ids": [match["indicator_id"] for match in matches],
                    "reason": "content_conflict",
                }
            )
            continue

        match = matches[0]
        row = rows_by_id.get(content_id, {})
        existing_indicator_id = row.get("nki_indicator_id")
        if existing_indicator_id == match["indicator_id"]:
            continue
        if existing_indicator_id and not force:
            skipped.append(
                {
                    "content_id": content_id,
                    "indicator_id": match["indicator_id"],
                    "existing_indicator_id": existing_indicator_id,
                    "reason": "existing_mapping",
                }
            )
            continue

        updates.append(match)

    updates.sort(key=lambda item: (item["content_id"], item["indicator_id"]))
    skipped.sort(key=lambda item: str(item))
    return updates, skipped


def main():
    parser = argparse.ArgumentParser(description="Backfill NKI indicator IDs onto content rows")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--force", action="store_true", help="Overwrite existing nki_indicator_id values")
    parser.add_argument("--limit", type=int, default=0, help="Max indicators to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=200, help="DB write batch size")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N indicators during matching (default: 50)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BACKFILL NKI INDICATOR IDS")
    print("=" * 60)

    print("\n[1/4] Ensuring database column exists...")
    ensure_nki_indicator_column(dry_run=args.dry_run)

    if args.dry_run and not has_nki_indicator_column():
        print("\nDry run stops before querying content because nki_indicator_id does not exist yet.")
        return

    print("\n[2/4] Loading candidate content rows...")
    content_rows = get_content_rows()
    print(f"  Loaded {len(content_rows)} content rows")

    if not content_rows:
        print("No content rows available.")
        return

    print("\n[3/4] Fetching NKI indicators from API...")
    try:
        indicators = fetch_indicators(limit=args.limit)
    except HelseDirectorateAPIError as exc:
        print(f"ERROR: Failed to fetch indicators: {exc}")
        sys.exit(1)

    print(f"  Loaded {len(indicators)} indicators")
    if not indicators:
        print("No indicators returned from API.")
        return

    print("\n[4/4] Matching indicators to content...")
    updates, skipped = _match_indicators_with_progress(
        indicators,
        content_rows,
        force=args.force,
        progress_every=max(1, args.progress_every),
    )

    print(f"  Planned updates: {len(updates)}")
    print(f"  Skipped: {len(skipped)}")

    if args.dry_run:
        print("\n[DRY RUN] First planned updates:")
        for update in updates[:20]:
            print(f"  {update}")
        _print_skip_summary(skipped)
        return

    saved_total = 0
    batch_size = max(1, args.batch_size)
    for start in range(0, len(updates), batch_size):
        saved_total += save_matches_batch(updates[start:start + batch_size])

    print(f"\nSaved {saved_total} mappings to the database.")
    _print_skip_summary(skipped)


if __name__ == "__main__":
    main()
