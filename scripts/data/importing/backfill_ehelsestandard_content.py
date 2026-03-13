#!/usr/bin/env python3
"""
Backfill e-helsestandard content from Helsedirektoratet file payloads.

Updates existing `content` rows using mostly existing columns:
- tekst
- has_text_content
- document_url
- forst_publisert
- sist_faglig_oppdatert
- attachments_json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.config import settings  # noqa: E402
from app.services.external.helsedir_api_service import (  # noqa: E402
    HelseDirectorateAPIError,
    helsedir_api_service,
)
from app.services.repositories.base import db_pool  # noqa: E402
from app.services.data.document_metadata import has_visible_text  # noqa: E402


def _require_db_connection(operation: str):
    conn = db_pool.get_connection()
    if not conn:
        raise RuntimeError(f"Database connection unavailable during {operation}")
    return conn


def _resolve_attachment_url(file_uri: Optional[str]) -> Optional[str]:
    if not file_uri or not file_uri.strip():
        return None

    normalized = file_uri.strip()
    parsed = urlparse(normalized)
    if parsed.scheme and parsed.netloc:
        return normalized

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return urljoin(settings.helsedir_public_base_url, normalized)


def _normalize_attachments(payload: Dict) -> List[Dict[str, Optional[str]]]:
    attachments = payload.get("attachments")
    if not isinstance(attachments, list):
        return []

    normalized = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        url = _resolve_attachment_url(
            attachment.get("fileUri")
            if isinstance(attachment.get("fileUri"), str)
            else attachment.get("file_uri")
            if isinstance(attachment.get("file_uri"), str)
            else attachment.get("url")
            if isinstance(attachment.get("url"), str)
            else attachment.get("href")
            if isinstance(attachment.get("href"), str)
            else None
        )
        if not url:
            continue
        title = (
            attachment.get("title")
            if isinstance(attachment.get("title"), str)
            else attachment.get("tittel")
            if isinstance(attachment.get("tittel"), str)
            else attachment.get("name")
            if isinstance(attachment.get("name"), str)
            else url.rstrip("/").rsplit("/", 1)[-1]
        )
        normalized.append(
            {
                "title": title.strip(),
                "url": url,
                "file_type": attachment.get("fileType")
                if isinstance(attachment.get("fileType"), str)
                else attachment.get("file_type")
                if isinstance(attachment.get("file_type"), str)
                else attachment.get("type")
                if isinstance(attachment.get("type"), str)
                else None,
            }
        )
    return normalized


def _fetch_rows(limit: int = 0, force: bool = False) -> List[Dict]:
    conn = _require_db_connection("fetching ehelsestandard rows")
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT id, tekst, has_text_content, document_url, attachments_json,
                   forst_publisert, sist_faglig_oppdatert
            FROM content
            WHERE info_type = 'ehelsestandard'
        """
        if not force:
            query += """
              AND (
                  attachments_json IS NULL
                  OR document_url IS NULL
                  OR has_text_content IS NULL
                  OR has_text_content = 0
                  OR forst_publisert IS NULL
                  OR sist_faglig_oppdatert IS NULL
              )
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


def _compute_update(row: Dict, payload: Dict) -> Tuple[str, str, int, Optional[str], Optional[str], Optional[str], str]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    purpose_html = data.get("formalBruksomrade") if isinstance(data.get("formalBruksomrade"), str) else None
    existing_text = row.get("tekst") or ""
    final_text = existing_text if has_visible_text(existing_text) else (purpose_html or existing_text)
    attachments = _normalize_attachments(payload)
    document_url = attachments[0]["url"] if attachments else row.get("document_url")
    has_text_content = int(has_visible_text(final_text))
    attachments_json = json.dumps(attachments, ensure_ascii=False)

    return (
        final_text,
        has_text_content,
        document_url,
        payload.get("forstPublisert") or row.get("forst_publisert"),
        payload.get("sistFagligOppdatert") or row.get("sist_faglig_oppdatert"),
        attachments_json,
        row["id"],
    )


def _fetch_and_compute(row: Dict):
    payload = helsedir_api_service.get_file_by_id(row["id"], timeout=15.0)
    return _compute_update(row, payload)


def _apply_updates(updates: List[Tuple[str, int, Optional[str], Optional[str], Optional[str], str, str]]) -> int:
    if not updates:
        return 0

    conn = _require_db_connection("updating ehelsestandard rows")
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """
            UPDATE content
            SET tekst = %s,
                has_text_content = %s,
                document_url = %s,
                forst_publisert = %s,
                sist_faglig_oppdatert = %s,
                attachments_json = %s
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


def main():
    parser = argparse.ArgumentParser(description="Backfill e-helsestandard content metadata")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API workers")
    parser.add_argument("--batch-size", type=int, default=200, help="DB update batch size")
    parser.add_argument("--force", action="store_true", help="Re-fetch all ehelsestandard rows")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    rows = _fetch_rows(limit=args.limit, force=args.force)
    if not rows:
        print("No ehelsestandard rows need backfill")
        return

    print(f"Loaded {len(rows)} ehelsestandard rows")
    updates = []
    processed = 0
    errors = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_fetch_and_compute, row): row["id"] for row in rows}
        for future in as_completed(futures):
            content_id = futures[future]
            try:
                updates.append(future.result())
            except HelseDirectorateAPIError as exc:
                errors += 1
                print(f"WARN {content_id}: {exc}")
            except Exception as exc:  # pragma: no cover - defensive batch logging
                errors += 1
                print(f"ERROR {content_id}: {type(exc).__name__}: {exc}")

            processed += 1
            if processed % 50 == 0 or processed == len(rows):
                elapsed = max(time.perf_counter() - started, 0.001)
                print(f"Processed {processed}/{len(rows)} rows at {processed / elapsed:.1f} rows/s")

    if args.dry_run:
        print(f"[DRY RUN] Prepared {len(updates)} updates, {errors} errors")
        return

    updated_total = 0
    for start in range(0, len(updates), max(1, args.batch_size)):
        batch = updates[start:start + max(1, args.batch_size)]
        updated_total += _apply_updates(batch)

    print(f"Updated {updated_total} rows, {errors} errors")


if __name__ == "__main__":
    main()
