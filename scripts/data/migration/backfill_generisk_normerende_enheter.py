#!/usr/bin/env python3
"""
Backfill generisk normerende enheter (GNE) into the local content hierarchy.

Why this exists:
- Some Helsedirektoratet veiledere expose real content pages as
  `generisk-normerende-enhet` objects.
- These pages are not linked back to chapters/veiledere through ordinary
  `forelder` / `barn` relations in the public API.
- Frontend expects one complete tree from backend.

This script imports GNE content into `content`, then synthesizes internal links:
- GNE -> `forelder` points to the closest existing ancestor by URL path
- GNE -> `root` / `publikasjon` point to the root publication in that tree
- Parent -> `barn` points back to the GNE

Matching rule:
- IDs are used for storage
- URL path prefix is used to decide placement in the hierarchy

Usage:
    python scripts/data/migration/backfill_generisk_normerende_enheter.py
    python scripts/data/migration/backfill_generisk_normerende_enheter.py --dry-run
    python scripts/data/migration/backfill_generisk_normerende_enheter.py --path-prefix /veiledere/ledelse-og-kvalitetsforbedring-i-helse-og-omsorgstjenesten
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from app.services.external.helsedir_api_service import helsedir_api_service  # noqa: E402
from app.services.repositories.base import db_pool  # noqa: E402
from app.services.data.database_service import database_service  # noqa: E402
from scripts.data.importing.link_utils import extract_helsedir_path  # noqa: E402


GNE_INFO_TYPE = "generisk-normerende-enhet"
REL_ROOTS = {"root", "publikasjon"}
REL_PARENT = "forelder"
REL_CHILD = "barn"
LINK_UPDATE_BATCH_SIZE = 200
UPSERT_BATCH_SIZE = 200


def _normalize_path(path: Optional[str]) -> Optional[str]:
    """Normalize a public Helsedir path."""
    if not path:
        return None
    value = path.strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return extract_helsedir_path(value)
    if not value.startswith("/"):
        value = f"/{value}"
    return value.rstrip("/") or "/"


def _parse_links(raw_links: Any) -> List[Dict[str, Any]]:
    """Parse a links column or payload field into a list."""
    if raw_links is None:
        return []
    if isinstance(raw_links, list):
        return [link for link in raw_links if isinstance(link, dict)]
    if isinstance(raw_links, str):
        try:
            data = json.loads(raw_links)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [link for link in data if isinstance(link, dict)]
    return []


def _serialize_links(links: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(list(links), ensure_ascii=False)


def _content_type_from_payload(item: Dict[str, Any]) -> str:
    """Resolve info type from either top-level or tekniskeData."""
    value = item.get("infoType")
    if isinstance(value, str) and value.strip():
        return value.strip()
    technical = item.get("tekniskeData")
    if isinstance(technical, dict):
        value = technical.get("infoType")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return item.get("info_type") or ""


def _path_depth(path: Optional[str]) -> int:
    normalized = _normalize_path(path)
    if not normalized or normalized == "/":
        return 0
    return len([segment for segment in normalized.split("/") if segment])


def _iter_parent_paths(path: str) -> Iterable[str]:
    """Yield parent paths from nearest to farthest."""
    current = _normalize_path(path)
    if not current or current == "/":
        return
    while current and current != "/":
        parent = current.rsplit("/", 1)[0] or "/"
        yield parent
        if parent == "/":
            break
        current = parent


def _find_best_parent(path: str, path_to_row: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the nearest ancestor row by path."""
    for candidate_path in _iter_parent_paths(path):
        candidate = path_to_row.get(candidate_path)
        if candidate:
            return candidate
    return None


def _first_link_by_rel(links: Sequence[Dict[str, Any]], rel: str) -> Optional[Dict[str, Any]]:
    for link in links:
        if link.get("rel") == rel:
            return link
    return None


def _make_internal_link(rel: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rel": rel,
        "type": row.get("info_type") or row.get("type") or "",
        "tittel": row.get("tittel"),
        "id": row.get("id"),
    }


def _resolve_root_publication(parent_row: Dict[str, Any], rows_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resolve the publication root for a hierarchy node.

    Prefers explicit `root` / `publikasjon` links when they point outside the
    current item. Falls back to the highest reachable ancestor.
    """
    current = parent_row
    last_known = parent_row
    visited = set()

    while current and current.get("id") not in visited:
        current_id = current.get("id")
        if current_id:
            visited.add(current_id)
            last_known = current

        links = _parse_links(current.get("links"))
        for rel in ("root", "publikasjon"):
            link = _first_link_by_rel(links, rel)
            if not link:
                continue
            target_id = link.get("id")
            if target_id and target_id != current_id:
                target_row = rows_by_id.get(target_id)
                if target_row:
                    return target_row
                return {
                    "id": target_id,
                    "tittel": link.get("tittel"),
                    "info_type": link.get("type") or "",
                    "links": [],
                }

        parent_link = _first_link_by_rel(links, REL_PARENT)
        if not parent_link or not parent_link.get("id"):
            break
        current = rows_by_id.get(parent_link["id"])

    return last_known


def _build_synthetic_gne_links(
    item: Dict[str, Any],
    parent_row: Dict[str, Any],
    root_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Replace self-referential nav links with backend-owned hierarchy links."""
    preserved = []
    for link in _parse_links(item.get("links")):
        rel = link.get("rel")
        if rel in REL_ROOTS or rel == REL_PARENT:
            continue
        preserved.append(link)

    return preserved + [
        _make_internal_link(REL_PARENT, parent_row),
        _make_internal_link("root", root_row),
        _make_internal_link("publikasjon", root_row),
    ]


def _ensure_child_link(parent_row: Dict[str, Any], child_row: Dict[str, Any]) -> bool:
    """Append a `barn` link if it does not already exist."""
    links = _parse_links(parent_row.get("links"))
    child_id = child_row.get("id")
    for link in links:
        if link.get("rel") == REL_CHILD and link.get("id") == child_id:
            return False

    links.append(_make_internal_link(REL_CHILD, child_row))
    parent_row["links"] = links
    return True


def _load_existing_rows() -> List[Dict[str, Any]]:
    conn = db_pool.get_connection()
    if not conn:
        raise RuntimeError("Database connection unavailable")

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, tittel, info_type, path, links
            FROM content
            """
        )
        rows = cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        conn.close()

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "id": row.get("id"),
                "tittel": row.get("tittel"),
                "info_type": row.get("info_type") or "",
                "path": _normalize_path(row.get("path")),
                "links": _parse_links(row.get("links")),
            }
        )
    return normalized_rows


def _fetch_gne_payloads(path_prefix: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    results = helsedir_api_service.search_infobits(
        query_text=None,
        filter_query=f"infoType eq '{GNE_INFO_TYPE}'",
        get_full_infobits=True,
        timeout=60.0,
    )
    if not isinstance(results, list):
        return []

    normalized_prefix = _normalize_path(path_prefix) if path_prefix else None
    items = []
    for item in results:
        path = _normalize_path(item.get("path") or item.get("url"))
        if normalized_prefix and not (path and (path == normalized_prefix or path.startswith(f"{normalized_prefix}/"))):
            continue
        item["path"] = path
        items.append(item)

    items.sort(key=lambda row: (_path_depth(row.get("path")), row.get("path") or "", row.get("id") or ""))
    if limit is not None and limit >= 0:
        return items[:limit]
    return items


def _plan_backfill(
    existing_rows: Sequence[Dict[str, Any]],
    gne_items: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Prepare upserts and link updates without touching the database.

    Returns:
    - items to upsert into content
    - row_id -> links to persist
    - skipped payloads with reason metadata
    """
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    path_to_row: Dict[str, Dict[str, Any]] = {}

    for row in existing_rows:
        row_copy = {
            "id": row.get("id"),
            "tittel": row.get("tittel"),
            "info_type": row.get("info_type") or "",
            "path": _normalize_path(row.get("path")),
            "links": _parse_links(row.get("links")),
        }
        if row_copy["id"]:
            rows_by_id[row_copy["id"]] = row_copy
        if row_copy["path"]:
            path_to_row[row_copy["path"]] = row_copy

    items_to_upsert: List[Dict[str, Any]] = []
    changed_rows: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []

    for item in gne_items:
        item_id = item.get("id")
        info_type = _content_type_from_payload(item)
        path = _normalize_path(item.get("path") or item.get("url"))

        if not item_id or not path:
            skipped.append({
                "id": item_id,
                "path": path,
                "reason": "missing_id_or_path",
            })
            continue

        if info_type != GNE_INFO_TYPE:
            skipped.append({
                "id": item_id,
                "path": path,
                "reason": f"unexpected_info_type:{info_type or 'missing'}",
            })
            continue

        parent_row = _find_best_parent(path, path_to_row)
        if not parent_row:
            skipped.append({
                "id": item_id,
                "path": path,
                "reason": "no_parent_match",
            })
            continue

        root_row = _resolve_root_publication(parent_row, rows_by_id)
        synthetic_links = _build_synthetic_gne_links(item, parent_row, root_row)

        upsert_item = dict(item)
        upsert_item["info_type"] = info_type
        upsert_item["path"] = path
        upsert_item["links"] = synthetic_links
        items_to_upsert.append(upsert_item)

        row_state = rows_by_id.get(item_id, {})
        row_state.update(
            {
                "id": item_id,
                "tittel": item.get("tittel"),
                "info_type": info_type,
                "path": path,
                "links": synthetic_links,
            }
        )
        rows_by_id[item_id] = row_state
        path_to_row[path] = row_state
        changed_rows[item_id] = row_state

        if _ensure_child_link(parent_row, row_state):
            changed_rows[parent_row["id"]] = parent_row

    link_updates = {
        row_id: _parse_links(row.get("links"))
        for row_id, row in changed_rows.items()
        if row_id
    }
    return items_to_upsert, link_updates, skipped


def _save_link_updates(link_updates: Dict[str, List[Dict[str, Any]]]) -> int:
    """Persist latest links JSON for the affected rows."""
    if not link_updates:
        return 0

    conn = db_pool.get_connection()
    if not conn:
        raise RuntimeError("Database connection unavailable")

    cursor = None
    updated = 0
    try:
        cursor = conn.cursor()
        pending = 0
        for row_id, links in link_updates.items():
            cursor.execute(
                "UPDATE content SET links = %s WHERE id = %s",
                (_serialize_links(links), row_id),
            )
            updated += 1
            pending += 1
            if pending >= LINK_UPDATE_BATCH_SIZE:
                conn.commit()
                pending = 0
        if pending:
            conn.commit()
    finally:
        if cursor:
            cursor.close()
        conn.close()

    return updated


def _chunked(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _save_upserts(items_to_upsert: Sequence[Dict[str, Any]]) -> int:
    """Persist GNE payloads using the existing content cache path."""
    saved = 0
    for batch in _chunked(list(items_to_upsert), UPSERT_BATCH_SIZE):
        saved += database_service.cache_content_batch(list(batch))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill generisk normerende enheter into the local content hierarchy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the planned imports and link changes without writing to the database",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help="Only process GNE paths under this public path prefix",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of fetched GNE payloads after filtering",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BACKFILL GENERISK NORMERENDE ENHETER")
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN - no database writes\n")

    if not database_service.is_connected():
        raise SystemExit("Could not connect to database")

    print("Loading existing hierarchy from database...")
    existing_rows = _load_existing_rows()
    print(f"  Loaded {len(existing_rows)} content rows")

    print("\nFetching generisk normerende enheter from Helsedir API...")
    gne_items = _fetch_gne_payloads(path_prefix=args.path_prefix, limit=args.limit)
    print(f"  Fetched {len(gne_items)} candidate GNE payloads")

    print("\nPlanning hierarchy backfill...")
    items_to_upsert, link_updates, skipped = _plan_backfill(existing_rows, gne_items)
    print(f"  Upserts planned: {len(items_to_upsert)}")
    print(f"  Link updates planned: {len(link_updates)}")
    print(f"  Skipped: {len(skipped)}")

    if skipped:
        skipped_by_reason = defaultdict(int)
        for item in skipped:
            skipped_by_reason[item["reason"]] += 1
        print("  Skip reasons:")
        for reason, count in sorted(skipped_by_reason.items()):
            print(f"    - {reason}: {count}")

    if args.dry_run:
        preview = items_to_upsert[:5]
        if preview:
            print("\nPreview of planned inserts:")
            for item in preview:
                parent = _first_link_by_rel(_parse_links(item.get("links")), REL_PARENT)
                print(
                    f"  - {item.get('id')}  path={item.get('path')}  "
                    f"parent={parent.get('id') if parent else 'missing'}"
                )
        return

    print("\nSaving GNE content...")
    saved = _save_upserts(items_to_upsert)
    print(f"  Saved {saved} content rows")

    print("\nSaving hierarchy link updates...")
    updated = _save_link_updates(link_updates)
    print(f"  Updated links for {updated} rows")

    print("\nDone.")
    print("Restart the backend to reload the in-memory content cache.")


if __name__ == "__main__":
    main()
