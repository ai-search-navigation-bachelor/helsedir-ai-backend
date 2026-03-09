#!/usr/bin/env python3
"""
Test script to visualize temaside passages for GPL training.

Shows how temasider look as passages with and without enrichment:
1. Raw passage (no enrichment) — what embedding model sees at runtime
2. Enriched passage (with DB linked content) — what GPL training sees
3. Fallback enrichment (hierarchy only, no DB) — when DB is unavailable

Usage:
    python scripts/test/ml/test_temaside_passages.py

    # Test specific temaside by ID
    python scripts/test/ml/test_temaside_passages.py --id 9999-0001-f36cce21-e174-58d1-b9bd-c9ba150d7863

    # Show all temasider
    python scripts/test/ml/test_temaside_passages.py --all

    # Skip database enrichment (offline mode)
    python scripts/test/ml/test_temaside_passages.py --no-db
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from app.ml.embedding_model import HealthContentEmbedding

THEME_PAGES_PATH = project_root / "data" / "theme_pages.json"
GPL_QUERIES_PATH = project_root / "data" / "gpl_queries.json"


def load_theme_pages() -> List[Dict[str, Any]]:
    with open(THEME_PAGES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gpl_queries() -> Dict[str, List[str]]:
    if not GPL_QUERIES_PATH.exists():
        return {}
    with open(GPL_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hierarchy(theme_pages: List[Dict[str, Any]]):
    """Build parent->children mapping and path->page lookup."""
    path_to_page = {tp["path"].rstrip("/"): tp for tp in theme_pages}
    children_by_id: Dict[str, List[Dict[str, Any]]] = {}

    for tp in theme_pages:
        for link in tp.get("links", []):
            if link.get("rel") == "barn":
                child_path = link["href"].rstrip("/")
                child_page = path_to_page.get(child_path)
                if child_page:
                    children_by_id.setdefault(tp["id"], []).append(child_page)

    return children_by_id, path_to_page


def enrich_from_db(theme_pages: List[Dict[str, Any]], children_by_id: Dict[str, List]) -> Dict[str, List[Dict]]:
    """Try to load linked content from database."""
    try:
        from app.services.repositories.base import db_pool
        conn = db_pool.get_connection()
        if not conn:
            return {}

        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT tpc.theme_page_id, c.id, c.tittel, c.tekst, c.info_type
            FROM theme_page_content tpc
            JOIN content c ON c.id = tpc.content_id
            ORDER BY tpc.theme_page_id, tpc.display_order
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        theme_to_content: Dict[str, List[Dict]] = {}
        for row in rows:
            theme_id = row["theme_page_id"]
            theme_to_content.setdefault(theme_id, []).append({
                "tittel": row["tittel"] or "",
                "tekst": row["tekst"] or "",
                "type": row["info_type"] or "",
            })

        # Also add grandchild content (child temaside's linked content)
        result: Dict[str, List[Dict]] = {}
        for tp in theme_pages:
            linked = list(theme_to_content.get(tp["id"], []))
            for child in children_by_id.get(tp["id"], []):
                linked.extend(theme_to_content.get(child["id"], []))
            if linked:
                result[tp["id"]] = linked

        print(f"  Loaded {len(rows)} theme-content links from DB")
        return result

    except Exception as e:
        print(f"  DB not available: {e}")
        return {}


def enrich_from_hierarchy(
    page: Dict[str, Any],
    children_by_id: Dict[str, List[Dict[str, Any]]],
) -> List[Dict]:
    """Fallback: create linked_content from hierarchy titles only."""
    linked = []
    for child in children_by_id.get(page["id"], []):
        linked.append({
            "tittel": child["tittel"],
            "tekst": "",
            "type": "temaside",
        })
    return linked


def print_header(text: str, char: str = "=", width: int = 90):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_passage(label: str, passage: str, width: int = 90):
    print(f"\n  [{label}] ({len(passage)} chars)")
    print(f"  {'-' * (width - 4)}")
    # Word-wrap at width
    words = passage.split()
    line = "  "
    for word in words:
        if len(line) + len(word) + 1 > width:
            print(line)
            line = "  " + word
        else:
            line = line + " " + word if line.strip() else "  " + word
    if line.strip():
        print(line)
    print(f"  {'-' * (width - 4)}")


def show_temaside(
    page: Dict[str, Any],
    children_by_id: Dict[str, List[Dict[str, Any]]],
    db_content: Dict[str, List[Dict]],
    gpl_queries: Dict[str, List[str]],
    use_db: bool = True,
):
    children = children_by_id.get(page["id"], [])
    is_leaf = len(children) == 0

    print_header(
        f"{page['tittel']}  "
        f"({'leaf' if is_leaf else f'{len(children)} barn'})  "
        f"id: {page['id'][:20]}..."
    )
    print(f"  Path: {page['path']}")
    print(f"  Links: {len(page.get('links', []))}")
    for link in page.get("links", []):
        print(f"    {link['rel']}: {link.get('type', '?')} -> {link['href']}")

    # 1. Raw passage (no enrichment)
    raw_item = {
        "tittel": page["tittel"],
        "info_type": "temaside",
        "tekst": page.get("tekst", ""),
    }
    raw_passage = HealthContentEmbedding.format_passage(raw_item)
    print_passage("RAW (ingen enrichment)", raw_passage)

    # 2. Hierarchy-only enrichment
    hierarchy_linked = enrich_from_hierarchy(page, children_by_id)
    if hierarchy_linked:
        hierarchy_item = {**raw_item, "linked_content": hierarchy_linked}
        hierarchy_passage = HealthContentEmbedding.format_passage(hierarchy_item)
        print_passage(f"HIERARKI (kun barnetitler, {len(hierarchy_linked)} barn)", hierarchy_passage)
    else:
        print(f"\n  [HIERARKI] Ingen barn i hierarkiet")

    # 3. DB enrichment
    if use_db and page["id"] in db_content:
        db_linked = db_content[page["id"]]
        db_item = {**raw_item, "linked_content": db_linked}
        db_passage = HealthContentEmbedding.format_passage(db_item)
        print_passage(f"DATABASE (theme_page_content, {len(db_linked)} lenket)", db_passage)
    elif use_db:
        print(f"\n  [DATABASE] Ingen lenket innhold funnet")

    # 4. GPL queries
    queries = gpl_queries.get(page["id"], [])
    if queries:
        print(f"\n  [GPL QUERIES] ({len(queries)} stk)")
        for i, q in enumerate(queries):
            print(f"    {i+1}. {q}")
    else:
        print(f"\n  [GPL QUERIES] Ingen generert enda")


def main():
    parser = argparse.ArgumentParser(description="Test temaside passage formatting")
    parser.add_argument("--id", type=str, help="Show specific temaside by ID")
    parser.add_argument("--all", action="store_true", help="Show all temasider")
    parser.add_argument("--no-db", action="store_true", help="Skip database enrichment")
    parser.add_argument("--limit", type=int, default=10, help="Max temasider to show (default: 10)")
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export all passages to JSON file (e.g. --export data/temaside_passages.json)",
    )
    args = parser.parse_args()

    print_header("TEMASIDE PASSAGE TEST", "=")

    # Load data
    print("\nLoading data...")
    theme_pages = load_theme_pages()
    gpl_queries = load_gpl_queries()
    children_by_id, path_to_page = build_hierarchy(theme_pages)
    print(f"  Theme pages: {len(theme_pages)}")
    print(f"  GPL queries: {len(gpl_queries)} documents")

    # Count types
    leaves = [tp for tp in theme_pages if not children_by_id.get(tp["id"])]
    parents = [tp for tp in theme_pages if children_by_id.get(tp["id"])]
    print(f"  Leaf temasider (ingen barn): {len(leaves)}")
    print(f"  Parent temasider (har barn): {len(parents)}")

    # DB enrichment
    db_content: Dict[str, List[Dict]] = {}
    if not args.no_db:
        print("\nLoading DB enrichment...")
        db_content = enrich_from_db(theme_pages, children_by_id)
        if db_content:
            print(f"  Enriched temasider: {len(db_content)}")
        else:
            print("  No DB enrichment available (use --no-db to skip)")

    # Select which to show
    if args.id:
        selected = [tp for tp in theme_pages if tp["id"] == args.id]
        if not selected:
            print(f"\nTemaside not found: {args.id}")
            # Fuzzy search
            matches = [tp for tp in theme_pages if args.id.lower() in tp["tittel"].lower()]
            if matches:
                print(f"Did you mean one of these?")
                for tp in matches[:5]:
                    print(f"  {tp['id']}  {tp['tittel']}")
            return
    elif args.all:
        selected = theme_pages
    else:
        # Show a mix: some parents, some leaves, some with DB content
        examples = []
        # Pick parents with most children
        for tp in sorted(parents, key=lambda t: len(children_by_id.get(t["id"], [])), reverse=True)[:3]:
            examples.append(tp)
        # Pick leaves with DB content
        for tp in leaves:
            if tp["id"] in db_content and len(examples) < 6:
                examples.append(tp)
        # Pick leaves without DB content
        for tp in leaves:
            if tp["id"] not in db_content and tp not in examples and len(examples) < 8:
                examples.append(tp)
                break
        # Fill up to limit
        for tp in theme_pages:
            if tp not in examples and len(examples) < args.limit:
                examples.append(tp)
        selected = examples[:args.limit]

    # Export to JSON if requested
    if args.export:
        export_path = Path(args.export)
        if not export_path.is_absolute():
            export_path = project_root / export_path

        print(f"\nExporting passages for all {len(theme_pages)} temasider...")
        export_data = {}
        for page in theme_pages:
            page_id = page["id"]
            children = children_by_id.get(page_id, [])

            raw_item = {
                "tittel": page["tittel"],
                "info_type": "temaside",
                "tekst": page.get("tekst", ""),
            }
            raw_passage = HealthContentEmbedding.format_passage(raw_item)

            entry = {
                "tittel": page["tittel"],
                "path": page["path"],
                "is_leaf": len(children) == 0,
                "num_children": len(children),
                "passages": {
                    "raw": raw_passage,
                    "raw_length": len(raw_passage),
                },
                "gpl_queries": gpl_queries.get(page_id, []),
            }

            # Hierarchy passage
            hierarchy_linked = enrich_from_hierarchy(page, children_by_id)
            if hierarchy_linked:
                h_item = {**raw_item, "linked_content": hierarchy_linked}
                h_passage = HealthContentEmbedding.format_passage(h_item)
                entry["passages"]["hierarchy"] = h_passage
                entry["passages"]["hierarchy_length"] = len(h_passage)

            # DB passage
            if not args.no_db and page_id in db_content:
                db_item = {**raw_item, "linked_content": db_content[page_id]}
                db_passage = HealthContentEmbedding.format_passage(db_item)
                entry["passages"]["database"] = db_passage
                entry["passages"]["database_length"] = len(db_passage)
                entry["passages"]["database_linked_count"] = len(db_content[page_id])

            export_data[page_id] = entry

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"Exported to {export_path}")
    else:
        # Show selected in terminal
        for page in selected:
            show_temaside(page, children_by_id, db_content, gpl_queries, use_db=not args.no_db)

    # Summary
    print_header("OPPSUMMERING")

    if not args.no_db and db_content:
        enriched_leaves = sum(1 for tp in leaves if tp["id"] in db_content)
        unenriched_leaves = len(leaves) - enriched_leaves
        print(f"  Leaves med DB-innhold:    {enriched_leaves}/{len(leaves)}")
        print(f"  Leaves uten DB-innhold:   {unenriched_leaves}/{len(leaves)}")
        if unenriched_leaves > 0:
            print(f"\n  OBS: {unenriched_leaves} leaf-temasider har ingen lenket innhold.")
            print(f"  Disse vil kun ha passage: '<tittel>. Dette er en temaside.'")

    avg_raw = sum(
        len(HealthContentEmbedding.format_passage({
            "tittel": tp["tittel"], "info_type": "temaside", "tekst": tp.get("tekst", "")
        }))
        for tp in theme_pages
    ) / len(theme_pages)
    print(f"\n  Gjennomsnittlig raw passage-lengde: {avg_raw:.0f} tegn")

    temaside_with_queries = sum(1 for tp in theme_pages if tp["id"] in gpl_queries)
    print(f"  Temasider med GPL queries: {temaside_with_queries}/{len(theme_pages)}")


if __name__ == "__main__":
    main()
