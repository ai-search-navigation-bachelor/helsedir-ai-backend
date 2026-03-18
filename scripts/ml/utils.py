"""
Shared utilities for ML scripts (training, embedding generation).

Provides enrichment functions used by both 2_finetune_gpl.py and
3_generate_embeddings.py to ensure identical passages.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# Norwegian stopwords excluded from title sub-query splitting
_NORWEGIAN_STOPWORDS = {
    "og", "eller", "for", "til", "med", "ved", "av", "på", "i", "om",
    "en", "et", "ei", "de", "den", "det", "er", "ut", "inn", "fra",
    "hos", "etter", "under", "over", "mot", "uten", "som", "om",
}


def get_title_subqueries(
    title: str,
    min_word_length: int = 3,
    max_subqueries: int = 3,
) -> List[str]:
    """
    Generate sub-queries from document title for training data augmentation.

    Splits multi-word titles into individual significant words so the embedding
    model learns that searching for a component word (e.g. "kreft") should match
    documents whose title contains it as a compound (e.g. "brystkreft").

    Single-word compound titles return an empty list — compound splitting is
    handled via LLM prompt instructions instead.

    Args:
        title: Document title
        min_word_length: Minimum character length for a word to be included
        max_subqueries: Maximum number of sub-queries to return

    Returns:
        Lowercase, deduplicated list of sub-queries
    """
    if not title:
        return []

    words = title.strip().split()
    if len(words) <= 1:
        # Single compound word — LLM handles splitting via prompt instruction
        return []

    subqueries: List[str] = []
    seen: set = set()
    for word in words:
        w = word.strip(".,!?-()[]").lower()
        if len(w) >= min_word_length and w not in _NORWEGIAN_STOPWORDS and w not in seen:
            subqueries.append(w)
            seen.add(w)
        if len(subqueries) >= max_subqueries:
            break

    return subqueries

project_root = Path(__file__).resolve().parents[2]


def enrich_with_child_content(
    content_items: List[Dict[str, Any]],
) -> None:
    """
    Enrich content items with their children's title and text.

    For each content item that has barn-links with resolved IDs:
    - Adds child's tittel + tekst as linked_content
    - If the child is a 'kapittel', also adds grandchildren (one extra depth level)

    Modifies content_items in-place by adding/extending 'linked_content'.
    """
    # Build lookup: id -> content item
    id_to_content = {str(item["id"]): item for item in content_items}

    enriched_count = 0

    for item in content_items:
        # Skip temasider — they get separate enrichment via theme_page_content
        info_type = (item.get("info_type") or "").lower()
        if info_type == "temaside":
            continue

        links_raw = item.get("links")
        if isinstance(links_raw, str):
            try:
                links = json.loads(links_raw)
            except json.JSONDecodeError:
                links = []
        else:
            links = links_raw or []

        linked_content = list(item.get("linked_content") or [])
        added = 0

        for link in links:
            if link.get("rel") != "barn":
                continue

            child_id = link.get("id")
            if not child_id:
                continue

            child = id_to_content.get(str(child_id))
            if not child:
                continue

            linked_content.append({
                "tittel": child.get("tittel") or "",
                "tekst": child.get("tekst") or "",
                "type": child.get("info_type") or "",
            })
            added += 1

            # If child is a kapittel, go one level deeper
            child_type = (child.get("info_type") or "").lower()
            if child_type == "kapittel":
                child_links_raw = child.get("links")
                if isinstance(child_links_raw, str):
                    try:
                        child_links = json.loads(child_links_raw)
                    except json.JSONDecodeError:
                        child_links = []
                else:
                    child_links = child_links_raw or []

                for child_link in child_links:
                    if child_link.get("rel") != "barn":
                        continue

                    grandchild_id = child_link.get("id")
                    if not grandchild_id:
                        continue

                    grandchild = id_to_content.get(str(grandchild_id))
                    if not grandchild:
                        continue

                    linked_content.append({
                        "tittel": grandchild.get("tittel") or "",
                        "tekst": grandchild.get("tekst") or "",
                        "type": grandchild.get("info_type") or "",
                    })
                    added += 1

        if added > 0:
            item["linked_content"] = linked_content
            enriched_count += 1

    print(f"  Enriched {enriched_count} items with child content")


def enrich_temasider_with_children(
    content_items: List[Dict[str, Any]],
) -> None:
    """
    Enrich temaside content items with linked content from the database.

    Uses theme_page_content junction table (populated by link_theme_pages.py)
    to find which real content belongs under each temaside. Also includes
    child temasider's linked content (grandchildren) for richer passages.

    Modifies content_items in-place by adding 'linked_content' to temasider.
    Falls back to theme_pages.json hierarchy titles if DB is unavailable.
    """
    from app.services.repositories.base import db_pool

    conn = db_pool.get_connection()
    if not conn:
        print("  Warning: No database connection, falling back to hierarchy")
        _enrich_from_hierarchy(content_items)
        return

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)

        # Load all theme_page_content links
        cursor.execute("""
            SELECT tpc.theme_page_id, c.id, c.tittel, c.tekst, c.info_type
            FROM theme_page_content tpc
            JOIN content c ON c.id = tpc.content_id
            ORDER BY tpc.theme_page_id, tpc.display_order
        """)
        rows = cursor.fetchall()

        # Group by theme page
        theme_to_content: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            theme_id = row["theme_page_id"]
            theme_to_content.setdefault(theme_id, []).append({
                "tittel": row["tittel"] or "",
                "tekst": row["tekst"] or "",
                "type": row["info_type"] or "",
            })

        print(f"  Loaded {len(rows)} theme-content links for {len(theme_to_content)} temasider")

    except Exception as e:
        print(f"  Warning: Could not load theme_page_content: {e}")
        theme_to_content = {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    if not theme_to_content:
        _enrich_from_hierarchy(content_items)
        return

    # Build temaside hierarchy for parent pages that have child temasider
    theme_pages_path = project_root / "data" / "theme_pages.json"
    child_temaside_ids: Dict[str, List[str]] = {}
    if theme_pages_path.exists():
        with open(theme_pages_path, "r", encoding="utf-8") as f:
            theme_pages = json.load(f)

        path_to_id = {tp["path"].rstrip("/"): tp["id"] for tp in theme_pages}
        for tp in theme_pages:
            parent_id = tp["id"]
            for link in tp.get("links", []):
                if link.get("rel") == "barn":
                    child_path = link["href"].rstrip("/")
                    child_id = path_to_id.get(child_path)
                    if child_id:
                        child_temaside_ids.setdefault(parent_id, []).append(child_id)

    enriched_count = 0
    for item in content_items:
        info_type = (item.get("info_type") or item.get("content_type") or "").lower()
        if info_type != "temaside":
            continue

        item_id = str(item.get("id", ""))
        linked = []

        # Direct content linked to this temaside
        if item_id in theme_to_content:
            linked.extend(theme_to_content[item_id])

        # Also include content from child temasider (grandchildren)
        for child_id in child_temaside_ids.get(item_id, []):
            if child_id in theme_to_content:
                linked.extend(theme_to_content[child_id])

        if linked:
            item["linked_content"] = linked
            enriched_count += 1

    print(f"  Enriched {enriched_count} temasider with linked content")


def _enrich_from_hierarchy(content_items: List[Dict[str, Any]]) -> None:
    """Fallback: enrich temasider using theme_pages.json titles only."""
    theme_pages_path = project_root / "data" / "theme_pages.json"
    if not theme_pages_path.exists():
        print("  Warning: theme_pages.json not found, no enrichment possible")
        return

    with open(theme_pages_path, "r", encoding="utf-8") as f:
        theme_pages = json.load(f)

    path_to_page = {tp["path"].rstrip("/"): tp for tp in theme_pages}
    path_to_children: Dict[str, List[str]] = {}
    for tp in theme_pages:
        for link in tp.get("links", []):
            if link.get("rel") == "barn":
                path_to_children.setdefault(
                    tp["path"].rstrip("/"), []
                ).append(link["href"].rstrip("/"))

    enriched_count = 0

    for item in content_items:
        info_type = (item.get("info_type") or item.get("content_type") or "").lower()
        if info_type != "temaside":
            continue

        item_path = (item.get("path") or "").rstrip("/")
        if not item_path:
            continue

        linked = []
        for child_path in path_to_children.get(item_path, []):
            child_page = path_to_page.get(child_path)
            if child_page:
                linked.append({"tittel": child_page["tittel"], "tekst": "", "type": "temaside"})
                for gc_path in path_to_children.get(child_path, []):
                    gc_page = path_to_page.get(gc_path)
                    if gc_page:
                        linked.append({"tittel": gc_page["tittel"], "tekst": "", "type": "temaside"})

        if linked:
            item["linked_content"] = linked
            enriched_count += 1

    print(f"  Fallback: enriched {enriched_count} temasider with child temaside titles")
