"""
Shared utilities for ML scripts (query generation, embedding generation).
"""

import json
import time
from typing import Any, Dict, List, Optional

import httpx


def fetch_linked_content(
    href: str,
    api_key: str,
    timeout: float = 10.0,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fetch content from Helsedirektoratet API by href URL.

    Args:
        href: Full API URL (e.g., https://api.helsedirektoratet.no/innhold/kapitler/...)
        api_key: Helsedirektoratet API key
        timeout: Request timeout in seconds
        debug: Print debug info about fetched content

    Returns:
        Dict with 'tittel', 'tekst', and 'type' fields, or None if fetch failed
    """
    try:
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Accept": "application/json",
        }
        if debug:
            print(f"    [DEBUG] Fetching: {href}")

        with httpx.Client() as client:
            response = client.get(href, headers=headers, timeout=timeout)
            if debug:
                print(f"    [DEBUG] Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if debug:
                    print(f"    [DEBUG] Response keys: {list(data.keys())}")
                    print(f"    [DEBUG] tittel: {data.get('tittel', '')[:50]}")
                    tekst = data.get("tekst", "")
                    print(f"    [DEBUG] tekst length: {len(tekst) if tekst else 0}")
                    if tekst:
                        print(f"    [DEBUG] tekst preview: {tekst[:100]}...")

                info_type = ""
                tekniske_data = data.get("tekniskeData", {})
                if tekniske_data:
                    info_type = tekniske_data.get("infoType", "")

                return {
                    "tittel": data.get("tittel", ""),
                    "tekst": data.get("tekst", ""),
                    "type": info_type,
                }
            else:
                if debug:
                    print(f"    [DEBUG] Failed: status {response.status_code}")
                    print(f"    [DEBUG] Response: {response.text[:200]}")
    except Exception as e:
        if debug:
            print(f"    [DEBUG] Error: {e}")
    return None


def enrich_content_with_links(
    content_items: List[Dict[str, Any]],
    api_key: str,
    max_links_per_item: int = 10,
    format_passage_fn=None,
) -> List[Dict[str, Any]]:
    """
    Enrich content items by fetching linked content from the Helsedir API.

    Fetches all link types (barn, forelder, root, publikasjon) for each item.

    Args:
        content_items: List of content dicts from the database
        api_key: Helsedirektoratet API key
        max_links_per_item: Maximum number of links to fetch per item
        format_passage_fn: Optional function to format passages for debug preview

    Returns:
        Enriched content items with 'linked_content' field added
    """
    enriched = []
    total_links_fetched = 0
    items_enriched = 0
    items_with_links = 0
    passages_shown = 0

    print(f"  Processing {len(content_items)} items...")

    for i, item in enumerate(content_items):
        links_raw = item.get("links")
        if isinstance(links_raw, str):
            try:
                links = json.loads(links_raw)
            except json.JSONDecodeError:
                links = []
        else:
            links = links_raw or []

        linked_content = []
        if links:
            items_with_links += 1
            links_to_fetch = links[:max_links_per_item]
            debug_this_item = passages_shown < 3

            for link in links_to_fetch:
                href = link.get("href")
                if href:
                    fetched = fetch_linked_content(href, api_key, debug=debug_this_item)
                    if fetched and (fetched.get("tittel") or fetched.get("tekst")):
                        linked_content.append(fetched)
                        total_links_fetched += 1
                    time.sleep(0.05)

            if linked_content:
                items_enriched += 1

        enriched_item = dict(item)
        enriched_item["linked_content"] = linked_content
        enriched.append(enriched_item)

        if passages_shown < 3 and linked_content and format_passage_fn:
            passages_shown += 1
            print(f"\n{'='*80}")
            print(f"PASSAGE {passages_shown}/3: {item.get('tittel', '')}")
            print(f"{'='*80}")
            print(f"ID: {item.get('id')}")
            print(f"Type: {item.get('info_type')}")
            print(f"Body length: {len(item.get('tekst') or '')} chars")
            print(f"Links fetched: {len(linked_content)}")
            print()

            print("-" * 40)
            print("MAIN CONTENT:")
            print("-" * 40)
            body = item.get("tekst") or ""
            if body:
                from app.ml.embedding_model import HealthContentEmbedding
                clean_body = HealthContentEmbedding.strip_html_tags(body)
                print(clean_body[:500] if len(clean_body) > 500 else clean_body)
                if len(clean_body) > 500:
                    print(f"[...{len(clean_body) - 500} more chars...]")
            else:
                print("(no body text)")
            print()

            print("-" * 40)
            print("LINKED CONTENT:")
            print("-" * 40)
            for j, lc in enumerate(linked_content, 1):
                print(f"\n  LINK {j}: {lc.get('tittel', '(no title)')}")
                lc_tekst = lc.get("tekst", "")
                if lc_tekst:
                    from app.ml.embedding_model import HealthContentEmbedding
                    clean_lc = HealthContentEmbedding.strip_html_tags(lc_tekst)
                    print(f"  Text ({len(clean_lc)} chars): {clean_lc[:300]}")
                    if len(clean_lc) > 300:
                        print(f"  [...{len(clean_lc) - 300} more chars...]")
                else:
                    print("  Text: (no text)")

            print()
            print("-" * 40)
            print("FULL FORMATTED PASSAGE:")
            print("-" * 40)
            passage = format_passage_fn(enriched_item)
            print(passage)
            print(f"\n[Total passage length: {len(passage)} chars]")
            print()

        if (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{len(content_items)}] "
                f"Items with links: {items_with_links}, "
                f"Enriched: {items_enriched}, "
                f"Links fetched: {total_links_fetched}"
            )

    print(
        f"\n  Done! Enriched {items_enriched}/{items_with_links} items "
        f"with {total_links_fetched} linked documents"
    )
    return enriched
