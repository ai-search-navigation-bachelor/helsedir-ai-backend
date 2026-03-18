#!/usr/bin/env python3
"""
Generate synthetic search queries for temasider using OpenAI API.

The main query generation script (1_generate_queries.py) excludes temasider.
This script fills that gap by generating queries specifically for temasider,
using the updated passage format that avoids dilution from child content.

Merges results into data/gpl_queries.json alongside existing queries.

Usage:
    python scripts/data/maintenance/generate_temaside_queries.py
    python scripts/data/maintenance/generate_temaside_queries.py --queries-per-doc 15
    python scripts/data/maintenance/generate_temaside_queries.py --force   # Regenerate all
    python scripts/data/maintenance/generate_temaside_queries.py --model gpt-4.1-nano  # Cheaper model

Requirements:
    OPENAI_API_KEY in .env
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml.utils import enrich_temasider_with_children, get_title_subqueries  # noqa: E402

TITLE_QUERY_COUNT = 2  # Number of explicit title name queries per temaside


async def get_compound_subqueries(
    client: httpx.AsyncClient,
    api_key: str,
    title: str,
    model: str = "gpt-4.1-mini",
) -> List[str]:
    """Split a single compound Norwegian word into meaningful parts via OpenAI."""
    try:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": (
                    f"Del opp det norske sammensatte ordet '{title}' i meningsfulle deler. "
                    "Returner kun delene, én per linje. Maks 2 deler. Ingen forklaringer."
                )}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=15.0,
        )
        if response.status_code != 200:
            return []
        text = response.json()["choices"][0]["message"]["content"].strip()
        parts = [p.strip().lower() for p in text.split("\n") if p.strip()]
        return [p for p in parts if len(p) >= 3 and p.lower() != title.lower()][:2]
    except Exception:
        return []


async def generate_queries_openai(
    client: httpx.AsyncClient,
    api_key: str,
    passage: str,
    title: str,
    num_queries: int = 10,
    model: str = "gpt-4.1-mini",
    _retries: int = 0,
) -> List[str]:
    """Generate synthetic search queries for a temaside using OpenAI."""
    prompt = f"""Du er en ekspert på norsk helseinformasjon. Generer {num_queries} realistiske søkeord/fraser som en helsepersonell ville brukt for å finne denne temasiden.

Temasiden heter: {title}

Innhold:
{passage[:2000]}

Regler:
- Skriv søk slik en lege eller sykepleier ville skrevet dem
- Bruk norske medisinske termer
- Varier mellom korte (1-2 ord) og lengre (3-5 ord) søk
- Inkluder forkortelser og synonymer hvis relevant
- Ikke gjenta tittelen direkte – den er allerede inkludert separat
- Ikke bruk anførselstegn eller nummerering

Returner kun søkefrasene, én per linje:"""

    try:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 300,
            },
            timeout=30.0,
        )

        if response.status_code == 429:
            if _retries >= 3:
                print("  Rate limited 3 times, skipping")
                return []
            wait = 5 * (_retries + 1)
            print(f"  Rate limited, waiting {wait}s (retry {_retries + 1}/3)...")
            await asyncio.sleep(wait)
            return await generate_queries_openai(
                client, api_key, passage, title, num_queries, model, _retries + 1
            )

        if response.status_code != 200:
            print(f"  OpenAI error: {response.status_code} - {response.text[:200]}")
            return []

        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        queries = [q.strip() for q in text.split("\n") if q.strip()]

        skip_phrases = [
            "her er", "søkeord", "fraser", "følgende", "liste",
            "kunne brukt", "ville brukt", "for å finne",
        ]

        cleaned = []
        for q in queries:
            q = q.lstrip("0123456789.-) *•")
            if q.lower().startswith("søk etter "):
                q = q[10:]
            q = q.strip("\"'").strip()

            if not q or len(q) > 100 or ":" in q:
                continue

            q_lower = q.lower()
            if any(phrase in q_lower for phrase in skip_phrases):
                continue

            cleaned.append(q)
            if len(cleaned) >= num_queries:
                break

        return cleaned

    except Exception as e:
        print(f"  Error with OpenAI: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate queries for temasider using OpenAI"
    )
    parser.add_argument(
        "--queries-per-doc", type=int, default=10,
        help="Number of queries per temaside (default: 10)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-mini",
        help="OpenAI model (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all temaside queries (ignore existing)",
    )
    args = parser.parse_args()

    from app.config import settings
    from app.ml.embedding_model import HealthContentEmbedding
    from app.services.data.database_service import database_service

    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY not configured in .env")
        sys.exit(1)

    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Load content
    print("=" * 60)
    print("TEMASIDE QUERY GENERATION (OpenAI)")
    print("=" * 60)

    print("\nLoading content from database...")
    all_content = database_service.get_all_content()

    temasider = [
        item for item in all_content
        if (item.get("info_type") or "").lower() == "temaside"
    ]
    print(f"Found {len(temasider)} temasider")

    if not temasider:
        print("No temasider found")
        sys.exit(1)

    # Enrich temasider with child content
    print("\nEnriching temasider with linked content...")
    enrich_temasider_with_children(temasider)

    # Format passages with the new temaside-optimized format
    for item in temasider:
        item["_passage"] = HealthContentEmbedding.format_passage(item)

    # Show sample passages
    print("\nSample passages:")
    for item in temasider[:3]:
        passage = item["_passage"]
        print(f"\n  {item.get('tittel', '')[:60]}")
        print(f"  Passage ({len(passage)} chars): {passage[:200]}...")

    # Load existing query cache
    cache_path = project_root / "data" / "gpl_queries.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"\nLoaded existing cache: {len(cached)} documents")
    else:
        cached = {}

    # Find temasider needing queries
    items_needing_queries = []
    for item in temasider:
        content_id = str(item["id"])
        existing = cached.get(content_id, []) if not args.force else []
        if len(existing) < args.queries_per_doc:
            item["_queries_needed"] = args.queries_per_doc - len(existing)
            items_needing_queries.append(item)

    if not items_needing_queries:
        print(f"\nAll {len(temasider)} temasider already have {args.queries_per_doc}+ queries")
        print("Use --force to regenerate")
        return

    total_needed = sum(item["_queries_needed"] for item in items_needing_queries)
    print(f"\nGeneration plan:")
    print(f"  Temasider needing queries: {len(items_needing_queries)}/{len(temasider)}")
    print(f"  Total queries to generate: {total_needed}")
    print(f"  Strategy: {TITLE_QUERY_COUNT} title + word splits (multi-word) + LLM = {args.queries_per_doc} per temaside")
    print(f"  Model: {args.model}")

    # Generate queries (saves to cache_path every 10 items)
    asyncio.run(_generate_all(
        items=items_needing_queries,
        api_key=settings.openai_api_key,
        cached=cached,
        force=args.force,
        model=args.model,
        cache_path=cache_path,
    ))

    # Final save
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    # Summary
    temaside_ids = {str(item["id"]) for item in temasider}
    temaside_query_count = sum(
        len(cached.get(tid, [])) for tid in temaside_ids
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Temaside queries: {temaside_query_count} for {len(temasider)} temasider")
    print(f"  Total queries in cache: {sum(len(q) for q in cached.values())}")
    print(f"  Saved to: {cache_path}")
    print(f"\nNext steps:")
    print(f"  1. Train model: python scripts/ml/2_finetune_gpl.py")
    print(f"  2. Generate embeddings: python scripts/ml/3_generate_embeddings.py")



async def _generate_all(
    items: List[Dict[str, Any]],
    api_key: str,
    cached: Dict[str, Any],
    force: bool,
    model: str,
    cache_path: Path = None,
) -> None:
    """Generate queries for all temasider. Saves intermediate results every 10 items.

    Each temaside gets TITLE_QUERY_COUNT explicit title queries + remaining from LLM.
    E.g. with queries_per_doc=10: 2 title + 8 LLM = 10 total.
    """
    generated_total = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(items):
            content_id = str(item["id"])
            title = item.get("tittel") or item.get("title") or ""
            passage = item.get("_passage", "")
            queries_needed = item["_queries_needed"]

            # Build explicit queries: TITLE_QUERY_COUNT exact title + word sub-queries
            title_queries = [title] * min(TITLE_QUERY_COUNT, queries_needed)
            if len(title.split()) == 1:
                word_subqueries = await get_compound_subqueries(client, api_key, title, model)
            else:
                word_subqueries = get_title_subqueries(title)

            explicit_queries = title_queries + word_subqueries
            llm_queries_needed = max(0, queries_needed - len(explicit_queries))

            if llm_queries_needed > 0:
                llm_queries = await generate_queries_openai(
                    client=client,
                    api_key=api_key,
                    passage=passage,
                    title=title,
                    num_queries=llm_queries_needed,
                    model=model,
                )
            else:
                llm_queries = []

            new_queries = explicit_queries + llm_queries

            if new_queries:
                existing = cached.get(content_id, []) if not force else []
                cached[content_id] = existing + new_queries
                generated_total += len(new_queries)

                if generated_total <= 20:
                    print(f"\n  {title[:60]}")
                    print(f"    [title x{TITLE_QUERY_COUNT}] {title}")
                    if word_subqueries:
                        print(f"    [word splits] {', '.join(word_subqueries)}")
                    for q in llm_queries[:3]:
                        print(f"    -> {q}")
            else:
                failed += 1

            if (i + 1) % 10 == 0:
                # Save intermediate results
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cached, f, ensure_ascii=False, indent=2)
                print(f"  [{i + 1}/{len(items)}] Generated: {generated_total}, Failed: {failed}")

            # Small delay to respect rate limits
            await asyncio.sleep(0.5)

    print(f"\n  Generated: {generated_total} queries, Failed: {failed}")


if __name__ == "__main__":
    main()
