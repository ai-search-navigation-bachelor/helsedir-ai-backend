#!/usr/bin/env python3
"""
Generate synthetic search queries for documents using LLM (Groq API).

Used for GPL (Generative Pseudo Labeling) fine-tuning of embedding models.
Queries are cached in data/gpl_queries.json for reuse.

Features:
- Automatically filters out 'temaside' content (has its own system)
- Incremental generation: only generates missing queries to reach target
- Caches all queries for fast reuse
- Uses same passage format as embeddings (includes linked content)
- Parallel generation with up to 4 Groq API keys (~4x faster)

Usage:
    python scripts/ml/1_generate_queries.py                     # Generate 10 queries per doc
    python scripts/ml/1_generate_queries.py --queries-per-doc 15 # Generate 15 queries per doc
    python scripts/ml/1_generate_queries.py --force              # Regenerate all queries

Requirements:
    OPENAI_API_KEY in .env (required — model: gpt-4o-mini)
    Child content enrichment is handled automatically via enrich_with_child_content
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml.utils import enrich_with_child_content, get_title_subqueries  # noqa: E402


async def generate_queries_async(
    client: httpx.AsyncClient,
    api_key: str,
    passage: str,
    info_type: str,
    num_queries: int = 3,
    _retries: int = 0,
) -> List[str]:
    """
    Use OpenAI API to generate synthetic search queries.

    Args:
        client: Shared async HTTP client
        api_key: OpenAI API key
        passage: Formatted passage (same as used for embeddings)
        info_type: Document type (e.g., 'retningslinje', 'veileder')
        num_queries: Number of queries to generate
        _retries: Internal retry counter

    Returns:
        List of generated query strings
    """
    passage_truncated = passage[:2000] if passage else ""

    prompt = f"""Du er en ekspert på norsk helseinformasjon. Generer {num_queries} realistiske søkeord/fraser som en helsepersonell ville brukt for å finne dette dokumentet.

Dokumenttype: {info_type}
Dokumentinnhold:
{passage_truncated}

Regler:
- Skriv søk slik en lege eller sykepleier ville skrevet dem
- Bruk norske medisinske termer
- Minst halvparten av søkene skal være korte (1-2 ord)
- Inkluder også noen lengre søk (3-5 ord)
- Del opp sammensatte ord fra tittelen og inkluder delene som egne søk (f.eks. "bryst" og "kreft" fra "brystkreft")
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
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 200,
            },
            timeout=30.0,
        )

        if response.status_code == 429:
            if _retries >= 3:
                print("  Rate limited 3 times, skipping")
                return []
            wait = 15 * (_retries + 1)
            print(f"  Rate limited, waiting {wait}s (retry {_retries + 1}/3)...")
            await asyncio.sleep(wait)
            return await generate_queries_async(client, api_key, passage, info_type, num_queries, _retries + 1)

        if response.status_code != 200:
            print(f"  OpenAI error: {response.status_code} - {response.text[:200]}")
            return []

        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        queries = [q.strip() for q in text.split("\n") if q.strip()]

        skip_phrases = [
            "her er",
            "søkeord",
            "fraser",
            "følgende",
            "liste",
            "kunne brukt",
            "ville brukt",
            "for å finne",
        ]

        cleaned = []
        for q in queries:
            q = q.lstrip("0123456789.-) *•")

            if q.lower().startswith("søk etter "):
                q = q[10:]

            q = q.strip("\"'").strip()

            if not q:
                continue
            if len(q) > 100:
                continue
            if ":" in q:
                continue
            if (q.startswith('"') and not q.endswith('"')) or \
               (q.startswith("'") and not q.endswith("'")):
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


async def api_worker(
    worker_id: int,
    api_key: str,
    queue: asyncio.Queue,
    cached: Dict[str, Any],
    force: bool,
    stats: Dict[str, int],
    save_lock: asyncio.Lock,
    output_path: Path,
) -> None:
    """
    Async worker that consumes items from the queue and calls the OpenAI API.

    Sleeps 0.1s between requests — OpenAI gpt-4o-mini supports ~500 req/min.
    """
    async with httpx.AsyncClient() as client:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            content_id = str(item["id"])
            info_type = item.get("info_type") or ""
            queries_needed = item["_queries_needed"]
            existing_count = item["_existing_count"]
            title = item.get("tittel") or item.get("title") or ""
            passage = item.get("_passage", "")

            # Inject title word sub-queries (multi-word titles only)
            word_subqueries = get_title_subqueries(title)
            llm_queries_needed = max(0, queries_needed - len(word_subqueries))

            if llm_queries_needed > 0:
                new_queries = await generate_queries_async(
                    client=client,
                    api_key=api_key,
                    passage=passage,
                    info_type=info_type,
                    num_queries=llm_queries_needed,
                )
            else:
                new_queries = []

            combined = word_subqueries + new_queries

            if combined:
                existing = cached.get(content_id, []) if not force else []
                cached[content_id] = existing + combined
                stats["generated"] += len(combined)
            else:
                stats["failed"] += 1
                if stats["failed"] <= 3:
                    print(f"  ⚠ Warning: No queries generated for '{title[:50]}'")

            stats["processed"] += 1

            if stats["processed"] % 10 == 0:
                async with save_lock:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(cached, f, ensure_ascii=False, indent=2)
                print(
                    f"  [{stats['processed']}/{stats['total']}] "
                    f"Workers: {stats['num_workers']} | "
                    f"Generated: {stats['generated']} | "
                    f"Failed: {stats['failed']}"
                )

            queue.task_done()
            await asyncio.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic queries for GPL training"
    )
    parser.add_argument(
        "--queries-per-doc",
        type=int,
        default=10,
        help="Number of queries to generate per document (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gpl_queries.json",
        help="Output file for cached queries (default: data/gpl_queries.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all queries (ignore cache)",
    )
    args = parser.parse_args()

    from app.config import settings
    from app.ml.embedding_model import HealthContentEmbedding
    from app.services.data.database_service import database_service

    # Check OpenAI API key
    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY not configured in .env")
        sys.exit(1)
    api_keys = [settings.openai_api_key]

    # Check database
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Load content
    print("=" * 60)
    print("QUERY GENERATION FOR GPL TRAINING")
    print("=" * 60)
    print(f"\nOpenAI API configured (model: gpt-4o-mini)")
    print("\nLoading content from database...")
    all_content = database_service.get_all_content()
    print(f"Found {len(all_content)} total content items")

    from app.services.repositories.content_repository import content_repository
    searchable_types = content_repository.get_searchable_info_types()
    print(f"Searchable content types ({len(searchable_types)}): {sorted(searchable_types)}")

    content_items = [
        item for item in all_content
        if item.get("info_type", "").lower() in searchable_types
        and item.get("info_type", "").lower() != "temaside"
    ]
    n_excluded = len(all_content) - len(content_items)
    n_temasider = sum(1 for item in all_content if item.get("info_type", "").lower() == "temaside")
    n_non_searchable = n_excluded - n_temasider
    print(
        f"Filtered: {len(content_items)} searchable documents "
        f"(excluded {n_temasider} temasider, {n_non_searchable} non-searchable)"
    )
    if n_temasider > 0:
        print(
            f"  Tip: Generate temaside queries separately with:\n"
            f"    python scripts/data/maintenance/generate_temaside_queries.py"
        )

    if not content_items:
        print("No content found to generate queries for")
        sys.exit(1)

    # Setup output path
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    cached = {}
    if output_path.exists() and not args.force:
        print(f"\nLoading cached queries from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  Found {len(cached)} cached documents")
    elif args.force:
        print("\n--force specified, regenerating all queries")

    # Find documents needing queries
    items_needing_queries = []
    for item in content_items:
        content_id = str(item["id"])
        existing_queries = cached.get(content_id, []) if not args.force else []
        if len(existing_queries) < args.queries_per_doc:
            item["_queries_needed"] = args.queries_per_doc - len(existing_queries)
            item["_existing_count"] = len(existing_queries)
            items_needing_queries.append(item)

    if not items_needing_queries:
        print(f"\n✓ All {len(cached)} documents already have {args.queries_per_doc}+ queries")
        print("  Use --force to regenerate, or --queries-per-doc to increase target")
        return

    # Enrich all content with child content (needed for linked_content lookup)
    print("\nEnriching content with child content...")
    enrich_with_child_content(content_items)

    # Pre-format passages so workers don't repeat the work
    for item in items_needing_queries:
        item["_passage"] = HealthContentEmbedding.format_passage(item)

    total_needed = sum(item["_queries_needed"] for item in items_needing_queries)
    num_workers = len(api_keys)

    print(f"\nQuery generation plan:")
    print(f"  Documents needing queries: {len(items_needing_queries)}/{len(content_items)}")
    print(f"  Total queries to generate: {total_needed}")
    print(f"  Target per document: {args.queries_per_doc}")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Rate: ~500 req/min (OpenAI gpt-4o-mini)")

    estimated_seconds = (len(items_needing_queries) / num_workers) * 0.1
    print(f"  Estimated time: ~{estimated_seconds / 60:.1f} minutes")

    print(f"\nPress Ctrl+C to cancel, or Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(0)

    # Run async worker pool
    asyncio.run(_run_workers(
        items=items_needing_queries,
        api_keys=api_keys,
        cached=cached,
        force=args.force,
        output_path=output_path,
    ))

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    # Summary
    total_queries = sum(len(q) for q in cached.values())
    avg_queries = total_queries / len(cached) if cached else 0

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Total documents: {len(cached)}")
    print(f"  Total queries: {total_queries}")
    print(f"  Average per document: {avg_queries:.1f}")
    print(f"  Output: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Train model: python scripts/ml/2_finetune_gpl.py")
    print(f"  2. Generate embeddings: python scripts/ml/3_generate_embeddings.py")


async def _run_workers(
    items: List[Dict[str, Any]],
    api_keys: List[str],
    cached: Dict[str, Any],
    force: bool,
    output_path: Path,
) -> None:
    """Fill the queue and launch one async worker per API key."""
    queue: asyncio.Queue = asyncio.Queue()
    for item in items:
        await queue.put(item)

    save_lock = asyncio.Lock()
    stats: Dict[str, int] = {
        "processed": 0,
        "generated": 0,
        "failed": 0,
        "total": len(items),
        "num_workers": len(api_keys),
    }

    workers = [
        api_worker(
            worker_id=i,
            api_key=key,
            queue=queue,
            cached=cached,
            force=force,
            stats=stats,
            save_lock=save_lock,
            output_path=output_path,
        )
        for i, key in enumerate(api_keys)
    ]

    print(f"\nGenerating queries ({len(api_keys)} parallel workers)...")
    print("-" * 60)
    await asyncio.gather(*workers)


if __name__ == "__main__":
    main()
