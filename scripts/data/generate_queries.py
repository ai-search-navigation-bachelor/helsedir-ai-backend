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

Usage:
    python scripts/data/generate_queries.py                     # Generate 10 queries per doc
    python scripts/data/generate_queries.py --queries-per-doc 15 # Generate 15 queries per doc
    python scripts/data/generate_queries.py --force              # Regenerate all queries

Requirements:
    GROQ_API_KEY in .env (free from https://console.groq.com/keys)
    HELSEDIR_API_KEY in .env (for fetching linked content)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_queries_with_groq(
    api_key: str,
    passage: str,
    info_type: str,
    num_queries: int = 3,
    _retries: int = 0,
    show_prompt: bool = False,
) -> List[str]:
    """
    Use Groq API (free tier) to generate synthetic search queries.

    Args:
        api_key: Groq API key
        passage: Formatted passage (same as used for embeddings)
        info_type: Document type (e.g., 'retningslinje', 'veileder')
        num_queries: Number of queries to generate
        _retries: Internal retry counter
        show_prompt: Show the prompt sent to Groq

    Returns:
        List of generated query strings
    """
    import httpx

    # Truncate passage if too long (keep it reasonable for LLM)
    passage_truncated = passage[:2000] if passage else ""

    prompt = f"""Du er en ekspert på norsk helseinformasjon. Generer {num_queries} realistiske søkeord/fraser som en helsepersonell ville brukt for å finne dette dokumentet.

Dokumenttype: {info_type}
Dokumentinnhold:
{passage_truncated}

Regler:
- Skriv søk slik en lege eller sykepleier ville skrevet dem
- Bruk norske medisinske termer
- Varier mellom korte (1-2 ord) og lengre (3-5 ord) søk
- Ikke bruk anførselstegn eller nummerering

Returner kun søkefrasene, én per linje:"""

    if show_prompt:
        print("\n" + "=" * 80)
        print("PROMPT SENT TO GROQ:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
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
            time.sleep(wait)
            return generate_queries_with_groq(api_key, passage, info_type, num_queries, _retries + 1, show_prompt)

        if response.status_code != 200:
            print(f"  Groq error: {response.status_code} - {response.text[:200]}")
            return []

        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        queries = [q.strip() for q in text.split("\n") if q.strip()]

        # Clean up and filter
        cleaned = []
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

        for q in queries:
            # Remove numbering, bullets, and leading symbols
            q = q.lstrip("0123456789.-) *•")

            # Remove "Søk etter" prefix (case insensitive)
            if q.lower().startswith("søk etter "):
                q = q[10:]  # Remove "Søk etter "

            # Strip quotes and whitespace
            q = q.strip('"\'').strip()

            # Skip if empty after cleaning
            if not q:
                continue

            # Skip if too long (likely explanation text, not a search query)
            if len(q) > 100:
                continue

            # Skip if contains colon (likely a label/header like "Her er søkeordene:")
            if ":" in q:
                continue

            # Skip if starts with quote but doesn't end with quote (incomplete)
            if (q.startswith('"') and not q.endswith('"')) or \
               (q.startswith("'") and not q.endswith("'")):
                continue

            # Skip if contains common intro phrases (case insensitive)
            q_lower = q.lower()
            if any(phrase in q_lower for phrase in skip_phrases):
                continue

            cleaned.append(q)

            # Stop when we have enough
            if len(cleaned) >= num_queries:
                break

        return cleaned

    except Exception as e:
        print(f"  Error with Groq: {e}")
        return []


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
    from app.services.data.database_service import database_service
    from app.ml.embedding_model import HealthContentEmbedding

    # Import enrich_content_with_links from generate_embeddings.py
    import sys
    from pathlib import Path
    embeddings_script = Path(__file__).parent / "generate_embeddings.py"

    # Load enrich_content_with_links function
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_embeddings", embeddings_script)
    gen_embeddings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_embeddings)
    enrich_content_with_links = gen_embeddings.enrich_content_with_links

    # Check Groq API key
    if not settings.groq_api_key:
        print("Error: GROQ_API_KEY not configured in .env")
        print("Get a free key at: https://console.groq.com/keys")
        sys.exit(1)

    # Check Helsedir API key (needed for linked content)
    if not settings.helsedir_api_key:
        print("Error: HELSEDIR_API_KEY not configured in .env")
        print("Needed for fetching linked content (barn, forelder, root)")
        sys.exit(1)

    # Check database
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Load content
    print("=" * 60)
    print("QUERY GENERATION FOR GPL TRAINING")
    print("=" * 60)
    print("\nLoading content from database...")
    all_content = database_service.get_all_content()
    print(f"Found {len(all_content)} total content items")

    # Filter out temasider
    content_items = [
        item for item in all_content
        if item.get("info_type", "").lower() != "temaside"
    ]
    print(f"Filtered: {len(content_items)} documents (excluding {len(all_content) - len(content_items)} temasider)")

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
        print(f"\n--force specified, regenerating all queries")

    # Find documents needing queries FIRST (before fetching linked content)
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
        print(f"  Use --force to regenerate, or --queries-per-doc to increase target")
        return

    # NOW enrich ONLY the documents that need queries (optimization!)
    print(f"\nFetching linked content for {len(items_needing_queries)} documents that need queries...")
    print(f"(Skipping {len(content_items) - len(items_needing_queries)} documents that already have queries)")
    print(f"This ensures queries match the embedding format (includes barn, forelder, root, publikasjon)")
    model = HealthContentEmbedding()
    items_needing_queries = enrich_content_with_links(
        items_needing_queries,
        api_key=settings.helsedir_api_key,
        max_links_per_item=10,
        format_passage_fn=model.format_passage,
    )

    total_needed = sum(item["_queries_needed"] for item in items_needing_queries)
    print(f"\nQuery generation needed:")
    print(f"  Documents needing queries: {len(items_needing_queries)}/{len(content_items)}")
    print(f"  Total queries to generate: {total_needed}")
    print(f"  Target per document: {args.queries_per_doc}")
    print(f"  Using: Groq API (llama-3.1-8b-instant)")

    # Estimate time
    estimated_seconds = total_needed * 3  # 3s per query (API rate limit)
    estimated_minutes = estimated_seconds / 60
    print(f"  Estimated time: ~{estimated_minutes:.1f} minutes")

    # Confirm
    print(f"\nPress Ctrl+C to cancel, or Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(0)

    # Generate queries
    print(f"\nGenerating queries...")
    print("-" * 60)

    failed = 0
    generated_count = 0
    shown_prompts = 0

    for i, item in enumerate(items_needing_queries):
        content_id = str(item["id"])
        info_type = item.get("info_type") or ""
        queries_needed = item["_queries_needed"]
        existing_count = item["_existing_count"]

        # Format passage using same format as embeddings
        passage = model.format_passage(item)

        # Show prompt for first 2 documents
        show_prompt = shown_prompts < 2

        # Generate missing queries
        new_queries = generate_queries_with_groq(
            api_key=settings.groq_api_key,
            passage=passage,
            info_type=info_type,
            num_queries=queries_needed,
            show_prompt=show_prompt,
        )

        if new_queries:
            # Append to existing or create new
            existing_queries = cached.get(content_id, []) if not args.force else []
            cached[content_id] = existing_queries + new_queries
            generated_count += len(new_queries)

            # Show generated queries for first 2 prompts
            if show_prompt:
                shown_prompts += 1
                action = "Added" if existing_count > 0 else "Generated"
                title = item.get("tittel") or item.get("title") or ""
                print(f"\n{'='*80}")
                print(f"GROQ RESPONSE (Example {shown_prompts}):")
                print(f"{'='*80}")
                print(f"Document: {title}")
                print(f"Type: {info_type}")
                print(f"Linked content: {len(item.get('linked_content', []))} items")
                print(f"Passage length: {len(passage)} chars")
                print(f"{action} {len(new_queries)} queries (had {existing_count}, now {len(cached[content_id])})")
                print(f"\nGenerated queries:")
                for j, q in enumerate(new_queries, 1):
                    print(f"  {j}. {q}")
                print("=" * 80)
        else:
            failed += 1
            if failed <= 3:
                print(f"  ⚠ Warning: No queries generated for '{title[:50]}'")

        # Progress every 10 items
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(items_needing_queries)}] Generated: {generated_count}, Failed: {failed}")

            # Save intermediate results
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)

        # Groq free tier: 30 req/min, use 3s delay to stay safe
        time.sleep(3)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Total documents: {len(cached)}")
    print(f"  New queries generated: {generated_count}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_path}")

    # Calculate total queries
    total_queries = sum(len(queries) for queries in cached.values())
    avg_queries = total_queries / len(cached) if cached else 0
    print(f"\nStatistics:")
    print(f"  Total queries: {total_queries}")
    print(f"  Average per document: {avg_queries:.1f}")

    print(f"\nNext steps:")
    print(f"  1. Train model: python scripts/train/finetune_gpl.py")
    print(f"  2. Generate embeddings: python scripts/data/generate_embeddings.py")


if __name__ == "__main__":
    main()
