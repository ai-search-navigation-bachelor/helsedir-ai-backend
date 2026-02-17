#!/usr/bin/env python3
"""
Generate embeddings for all content and store in database.

By default, uses the fine-tuned model (models/finetuned-e5-gpl) which is optimized
for Norwegian health content. Falls back to base model if fine-tuned not available.

Fetches linked content (barn, forelder, root, publikasjon) from Helsedirektoratet API
to enrich all documents with related content.

Usage:
    python scripts/ml/3_generate_embeddings.py                                    # Use fine-tuned model
    python scripts/ml/3_generate_embeddings.py --batch-size 16                    # Adjust batch size
    python scripts/ml/3_generate_embeddings.py --no-fetch-links                   # Skip API calls (faster)
    python scripts/ml/3_generate_embeddings.py --model-name intfloat/multilingual-e5-base  # Use base model
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def fetch_linked_content(
    href: str,
    api_key: str,
    timeout: float = 10.0,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Fetch content from Helsedirektoratet API by href URL.

    Args:
        href: Full API URL (e.g., https://api.helsedirektoratet.no/innhold/kapitler/...)
        api_key: Helsedirektoratet API key
        timeout: Request timeout in seconds
        debug: Print debug info about fetched content

    Returns:
        Dict with 'tittel' and 'tekst' fields, or None if fetch failed
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
                    tekst = data.get('tekst', '')
                    print(f"    [DEBUG] tekst length: {len(tekst) if tekst else 0}")
                    if tekst:
                        print(f"    [DEBUG] tekst preview: {tekst[:100]}...")

                # Get type from tekniskeData.infoType
                info_type = ""
                tekniske_data = data.get("tekniskeData", {})
                if tekniske_data:
                    info_type = tekniske_data.get("infoType", "")

                result = {
                    "tittel": data.get("tittel", ""),
                    "tekst": data.get("tekst", ""),
                    "type": info_type,
                }
                return result
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
    Enrich content items by fetching linked content from API.

    Fetches all link types (barn, forelder, root, publikasjon) for all items.

    Args:
        content_items: List of content dicts from database
        api_key: Helsedirektoratet API key
        max_links_per_item: Maximum number of links to fetch per item
        format_passage_fn: Optional function to format passages for preview

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
        # Parse links from JSON if needed
        links_raw = item.get("links")
        if isinstance(links_raw, str):
            try:
                links = json.loads(links_raw)
            except json.JSONDecodeError:
                links = []
        else:
            links = links_raw or []

        # Fetch linked content for all items with links
        linked_content = []
        if links:
            items_with_links += 1
            links_to_fetch = links[:max_links_per_item]

            # Fetch linked content (debug for first 3 items)
            debug_this_item = passages_shown < 3
            for link in links_to_fetch:
                href = link.get("href")
                if href:
                    fetched = fetch_linked_content(href, api_key, debug=debug_this_item)
                    if fetched and (fetched.get("tittel") or fetched.get("tekst")):
                        linked_content.append(fetched)
                        total_links_fetched += 1
                    # Small delay to avoid rate limiting
                    time.sleep(0.05)

            if linked_content:
                items_enriched += 1

        # Add linked content to item
        enriched_item = dict(item)
        enriched_item["linked_content"] = linked_content
        enriched.append(enriched_item)

        # Show detailed passage for first 3 items with linked content
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

            # Show main content
            print("-" * 40)
            print("MAIN CONTENT:")
            print("-" * 40)
            body = item.get('tekst') or ''
            if body:
                from app.ml.embedding_model import HealthContentEmbedding
                clean_body = HealthContentEmbedding.strip_html_tags(body)
                print(clean_body[:500] if len(clean_body) > 500 else clean_body)
                if len(clean_body) > 500:
                    print(f"[...{len(clean_body) - 500} more chars...]")
            else:
                print("(no body text)")
            print()

            # Show each link separately
            print("-" * 40)
            print("LINKED CONTENT:")
            print("-" * 40)
            for j, lc in enumerate(linked_content, 1):
                print(f"\n  LINK {j}: {lc.get('tittel', '(no title)')}")
                lc_tekst = lc.get('tekst', '')
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

        # Progress update every 10 items
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(content_items)}] "
                  f"Items with links: {items_with_links}, "
                  f"Enriched: {items_enriched}, "
                  f"Links fetched: {total_links_fetched}")

    print(f"\n  Done! Enriched {items_enriched}/{items_with_links} items with {total_links_fetched} linked documents")
    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for all content using E5 model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="models/finetuned-e5-gpl",
        help="Model name (HuggingFace) or local path. Default: models/finetuned-e5-gpl (fine-tuned). Use 'intfloat/multilingual-e5-base' for base model",
    )
    parser.add_argument(
        "--no-fetch-links",
        action="store_true",
        help="Skip fetching linked content from API (not recommended)",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=10,
        help="Maximum number of links to fetch per item (default: 10)",
    )
    args = parser.parse_args()

    # Check for sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers available")
    except ImportError:
        print("Error: sentence-transformers required.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    from app.services.data.database_service import database_service
    from app.ml.embedding_model import HealthContentEmbedding

    # Check database
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Load E5 model (will download from HuggingFace first time)
    model_name = args.model_name

    # Check if fine-tuned model exists, fallback to base if not
    if model_name.startswith("models/"):
        model_path = Path(model_name)
        if not model_path.exists():
            print(f"\n⚠ Fine-tuned model not found at: {model_name}")
            print("Falling back to base model: intfloat/multilingual-e5-base")
            print("To train a fine-tuned model, run: python scripts/train/finetune_gpl.py")
            model_name = "intfloat/multilingual-e5-base"

    print(f"\nLoading model: {model_name}")
    print("(This may take a few minutes on first run...)")
    model = HealthContentEmbedding(model_name=model_name)

    # Load content
    print("\nLoading content from database...")
    content_items = database_service.get_all_content()
    print(f"Found {len(content_items)} content items")

    if not content_items:
        print("No content found. Run: python scripts/data/import_content.py")
        sys.exit(1)

    # Sort to put 'anbefaling' and 'retningslinje' first (they have more content)
    priority_types = ['anbefaling', 'retningslinje', 'pakkeforlop-anbefaling', 'veileder']
    def sort_key(item):
        info_type = item.get('info_type', '')
        if info_type in priority_types:
            return (0, priority_types.index(info_type))
        return (1, info_type)
    content_items = sorted(content_items, key=sort_key)
    print(f"Sorted content (priority: {', '.join(priority_types)})")

    # Fetch linked content from API (default behavior)
    if not args.no_fetch_links:
        from app.config import settings
        if not settings.helsedir_api_key:
            print("Error: HELSEDIR_API_KEY not configured in .env")
            print("Cannot fetch linked content without API key")
            sys.exit(1)

        print(f"\nFetching linked content from API...")
        print(f"  Max links per item: {args.max_links}")
        content_items = enrich_content_with_links(
            content_items,
            api_key=settings.helsedir_api_key,
            max_links_per_item=args.max_links,
            format_passage_fn=model.format_passage,
        )
    else:
        print("\nSkipping link fetching (--no-fetch-links specified)")

    # Generate embeddings in batches using structured passages
    print(f"\nGenerating embeddings (batch size: {args.batch_size})...")
    print("Formatting content as structured passages with metadata...")
    if not args.no_fetch_links:
        print("Including linked content in passages...")

    all_embeddings = []
    ids = []

    for i in range(0, len(content_items), args.batch_size):
        batch = content_items[i:i + args.batch_size]

        # Extract IDs
        batch_ids = [item.get("id") for item in batch]
        ids.extend(batch_ids)

        # Generate embeddings for batch (uses format_passage internally)
        batch_embeddings = model.encode_passages(batch, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)

        print(f"  Processed {min(i + args.batch_size, len(content_items))}/{len(content_items)}...")

    embeddings = np.vstack(all_embeddings)
    print(f"\n✓ Generated embeddings: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]} (E5-base: 768-dim)")

    # Store embeddings in database
    print("\nStoring embeddings in database...")
    stored = 0
    for i, (content_id, embedding) in enumerate(zip(ids, embeddings)):
        # Convert to bytes for BLOB storage (float32)
        embedding_bytes = embedding.astype(np.float32).tobytes()

        if store_embedding(database_service, content_id, embedding_bytes):
            stored += 1

        if (i + 1) % 100 == 0:
            print(f"  Stored {i + 1}/{len(ids)}...")

    print(f"\n✓ Successfully stored {stored}/{len(ids)} embeddings")

    # Verify
    print("\nVerifying stored embeddings...")
    sample_id = ids[0]
    loaded = load_embedding(database_service, sample_id)
    if loaded is not None:
        loaded_emb = np.frombuffer(loaded, dtype=np.float32)
        original_emb = embeddings[0].astype(np.float32)
        match = np.allclose(loaded_emb, original_emb)
        print(f"  Sample embedding shape: {loaded_emb.shape}")
        print(f"  Verification: {'✓ PASSED' if match else '✗ FAILED'}")
    else:
        print("  ✗ Verification: Could not load embedding")

    print("\n" + "="*60)
    print("✓ Done! Embeddings are ready for semantic search.")
    print("="*60)
    print("\nTo enable semantic search, ensure in .env:")
    print("  ML_EMBEDDING_ENABLED=true")


def store_embedding(db_service, content_id: int, embedding_bytes: bytes) -> bool:
    """Store embedding in database."""
    conn = db_service._get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE content SET embedding = %s WHERE id = %s",
            (embedding_bytes, content_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing embedding for {content_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def load_embedding(db_service, content_id: str) -> bytes:
    """Load embedding from database."""
    conn = db_service._get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT embedding FROM content WHERE id = %s",
            (content_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error loading embedding: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
