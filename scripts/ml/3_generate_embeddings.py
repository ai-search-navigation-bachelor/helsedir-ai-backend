#!/usr/bin/env python3
"""
Generate embeddings for all content and store in database.

By default, uses the fine-tuned model (models/finetuned-e5-gpl) which is optimized
for Norwegian health content. Falls back to base model if fine-tuned not available.

Enriches documents using the same logic as 2_finetune_gpl.py:
- Temasider: enriched with child content via theme_page_content junction table
- Other content: enriched with barn's title+text (extra depth for kapittel children)

Usage:
    python scripts/ml/3_generate_embeddings.py
    python scripts/ml/3_generate_embeddings.py --batch-size 16
    python scripts/ml/3_generate_embeddings.py --no-enrich
    python scripts/ml/3_generate_embeddings.py --model-name intfloat/multilingual-e5-base
    python scripts/ml/3_generate_embeddings.py --commit-batch-size 500
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml.utils import enrich_temasider_with_children, enrich_with_child_content  # noqa: E402


def store_embeddings_batch(
    db_service,
    ids_and_embeddings: List[tuple],
) -> int:
    """Store a batch of embeddings in database with a single commit."""
    conn = db_service._get_connection()
    if not conn:
        return 0

    cursor = None
    try:
        cursor = conn.cursor()
        # executemany is faster than individual execute calls
        cursor.executemany(
            "UPDATE content SET embedding = %s WHERE id = %s",
            [(emb, cid) for cid, emb in ids_and_embeddings],
        )
        conn.commit()
        return len(ids_and_embeddings)
    except Exception as e:
        print(f"  Error committing batch: {e}")
        return 0
    finally:
        if cursor:
            cursor.close()
        conn.close()


def load_embedding(db_service, content_id: str) -> bytes:
    """Load embedding from database for verification."""
    conn = db_service._get_connection()
    if not conn:
        return None

    cursor = None
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
        if cursor:
            cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for all content using E5 model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--model-name", type=str, default="models/finetuned-e5-gpl",
        help="Model name or local path. Default: models/finetuned-e5-gpl",
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Skip link enrichment (faster, but passages won't include related content)",
    )
    parser.add_argument(
        "--commit-batch-size", type=int, default=500,
        help="Number of embeddings per DB commit (default: 500)",
    )
    args = parser.parse_args()

    # Check for sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("sentence-transformers available")
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

    # Load E5 model
    model_name = args.model_name

    if model_name.startswith("models/"):
        model_path = Path(model_name)
        if not model_path.exists():
            print(f"\nFine-tuned model not found at: {model_name}")
            print("Falling back to base model: intfloat/multilingual-e5-base")
            model_name = "intfloat/multilingual-e5-base"

    print(f"\nLoading model: {model_name}")
    model = HealthContentEmbedding(model_name=model_name)

    # Load content from database
    print("\nLoading content from database...")
    content_items = database_service.get_all_content()
    print(f"Found {len(content_items)} content items")

    if not content_items:
        print("No content found. Run: python scripts/data/importing/import_content.py")
        sys.exit(1)

    # Count types
    type_counts: Dict[str, int] = {}
    for item in content_items:
        t = item.get("info_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Content types: {type_counts}")

    # Enrich content — same logic as 2_finetune_gpl.py
    if not args.no_enrich:
        print("\nEnriching temasider with linked content...")
        enrich_temasider_with_children(content_items)

        print("\nEnriching content with child content...")
        enrich_with_child_content(content_items)
    else:
        print("\nSkipping enrichment (--no-enrich specified)")

    # Show sample passages (encode_passages calls format_passage internally)
    print(f"\nSample passages:")
    shown = 0
    for item in content_items:
        if shown >= 3:
            break
        linked_count = len(item.get("linked_content") or [])
        if linked_count > 0 or shown == 0:
            passage = HealthContentEmbedding.format_passage(item)
            print(f"\n  [{item.get('info_type')}] {item.get('tittel', '')[:60]}")
            print(f"  Linked: {linked_count}, Passage length: {len(passage)} chars")
            print(f"  Preview: {passage[:200]}...")
            shown += 1

    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch size: {args.batch_size})...")

    all_embeddings = []
    ids = []

    for i in range(0, len(content_items), args.batch_size):
        batch = content_items[i:i + args.batch_size]

        batch_ids = [str(item.get("id")) for item in batch]
        ids.extend(batch_ids)

        batch_embeddings = model.encode_passages(batch, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)

        print(f"  Processed {min(i + args.batch_size, len(content_items))}/{len(content_items)}...")

    embeddings = np.vstack(all_embeddings)
    print(f"\nGenerated embeddings: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    # Store embeddings in database with batch commits
    commit_batch_size = args.commit_batch_size
    print(f"\nStoring embeddings in database (batch commit size: {commit_batch_size})...")

    stored_total = 0
    batch_buffer = []

    for i, (content_id, embedding) in enumerate(zip(ids, embeddings)):
        embedding_bytes = embedding.astype(np.float32).tobytes()
        batch_buffer.append((content_id, embedding_bytes))

        if len(batch_buffer) >= commit_batch_size:
            stored = store_embeddings_batch(database_service, batch_buffer)
            stored_total += stored
            print(f"  Committed {stored_total}/{len(ids)} embeddings...")
            batch_buffer = []

    # Commit remaining
    if batch_buffer:
        stored = store_embeddings_batch(database_service, batch_buffer)
        stored_total += stored

    print(f"\nSuccessfully stored {stored_total}/{len(ids)} embeddings")

    # Verify
    print("\nVerifying stored embeddings...")
    sample_id = ids[0]
    loaded = load_embedding(database_service, sample_id)
    if loaded is not None:
        loaded_emb = np.frombuffer(loaded, dtype=np.float32)
        original_emb = embeddings[0].astype(np.float32)
        match = np.allclose(loaded_emb, original_emb)
        print(f"  Sample embedding shape: {loaded_emb.shape}")
        print(f"  Verification: {'PASSED' if match else 'FAILED'}")
    else:
        print("  Verification: Could not load embedding")

    print("\n" + "="*60)
    print("Done! Embeddings are ready for semantic search.")
    print("="*60)
    print("\nTo enable semantic search, ensure in .env:")
    print("  ML_EMBEDDING_ENABLED=true")


if __name__ == "__main__":
    main()
