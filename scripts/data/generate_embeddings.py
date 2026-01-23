#!/usr/bin/env python3
"""
Generate embeddings for all content using multilingual-e5-base and store in database.

This script uses the pre-trained intfloat/multilingual-e5-base model.
No training required - generates embeddings directly.

Usage:
    python scripts/data/generate_embeddings.py
    python scripts/data/generate_embeddings.py --batch-size 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


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
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model name",
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
    print(f"\nLoading model: {args.model_name}")
    print("(This may take a few minutes on first run...)")
    model = HealthContentEmbedding(model_name=args.model_name)

    # Load content
    print("\nLoading content from database...")
    content_items = database_service.get_all_content()
    print(f"Found {len(content_items)} content items")

    if not content_items:
        print("No content found. Run: python scripts/data/import_content.py")
        sys.exit(1)

    # Generate embeddings in batches using structured passages
    print(f"\nGenerating embeddings (batch size: {args.batch_size})...")
    print("Formatting content as structured passages with metadata...")
    
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
