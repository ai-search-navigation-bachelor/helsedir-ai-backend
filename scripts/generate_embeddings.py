#!/usr/bin/env python3
"""
Generate embeddings for all content and store in database.

Run this after training the embedding model.

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --model-path models/embedding/model
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for all content"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/embedding/model",
        help="Path to trained embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    args = parser.parse_args()

    # Check for TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("Error: TensorFlow required. Install with: pip install tensorflow")
        sys.exit(1)

    from app.services.database_service import database_service
    from app.ml.embedding_model import HealthContentEmbedding

    # Check database
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Load model
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".keras").exists():
        print(f"Error: Model not found at {model_path}.keras")
        print("Run: python scripts/train_embedding_model.py")
        sys.exit(1)

    print(f"Loading model from {model_path}.keras...")
    model = HealthContentEmbedding.load(str(model_path))

    # Load content
    print("Loading content from database...")
    content_items = database_service.get_all_content()
    print(f"Found {len(content_items)} content items")

    # Prepare texts
    texts = []
    ids = []
    for item in content_items:
        text = f"{item.get('tittel', '')} {item.get('tekst', '')}"
        texts.append(text)
        ids.append(item.get("id"))

    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch size: {args.batch_size})...")
    all_embeddings = []

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i + args.batch_size]
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.append(batch_embeddings)

        if (i // args.batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + args.batch_size, len(texts))}/{len(texts)}...")

    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Store embeddings in database
    print("\nStoring embeddings in database...")
    stored = 0
    for i, (content_id, embedding) in enumerate(zip(ids, embeddings)):
        # Convert to bytes for BLOB storage
        embedding_bytes = embedding.tobytes()

        if store_embedding(database_service, content_id, embedding_bytes):
            stored += 1

        if (i + 1) % 100 == 0:
            print(f"  Stored {i + 1}/{len(ids)}...")

    print(f"\nSuccessfully stored {stored}/{len(ids)} embeddings")

    # Verify
    print("\nVerifying stored embeddings...")
    sample_id = ids[0]
    loaded = load_embedding(database_service, sample_id)
    if loaded is not None:
        loaded_emb = np.frombuffer(loaded, dtype=np.float32)
        original_emb = embeddings[0]
        match = np.allclose(loaded_emb, original_emb)
        print(f"  Verification: {'PASSED' if match else 'FAILED'}")
    else:
        print("  Verification: Could not load embedding")

    print("\nDone! Embeddings are ready for semantic search.")


def store_embedding(db_service, content_id: str, embedding_bytes: bytes) -> bool:
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
