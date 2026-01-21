#!/usr/bin/env python3
"""
Training script for the health content embedding model.

This script:
1. Loads content from data/content.json
2. Trains the embedding model on the content
3. Saves the trained model to models/embedding/

Usage:
    python scripts/train_embedding_model.py
    python scripts/train_embedding_model.py --epochs 20 --output-dim 128
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_content(content_file: str) -> list:
    """Load content from JSON file."""
    with open(content_file, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_corpus(content_items: list) -> list:
    """Prepare text corpus from content items."""
    corpus = []
    for item in content_items:
        # Combine title, body, and tags for better embeddings
        text_parts = [item.get("title", ""), item.get("body", "")]
        if item.get("tags"):
            text_parts.append(" ".join(item["tags"]))
        corpus.append(" ".join(text_parts))
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="Train the health content embedding model"
    )
    parser.add_argument(
        "--content-file",
        type=str,
        default="data/content.json",
        help="Path to content JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/embedding",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Maximum vocabulary size",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Token embedding dimension",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
        help="Output embedding dimension",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    args = parser.parse_args()

    # Check for TensorFlow
    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("Error: TensorFlow is required. Install with: pip install tensorflow>=2.15.0")
        sys.exit(1)

    from app.ml.embedding_model import HealthContentEmbedding

    # Load content
    content_file = Path(args.content_file)
    if not content_file.exists():
        print(f"Error: Content file not found: {content_file}")
        sys.exit(1)

    print(f"Loading content from {content_file}...")
    content_items = load_content(content_file)
    print(f"Loaded {len(content_items)} content items")

    # Prepare corpus
    corpus = prepare_corpus(content_items)
    print(f"Prepared corpus with {len(corpus)} documents")

    # Create model
    print("\nCreating embedding model...")
    model = HealthContentEmbedding(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        output_dim=args.output_dim,
        max_sequence_length=args.max_seq_length,
    )

    # Adapt vectorizer to corpus
    print("Adapting vectorizer to corpus...")
    model.adapt(corpus)

    # Get vocabulary info
    vocab = model.vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")

    # Test encoding
    print("\nTesting encoding...")
    sample_texts = corpus[:3] if len(corpus) >= 3 else corpus
    embeddings = model.encode(sample_texts)
    print(f"Sample embedding shape: {embeddings.shape}")

    # Test similarity
    if len(corpus) >= 2:
        query_emb = model.encode([corpus[0]])
        doc_embs = model.encode(corpus[:5])
        sims = model.compute_similarity(query_emb, doc_embs)
        print(f"Self-similarity (should be ~1.0): {sims[0]:.4f}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model"

    print(f"\nSaving model to {model_path}.keras...")
    model.save(str(model_path))

    # Verify saved model
    print("Verifying saved model...")
    loaded_model = HealthContentEmbedding.load(str(model_path))
    test_emb = loaded_model.encode(sample_texts)
    print(f"Loaded model embedding shape: {test_emb.shape}")

    print("\n✓ Training complete!")
    print(f"Model saved to: {model_path}.keras")
    print("\nTo enable semantic search, set in .env:")
    print("  ML_EMBEDDING_ENABLED=true")


if __name__ == "__main__":
    main()
