#!/usr/bin/env python3
"""
Training script for the health content embedding model.

This script:
1. Loads content from MySQL database cache
2. Extracts tags for multi-label supervised learning
3. Creates contrastive pairs (weighted by tag overlap)
4. Trains model with triplet loss
5. Saves the trained model to models/embedding/

Usage:
    python scripts/train_embedding_model.py
    python scripts/train_embedding_model.py --epochs 10 --batch-size 32
"""

import os
import sys

# Fix Windows encoding issue with Norwegian characters
os.environ["PYTHONUTF8"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def prepare_corpus(content_items: list) -> list:
    """Prepare text corpus from content items."""
    corpus = []
    for item in content_items:
        # Combine title and body for better embeddings
        text_parts = [
            item.get("tittel", item.get("title", "")),
            item.get("tekst", item.get("body", "")),
        ]
        corpus.append(" ".join(filter(None, text_parts)))
    return corpus


def extract_labels_from_tags(content_items: list, num_tags: int = 5) -> tuple:
    """
    Extract labels from tags for supervised contrastive learning.
    
    Uses the top N tags as labels for each document. This allows for
    multi-label learning where documents can be similar on multiple dimensions.
    
    Example:
        Doc A: ["diabetes", "behandling", "type_2"] → Can group with docs about
               diabetes, OR about behandling, OR about type 2
        Doc B: ["diabetes", "barn", "pediatri"] → Shares "diabetes" with A
               but focuses on children
    
    Args:
        content_items: List of content items with tags
        num_tags: Number of top tags to use as labels (default: 5)
    
    Returns:
        Tuple of (labels_list, tag_to_id mapping, label_stats)
        - labels_list: List of lists, each containing up to num_tags label IDs
        - tag_to_id: Mapping from tag string to numeric ID
        - label_stats: Dict with statistics about label distribution
    """
    labels = []
    tag_to_id = {}
    next_id = 0
    all_tags_flat = []
    
    for item in content_items:
        tags_json = item.get("tags")
        item_labels = []
        
        # Parse tags
        if tags_json:
            try:
                if isinstance(tags_json, str):
                    tags = json.loads(tags_json)
                else:
                    tags = tags_json
                    
                if tags and len(tags) > 0:
                    # Use top N tags (they're already sorted by importance)
                    top_tags = tags[:num_tags]
                else:
                    top_tags = ["unknown"]
            except (json.JSONDecodeError, TypeError):
                top_tags = ["unknown"]
        else:
            top_tags = ["unknown"]
        
        # Map each tag to numeric ID
        for tag in top_tags:
            if tag not in tag_to_id:
                tag_to_id[tag] = next_id
                next_id += 1
            
            item_labels.append(tag_to_id[tag])
            all_tags_flat.append(tag)
        
        labels.append(item_labels)
    
    # Calculate statistics
    tag_counts = Counter(all_tags_flat)
    
    # Count how many docs have 1, 2, 3 tags
    docs_by_tag_count = Counter(len(item_labels) for item_labels in labels)
    
    label_stats = {
        "unique_tags": len(tag_to_id),
        "total_tag_assignments": len(all_tags_flat),
        "avg_tags_per_doc": len(all_tags_flat) / len(labels) if labels else 0,
        "docs_by_tag_count": dict(docs_by_tag_count),
        "top_tags": tag_counts.most_common(20),
    }
    
    return labels, tag_to_id, label_stats


def main():
    parser = argparse.ArgumentParser(
        description="Train the health content embedding model"
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin for triplet loss",
    )
    parser.add_argument(
        "--pairs-per-doc",
        type=int,
        default=3,
        help="Number of contrastive pairs per document",
    )
    args = parser.parse_args()

    # Check for TensorFlow
    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("Error: TensorFlow is required. Install with: pip install tensorflow>=2.15.0")
        sys.exit(1)

    from app.services.database_service import database_service
    from app.ml.embedding_model import HealthContentEmbedding

    # Check database connection
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        print("Check your MySQL settings in .env")
        sys.exit(1)

    # Load content from database
    print("Loading content from database...")
    content_items = database_service.get_all_content()
    content_count = database_service.get_content_count()

    print(f"Found {content_count} content items in database")

    if not content_items:
        print("\nNo content in database cache.")
        print("Run: python scripts/import_content.py")
        sys.exit(1)

    # Check for tags
    items_with_tags = sum(1 for item in content_items if item.get("tags"))
    if items_with_tags == 0:
        print("\nNo tags found in content!")
        print("Run: python scripts/generate_tags.py")
        sys.exit(1)
    
    print(f"Items with tags: {items_with_tags}/{content_count}")

    # Prepare corpus
    corpus = prepare_corpus(content_items)
    print(f"Prepared corpus with {len(corpus)} documents")

    # Extract labels from tags for supervised learning
    print("\nExtracting labels from tags (using top 5 tags per document)...")
    labels, tag_to_id, label_stats = extract_labels_from_tags(content_items, num_tags=5)
    
    # Show label distribution
    print(f"\nLabel statistics:")
    print(f"  Unique tags: {label_stats['unique_tags']}")
    print(f"  Total tag assignments: {label_stats['total_tag_assignments']}")
    print(f"  Avg tags per document: {label_stats['avg_tags_per_doc']:.1f}")
    print(f"\n  Documents by tag count:")
    for count, num_docs in sorted(label_stats['docs_by_tag_count'].items()):
        print(f"    {count} tag(s): {num_docs} documents")
    
    print(f"\n  Top 15 most common tags:")
    for tag, count in label_stats['top_tags'][:15]:
        print(f"    {tag}: {count} documents")

    # Create model
    print("\nCreating embedding model...")
    model = HealthContentEmbedding(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        output_dim=args.output_dim,
        max_sequence_length=args.max_seq_length,
    )

    # Adapt vectorizer to corpus
    print("\nAdapting vectorizer to Norwegian health vocabulary...")
    model.adapt(corpus)

    # Get vocabulary info
    vocab = model.vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vocab: {vocab[1:11]}")  # Skip [UNK]

    # Create contrastive pairs with multi-label support
    from app.ml.embedding_model import create_contrastive_pairs, triplet_loss

    print("\nCreating contrastive training pairs...")
    anchors, positives, negatives, weights = create_contrastive_pairs(
        texts=corpus,
        labels=labels,
        pairs_per_doc=args.pairs_per_doc,
        weight_by_overlap=True,
    )
    print(f"Created {len(anchors)} training triplets")
    print(f"Weight distribution: min={min(weights):.1f}, max={max(weights):.1f}, avg={sum(weights)/len(weights):.2f}")

    # Training loop
    print(f"\nTraining with triplet loss for {args.epochs} epochs...")

    batch_size = args.batch_size
    n_samples = len(anchors)
    n_batches = (n_samples + batch_size - 1) // batch_size

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    import time

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Shuffle training data
        indices = list(range(n_samples))
        random.shuffle(indices)

        epoch_loss = 0.0
        batch_losses = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]

            batch_anchors = [anchors[i] for i in batch_indices]
            batch_positives = [positives[i] for i in batch_indices]
            batch_negatives = [negatives[i] for i in batch_indices]
            batch_weights = tf.constant([weights[i] for i in batch_indices], dtype=tf.float32)

            # Convert to tensors for training (model() requires this for gradient tracking)
            anchor_tensor = tf.constant([[t] for t in batch_anchors], dtype=tf.string)
            positive_tensor = tf.constant([[t] for t in batch_positives], dtype=tf.string)
            negative_tensor = tf.constant([[t] for t in batch_negatives], dtype=tf.string)

            with tf.GradientTape() as tape:
                # Call model directly (not .predict()) to track gradients
                anchor_emb = model.model(anchor_tensor, training=True)
                positive_emb = model.model(positive_tensor, training=True)
                negative_emb = model.model(negative_tensor, training=True)

                # Compute weighted triplet loss
                loss = triplet_loss(
                    anchor_emb, positive_emb, negative_emb,
                    margin=args.margin,
                    weights=batch_weights
                )

            # Update weights
            gradients = tape.gradient(loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

            batch_loss = loss.numpy()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)

            # Progress logging every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{n_batches} - Loss: {batch_loss:.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        min_loss = min(batch_losses)
        max_loss = max(batch_losses)

        print(f"\n  Epoch {epoch + 1}/{args.epochs}:")
        print(f"    Avg Loss: {avg_loss:.4f} (min: {min_loss:.4f}, max: {max_loss:.4f})")
        print(f"    Time: {epoch_time:.1f}s ({epoch_time/n_batches:.2f}s/batch)")

        # Sample similarity check every few epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            sample_emb = model.encode(corpus[:5])
            self_sim = float(np.dot(sample_emb[0], sample_emb[0]))
            other_sim = float(np.dot(sample_emb[0], sample_emb[1]))
            print(f"    Sample similarities: self={self_sim:.3f}, other={other_sim:.3f}")

    # Test similarity after training
    print("\nTesting learned similarities...")
    if len(corpus) >= 10:
        test_emb = model.encode(corpus[:10])
        sims = model.compute_similarity(test_emb[:1], test_emb)
        print(f"Self-similarity: {sims[0]:.4f}")
        print(f"Other similarities: {sims[1:5]}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model"

    print(f"\nSaving trained model to {model_path}.keras...")
    model.save(str(model_path))

    # Save tag mapping for later use
    tag_mapping_path = output_dir / "tag_to_id.json"
    with open(tag_mapping_path, "w", encoding="utf-8") as f:
        json.dump(tag_to_id, f, ensure_ascii=False, indent=2)
    print(f"Tag mapping saved to: {tag_mapping_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nModel saved to: {model_path}.keras")
    print(f"Training triplets: {len(anchors)}")
    print(f"Epochs: {args.epochs}")
    print(f"Final loss: {avg_loss:.4f}")

    print("\nTo enable in search, set in .env:")
    print("  ML_EMBEDDING_ENABLED=true")


if __name__ == "__main__":
    main()
