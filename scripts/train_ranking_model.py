#!/usr/bin/env python3
"""
Training script for the health content ranking model.

This script:
1. Loads training data from MySQL database (search_logs, click_logs)
2. Prepares training data from search/click events
3. Trains the ranking model
4. Saves the trained model to models/ranking/

Usage:
    python scripts/train_ranking_model.py
    python scripts/train_ranking_model.py --epochs 50 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Train the health content ranking model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ranking",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation data fraction",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum training samples required",
    )
    parser.add_argument(
        "--hidden-units",
        type=str,
        default="64,32",
        help="Hidden layer sizes (comma-separated)",
    )
    args = parser.parse_args()

    # Check for TensorFlow
    try:
        import tensorflow as tf
        import numpy as np

        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("Error: TensorFlow is required. Install with: pip install tensorflow>=2.15.0")
        sys.exit(1)

    from app.services.database_service import database_service
    from app.ml.ranking_model import (
        HealthContentRanker,
        RANKING_FEATURES,
    )

    # Check database connection
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        print("Check your MySQL settings in .env")
        sys.exit(1)

    # Get training data from database
    print("Loading training data from database...")
    training_data = database_service.get_training_data()

    search_count = database_service.get_search_count()
    click_count = database_service.get_click_count()

    print(f"\nDatabase Statistics:")
    print(f"  Total searches: {search_count}")
    print(f"  Total clicks: {click_count}")
    print(f"  Training samples: {len(training_data)}")

    if not training_data:
        print("\nNo training data available.")
        print("Training data requires searches that have at least one click.")
        print("\nTo collect training data:")
        print("1. Ensure frontend sends search_id with search events")
        print("2. Ensure frontend sends search_id with click events")
        print("3. Collect data over time until you have sufficient examples")
        sys.exit(1)

    # Convert training data to features and labels
    positive_count = sum(1 for d in training_data if d["clicked"] == 1)
    negative_count = len(training_data) - positive_count

    print(f"  Positive samples (clicks): {positive_count}")
    print(f"  Negative samples (no click): {negative_count}")

    if len(training_data) < args.min_samples:
        print(f"\nError: Not enough training samples ({len(training_data)} < {args.min_samples})")
        print("Continue collecting click data and try again later.")
        sys.exit(1)

    # Prepare features and labels
    features = []
    labels = []

    for sample in training_data:
        feature_dict = {
            "title_keyword_score": 0.0,
            "body_keyword_score": 0.0,
            "tag_match_score": 0.0,
            "semantic_similarity": 0.0,
            "content_type_encoded": 0.0,
            "days_since_published": 0.0,
            "historical_ctr": database_service.get_ctr(sample["content_id"]),
            "role_match": 1.0 if sample.get("role") else 0.0,
            "position_shown": float(sample.get("position", 0)) / 10.0,
            "baseline_score": float(sample.get("score", 0)) / 20.0,
        }
        features.append([feature_dict.get(name, 0.0) for name in RANKING_FEATURES])
        labels.append(sample["clicked"])

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Parse hidden units
    hidden_units = [int(x.strip()) for x in args.hidden_units.split(",")]

    # Create model
    print(f"\nCreating ranking model...")
    print(f"  Features: {len(RANKING_FEATURES)}")
    print(f"  Hidden units: {hidden_units}")

    model = HealthContentRanker(
        num_features=len(RANKING_FEATURES),
        hidden_units=hidden_units,
    )
    model.compile()

    # Train model
    print(f"\nTraining model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Validation split: {args.validation_split}")

    history = model.train(
        features,
        labels,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Print final metrics
    final_metrics = {
        key: values[-1] for key, values in history.history.items() if values
    }
    print(f"\nFinal metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model"

    print(f"\nSaving model to {model_path}.keras...")
    model.save(str(model_path))

    # Verify saved model
    print("Verifying saved model...")
    loaded_model = HealthContentRanker.load(str(model_path))

    # Test prediction
    test_features = [{name: 0.5 for name in RANKING_FEATURES}]
    test_pred = loaded_model.predict(test_features)
    print(f"Test prediction: {test_pred[0]:.4f}")

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}.keras")
    print("\nTo enable learning-to-rank, set in .env:")
    print("  ML_RANKING_ENABLED=true")


if __name__ == "__main__":
    main()
