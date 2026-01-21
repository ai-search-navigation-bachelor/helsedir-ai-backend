#!/usr/bin/env python3
"""
Training script for the health content ranking model.

This script:
1. Loads click logs from logs/logs.jsonl
2. Prepares training data from search/click events
3. Trains the ranking model
4. Saves the trained model to models/ranking/

Usage:
    python scripts/train_ranking_model.py
    python scripts/train_ranking_model.py --data logs/logs.jsonl --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_logs(logs_file: str) -> list:
    """Load logs from JSONL file."""
    logs = []
    with open(logs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def analyze_logs(logs: list) -> dict:
    """Analyze logs for training statistics."""
    search_events = [log for log in logs if log.get("event_type") == "search"]
    click_events = [log for log in logs if log.get("event_type") == "click"]

    # Count searches with results_shown (needed for training)
    valid_searches = [
        s for s in search_events if s.get("results_shown") and len(s["results_shown"]) > 0
    ]

    return {
        "total_logs": len(logs),
        "search_events": len(search_events),
        "click_events": len(click_events),
        "valid_searches": len(valid_searches),
        "click_rate": len(click_events) / max(len(search_events), 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the health content ranking model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="logs/logs.jsonl",
        help="Path to logs JSONL file",
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

    from app.ml.ranking_model import (
        HealthContentRanker,
        prepare_training_data_from_logs,
        RANKING_FEATURES,
    )

    # Load logs
    logs_file = Path(args.data)
    if not logs_file.exists():
        print(f"Error: Logs file not found: {logs_file}")
        print("\nTo collect training data:")
        print("1. Ensure your frontend logs search events with results_shown")
        print("2. Ensure click events include query and position")
        print("3. Collect data over time until you have sufficient examples")
        sys.exit(1)

    print(f"Loading logs from {logs_file}...")
    logs = load_logs(logs_file)

    # Analyze logs
    stats = analyze_logs(logs)
    print(f"\nLog Statistics:")
    print(f"  Total logs: {stats['total_logs']}")
    print(f"  Search events: {stats['search_events']}")
    print(f"  Click events: {stats['click_events']}")
    print(f"  Valid searches (with results): {stats['valid_searches']}")
    print(f"  Click rate: {stats['click_rate']:.2%}")

    # Check for minimum data
    if stats["valid_searches"] < 10:
        print(f"\nWarning: Not enough valid search events with results_shown.")
        print("Ensure your frontend logs search results like this:")
        print("""
{
  "event_type": "search",
  "query": "diabetes",
  "results_shown": [
    {"content_id": "1", "position": 0, "score": 12.5},
    {"content_id": "2", "position": 1, "score": 8.3}
  ]
}
""")

    # Prepare training data
    print("\nPreparing training data...")
    features, labels = prepare_training_data_from_logs(logs)
    print(f"Training samples: {len(features)}")
    print(f"Positive samples (clicks): {int(labels.sum())}")
    print(f"Negative samples (no click): {int(len(labels) - labels.sum())}")

    if len(features) < args.min_samples:
        print(f"\nError: Not enough training samples ({len(features)} < {args.min_samples})")
        print("Continue collecting click data and try again later.")
        sys.exit(1)

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
    test_features = [
        {name: 0.5 for name in RANKING_FEATURES}
    ]
    test_pred = loaded_model.predict(test_features)
    print(f"Test prediction: {test_pred[0]:.4f}")

    print("\n✓ Training complete!")
    print(f"Model saved to: {model_path}.keras")
    print("\nTo enable learning-to-rank, set in .env:")
    print("  ML_RANKING_ENABLED=true")


if __name__ == "__main__":
    main()
