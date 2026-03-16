#!/usr/bin/env python3
"""
Train the learning-to-rank reranker model.

This script trains a XGBoost LambdaMART model from database logs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.ml.ranking_model import HealthContentReranker


def main():
    print("=" * 50)
    print("Training Learning-to-Rank Reranker")
    print("=" * 50)

    # Create reranker
    reranker = HealthContentReranker()

    # Train from database
    print("\nTraining from database logs...")
    print("This may take a few minutes depending on the amount of data.\n")

    results = reranker.train_from_database(
        days_back=180,           # Use last 6 months of data
        min_group_size=5,        # Only use searches with at least 5 results
        require_any_click=True,  # Only use searches where user clicked something
        use_db_propensity=True,  # Use position propensities from database
        verbose=True,            # Show training progress
    )

    print("\n" + "=" * 50)
    print("Training Results")
    print("=" * 50)
    print(f"Trained: {'Yes' if results['trained'] else 'No'}")
    print(f"Training groups (searches): {int(results['groups'])}")
    print(f"Training rows (results): {int(results['rows'])}")

    if results["trained"]:
        # Save model
        model_path = project_root / "models" / "ranking" / "reranker.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        reranker.save(str(model_path))
        print(f"\nModel saved to: {model_path}")

        # Verify saved model loads correctly
        print("Verifying saved model...")
        loaded = HealthContentReranker.load(str(model_path))
        assert loaded.model is not None, "Loaded model is None"
        assert len(loaded.feature_names) == len(reranker.feature_names), "Feature count mismatch"
        print(f"  Model verified: {len(loaded.feature_names)} features")
    else:
        print("\nTraining failed - not enough data")
        print("  Make sure you have:")
        print("  - Search logs with results shown (search_results_shown table)")
        print("  - Click logs (click_logs table)")
        print("  - At least a few searches with clicks")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
