#!/usr/bin/env python3
"""
Train the learning-to-rank (LTR) reranker model from real user click data.

The model is an XGBoost LambdaMART ranker. It learns to reorder search results
based on 6 features extracted from the search_results_shown and click_logs tables:

  - semantic_score      — cosine similarity from the E5 embedding model
  - bm25_score          — BM25 keyword relevance score
  - smoothed_ctr        — Bayesian-smoothed click-through rate (last 30 days)
  - role_match          — how well the content's target group matches the user's role
  - query_length        — number of terms in the query
  - title_query_overlap — Jaccard overlap between query terms and document title

Training uses IPS (Inverse Propensity Scoring) to correct for position bias:
results shown higher on the page receive more clicks regardless of relevance.
IPS divides each click signal by the probability of being clicked at that position,
so the model learns from relevance rather than position.

Requires sufficient data in the database to produce a useful model.
Enable the trained model with ML_RANKING_ENABLED=true in .env.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.ml.ranking_model import HealthContentReranker


def main():
    print("=" * 50)
    print("Training Learning-to-Rank Reranker")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # Step 1: Load training data from the database
    # train_from_database() reads the last 6 months of search sessions from
    # search_results_shown (one row per result shown) joined with click_logs.
    # Only sessions with at least 5 results AND at least one click are used —
    # sessions with no clicks give no learning signal for ranking.
    # -------------------------------------------------------------------------
    reranker = HealthContentReranker()

    print("\nTraining from database logs...")
    print("This may take a few minutes depending on the amount of data.\n")

    results = reranker.train_from_database(
        days_back=180,           # Use last 6 months of data
        min_group_size=5,        # Only use searches with at least 5 results shown
        require_any_click=True,  # Skip sessions with no clicks (no label signal)
        use_db_propensity=True,  # Use position propensities from DB for IPS weighting
        verbose=True,            # Show XGBoost training progress
    )

    # -------------------------------------------------------------------------
    # Step 2: Report results and save model
    # Training is grouped by search_id (each search session = one query group).
    # LambdaMART requires groups — it learns to rank within a session, not globally.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Results")
    print("=" * 50)
    print(f"Trained: {'Yes' if results['trained'] else 'No'}")
    print(f"Training groups (search sessions): {int(results['groups'])}")
    print(f"Training rows (results shown): {int(results['rows'])}")

    if results["trained"]:
        model_path = project_root / "models" / "ranking" / "reranker.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        reranker.save(str(model_path))
        print(f"\nModel saved to: {model_path}")

        # Reload and verify to catch any serialization issues before enabling
        print("Verifying saved model...")
        loaded = HealthContentReranker.load(str(model_path))
        assert loaded.model is not None, "Loaded model is None"
        assert len(loaded.feature_names) == len(reranker.feature_names), "Feature count mismatch"
        print(f"  Model verified: {len(loaded.feature_names)} features")

        print("\nTo enable reranking, set in .env:")
        print("  ML_RANKING_ENABLED=true")
    else:
        print("\nTraining failed — not enough data.")
        print("  The model needs search sessions where users clicked results.")
        print("  Make sure these tables have data:")
        print("  - search_results_shown (one row per result shown per search)")
        print("  - click_logs (click events with search_id and content_id)")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
