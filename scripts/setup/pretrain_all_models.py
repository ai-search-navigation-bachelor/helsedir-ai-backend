"""
Pre-train reranker models for all presets.

Generates training data and trains an XGBoost model for each preset,
saving them to models/ranking/<preset_id>.json.

Usage:
    python scripts/setup/pretrain_all_models.py
    python scripts/setup/pretrain_all_models.py --preset 1  # Train only one
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.repositories.training_repository import DATASETS_DIR, training_repository
from app.ml.ranking_model import HealthContentReranker
from scripts.setup.generate_training_data import (
    parse_page_view_csv,
    match_csv_to_content,
    generate_training_data,
    get_connection,
)


def train_preset(preset_id: int) -> dict:
    """Generate data and train model for a single preset."""
    preset = training_repository.get_preset(preset_id)
    if preset is None:
        print(f"  Preset {preset_id} not found, skipping")
        return {}

    print(f"\n{'='*60}")
    print(f"  Preset #{preset_id}: {preset['name']}")
    print(f"  Dataset: {preset['dataset_filename']}")
    print(f"{'='*60}")

    csv_path = DATASETS_DIR / preset["dataset_filename"]
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return {}

    csv_rows = parse_page_view_csv(str(csv_path))
    matched = match_csv_to_content(csv_rows)
    if not matched:
        print("  No pages matched content in DB")
        return {}

    matched.sort(key=lambda x: -x["views"])
    pages = matched[: preset["top_n"]]
    print(f"  Using {len(pages)} pages")

    conn = get_connection()
    try:
        # Clear existing logs (dev-only safety gate)
        import os
        if os.environ.get("PRETRAIN_ALLOW_TRUNCATE", "").lower() != "true":
            print("  ERROR: Set PRETRAIN_ALLOW_TRUNCATE=true to allow log clearing")
            return {}
        cursor = conn.cursor()
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        for tbl in ("click_logs", "search_results_shown", "search_logs"):
            cursor.execute(f"TRUNCATE TABLE {tbl}")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()
        cursor.close()

        # Generate training data
        print("  Generating training data...")
        stats = generate_training_data(
            conn, pages,
            k=preset["k"],
            target_click_min=preset["target_click_min"],
            target_click_max=preset["target_click_max"],
            temaside_click_weight=preset["temaside_click_weight"],
            retningslinje_click_weight=preset["retningslinje_click_weight"],
            other_click_weight=preset["other_click_weight"],
        )
    finally:
        conn.close()

    # Train model
    print("  Training model...")
    reranker = HealthContentReranker()
    metrics = reranker.train_from_database(
        days_back=180,
        min_group_size=5,
        require_any_click=True,
        verbose=False,
    )

    if metrics.get("trained", 0) == 0:
        print("  No usable training groups!")
        return metrics

    # Save model
    model_dir = project_root / "models" / "ranking"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{preset_id}.json"
    reranker.save(str(model_path))
    print(f"  Saved to {model_path}")
    print(f"  Groups: {int(metrics['groups'])}, Rows: {int(metrics['rows'])}")

    # Show feature importances
    raw = reranker.model.get_booster().get_score(importance_type="gain")
    total = sum(raw.values()) or 1
    print("  Feature importances:")
    for k, v in sorted(raw.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v/total*100:.1f}%")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-train reranker models for all presets")
    parser.add_argument("--preset", type=int, help="Train only this preset ID")
    args = parser.parse_args()

    presets = training_repository.get_all_presets()
    if not presets:
        print("No presets found in database")
        return

    if args.preset:
        train_preset(args.preset)
    else:
        for preset in presets:
            train_preset(preset["id"])

    print(f"\nDone! Models saved to models/ranking/")


if __name__ == "__main__":
    main()
