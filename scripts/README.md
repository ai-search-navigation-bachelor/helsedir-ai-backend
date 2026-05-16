# Scripts

All offline tooling lives here. Unlike the `app/` directory (the live API), these scripts are run manually for one-off tasks: importing data, training models, running evaluations, and setting up the database.

## Quick orientation

If you are new to this codebase, the two most important pipelines are:

1. **Semantic search model training** → [`ml/`](ml/) — fine-tunes the E5 embedding model on Norwegian health content using Generative Pseudo Labeling (GPL)
2. **Learning-to-rank model training** → [`train/`](train/) — trains an XGBoost LambdaMART reranker from real click data

Everything else (data import, migrations, evaluation) exists to support these two pipelines and the running system.

---

## Directory Overview

```text
scripts/
├── ml/               ← ML pipeline: E5 fine-tuning and embedding generation
│   ├── 1_generate_queries.py    # Step 1: generate synthetic queries with LLM
│   ├── 2_finetune_gpl.py        # Step 2: fine-tune E5 model
│   ├── 3_generate_embeddings.py # Step 3: embed all documents
│   ├── 4_evaluate_model.py      # Step 4: measure model quality
│   └── utils.py                 # Shared passage formatting, content enrichment
│
├── train/            ← LTR model training
│   └── train_ranking_model.py   # Train XGBoost LambdaMART from click logs
│
├── data/             ← Data management
│   ├── importing/    # Import content from the Helsedirektoratet API
│   ├── migration/    # Database schema migrations and field backfills
│   ├── generation/   # Generate theme pages and role tags
│   └── maintenance/  # Database cleanup and query enrichment
│
├── setup/            ← Initial setup
│   ├── init_database.sql         # Full database schema
│   ├── generate_training_data.py # Synthetic training data for LTR
│   ├── generate_filtered_csv.py  # Filter and prepare CSV training data
│   └── pretrain_all_models.py    # Pre-train LTR models for all presets
│
├── test/             ← Evaluation and testing
│   ├── search/       # Search quality evaluation (NDCG, MRR, Recall)
│   ├── ml/           # Embedding model tests and comparisons
│   ├── api/          # API endpoint smoke tests
│   └── data/         # Data import and DB connection tests
│
└── run.py            ← Convenience server launcher (reads .env automatically)
```

---

## ML Pipeline (`ml/`)

The semantic search model is trained in four steps. See [`ml/README.md`](ml/README.md) for full details.

```bash
# Run the full pipeline
python scripts/ml/1_generate_queries.py   # ~1-2 hours
python scripts/ml/2_finetune_gpl.py       # ~30-60 minutes (GPU)
python scripts/ml/3_generate_embeddings.py # ~15-30 minutes
python scripts/ml/4_evaluate_model.py      # minutes
```

**Why fine-tune?** The base multilingual-e5 model was not trained on Norwegian health text. Fine-tuning with domain-specific synthetic queries measurably improves NDCG and Recall on health content retrieval.

---

## LTR Training (`train/`)

The learning-to-rank model is trained from real user click data collected by the API.

```bash
python scripts/train/train_ranking_model.py
```

The model uses 6 features: semantic score, BM25 score, smoothed CTR (30-day window), role match, query length, and title–query Jaccard overlap. Training is also available through the developer API at `/dev/train`.

---

## Data Import (`data/importing/`)

Content is imported from the Helsedirektoratet API and cached in MySQL.

```bash
# Full import
python scripts/data/importing/import_content.py --target 3000

# Backfill specific fields on existing content
python scripts/data/importing/backfill_anbefaling_details.py
python scripts/data/importing/backfill_document_metadata.py
```

---

## Evaluation (`test/search/`)

Objective search quality measurement using held-out test queries (from the GPL training split):

```bash
# Compare BM25 vs semantic vs hybrid — reports NDCG@10, MRR@10, Recall@10
python scripts/test/search/evaluate_search_methods.py

# Evaluate the LTR reranker
python scripts/test/search/evaluate_ranking.py

# Compare hybrid search weight variants
python scripts/test/search/test_ranking_comparison.py
```

The test triplets (`data/gpl_test_triplets.json`) are automatically split off during step 2 of the ML pipeline and held out from training.

---

## Setup (`setup/`)

Run once when initializing the project:

```bash
# Create database tables
mysql -u root -p helsedir_ai < scripts/setup/init_database.sql

# Generate synthetic training data for LTR (if no real click data yet)
python scripts/setup/generate_training_data.py
```
