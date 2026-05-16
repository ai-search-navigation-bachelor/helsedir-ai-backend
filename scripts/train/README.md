# LTR Model Training

Train the **Learning-to-Rank (LTR)** reranker from real user click data.

## Overview

The LTR model is an XGBoost LambdaMART model that reranks the top search results returned by the hybrid search pipeline. It is trained on historical (query, document, click) data collected by the API.

## Usage

```bash
python scripts/train/train_ranking_model.py
```

Training is also available through the developer web UI at `/dev/train`, which calls this same logic.

## Features

The model uses 6 features per (query, document) pair:

| Feature | Description |
|---|---|
| `semantic_score` | Normalized cosine similarity from E5 embeddings (0–1) |
| `bm25_score` | Normalized BM25 score (0–1) |
| `smoothed_ctr` | Click-through rate: `(clicks + 1) / (impressions + 21)`, 30-day window |
| `role_match` | Whether the document's target groups match the user's role (0 or 1) |
| `query_length` | Number of tokens in the query |
| `title_query_overlap` | Jaccard overlap between query tokens and document title tokens |

## Position Bias Correction

Click data is position-biased: results shown first are more likely to be clicked regardless of relevance. The model corrects for this using **IPS (Inverse Propensity Scoring)**: each click is weighted by the inverse of the click probability at that position, estimated from the `position_propensity` table.

## Data Requirements

- Real click data in `search_logs`, `search_results_shown`, and `click_logs` tables
- If no real click data exists, use `scripts/setup/generate_training_data.py` to generate synthetic training data

## Enabling the Model

After training, enable reranking in `.env`:

```bash
ML_RANKING_ENABLED=true
```

The active model is stored in `models/ranking/`. Pre-trained models for different training presets can be browsed and activated via the `/dev/models` and `/dev/model/select` API endpoints.
