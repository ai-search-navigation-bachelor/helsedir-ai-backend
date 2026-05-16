# Search Evaluation Scripts

Objective evaluation of search quality using held-out test queries.

## Scripts

### `evaluate_search_methods.py` — Primary evaluation script

Compares BM25, semantic search, and hybrid search head-to-head. Reports NDCG@10, MRR@10, and Recall@10.

```bash
# Compare all three methods (default)
python scripts/test/search/evaluate_search_methods.py

# Adjust cutoff
python scripts/test/search/evaluate_search_methods.py --k 5

# Custom test triplets file
python scripts/test/search/evaluate_search_methods.py \
    --test-triplets data/gpl_test_triplets.json
```

**Requires**: `data/gpl_test_triplets.json` — generated automatically by `scripts/ml/2_finetune_gpl.py` (held out from training). This ensures evaluation is on unseen queries.

### `evaluate_ranking.py` — LTR reranker evaluation

Measures whether the XGBoost reranker improves search quality over the base hybrid results.

```bash
python scripts/test/search/evaluate_ranking.py
```

### `test_ranking_comparison.py` — Hybrid weight comparison

Evaluates multiple RRF weight variants (e.g. different BM25/semantic ratios) and reports which combination performs best.

```bash
python scripts/test/search/test_ranking_comparison.py
```

### `test_search.py` — Live search smoke tests

Tests the search API against a running server with a set of representative queries.

### `eval_popularity_recall.py` — CTR signal evaluation

Evaluates how well the CTR-based popularity signal (smoothed click-through rate) contributes to recall.

## Metrics

| Metric | Description |
|---|---|
| NDCG@k | Normalized Discounted Cumulative Gain — rewards ranking the correct document higher |
| MRR@k | Mean Reciprocal Rank — `1/rank` of the first correct result |
| Recall@k | Fraction of queries where the correct document appears in the top-k results |

## Test Data

The test triplets (`data/gpl_test_triplets.json`) have the format:

```json
[
  {
    "query": "diabetes behandling",
    "positive_id": "12345",
    "negative_ids": ["67890", "11111"]
  }
]
```

Each triplet was generated during GPL training and held out from the fine-tuning data split.
