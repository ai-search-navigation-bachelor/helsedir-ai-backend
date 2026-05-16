# ML Tests

Tests for machine learning models and embeddings.

## Test Files

- **`test_e5_embedding.py`** - Test E5 embedding generation and similarity ranking
- **`test_e5_passages.py`** - Comprehensive passage formatting visualization with sample data
- **`test_temaside_passages.py`** - Visualize temaside passages with and without DB enrichment
- **`test_ml_integration.py`** - Integration test for the full ML pipeline (embedding + ranking)
- **`test_model.py`** - Compare base vs fine-tuned model on sample queries

## Usage

```bash
# Test E5 embeddings
python scripts/test/ml/test_e5_embedding.py

# Compare base vs fine-tuned model on sample queries
python scripts/test/ml/test_model.py

# Visualize passage formatting and similarity ranking
python scripts/test/ml/test_e5_passages.py

# Inspect temaside passages with/without DB enrichment
python scripts/test/ml/test_temaside_passages.py

# Full ML pipeline integration test
python scripts/test/ml/test_ml_integration.py
```

## Model Comparison

The `test_model.py` script compares:
- Base E5 model (`intfloat/multilingual-e5-base`)
- Fine-tuned model (`models/finetuned-e5-gpl`)

It shows similarity scores and margins for test queries to verify that fine-tuning improved retrieval quality.
