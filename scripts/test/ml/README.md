# ML Tests

Tests for machine learning models and embeddings.

## Test Files

- **`test_e5_embedding.py`** - Test E5 embedding generation
- **`test_e5_passages.py`** - Test passage formatting
- **`test_ml_integration.py`** - Test ML pipeline integration
- **`test_ntnu_llm.py`** - Test NTNU LLM integration
- **`test_model.py`** - Compare base vs fine-tuned model

## Usage

```bash
# Test E5 embeddings
python scripts/test/ml/test_e5_embedding.py

# Compare models
python scripts/test/ml/test_model.py

# Test passage formatting
python scripts/test/ml/test_e5_passages.py
```

## Model Comparison

The `test_model.py` script compares:
- Base E5 model (`intfloat/multilingual-e5-base`)
- Fine-tuned model (`models/finetuned-e5-gpl`)

It shows similarity scores and margins for test queries to verify that fine-tuning improved retrieval quality.
