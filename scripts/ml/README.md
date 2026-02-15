# ML Workflow Scripts

Machine learning pipeline for training and generating embeddings. These scripts have long execution times.

## Workflow (in order)

1. **`1_generate_queries.py`** - Generate synthetic search queries using Groq LLM
   - Generates 10 queries per document
   - Uses GPL (Generative Pseudo Labeling) method
   - Caches results in `data/gpl_queries.json`
   - Time: ~3 hours for 3000 documents

2. **`2_finetune_gpl.py`** - Fine-tune E5 embedding model
   - Trains on generated queries using contrastive learning
   - Uses hard negative mining
   - Saves model to `models/finetuned-e5-gpl/`
   - Time: ~30-60 minutes

3. **`3_generate_embeddings.py`** - Generate embeddings for all content
   - Uses fine-tuned model (or base E5 if unavailable)
   - Fetches and includes linked content
   - Stores embeddings in database
   - Time: ~15-30 minutes

## Quick Start

```bash
# Full ML pipeline
python scripts/ml/1_generate_queries.py
python scripts/ml/2_finetune_gpl.py
python scripts/ml/3_generate_embeddings.py
```

## Requirements

- `GROQ_API_KEY` in `.env` (for step 1)
- `HELSEDIR_API_KEY` in `.env` (for step 3)
- `sentence-transformers` package
