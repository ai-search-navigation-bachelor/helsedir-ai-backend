"""
ML module — embedding model and learning-to-rank reranker.

embedding_model.py
    HealthContentEmbedding: wraps intfloat/multilingual-e5-base (fine-tuned
    with GPL on Norwegian health content). Encodes documents and queries into
    768-dimensional vectors for cosine similarity search.
    Enabled with ML_EMBEDDING_ENABLED=true.

ranking_model.py
    HealthContentReranker: XGBoost LambdaMART reranker trained on click logs.
    Uses 6 features: semantic score, BM25 score, smoothed CTR, role match,
    query length, title-query overlap.
    Enabled with ML_RANKING_ENABLED=true.

Models are loaded once at startup (lifespan hook in app/main.py) and reused
for every request. Both are optional — the API runs without them.

Training:
    scripts/ml/     — fine-tune the E5 embedding model (GPL pipeline)
    scripts/train/  — train the XGBoost reranker from click data
"""
