# helsedir-ai-backend

FastAPI backend for AI-powered health content search for Helsedirektoratet (Norwegian Directorate of Health).

The system implements **hybrid search** — combining BM25 keyword matching with a domain-adapted semantic embedding model, fused via Reciprocal Rank Fusion (RRF). The semantic model is fine-tuned specifically on Norwegian health content using GPL (Generative Pseudo Labeling), requiring no manual data annotation. The training pipeline is described in [How the Semantic Search Model Was Trained](#how-the-semantic-search-model-was-trained).

## Features

- **Hybrid search** - BM25 keyword + semantic embeddings combined with Reciprocal Rank Fusion (RRF)
- **Domain-adapted semantic model** - E5 embedding model fine-tuned on Norwegian health content via GPL ([see training pipeline →](#how-the-semantic-search-model-was-trained))
- **Learning-to-rank** - XGBoost LambdaMART reranker trained on click data (disabled by default — requires sufficient click volume)
- **BM25 hierarchy** - Parent content inherits relevance from child matches for better recall
- **Click tracking** - Search logs, impressions, and clicks with CTR-based popularity signals
- **NKI statistics** - Quality indicator integration from Helsedirektoratet's NKI API
- **Theme pages** - Curated navigation pages with content grouping by category
- **Role personalization** - Score boosting/penalty based on user role matching
- **Norwegian NLP** - Snowball stemming and synonym expansion for Norwegian text

---

## How Search Works

Each search query goes through a multi-stage pipeline:

```
Query
  │
  ▼
┌─────────────────────────────────────────────┐
│  1. Norwegian NLP preprocessing             │
│     - Snowball stemmer (Norwegian Bokmål)   │
│     - Synonym expansion (e.g. hjerteinfarkt │
│       → myokardinfarkt)                     │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌─────────────────┐  ┌──────────────────────┐
│ 2. BM25 Search  │  │ 3. Semantic Search   │
│  Title/body     │  │  Fine-tuned E5 model │
│  keyword score  │  │  Cosine similarity   │
│  + hierarchy    │  │  against stored      │
│  boosting       │  │  embeddings          │
└────────┬────────┘  └──────────┬───────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────┐
│  4. RRF Fusion (Reciprocal Rank Fusion)     │
│     score = w_bm25/(k + rank_bm25)          │
│           + w_sem/(k  + rank_sem)           │
│     Default: w_bm25=0.3, w_sem=0.7, k=60   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  5. Score boosting                          │
│     - Temaside (theme page): ×1.15          │
│     - Retningslinje (guideline): ×1.10      │
│     - Role match: ×1.15 / mismatch: ×0.85  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  6. ML reranking (optional)                 │
│     XGBoost LambdaMART — 6 features:        │
│     semantic score, BM25 score, CTR,        │
│     role match, query length, title overlap │
└──────────────────┬──────────────────────────┘
                   ▼
               Results
```

### BM25 Hierarchy

Documents in Helsedirektoratet often follow a parent–child structure (e.g. a retningslinje page contains many anbefaling sub-pages). To improve recall on broad queries, child relevance propagates to the parent: if several children of a page rank highly for a query, the parent page's BM25 score is boosted. This is controlled by `SEARCH_BM25_HIERARCHY_ENABLED`.

### Reciprocal Rank Fusion (RRF)

RRF is a rank-based fusion method that is robust to score scale differences between BM25 and semantic search. Rather than combining raw scores (which use different scales), it combines rank positions:

```
rrf_score = w_bm25 / (k + rank_bm25) + w_semantic / (k + rank_semantic)
```

The constant `k=60` prevents top-ranked results from dominating too heavily.

### Role Personalization

When a user role is provided (e.g. `lege`, `sykepleier`), results whose target groups match the role receive a score boost, while mismatches receive a penalty. Role tags are stored per content item.

---

## How the Semantic Search Model Was Trained

The semantic search component uses a **domain-adapted embedding model** fine-tuned specifically on Norwegian health content. This is one of the core technical contributions of the project.

### The Challenge

Standard embedding models — even multilingual ones — are trained on general web text. They do not understand Norwegian medical terminology, the structure of Helsedirektoratet's content types, or that a query for "kreft" should surface pages about "kreftsykdommer" and "onkologi". Fine-tuning on domain-specific data addresses this, but normally requires thousands of manually labeled (query → relevant document) pairs.

### Solution: GPL — No Labeled Data Needed

We use **GPL (Generative Pseudo Labeling)**, an unsupervised domain adaptation technique. An LLM generates realistic synthetic queries for each document, and the model is then fine-tuned contrastively on those pairs — no manual annotation required.

**Base model**: `intfloat/multilingual-e5-base` — a multilingual transformer pretrained on 1.1 billion text pairs, producing 768-dimensional embeddings.

### Training Pipeline

The full pipeline lives in [`scripts/ml/`](scripts/ml/) and runs in four steps:

```
Step 1 — Query Generation     (scripts/ml/1_generate_queries.py)
  └─ For every document in the corpus, use OpenAI GPT-4o-mini
     to generate 10 realistic Norwegian search queries.
     Queries are cached in data/gpl_queries.json.
     Time: ~1–2 hours for ~3000 documents.

Step 2 — Model Fine-tuning    (scripts/ml/2_finetune_gpl.py)
  └─ Fine-tune E5 using Multiple Negatives Ranking Loss (MNRL):
     • Each (query, document) pair is a positive example.
     • Hard negatives are mined: for each query, find documents
       most similar to the correct one but actually wrong —
       forcing the model to learn fine-grained distinctions.
     • In-batch negatives give additional signal at no extra cost.
     • Mixed precision (AMP) for 2× training speed.
     • Saves fine-tuned model to models/finetuned-e5-gpl/.
     Time: ~30–60 minutes (GPU recommended).

Step 3 — Embedding Generation (scripts/ml/3_generate_embeddings.py)
  └─ Encode every document in the corpus with the fine-tuned model.
     Embeddings are stored in the database for live retrieval.
     Time: ~15–30 minutes.

Step 4 — Evaluation           (scripts/ml/4_evaluate_model.py)
  └─ Compare base model vs fine-tuned on held-out test triplets
     using NDCG@10, MRR@10, and Recall@10 over the full corpus.
```

See [`scripts/ml/README.md`](scripts/ml/README.md) for a detailed walkthrough of each step.

### Why It Works

GPT-4o-mini generates queries that health professionals would actually type. Fine-tuning then teaches the embedding model two things:
- *"This query belongs to this document"* — positive signal
- *"This query does NOT belong to these similar-looking documents"* — hard negative signal

The result is an embedding space where Norwegian health queries land closer to their correct documents than the generic multilingual baseline achieves.

### Evaluating Search Quality

To compare search methods objectively on held-out test queries:

```bash
# Compare BM25 vs semantic vs hybrid
python scripts/test/search/evaluate_search_methods.py

# Compare base model vs fine-tuned model
python scripts/ml/4_evaluate_model.py
```

Both scripts report NDCG@10, MRR@10, and Recall@10. See [`scripts/test/search/`](scripts/test/search/) for the full evaluation suite.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Run development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest
```

### Docker

```bash
docker compose up --build
```

This starts MySQL 8 and the backend. The database schema is automatically initialized from `scripts/setup/init_database.sql`.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/search` | GET | Hybrid search with pagination |
| `/search/suggestions` | GET | Autocomplete suggestions |
| `/search/categorized` | GET | Results grouped by content type |
| `/search/category` | GET | All results in a specific category |
| `/content/{id}` | GET | Content by ID (logs clicks with `search_id`) |
| `/content/by-path` | GET | Content by helsedirektoratet.no path |
| `/content/{id}/statistics` | GET | NKI quality indicator statistics |
| `/helsedir/search` | GET | Direct search against Helsedir API |
| `/helsedir/infobit/{id}` | GET | Get infobit with nested children |
| `/theme-pages` | GET | List theme pages (optional category filter) |
| `/roles` | GET | Available user roles |
| `/log` | POST | Log user interaction events |
| `/health` | GET | Health check |
| `/dev/*` | Various | Developer tools (training, model management) |

Full interactive docs at `http://localhost:8000/docs`.

---

## Code Architecture

```text
app/
├── routes/              # HTTP endpoints (thin layer)
├── controllers/         # Business logic orchestration
├── services/
│   ├── repositories/    # MySQL data access (Repository pattern)
│   ├── data/            # Content loading, caching, metadata
│   ├── search/          # BM25, semantic, hybrid, RRF, ML reranking
│   ├── statistics/      # NKI quality indicators
│   ├── analytics/       # Event logging
│   └── external/        # Helsedirektoratet API client
├── dto/                 # Request/Response Pydantic models
├── entities/            # Domain models
├── ml/                  # Embedding model (E5) + ranking model (XGBoost)
└── config.py            # Settings (pydantic-settings)
```

---

## Database

MySQL 8 with the following tables:

| Table | Purpose |
|---|---|
| `content` | Cached health content with embeddings |
| `anbefaling_details` | Recommendation-specific fields |
| `theme_page_content` | Theme page to content mapping |
| `search_logs` | Search events |
| `search_results_shown` | Shown results with retrieval scores |
| `click_logs` | Click events with position |
| `position_propensity` | Position bias for IPS weighting |
| `content_type_config` | Searchable content type configuration |
| `training_datasets` | Uploaded training datasets |
| `training_presets` | Training configuration presets |

Schema: [`scripts/setup/init_database.sql`](scripts/setup/init_database.sql)

---

## Scripts

The `scripts/` directory contains all offline tooling — data pipelines, ML training, evaluation, and utilities. See [`scripts/README.md`](scripts/README.md) for a complete guide.

| Directory | Purpose |
|---|---|
| [`scripts/ml/`](scripts/ml/) | **ML pipeline**: query generation → E5 fine-tuning → embedding generation → evaluation |
| [`scripts/train/`](scripts/train/) | **LTR model training**: XGBoost LambdaMART from click logs |
| [`scripts/data/importing/`](scripts/data/importing/) | Import content from Helsedir API |
| [`scripts/data/migration/`](scripts/data/migration/) | Database migrations and backfills |
| [`scripts/data/generation/`](scripts/data/generation/) | Generate theme pages, role tags |
| [`scripts/data/maintenance/`](scripts/data/maintenance/) | Database cleanup utilities |
| [`scripts/setup/`](scripts/setup/) | Database schema, synthetic training data generation |
| [`scripts/test/`](scripts/test/) | Search evaluation and model testing |

---

## Configuration

See `.env.example` for all available settings. Key variables:

```bash
# Database
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_DATABASE=helsedir_ai

# Search weights
SEARCH_RRF_WEIGHT_BM25=0.3
SEARCH_RRF_WEIGHT_SEMANTIC=0.7
SEARCH_BOOST_TEMASIDE=1.15
SEARCH_ROLE_MATCH_BOOST=1.15

# ML (disabled by default — requires trained models)
ML_EMBEDDING_ENABLED=false   # Enable semantic search
ML_RANKING_ENABLED=false     # Enable LTR reranking

# LLM APIs (required for ML training pipeline only)
OPENAI_API_KEY=...           # For query generation (step 1)
```

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Full architecture and development reference
- **[scripts/README.md](scripts/README.md)** - Guide to all offline scripts and pipelines

---

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: MySQL 8
- **ML**: sentence-transformers (ONNX), XGBoost, PyTorch
- **NLP**: Snowball stemmer (Norwegian)
- **Container**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Python**: 3.11+
