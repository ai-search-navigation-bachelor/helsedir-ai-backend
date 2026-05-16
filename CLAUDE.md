# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

FastAPI backend for AI-powered content search for Helsedirektoratet (Norwegian Directorate of Health). Features hybrid search (BM25 keyword + semantic embeddings), Reciprocal Rank Fusion (RRF), learning-to-rank with XGBoost LambdaMART, click tracking, NKI statistics integration, and a developer training pipeline for iterative model improvement.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Alternative: use the convenience runner script
python scripts/run.py

# Run tests
pytest

# Run with Docker
docker compose up --build

# API docs at http://localhost:8000/docs
```

## Architecture

### Layered Structure

```text
app/
├── routes/           # HTTP endpoints (thin layer)
├── controllers/      # Business logic orchestration
├── services/         # Core algorithms and data access
├── dto/              # Request/Response models (Pydantic)
├── entities/         # Domain models
├── exceptions/       # Custom exception classes
├── ml/               # Machine learning models
├── config.py         # Settings (pydantic-settings from .env)
└── constants.py      # Content types, roles, theme categories
```

### Services Layer (Repository Pattern)

**Repositories** (`app/services/repositories/`):
- `base.py` - MySQL connection pool management
- `content_repository.py` - Content CRUD, theme page links, column checks
- `search_repository.py` - Search logs, click logs, impressions
- `ltr_repository.py` - Learning-to-rank training data queries
- `training_repository.py` - Training datasets and presets management

**Data Services** (`app/services/data/`):
- `database_service.py` - Facade for all database operations
- `content_service.py` - Content loading and caching (by ID, by path)
- `document_metadata.py` - PDF/document URL detection, `has_text_content` flag
- `ehelsestandard_utils.py` - E-helsestandard attachment processing

**Search Services** (`app/services/search/`):
- `search_service.py` - Facade for all search methods
- `keyword_search.py` - Title-based keyword scoring
- `bm25_search.py` - BM25 search with hierarchy (parent/child relevance boosting)
- `bm25_hierarchy.py` - BM25HierarchyIndex: builds parent graph and propagates scores up the tree
- `semantic_search.py` - E5 embedding-based cosine similarity search
- `hybrid_search.py` - RRF fusion of BM25 + semantic, role/type boosting, ML reranking
- `rrf_fusion.py` - Reciprocal Rank Fusion algorithm
- `ml_service.py` - ML model inference, SHAP explanations, CTR caching
- `synonyms.py` - Norwegian synonym expansion

**Statistics** (`app/services/statistics/`):
- `nki_statistics_service.py` - NKI quality indicator fetching with caching
- `nki_matching.py` - NKI indicator ID matching logic

**Analytics** (`app/services/analytics/`):
- `logging_service.py` - Event logging to database

**External** (`app/services/external/`):
- `helsedir_api_service.py` - Async HTTP client for Helsedirektoratet API

### Hybrid Search Pipeline

1. **BM25 retrieval** - Title/body keyword scoring with Norwegian stemming and synonym expansion
2. **Semantic retrieval** - E5 embedding cosine similarity (if enabled)
3. **RRF fusion** - Reciprocal Rank Fusion combines both result lists (configurable weights)
4. **Boosting** - Content type boosts (temaside, retningslinje) and role match boost/penalty
5. **ML reranking** - Optional XGBoost LambdaMART reranking (if enabled)

BM25 hierarchy: parent content inherits relevance from top child matches for better recall on broad pages.

Configured via environment variables:
```bash
SEARCH_RRF_K=60                        # RRF fusion constant
SEARCH_RRF_WEIGHT_BM25=0.3            # BM25 weight in RRF
SEARCH_RRF_WEIGHT_SEMANTIC=0.7        # Semantic weight in RRF
SEARCH_BM25_HIERARCHY_ENABLED=true             # Parent/child relevance boosting
SEARCH_BM25_HIERARCHY_MAX_DEPTH=4             # How many levels up to propagate
SEARCH_BM25_HIERARCHY_DECAY=0.65             # Score decay per level
SEARCH_BM25_HIERARCHY_SOURCE_TOP_K=400       # Top child results to consider
SEARCH_BM25_HIERARCHY_TOP_CHILDREN=3         # Children per parent used for boost
SEARCH_BM25_HIERARCHY_TAIL_WEIGHT=0.35       # Weight for lower-ranked children
SEARCH_BM25_HIERARCHY_WEIGHT=0.8             # Overall hierarchy contribution weight
SEARCH_BM25_HIERARCHY_MIN_CONTRIBUTION=0.0001 # Minimum score to propagate
SEARCH_BOOST_TEMASIDE=1.15                   # Theme page score multiplier
SEARCH_BOOST_RETNINGSLINJE=1.10       # Guideline score multiplier
SEARCH_ROLE_MATCH_BOOST=1.15          # Role match multiplier
SEARCH_ROLE_MISMATCH_PENALTY=0.85     # Role mismatch multiplier
```

### Learning-to-Rank Model

XGBoost LambdaMART model trained on click data.

**Features (6):**
1. `semantic_score` - Normalized semantic similarity (0-1)
2. `bm25_score` - Normalized BM25 score (0-1)
3. `smoothed_ctr` - Windowed CTR (last 30 days, Bayesian-smoothed)
4. `role_match` - User role match with target groups (0-1)
5. `query_length` - Number of terms in query
6. `title_query_overlap` - Jaccard overlap between query and title terms

**Training:**
- Groups by `search_id` (LTR requires grouping)
- Uses IPS (Inverse Propensity Scoring) for position bias correction
- Windowed CTR (30 days) for popularity signal
- Smoothed CTR: `(clicks + 1) / (impressions + 21)`
- Supports training presets with configurable click simulation parameters

Enable with `ML_RANKING_ENABLED=true`.

## File Organization

```text
helsedir-ai-backend/
├── app/
│   ├── main.py                        # FastAPI app entry point with lifespan hooks
│   ├── config.py                      # Settings (pydantic-settings from .env)
│   ├── constants.py                   # Content types, roles, theme categories
│   │
│   ├── routes/                        # HTTP endpoints
│   │   ├── search.py                  # /search, /search/suggestions, /search/categorized, /search/category
│   │   ├── content.py                 # /content/{id}, /content/by-path, /content/{id}/statistics
│   │   ├── health.py                  # /health
│   │   ├── helsedir.py                # /helsedir/search, /helsedir/infobit/{id}
│   │   ├── temaside.py                # /theme-pages
│   │   ├── roles.py                   # /roles
│   │   ├── logging.py                 # /log
│   │   └── dev.py                     # /dev/* (training pipeline, model management)
│   │
│   ├── controllers/                   # Business logic
│   │   ├── search_controller.py       # Search orchestration, caching, pagination
│   │   ├── health_controller.py       # Health checks
│   │   ├── helsedir_controller.py     # Helsedir API integration
│   │   └── logging_controller.py      # Event logging
│   │
│   ├── services/
│   │   ├── repositories/              # Data access layer (MySQL)
│   │   │   ├── base.py                # Connection pool
│   │   │   ├── content_repository.py  # Content CRUD, theme pages
│   │   │   ├── search_repository.py   # Search/click logging
│   │   │   ├── ltr_repository.py      # LTR training data
│   │   │   └── training_repository.py # Datasets, presets, models
│   │   │
│   │   ├── data/                      # Data services
│   │   │   ├── database_service.py    # Facade for all DB operations
│   │   │   ├── content_service.py     # Content loading and caching
│   │   │   ├── document_metadata.py   # PDF/document URL detection
│   │   │   └── ehelsestandard_utils.py# E-helsestandard attachments
│   │   │
│   │   ├── search/                    # Search algorithms
│   │   │   ├── search_service.py      # Facade for all search methods
│   │   │   ├── keyword_search.py      # Title-based keyword scoring
│   │   │   ├── bm25_search.py         # BM25 with hierarchy boosting
│   │   │   ├── bm25_hierarchy.py      # Parent graph + score propagation
│   │   │   ├── semantic_search.py     # E5 embedding similarity
│   │   │   ├── hybrid_search.py       # RRF fusion + boosting + ML rerank
│   │   │   ├── rrf_fusion.py          # Reciprocal Rank Fusion
│   │   │   ├── ml_service.py          # ML inference + SHAP
│   │   │   └── synonyms.py            # Norwegian synonym expansion
│   │   │
│   │   ├── statistics/                # Quality indicators
│   │   │   ├── nki_statistics_service.py # NKI indicator fetching
│   │   │   └── nki_matching.py        # Indicator ID matching
│   │   │
│   │   ├── analytics/                 # Event logging
│   │   │   └── logging_service.py
│   │   │
│   │   └── external/                  # External APIs
│   │       └── helsedir_api_service.py
│   │
│   ├── dto/                           # Data transfer objects (Pydantic)
│   │   ├── request/
│   │   │   ├── search.py              # SearchRequest, CategorizedSearchRequest
│   │   │   └── logging.py             # LogRequest
│   │   └── response/
│   │       ├── search.py              # SearchResponse, SearchResult
│   │       ├── content.py             # ContentResponse
│   │       ├── health.py              # HealthResponse
│   │       ├── helsedir.py            # Helsedir API models
│   │       ├── logging.py             # LogResponse
│   │       └── statistics.py          # ContentStatisticsResponse
│   │
│   ├── entities/                      # Domain models
│   │   ├── content.py                 # ContentItem, ContentLink, AnbefalingFields, EhelsestandardFields
│   │   ├── search_log.py              # SearchLog entity
│   │   ├── click_log.py               # ClickLog entity
│   │   ├── search_result_shown.py     # SearchResultShown entity (LTR features)
│   │   └── content_stats.py           # ContentStats entity (aggregations)
│   │
│   ├── exceptions/                    # Custom exceptions
│   │   └── helsedir.py                # HelseDirectorateAPIError
│   │
│   └── ml/                            # ML models
│       ├── embedding_model.py         # E5 embeddings (sentence-transformers, ONNX)
│       └── ranking_model.py           # XGBoost LambdaMART reranker
│
├── scripts/
│   ├── ml/                            # ML pipeline (long running)
│   │   ├── 1_generate_queries.py      # Generate queries with Groq LLM (~3h)
│   │   ├── 2_finetune_gpl.py          # Fine-tune E5 model with GPL (~30-60min)
│   │   ├── 3_generate_embeddings.py   # Generate embeddings (~15-30min)
│   │   └── utils.py                   # Shared ML utilities
│   │
│   ├── data/
│   │   ├── importing/                 # Import from Helsedir API
│   │   │   ├── import_content.py      # Main content import
│   │   │   ├── backfill_anbefaling_details.py
│   │   │   ├── backfill_links.py
│   │   │   ├── backfill_paths.py
│   │   │   ├── backfill_document_metadata.py
│   │   │   ├── backfill_ehelsestandard_content.py
│   │   │   └── link_utils.py          # Shared utilities
│   │   ├── migration/                 # Database migrations
│   │   │   ├── migrate_links.py
│   │   │   ├── migrate_search_results_shown.py
│   │   │   ├── backfill_publish_dates.py
│   │   │   ├── backfill_short_titles.py
│   │   │   ├── backfill_nki_indicator_ids.py
│   │   │   ├── backfill_generisk_normerende_enheter.py
│   │   │   ├── backfill_dead_end_theme_pages.py
│   │   │   └── backfill_pdf_report_chapter_urls.py
│   │   ├── generation/                # Generate static data
│   │   │   ├── generate_theme_pages.py
│   │   │   ├── populate_theme_pages.py
│   │   │   ├── link_theme_pages.py
│   │   │   └── generate_role_tags.py
│   │   └── maintenance/               # Database cleanup
│   │       ├── reduce_content.py
│   │       ├── enrich_gpl_queries.py
│   │       └── generate_temaside_queries.py
│   │
│   ├── setup/                         # Initial setup
│   │   ├── init_database.sql          # Full database schema
│   │   ├── generate_training_data.py  # Synthetic training data generation
│   │   ├── generate_filtered_csv.py   # Filter CSV data for training
│   │   └── pretrain_all_models.py     # Pre-train models for presets
│   │
│   ├── train/                         # Model training
│   │   └── train_ranking_model.py     # Train LTR ranking model
│   │
│   └── test/                          # Test/eval scripts
│       ├── api/                       # API endpoint tests
│       ├── ml/                        # ML model and embedding tests
│       ├── search/                    # Search and ranking evaluation
│       └── data/                      # Data import and DB tests
│
├── tests/                             # Unit and integration tests (pytest)
│   ├── conftest.py                    # Pytest fixtures
│   ├── fixtures/                      # Test data fixtures
│   ├── unit/                          # Unit tests
│   │   ├── controllers/
│   │   ├── entities/
│   │   ├── routes/
│   │   ├── services/
│   │   └── scripts/
│   └── integration/                   # Integration tests
│       ├── test_routes_search.py
│       ├── test_routes_content.py
│       └── test_routes_health.py
│
├── models/                            # Trained model files
│   ├── finetuned-e5-gpl/              # Fine-tuned E5 embedding model
│   └── ranking/                       # XGBoost ranking models (by preset ID)
│
├── Dockerfile                         # Python 3.11-slim container
├── docker-compose.yml                 # MySQL 8 + Backend services
├── .github/workflows/                 # CI/CD
│   ├── tests-pr-dev.yml               # pytest on PR to dev
│   └── deploy.yml                     # Deployment pipeline
├── .env.example                       # Environment template
└── requirements.txt                   # Python dependencies
```

## API Endpoints

### GET /search
Search with pagination and RRF fusion.

**Query params:**
- `query` (required): Search text
- `role`: User role for personalization
- `method`: 'keyword', 'semantic', or 'hybrid' (default)
- `offset`: Results to skip (default: 0)
- `limit`: Results per page (default: 10, max: 50)
- `search_id`: Existing search_id for pagination

**Response:**
```json
{
  "results": [{"id": "...", "title": "...", "score": 0.85, ...}],
  "query": "diabetes",
  "total": 42,
  "search_id": "uuid",
  "offset": 0,
  "limit": 10,
  "has_next": true,
  "has_prev": false
}
```

### GET /search/suggestions
Autocomplete suggestions (top 5 titles matching query).

### GET /search/categorized
Results grouped by content type. Priority categories show all results.

### GET /search/category
Get all results in a specific content category.

### GET /content/{content_id}
Get content by ID with full enrichment (linked content, relations, anbefaling fields, ehelsestandard fields, NKI stats). Include `search_id` query param to log click.

### GET /content/by-path
Get content by helsedirektoratet.no path. Optional `search_id` for click tracking.

### GET /content/{content_id}/statistics
Get NKI quality indicator statistics for a content item.

### GET /helsedir/search
Live search against Helsedirektoratet API. Params: `QueryText`, `Filter` (OData), `SearchMode`, `QueryType`, `getFullInfobits`.

### GET /helsedir/infobit/{infobit_id}
Get complete infobit with nested children. Optional: `include_children`, `depth` (max: 10).

### GET /theme-pages
List all theme pages. Optional `category` filter (slug).

### GET /roles
Get available user roles for personalization.

### POST /log
Log user interaction events (clicks, impressions, etc.).

### /dev/* (Developer Tools)
- `GET /dev/search` - Search with configurable feature weights, SHAP explanations
- `POST /dev/generate` - Start background job for synthetic training data
- `GET /dev/generate/status/{job_id}` - Poll generation job progress
- `POST /dev/train` - Retrain LTR model from click logs
- `GET /dev/model` - Current model info (feature importances)
- `GET /dev/models` - List all pre-trained models
- `POST /dev/model/select` - Activate a pre-trained model
- `GET /dev/datasets` - List training datasets
- `POST /dev/datasets/upload` - Upload CSV training dataset
- `DELETE /dev/datasets/{dataset_id}` - Delete dataset
- `GET /dev/presets` - List training presets
- `POST /dev/presets` - Create training preset
- `DELETE /dev/presets/{preset_id}` - Delete preset

## Database Tables

- `content` - Cached content from Helsedir API (id, tittel, tekst, info_type, embedding, path, links, role_tags, has_text_content, document_url, nki_indicator_id, attachments, ehelsestandard fields, publish dates)
- `anbefaling_details` - Anbefaling-specific fields (praktisk, rasjonale, fordeler/ulemper, etc.)
- `theme_page_content` - Junction table linking theme pages to content items (many-to-many)
- `search_logs` - Search events (search_id UUID, query, role, timestamp)
- `search_results_shown` - Results shown per search with retrieval scores (semantic, BM25, RRF, role_match)
- `click_logs` - Click events (search_id, content_id, position)
- `position_propensity` - Position bias for IPS weighting (positions 1-10)
- `content_type_config` - Controls which info_types appear in search results (searchable flag)
- `training_datasets` - Uploaded CSV training datasets metadata
- `training_presets` - Named configs combining dataset with click simulation params

## Configuration

Key environment variables:

```bash
# Server
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=WARNING

# Database
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_DATABASE=helsedir_ai
MYSQL_POOL_SIZE=10  # max 32

# Helsedir API
HELSEDIR_API_KEY=...
HELSEDIR_NKI_API_KEY=...
HELSEDIR_API_URL=https://api.helsedirektoratet.no
NKI_STATISTICS_CACHE_TTL_SECONDS=21600

# Search weights
SEARCH_EXACT_PHRASE_TITLE_WEIGHT=10.0
SEARCH_FULL_TITLE_COVERAGE_WEIGHT=7.0
SEARCH_KEYWORD_TITLE_WEIGHT=3.0

# Hybrid search
SEARCH_RRF_K=60
SEARCH_RRF_WEIGHT_BM25=0.3
SEARCH_RRF_WEIGHT_SEMANTIC=0.7
SEARCH_BM25_HIERARCHY_ENABLED=true
SEARCH_BOOST_TEMASIDE=1.15
SEARCH_BOOST_RETNINGSLINJE=1.10
SEARCH_ROLE_MATCH_BOOST=1.15
SEARCH_ROLE_MISMATCH_PENALTY=0.85

# Categorized search
SEARCH_MIN_SCORE=0.45
SEARCH_CATEGORY_PREVIEW_COUNT=3

# ML
ML_EMBEDDING_ENABLED=false
ML_EMBEDDING_MODEL=models/finetuned-e5-gpl
ML_RANKING_ENABLED=false
ML_MODELS_DIR=models

# LLM APIs (for GPL training)
OPENAI_API_KEY=...
GROQ_API_KEY=...          # Up to 6 keys for parallel generation
```

## Scripts Organization

Scripts are organized by function in `scripts/`:

**ML Pipeline** (`scripts/ml/`):
- Numbered 1-2-3 to show execution order
- `1_generate_queries.py` - Generate synthetic queries using Groq LLM (~3h)
- `2_finetune_gpl.py` - Fine-tune E5 model with GPL (Generative Pseudo Labeling)
- `3_generate_embeddings.py` - Generate + store embeddings in database

**Data Management** (`scripts/data/`):
- `importing/` - Import content from Helsedir API, backfill various fields
- `migration/` - Database migrations and backfills
- `generation/` - Generate theme pages, role tags, and static data
- `maintenance/` - Database cleanup and query enrichment

**Setup** (`scripts/setup/`):
- `init_database.sql` - Full database schema
- `generate_training_data.py` - Synthetic training data generation
- `pretrain_all_models.py` - Pre-train models for all presets

**Training** (`scripts/train/`):
- `train_ranking_model.py` - Train the LTR ranking model

**Test/Evaluation** (`scripts/test/`):
- `api/` - API endpoint tests
- `ml/` - ML model and embedding tests
- `search/` - Search evaluation and ranking comparison
- `data/` - Data import and DB connection tests

## Design Patterns

**Singleton Services**: Global instances for stateful services:
```python
# In app/services/search/search_service.py
search_service = SearchService()

# Usage
from app.services.search.search_service import search_service
```

**Facade Pattern**: Main service classes delegate to specialized modules:
```python
# database_service delegates to repositories
# search_service delegates to keyword/semantic/hybrid search
```

**Repository Pattern**: Data access separated into focused repositories.

**Caching**: Multiple caching layers:
- Content service: in-memory cache by ID and path
- Search results: 30-second TTL cache in search controller
- Theme page children: pre-loaded on startup
- NKI statistics: configurable TTL (default 6 hours)
- CTR map: 10-minute refresh in ml_service

**Background Tasks**: Click logging, NKI stats, training data generation run asynchronously.

**Lifespan Hooks**: BM25 index pre-build, embedding model load, and ranking model load happen at startup.
