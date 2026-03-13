# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

FastAPI backend for AI-powered content search for Helsedirektoratet (Norwegian Directorate of Health). Features hybrid search (keyword + semantic), learning-to-rank, and click tracking.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# API docs at http://localhost:8000/docs
```

## Architecture

### Layered Structure

```text
app/
├── routes/           # HTTP endpoints (thin layer)
├── controllers/      # Business logic orchestration
├── services/         # Core algorithms and data access
├── dto/              # Request/Response models
├── entities/         # Domain models
├── ml/               # Machine learning models
└── config.py         # Settings
```

### Services Layer (Repository Pattern)

**Repositories** (`app/services/repositories/`):
- `base.py` - MySQL connection pool
- `content_repository.py` - Content CRUD operations
- `stats_repository.py` - CTR, impressions, clicks statistics
- `search_repository.py` - Search and click logging
- `ltr_repository.py` - Learning-to-rank training data

**Data Services** (`app/services/data/`):
- `database_service.py` - Facade for all database operations
- `content_service.py` - Content loading and caching

**Search Services** (`app/services/search/`):
- `search_service.py` - Facade for all search methods
- `keyword_search.py` - Title-based keyword scoring
- `semantic_search.py` - E5 embedding-based search
- `hybrid_search.py` - Combined keyword + semantic search
- `ml_service.py` - Ranking model inference

### Search Scoring (Title-Only)

Keyword scoring uses title matches only:
- Exact phrase in title: +10.0
- Full title coverage (all title words in query): +7.0
- Keyword matches in title: +3.0 per keyword

Configured via environment variables:
```bash
SEARCH_EXACT_PHRASE_TITLE_WEIGHT=10.0
SEARCH_FULL_TITLE_COVERAGE_WEIGHT=7.0
SEARCH_KEYWORD_TITLE_WEIGHT=3.0
```

### Learning-to-Rank Model

XGBoost LambdaMART model trained on click data.

**Features (7):**
1. `semantic_score` - Normalized semantic similarity (0-1)
2. `bm25_score` - Normalized BM25 score (0-1)
3. `smoothed_ctr` - Windowed CTR (last 30 days, Bayesian-smoothed)
4. `role_match` - User role match with target groups
5. `query_length` - Number of terms in query
6. `title_query_overlap` - Jaccard overlap between query and title terms
7. `content_freshness` - Decay function on `sist_faglig_oppdatert`

**Training:**
- Groups by `search_id` (LTR requires grouping)
- Uses IPS (Inverse Propensity Scoring) for position bias correction
- Windowed CTR (30 days) for popularity signal
- Smoothed CTR: `(clicks + 1) / (impressions + 21)`

Enable with `ML_RANKING_ENABLED=true`.

## File Organization

```text
helsedir-ai-backend/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── constants.py               # Allowed info types, etc.
│   │
│   ├── routes/                    # HTTP endpoints
│   │   ├── search.py              # GET /search
│   │   ├── content.py             # GET /content/{id}
│   │   ├── health.py              # GET /health
│   │   └── helsedir.py            # GET /helsedir/search
│   │
│   ├── controllers/               # Business logic
│   │   ├── search_controller.py   # Search orchestration
│   │   └── ...
│   │
│   ├── services/
│   │   ├── repositories/          # Data access layer
│   │   │   ├── base.py            # Connection pool
│   │   │   ├── content_repository.py
│   │   │   ├── stats_repository.py
│   │   │   ├── search_repository.py
│   │   │   └── ltr_repository.py
│   │   │
│   │   ├── data/                  # Data services
│   │   │   ├── database_service.py # Facade
│   │   │   └── content_service.py
│   │   │
│   │   ├── search/                # Search algorithms
│   │   │   ├── search_service.py  # Facade
│   │   │   ├── keyword_search.py
│   │   │   ├── semantic_search.py
│   │   │   ├── hybrid_search.py
│   │   │   └── ml_service.py
│   │   │
│   │   ├── analytics/             # Logging
│   │   │   └── logging_service.py
│   │   │
│   │   └── external/              # External APIs
│   │       └── helsedir_api_service.py
│   │
│   ├── dto/                       # Data transfer objects
│   │   ├── request/
│   │   └── response/
│   │
│   ├── entities/                  # Domain models
│   │   └── content.py
│   │
│   └── ml/                        # ML models
│       ├── embedding_model.py     # E5 embeddings
│       └── ranking_model.py       # XGBoost LTR
│
├── scripts/
│   ├── ml/                        # ML workflow (long running)
│   │   ├── 1_generate_queries.py  # Generate queries with LLM (~3h)
│   │   ├── 2_finetune_gpl.py      # Fine-tune E5 model (~30-60min)
│   │   └── 3_generate_embeddings.py # Generate embeddings (~15-30min)
│   │
│   ├── data/
│   │   ├── importing/             # Import from Helsedir API
│   │   │   ├── import_content.py
│   │   │   ├── backfill_anbefaling_details.py
│   │   │   └── link_utils.py      # Shared utilities
│   │   ├── migration/             # Database migrations
│   │   │   └── migrate_links.py
│   │   ├── generation/            # Generate static data
│   │   │   ├── generate_theme_pages.py
│   │   │   ├── populate_theme_pages.py
│   │   │   └── link_theme_pages.py
│   │   └── maintenance/           # Database cleanup
│   │       └── reduce_content.py
│   │
│   ├── test/
│   │   ├── api/                   # API tests
│   │   ├── ml/                    # ML & embedding tests
│   │   ├── search/                # Search & ranking tests
│   │   └── data/                  # Data & DB tests
│   │
│   └── setup/
│       └── init_database.sql      # Database schema
│
├── models/                        # Trained model files
├── .env.example                   # Environment template
└── requirements.txt
```

## API Endpoints

### GET /search
Search with pagination.

**Query params:**
- `query` (required): Search text
- `role`: User role for filtering
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

### GET /content/{id}
Get content by ID. Include `search_id` query param to log click.

```http
GET /content/abc123?search_id=uuid
```

## Database Tables

- `content` - Cached content from Helsedir API
- `content_stats` - All-time clicks/impressions per content
- `search_logs` - Search events (search_id, query, role)
- `search_results_shown` - Results shown with ML features
- `click_logs` - Click events with position

## Configuration

Key environment variables:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Database
MYSQL_HOST=localhost
MYSQL_DATABASE=helsedir_ai

# ML
ML_EMBEDDING_ENABLED=false
ML_RANKING_ENABLED=false

# Search weights
SEARCH_EXACT_PHRASE_TITLE_WEIGHT=10.0
SEARCH_FULL_TITLE_COVERAGE_WEIGHT=7.0
SEARCH_KEYWORD_TITLE_WEIGHT=3.0
```

## Scripts Organization

Scripts are organized by function in `scripts/`:

**ML Workflow** (`scripts/ml/`):
- Numbered 1-2-3 to show execution order
- `1_generate_queries.py` - Generate synthetic queries using Groq LLM
- `2_finetune_gpl.py` - Fine-tune E5 model with GPL (Generative Pseudo Labeling)
- `3_generate_embeddings.py` - Generate embeddings with fine-tuned model

**Data Management** (`scripts/data/`):
- `importing/` - Import content from Helsedir API
- `migration/` - Database schema and data migrations
- `generation/` - Generate theme pages and static data
- `maintenance/` - Database cleanup utilities

**Testing** (`scripts/test/`):
- `api/` - API endpoint tests
- `ml/` - ML model and embedding tests
- `search/` - Search and ranking tests
- `data/` - Data import and validation tests

Each subdirectory contains a README.md with usage examples.

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
