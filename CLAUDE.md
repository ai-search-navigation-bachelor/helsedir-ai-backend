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

```
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

**Data Services** (`app/services/data/`):
- `base.py` - MySQL connection pool
- `database_service.py` - Facade for all database operations
- `content_repository.py` - Content CRUD operations
- `stats_repository.py` - CTR, impressions, clicks statistics
- `search_repository.py` - Search and click logging
- `ltr_repository.py` - Learning-to-rank training data
- `content_service.py` - Content loading and caching

**Search Services** (`app/services/search/`):
- `search_service.py` - Facade for all search methods
- `keyword_search.py` - Title-based keyword scoring
- `semantic_search.py` - E5 embedding-based search
- `hybrid_search.py` - Combined keyword + semantic search
- `feature_extractor.py` - ML feature extraction for logging
- `ml_service.py` - Ranking model inference

### Search Scoring (Title-Only)

Keyword scoring uses title matches only:
- Exact phrase in title: +10.0
- Full title coverage (all title words in query): +7.0
- Keyword matches in title: +3.0 per keyword

Configured via environment variables:
```
SEARCH_EXACT_PHRASE_TITLE_WEIGHT=10.0
SEARCH_FULL_TITLE_COVERAGE_WEIGHT=7.0
SEARCH_KEYWORD_TITLE_WEIGHT=3.0
```

### Learning-to-Rank Model

XGBoost LambdaMART model trained on click data.

**Features (12):**
1. `semantic_similarity` - Cosine similarity from E5 embeddings
2. `keyword_score_total` - Normalized keyword score (0-1)
3. `exact_title_proportion` - Proportion from exact title match
4. `full_coverage_proportion` - Proportion from full title coverage
5. `title_keyword_proportion` - Proportion from title keyword matches
6. `type_match` - Content type authority (retningslinje=0.9, veileder=0.8, etc.)
7. `role_match` - User role match with target groups
8. `code_match_count` - Number of matched codes (ICD/ICPC/SNOMED/LIS)
9. `lis_match` - LIS code match
10. `maalgruppe_match` - Target group match
11. `smoothed_ctr` - Windowed CTR (last 30 days)
12. `position` - Shown position (for IPS weighting)

**Training:**
- Groups by `search_id` (LTR requires grouping)
- Uses IPS (Inverse Propensity Scoring) for position bias correction
- Windowed CTR (30 days) for popularity signal
- Smoothed CTR: `(clicks + 1) / (impressions + 21)`

Enable with `ML_RANKING_ENABLED=true`.

## File Organization

```
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
│   │   ├── data/                  # Data access layer
│   │   │   ├── base.py            # Connection pool
│   │   │   ├── database_service.py # Facade
│   │   │   ├── content_repository.py
│   │   │   ├── stats_repository.py
│   │   │   ├── search_repository.py
│   │   │   ├── ltr_repository.py
│   │   │   └── content_service.py
│   │   │
│   │   ├── search/                # Search algorithms
│   │   │   ├── search_service.py  # Facade
│   │   │   ├── keyword_search.py
│   │   │   ├── semantic_search.py
│   │   │   ├── hybrid_search.py
│   │   │   ├── feature_extractor.py
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
│   ├── setup/
│   │   └── init_database.sql      # Database schema
│   └── ...
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

```
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
