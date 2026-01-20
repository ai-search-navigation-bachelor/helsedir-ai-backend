# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based backend for an AI-powered content search system for Helsedirektoratet (Norwegian Directorate of Health). The system provides search, logging, and optional AI-enhanced features for health-related content.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp .env.example .env

# Or use the setup scripts
./scripts/setup_venv.sh    # Linux/Mac
scripts\setup_venv.bat     # Windows
```

### Running the Application
```bash
# Development mode (with auto-reload)
python scripts/run.py

# Or run directly with uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# With custom settings
HOST=127.0.0.1 PORT=8080 python scripts/run.py
```

### Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t helsedir-backend .
docker run -p 8000:8000 helsedir-backend
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_search.py

# Run specific test
pytest tests/test_search.py::test_search_endpoint

# Manual API testing
python scripts/test_api.py
```

### API Documentation
Once running, access interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

### Layered Structure
The codebase follows a clean layered architecture. All application code is under the `app/` directory:

1. **Routes Layer** (`app/routes/`): HTTP endpoint definitions
   - Handles request/response validation via Pydantic models
   - Thin layer that delegates to services
   - Each route file corresponds to a feature domain

2. **Services Layer** (`app/services/`): Business logic
   - `content_service.py`: Manages content loading and caching
   - `search_service.py`: Implements search algorithms (baseline → semantic)
   - `logging_service.py`: Handles event logging to JSONL
   - `helsedir_api_service.py`: Integration with Helsedirektoratet's external API

3. **Models Layer** (`app/models/`): Data validation
   - `schemas.py`: Pydantic models for all request/response types
   - Ensures type safety and automatic validation

4. **Configuration** (`app/config.py`): Centralized settings management
   - Uses `pydantic-settings` for environment-based configuration
   - Validates settings on startup

### Key Design Patterns

**Singleton Services**: Services are instantiated once as global instances:
```python
# In app/services/search_service.py
search_service = SearchService()

# In app/routes/search.py
from app.services.search_service import search_service
```

**Dependency Injection**: Services depend on each other via imports rather than FastAPI's DI system for simplicity in this MVP.

**JSONL Logging**: Events are appended to `logs/logs.jsonl` for easy streaming and analysis. Each line is a complete JSON object.

### Data Flow

1. **Search Request Flow**:
   - Client → POST `/search` (app/routes/search.py)
   - Route validates request with `SearchRequest` model
   - Route calls `search_service.search()`
   - Service loads content from `content_service`
   - Service scores and ranks results
   - Route logs search event via `logging_service`
   - Response validated with `SearchResponse` model
   - Client receives JSON results

2. **Content Loading**:
   - On startup, `content_service` loads `data/content.json`
   - Content cached in memory for fast access
   - Call `content_service.reload_content()` to refresh
   - Alternative: `content_service.load_from_api()` to fetch from Helsedirektoratet API

### Search Implementation

**Current (Baseline)**: Keyword-based scoring in `app/services/search_service.py`:
- Exact phrase in title: +10.0
- Keyword matches in title: +3.0 per keyword
- Keyword matches in body: +1.0 per keyword
- Exact phrase in body: +2.0
- Tag matches: +2.0 per keyword
- Role-based filtering
- Deterministic ordering (score DESC, id ASC)

**Future (Semantic)**: To upgrade to embeddings-based search:
1. Uncomment `sentence-transformers` and `faiss-cpu` in requirements.txt
2. Create `app/services/embedding_service.py` to generate embeddings
3. Modify `search_service.py` to use vector similarity
4. Store embeddings in `data/embeddings/` directory

## File Organization

```
helsedir-ai-backend/
├── app/                          # Application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point, CORS, router registration
│   ├── config.py                 # Settings management (env vars, paths)
│   │
│   ├── routes/                   # HTTP endpoints (thin layer)
│   │   ├── __init__.py
│   │   ├── health.py             # GET /health
│   │   ├── search.py             # POST /search
│   │   ├── logging.py            # POST /log
│   │   └── helsedir.py           # GET /helsedir/search (external API)
│   │
│   ├── services/                 # Business logic (core algorithms)
│   │   ├── __init__.py
│   │   ├── content_service.py    # Content loading and caching
│   │   ├── search_service.py     # Search algorithm implementation
│   │   ├── logging_service.py    # Event logging to JSONL
│   │   └── helsedir_api_service.py  # Helsedirektoratet API client
│   │
│   └── models/                   # Data validation
│       ├── __init__.py
│       └── schemas.py            # All Pydantic models
│
├── data/                         # Application data
│   └── content.json              # Content dataset (can be updated)
│
├── logs/                         # Runtime logs (gitignored)
│   └── logs.jsonl                # Event logs
│
├── scripts/                      # Utility scripts
│   ├── run.py                    # Alternative entry point
│   ├── test_api.py               # Manual API testing
│   ├── test_setup.py             # Setup verification
│   ├── setup_venv.sh             # Linux/Mac venv setup
│   └── setup_venv.bat            # Windows venv setup
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── .env.example                  # Environment variable template
├── .gitignore                    # Git exclusions
├── README.md                     # Project readme
├── CLAUDE.md                     # This file
├── TEAM_GUIDE.md                 # Team collaboration guide
└── API_INTEGRATION.md            # Helsedirektoratet API documentation
```

## API Endpoints

### POST /search
Search content with query and optional role filtering.

**Request**:
```json
{
  "query": "diabetes behandling",
  "role": "fastlege",
  "k": 10
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "2",
      "title": "Retningslinjer for behandling av diabetes type 2",
      "url": "https://...",
      "snippet": "...diabetes type 2 hos voksne...",
      "score": 12.5,
      "explanation": "Relevant: matches in title; relevant for fastlege"
    }
  ],
  "query": "diabetes behandling",
  "total": 1
}
```

### GET /helsedir/search
Direct search against the Helsedirektoratet external API.

**Query Parameters**:
- `QueryText` (required): Search query string
- `Filter`: OData filter expression (e.g., `infoType eq 'Retningslinje'`)
- `SearchMode`: `any` (default) or `all`
- `QueryType`: `simple` (default) or `full`
- `getFullInfobits`: `true` or `false` (default)

**Example**:
```
GET /helsedir/search?QueryText=diabetes&Filter=infoType eq 'Retningslinje'
```

**Response**:
```json
{
  "results": [
    {
      "id": "abc123",
      "title": "Diabetes - nasjonal faglig retningslinje",
      "body": "...",
      "url": "https://...",
      "content_type": "Retningslinje",
      "target_groups": ["Fastlege", "Sykepleier"]
    }
  ],
  "total": 15
}
```

### POST /log
Log user interactions for analytics.

**Request**:
```json
{
  "event_type": "click",
  "content_id": "2",
  "role": "fastlege"
}
```

**Response**:
```json
{
  "success": true
}
```

Event types: `search`, `click`, `role_change`

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Configuration

Environment variables (create `.env` from `.env.example`):

```bash
# Server
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
DEBUG=false

# Data paths
CONTENT_FILE=data/content.json
LOGS_FILE=logs/logs.jsonl

# Helsedirektoratet API
HELSEDIR_API_KEY=your_key_here
HELSEDIR_API_URL=https://api.helsedirektoratet.no

# Optional: OpenAI for chat/RAG features
# OPENAI_API_KEY=sk-...
```

## Content Data Format

The `data/content.json` file should contain an array of content items:

```json
[
  {
    "id": "unique-id",
    "title": "Content title",
    "body": "Full content text...",
    "url": "https://link-to-content",
    "content_type": "veileder|retningslinje|informasjon",
    "published_at": "2024-01-15T10:00:00Z",
    "target_groups": ["fastlege", "sykepleier"],
    "tags": ["diabetes", "behandling"]
  }
]
```

## Future Features

Models for these features are already defined in `app/models/schemas.py`:

### AI Tagging (POST /tag)
Add `app/routes/tagging.py` and implement rule-based or ML-based tag extraction.
Uses: `TagRequest`, `TagResponse`

### Chat/RAG (POST /chat)
Add `app/routes/chat.py` with LLM integration:
1. Use search to retrieve relevant content
2. Pass content + question to LLM
3. Return grounded answer with sources
Uses: `ChatRequest`, `ChatSource`, `ChatResponse`

## Upgrading to Semantic Search

To implement embeddings-based search:

1. **Install dependencies**:
   ```bash
   pip install sentence-transformers faiss-cpu
   ```

2. **Create embedding service** (`app/services/embedding_service.py`):
   - Load model: `SentenceTransformer('intfloat/multilingual-e5-base')`
   - Generate embeddings for all content
   - Save to disk for persistence

3. **Modify search service**:
   - Create FAISS index from embeddings
   - Convert query to embedding
   - Find K nearest neighbors
   - Return results with cosine similarity scores

4. **Precompute embeddings**:
   - Run embedding generation on startup or as separate script
   - Cache embeddings to avoid recomputation

## Norwegian Language Considerations

The content is in Norwegian (Bokmål). When implementing semantic search:
- Use multilingual models: `intfloat/multilingual-e5-base` or `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Consider Norwegian-specific models from NB (Nasjonalbiblioteket) if available

## Helsedirektoratet API Integration

The API integration is implemented in `app/services/helsedir_api_service.py`.

**Setup**:
1. Register at https://utvikler.helsedirektoratet.no
2. Subscribe to relevant API products
3. Get API key and add to `.env` as `HELSEDIR_API_KEY`

**Authentication**: Uses `Ocp-Apim-Subscription-Key` header.

**Available methods**:
- `search_infobits()` / `search_infobits_async()`: Search content
- `get_infobit_by_id()`: Get single item

See `API_INTEGRATION.md` for detailed documentation.

## Error Handling

All endpoints use FastAPI's HTTPException for errors:
- 400: Validation errors (automatic via Pydantic)
- 401/403: Authentication errors (Helsedir API)
- 404: Resource not found
- 500: Internal server errors
- 503: External service unavailable (Helsedir API)

Exceptions are logged to console in development mode.

## CORS Configuration

CORS is configured in `app/main.py` to allow all origins (`*`) for development.

**Important**: Update `allow_origins` to specific domains in production:
```python
allow_origins=["https://your-frontend-domain.no"]
```

## Related Documentation

- `README.md`: Project overview and quick start
- `TEAM_GUIDE.md`: Team collaboration, git workflow, code standards
- `API_INTEGRATION.md`: Detailed Helsedirektoratet API documentation
