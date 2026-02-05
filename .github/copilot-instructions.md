# Copilot Instructions: Helsedirektoratet AI Backend

## Project Overview

FastAPI backend for AI-powered search and recommendations of Norwegian health content. Integrates with Helsedirektoratet's external API while maintaining local caching layer.

## Architecture

**Three-layer pattern:**
- **Routes** ([app/routes/](app/routes/)) - FastAPI endpoints with schema validation
- **Services** ([app/services/](app/services/)) - Business logic layer (singleton pattern with global instances)
- **Models** ([app/models/schemas.py](app/models/schemas.py)) - Pydantic models for validation

**Dual data sources:**
- **Local cache:** [data/content.json](data/content.json) loaded by `content_service` (default)
- **Live API:** Helsedirektoratet API via `helsedir_api_service` (optional)
- Route distinction: `/search` (local), `/helsedir/search` (live API)

**Service singletons pattern:** Services instantiate global objects at module level:
```python
# In service file:
search_service = SearchService()

# In routes:
from app.services.search_service import search_service
```

## Configuration

**Environment-based:** [app/config.py](app/config.py) uses `pydantic-settings` with `.env` file:
```python
from app.config import settings
settings.helsedir_api_key  # Access like this
```

**Critical env vars:**
- `HELSEDIR_API_KEY` - External API auth (get from https://utvikler.helsedirektoratet.no)
- `CONTENT_FILE`, `LOGS_FILE` - Relative paths from project root

## Development Workflows

**Local setup (two options):**
```bash
# Option 1: Virtual environment (preferred for quick dev)
setup_venv.bat  # Windows
./setup_venv.sh  # Unix
python run.py  # Starts uvicorn on port 8000

# Option 2: Docker (production-like)
docker-compose up --build
```

**Testing API:**
```bash
python scripts/test_api.py  # Tests Helsedirektoratet API connection
```

## Code Conventions

**Norwegian API field mapping:** External API returns Norwegian names - map to English internally:
```python
# API fields (Norwegian) → Internal fields (English)
tittel → title
tekst → body
maalgruppe → target_groups
infoType → content_type
```

**Type hints are mandatory:** All functions use full type annotations:
```python
def search(query: str, role: Optional[str] = None, k: int = 10) -> List[SearchResult]:
```

**FastAPI parameter aliases:** Query params use PascalCase aliases for external API compatibility:
```python
query: str = Query(..., alias="QueryText")  # Accepts ?QueryText=diabetes
```

**Error handling pattern:** Catch service-specific exceptions and convert to HTTPException:
```python
try:
    results = helsedir_api_service.search_infobits(...)
except HelseDirectorateAPIError as e:
    raise HTTPException(status_code=503, detail=f"API unavailable: {str(e)}")
```

## Search Implementation

**Baseline keyword scoring** ([app/services/search_service.py](app/services/search_service.py)):
- Exact phrase in title: 10 points
- Keyword in title: 3 points each
- Keyword in body: 1 point each
- Tag matches: 2 points each

**Future extension points:** Comments indicate where to add semantic search (sentence-transformers, FAISS) - currently using basic keyword matching.

## API Integration Details

**Async pattern for external calls:**
```python
# Service provides both sync and async
results = helsedir_api_service.search_infobits(...)  # Sync
results = await helsedir_api_service.search_infobits_async(...)  # Async in routes
```

**OData filter syntax** for `/helsedir/search`:
```python
filter_query="infoType eq 'veileder' and targetGroups/any(t: t eq 'fastlege')"
```

## Logging

**JSONL append-only log** ([logs/logs.jsonl](logs/logs.jsonl)) via `logging_service`:
```python
logging_service.log_event(event_type="search", query="diabetes", role="fastlege")
```

Track events: `search`, `click`, `role_change` for future analytics.

## Key Files to Reference

- [app/main.py](app/main.py) - App initialization, CORS, router registration
- [app/routes/search.py](app/routes/search.py) - Local cache search endpoint
- [app/routes/helsedir.py](app/routes/helsedir.py) - Live API passthrough endpoint
- [app/services/search_service.py](app/services/search_service.py) - Keyword scoring algorithm
- [app/services/helsedir_api_service.py](app/services/helsedir_api_service.py) - External API client
- [API_INTEGRATION.md](API_INTEGRATION.md) - Complete API usage guide with examples

## Dependencies

**Core stack:** FastAPI + Uvicorn + Pydantic v2 + httpx

**Commented out in requirements.txt:** ML dependencies (sentence-transformers, scikit-learn, openai, langchain) - uncomment when implementing semantic/AI features.

## Docker Considerations

**Volume mounts for development:**
- `./app:/app/app` - Live code reloading
- `./data:/app/data`, `./logs:/app/logs` - Data persistence

**Health check:** Polls `/health` endpoint every 30s.
