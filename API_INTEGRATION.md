# Helsedirektoratet API Integration

This document explains how to use the Helsedirektoratet API integration.

## Setup

### 1. Get API Key

1. Register at https://utvikler.helsedirektoratet.no
2. Subscribe to the "Innhold API" product
3. Copy your subscription key

### 2. Configure Environment

Add your API key to `.env`:

```bash
HELSEDIR_API_KEY=your_subscription_key_here
HELSEDIR_API_URL=https://api.helsedirektoratet.no
```

### 3. Test Connection

Run the test script to verify your setup:

```bash
python scripts/test_api.py
```

## Usage

### Basic Search

```python
from app.services.helsedir_api_service import helsedir_api_service

# Simple search
results = helsedir_api_service.search_infobits(
    query_text="diabetes"
)

# Print results
for item in results['value']:
    print(f"- {item['title']}")
```

### Search with Filters

```python
# Filter by target group (role)
results = helsedir_api_service.search_infobits(
    query_text="vaksine",
    filter_query="targetGroups/any(t: t eq 'fastlege')"
)

# Filter by content type
results = helsedir_api_service.search_infobits(
    query_text="behandling",
    filter_query="contentType eq 'veileder'"
)

# Combine filters
results = helsedir_api_service.search_infobits(
    query_text="psykisk helse",
    filter_query="contentType eq 'retningslinje' and targetGroups/any(t: t eq 'psykolog')"
)
```

### Get Full Content

By default, the API returns only metadata. To get full content:

```python
results = helsedir_api_service.search_infobits(
    query_text="smittevern",
    get_full_infobits=True  # Include full body/content
)

# Access full content
for item in results['value']:
    print(f"Title: {item['title']}")
    print(f"Body: {item.get('body', 'N/A')[:200]}...")  # First 200 chars
```

### Async Usage (in FastAPI endpoints)

```python
from app.services.helsedir_api_service import helsedir_api_service

@router.post("/search-live")
async def search_live_content(query: str):
    """Search live content from Helsedirektoratet API."""
    results = await helsedir_api_service.search_infobits_async(
        query_text=query,
        get_full_infobits=True
    )
    return results
```

### Load Content from API

Replace local `content.json` with live API data:

```python
from app.services.content_service import content_service

# Load all content from API
content_service.load_from_api(max_items=100)

# Load filtered content
content_service.load_from_api(
    query_text="helse",  # Only health-related content
    max_items=50
)

# Now search works with live data
from app.services.search_service import search_service
results = search_service.search("diabetes", role="fastlege")
```

## Query Parameters

### QueryText
Free text search query:
```python
query_text="diabetes behandling"
```

### Filter (OData syntax)

**Filter by content type:**
```python
filter_query="contentType eq 'veileder'"
filter_query="contentType eq 'retningslinje'"
filter_query="contentType eq 'oversikt'"
```

**Filter by target group:**
```python
filter_query="targetGroups/any(t: t eq 'fastlege')"
filter_query="targetGroups/any(t: t eq 'sykepleier')"
```

**Filter by tags:**
```python
filter_query="tags/any(t: t eq 'diabetes')"
```

**Combine filters with `and`:**
```python
filter_query="contentType eq 'veileder' and targetGroups/any(t: t eq 'fastlege')"
```

**Combine filters with `or`:**
```python
filter_query="contentType eq 'veileder' or contentType eq 'retningslinje'"
```

### SearchMode
- `"any"`: Match any of the search terms (default)
- `"all"`: Match all search terms

```python
search_mode="all"  # More restrictive
```

### QueryType
- `"simple"`: Simple query syntax (default)
- `"full"`: Lucene query syntax for advanced queries

```python
query_type="full"
```

## Error Handling

```python
from app.services.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError
)

try:
    results = helsedir_api_service.search_infobits(
        query_text="diabetes"
    )
except HelseDirectorateAPIError as e:
    print(f"API Error: {e}")
    # Handle error (e.g., use cached data, return error to user)
```

## Common Errors

### 401 Unauthorized
- **Cause**: Invalid or missing API key
- **Fix**: Check `HELSEDIR_API_KEY` in `.env`

### 403 Forbidden
- **Cause**: API key doesn't have access to the resource
- **Fix**: Verify your subscription includes "Innhold API"

### Timeout
- **Cause**: API is slow or unreachable
- **Fix**: Increase timeout or implement retry logic

```python
results = helsedir_api_service.search_infobits(
    query_text="diabetes",
    timeout=30.0  # Increase timeout to 30 seconds
)
```

## Best Practices

### 1. Cache API Results

```python
import time

class CachedAPIService:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    def search_with_cache(self, query_text):
        cache_key = f"search:{query_text}"

        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data

        # Fetch from API
        results = helsedir_api_service.search_infobits(query_text=query_text)

        # Store in cache
        self.cache[cache_key] = (results, time.time())

        return results
```

### 2. Handle Rate Limits

```python
import time
from functools import wraps

def rate_limit(max_calls=10, period=60):
    """Rate limit decorator: max_calls per period (seconds)."""
    calls = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls
            calls[:] = [c for c in calls if now - c < period]

            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                print(f"Rate limit reached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)

            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=10, period=60)
def search_with_rate_limit(query):
    return helsedir_api_service.search_infobits(query_text=query)
```

### 3. Fallback to Local Data

```python
try:
    # Try API first
    results = helsedir_api_service.search_infobits(query_text="diabetes")
except HelseDirectorateAPIError:
    # Fallback to local search
    from app.services.search_service import search_service
    results = {"value": search_service.search("diabetes")}
```

## Example: Live Search Endpoint

Create a new endpoint that searches live API data:

```python
# In app/routes/search.py

from app.services.helsedir_api_service import (
    helsedir_api_service,
    HelseDirectorateAPIError
)

@router.post("/search/live")
async def search_live(request: SearchRequest):
    """
    Search live content from Helsedirektoratet API.

    This bypasses the local content cache and searches directly.
    """
    try:
        # Build filter for role
        filter_query = None
        if request.role:
            filter_query = f"targetGroups/any(t: t eq '{request.role}')"

        # Search API
        api_results = await helsedir_api_service.search_infobits_async(
            query_text=request.query,
            filter_query=filter_query,
            get_full_infobits=True,
        )

        # Convert to SearchResult format
        results = []
        for item in api_results.get('value', [])[:request.k]:
            results.append(SearchResult(
                id=str(item.get('id', '')),
                title=item.get('title', ''),
                url=item.get('url', ''),
                snippet=item.get('body', '')[:200],
                score=1.0,  # API doesn't provide scores
                explanation="Live result from Helsedirektoratet API"
            ))

        return SearchResponse(
            results=results,
            query=request.query,
            total=len(results)
        )

    except HelseDirectorateAPIError as e:
        raise HTTPException(status_code=503, detail=f"API unavailable: {str(e)}")
```

## Testing

Run comprehensive API tests:

```bash
# Test basic connectivity
python scripts/test_api.py

# Test with pytest (if you create test files)
pytest tests/test_helsedir_api.py -v
```

## Resources

- **API Portal**: https://utvikler.helsedirektoratet.no
- **OData Filter Guide**: https://docs.microsoft.com/en-us/odata/concepts/queryoptions-overview
- **API Documentation**: Check the developer portal for full API specs
