# Log Format

All events are logged to `logs.jsonl` in JSONL format (one JSON object per line).

## Event Types

### Search Event

Logged when a user performs a search. Include `search_id` and `results_shown` for learning-to-rank training.

```json
{
  "event_type": "search",
  "timestamp": "2026-01-21T00:50:00",
  "search_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "diabetes",
  "role": "fastlege",
  "results_shown": [
    {"content_id": "1", "position": 0, "score": 12.5},
    {"content_id": "2", "position": 1, "score": 8.3}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | Always `"search"` |
| `timestamp` | string | ISO 8601 timestamp |
| `search_id` | string | **Required for ML** - UUID linking this search to its clicks |
| `query` | string | The search query |
| `role` | string? | User's role (optional) |
| `results_shown` | array | **Required for ML** - Results shown to user |

### Click Event

Logged when a user clicks on a search result. Include `search_id` to link to the originating search.

```json
{
  "event_type": "click",
  "timestamp": "2026-01-21T00:50:05",
  "search_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "diabetes",
  "content_id": "1",
  "position": 0,
  "role": "fastlege"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | Always `"click"` |
| `timestamp` | string | ISO 8601 timestamp |
| `search_id` | string | **Required for ML** - Must match the search that showed this result |
| `query` | string | The search query |
| `content_id` | string | ID of the clicked content |
| `position` | int? | Position in search results, 0-indexed |
| `role` | string? | User's role (optional) |

### Role Change Event

Logged when a user changes their role.

```json
{
  "event_type": "role_change",
  "timestamp": "2026-01-21T00:50:10",
  "role": "sykepleier"
}
```

## Important: search_id

The `search_id` field is **critical** for ML training. It links searches to their clicks.

**Why is this needed?**

Without `search_id`, we can't know which search a click belongs to:

```
User searches "diabetes"      → results [A, B, C]
User searches "diabetes type 2" → results [D, E, F]
User clicks result D
```

Without `search_id`, we might incorrectly link the click to the first search.
With `search_id`, we correctly link it to the second search.

**How it works:**

1. Frontend generates a UUID when performing a search
2. Frontend logs the search with that `search_id`
3. When user clicks a result, frontend includes the same `search_id`
4. ML training only uses searches that have at least one click

**Searches without clicks are ignored** during training to avoid polluting
the model with potentially irrelevant negative examples.

## API Examples

```bash
# Log search with results (note: search_id is a UUID)
curl -X POST http://localhost:8000/log \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "search",
    "search_id": "550e8400-e29b-41d4-a716-446655440000",
    "query": "diabetes behandling",
    "results_shown": [
      {"content_id": "1", "position": 0, "score": 12.5},
      {"content_id": "2", "position": 1, "score": 8.3}
    ]
  }'

# Log click on result (same search_id!)
curl -X POST http://localhost:8000/log \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "click",
    "search_id": "550e8400-e29b-41d4-a716-446655440000",
    "query": "diabetes behandling",
    "content_id": "1",
    "position": 0
  }'
```

## Frontend Implementation Example

```javascript
// Generate UUID (or use crypto.randomUUID() in modern browsers)
function generateSearchId() {
  return crypto.randomUUID();
}

// When user performs a search
async function performSearch(query) {
  const searchId = generateSearchId();

  // Call search API
  const response = await fetch(`/helsedir/search?QueryText=${query}`);
  const data = await response.json();

  // Map results for logging
  const resultsShown = data.results.map((result, index) => ({
    content_id: result.id,
    position: index,
    score: 0
  }));

  // Log the search
  await fetch('/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: 'search',
      search_id: searchId,
      query: data.query,
      results_shown: resultsShown
    })
  });

  // Store searchId for click logging
  return { ...data, searchId };
}

// When user clicks a result
async function logClick(searchId, query, contentId, position) {
  await fetch('/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: 'click',
      search_id: searchId,
      query: query,
      content_id: contentId,
      position: position
    })
  });
}
```
