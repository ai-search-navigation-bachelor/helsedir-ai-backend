# Team Collaboration Guide

Guide for everyone working on the Helsedirektoratet AI Backend.

## First-Time Setup

### For All Team Members

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd helsedir-ai-backend
   ```

2. **Choose your development environment** (choose ONE):

   **Option A: Virtual Environment (recommended for local development)**
   ```bash
   # Windows
   setup_venv.bat

   # Linux/macOS
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

   **Option B: Docker (recommended for consistent environment)**
   ```bash
   # Build and start
   docker-compose up --build

   # Run in background
   docker-compose up -d
   ```

3. **Configure .env**:
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

## Daily Development

### With Virtual Environment

```bash
# Activate venv (must be done each time you open a new terminal)
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Start the server
python run.py

# When done
deactivate
```

### With Docker

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after dependency changes
docker-compose up --build
```

## Git Workflow

### Branching Strategy

```
main/master     <- Production-ready code
  └── dev       <- Development branch (default)
       └── feature/feature-name  <- Feature branches
```

### Workflow

1. **Before you start coding**:
   ```bash
   # Make sure you're up to date
   git checkout dev
   git pull origin dev
   ```

2. **Create a feature branch**:
   ```bash
   # Use descriptive names
   git checkout -b feature/search-improvements
   # or
   git checkout -b bugfix/logging-error
   ```

3. **Commit often with good messages**:
   ```bash
   git add .
   git commit -m "Add semantic search with embeddings"

   # Good commit messages:
   # - "Add endpoint for AI tagging"
   # - "Fix search scoring bug for role filtering"
   # - "Update content schema with new fields"
   # - "Improve search performance by caching"
   ```

4. **Push to remote**:
   ```bash
   git push origin feature/search-improvements
   ```

5. **Create Pull Request**:
   - Go to GitHub/GitLab
   - Create PR from your branch to `dev`
   - Describe your changes
   - Request code review from another team member

6. **Code Review**:
   - At least one other person must approve
   - Address any comments
   - Merge when approved

### Merge Conflicts

If you encounter merge conflicts:

```bash
# Update your branch with dev
git checkout dev
git pull origin dev
git checkout feature/your-feature
git merge dev

# Resolve conflicts in your editor
# Open files with conflicts and choose the correct code

# When done:
git add .
git commit -m "Merge dev into feature branch"
git push origin feature/your-feature
```

## Code Standards

### Python Style Guide

Follow **PEP 8** standards:

```python
# Good code:
def search_content(query: str, role: Optional[str] = None) -> List[SearchResult]:
    """Search for content based on query."""
    results = []
    # ...
    return results

# Avoid:
def searchContent(q, r=None):  # CamelCase, short variable names
    res = []
    return res
```

### Type Hints

Always use type hints:

```python
from typing import List, Optional

def get_content_by_id(content_id: str) -> Optional[ContentItem]:
    """Get a specific content item."""
    return content_service.get_content_by_id(content_id)
```

### Docstrings

Write docstrings for all functions:

```python
def calculate_score(item: ContentItem, query: str) -> float:
    """
    Calculate relevance score for a content item.

    Args:
        item: The content item to score
        query: Search query string

    Returns:
        Relevance score (higher is better)
    """
    # Implementation...
```

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test
pytest tests/test_search.py::test_search_with_role
```

### Writing Tests

Create tests for new features in `tests/`:

```python
def test_search_endpoint():
    """Test that search returns correct results."""
    response = client.post("/search", json={
        "query": "diabetes",
        "role": "fastlege",
        "k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 5
```
