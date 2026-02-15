# Test Scripts

Test suites organized by domain.

## Directory Structure

- **`api/`** - API endpoint tests
- **`ml/`** - Machine learning and embedding tests
- **`search/`** - Search and ranking tests
- **`data/`** - Data import and validation tests

## Quick Start

```bash
# Run all tests in a category
python -m pytest scripts/test/ml/
python -m pytest scripts/test/search/

# Run specific test
python scripts/test/ml/test_e5_embedding.py
```

## Test Categories

### ML Tests
- E5 embedding generation
- Passage formatting
- Model integration

### Search Tests
- Keyword, semantic, hybrid search
- Ranking model performance
- Ranking comparisons

### API Tests
- Endpoint responses
- Error handling

### Data Tests
- Import validation
- Ranking data quality
