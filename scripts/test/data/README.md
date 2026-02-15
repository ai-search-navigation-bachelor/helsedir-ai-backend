# Data Tests

Tests for data import, validation, and database operations.

## Test Files

- **`test_import.py`** - Test content import from API
- **`test_setup.py`** - Test database setup
- **`check_ranking_data.py`** - Validate ranking training data
- **`test_db_connection.py`** - Quick database connection test

## Usage

```bash
# Test database connection
python scripts/test/data/test_db_connection.py

# Test import functionality
python scripts/test/data/test_import.py

# Check ranking data quality
python scripts/test/data/check_ranking_data.py
```

## Quick DB Check

The `test_db_connection.py` script provides a quick overview:
- Total content items
- Items with embeddings
- Search logs count
- Click logs count
