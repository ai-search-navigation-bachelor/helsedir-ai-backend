# Data Tests

Diagnostic scripts for validating database state and API connectivity.

## Scripts

- **`test_db_connection.py`** - Quick database connection check; shows content count, embedding count, search and click log totals
- **`test_import.py`** - Test a live import from the Helsedirektoratet API without saving to the database; useful for inspecting API response fields
- **`check_ranking_data.py`** - Verify there is enough click data to train the LTR ranking model

## Usage

```bash
# Check database connection and row counts
python scripts/test/data/test_db_connection.py

# Preview an API import (no DB writes)
python scripts/test/data/test_import.py --type retningslinje --count 3

# Check ranking data availability
python scripts/test/data/check_ranking_data.py
```
