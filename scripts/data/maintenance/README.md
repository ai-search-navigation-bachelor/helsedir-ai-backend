# Maintenance Scripts

Database maintenance and cleanup utilities.

## Scripts

- **`reduce_content.py`** - Reduce database to target size
  - Keeps proportional distribution of info types
  - Useful for development/testing with smaller datasets
  - Preserves diversity

## Usage

```bash
# Reduce to 400 items (proportional)
python scripts/data/maintenance/reduce_content.py --target 400

# Ensure minimum 10 items per type
python scripts/data/maintenance/reduce_content.py --target 500 --min-per-type 10

# Preview changes
python scripts/data/maintenance/reduce_content.py --dry-run
```
