# Importing Scripts

Import and enrich content from Helsedirektoratet API.

## Scripts


- **`import_content.py`** - Main import script
  - Fetches content from Helsedir API
  - Processes links (internal/external)
  - Caches in database

- **`backfill_anbefaling_details.py`** - Fetch additional fields for anbefalinger
  - Fetches praktisk, rasjonale, fordeler_ulemper, etc.
  - Updates existing anbefaling records

- **`link_utils.py`** - Shared utilities for link processing
  - `extract_content_id_from_href()` - Extract ID from API URLs

## Usage

```bash
# Import 1000 content items
python scripts/data/importing/import_content.py --target 1000

# Import specific info type
python scripts/data/importing/import_content.py --info-type anbefaling --limit 500

# Backfill anbefaling details
python scripts/data/importing/backfill_anbefaling_details.py
```
