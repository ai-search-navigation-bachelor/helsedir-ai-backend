# Data Scripts

Scripts for importing, migrating, and managing content data.

## Directory Structure

- **`importing/`** - Import content from Helsedirektoratet API
- **`migration/`** - Database migrations and schema updates
- **`generation/`** - Generate static data (theme pages, etc.)
- **`maintenance/`** - Database maintenance and cleanup

## Quick Start

```bash
# Import content from API
python scripts/data/importing/import_content.py --target 1000

# Migrate links to new format
python scripts/data/migration/migrate_links.py

# Generate theme pages
python scripts/data/generation/generate_theme_pages.py
```
