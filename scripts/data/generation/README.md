# Generation Scripts

Generate static data and theme pages.

## Scripts

- **`generate_theme_pages.py`** - Generate theme page structure
  - Creates theme pages from paths
  - Assigns unique IDs
  - Builds parent/child relationships
  - Outputs to `data/theme_pages.json`

- **`link_theme_pages.py`** - Link theme pages to content
  - Scrapes Helsedir website for content under theme pages
  - Links content to theme categories
  - Updates database with theme associations

- **`populate_theme_pages.py`** - Populate theme pages into database
  - Inserts theme pages from JSON into content table
  - Uses batch insert for performance
  - Run after clearing content table or updating theme pages

## Usage

```bash
# Generate theme pages JSON
python scripts/data/generation/generate_theme_pages.py

# Populate theme pages into database
python scripts/data/generation/populate_theme_pages.py

# Link theme pages to content
python scripts/data/generation/link_theme_pages.py
```

## Theme Page Structure

Theme pages follow a hierarchical structure:
```
/forebygging-diagnose-og-behandling
  /kreft
    /brystkreft
  /hjerte-kar
```

Each theme page gets an ID: `9999-{category}-{uuid}`
