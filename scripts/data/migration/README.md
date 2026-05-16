# Migration Scripts

One-time database backfills for enriching existing content rows. These scripts are safe to re-run — they skip rows that are already populated.

If you are setting up the project from scratch, run `scripts/setup/init_database.sql` instead — the schema is already up to date.

## Scripts

- **`backfill_dead_end_theme_pages.py`** - Flag theme pages that have no linked content (`is_dead_end_theme_page`)
- **`backfill_generisk_normerende_enheter.py`** - Import GNE content that is not exposed through normal API relations
- **`backfill_nki_indicator_ids.py`** - Match NKI quality indicators to content pages by URL and title
- **`backfill_pdf_report_chapter_urls.py`** - Set `document_url` for chapter pages that link to a PDF report
- **`backfill_publish_dates.py`** - Populate `forst_publisert` and `sist_faglig_oppdatert` from the API
- **`backfill_short_titles.py`** - Populate `kort_tittel` from the API for items where it is missing

## Usage

```bash
# Preview changes before applying (scripts that support --dry-run)
python scripts/data/migration/backfill_pdf_report_chapter_urls.py --dry-run

# Apply
python scripts/data/migration/backfill_publish_dates.py
python scripts/data/migration/backfill_nki_indicator_ids.py
```
