# Migration Scripts

Database migrations for schema and data updates.

## Scripts

- **`migrate_links.py`** - Migrate links to new id/href format
  - Converts internal links from `href` to `id` format
  - Removes invalid links with empty `href`
  - Normalizes `"None"` strings to actual `None`
- **`backfill_pdf_report_chapter_urls.py`** - Backfill `document_url` for `pdf-av-rapporten` chapters
  - Fetches the public Helsedirektoratet HTML page for each matching chapter
  - Extracts the first `<a href="...pdf">` link and stores it in `content.document_url`
  - Reclassifies shortcode-only PDF chapters as `has_text_content = 0`

## Usage

```bash
# Preview changes (dry-run)
python scripts/data/migration/migrate_links.py --dry-run

# Apply migration
python scripts/data/migration/migrate_links.py

# Preview PDF chapter URL backfill
python scripts/data/migration/backfill_pdf_report_chapter_urls.py --dry-run

# Apply PDF chapter URL backfill
python scripts/data/migration/backfill_pdf_report_chapter_urls.py
```

## Link Format

Old format (before migration):
```json
{
  "rel": "forelder",
  "type": "kapittel",
  "href": "https://api.helsedirektoratet.no/innhold/kapitler/0006-0041-xxx",
  "strukturId": "xxx"
}
```

New format (after migration):
```json
{
  "rel": "forelder",
  "type": "kapittel",
  "id": "0006-0041-xxx"
}
```
