# Migration Scripts

Database migrations for schema and data updates.

## Scripts

- **`migrate_links.py`** - Migrate links to new id/href format
  - Converts internal links from `href` to `id` format
  - Removes invalid links with empty `href`
  - Normalizes `"None"` strings to actual `None`

## Usage

```bash
# Preview changes (dry-run)
python scripts/data/migration/migrate_links.py --dry-run

# Apply migration
python scripts/data/migration/migrate_links.py
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
