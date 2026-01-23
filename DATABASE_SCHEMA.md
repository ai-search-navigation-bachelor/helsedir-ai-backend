# Database Schema

Dette dokumentet beskriver tabellstrukturen i MySQL-databasen for helsedir-ai-backend.

## Database: `helsedir`

---

## Tabell: `content`

Lagrer alt innhold fra Helsedirektoratet API.

| Kolonne      | Type            | Beskrivelse                                         |
| ------------ | --------------- | --------------------------------------------------- |
| `id`         | VARCHAR(100) PK | Unik ID fra Helsedir API                            |
| `tittel`     | TEXT            | Dokumenttittel                                      |
| `tekst`      | LONGTEXT        | Fulltekst innhold                                   |
| `url`        | TEXT            | URL til originaldokument                            |
| `info_type`  | VARCHAR(50)     | Dokumenttype (retningslinje, veileder, informasjon) |
| `koder`      | JSON            | Fagkoder fra Helsedir (LIS-koder, etc.)             |
| `maalgruppe` | JSON            | Målgrupper (Fastlege, Sykepleier, etc.)             |
| `tags`       | JSON            | Auto-genererte semantiske tags                      |
| `embedding`  | BLOB            | Embedding-vektor (256D) for semantisk søk           |

**Indekser:**

- PRIMARY KEY: `id`
- INDEX: `info_type`

**Eksempel:**

```json
{
  "id": "abc123",
  "tittel": "Nasjonal faglig retningslinje for diabetes type 2",
  "tekst": "Denne retningslinjen omhandler...",
  "url": "https://helsedirektoratet.no/...",
  "info_type": "Retningslinje",
  "koder": { "lis-laeringsmal": ["NKI 036"] },
  "maalgruppe": ["Fastlege", "Sykepleier"],
  "tags": ["diabetes", "behandling", "type_2_diabetes", "retningslinje"],
  "embedding": null
}
```

---

## Tabell: `content_stats`

Sporer impressions og klikk for hver content item (brukes til CTR-beregning).

| Kolonne       | Type            | Beskrivelse                                       |
| ------------- | --------------- | ------------------------------------------------- |
| `content_id`  | VARCHAR(100) PK | Referanse til content.id                          |
| `impressions` | INT             | Antall ganger vist i søkeresultater               |
| `clicks`      | INT             | Antall ganger klikket på                          |
| `ctr`         | FLOAT           | Click-through rate (beregnet: clicks/impressions) |

**Indekser:**

- PRIMARY KEY: `content_id`
- FOREIGN KEY: `content_id` → `content(id)`

**Eksempel:**

```
content_id  | impressions | clicks | ctr
------------|-------------|--------|-------
abc123      | 150         | 12     | 0.0800
def456      | 200         | 5      | 0.0250
```

---

## Tabell: `search_logs`

Logger alle søk som utføres.

| Kolonne         | Type               | Beskrivelse                                  |
| --------------- | ------------------ | -------------------------------------------- |
| `id`            | INT AI PK          | Auto-increment ID                            |
| `search_id`     | VARCHAR(36) UNIQUE | UUID for dette søket                         |
| `query`         | TEXT               | Søketekst fra bruker                         |
| `role`          | VARCHAR(50)        | Brukerens rolle (fastlege, sykepleier, etc.) |
| `results_count` | INT                | Antall resultater returnert                  |
| `timestamp`     | DATETIME           | Tidspunkt for søk                            |

**Indekser:**

- PRIMARY KEY: `id`
- UNIQUE: `search_id`
- INDEX: `timestamp`

**Eksempel:**

```
id | search_id                            | query              | role     | results_count | timestamp
---|--------------------------------------|--------------------|-----------|--------------|-----------
1  | 550e8400-e29b-41d4-a716-446655440000 | diabetes behandling| fastlege | 10           | 2026-01-21 14:30:00
```

---

## Tabell: `search_results_shown`

Logger hvilke resultater som ble vist for hvert søk (brukes til ML-trening).

| Kolonne      | Type         | Beskrivelse                          |
| ------------ | ------------ | ------------------------------------ |
| `id`         | INT AI PK    | Auto-increment ID                    |
| `search_id`  | VARCHAR(36)  | Referanse til search_logs.search_id  |
| `content_id` | VARCHAR(100) | Referanse til content.id             |
| `position`   | INT          | Posisjon i resultatlisten (0-basert) |
| `score`      | FLOAT        | Relevans-score fra søkemotoren       |

**Indekser:**

- PRIMARY KEY: `id`
- INDEX: `search_id`
- FOREIGN KEY: `search_id` → `search_logs(search_id)`
- FOREIGN KEY: `content_id` → `content(id)`

**Eksempel:**

```
id | search_id                            | content_id | position | score
---|--------------------------------------|------------|----------|-------
1  | 550e8400-e29b-41d4-a716-446655440000 | abc123     | 0        | 15.3
2  | 550e8400-e29b-41d4-a716-446655440000 | def456     | 1        | 12.1
3  | 550e8400-e29b-41d4-a716-446655440000 | ghi789     | 2        | 8.7
```

---

## Tabell: `click_logs`

Logger bruker-klikk på søkeresultater (brukes til ML-trening).

| Kolonne      | Type         | Beskrivelse                                   |
| ------------ | ------------ | --------------------------------------------- |
| `id`         | INT AI PK    | Auto-increment ID                             |
| `search_id`  | VARCHAR(36)  | Referanse til search_logs.search_id           |
| `content_id` | VARCHAR(100) | Hvilken content som ble klikket               |
| `position`   | INT          | Posisjon i resultatlisten (0-basert)          |
| `query`      | TEXT         | Søketekst (denormalisert for lettere analyse) |
| `role`       | VARCHAR(50)  | Brukerens rolle (denormalisert)               |
| `timestamp`  | DATETIME     | Tidspunkt for klikk                           |

**Indekser:**

- PRIMARY KEY: `id`
- INDEX: `search_id`
- INDEX: `content_id`
- INDEX: `timestamp`
- FOREIGN KEY: `search_id` → `search_logs(search_id)`
- FOREIGN KEY: `content_id` → `content(id)`

**Eksempel:**

```
id | search_id                            | content_id | position | query               | role     | timestamp
---|--------------------------------------|------------|----------|---------------------|----------|----------
1  | 550e8400-e29b-41d4-a716-446655440000 | abc123     | 0        | diabetes behandling | fastlege | 2026-01-21 14:30:15
2  | 550e8400-e29b-41d4-a716-446655440000 | ghi789     | 2        | diabetes behandling | fastlege | 2026-01-21 14:30:45
```

---

## Relasjoner

```
content
  ├─→ content_stats (1:1)
  ├─→ search_results_shown (1:N)
  └─→ click_logs (1:N)

search_logs
  ├─→ search_results_shown (1:N)
  └─→ click_logs (1:N)
```

---

## ML Training Data Flow

### For Ranking Model:

```sql
-- Hent treningsdata: søk med minst ett klikk
SELECT
    sl.search_id,
    sl.query,
    sl.role,
    srs.content_id,
    srs.position,
    srs.score,
    CASE WHEN cl.content_id IS NOT NULL THEN 1 ELSE 0 END as clicked
FROM search_logs sl
INNER JOIN search_results_shown srs ON sl.search_id = srs.search_id
LEFT JOIN click_logs cl ON sl.search_id = cl.search_id
                        AND srs.content_id = cl.content_id
WHERE sl.search_id IN (
    SELECT DISTINCT search_id FROM click_logs
)
ORDER BY sl.search_id, srs.position;
```

**Resultat:**

- Positive samples: `clicked = 1` (brukeren klikket)
- Negative samples: `clicked = 0` (vist men ikke klikket)

### For Embedding Model:

```sql
-- Hent innhold med tags for supervised learning
SELECT
    id,
    tittel,
    tekst,
    tags
FROM content
WHERE tags IS NOT NULL;
```

**Resultat:**

- Primary tag (første i tags-arrayet) brukes som label
- Dokumenter med samme primary tag = positive pairs
- Dokumenter med ulike primary tags = negative pairs

---

## Statistikk-queries

### CTR per dokument:

```sql
SELECT
    c.id,
    c.tittel,
    cs.impressions,
    cs.clicks,
    cs.clicks / cs.impressions as ctr
FROM content c
LEFT JOIN content_stats cs ON c.id = cs.content_id
WHERE cs.impressions > 0
ORDER BY ctr DESC;
```

### Mest søkte termer:

```sql
SELECT
    query,
    COUNT(*) as search_count,
    SUM(results_count) as total_results_shown
FROM search_logs
GROUP BY query
ORDER BY search_count DESC
LIMIT 20;
```

### Mest klikkede dokumenter:

```sql
SELECT
    c.id,
    c.tittel,
    COUNT(cl.id) as click_count
FROM content c
INNER JOIN click_logs cl ON c.id = cl.content_id
GROUP BY c.id, c.tittel
ORDER BY click_count DESC
LIMIT 20;
```

---

## Oppsett

### Opprett database:

```bash
mysql -u root -p < scripts/setup_database.sql
```

### Legg til tags kolonne (hvis ikke eksisterer):

```bash
mysql -u root -p helsedir < scripts/add_tags_column.sql
```

### Importer innhold:

```bash
python scripts/import_content.py
```

### Generer tags:

```bash
python scripts/generate_tags.py
```

---

## Vedlikehold

### Nullstill statistikk:

```sql
TRUNCATE TABLE content_stats;
TRUNCATE TABLE search_logs;
TRUNCATE TABLE search_results_shown;
TRUNCATE TABLE click_logs;
```

### Slett gammelt innhold:

```sql
DELETE FROM content WHERE id NOT IN (
    SELECT DISTINCT content_id FROM search_results_shown
    WHERE timestamp > DATE_SUB(NOW(), INTERVAL 30 DAY)
);
```

### Database backup:

```bash
mysqldump -u root -p helsedir > backup_$(date +%Y%m%d).sql
```
