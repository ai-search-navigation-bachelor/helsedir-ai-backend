# Database Schema

Dette dokumentet beskriver tabellstrukturen i MySQL-databasen for helsedir-ai-backend.

## Database: `helsedir_ai`

---

## Tabell: `content`

Lagrer alt innhold fra Helsedirektoratet API.

| Kolonne      | Type            | Beskrivelse                                         |
| ------------ | --------------- | --------------------------------------------------- |
| `id`         | VARCHAR(100) PK | Unik ID fra Helsedir API                            |
| `tittel`     | TEXT            | Dokumenttittel                                      |
| `tekst`      | LONGTEXT        | Fulltekst innhold                                   |
| `info_type`  | VARCHAR(50)     | Dokumenttype (retningslinje, veileder, informasjon) |
| `koder`      | JSON            | Fagkoder fra Helsedir (ICD, ICPC, SNOMED, LIS)      |
| `maalgruppe` | JSON            | Målgrupper (Fastlege, Sykepleier, etc.)             |
| `embedding`  | BLOB            | Embedding-vektor for semantisk søk                  |

**Indekser:**

- PRIMARY KEY: `id`
- INDEX: `info_type`
- FULLTEXT: `tittel`, `tekst`

---

## Tabell: `search_logs`

Logger alle søk som utføres.

| Kolonne      | Type               | Beskrivelse                                  |
| ------------ | ------------------ | -------------------------------------------- |
| `id`         | INT AI PK          | Auto-increment ID                            |
| `search_id`  | VARCHAR(36) UNIQUE | UUID for dette søket                         |
| `query`      | TEXT               | Søketekst fra bruker                         |
| `role`       | VARCHAR(100)       | Brukerens rolle (fastlege, sykepleier, etc.) |
| `session_id` | VARCHAR(36)        | Sesjon-ID for å gruppere brukerinteraksjoner |
| `user_id`    | VARCHAR(36)        | Bruker-ID (hvis innlogget)                   |
| `timestamp`  | DATETIME           | Tidspunkt for søk                            |

**Indekser:**

- PRIMARY KEY: `id`
- UNIQUE: `search_id`
- INDEX: `timestamp`, `session_id`

---

## Tabell: `search_results_shown`

Logger hvilke resultater som ble vist for hvert søk med ML-features for LTR-trening.

| Kolonne                    | Type         | Beskrivelse                                       |
| -------------------------- | ------------ | ------------------------------------------------- |
| `id`                       | INT AI PK    | Auto-increment ID                                 |
| `search_id`                | VARCHAR(36)  | Referanse til search_logs.search_id               |
| `content_id`               | VARCHAR(100) | Referanse til content.id                          |
| `position`                 | INT          | Posisjon i resultatlisten (1, 2, 3, ...)          |
| `semantic_similarity`      | FLOAT        | Cosine similarity fra embedding (-1 til 1)        |
| `keyword_score_total`      | FLOAT        | Rå total keyword score (normaliseres ved trening) |
| `exact_title_proportion`   | FLOAT        | Andel av score fra eksakt tittel-match (0-1)      |
| `full_coverage_proportion` | FLOAT        | Andel fra full tittel-dekning (0-1)               |
| `title_keyword_proportion` | FLOAT        | Andel fra tittel keyword-matcher (0-1)            |
| `body_keyword_proportion`  | FLOAT        | Andel fra body keyword-matcher (0-1)              |
| `exact_body_proportion`    | FLOAT        | Andel fra eksakt body-match (0-1)                 |
| `type_match`               | FLOAT        | Innholdstype autoritetsnivå (0-1)                 |
| `role_match`               | FLOAT        | Brukerrolle-match score (0-1)                     |
| `code_match_count`         | INT          | Antall matchede koder (ICD/ICPC/SNOMED/LIS)       |
| `lis_match`                | TINYINT      | LIS-kode match (0/1)                              |
| `maalgruppe_match`         | TINYINT      | Målgruppe match (0/1)                             |
| `timestamp`                | DATETIME     | Tidspunkt resultat ble vist                       |

**Indekser:**

- PRIMARY KEY: `id`
- INDEX: `search_id`, `content_id`
- FOREIGN KEY: `search_id` → `search_logs(search_id)`

**Feature-forklaring:**

Proporsjonene (`*_proportion`) summerer til 1.0 og viser hvor keyword-scoren kom fra:
```
exact_title_proportion   = exact_title_score / total_keyword_score
full_coverage_proportion = full_coverage_score / total_keyword_score
title_keyword_proportion = title_keyword_score / total_keyword_score
body_keyword_proportion  = body_keyword_score / total_keyword_score
exact_body_proportion    = exact_body_score / total_keyword_score
```

---

## Tabell: `click_logs`

Logger bruker-klikk på søkeresultater.

| Kolonne      | Type         | Beskrivelse                         |
| ------------ | ------------ | ----------------------------------- |
| `id`         | INT AI PK    | Auto-increment ID                   |
| `search_id`  | VARCHAR(36)  | Referanse til search_logs.search_id |
| `content_id` | VARCHAR(100) | Hvilken content som ble klikket     |
| `position`   | INT          | Posisjon i resultatlisten           |
| `timestamp`  | DATETIME     | Tidspunkt for klikk                 |

**Indekser:**

- PRIMARY KEY: `id`
- INDEX: `search_id`, `content_id`
- FOREIGN KEY: `search_id` → `search_logs(search_id)`

---

## Tabell: `content_stats`

Aggregert statistikk for innholdsperformance.

| Kolonne        | Type            | Beskrivelse                              |
| -------------- | --------------- | ---------------------------------------- |
| `content_id`   | VARCHAR(100) PK | Referanse til content.id                 |
| `clicks`       | INT             | Totalt antall klikk                      |
| `impressions`  | INT             | Totalt antall visninger i søkeresultater |
| `last_updated` | DATETIME        | Sist oppdatert                           |

**Indekser:**

- PRIMARY KEY: `content_id`
- FOREIGN KEY: `content_id` → `content(id)`

---

## Tabell: `position_propensity`

Lagrer lært position bias for IPS-vekting i LTR-trening.

| Kolonne     | Type      | Beskrivelse                            |
| ----------- | --------- | -------------------------------------- |
| `position`  | INT PK    | Posisjon i resultatlisten (1, 2, ...) |
| `propensity`| FLOAT     | P(click \| position) - klikksannsynlighet |

**Default verdier:**

| Position | Propensity |
| -------- | ---------- |
| 1        | 1.00       |
| 2        | 0.70       |
| 3        | 0.55       |
| 4        | 0.45       |
| 5        | 0.40       |
| 6        | 0.35       |
| 7        | 0.30       |
| 8        | 0.28       |
| 9        | 0.26       |
| 10       | 0.24       |

**Brukes til:**
- Inverse Propensity Scoring (IPS) i LTR-trening
- Korrigerer for position bias (høyere posisjoner får flere klikk)

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

## LTR Training Data Query

Hent treningsdata for Learning-to-Rank modellen:

```sql
SELECT
    sl.search_id,
    srs.content_id,
    srs.position,

    -- Features
    srs.semantic_similarity,
    srs.keyword_score_total,
    srs.exact_title_proportion,
    srs.full_coverage_proportion,
    srs.title_keyword_proportion,
    srs.body_keyword_proportion,
    srs.exact_body_proportion,
    srs.type_match,
    srs.role_match,
    srs.code_match_count,
    srs.lis_match,
    srs.maalgruppe_match,

    -- Labels
    CASE WHEN cl.id IS NOT NULL THEN 1 ELSE 0 END AS clicked

FROM search_results_shown srs
JOIN search_logs sl ON srs.search_id = sl.search_id
LEFT JOIN click_logs cl ON srs.search_id = cl.search_id
                        AND srs.content_id = cl.content_id
WHERE sl.timestamp > DATE_SUB(NOW(), INTERVAL 180 DAY)
ORDER BY sl.search_id, srs.position;
```

**Normalisering ved trening:**

`keyword_score_total` normaliseres per `search_id` gruppe:
```python
max_kw = max(row["keyword_score_total"] for row in group)
for row in group:
    row["keyword_score_total"] /= max_kw  # Beste match = 1.0
```

---

## Statistikk-queries

### Smoothed CTR per dokument:

```sql
SELECT
    c.id,
    c.tittel,
    cs.impressions,
    cs.clicks,
    (cs.clicks + 1) / (cs.impressions + 21) as smoothed_ctr
FROM content c
LEFT JOIN content_stats cs ON c.id = cs.content_id
WHERE cs.impressions > 0
ORDER BY smoothed_ctr DESC;
```

### Mest søkte termer:

```sql
SELECT
    query,
    COUNT(*) as search_count
FROM search_logs
WHERE timestamp > DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY query
ORDER BY search_count DESC
LIMIT 20;
```

### Position bias analyse:

```sql
SELECT
    srs.position,
    COUNT(*) as impressions,
    SUM(CASE WHEN cl.id IS NOT NULL THEN 1 ELSE 0 END) as clicks,
    SUM(CASE WHEN cl.id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) as ctr
FROM search_results_shown srs
LEFT JOIN click_logs cl ON srs.search_id = cl.search_id
                        AND srs.content_id = cl.content_id
GROUP BY srs.position
ORDER BY srs.position;
```

---

## Oppsett

### Opprett database:

```bash
mysql -u root -p < scripts/setup/init_database.sql
```

### Importer innhold:

```bash
python scripts/import_content.py
```

---

## Vedlikehold

### Nullstill statistikk:

```sql
TRUNCATE TABLE content_stats;
TRUNCATE TABLE click_logs;
TRUNCATE TABLE search_results_shown;
TRUNCATE TABLE search_logs;
```

### Oppdater position propensity fra faktiske data:

```sql
REPLACE INTO position_propensity (position, propensity)
SELECT
    position,
    SUM(CASE WHEN cl.id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) as propensity
FROM search_results_shown srs
LEFT JOIN click_logs cl ON srs.search_id = cl.search_id
                        AND srs.content_id = cl.content_id
WHERE srs.position <= 10
GROUP BY srs.position;
```

### Database backup:

```bash
mysqldump -u root -p helsedir_ai > backup_$(date +%Y%m%d).sql
```
