# Full Content API Changes

**Date:** 2026-01-23

## Summary

Endret API til å returnere **full informasjon** som standard i stedet for begrenset data med "les mer" knapper. Systemet lager nå en komplett "mock" av hele nettsiden.

## Changes Made

### 1. Search Endpoint (`/helsedir/search`)
**Før:**
- `getFullInfobits=false` (default)
- Returnerte bare tittel og sammendrag
- Måtte klikke "les mer" for å få full tekst

**Nå:**
- `getFullInfobits=true` (default) ✅
- Returnerer **FULL tekst** for alle resultater
- Alt innhold er synlig umiddelbart
- Kan sette `getFullInfobits=false` hvis man vil ha mindre data

### 2. Infobit Detail Endpoint (`/helsedir/infobit/{id}`)
**Før:**
- `include_children=false` (default)
- `depth=1` (default)
- `MAX_DEPTH=5`
- Hentet bare hovedinformasjon
- Måtte manuelt hente children

**Nå:**
- `include_children=true` (default) ✅
- `depth=10` (default) ✅
- `MAX_DEPTH=10` ✅
- Henter **ALLE children rekursivt**
- Komplett nøstet struktur med alt innhold
- Full "mock" av nettsiden

## API Behavior

### Search Results
```bash
# Returns FULL content automatically
GET /helsedir/search?QueryText=diabetes

# Response includes complete "tekst" field with all content
{
  "results": [
    {
      "tittel": "Diabetes type 2",
      "tekst": "FULL TEXT HERE... (complete content, not truncated)",
      "infoType": "veileder",
      ...
    }
  ]
}
```

### Infobit Details
```bash
# Returns complete nested structure automatically
GET /helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e

# Response includes ALL nested children to depth 10
{
  "tittel": "Main Content",
  "tekst": "COMPLETE TEXT...",
  "children": [
    {
      "tittel": "Kapittel 1",
      "data": {
        "tekst": "FULL CHAPTER TEXT..."
      },
      "children": [
        {
          "tittel": "Sub-section",
          "data": { ... },
          "children": [ ... ]  # Continues to depth 10
        }
      ]
    }
  ]
}
```

## Files Modified

1. **[app/dto/request/search.py](app/dto/request/search.py)**
   - Changed `get_full_infobits` default: `False` → `True`

2. **[app/routes/helsedir.py](app/routes/helsedir.py)**
   - Changed `MAX_DEPTH`: `5` → `10`
   - Changed `/infobit/{id}` defaults:
     - `include_children`: `False` → `True`
     - `depth`: `1` → `10`
   - Updated documentation to explain full content behavior

3. **[app/controllers/helsedir_controller.py](app/controllers/helsedir_controller.py)**
   - Updated method signatures with new defaults
   - Updated docstrings to reflect "full mock" behavior

## Benefits

✅ **Frontend får all data umiddelbart** - ingen "les mer" knapper nødvendig
✅ **Komplett "mock" av nettsiden** - all informasjon tilgjengelig
✅ **Færre API-kall** - ett søk gir all informasjon
✅ **Bedre brukeropplevelse** - ingen venting på ekstra data
✅ **Kan fortsatt optimalisere** - kan sette `getFullInfobits=false` eller lavere `depth` ved behov

## Backward Compatibility

Gamle API-kall fungerer fortsatt, men får nå **mer data** enn før:
- `/helsedir/search?QueryText=test` gir nå full innhold (før: bare sammendrag)
- `/helsedir/infobit/123` gir nå alle children (før: bare hovedinfo)

For å få **gammelt** oppførsel:
```bash
# Limit content
GET /helsedir/search?QueryText=test&getFullInfobits=false

# No children
GET /helsedir/infobit/123?include_children=false
```

## Testing

Start serveren og test:
```bash
python run.py
```

Test endpoints:
```bash
# Full search results
curl "http://localhost:8000/helsedir/search?QueryText=diabetes"

# Complete infobit with all nested content
curl "http://localhost:8000/helsedir/infobit/0006-0007-4569133a-5426-4072-a96b-3a4dc43def2e"
```

## Next Steps

Frontend kan nå:
1. Vise all tekst direkte uten "les mer" knapper
2. Navigere gjennom full content-struktur (children/kapitler)
3. Lage komplett "mock" av Helsedirektoratet sin nettside
4. Redusere antall API-kall dramatisk

---

*Endret av: Copilot*
*Dato: 2026-01-23*
