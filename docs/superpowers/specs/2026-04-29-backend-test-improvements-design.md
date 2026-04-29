# Backend Test Improvements — Design Spec

**Date:** 2026-04-29
**Branch:** feat/synonyms-pregnancy-exercise

## Background

Baseline coverage run revealed 50% total coverage and 29 failing tests.
This spec defines the work to eliminate all failures and raise coverage to ~65–70%.

---

## Current State

| Module | Coverage | Issue |
|--------|----------|-------|
| `hybrid_search.py` | 32% | No tests |
| `ml_service.py` | 12% | No tests |
| `semantic_search.py` | 29% | No tests (requires model weights) |
| `synonyms.py` | 82% | Multi-word expansion paths uncovered |
| Total | **50%** | 29 failing tests |

### Failing tests (29)

- **28 async failures** — `pytest-asyncio` is not installed (missing from environment)
- **1 stemming failure** — `test_applies_snowball_stemming` expects `{'behandlinger'}` but
  the snowball stemmer correctly returns `{'behandling'}` (test expectation is wrong)

---

## Approach

### Phase 1 — Fix existing failures

**1a. Install pytest-asyncio**
- Add `pytest-asyncio>=0.24.0` to `requirements.txt` (already declared, just not installed in current env)
- Run `pip install pytest-asyncio` to verify fix
- Expected: 28 tests go from FAILED → PASSED

**1b. Fix stemming test**
- File: `tests/unit/services/search/test_keyword_search.py`
- Change: `assert stem_a == stem_b` where `stem_b = _tokenize("behandlinger")`
- Fix: Update assertion to reflect correct stemmer output — both forms should produce `{'behandling'}`
- The test intent (verify stemming works) is correct; only the expected value is wrong

---

### Phase 2 — New test files

#### `tests/unit/services/search/test_synonyms.py`

Target: synonyms.py lines 751, 756–763, 768 (currently at 82%)

Tests:
- `expand_terms` with single-word synonyms (e.g. `["diabetes"]` → includes `"sukkersyke"`)
- `expand_terms` with multi-word query match (e.g. `["høyt", "blodtrykk"]` → triggers multi-word path, includes `"hypertensjon"`)
- `expand_terms` where multi-word synonym overlaps — covered indices block re-expansion of individual tokens
- Original terms keep weight 1.0, synonyms get weight 0.5 (`SYNONYM_WEIGHT`)
- Terms not in lookup pass through unchanged
- Empty input returns only original term with weight 1.0
- Multi-word synonym where span is at end of term list (boundary check)

#### `tests/unit/services/search/test_hybrid_search.py`

**Static method tests (no mocking needed):**

`_normalize_combined_scores`:
- Empty list returns empty list
- Single item gets score 1.0
- All-same scores give 1.0 each
- Scores normalized to [0, 1] range
- Order preserved after normalization

`_extract_ranking_features`:
- Role match = 1.0/len(tags) when role is in tags
- Role match = 0.5 when both role and tags are None
- Role match = 0.3 when role is None but item has tags
- Role match = 0.0 when role set but no match in tags
- Default CTR used when content_id not in ctr_map
- Title-query Jaccard overlap computed correctly
- All 6 expected feature keys present in output

`_build_results`:
- Trims to k results
- Sets `id`, `title`, `info_type`, `score` correctly
- `rerank_info` is None when `rerank_score` is None
- `rerank_info` populated when `rerank_score` and `pre_rerank_position` are set

**Full `search()` flow (with mocks):**

Mock targets: `bm25_search.search`, `semantic_search.search`, `semantic_search.is_available`

- BM25-only flow (semantic unavailable): returns ranked results
- Role boost applied: item with matching role tag scores higher than same item without
- Role penalty applied: item with non-matching tags scores lower
- `rerank=False` skips ML model
- Empty BM25 + no semantic returns empty list
- Exception in RRF fusion returns empty list gracefully

#### `tests/unit/services/search/test_ml_service.py`

Mock strategy: patch `xgboost.Booster` and file system checks entirely.

Tests:
- `is_ranking_available()` returns False when no model loaded
- `load_ranking_model()` sets `_model` when file exists (mock `xgboost.Booster.load_model`)
- `get_ranking_scores(features_list)` returns list of floats with correct length
- `get_ctr_map()` returns dict (mock database call)
- `get_ranking_scores_with_contributions()` returns (score, dict) tuples
- Model not available gracefully returns default scores

---

## File Changes

| Action | File |
|--------|------|
| Edit | `requirements.txt` — ensure `pytest-asyncio>=0.24.0` present |
| Edit | `tests/unit/services/search/test_keyword_search.py` — fix stemming assertion |
| Create | `tests/unit/services/search/test_synonyms.py` |
| Create | `tests/unit/services/search/test_hybrid_search.py` |
| Create | `tests/unit/services/search/test_ml_service.py` |

---

## Success Criteria

- `pytest` exits with 0 failures
- `pytest --cov=app` reports total coverage ≥ 65%
- `hybrid_search.py` coverage ≥ 60%
- `synonyms.py` coverage ≥ 95%
- `ml_service.py` coverage ≥ 50%
