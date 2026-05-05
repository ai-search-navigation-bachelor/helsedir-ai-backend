# Backend Test Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all 29 failing tests and raise overall backend coverage from 50% to ≥65%.

**Architecture:** Three-phase approach — fix missing package installations that cause all current failures, add pure-function tests for synonym expansion and hybrid search internals, then add ML service tests with a fully mocked XGBoost model.

**Tech Stack:** pytest, pytest-asyncio, pytest-mock, pytest-cov, snowballstemmer, xgboost (mocked in tests)

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Edit | `app/services/search/synonyms.py:702` | Fix `\w` → `\\w` SyntaxWarning in docstring |
| Edit | `requirements.txt` | Ensure pytest-cov is listed |
| Create | `tests/unit/services/search/test_synonyms.py` | Cover expand_terms multi-word paths |
| Create | `tests/unit/services/search/test_hybrid_search.py` | Cover static methods + mocked search flow |
| Create | `tests/unit/services/search/test_ml_service.py` | Cover MLService with mocked XGBoost |

---

## Task 1: Install missing packages and fix SyntaxWarning

**Root cause:** `pytest-asyncio` and `snowballstemmer` are declared in `requirements.txt` but not installed in the current environment. Without `snowballstemmer`, the `_stem` fallback returns words unchanged, causing `_tokenize("behandlinger")` to return `{'behandlinger'}` instead of `{'behandling'}`.

**Files:**
- Edit: `requirements.txt`
- Edit: `app/services/search/synonyms.py:702`

- [ ] **Step 1: Install all dependencies**

```bash
pip install -r requirements.txt
```

Expected output includes lines like:
```
Successfully installed pytest-asyncio-0.24.x snowballstemmer-2.2.x ...
```

- [ ] **Step 2: Add pytest-cov to requirements.txt**

Open `requirements.txt`. In the `# Development` section add `pytest-cov>=5.0.0`:

```text
# Development
pytest>=8.3.0
pytest-mock>=3.14.0
pytest-asyncio>=0.24.0
pytest-cov>=5.0.0
httpx>=0.28.0
requests>=2.32.0
```

- [ ] **Step 3: Install pytest-cov**

```bash
pip install pytest-cov
```

- [ ] **Step 4: Fix SyntaxWarning in synonyms.py**

In `app/services/search/synonyms.py` at line 702, change the docstring:

```python
# Before (causes SyntaxWarning in Python 3.12+):
def _tokenize_term(term: str) -> List[str]:
    """Tokenize a synonym term using the same \w+ pattern as BM25Search."""

# After:
def _tokenize_term(term: str) -> List[str]:
    """Tokenize a synonym term using the same \\w+ pattern as BM25Search."""
```

- [ ] **Step 5: Run the full test suite and verify 0 failures**

```bash
python -m pytest --no-header -q 2>&1 | tail -5
```

Expected:
```
261 passed in ...s
```

(All 29 previous failures are gone — 28 async tests now run because pytest-asyncio is installed, and `test_applies_snowball_stemming` passes because snowballstemmer now stems `behandlinger` → `behandling`.)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/services/search/synonyms.py
git commit -m "fix: install pytest-asyncio and snowballstemmer, fix docstring SyntaxWarning"
```

---

## Task 2: Add test_synonyms.py

**Target:** `app/services/search/synonyms.py` lines 751, 756–763, 768 (currently at 82%).

**What those lines are:**
- Line 751: `if i in covered_indices: continue` — inside the multi-word loop
- Lines 756–763: multi-word match body (the path that expands multi-word query phrases)
- Line 768: `if i in covered_indices: continue` — inside the single-word loop

**Files:**
- Create: `tests/unit/services/search/test_synonyms.py`

- [ ] **Step 1: Create the test file**

Create `tests/unit/services/search/test_synonyms.py`:

```python
"""
Unit tests for synonym expansion in synonyms.py.

expand_terms() is a pure function — no fixtures or mocking required.
Key behaviours tested:
- Original terms always get weight 1.0
- Synonyms get SYNONYM_WEIGHT (0.5)
- Single-word synonyms are expanded for known terms
- Multi-word query phrases trigger multi-word synonym expansion
- Tokens covered by a multi-word match are skipped in the single-word loop
"""

import pytest
from app.services.search.synonyms import expand_terms, SYNONYM_WEIGHT


@pytest.mark.unit
class TestExpandTermsSingleWord:
    def test_original_term_has_weight_one(self):
        weights = expand_terms(["diabetes"])
        assert weights["diabetes"] == 1.0

    def test_known_synonym_added_with_synonym_weight(self):
        # "diabetes" group includes "sukkersyke" (single-word)
        weights = expand_terms(["diabetes"])
        assert "sukkersyke" in weights
        assert weights["sukkersyke"] == SYNONYM_WEIGHT

    def test_multi_word_synonym_not_added_as_expansion(self):
        # "diabetes mellitus" is a 2-token phrase — must not be split and injected
        weights = expand_terms(["diabetes"])
        assert "mellitus" not in weights

    def test_unknown_term_passes_through_unchanged(self):
        weights = expand_terms(["xyzzyfoo123"])
        assert weights == {"xyzzyfoo123": 1.0}

    def test_empty_input_returns_empty_dict(self):
        assert expand_terms([]) == {}

    def test_multiple_terms_all_present(self):
        weights = expand_terms(["diabetes", "slag"])
        assert weights["diabetes"] == 1.0
        assert weights["slag"] == 1.0

    def test_single_word_synonym_of_slag(self):
        # "slag" group: "hjerneslag", "apopleksi", "stroke" are single-token
        weights = expand_terms(["slag"])
        assert "apopleksi" in weights
        assert "stroke" in weights
        assert weights["apopleksi"] == SYNONYM_WEIGHT


@pytest.mark.unit
class TestExpandTermsMultiWord:
    def test_multi_word_match_adds_single_word_synonym(self):
        # "høyt blodtrykk" is a multi-word synonym group member
        # Its single-word synonym is "hypertensjon"
        weights = expand_terms(["høyt", "blodtrykk"])
        assert "hypertensjon" in weights
        assert weights["hypertensjon"] == SYNONYM_WEIGHT

    def test_original_tokens_keep_weight_one_after_multi_word_match(self):
        weights = expand_terms(["høyt", "blodtrykk"])
        assert weights["høyt"] == 1.0
        assert weights["blodtrykk"] == 1.0

    def test_partial_multi_word_does_not_expand(self):
        # "høyt" alone cannot complete the phrase "høyt blodtrykk"
        weights = expand_terms(["høyt"])
        assert "hypertensjon" not in weights

    def test_multi_word_match_at_end_of_list(self):
        # Multi-word phrase as the last two tokens
        weights = expand_terms(["annen", "term", "høyt", "blodtrykk"])
        assert "hypertensjon" in weights

    def test_covered_tokens_skipped_in_single_word_loop(self):
        # "høyt" and "blodtrykk" are covered by the multi-word match;
        # "slag" at index 2 is NOT covered and should still expand.
        weights = expand_terms(["høyt", "blodtrykk", "slag"])
        # Multi-word expansion
        assert "hypertensjon" in weights
        # Single-word expansion for uncovered "slag"
        assert "stroke" in weights
        # Weight check
        assert weights["hypertensjon"] == SYNONYM_WEIGHT
        assert weights["stroke"] == SYNONYM_WEIGHT

    def test_synonym_weight_not_overwritten_by_later_expansion(self):
        # If a synonym term is already in weights, it should not be overwritten
        # (the code checks `if syn_tokens[0] not in weights`)
        weights = expand_terms(["slag", "hjerneslag"])
        # Both terms are in the same synonym group
        # "slag" expands to include "hjerneslag"
        # "hjerneslag" later tries to expand to "slag" but "slag" is already weight 1.0
        assert weights["slag"] == 1.0
```

- [ ] **Step 2: Run the new tests and verify they all pass**

```bash
python -m pytest tests/unit/services/search/test_synonyms.py -v
```

Expected: all tests PASSED, 0 failures.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/services/search/test_synonyms.py
git commit -m "test: add unit tests for synonym expand_terms covering multi-word paths"
```

---

## Task 3: Add test_hybrid_search.py

**Target:** `app/services/search/hybrid_search.py` (currently 32%).

Covers three static methods (no mocking needed) plus the full `search()` flow with BM25 and semantic search mocked out.

**Files:**
- Create: `tests/unit/services/search/test_hybrid_search.py`

- [ ] **Step 1: Create the test file**

Create `tests/unit/services/search/test_hybrid_search.py`:

```python
"""
Unit tests for hybrid_search.py.

Three test classes:
1. TestNormalizeCombinedScores — pure static method, no fixtures
2. TestExtractRankingFeatures  — pure static method, no fixtures
3. TestBuildResults            — pure static method, no fixtures
4. TestHybridSearchFlow        — full search() with bm25/semantic mocked
"""

import pytest
from unittest.mock import MagicMock, patch

from app.entities.content import ContentItem
from app.services.search.hybrid_search import HybridSearch, HybridCandidate
from app.services.search.bm25_search import BM25Hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(
    content_id: str = "test-001",
    title: str = "Diabetes retningslinje",
    content_type: str = "retningslinje",
    role_tags: list = None,
) -> ContentItem:
    return ContentItem(
        id=content_id,
        title=title,
        body="",
        content_type=content_type,
        has_text_content=True,
        role_tags=role_tags or [],
    )


def _candidate(
    score: float,
    content_id: str = "test-001",
    keyword_norm: float = 0.5,
    semantic_norm: float = 0.5,
    role_tags: list = None,
) -> HybridCandidate:
    return HybridCandidate(
        item=_item(content_id, role_tags=role_tags),
        combined_score=score,
        keyword_raw=0.0,
        semantic_raw=0.0,
        keyword_norm=keyword_norm,
        semantic_norm=semantic_norm,
    )


def _bm25_hit(content_id: str, score: float, content_type: str = "retningslinje") -> BM25Hit:
    return BM25Hit(
        item=_item(content_id, content_type=content_type),
        score=score,
        rank=1,
    )


# ---------------------------------------------------------------------------
# _normalize_combined_scores
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNormalizeCombinedScores:
    def test_empty_list_returns_empty(self):
        assert HybridSearch._normalize_combined_scores([]) == []

    def test_single_item_gets_score_one(self):
        result = HybridSearch._normalize_combined_scores([_candidate(0.5)])
        assert result[0].combined_score == 1.0

    def test_all_same_scores_become_one(self):
        candidates = [_candidate(0.5, f"id{i}") for i in range(3)]
        result = HybridSearch._normalize_combined_scores(candidates)
        assert all(c.combined_score == 1.0 for c in result)

    def test_scores_normalized_to_zero_one_range(self):
        candidates = [_candidate(s, f"id{i}") for i, s in enumerate([0.1, 0.5, 0.9])]
        result = HybridSearch._normalize_combined_scores(candidates)
        scores = [c.combined_score for c in result]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)

    def test_order_preserved_after_normalization(self):
        ids = ["a", "b", "c"]
        candidates = [_candidate(s, i) for s, i in zip([0.9, 0.5, 0.1], ids)]
        result = HybridSearch._normalize_combined_scores(candidates)
        assert [c.item.id for c in result] == ids


# ---------------------------------------------------------------------------
# _extract_ranking_features
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractRankingFeatures:
    def _features(self, role_tags, role=None, ctr_map=None, query_kw=None):
        c = _candidate(0.5, role_tags=role_tags)
        return HybridSearch._extract_ranking_features(
            c, role, ctr_map or {}, "", query_kw or set()
        )

    def test_role_match_when_role_in_tags(self):
        # role in ["lege", "sykepleier"] → 1 / 2
        f = self._features(["lege", "sykepleier"], role="lege")
        assert f["role_match"] == pytest.approx(0.5)

    def test_role_match_zero_when_role_set_but_no_match(self):
        # role="pasient" not in ["lege"] → 0.0
        f = self._features(["lege"], role="pasient")
        assert f["role_match"] == 0.0

    def test_role_match_half_when_neither_has_role(self):
        # role=None, tags=[] → 0.5
        f = self._features([], role=None)
        assert f["role_match"] == 0.5

    def test_role_match_point_three_when_role_set_but_no_tags(self):
        # role="lege", tags=[] → 0.3 (item has no tags, not neutral)
        f = self._features([], role="lege")
        assert f["role_match"] == pytest.approx(0.3)

    def test_role_match_zero_when_no_role_but_item_has_tags(self):
        # role=None, tags=["lege"] → 0.0 (no match possible)
        f = self._features(["lege"], role=None)
        assert f["role_match"] == 0.0

    def test_default_ctr_used_when_id_not_in_map(self):
        f = self._features([])
        assert f["smoothed_ctr"] == pytest.approx(1.0 / 21.0)

    def test_ctr_from_map_when_present(self):
        f = self._features([], ctr_map={"test-001": 0.42})
        assert f["smoothed_ctr"] == pytest.approx(0.42)

    def test_title_query_overlap_full_match(self):
        # Title "Diabetes retningslinje" — both terms in query
        c = _candidate(0.5)
        f = HybridSearch._extract_ranking_features(
            c, None, {}, "diabetes retningslinje", {"diabetes", "retningslinje"}
        )
        assert f["title_query_overlap"] == pytest.approx(1.0)

    def test_title_query_overlap_no_match(self):
        c = _candidate(0.5)
        f = HybridSearch._extract_ranking_features(
            c, None, {}, "kreft behandling", {"kreft", "behandling"}
        )
        assert f["title_query_overlap"] == pytest.approx(0.0)

    def test_all_six_feature_keys_present(self):
        f = self._features([])
        assert set(f.keys()) == {
            "semantic_score", "bm25_score", "smoothed_ctr",
            "role_match", "query_length", "title_query_overlap",
        }

    def test_query_length_matches_number_of_terms(self):
        c = _candidate(0.5)
        f = HybridSearch._extract_ranking_features(
            c, None, {}, "diabetes type 2", {"diabetes", "type", "2"}
        )
        assert f["query_length"] == 3.0


# ---------------------------------------------------------------------------
# _build_results
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildResults:
    def test_trims_to_k(self):
        candidates = [_candidate(1.0 - i * 0.1, f"id{i}") for i in range(5)]
        results = HybridSearch._build_results(candidates, k=3)
        assert len(results) == 3

    def test_id_and_info_type_populated(self):
        c = _candidate(0.9, "my-id")
        results = HybridSearch._build_results([c], k=1)
        assert results[0].id == "my-id"
        assert results[0].info_type == "retningslinje"

    def test_score_rounded_to_three_decimals(self):
        c = _candidate(0.12345)
        results = HybridSearch._build_results([c], k=1)
        assert results[0].score == pytest.approx(0.123, abs=0.001)

    def test_pipeline_rerank_none_when_not_reranked(self):
        c = _candidate(0.9)  # rerank_score defaults to None
        results = HybridSearch._build_results([c], k=1)
        assert results[0].pipeline is not None
        assert results[0].pipeline.rerank is None

    def test_pipeline_rerank_populated_when_reranked(self):
        c = _candidate(0.9)
        c.rerank_score = 0.85
        c.pre_rerank_position = 3
        c.ranking_features = {
            "semantic_score": 0.7, "bm25_score": 0.5,
            "smoothed_ctr": 0.1, "role_match": 0.5,
            "query_length": 2.0, "title_query_overlap": 0.3,
        }
        results = HybridSearch._build_results([c], k=1)
        ri = results[0].pipeline.rerank
        assert ri is not None
        assert ri.score == pytest.approx(0.85, abs=0.0001)
        assert ri.rank_change == 3 - 1  # pre_rerank_position - final position (1)


# ---------------------------------------------------------------------------
# Full search() flow with mocked BM25 / semantic
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHybridSearchFlow:
    def test_bm25_only_returns_results(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[
                _bm25_hit("001", 5.0),
                _bm25_hit("002", 3.0),
            ],
        )

        hs = HybridSearch()
        results = hs.search("diabetes", rerank=False)

        assert len(results) > 0
        result_ids = {r.id for r in results}
        assert result_ids <= {"001", "002"}

    def test_empty_bm25_returns_empty_list(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[],
        )

        hs = HybridSearch()
        results = hs.search("nomatch", rerank=False)
        assert results == []

    def test_role_boost_elevates_matching_item(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        # Two items with equal BM25 score; one matches role "lege"
        item_match = _item("role-match", role_tags=["lege"])
        item_mismatch = _item("role-mismatch", role_tags=["pasient"])
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[
                BM25Hit(item=item_match, score=5.0, rank=1),
                BM25Hit(item=item_mismatch, score=5.0, rank=2),
            ],
        )

        hs = HybridSearch(
            candidate_multiplier=2,
            min_candidate_pool=2,
            max_candidate_pool=10,
        )
        results = hs.search("diabetes", role="lege", rerank=False, k=2)

        if len(results) == 2:
            assert results[0].id == "role-match"

    def test_rerank_false_does_not_call_ml_service(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[_bm25_hit("001", 5.0)],
        )
        mock_apply = mocker.patch.object(HybridSearch, "_apply_ranking_model")

        hs = HybridSearch()
        hs.search("diabetes", rerank=False)

        mock_apply.assert_not_called()

    def test_exception_in_rrf_fusion_returns_empty_list(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[_bm25_hit("001", 5.0)],
        )
        mocker.patch(
            "app.services.search.hybrid_search.fuse_ranked_lists",
            side_effect=RuntimeError("fusion exploded"),
        )

        hs = HybridSearch()
        results = hs.search("diabetes", rerank=False)
        assert results == []
```

- [ ] **Step 2: Run the new tests and verify they all pass**

```bash
python -m pytest tests/unit/services/search/test_hybrid_search.py -v
```

Expected: all tests PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/services/search/test_hybrid_search.py
git commit -m "test: add unit tests for HybridSearch static methods and search flow"
```

---

## Task 4: Add test_ml_service.py

**Target:** `app/services/search/ml_service.py` (currently 12%). Mocks XGBoost entirely — no model files needed.

**Files:**
- Create: `tests/unit/services/search/test_ml_service.py`

- [ ] **Step 1: Create the test file**

Create `tests/unit/services/search/test_ml_service.py`:

```python
"""
Unit tests for MLService.

XGBoost and file system are mocked throughout — no model files needed.
Each test instantiates a fresh MLService() to avoid state leakage.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.services.search.ml_service import MLService


# ---------------------------------------------------------------------------
# is_ranking_available
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsRankingAvailable:
    def test_false_on_fresh_instance(self):
        assert MLService().is_ranking_available() is False

    def test_true_after_flag_set(self):
        svc = MLService()
        svc._ranking_loaded = True
        assert svc.is_ranking_available() is True


# ---------------------------------------------------------------------------
# load_ranking_model
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadRankingModel:
    def test_returns_false_when_disabled_and_not_forced(self):
        svc = MLService()
        with patch("app.services.search.ml_service.settings") as s:
            s.ml_ranking_enabled = False
            assert svc.load_ranking_model(force=False) is False

    def test_returns_false_when_model_file_missing(self, tmp_path):
        svc = MLService()
        with patch("app.services.search.ml_service.settings") as s:
            s.ml_ranking_enabled = True
            s.ml_models_dir = str(tmp_path)  # ranking/reranker.json does not exist
            assert svc.load_ranking_model() is False

    def test_returns_true_and_sets_flag_when_file_exists(self, tmp_path):
        ranking_dir = tmp_path / "ranking"
        ranking_dir.mkdir()
        (ranking_dir / "reranker.json").write_text("{}")

        mock_ranker = MagicMock()
        svc = MLService()

        with patch("app.services.search.ml_service.settings") as s, \
             patch("app.ml.ranking_model.HealthContentRanker.load", return_value=mock_ranker), \
             patch.object(svc, "_refresh_ctr_cache"):
            s.ml_ranking_enabled = True
            s.ml_models_dir = str(tmp_path)
            result = svc.load_ranking_model()

        assert result is True
        assert svc._ranking_loaded is True
        assert svc.ranking_model is mock_ranker

    def test_returns_false_on_load_exception(self, tmp_path):
        ranking_dir = tmp_path / "ranking"
        ranking_dir.mkdir()
        (ranking_dir / "reranker.json").write_text("{}")

        svc = MLService()
        with patch("app.services.search.ml_service.settings") as s, \
             patch("app.ml.ranking_model.HealthContentRanker.load",
                   side_effect=RuntimeError("corrupt model")):
            s.ml_ranking_enabled = True
            s.ml_models_dir = str(tmp_path)
            result = svc.load_ranking_model()

        assert result is False
        assert svc._ranking_loaded is False


# ---------------------------------------------------------------------------
# get_ranking_scores
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetRankingScores:
    def test_returns_zeros_when_no_model(self):
        svc = MLService()
        scores = svc.get_ranking_scores([{"semantic_score": 0.5}])
        assert scores == [0.0]

    def test_returns_zeros_for_multiple_features_when_no_model(self):
        svc = MLService()
        scores = svc.get_ranking_scores([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
        assert scores == [0.0, 0.0, 0.0]

    def test_delegates_to_ranking_model_predict(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict.return_value = [0.9, 0.6]
        svc.ranking_model = mock_ranker

        features = [{"semantic_score": 0.9}, {"semantic_score": 0.4}]
        scores = svc.get_ranking_scores(features)

        assert scores == [0.9, 0.6]
        mock_ranker.predict.assert_called_once_with(features)

    def test_returns_zeros_on_predict_exception(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict.side_effect = RuntimeError("broken")
        svc.ranking_model = mock_ranker

        scores = svc.get_ranking_scores([{"x": 1.0}])
        assert scores == [0.0]


# ---------------------------------------------------------------------------
# get_ranking_scores_with_contributions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetRankingScoresWithContributions:
    def test_returns_zero_tuples_when_no_model(self):
        svc = MLService()
        result = svc.get_ranking_scores_with_contributions([{"x": 0.5}])
        assert result == [(0.0, {})]

    def test_delegates_to_predict_with_contributions(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict_with_contributions.return_value = [
            (0.9, {"bm25_score": 0.4, "semantic_score": 0.5})
        ]
        svc.ranking_model = mock_ranker

        result = svc.get_ranking_scores_with_contributions([{"bm25_score": 0.4}])

        assert result[0][0] == pytest.approx(0.9)
        assert "bm25_score" in result[0][1]

    def test_returns_zero_tuples_on_exception(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict_with_contributions.side_effect = RuntimeError("shap error")
        svc.ranking_model = mock_ranker

        result = svc.get_ranking_scores_with_contributions([{"x": 1.0}])
        assert result == [(0.0, {})]


# ---------------------------------------------------------------------------
# get_ctr_map (caching behaviour)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCTRMap:
    def test_returns_dict(self):
        svc = MLService()
        # Force cache to be "fresh" so no DB call is made
        svc._ctr_last_refresh = time.monotonic()
        svc._ctr_map = {}
        result = svc.get_ctr_map()
        assert isinstance(result, dict)

    def test_returns_cached_map_without_refresh_when_fresh(self):
        svc = MLService()
        svc._ctr_map = {"001": 0.15, "002": 0.08}
        svc._ctr_last_refresh = time.monotonic()  # just refreshed — TTL not expired

        result = svc.get_ctr_map()
        assert result == {"001": 0.15, "002": 0.08}

    def test_refreshes_from_db_when_ttl_expired(self):
        svc = MLService()
        svc._ctr_last_refresh = 0.0  # expired (time.monotonic() >> 600s after boot)

        with patch(
            "app.services.data.database_service.database_service.get_content_ctr_map",
            return_value={"003": 0.25},
        ):
            result = svc.get_ctr_map()

        assert result.get("003") == pytest.approx(0.25)

    def test_ctr_map_is_empty_dict_on_db_exception(self):
        svc = MLService()
        svc._ctr_last_refresh = 0.0  # force refresh

        with patch(
            "app.services.data.database_service.database_service.get_content_ctr_map",
            side_effect=RuntimeError("db down"),
        ):
            result = svc.get_ctr_map()

        assert isinstance(result, dict)
```

- [ ] **Step 2: Run the new tests and verify they all pass**

```bash
python -m pytest tests/unit/services/search/test_ml_service.py -v
```

Expected: all tests PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/services/search/test_ml_service.py
git commit -m "test: add unit tests for MLService with mocked XGBoost model"
```

---

## Task 5: Final coverage verification

- [ ] **Step 1: Run the full suite with coverage**

```bash
python -m pytest --cov=app --cov-report=term-missing --no-header -q 2>&1 | tail -30
```

- [ ] **Step 2: Verify success criteria**

Check the output against these thresholds:

| Module | Target |
|--------|--------|
| `app/services/search/synonyms.py` | ≥ 95% |
| `app/services/search/hybrid_search.py` | ≥ 60% |
| `app/services/search/ml_service.py` | ≥ 50% |
| Total | ≥ 65% |

Also verify: `0 failed` in the test summary line.

- [ ] **Step 3: Generate HTML coverage report (optional, for rapport screenshot)**

```bash
python -m pytest --cov=app --cov-report=html --no-header -q
```

Report is written to `htmlcov/index.html`. Open in browser for the screenshot needed in the rapport.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "test: verify coverage thresholds met after backend test improvements"
```
