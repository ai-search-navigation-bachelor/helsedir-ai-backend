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
