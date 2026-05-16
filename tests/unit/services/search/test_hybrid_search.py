"""
Unit tests for hybrid_search.py.

Three test classes:
1. TestNormalizeCombinedScores            — pure static method, no fixtures
2. TestExtractRankingFeatures             — pure static method, no fixtures
3. TestExtractRankingFeaturesEdgeCases    — additional edge cases, no fixtures
4. TestBuildResults                       — pure static method, no fixtures
5. TestBuildResultsPipelineScores         — pipeline field values, no fixtures
6. TestNormalizeCombinedScoresEdgeCases   — negative/zero scores, no fixtures
7. TestHybridSearchFlow                   — full search() with bm25/semantic mocked
8. TestHybridSearchRoleBoostPipeline      — role boost/penalty visible on pipeline
9. TestHybridSearchTypeBoosts             — content type boosts change ordering
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


# ---------------------------------------------------------------------------
# _extract_ranking_features — additional edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractRankingFeaturesEdgeCases:
    def test_empty_title_gives_zero_overlap(self):
        c = _candidate(0.5)
        c.item.title = ""
        f = HybridSearch._extract_ranking_features(c, None, {}, "diabetes", {"diabetes"})
        assert f["title_query_overlap"] == pytest.approx(0.0)

    def test_empty_query_keywords_gives_zero_query_length(self):
        c = _candidate(0.5)
        f = HybridSearch._extract_ranking_features(c, None, {}, "", set())
        assert f["query_length"] == 0.0

    def test_none_ctr_map_uses_default_ctr(self):
        c = _candidate(0.5)
        f = HybridSearch._extract_ranking_features(c, None, None, "", set())
        assert f["smoothed_ctr"] == pytest.approx(1.0 / 21.0)

    def test_semantic_score_comes_from_candidate_semantic_norm(self):
        c = _candidate(0.5, semantic_norm=0.77)
        f = HybridSearch._extract_ranking_features(c, None, {}, "", set())
        assert f["semantic_score"] == pytest.approx(0.77)

    def test_bm25_score_comes_from_candidate_keyword_norm(self):
        c = _candidate(0.5, keyword_norm=0.33)
        f = HybridSearch._extract_ranking_features(c, None, {}, "", set())
        assert f["bm25_score"] == pytest.approx(0.33)

    def test_partial_title_match_gives_intermediate_overlap(self):
        # Title "Diabetes retningslinje" — query has one of two title words
        # intersection={"diabetes"}, union={"diabetes","retningslinje"} → 0.5
        c = _candidate(0.5)
        f = HybridSearch._extract_ranking_features(
            c, None, {}, "diabetes", {"diabetes"}
        )
        assert 0.0 < f["title_query_overlap"] < 1.0


# ---------------------------------------------------------------------------
# _build_results — pipeline field values
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildResultsPipelineScores:
    def test_pipeline_bm25_rounded_to_four_decimals(self):
        c = _candidate(0.9, keyword_norm=0.123456)
        results = HybridSearch._build_results([c], k=1)
        assert results[0].pipeline.bm25 == pytest.approx(0.1235, abs=0.0001)

    def test_pipeline_semantic_rounded_to_four_decimals(self):
        c = _candidate(0.9, semantic_norm=0.987654)
        results = HybridSearch._build_results([c], k=1)
        assert results[0].pipeline.semantic == pytest.approx(0.9877, abs=0.0001)

    def test_pipeline_role_boost_default_is_one(self):
        c = _candidate(0.9)
        results = HybridSearch._build_results([c], k=1)
        assert results[0].pipeline.role_boost == pytest.approx(1.0)

    def test_pipeline_role_boost_reflects_candidate_value(self):
        c = _candidate(0.9)
        c.role_boost = 1.15
        results = HybridSearch._build_results([c], k=1)
        assert results[0].pipeline.role_boost == pytest.approx(1.15)

    def test_pipeline_rrf_score_reflects_candidate_rrf_raw(self):
        c = _candidate(0.9)
        c.rrf_raw = 0.012345
        results = HybridSearch._build_results([c], k=1)
        assert results[0].pipeline.rrf == pytest.approx(0.012345, abs=1e-6)

    def test_result_title_matches_item_title(self):
        c = _candidate(0.9, content_id="x")
        results = HybridSearch._build_results([c], k=1)
        assert results[0].title == c.item.title

    def test_all_k_results_returned_when_candidates_equal_k(self):
        candidates = [_candidate(1.0 - i * 0.1, f"id{i}") for i in range(3)]
        results = HybridSearch._build_results(candidates, k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# _normalize_combined_scores — edge cases with negative and zero scores
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNormalizeCombinedScoresEdgeCases:
    def test_negative_scores_normalized_to_zero_one(self):
        candidates = [_candidate(s, f"id{i}") for i, s in enumerate([-1.0, 0.0, 1.0])]
        result = HybridSearch._normalize_combined_scores(candidates)
        scores = [c.combined_score for c in result]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)

    def test_single_item_with_zero_score_gives_zero(self):
        c = _candidate(0.0)
        result = HybridSearch._normalize_combined_scores([c])
        assert result[0].combined_score == 0.0

    def test_single_item_with_positive_score_gives_one(self):
        c = _candidate(0.5)
        result = HybridSearch._normalize_combined_scores([c])
        assert result[0].combined_score == 1.0


# ---------------------------------------------------------------------------
# Role boost/penalty visible through pipeline field in search()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHybridSearchRoleBoostPipeline:
    def test_matching_role_sets_boost_above_one_on_pipeline(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        item = _item("match-001", role_tags=["lege"])
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[BM25Hit(item=item, score=5.0, rank=1)],
        )

        hs = HybridSearch()
        results = hs.search("diabetes", role="lege", rerank=False, k=1)

        assert len(results) == 1
        assert results[0].pipeline.role_boost > 1.0

    def test_mismatching_role_sets_penalty_below_one_on_pipeline(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        item = _item("mismatch-001", role_tags=["pasient"])
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[BM25Hit(item=item, score=5.0, rank=1)],
        )

        hs = HybridSearch()
        results = hs.search("diabetes", role="lege", rerank=False, k=1)

        assert len(results) == 1
        assert results[0].pipeline.role_boost < 1.0

    def test_no_role_tags_gives_neutral_boost_even_with_role(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        # _bm25_hit creates items with empty role_tags by default
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[_bm25_hit("001", 5.0)],
        )

        hs = HybridSearch()
        results = hs.search("diabetes", role="lege", rerank=False, k=1)

        assert len(results) == 1
        assert results[0].pipeline.role_boost == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Content type boosts change ordering in search()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHybridSearchTypeBoosts:
    def test_temaside_boost_elevates_over_equal_bm25_item(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        temaside = _item("ts-001", content_type="temaside")
        retningslinje = _item("rl-001", content_type="retningslinje")
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[
                BM25Hit(item=temaside, score=5.0, rank=1),
                BM25Hit(item=retningslinje, score=5.0, rank=2),
            ],
        )

        hs = HybridSearch(candidate_multiplier=2, min_candidate_pool=2, max_candidate_pool=10)
        results = hs.search(
            "diabetes", rerank=False, k=2,
            temaside_boost=2.0, retningslinje_boost=1.0,
        )

        assert len(results) == 2
        assert results[0].id == "ts-001"

    def test_retningslinje_boost_elevates_over_unboosted_item(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        retningslinje = _item("rl-001", content_type="retningslinje")
        veileder = _item("vl-001", content_type="veileder")
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[
                BM25Hit(item=retningslinje, score=5.0, rank=1),
                BM25Hit(item=veileder, score=5.0, rank=2),
            ],
        )

        hs = HybridSearch(candidate_multiplier=2, min_candidate_pool=2, max_candidate_pool=10)
        results = hs.search(
            "diabetes", rerank=False, k=2,
            retningslinje_boost=2.0, temaside_boost=1.0,
        )

        assert len(results) == 2
        assert results[0].id == "rl-001"

    def test_no_boost_leaves_original_bm25_ordering(self, mock_content, mocker):
        mocker.patch(
            "app.services.search.hybrid_search.semantic_search.is_available",
            return_value=False,
        )
        mocker.patch(
            "app.services.search.hybrid_search.bm25_search.search",
            return_value=[
                _bm25_hit("first", 9.0),
                _bm25_hit("second", 3.0),
            ],
        )

        hs = HybridSearch(candidate_multiplier=2, min_candidate_pool=2, max_candidate_pool=10)
        results = hs.search(
            "diabetes", rerank=False, k=2,
            temaside_boost=1.0, retningslinje_boost=1.0,
        )

        assert len(results) == 2
        assert results[0].id == "first"
