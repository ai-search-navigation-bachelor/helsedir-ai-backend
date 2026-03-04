"""
Unit tests for RRF (Reciprocal Rank Fusion) utility.

fuse_ranked_lists is a pure function with zero external dependencies —
no fixtures or mocking required.
"""

import pytest
from app.services.search.rrf_fusion import fuse_ranked_lists, RRFResult


@pytest.mark.unit
class TestFuseRankedLists:
    def test_empty_input_returns_empty_list(self):
        result = fuse_ranked_lists({})
        assert result == []

    def test_single_source_single_item(self):
        result = fuse_ranked_lists({"bm25": ["a"]})
        assert len(result) == 1
        assert result[0].content_id == "a"

    def test_single_source_preserves_order(self):
        result = fuse_ranked_lists({"bm25": ["a", "b", "c"]}, rrf_k=60)
        ids = [r.content_id for r in result]
        assert ids == ["a", "b", "c"]

    def test_scores_decrease_with_rank(self):
        result = fuse_ranked_lists({"bm25": ["first", "second", "third"]})
        assert result[0].score > result[1].score > result[2].score

    def test_item_in_both_lists_scores_higher_than_item_in_one(self):
        result = fuse_ranked_lists(
            {"bm25": ["shared", "only_bm25"], "semantic": ["shared", "only_semantic"]},
        )
        scores = {r.content_id: r.score for r in result}
        assert scores["shared"] > scores["only_bm25"]
        assert scores["shared"] > scores["only_semantic"]

    def test_two_symmetric_lists_give_equal_scores(self):
        result = fuse_ranked_lists(
            {"bm25": ["a", "b"], "semantic": ["b", "a"]},
            rrf_k=60,
        )
        scores = {r.content_id: r.score for r in result}
        assert abs(scores["a"] - scores["b"]) < 1e-9

    def test_weights_applied_correctly(self):
        result = fuse_ranked_lists(
            {"bm25": ["a"], "semantic": ["b"]},
            rrf_k=60,
            weights={"bm25": 0.3, "semantic": 0.7},
        )
        scores = {r.content_id: r.score for r in result}
        # semantic-only item should beat bm25-only item
        assert scores["b"] > scores["a"]

    def test_equal_weights_produce_equal_single_item_scores(self):
        result = fuse_ranked_lists(
            {"bm25": ["a"], "semantic": ["b"]},
            rrf_k=60,
            weights={"bm25": 1.0, "semantic": 1.0},
        )
        scores = {r.content_id: r.score for r in result}
        assert abs(scores["a"] - scores["b"]) < 1e-9

    def test_rrf_k_zero_clamped_to_one(self):
        """rrf_k=0 must not cause ZeroDivisionError."""
        result = fuse_ranked_lists({"bm25": ["a", "b"]}, rrf_k=0)
        assert len(result) == 2

    def test_empty_content_id_is_skipped(self):
        result = fuse_ranked_lists({"bm25": ["", "a", ""]})
        assert len(result) == 1
        assert result[0].content_id == "a"

    def test_result_sorted_descending_by_score(self):
        result = fuse_ranked_lists(
            {"bm25": ["a", "b", "c"], "semantic": ["a", "c", "b"]},
        )
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_recorded_per_source(self):
        result = fuse_ranked_lists({"bm25": ["x", "y"], "semantic": ["y", "x"]})
        ranks_by_id = {r.content_id: r.ranks for r in result}
        assert ranks_by_id["x"]["bm25"] == 1
        assert ranks_by_id["x"]["semantic"] == 2
        assert ranks_by_id["y"]["bm25"] == 2
        assert ranks_by_id["y"]["semantic"] == 1

    def test_source_without_explicit_weight_defaults_to_one(self):
        result_no_weights = fuse_ranked_lists({"bm25": ["a"]}, rrf_k=60)
        result_weight_one = fuse_ranked_lists(
            {"bm25": ["a"]}, rrf_k=60, weights={"bm25": 1.0}
        )
        assert abs(result_no_weights[0].score - result_weight_one[0].score) < 1e-9

    def test_missing_weight_for_source_defaults_to_one(self):
        """Source not in weights dict still uses 1.0."""
        result = fuse_ranked_lists(
            {"bm25": ["a"], "semantic": ["b"]},
            weights={"bm25": 1.0},  # semantic not specified
        )
        scores = {r.content_id: r.score for r in result}
        # Both have weight 1.0 (semantic defaults), so equal for equal rank
        assert abs(scores["a"] - scores["b"]) < 1e-9

    def test_large_rrf_k_reduces_score_differences(self):
        """Higher k → smaller score difference between ranks."""
        result_low_k = fuse_ranked_lists({"bm25": ["a", "b"]}, rrf_k=1)
        result_high_k = fuse_ranked_lists({"bm25": ["a", "b"]}, rrf_k=1000)
        diff_low = result_low_k[0].score - result_low_k[1].score
        diff_high = result_high_k[0].score - result_high_k[1].score
        assert diff_low > diff_high
