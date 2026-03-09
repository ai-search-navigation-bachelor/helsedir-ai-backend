"""
Unit tests for keyword search service.

Tests both the pure tokenization/scoring helpers and the full search pipeline.
The mock_content fixture is required for tests that call KeywordSearch.search()
because that method iterates content_service.get_all_content().
"""

import pytest
from typing import List, Optional
from app.entities.content import ContentItem
from app.services.search.keyword_search import (
    KeywordSearch,
    _tokenize,
    _expand_query,
    STOP_WORDS,
)


def _item(
    title: str,
    content_type: str = "retningslinje",
    id: str = "test-x",
    target_groups: Optional[List[str]] = None,
) -> ContentItem:
    return ContentItem(
        id=id,
        title=title,
        body="",
        content_type=content_type,
        target_groups=target_groups or [],
    )


@pytest.mark.unit
class TestTokenize:
    def test_removes_stop_words(self):
        tokens = _tokenize("og i er en et diabetes")
        assert "og" not in tokens
        assert "i" not in tokens
        assert "er" not in tokens

    def test_empty_string_returns_empty_set(self):
        assert _tokenize("") == set()

    def test_lowercases_input(self):
        tokens = _tokenize("Diabetes BEHANDLING")
        assert all(t == t.lower() for t in tokens)

    def test_applies_snowball_stemming(self):
        stem_a = _tokenize("behandling")
        stem_b = _tokenize("behandlinger")
        assert stem_a == stem_b

    def test_punctuation_stripped(self):
        tokens = _tokenize("diabetes, behandling.")
        assert all(t.isalpha() or t.isdigit() or "_" in t for t in tokens)

    def test_known_stop_word_subset(self):
        for word in ["og", "i", "er", "til", "for"]:
            assert word in STOP_WORDS


@pytest.mark.unit
class TestExpandQuery:
    def test_synonym_expansion_diabetes(self):
        stemmed = _tokenize("diabetes")
        expanded = _expand_query("diabetes", stemmed)
        sukkersyke_stem = _tokenize("sukkersyke")
        assert sukkersyke_stem.issubset(expanded)

    def test_synonym_expansion_hjerteinfarkt(self):
        stemmed = _tokenize("hjerteinfarkt")
        expanded = _expand_query("hjerteinfarkt", stemmed)
        hjerteanfall_stem = _tokenize("hjerteanfall")
        assert hjerteanfall_stem.issubset(expanded)

    def test_no_expansion_for_unknown_word(self):
        stemmed = _tokenize("ukjentord")
        expanded = _expand_query("ukjentord", stemmed)
        assert expanded == stemmed


@pytest.mark.unit
class TestKeywordSearchScoring:
    """Tests for KeywordSearch.calculate_score_with_breakdown — no DB needed."""

    def setup_method(self):
        self.ks = KeywordSearch()

    def test_exact_phrase_in_title_adds_score(self):
        item = _item("Diabetes retningslinje")
        query_lower = "diabetes retningslinje"
        query_kws = _tokenize(query_lower)
        score, breakdown = self.ks.calculate_score_with_breakdown(
            item, query_lower, query_kws
        )
        assert "exact_title" in breakdown
        assert score > 0

    def test_keyword_match_in_title_adds_score(self):
        # Use _expand_query to mirror how search() prepares query_keywords
        item = _item("Diabetes behandling")
        query_lower = "diabetes"
        stemmed = _tokenize(query_lower)
        query_kws = _expand_query(query_lower, stemmed)
        score, breakdown = self.ks.calculate_score_with_breakdown(
            item, query_lower, query_kws
        )
        assert score > 0
        # Either exact_title or title_keywords (or both) should be present
        assert "exact_title" in breakdown or "title_keywords" in breakdown

    def test_no_match_returns_zero(self):
        item = _item("Allergi behandling")
        query_lower = "diabetes"
        query_kws = _tokenize(query_lower)
        score, _ = self.ks.calculate_score_with_breakdown(item, query_lower, query_kws)
        assert score == 0.0

    def test_full_title_coverage_bonus_present(self):
        """
        Full title coverage fires when all title stems are in the query.

        Pass raw words (not pre-stemmed) to calculate_score_with_breakdown,
        as _normalize_query_keywords is designed to handle raw text input.
        """
        item = _item("Diabetes")  # single-word title fully covered by query
        query_lower = "diabetes behandling"
        # Pass raw words — _normalize_query_keywords will stem them correctly
        query_kws = {"diabetes", "behandling"}
        _, breakdown = self.ks.calculate_score_with_breakdown(
            item, query_lower, query_kws
        )
        assert "full_title_coverage" in breakdown

    def test_calculate_score_matches_calculate_score_with_breakdown(self):
        item = _item("Diabetes retningslinje")
        query_lower = "diabetes"
        query_kws = _tokenize(query_lower)
        score_only = self.ks.calculate_score(item, query_lower, query_kws)
        score_with_bd, _ = self.ks.calculate_score_with_breakdown(
            item, query_lower, query_kws
        )
        assert score_only == score_with_bd


@pytest.mark.unit
class TestKeywordSearchPipeline:
    """Tests for KeywordSearch.search() — requires mock_content fixture."""

    def test_search_returns_normalized_scores(self, mock_content):
        ks = KeywordSearch()
        results = ks.search("diabetes", k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range for {r.id}"

    def test_search_returns_list_of_search_results(self, mock_content):
        from app.dto.response.search import SearchResult
        ks = KeywordSearch()
        results = ks.search("diabetes")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_search_no_match_returns_empty(self, mock_content):
        ks = KeywordSearch()
        results = ks.search("xyznomatch99999")
        assert results == []

    def test_search_limits_results_to_k(self, mock_content):
        ks = KeywordSearch()
        # Add more items so we can test limit
        from app.services.data.content_service import content_service
        for i in range(10):
            extra = ContentItem(
                id=f"extra-{i}",
                title=f"Diabetes veileder {i}",
                body="",
                content_type="veileder",
            )
            content_service.content.append(extra)
            content_service.content_by_id[extra.id] = extra
            content_service.searchable_types.add("veileder")
        results = ks.search("diabetes", k=3)
        assert len(results) <= 3

    def test_role_filter_excludes_items_without_role(self, mock_content):
        """
        When a role is specified, items without that role in target_groups
        are excluded. SAMPLE_CONTENT items have empty target_groups, so
        a role search returns no results unless items explicitly include the role.
        """
        from app.services.data.content_service import content_service
        restricted = ContentItem(
            id="role-001",
            title="Diabetes for lege",
            body="",
            content_type="retningslinje",
            role_tags=["lege"],
        )
        content_service.content.append(restricted)
        content_service.content_by_id["role-001"] = restricted

        ks = KeywordSearch()
        results = ks.search("diabetes", role="lege", k=10)
        result_ids = [r.id for r in results]
        assert "role-001" in result_ids

    def test_role_filter_excludes_wrong_role(self, mock_content):
        from app.services.data.content_service import content_service
        restricted = ContentItem(
            id="role-002",
            title="Diabetes for lege guide",
            body="",
            content_type="retningslinje",
            role_tags=["lege"],
        )
        content_service.content.append(restricted)
        content_service.content_by_id["role-002"] = restricted

        ks = KeywordSearch()
        results = ks.search("diabetes lege guide", role="sykepleier", k=10)
        result_ids = [r.id for r in results]
        assert "role-002" not in result_ids

    def test_search_results_have_explanation(self, mock_content):
        ks = KeywordSearch()
        results = ks.search("diabetes")
        for r in results:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 0
