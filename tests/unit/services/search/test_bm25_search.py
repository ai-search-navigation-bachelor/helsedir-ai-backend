"""
Unit tests for BM25 search service.

Tests use the mock_content fixture to inject synthetic content into
content_service, then call _build_index() to force index rebuild.
"""

import pytest
from app.entities.content import ContentItem
from app.services.search.bm25_search import BM25Search, BM25Hit


@pytest.mark.unit
class TestBM25Tokenize:
    def test_lowercases_text(self):
        tokens = BM25Search._tokenize("Diabetes BEHANDLING")
        assert "diabetes" in tokens
        assert "behandling" in tokens

    def test_empty_string_returns_empty_list(self):
        assert BM25Search._tokenize("") == []

    def test_none_like_input_returns_empty(self):
        assert BM25Search._tokenize("") == []

    def test_splits_on_word_boundaries(self):
        tokens = BM25Search._tokenize("hjerte-infarkt, diabetes.")
        assert "hjerte" in tokens or "hjerteinfarkt" in tokens or "infarkt" in tokens
        assert "diabetes" in tokens


@pytest.mark.unit
class TestBM25Index:
    def test_empty_content_builds_empty_index(self, mock_content):
        from app.services.data.content_service import content_service
        content_service.content = []
        bm25 = BM25Search()
        bm25._build_index()
        assert bm25._items == []
        assert bm25._idf == {}

    def test_prebuild_returns_item_count(self, mock_content):
        bm25 = BM25Search()
        count = bm25.prebuild()
        assert count == len(mock_content.content)

    def test_needs_rebuild_after_fresh_init(self, mock_content):
        """A new BM25Search instance should need a rebuild (no index yet)."""
        bm25 = BM25Search()
        assert bm25._needs_rebuild()

    def test_no_rebuild_needed_after_build(self, mock_content):
        bm25 = BM25Search()
        bm25._build_index()
        assert not bm25._needs_rebuild()

    def test_needs_rebuild_after_content_change(self, mock_content):
        from app.services.data.content_service import content_service
        bm25 = BM25Search()
        bm25._build_index()
        new_item = ContentItem(
            id="new-rebuild",
            title="Ny innhold",
            body="",
            content_type="veileder",
        )
        content_service.content.append(new_item)
        assert bm25._needs_rebuild()


@pytest.mark.unit
class TestBM25Search:
    def test_search_empty_content_returns_empty(self, mock_content):
        from app.services.data.content_service import content_service
        content_service.content = []
        bm25 = BM25Search()
        results = bm25.search("diabetes")
        assert results == []

    def test_search_matching_title_returns_hit(self, mock_content):
        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes")
        result_ids = [r.item.id for r in results]
        assert "001" in result_ids

    def test_search_non_matching_query_returns_empty(self, mock_content):
        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("xyznomatch99999")
        assert results == []

    def test_search_results_ranked_descending_by_score(self, mock_content):
        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes retningslinje")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_bm25hit_instances(self, mock_content):
        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes")
        for r in results:
            assert isinstance(r, BM25Hit)

    def test_search_rank_field_is_sequential(self, mock_content):
        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes")
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_search_limits_results_with_k(self, mock_content):
        from app.services.data.content_service import content_service
        # Add extra items that all match "veileder"
        for i in range(10):
            item = ContentItem(
                id=f"bm25-extra-{i}",
                title=f"Diabetes veileder {i}",
                body=f"Tekst {i}.",
                content_type="veileder",
            )
            content_service.content.append(item)
            content_service.content_by_id[item.id] = item

        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes", k=3)
        assert len(results) <= 3

    def test_role_filter_includes_matching_role(self, mock_content):
        from app.services.data.content_service import content_service
        role_item = ContentItem(
            id="bm25-role-001",
            title="Diabetes for lege",
            body="",
            content_type="retningslinje",
            target_groups=["lege"],
        )
        content_service.content.append(role_item)
        content_service.content_by_id["bm25-role-001"] = role_item

        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes lege", role="lege", k=20)
        result_ids = [r.item.id for r in results]
        assert "bm25-role-001" in result_ids

    def test_role_filter_excludes_non_matching_role(self, mock_content):
        from app.services.data.content_service import content_service
        role_item = ContentItem(
            id="bm25-role-002",
            title="Diabetes lege spesial",
            body="",
            content_type="retningslinje",
            target_groups=["lege"],
        )
        content_service.content.append(role_item)
        content_service.content_by_id["bm25-role-002"] = role_item

        bm25 = BM25Search()
        bm25._build_index()
        results = bm25.search("diabetes lege spesial", role="sykepleier", k=20)
        result_ids = [r.item.id for r in results]
        assert "bm25-role-002" not in result_ids

    def test_compute_signature_is_stable(self, mock_content):
        from app.services.data.content_service import content_service
        sig1 = BM25Search._compute_signature(content_service.content)
        sig2 = BM25Search._compute_signature(content_service.content)
        assert sig1 == sig2

    def test_compute_signature_changes_when_content_changes(self, mock_content):
        from app.services.data.content_service import content_service
        sig_before = BM25Search._compute_signature(content_service.content)
        new_item = ContentItem(
            id="sig-change-001",
            title="Ny innhold",
            body="",
            content_type="veileder",
        )
        content_service.content.append(new_item)
        sig_after = BM25Search._compute_signature(content_service.content)
        assert sig_before != sig_after
