"""
Unit tests for SearchController.

Three test classes:
1. TestSearchControllerHelpers — pure static/class methods, no I/O
2. TestSearchControllerCaching — _execute_search caching behavior
3. TestSearchControllerSearchAsync — async search() and get_suggestions()
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.controllers.search_controller import SearchController
from app.dto.response.search import SearchResult, SearchResponse, GroupedContent
from app.entities.content import ContentItem, ContentLink


def _make_result(result_id: str, score: float, info_type: str = "retningslinje") -> SearchResult:
    return SearchResult(
        id=result_id,
        title=f"Title {result_id}",
        short_title=f"Short {result_id}",
        display_title=f"Short {result_id}",
        info_type=info_type,
        has_text_content=True,
        document_url=None,
        is_pdf_only=False,
        score=score,
        explanation="test",
    )


@pytest.mark.unit
class TestSearchControllerHelpers:
    """Tests for pure static/class methods — no fixtures or mocking needed."""

    # ── _compute_role_match ──────────────────────────────────────────────────

    def test_role_match_exact_single_group(self):
        score = SearchController._compute_role_match("lege", ["lege"])
        assert score == pytest.approx(1.0)

    def test_role_match_exact_two_groups(self):
        score = SearchController._compute_role_match("lege", ["lege", "sykepleier"])
        assert score == pytest.approx(0.5)

    def test_role_match_no_role_no_groups(self):
        assert SearchController._compute_role_match(None, []) == 0.5

    def test_role_match_no_role_with_groups(self):
        assert SearchController._compute_role_match(None, ["lege"]) == 0.0

    def test_role_match_role_not_in_groups(self):
        assert SearchController._compute_role_match("lege", ["sykepleier"]) == 0.0

    def test_role_match_role_with_no_groups(self):
        assert SearchController._compute_role_match("lege", []) == 0.3

    def test_role_match_role_with_none_groups(self):
        assert SearchController._compute_role_match("lege", None) == 0.3

    # ── _needs_theme_fuzzy_fallback ──────────────────────────────────────────

    def test_needs_fallback_no_theme_results(self):
        results = [_make_result("1", 0.9, "retningslinje")]
        ctrl = SearchController()
        assert ctrl._needs_theme_fuzzy_fallback(results) is True

    def test_needs_fallback_empty_results(self):
        ctrl = SearchController()
        assert ctrl._needs_theme_fuzzy_fallback([]) is True

    def test_no_fallback_when_good_theme_score(self):
        from app.config import settings
        high_score = settings.search_min_score + 0.1
        results = [_make_result("1", high_score, "temaside")]
        ctrl = SearchController()
        assert ctrl._needs_theme_fuzzy_fallback(results) is False

    def test_needs_fallback_when_theme_score_below_min(self):
        from app.config import settings
        low_score = settings.search_min_score - 0.01
        results = [_make_result("1", low_score, "temaside")]
        ctrl = SearchController()
        assert ctrl._needs_theme_fuzzy_fallback(results) is True

    # ── _build_search_signature / _build_signed_search_id / _extract ────────

    def test_build_signature_is_deterministic(self):
        sig1 = SearchController._build_search_signature(
            "hybrid", None, None, None, None, None
        )
        sig2 = SearchController._build_search_signature(
            "hybrid", None, None, None, None, None
        )
        assert sig1 == sig2

    def test_build_signature_differs_for_different_methods(self):
        sig_hybrid = SearchController._build_search_signature(
            "hybrid", None, None, None, None, None
        )
        sig_keyword = SearchController._build_search_signature(
            "keyword", None, None, None, None, None
        )
        assert sig_hybrid != sig_keyword

    def test_signed_search_id_round_trip(self):
        sig = SearchController._build_search_signature(
            "hybrid", None, None, None, None, None
        )
        search_id = SearchController._build_signed_search_id(sig)
        extracted = SearchController._extract_search_signature(search_id)
        assert extracted == sig

    def test_extract_signature_from_invalid_uuid_returns_none(self):
        result = SearchController._extract_search_signature("not-a-uuid")
        assert result is None

    def test_extract_signature_from_unsigned_uuid_returns_none(self):
        import uuid
        # A plain uuid4() will not have the "c0de" marker in the last 12 hex chars
        # with near-certainty. We force a UUID that provably lacks the marker.
        unsigned_id = "12345678-1234-1234-1234-123456789abc"
        result = SearchController._extract_search_signature(unsigned_id)
        assert result is None

    # ── _merge_theme_fallback_results ────────────────────────────────────────

    def test_merge_fallback_does_not_override_existing_result(self):
        ctrl = SearchController()
        regular = [_make_result("shared", 0.9)]
        fallback = [_make_result("shared", 0.5), _make_result("new", 0.75)]
        merged = ctrl._merge_theme_fallback_results(regular, fallback)
        shared = next(r for r in merged if r.id == "shared")
        assert shared.score == 0.9  # original preserved

    def test_merge_fallback_adds_new_results(self):
        ctrl = SearchController()
        regular = [_make_result("existing", 0.8)]
        fallback = [_make_result("new", 0.6)]
        merged = ctrl._merge_theme_fallback_results(regular, fallback)
        ids = [r.id for r in merged]
        assert "new" in ids

    def test_merge_fallback_sorted_by_score_descending(self):
        ctrl = SearchController()
        regular = [_make_result("a", 0.5)]
        fallback = [_make_result("b", 0.9)]
        merged = ctrl._merge_theme_fallback_results(regular, fallback)
        assert merged[0].score >= merged[1].score

    def test_merge_fallback_empty_fallback_returns_original(self):
        ctrl = SearchController()
        regular = [_make_result("a", 0.9)]
        merged = ctrl._merge_theme_fallback_results(regular, [])
        assert merged == regular

    def test_merge_fallback_respects_max_results(self):
        ctrl = SearchController()
        regular = [_make_result(f"r{i}", 0.9 - i * 0.1) for i in range(3)]
        fallback = [_make_result(f"f{i}", 0.5 - i * 0.1) for i in range(3)]
        merged = ctrl._merge_theme_fallback_results(regular, fallback, max_results=4)
        assert len(merged) <= 4

    def test_filter_empty_theme_page_results_removes_empty_theme_pages(self, mocker):
        ctrl = SearchController()
        empty_theme = _make_result("empty-theme", 0.8, "temaside")
        empty_theme.has_text_content = False
        filled_theme = _make_result("filled-theme", 0.7, "temaside")
        filled_theme.has_text_content = False
        results = [
            empty_theme,
            filled_theme,
            _make_result("guide", 0.6, "veileder"),
        ]
        mocker.patch.object(
            ctrl,
            "_get_non_empty_theme_page_ids",
            return_value={"filled-theme"},
        )

        filtered = ctrl._filter_empty_theme_page_results(results)

        assert [result.id for result in filtered] == ["filled-theme", "guide"]

    def test_filter_empty_theme_page_results_keeps_theme_pages_with_text_content(self, mocker):
        ctrl = SearchController()
        text_theme = _make_result("text-theme", 0.8, "temaside")
        text_theme.has_text_content = True
        mocker.patch.object(ctrl, "_get_non_empty_theme_page_ids", return_value=set())

        filtered = ctrl._filter_empty_theme_page_results([text_theme])

        assert [result.id for result in filtered] == ["text-theme"]


@pytest.mark.unit
class TestSearchControllerCaching:
    """Tests for _execute_search caching behavior."""

    def test_cache_hit_on_second_identical_call(self, mocker):
        ctrl = SearchController()
        mock_search_svc = MagicMock()
        mock_search_svc.search.return_value = [_make_result("001", 0.9)]
        ctrl.search_service = mock_search_svc

        mocker.patch(
            "app.controllers.search_controller.content_service.get_all_content",
            return_value=[],
        )

        ctrl._execute_search("diabetes", None, "keyword", 10)
        ctrl._execute_search("diabetes", None, "keyword", 10)

        assert mock_search_svc.search.call_count == 1

    def test_cache_miss_on_different_query(self, mocker):
        ctrl = SearchController()
        mock_search_svc = MagicMock()
        mock_search_svc.search.return_value = []
        ctrl.search_service = mock_search_svc

        mocker.patch(
            "app.controllers.search_controller.content_service.get_all_content",
            return_value=[],
        )

        ctrl._execute_search("diabetes", None, "keyword", 10)
        ctrl._execute_search("adhd", None, "keyword", 10)

        assert mock_search_svc.search.call_count == 2

    def test_cache_miss_on_different_role(self, mocker):
        ctrl = SearchController()
        mock_search_svc = MagicMock()
        mock_search_svc.search.return_value = []
        ctrl.search_service = mock_search_svc

        mocker.patch(
            "app.controllers.search_controller.content_service.get_all_content",
            return_value=[],
        )

        ctrl._execute_search("diabetes", None, "keyword", 10)
        ctrl._execute_search("diabetes", "lege", "keyword", 10)

        assert mock_search_svc.search.call_count == 2


@pytest.mark.unit
class TestSearchControllerSearchAsync:
    """Tests for async search() and synchronous get_suggestions()."""

    async def test_search_invalid_method_raises_value_error(self):
        ctrl = SearchController()
        with pytest.raises(ValueError, match="Invalid search method"):
            await ctrl.search(query="diabetes", method="unknown_method")

    async def test_search_returns_search_response(self, mock_content, mock_database_service, mocker):
        ctrl = SearchController()
        mocker.patch.object(
            ctrl,
            "_execute_search",
            return_value=[_make_result("001", 0.9), _make_result("002", 0.6)],
        )
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search(query="diabetes", method="keyword", log=False)

        assert isinstance(response, SearchResponse)
        assert response.query == "diabetes"
        assert isinstance(response.search_id, str)
        assert response.total >= 0

    async def test_search_filters_results_below_min_score(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        mocker.patch.object(
            ctrl,
            "_execute_search",
            return_value=[
                _make_result("above", settings.search_min_score + 0.1),
                _make_result("below", max(0.0, settings.search_min_score - 0.1)),
            ],
        )
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search(query="test", log=False)
        result_ids = [r.id for r in response.results]
        assert "above" in result_ids
        # "below" is only excluded if its score is actually below the threshold
        # (score 0.0 is always below)
        if settings.search_min_score > 0:
            assert "below" not in result_ids

    async def test_search_pagination_offsets_correctly(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        results = [_make_result(f"item-{i}", settings.search_min_score + 0.01) for i in range(5)]
        ctrl = SearchController()
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search(query="test", offset=2, limit=2, log=False)
        assert len(response.results) <= 2
        assert response.offset == 2
        assert response.has_prev is True

    async def test_search_includes_parent_and_root_publication(
        self, mock_content, mock_database_service, mocker
    ):
        from app.services.data.content_service import content_service

        recommendation = ContentItem(
            id="100",
            title="ADHD-anbefaling",
            body="",
            content_type="anbefaling",
            path="/anbefalinger/adhd",
            has_text_content=True,
            links=[
                ContentLink(rel="forelder", type="kapittel", id="500"),
                ContentLink(rel="root", type="retningslinje", id="600"),
            ],
        )
        parent = ContentItem(
            id="500",
            title="Kapittel om ADHD",
            body="",
            content_type="kapittel",
            path="/kapitler/adhd",
        )
        root = ContentItem(
            id="600",
            title="Nasjonal faglig retningslinje for ADHD",
            body="",
            content_type="retningslinje",
            path="/retningslinjer/adhd",
        )
        content_service.content = [recommendation, parent, root]
        content_service.content_by_id = {item.id: item for item in content_service.content}
        content_service.content_by_path = {
            item.path: item for item in content_service.content if item.path
        }
        content_service.searchable_types = {item.content_type for item in content_service.content}

        ctrl = SearchController()
        mocker.patch.object(
            ctrl,
            "_execute_search",
            return_value=[_make_result("100", 0.9, "anbefaling")],
        )
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search(query="adhd", method="keyword", log=False)

        assert response.results[0].parent is not None
        assert response.results[0].parent.id == "500"
        assert response.results[0].root_publication is not None
        assert response.results[0].root_publication.id == "600"

    def test_populate_result_relations_updates_theme_page_children(self, mock_content):
        from app.services.data.content_service import content_service

        child = ContentItem(
            id="100",
            title="ADHD-anbefaling",
            body="",
            content_type="anbefaling",
            path="/anbefalinger/adhd",
            has_text_content=True,
            links=[ContentLink(rel="root", type="retningslinje", id="600")],
        )
        root = ContentItem(
            id="600",
            title="Nasjonal faglig retningslinje for ADHD",
            body="",
            content_type="retningslinje",
            path="/retningslinjer/adhd",
        )
        content_service.content = [child, root]
        content_service.content_by_id = {item.id: item for item in content_service.content}
        content_service.content_by_path = {
            item.path: item for item in content_service.content if item.path
        }
        content_service.searchable_types = {item.content_type for item in content_service.content}

        ctrl = SearchController()
        theme_result = SearchResult(
            id="004",
            title="ADHD temaside",
            short_title="ADHD",
            display_title="ADHD",
            info_type="temaside",
            has_text_content=False,
            document_url=None,
            is_pdf_only=False,
            score=0.9,
            children=[
                GroupedContent(
                    info_type="anbefaling",
                    display_name="Anbefalinger",
                    items=[
                        SearchResult(
                            id="100",
                            title="ADHD-anbefaling",
                            short_title="ADHD-anbefaling",
                            display_title="ADHD-anbefaling",
                            info_type="anbefaling",
                            has_text_content=True,
                            document_url=None,
                            is_pdf_only=False,
                            score=1.0,
                        )
                    ],
                )
            ],
        )

        ctrl._populate_result_relations([theme_result])

        assert theme_result.children is not None
        assert theme_result.children[0].items[0].root_publication is not None
        assert theme_result.children[0].items[0].root_publication.id == "600"

    # ── get_suggestions ──────────────────────────────────────────────────────

    def test_get_suggestions_empty_query_returns_empty(self, mock_content):
        ctrl = SearchController()
        response = ctrl.get_suggestions("")
        assert response.suggestions == []

    def test_get_suggestions_prefix_match(self, mock_content):
        ctrl = SearchController()
        with patch.object(ctrl, "_get_non_empty_theme_page_ids", return_value={"004"}):
            response = ctrl.get_suggestions("psykisk")
        ids = [s.id for s in response.suggestions]
        assert "004" in ids  # "Psykisk helse temaside"

    def test_get_suggestions_case_insensitive(self, mock_content):
        ctrl = SearchController()
        with patch.object(ctrl, "_get_non_empty_theme_page_ids", return_value={"004"}):
            response_lower = ctrl.get_suggestions("psykisk")
            response_upper = ctrl.get_suggestions("PSYKISK")
        assert {s.id for s in response_lower.suggestions} == {
            s.id for s in response_upper.suggestions
        }

    def test_get_suggestions_returns_temasider_and_retningslinjer(self, mock_content):
        ctrl = SearchController()
        # "diabetes" matches "Diabetes retningslinje" (retningslinje, not temaside)
        # suggestions now include both temasider and retningslinjer
        response = ctrl.get_suggestions("diabetes")
        allowed_types = {"temaside", "retningslinje"}
        for s in response.suggestions:
            item = mock_content.content_by_id.get(s.id)
            if item:
                assert item.content_type in allowed_types

    def test_get_suggestions_max_five_results(self, mock_content):
        from app.services.data.content_service import content_service
        from app.entities.content import ContentItem
        # Add extra temasider that all start with "test"
        for i in range(10):
            item = ContentItem(
                id=f"ts-{i}", title=f"Test temaside {i}", body="",
                content_type="temaside", path=f"/temasider/test-{i}"
            )
            content_service.content.append(item)
            content_service.content_by_id[item.id] = item

        ctrl = SearchController()
        response = ctrl.get_suggestions("test")
        assert len(response.suggestions) <= 5

    def test_get_suggestions_excludes_empty_theme_pages(self, mock_content, mocker):
        ctrl = SearchController()
        mocker.patch.object(
            ctrl,
            "_get_non_empty_theme_page_ids",
            return_value=set(),
        )

        response = ctrl.get_suggestions("psykisk")

        assert response.suggestions == []

    async def test_get_theme_pages_excludes_empty_theme_pages(self, mocker):
        ctrl = SearchController()
        mocker.patch(
            "app.controllers.search_controller.content_repository.get_theme_pages",
            return_value=[
                {"id": "empty", "tittel": "Tom temaside", "path": "/tema/tom", "has_text_content": 0},
                {"id": "filled", "tittel": "Fylt temaside", "path": "/tema/fylt", "has_text_content": 0},
            ],
        )
        mocker.patch.object(
            ctrl,
            "_get_non_empty_theme_page_ids",
            return_value={"filled"},
        )

        response = await ctrl.get_theme_pages()

        assert [result.id for result in response.results] == ["filled"]


@pytest.mark.unit
class TestSearchControllerSearchCategorized:
    """Tests for async search_categorized()."""

    async def test_returns_categorized_search_response(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings
        from app.dto.response.search import CategorizedSearchResponse

        ctrl = SearchController()
        results = [
            _make_result("001", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("002", settings.search_min_score + 0.05, "veileder"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="diabetes")

        assert isinstance(response, CategorizedSearchResponse)
        assert response.query == "diabetes"
        assert isinstance(response.search_id, str)
        assert response.total >= 0

    async def test_priority_categories_contain_retningslinje(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result("001", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("002", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("003", settings.search_min_score + 0.05, "veileder"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="diabetes")

        priority_categories = {cat.category for cat in response.priority_categories}
        assert "retningslinje" in priority_categories

    async def test_priority_category_contains_all_results(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result(f"r{i}", settings.search_min_score + 0.1, "retningslinje")
            for i in range(5)
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="diabetes")

        retningslinje_cat = next(
            (c for c in response.priority_categories if c.category == "retningslinje"), None
        )
        assert retningslinje_cat is not None
        assert retningslinje_cat.count == 5
        assert len(retningslinje_cat.results) == 5  # All results shown (is_priority)

    async def test_other_categories_limited_to_preview_count(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result(f"v{i}", settings.search_min_score + 0.1, "veileder")
            for i in range(10)
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="diabetes")

        veileder_cat = next(
            (c for c in response.other_categories if c.category == "veileder"), None
        )
        assert veileder_cat is not None
        assert veileder_cat.count == 10  # Full count reported
        assert len(veileder_cat.results) <= settings.search_category_preview_count

    async def test_results_below_min_score_excluded(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result("above", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("below", max(0.0, settings.search_min_score - 0.1), "retningslinje"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="test")

        all_result_ids = [
            r.id
            for cat in response.priority_categories + response.other_categories
            for r in cat.results
        ]
        assert "above" in all_result_ids
        if settings.search_min_score > 0:
            assert "below" not in all_result_ids

    async def test_total_matches_filtered_result_count(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        above_min = settings.search_min_score + 0.1
        results = [
            _make_result("001", above_min, "retningslinje"),
            _make_result("002", above_min, "veileder"),
            _make_result("003", above_min, "temaside"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="test")

        assert response.total == 3

    async def test_invalid_method_raises_value_error(self):
        ctrl = SearchController()
        with pytest.raises(ValueError, match="Invalid search method"):
            await ctrl.search_categorized(query="diabetes", method="invalid")

    async def test_logs_search_to_database(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        mocker.patch.object(ctrl, "_execute_search", return_value=[
            _make_result("001", settings.search_min_score + 0.1, "retningslinje"),
        ])
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        await ctrl.search_categorized(query="diabetes")

        mock_database_service.log_search.assert_called_once()
        call_kwargs = mock_database_service.log_search.call_args.kwargs
        assert call_kwargs.get("query") == "diabetes"

    async def test_empty_results_returns_empty_categories(
        self, mock_content, mock_database_service, mocker
    ):
        ctrl = SearchController()
        mocker.patch.object(ctrl, "_execute_search", return_value=[])
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_categorized(query="xyznomatch")

        assert response.total == 0
        assert response.priority_categories == []
        assert response.other_categories == []


@pytest.mark.unit
class TestSearchControllerSearchCategory:
    """
    Tests for async search_category() — the follow-up endpoint used when
    a user clicks 'Vis alle' for a specific category.

    Key behavior:
    - Validates search_id against DB (raises ValueError if not found)
    - Filters results to exactly the requested category
    - Filters out results below min_score
    - Returns full result list with no pagination
    """

    async def test_returns_search_response(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings
        from app.dto.response.search import SearchResponse

        ctrl = SearchController()
        results = [
            _make_result("001", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("002", settings.search_min_score + 0.1, "veileder"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_category(
            query="diabetes", category="retningslinje", search_id="test-id"
        )

        assert isinstance(response, SearchResponse)
        assert response.query == "diabetes"
        assert response.search_id == "test-id"

    async def test_filters_to_requested_category_only(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result("r1", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("v1", settings.search_min_score + 0.1, "veileder"),
            _make_result("r2", settings.search_min_score + 0.1, "retningslinje"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_category(
            query="diabetes", category="retningslinje", search_id="test-id"
        )

        result_ids = [r.id for r in response.results]
        assert "r1" in result_ids
        assert "r2" in result_ids
        assert "v1" not in result_ids  # Different category excluded

    async def test_all_results_returned_no_pagination(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result(f"r{i}", settings.search_min_score + 0.1, "retningslinje")
            for i in range(8)
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_category(
            query="diabetes", category="retningslinje", search_id="test-id"
        )

        assert response.total == 8
        assert len(response.results) == 8
        assert response.has_next is False
        assert response.has_prev is False

    async def test_results_below_min_score_excluded(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result("above", settings.search_min_score + 0.1, "retningslinje"),
            _make_result("below", max(0.0, settings.search_min_score - 0.1), "retningslinje"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_category(
            query="test", category="retningslinje", search_id="test-id"
        )

        result_ids = [r.id for r in response.results]
        assert "above" in result_ids
        if settings.search_min_score > 0:
            assert "below" not in result_ids

    async def test_invalid_search_id_raises_value_error(
        self, mock_content, mock_database_service
    ):
        mock_database_service.get_search_by_id.return_value = None

        ctrl = SearchController()
        with pytest.raises(ValueError, match="Invalid search_id"):
            await ctrl.search_category(
                query="diabetes", category="retningslinje", search_id="nonexistent"
            )

    async def test_invalid_method_raises_value_error(self, mock_database_service):
        ctrl = SearchController()
        with pytest.raises(ValueError, match="Invalid search method"):
            await ctrl.search_category(
                query="diabetes",
                category="retningslinje",
                method="invalid",
                search_id="test-id",
            )

    async def test_category_filter_case_insensitive(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result("001", settings.search_min_score + 0.1, "retningslinje"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_category(
            query="test", category="Retningslinje", search_id="test-id"
        )

        assert response.total == 1
        assert response.results[0].id == "001"

    async def test_empty_category_results_returns_empty_list(
        self, mock_content, mock_database_service, mocker
    ):
        from app.config import settings

        ctrl = SearchController()
        results = [
            _make_result("001", settings.search_min_score + 0.1, "veileder"),
        ]
        mocker.patch.object(ctrl, "_execute_search", return_value=results)
        mocker.patch.object(ctrl, "_populate_theme_page_children", side_effect=lambda x: x)
        mocker.patch.object(ctrl, "_log_results")

        response = await ctrl.search_category(
            query="test", category="retningslinje", search_id="test-id"
        )

        assert response.total == 0
        assert response.results == []
