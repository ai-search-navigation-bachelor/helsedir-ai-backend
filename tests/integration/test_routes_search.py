"""
Integration tests for search routes.

Uses session-scoped TestClient and patches search_controller methods
at the point of use in app.routes.search.
"""

import pytest
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

from app.dto.response.search import (
    SearchResponse,
    SearchResult,
    SuggestionResponse,
    Suggestion,
    CategorizedSearchResponse,
    CategoryResults,
)
from app.dto.response.content import ContentSummaryResponse


def _mock_search_response(
    query: str = "diabetes",
    results: Optional[List[SearchResult]] = None,
) -> SearchResponse:
    if results is None:
        results = [
            SearchResult(
                id="001",
                title="Diabetes retningslinje",
                short_title="Diabetes",
                display_title="Diabetes",
                info_type="retningslinje",
                has_text_content=False,
                document_url="https://example.test/doc/1",
                is_pdf_only=True,
                score=0.9,
                explanation="test",
            )
        ]
    return SearchResponse(
        results=results,
        query=query,
        total=len(results),
        search_id="00000000-0000-0000-c0de-00000000abcd",
        offset=0,
        limit=15,
        has_next=False,
        has_prev=False,
    )


def _mock_categorized_response(query: str = "diabetes") -> CategorizedSearchResponse:
    return CategorizedSearchResponse(
        query=query,
        total=1,
        min_score=0.45,
        search_id="test-search-id",
        priority_categories=[
            CategoryResults(
                category="retningslinje",
                display_name="Retningslinjer",
                count=1,
                is_priority=True,
                results=[
                    SearchResult(
                        id="001",
                        title="Diabetes",
                        short_title="Diabetes",
                        display_title="Diabetes",
                        info_type="retningslinje",
                        has_text_content=False,
                        document_url="https://example.test/doc/1",
                        is_pdf_only=True,
                        score=0.9,
                        explanation="test",
                    )
                ],
            )
        ],
        other_categories=[],
    )


@pytest.mark.integration
class TestSearchRoute:
    def test_missing_query_returns_422(self, client):
        response = client.get("/search")
        assert response.status_code == 422

    def test_empty_query_returns_422(self, client):
        response = client.get("/search?query=")
        assert response.status_code == 422

    def test_valid_query_returns_200(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        response = client.get("/search?query=diabetes")
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        data = client.get("/search?query=diabetes").json()
        assert "results" in data
        assert "search_id" in data
        assert "total" in data
        assert "has_next" in data
        assert "has_prev" in data
        assert "query" in data

    def test_result_items_have_required_fields(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        data = client.get("/search?query=diabetes").json()
        for result in data["results"]:
            assert "id" in result
            assert "title" in result
            assert "score" in result
            assert "info_type" in result
            assert result["has_text_content"] is False
            assert result["document_url"] == "https://example.test/doc/1"
            assert result["is_pdf_only"] is True

    def test_result_items_can_include_parent_and_root_publication(self, client, mocker):
        result = SearchResult(
            id="001",
            title="ADHD-anbefaling",
            short_title="ADHD",
            display_title="ADHD",
            info_type="anbefaling",
            has_text_content=True,
            document_url=None,
            is_pdf_only=False,
            score=0.9,
            parent=ContentSummaryResponse(
                id="500",
                title="Kapittel om ADHD",
                content_type="kapittel",
                path="/kapitler/adhd",
            ),
            root_publication=ContentSummaryResponse(
                id="600",
                title="Nasjonal faglig retningslinje for ADHD",
                content_type="retningslinje",
                path="/retningslinjer/adhd",
            ),
        )
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response(results=[result])),
        )

        data = client.get("/search?query=adhd").json()

        assert data["results"][0]["parent"]["id"] == "500"
        assert data["results"][0]["root_publication"]["id"] == "600"

    def test_invalid_method_returns_400(self, client):
        response = client.get("/search?query=diabetes&method=invalid_method")
        assert response.status_code == 400

    def test_controller_value_error_returns_400(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(side_effect=ValueError("Invalid search method: xyz")),
        )
        response = client.get("/search?query=diabetes")
        assert response.status_code == 400

    def test_controller_runtime_error_returns_500(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(side_effect=RuntimeError("Something broke")),
        )
        response = client.get("/search?query=diabetes")
        assert response.status_code == 500

    def test_method_parameter_keyword_is_accepted(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        response = client.get("/search?query=diabetes&method=keyword")
        assert response.status_code == 200

    def test_method_parameter_semantic_is_accepted(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        response = client.get("/search?query=diabetes&method=semantic")
        assert response.status_code == 200

    def test_method_parameter_hybrid_is_accepted(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        response = client.get("/search?query=diabetes&method=hybrid")
        assert response.status_code == 200

    def test_pagination_params_passed_correctly(self, client, mocker):
        mock_search = AsyncMock(return_value=_mock_search_response())
        mocker.patch("app.routes.search.search_controller.search", new=mock_search)
        client.get("/search?query=diabetes&offset=10&limit=5")
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("offset") == 10
        assert call_kwargs.get("limit") == 5


@pytest.mark.integration
class TestSuggestionsRoute:
    def test_missing_query_returns_422(self, client):
        response = client.get("/search/suggestions")
        assert response.status_code == 422

    def test_valid_query_returns_200(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.get_suggestions",
            return_value=SuggestionResponse(suggestions=[]),
        )
        response = client.get("/search/suggestions?query=dia")
        assert response.status_code == 200

    def test_response_has_suggestions_field(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.get_suggestions",
            return_value=SuggestionResponse(
                suggestions=[
                    Suggestion(
                        id="004",
                        title="Psykisk helse temaside",
                        short_title="Psykisk helse",
                        display_title="Psykisk helse",
                    )
                ]
            ),
        )
        data = client.get("/search/suggestions?query=psykisk").json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_suggestions_items_have_id_and_title(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.get_suggestions",
            return_value=SuggestionResponse(
                suggestions=[
                    Suggestion(
                        id="004",
                        title="Psykisk helse temaside",
                        short_title="Psykisk helse",
                        display_title="Psykisk helse",
                    )
                ]
            ),
        )
        data = client.get("/search/suggestions?query=psykisk").json()
        for s in data["suggestions"]:
            assert "id" in s
            assert "title" in s

    def test_runtime_error_returns_500(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.get_suggestions",
            side_effect=RuntimeError("unexpected"),
        )
        response = client.get("/search/suggestions?query=test")
        assert response.status_code == 500


@pytest.mark.integration
class TestCategorizedSearchRoute:
    def test_missing_query_returns_422(self, client):
        response = client.get("/search/categorized")
        assert response.status_code == 422

    def test_valid_query_returns_200(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search_categorized",
            new=AsyncMock(return_value=_mock_categorized_response()),
        )
        response = client.get("/search/categorized?query=diabetes")
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search_categorized",
            new=AsyncMock(return_value=_mock_categorized_response()),
        )
        data = client.get("/search/categorized?query=diabetes").json()
        assert "query" in data
        assert "total" in data
        assert "search_id" in data
        assert "priority_categories" in data
        assert "other_categories" in data


@pytest.mark.integration
class TestCategorySearchRoute:
    BASE_URL = "/search/category?query=diabetes&category=retningslinje&search_id=test-id"

    def test_missing_query_returns_422(self, client):
        response = client.get("/search/category?category=retningslinje&search_id=test-id")
        assert response.status_code == 422

    def test_missing_category_returns_422(self, client):
        response = client.get("/search/category?query=diabetes&search_id=test-id")
        assert response.status_code == 422

    def test_missing_search_id_returns_422(self, client):
        response = client.get("/search/category?query=diabetes&category=retningslinje")
        assert response.status_code == 422

    def test_valid_request_returns_200(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search_category",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        response = client.get(self.BASE_URL)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search_category",
            new=AsyncMock(return_value=_mock_search_response()),
        )
        data = client.get(self.BASE_URL).json()
        assert "results" in data
        assert "search_id" in data
        assert "total" in data
        assert "has_next" in data
        assert "has_prev" in data

    def test_value_error_from_controller_returns_400(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search_category",
            new=AsyncMock(side_effect=ValueError("Invalid search_id")),
        )
        response = client.get(self.BASE_URL)
        assert response.status_code == 400

    def test_runtime_error_returns_500(self, client, mocker):
        mocker.patch(
            "app.routes.search.search_controller.search_category",
            new=AsyncMock(side_effect=RuntimeError("Unexpected error")),
        )
        response = client.get(self.BASE_URL)
        assert response.status_code == 500
