"""
Integration tests for content routes.

Uses mock_content to populate content_service in-memory store,
and patches _build_links_with_children to avoid external API calls.
"""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.integration
@pytest.mark.usefixtures("mock_content")
class TestGetContentById:
    def test_nonexistent_id_returns_404(self, client):
        response = client.get("/content/nonexistent-id-xyz")
        assert response.status_code == 404

    def test_existing_id_returns_200(self, client, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        response = client.get("/content/001")
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        data = client.get("/content/001").json()
        assert "id" in data
        assert "title" in data
        assert "body" in data
        assert "content_type" in data
        assert data["has_text_content"] is True
        assert data["document_url"] is None
        assert data["is_pdf_only"] is False

    def test_response_id_matches_requested(self, client, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        data = client.get("/content/001").json()
        assert data["id"] == "001"

    def test_content_type_is_json(self, client, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        response = client.get("/content/001")
        assert "application/json" in response.headers["content-type"]


@pytest.mark.integration
@pytest.mark.usefixtures("mock_content")
class TestGetContentByPath:
    def test_missing_path_param_returns_422(self, client):
        response = client.get("/content/by-path")
        assert response.status_code == 422

    def test_nonexistent_path_returns_404(self, client):
        response = client.get("/content/by-path?path=/nonexistent/path")
        assert response.status_code == 404

    def test_existing_path_returns_200(self, client, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        response = client.get("/content/by-path?path=/retningslinjer/diabetes")
        assert response.status_code == 200

    def test_response_matches_path_content(self, client, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        data = client.get("/content/by-path?path=/retningslinjer/diabetes").json()
        assert data["id"] == "001"
        assert data["content_type"] == "retningslinje"
