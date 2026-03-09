"""
Integration tests for content routes.

Uses mock_content to populate content_service in-memory store,
and patches _build_links_with_children to avoid external API calls.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.entities.content import ContentItem


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

    def test_textless_report_returns_related_links(self, client, mock_content, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        mocker.patch(
            "app.routes.content.resolve_public_related_links",
            return_value=[
                {
                    "title": "Utviklingen i norsk kosthold 2024",
                    "url": "https://www.helsedirektoratet.no/rapporter/utviklingen-i-norsk-kosthold-2024",
                    "is_document": False,
                    "file_type": "UNDEF",
                    "url_type": "internal",
                    "target": "",
                },
                {
                    "title": "Utviklingen i norsk kosthold 2024 – Fullversjon",
                    "url": "https://www.helsedirektoratet.no/rapporter/utviklingen-i-norsk-kosthold/fullversjon.pdf?download=false",
                    "is_document": True,
                    "file_type": "PDF",
                    "url_type": "internal",
                    "target": "",
                },
            ],
        )
        report = ContentItem(
            id="report-no-text",
            title="Utviklingen i norsk kosthold",
            body="",
            content_type="rapport",
            path="/rapporter/utviklingen-i-norsk-kosthold",
            has_text_content=False,
            document_url=None,
        )
        mock_content.content.append(report)
        mock_content.content_by_id[report.id] = report
        mock_content.content_by_path[report.path] = report

        data = client.get("/content/report-no-text").json()
        assert data["is_pdf_only"] is False
        assert data["document_url"] is None
        assert data["related_links"] == [
            {
                "title": "Utviklingen i norsk kosthold 2024",
                "url": "https://www.helsedirektoratet.no/rapporter/utviklingen-i-norsk-kosthold-2024",
                "is_document": False,
                "file_type": "UNDEF",
                "url_type": "internal",
                "target": "",
            },
            {
                "title": "Utviklingen i norsk kosthold 2024 – Fullversjon",
                "url": "https://www.helsedirektoratet.no/rapporter/utviklingen-i-norsk-kosthold/fullversjon.pdf?download=false",
                "is_document": True,
                "file_type": "PDF",
                "url_type": "internal",
                "target": "",
            },
        ]

    def test_textless_report_with_only_navigation_links_returns_related_links(self, client, mock_content, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        mocker.patch(
            "app.routes.content.resolve_public_related_links",
            return_value=[
                {
                    "title": "Folkehelsepolitisk rapport 2017",
                    "url": "https://www.helsedirektoratet.no/rapporter/folkehelsepolitisk-rapport/Folkehelsepolitisk%20rapport%202017.pdf?download=false",
                    "is_document": True,
                    "file_type": "PDF",
                    "url_type": "internal",
                    "target": "",
                }
            ],
        )
        report = ContentItem(
            id="report-no-text-temaside",
            title="Folkehelsepolitisk rapport",
            body="",
            content_type="rapport",
            path="/rapporter/folkehelsepolitisk-rapport",
            has_text_content=False,
            document_url=None,
            links=[
                {"rel": "root", "type": "rapport", "id": "root-1"},
                {"rel": "publikasjon", "type": "rapport", "id": "pub-1"},
                {"rel": "temaside", "type": "temaside", "id": "theme-1"},
            ],
        )
        mock_content.content.append(report)
        mock_content.content_by_id[report.id] = report
        mock_content.content_by_path[report.path] = report

        data = client.get("/content/report-no-text-temaside").json()
        assert data["is_pdf_only"] is False
        assert data["document_url"] is None
        assert data["related_links"] == [
            {
                "title": "Folkehelsepolitisk rapport 2017",
                "url": "https://www.helsedirektoratet.no/rapporter/folkehelsepolitisk-rapport/Folkehelsepolitisk%20rapport%202017.pdf?download=false",
                "is_document": True,
                "file_type": "PDF",
                "url_type": "internal",
                "target": "",
            }
        ]


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

    def test_pdf_report_chapter_returns_pdf_document_url(self, client, mock_content, mocker):
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        pdf_item = ContentItem(
            id="pdf-chapter-1",
            title="PDF av rapporten",
            body='[knapp text="Last ned PDF av rapporten" internalLink="abc"/]',
            content_type="kapittel",
            path="/rapporter/test/pdf-av-rapporten",
            has_text_content=False,
            document_url="https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf",
        )
        mock_content.content.append(pdf_item)
        mock_content.content_by_id[pdf_item.id] = pdf_item
        mock_content.content_by_path[pdf_item.path] = pdf_item

        data = client.get("/content/pdf-chapter-1").json()
        assert data["document_url"] == "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf"
        assert data["is_pdf_only"] is True
