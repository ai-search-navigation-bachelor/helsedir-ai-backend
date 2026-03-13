"""
Integration tests for content routes.

Uses mock_content to populate content_service in-memory store,
and patches _build_links_with_children to avoid external API calls.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.dto.response.content import ContentLinkResponse
from app.entities.content import (
    ContentItem,
    ContentLink,
    EhelsestandardAttachment,
    EhelsestandardFields,
)


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
                "path": "/rapporter/utviklingen-i-norsk-kosthold-2024",
                "content_id": None,
            },
            {
                "title": "Utviklingen i norsk kosthold 2024 – Fullversjon",
                "url": "https://www.helsedirektoratet.no/rapporter/utviklingen-i-norsk-kosthold/fullversjon.pdf?download=false",
                "is_document": True,
                "file_type": "PDF",
                "url_type": "internal",
                "target": "",
                "path": "/rapporter/utviklingen-i-norsk-kosthold/fullversjon.pdf",
                "content_id": None,
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
                "path": "/rapporter/folkehelsepolitisk-rapport/Folkehelsepolitisk%20rapport%202017.pdf",
                "content_id": None,
            }
        ]

    def test_textless_report_maps_internal_related_pages_to_internal_paths(self, client, mock_content, mocker):
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
                    "title": "Årsrapport for NKI 2024",
                    "url": "https://www.helsedirektoratet.no/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2024",
                    "is_document": False,
                    "file_type": None,
                    "url_type": "external",
                    "target": "",
                },
                {
                    "title": "Årsrapport for NKI 2020",
                    "url": "https://www.helsedirektoratet.no/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2020.pdf",
                    "is_document": True,
                    "file_type": "PDF",
                    "url_type": "internal",
                    "target": "",
                },
            ],
        )
        report = ContentItem(
            id="nki-root",
            title="Nasjonalt kvalitetsindikatorsystem (NKI) – Årsrapporter",
            body="",
            content_type="rapport",
            path="/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapporter",
            has_text_content=False,
            document_url=None,
        )
        child_report = ContentItem(
            id="nki-2024",
            title="Årsrapport for NKI 2024",
            body="<p>Intern rapportside</p>",
            content_type="rapport",
            path="/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2024",
            has_text_content=True,
            document_url=None,
        )
        mock_content.content.extend([report, child_report])
        mock_content.content_by_id[report.id] = report
        mock_content.content_by_id[child_report.id] = child_report
        mock_content.content_by_path[report.path] = report
        mock_content.content_by_path[child_report.path] = child_report

        data = client.get("/content/nki-root").json()

        assert data["related_links"] == [
            {
                "title": "Årsrapport for NKI 2024",
                "url": "/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2024",
                "is_document": False,
                "file_type": None,
                "url_type": "internal",
                "target": "",
                "path": "/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2024",
                "content_id": "nki-2024",
            },
            {
                "title": "Årsrapport for NKI 2020",
                "url": "https://www.helsedirektoratet.no/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2020.pdf",
                "is_document": True,
                "file_type": "PDF",
                "url_type": "internal",
                "target": "",
                "path": "/rapporter/nasjonalt-kvalitetsindikatorsystem-nki-arsrapport-2020.pdf",
                "content_id": None,
            },
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


@pytest.mark.integration
@pytest.mark.usefixtures("mock_content")
class TestEhelsestandardEnrichment:
    @staticmethod
    def _add_ehelsestandard(mock_content, *, body="", has_text_content=False):
        item = ContentItem(
            id="0006-0072-526cd1f3-2f11-46b8-8b15-eba47a16a403",
            title="Standard for svarrapportering av medisinske tjenester v1.4 (HIS 80822:2014)",
            body=body,
            content_type="ehelsestandard",
            path="/standarder/svarrapportering-av-medisinske-tjenester-v1.4",
            has_text_content=has_text_content,
        )
        mock_content.content.append(item)
        mock_content.content_by_id[item.id] = item
        mock_content.content_by_path[item.path] = item
        return item

    @staticmethod
    def _external_payload():
        return {
            "url": "https://www.helsedirektoratet.no/standarder/svarrapportering-av-medisinske-tjenester-v1.4",
            "forstPublisert": "2014-06-15T00:00:00",
            "sistFagligOppdatert": "2025-12-15T00:00:00",
            "data": {
                "idStandard": "HIS 80822:2014",
                "typeStandard": "obligatoriskStandard",
                "formalBruksomrade": "<p>Dokumentet beskriver ...</p><p>Standarden brukes ikke selvstendig ...</p>",
                "standardenGjelderFor": "",
            },
            "attachments": [
                {
                    "title": "Svarrapportering av medisinske tjenester v1.4.pdf",
                    "fileUri": "/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf",
                    "fileType": "PDF",
                }
            ],
        }

    def test_content_by_id_enriches_textless_ehelsestandard(self, client, mock_content, mocker):
        self._add_ehelsestandard(mock_content, body="", has_text_content=False)
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        file_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_file_by_id_async",
            new=AsyncMock(return_value=self._external_payload()),
        )

        data = client.get("/content/0006-0072-526cd1f3-2f11-46b8-8b15-eba47a16a403").json()

        assert data["body"] == "<p>Dokumentet beskriver ...</p><p>Standarden brukes ikke selvstendig ...</p>"
        assert data["has_text_content"] is True
        assert data["document_url"] == (
            "https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf"
        )
        assert data["is_pdf_only"] is False
        assert data["url"] == "https://www.helsedirektoratet.no/standarder/svarrapportering-av-medisinske-tjenester-v1.4"
        assert data["first_published"] == "2014-06-15T00:00:00"
        assert data["last_reviewed_date"] == "2025-12-15T00:00:00"
        assert data["ehelsestandard_fields"] == {
            "standard_id": "HIS 80822:2014",
            "standard_type": "obligatoriskStandard",
            "purpose_html": "<p>Dokumentet beskriver ...</p><p>Standarden brukes ikke selvstendig ...</p>",
            "applies_to_html": "",
            "attachments": [
                {
                    "title": "Svarrapportering av medisinske tjenester v1.4.pdf",
                    "url": "https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf",
                    "file_type": "PDF",
                }
            ],
        }
        file_mock.assert_awaited_once()

    def test_content_by_path_returns_same_enriched_shape(self, client, mock_content, mocker):
        item = self._add_ehelsestandard(mock_content, body="", has_text_content=False)
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        mocker.patch(
            "app.routes.content.helsedir_api_service.get_file_by_id_async",
            new=AsyncMock(return_value=self._external_payload()),
        )

        by_id = client.get(f"/content/{item.id}").json()
        by_path = client.get(f"/content/by-path?path={item.path}").json()

        assert by_path == by_id

    def test_empty_html_body_triggers_external_enrichment(self, client, mock_content, mocker):
        item = self._add_ehelsestandard(mock_content, body="<h2>&nbsp;</h2><p> </p>", has_text_content=False)
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        file_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_file_by_id_async",
            new=AsyncMock(return_value=self._external_payload()),
        )

        data = client.get(f"/content/{item.id}").json()

        assert data["has_text_content"] is True
        assert data["body"] == "<p>Dokumentet beskriver ...</p><p>Standarden brukes ikke selvstendig ...</p>"
        file_mock.assert_awaited_once()

    def test_db_backfilled_ehelsestandard_skips_external_enrichment(self, client, mock_content, mocker):
        item = self._add_ehelsestandard(
            mock_content,
            body="<p>Dette er intern tekst som skal beholdes.</p>",
            has_text_content=True,
        )
        item.document_url = "https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf"
        item.ehelsestandard_fields = EhelsestandardFields(
            attachments=[
                EhelsestandardAttachment(
                    title="Svarrapportering av medisinske tjenester v1.4.pdf",
                    url="https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf",
                    file_type="PDF",
                )
            ]
        )
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        file_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_file_by_id_async",
            new=AsyncMock(return_value=self._external_payload()),
        )

        data = client.get(f"/content/{item.id}").json()

        assert data["body"] == "<p>Dette er intern tekst som skal beholdes.</p>"
        assert data["has_text_content"] is True
        assert data["document_url"] == "https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf"
        assert data["url"] == "https://www.helsedirektoratet.no/standarder/svarrapportering-av-medisinske-tjenester-v1.4"
        assert data["ehelsestandard_fields"] == {
            "standard_id": None,
            "standard_type": None,
            "purpose_html": "<p>Dette er intern tekst som skal beholdes.</p>",
            "applies_to_html": None,
            "attachments": [
                {
                    "title": "Svarrapportering av medisinske tjenester v1.4.pdf",
                    "url": "https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf",
                    "file_type": "PDF",
                }
            ],
        }
        file_mock.assert_not_awaited()

    def test_visible_internal_text_without_db_attachments_still_uses_fallback_for_files(self, client, mock_content, mocker):
        item = self._add_ehelsestandard(
            mock_content,
            body="<p>Dette er intern tekst som skal beholdes.</p>",
            has_text_content=True,
        )
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        file_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_file_by_id_async",
            new=AsyncMock(return_value=self._external_payload()),
        )

        data = client.get(f"/content/{item.id}").json()

        assert data["body"] == "<p>Dette er intern tekst som skal beholdes.</p>"
        assert data["document_url"] == (
            "https://www.helsedirektoratet.no/guillotine/helsedirektoratet/master/_/attachment/inline/test.pdf"
        )
        file_mock.assert_awaited_once()

    def test_external_failure_returns_original_content(self, client, mock_content, mocker):
        item = self._add_ehelsestandard(mock_content, body="", has_text_content=False)
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )
        mocker.patch(
            "app.routes.content.helsedir_api_service.get_file_by_id_async",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        )

        data = client.get(f"/content/{item.id}").json()

        assert data["body"] == ""
        assert data["has_text_content"] is False
        assert data["document_url"] is None
        assert data["ehelsestandard_fields"] is None

@pytest.mark.integration
@pytest.mark.usefixtures("mock_content")
class TestNormalizedContentRelations:
    def _install_recommendation_fixture(self, content_service):
        recommendation = ContentItem(
            id="100",
            title="Anbefaling om oppfolging",
            body="Detaljert anbefalingstekst.",
            content_type="pakkeforlop-anbefaling",
            path="/anbefalinger/oppfolging",
            links=[
                ContentLink(rel="barn", type="referanse", id="200", tittel="Kildegrunnlag"),
                ContentLink(rel="barn", type="pico", id="300", tittel="PICO-sporsmal"),
                ContentLink(rel="barn", type="kapittel", id="400", tittel="Kapittel A"),
                ContentLink(rel="forelder", type="kapittel", id="500", tittel="Overordnet kapittel"),
                ContentLink(rel="root", type="retningslinje", id="600", tittel="Retningslinje rot"),
            ],
        )
        reference = ContentItem(
            id="200",
            title="Kildegrunnlag",
            body="",
            content_type="referanse",
            path="/referanser/kildegrunnlag",
        )
        pico = ContentItem(
            id="300",
            title="PICO-sporsmal",
            body="",
            content_type="pico",
            path="/pico/sporsmal",
        )
        chapter = ContentItem(
            id="400",
            title="Kapittel A",
            body="",
            content_type="kapittel",
            path="/kapitler/a",
        )
        parent = ContentItem(
            id="500",
            title="Overordnet kapittel",
            body="",
            content_type="kapittel",
            path="/kapitler/overordnet",
        )
        root = ContentItem(
            id="600",
            title="Retningslinje rot",
            body="",
            content_type="retningslinje",
            path="/retningslinjer/rot",
        )

        content_service.content = [
            recommendation,
            reference,
            pico,
            chapter,
            parent,
            root,
        ]
        content_service.content_by_id = {item.id: item for item in content_service.content}
        content_service.content_by_path = {
            item.path: item for item in content_service.content if item.path
        }
        content_service.searchable_types = {item.content_type for item in content_service.content}

        return [
            ContentLinkResponse(rel="barn", type="referanse", id="200", title="Kildegrunnlag"),
            ContentLinkResponse(rel="barn", type="pico", id="300", title="PICO-sporsmal"),
            ContentLinkResponse(rel="barn", type="kapittel", id="400", title="Kapittel A"),
            ContentLinkResponse(rel="forelder", type="kapittel", id="500", title="Overordnet kapittel"),
            ContentLinkResponse(rel="root", type="retningslinje", id="600", title="Retningslinje rot"),
        ]

    def test_recommendation_has_explicit_references_and_related_content(
        self,
        client,
        mocker,
        mock_content,
    ):
        links_response = self._install_recommendation_fixture(mock_content)
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=links_response),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )

        data = client.get("/content/100").json()

        assert data["detail_level"] == "full"
        assert data["content_type"] == "anbefaling"
        assert [item["id"] for item in data["references"]] == ["200"]
        assert [item["id"] for item in data["related_content"]] == ["300"]
        assert [item["id"] for item in data["chapters"]] == ["400"]
        assert data["references"][0]["content_type"] == "referanse"
        assert data["related_content"][0]["content_type"] == "pico"
        assert data["parent"]["id"] == "500"
        assert data["root_publication"]["id"] == "600"
        assert [group["key"] for group in data["child_groups"]] == [
            "chapters",
            "references",
            "related_content",
        ]

    def test_content_by_id_and_path_share_same_normalized_shape(
        self,
        client,
        mocker,
        mock_content,
    ):
        links_response = self._install_recommendation_fixture(mock_content)
        mocker.patch(
            "app.routes.content._build_links_with_children",
            new=AsyncMock(return_value=links_response),
        )
        mocker.patch(
            "app.routes.content.content_repository.get_theme_page_content",
            return_value=[],
        )

        by_id = client.get("/content/100").json()
        by_path = client.get("/content/by-path?path=/anbefalinger/oppfolging").json()

        assert by_path["content_type"] == by_id["content_type"] == "anbefaling"
        assert by_path["references"] == by_id["references"]
        assert by_path["related_content"] == by_id["related_content"]
        assert by_path["chapters"] == by_id["chapters"]
        assert by_path["child_groups"] == by_id["child_groups"]


@pytest.mark.integration
@pytest.mark.usefixtures("mock_content")
class TestPdfReportChapters:
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
