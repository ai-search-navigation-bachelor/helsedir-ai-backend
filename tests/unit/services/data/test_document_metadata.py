import pytest

from app.entities.content import ContentItem
from app.services.data.document_metadata import (
    compute_document_metadata,
    has_visible_text,
    resolve_public_document_url,
)


@pytest.mark.unit
class TestDocumentMetadata:
    def test_has_visible_text_ignores_empty_html(self):
        assert has_visible_text("<p>&nbsp;</p>") is False

    def test_compute_document_metadata_detects_pdf_url(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "data": {
                    "fil": "https://helsedirektoratet.no/rapporter/test/test.pdf",
                },
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] == "https://helsedirektoratet.no/rapporter/test/test.pdf"

    def test_compute_document_metadata_detects_visible_text(self):
        meta = compute_document_metadata({"tekst": "<p>Faktisk innhold</p>"})
        assert meta["has_text_content"] is True
        assert meta["document_url"] is None

    def test_resolve_public_document_url_prefers_path_over_file_url(self):
        meta = compute_document_metadata(
            {
                "path": "/rapporter/test-side",
                "data": {
                    "fil": "https://helsedirektoratet.no/rapporter/test/test.pdf",
                },
            }
        )
        assert meta["document_url"] == "https://helsedirektoratet.no/rapporter/test/test.pdf"
        assert (
            resolve_public_document_url("/rapporter/test-side", meta["document_url"])
            == "https://www.helsedirektoratet.no/rapporter/test-side"
        )

    def test_resolve_public_document_url_falls_back_to_file_when_path_missing(self):
        meta = compute_document_metadata(
            {
                "data": {
                    "fil": "https://helsedirektoratet.no/rapporter/test/test.pdf",
                },
            }
        )
        assert resolve_public_document_url(None, meta["document_url"]) == meta["document_url"]

    def test_resolve_public_document_url_falls_back_to_file_when_path_empty(self):
        file_url = "https://helsedirektoratet.no/rapporter/test/test.pdf"
        assert resolve_public_document_url("   ", file_url) == file_url

    def test_content_item_public_document_url_prefers_path(self):
        item = ContentItem(
            id="pdf-1",
            title="PDF",
            body="",
            content_type="rapport",
            path="/innhold/pdf-1",
            document_url="https://helsedirektoratet.no/rapporter/test/test.pdf",
        )
        assert item.public_document_url == "https://www.helsedirektoratet.no/innhold/pdf-1"
