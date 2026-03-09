import pytest

from app.entities.content import ContentItem
from app.services.data.document_metadata import (
    compute_document_metadata,
    compute_document_metadata_with_fallback,
    extract_pdf_url_from_public_html,
    has_visible_text,
    resolve_pdf_report_chapter_document_url,
    resolve_public_document_url,
)


@pytest.mark.unit
class TestDocumentMetadata:
    def test_has_visible_text_ignores_empty_html(self):
        assert has_visible_text("<p>&nbsp;</p>") is False

    def test_has_visible_text_ignores_shortcode_only_pdf_chapter(self):
        assert (
            has_visible_text(
                '<p>[knapp text="Last ned PDF av rapporten" internalLink="abc"/]</p>',
                is_pdf_report_chapter=True,
            )
            is False
        )

    def test_has_visible_text_keeps_real_text_for_pdf_chapter(self):
        assert (
            has_visible_text(
                '<p>[knapp text="Last ned PDF av rapporten" internalLink="abc"/]</p><p>Faktisk tekst</p>',
                is_pdf_report_chapter=True,
            )
            is True
        )

    def test_has_visible_text_ignores_anchor_without_href_for_pdf_chapter(self):
        assert has_visible_text("<p><a>Last ned PDF av rapporten</a></p>", is_pdf_report_chapter=True) is False

    def test_compute_document_metadata_detects_data_fil_url(self):
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

    def test_compute_document_metadata_detects_attachment_non_pdf_url(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "attachments": [
                    {
                        "url": "https://helsedirektoratet.no/rapporter/test/test.docx",
                    }
                ],
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] == "https://helsedirektoratet.no/rapporter/test/test.docx"

    def test_compute_document_metadata_detects_attachment_pdf_url(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "attachments": [
                    {
                        "href": "https://helsedirektoratet.no/rapporter/test/test.pdf",
                    }
                ],
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] == "https://helsedirektoratet.no/rapporter/test/test.pdf"

    def test_compute_document_metadata_ignores_non_pdf_links(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "links": [
                    {
                        "url": "https://helsedirektoratet.no/rapporter/test/test.docx",
                    }
                ],
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] is None

    def test_compute_document_metadata_detects_pdf_link(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "links": [
                    {
                        "href": "https://helsedirektoratet.no/rapporter/test/test.pdf",
                    }
                ],
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] == "https://helsedirektoratet.no/rapporter/test/test.pdf"

    def test_compute_document_metadata_ignores_non_pdf_lenker(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "lenker": [
                    {
                        "fil": "https://helsedirektoratet.no/rapporter/test/test.docx",
                    }
                ],
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] is None

    def test_compute_document_metadata_detects_pdf_lenker(self):
        meta = compute_document_metadata(
            {
                "tekst": "",
                "lenker": [
                    {
                        "fil": "https://helsedirektoratet.no/rapporter/test/test.pdf",
                    }
                ],
            }
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] == "https://helsedirektoratet.no/rapporter/test/test.pdf"

    def test_compute_document_metadata_detects_visible_text(self):
        meta = compute_document_metadata({"tekst": "<p>Faktisk innhold</p>"})
        assert meta["has_text_content"] is True
        assert meta["document_url"] is None

    def test_compute_document_metadata_treats_pdf_chapter_shortcode_as_non_text(self):
        meta = compute_document_metadata(
            {
                "path": "/rapporter/test/pdf-av-rapporten",
                "tekst": '<p>[knapp text="Last ned PDF av rapporten" internalLink="abc"/]</p>',
            }
        )
        assert meta["has_text_content"] is False
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

    def test_resolve_public_document_url_prefers_pdf_file_for_pdf_chapter(self):
        file_url = "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf"
        assert resolve_public_document_url("/rapporter/test/pdf-av-rapporten", file_url) == file_url

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

    def test_content_item_public_document_url_prefers_pdf_for_pdf_chapter(self):
        item = ContentItem(
            id="pdf-2",
            title="PDF-kapittel",
            body="",
            content_type="kapittel",
            path="/rapporter/test/pdf-av-rapporten",
            has_text_content=False,
            document_url="https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf",
        )
        assert item.public_document_url == "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf"

    def test_extract_pdf_url_from_public_html_returns_first_pdf_link(self):
        html = """
        <html><body>
        <a href="/ikke/denne">Ikke PDF</a>
        <a href="/rapporter/test/file.pdf">PDF</a>
        <a href="/rapporter/test/file-2.pdf">PDF 2</a>
        </body></html>
        """
        assert (
            extract_pdf_url_from_public_html(
                html,
                base_url="https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten",
            )
            == "https://www.helsedirektoratet.no/rapporter/test/file.pdf"
        )

    def test_extract_pdf_url_from_public_html_returns_none_when_missing(self):
        assert extract_pdf_url_from_public_html("<html><body><a href='/foo'>Hei</a></body></html>") is None

    def test_resolve_pdf_report_chapter_document_url_fetches_public_page(self):
        class DummyResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class DummyClient:
            def get(self, url):
                assert url == "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten"
                return DummyResponse('<a href="/rapporter/test/pdf-av-rapporten/test.pdf">Last ned</a>')

            def close(self):
                return None

        assert (
            resolve_pdf_report_chapter_document_url(
                "/rapporter/test/pdf-av-rapporten",
                client=DummyClient(),
            )
            == "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf"
        )

    def test_compute_document_metadata_with_fallback_resolves_pdf_report_chapter(self):
        class DummyResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class DummyClient:
            def get(self, url):
                assert url == "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten"
                return DummyResponse('<a href="https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf">Last ned</a>')

            def close(self):
                return None

        meta = compute_document_metadata_with_fallback(
            {
                "path": "/rapporter/test/pdf-av-rapporten",
                "tekst": '<p>[knapp text="Last ned PDF av rapporten" internalLink="abc"/]</p>',
            },
            client=DummyClient(),
        )

        assert meta["has_text_content"] is False
        assert meta["document_url"] == "https://www.helsedirektoratet.no/rapporter/test/pdf-av-rapporten/test.pdf"
