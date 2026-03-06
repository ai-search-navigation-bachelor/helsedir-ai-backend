import pytest

from app.services.data.document_metadata import compute_document_metadata, has_visible_text


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
