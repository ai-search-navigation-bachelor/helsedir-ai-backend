import pytest

from scripts.data.importing.backfill_ehelsestandard_content import (
    _compute_update,
    _normalize_attachments,
)


@pytest.mark.unit
def test_normalize_attachments_builds_absolute_public_urls():
    attachments = _normalize_attachments(
        {
            "attachments": [
                {
                    "title": "Standard.pdf",
                    "fileUri": "/guillotine/helsedir/standard.pdf",
                    "fileType": "PDF",
                }
            ]
        }
    )

    assert attachments == [
        {
            "title": "Standard.pdf",
            "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
            "file_type": "PDF",
        }
    ]


@pytest.mark.unit
def test_compute_update_uses_formal_bruksomrade_when_existing_text_is_empty():
    update = _compute_update(
        {
            "id": "file-1",
            "tekst": "<p>&nbsp;</p>",
            "document_url": None,
            "forst_publisert": None,
            "sist_faglig_oppdatert": None,
        },
        {
            "forstPublisert": "2014-06-15T00:00:00",
            "sistFagligOppdatert": "2025-12-15T00:00:00",
            "data": {
                "formalBruksomrade": "<p>Dokumentet beskriver ...</p>",
            },
            "attachments": [
                {
                    "title": "Standard.pdf",
                    "fileUri": "/guillotine/helsedir/standard.pdf",
                    "fileType": "PDF",
                }
            ],
        },
    )

    assert update == (
        "<p>Dokumentet beskriver ...</p>",
        1,
        "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
        "2014-06-15T00:00:00",
        "2025-12-15T00:00:00",
        '[{"title": "Standard.pdf", "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf", "file_type": "PDF"}]',
        "file-1",
    )
