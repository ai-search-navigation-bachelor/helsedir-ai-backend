def test_parse_ehelsestandard_fields_reads_attachments_json():
    from app.services.data.content_service import content_service

    parsed = content_service._parse_ehelsestandard_fields(
        {
            "attachments_json": [
                {
                    "title": "Standard.pdf",
                    "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
                    "file_type": "PDF",
                }
            ]
        }
    )

    assert parsed is not None
    assert [attachment.model_dump() for attachment in parsed.attachments] == [
        {
            "title": "Standard.pdf",
            "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
            "file_type": "PDF",
        }
    ]


def test_parse_ehelsestandard_fields_prefers_stored_metadata_json():
    from app.services.data.content_service import content_service

    parsed = content_service._parse_ehelsestandard_fields(
        {
            "ehelsestandard_fields_json": {
                "standard_id": "STD-1",
                "standard_type": "Norm",
                "purpose_html": "<p>Formaal</p>",
                "applies_to_html": "<p>Gjelder for</p>",
                "attachments": [
                    {
                        "title": "Standard.pdf",
                        "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
                        "file_type": "PDF",
                    }
                ],
            }
        }
    )

    assert parsed is not None
    assert parsed.standard_id == "STD-1"
    assert parsed.standard_type == "Norm"
    assert parsed.purpose_html == "<p>Formaal</p>"
    assert parsed.applies_to_html == "<p>Gjelder for</p>"
    assert [attachment.model_dump() for attachment in parsed.attachments] == [
        {
            "title": "Standard.pdf",
            "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
            "file_type": "PDF",
        }
    ]
