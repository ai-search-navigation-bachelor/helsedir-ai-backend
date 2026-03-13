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
