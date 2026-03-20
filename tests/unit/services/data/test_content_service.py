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


def test_load_from_api_reloads_from_database_cache_after_refresh(mocker):
    from app.services.data.content_service import content_service

    search_mock = mocker.patch(
        "app.services.external.helsedir_api_service.helsedir_api_service.search_infobits",
        return_value=[{"id": "001", "tittel": "Tittel", "tekst": "Tekst", "infoType": "statistikk"}],
    )
    cache_mock = mocker.patch(
        "app.services.data.content_service.database_service.cache_content_batch",
        return_value=1,
    )
    load_mock = mocker.patch.object(content_service, "load_content")

    content_service.load_from_api(query_text="test", max_items=1)

    search_mock.assert_called_once_with(query_text="test", get_full_infobits=True)
    cache_mock.assert_called_once()
    load_mock.assert_called_once()
