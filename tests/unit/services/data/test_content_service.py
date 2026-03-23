from app.entities.content import ContentItem, ContentLink


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


def test_refresh_dead_end_theme_page_flags_marks_only_true_dead_ends(mocker):
    from app.services.data.content_service import content_service

    parent = ContentItem(
        id="theme-parent",
        title="Parent theme",
        body="",
        content_type="temaside",
        path="/tema/parent",
        links=[ContentLink(rel="barn", type="temaside", id="theme-active-child", tittel="Active child")],
    )
    active_child = ContentItem(
        id="theme-active-child",
        title="Active child",
        body="",
        content_type="temaside",
        path="/tema/active-child",
        links=[ContentLink(rel="barn", type="rapport", id="report-1", tittel="Report")],
    )
    dead_end_child = ContentItem(
        id="theme-dead-end-child",
        title="Dead-end child",
        body="",
        content_type="temaside",
        path="/tema/dead-end-child",
    )
    report = ContentItem(
        id="report-1",
        title="Report",
        body="Report body",
        content_type="rapport",
        path="/rapport/report-1",
        has_text_content=True,
    )

    orig_content = content_service.content[:]
    orig_by_id = dict(content_service.content_by_id)
    orig_by_path = dict(content_service.content_by_path)
    orig_types = set(content_service.searchable_types)
    try:
        content_service.content = [parent, active_child, dead_end_child, report]
        content_service._rebuild_lookup_dicts()
        content_service.searchable_types = {
            item.content_type for item in content_service.content
        }

        batch_mock = mocker.patch(
            "app.services.data.content_service.content_repository.get_theme_pages_content_batch",
            return_value={},
        )
        update_mock = mocker.patch(
            "app.services.data.content_service.content_repository.update_dead_end_theme_page_flags",
            return_value=3,
        )

        flags = content_service.refresh_dead_end_theme_page_flags(persist=True)

        batch_mock.assert_called_once_with(
            ["theme-parent", "theme-active-child", "theme-dead-end-child"]
        )
        update_mock.assert_called_once_with(
            {
                "theme-parent": False,
                "theme-active-child": False,
                "theme-dead-end-child": True,
            }
        )
        assert flags == {
            "theme-parent": False,
            "theme-active-child": False,
            "theme-dead-end-child": True,
        }
        assert content_service.content_by_id["theme-parent"].is_dead_end_theme_page is False
        assert content_service.content_by_id["theme-active-child"].is_dead_end_theme_page is False
        assert content_service.content_by_id["theme-dead-end-child"].is_dead_end_theme_page is True
    finally:
        content_service.content = orig_content
        content_service.content_by_id = orig_by_id
        content_service.content_by_path = orig_by_path
        content_service.searchable_types = orig_types
