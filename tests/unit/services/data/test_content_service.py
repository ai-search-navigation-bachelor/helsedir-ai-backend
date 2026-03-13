from app.entities.content import ContentItem, ContentLink


def _theme_page(
    content_id: str,
    title: str,
    path: str,
    *,
    body: str = "",
    child_ids: list[str] | None = None,
) -> ContentItem:
    links = [
        ContentLink(
            rel="barn",
            type="temaside",
            id=child_id,
            tittel=f"Child {child_id}",
        )
        for child_id in (child_ids or [])
    ]
    return ContentItem(
        id=content_id,
        title=title,
        body=body,
        content_type="temaside",
        path=path,
        links=links,
        has_text_content=bool(body),
    )


def test_refresh_theme_page_visibility_keeps_navigation_nodes_with_visible_descendants(mocker):
    from app.services.data.content_service import content_service

    root = _theme_page("root", "Root", "/root", child_ids=["nav", "empty-leaf"])
    nav = _theme_page("nav", "Nav", "/root/nav", child_ids=["leaf"])
    leaf = _theme_page("leaf", "Leaf", "/root/nav/leaf", body="Har innhold")
    empty_leaf = _theme_page("empty-leaf", "Empty", "/root/empty")

    orig_content = content_service.content
    orig_by_id = dict(content_service.content_by_id)
    orig_by_path = dict(content_service.content_by_path)
    orig_visible = content_service.visible_theme_page_ids
    orig_child_counts = dict(content_service.theme_page_visible_child_counts)

    content_service.content = [root, nav, leaf, empty_leaf]
    content_service.content_by_id = {item.id: item for item in content_service.content}
    content_service.content_by_path = {
        item.path: item for item in content_service.content if item.path
    }
    mocker.patch(
        "app.services.data.content_service.content_repository.get_non_empty_theme_page_ids",
        return_value=set(),
    )

    try:
        content_service._refresh_theme_page_visibility()

        assert content_service.visible_theme_page_ids == {"root", "nav", "leaf"}
        assert content_service.get_theme_page_visible_child_count("root") == 1
        assert content_service.get_theme_page_visible_child_count("nav") == 1
        assert content_service.get_theme_page_visible_child_count("leaf") == 0
        assert content_service.get_theme_page_visible_child_count("empty-leaf") == 0
        assert content_service.is_theme_page_id_visible("empty-leaf") is False
    finally:
        content_service.content = orig_content
        content_service.content_by_id = orig_by_id
        content_service.content_by_path = orig_by_path
        content_service.visible_theme_page_ids = orig_visible
        content_service.theme_page_visible_child_counts = orig_child_counts


def test_refresh_theme_page_visibility_marks_linked_leaf_as_visible(mocker):
    from app.services.data.content_service import content_service

    parent = _theme_page("parent", "Parent", "/parent", child_ids=["linked-leaf"])
    linked_leaf = _theme_page("linked-leaf", "Linked", "/parent/linked")

    orig_content = content_service.content
    orig_by_id = dict(content_service.content_by_id)
    orig_by_path = dict(content_service.content_by_path)
    orig_visible = content_service.visible_theme_page_ids
    orig_child_counts = dict(content_service.theme_page_visible_child_counts)

    content_service.content = [parent, linked_leaf]
    content_service.content_by_id = {item.id: item for item in content_service.content}
    content_service.content_by_path = {
        item.path: item for item in content_service.content if item.path
    }
    mocker.patch(
        "app.services.data.content_service.content_repository.get_non_empty_theme_page_ids",
        return_value={"linked-leaf"},
    )

    try:
        content_service._refresh_theme_page_visibility()

        assert content_service.visible_theme_page_ids == {"parent", "linked-leaf"}
        assert content_service.get_theme_page_visible_child_count("parent") == 1
        assert content_service.is_theme_page_id_visible("linked-leaf") is True
    finally:
        content_service.content = orig_content
        content_service.content_by_id = orig_by_id
        content_service.content_by_path = orig_by_path
        content_service.visible_theme_page_ids = orig_visible
        content_service.theme_page_visible_child_counts = orig_child_counts


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
