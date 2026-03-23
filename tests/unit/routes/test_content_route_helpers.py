"""
Unit tests for content route helper functions.
"""

import pytest

from app.entities.content import ContentItem, ContentLink
from app.routes.content import (
    _build_links_with_children,
    _normalize_ehelsestandard_attachments,
    _resolve_ehelsestandard_attachment_url,
)


@pytest.mark.unit
class TestBuildLinksWithChildren:
    async def test_public_theme_href_uses_path_lookup_without_api_call(self, mock_content, mocker):
        from app.services.data.content_service import content_service

        grandchild = ContentItem(
            id="grandchild-001",
            title="Grandchild",
            body="",
            content_type="rapport",
            path="/rapporter/grandchild",
        )
        child = ContentItem(
            id="child-theme-001",
            title="Theme child",
            body="",
            content_type="temaside",
            path="/digitalisering-og-e-helse/kunstig-intelligens/rapporter-om-store-sprakmodeller-i-helse-og-omsorgstjenesten",
            links=[ContentLink(rel="barn", type="rapport", id="grandchild-001", tittel="Grandchild")],
        )

        content_service.content.extend([child, grandchild])
        content_service.content_by_id[child.id] = child
        content_service.content_by_id[grandchild.id] = grandchild
        content_service.content_by_path[child.path] = child
        content_service.content_by_path[grandchild.path] = grandchild

        api_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_content_by_href_async"
        )

        results = await _build_links_with_children([
            ContentLink(
                rel="barn",
                type="temaside",
                href=child.path,
                tittel=child.title,
            )
        ])

        assert len(results) == 1
        assert len(results[0].children or []) == 1
        assert results[0].children[0].id == "grandchild-001"
        api_mock.assert_not_called()

    async def test_api_href_still_fetches_when_not_in_cache(self, mock_content, mocker):
        api_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_content_by_href_async",
            return_value={
                "links": [
                    {
                        "rel": "barn",
                        "type": "rapport",
                        "id": "api-child-001",
                        "tittel": "API child",
                    }
                ]
            },
        )

        api_href = "https://api.helsedirektoratet.no/innhold/infobit/0006-1234-abcdef12-3456-7890-abcd-ef1234567890"
        results = await _build_links_with_children([
            ContentLink(rel="barn", type="temaside", href=api_href, tittel="API parent")
        ])

        assert len(results) == 1
        assert len(results[0].children or []) == 1
        assert results[0].children[0].id == "api-child-001"
        api_mock.assert_called_once()

    async def test_non_api_external_href_does_not_fetch(self, mock_content, mocker):
        api_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_content_by_href_async"
        )

        results = await _build_links_with_children([
            ContentLink(rel="barn", type="artikkel", href="https://example.com/whatever", tittel="External")
        ])

        assert len(results) == 1
        assert results[0].children == []
        api_mock.assert_not_called()

    async def test_public_theme_href_includes_no_content_tag_for_dead_end_child(self, mock_content, mocker):
        from app.services.data.content_service import content_service

        dead_end_child = ContentItem(
            id="child-theme-002",
            title="Dead-end child",
            body="",
            content_type="temaside",
            path="/digitalisering-og-e-helse/dead-end-child",
            is_dead_end_theme_page=True,
        )

        content_service.content.append(dead_end_child)
        content_service.content_by_id[dead_end_child.id] = dead_end_child
        content_service.content_by_path[dead_end_child.path] = dead_end_child

        api_mock = mocker.patch(
            "app.routes.content.helsedir_api_service.get_content_by_href_async"
        )

        results = await _build_links_with_children([
            ContentLink(
                rel="barn",
                type="temaside",
                href=dead_end_child.path,
                tittel=dead_end_child.title,
            )
        ])

        assert len(results) == 1
        assert results[0].tags == ["no_content"]
        api_mock.assert_not_called()


@pytest.mark.unit
class TestEhelsestandardHelpers:
    def test_resolve_ehelsestandard_attachment_url_builds_public_url_for_relative_path(self):
        assert (
            _resolve_ehelsestandard_attachment_url("/guillotine/test.pdf")
            == "https://www.helsedirektoratet.no/guillotine/test.pdf"
        )

    def test_normalize_ehelsestandard_attachments_maps_relative_file_uri(self):
        attachments = _normalize_ehelsestandard_attachments(
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

        assert [attachment.model_dump() for attachment in attachments] == [
            {
                "title": "Standard.pdf",
                "url": "https://www.helsedirektoratet.no/guillotine/helsedir/standard.pdf",
                "file_type": "PDF",
            }
        ]
