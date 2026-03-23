"""
Unit tests for ContentLink and ContentItem entities.

These tests need no fixtures — entities are pure Pydantic models.
"""

import pytest
from pydantic import ValidationError

from app.entities.content import ContentItem, ContentLink, AnbefalingFields


@pytest.mark.unit
class TestContentLink:
    def test_valid_internal_link_with_id(self):
        link = ContentLink(rel="barn", type="kapittel", id="abc-123")
        assert link.id == "abc-123"
        assert link.href is None

    def test_valid_external_link_with_href(self):
        link = ContentLink(
            rel="publikasjon",
            type="artikkel",
            href="https://example.com/innhold/a/1234-5678",
        )
        assert link.href == "https://example.com/innhold/a/1234-5678"
        assert link.id is None

    def test_missing_both_id_and_href_raises(self):
        with pytest.raises(ValidationError, match="must have either"):
            ContentLink(rel="barn", type="kapittel")

    def test_both_id_and_href_raises(self):
        with pytest.raises(ValidationError, match="cannot have both"):
            ContentLink(
                rel="barn",
                type="kapittel",
                id="abc",
                href="https://example.com",
            )

    def test_whitespace_only_id_treated_as_missing(self):
        with pytest.raises(ValidationError, match="must have either"):
            ContentLink(rel="barn", type="kapittel", id="   ")

    def test_whitespace_only_href_treated_as_missing(self):
        with pytest.raises(ValidationError, match="must have either"):
            ContentLink(rel="barn", type="kapittel", href="   ")

    def test_optional_fields_default_to_none(self):
        link = ContentLink(rel="barn", type="kapittel", id="x")
        assert link.tittel is None
        assert link.path is None
        assert link.strukturId is None

    def test_tittel_is_stored(self):
        link = ContentLink(rel="barn", type="kapittel", id="x", tittel="Overskrift")
        assert link.tittel == "Overskrift"


@pytest.mark.unit
class TestContentItem:
    def test_info_type_property_aliases_content_type(self):
        item = ContentItem(id="1", title="T", body="B", content_type="retningslinje")
        assert item.info_type == "retningslinje"

    def test_target_groups_defaults_to_empty_list(self):
        item = ContentItem(id="1", title="T", body="B", content_type="veileder")
        assert item.target_groups == []
        assert item.has_text_content is False  # defaults to False; caller must set explicitly

    def test_links_defaults_to_empty_list(self):
        item = ContentItem(id="1", title="T", body="B", content_type="veileder")
        assert item.links == []

    def test_path_defaults_to_none(self):
        item = ContentItem(id="1", title="T", body="B", content_type="faktaark")
        assert item.path is None
        assert item.nki_indicator_id is None

    def test_anbefaling_fields_defaults_to_none(self):
        item = ContentItem(id="1", title="T", body="B", content_type="anbefaling")
        assert item.anbefaling_fields is None
        assert item.document_url is None
        assert item.is_pdf_only is False

    def test_content_item_with_full_fields(self):
        from datetime import datetime

        link = ContentLink(rel="forelder", type="retningslinje", id="parent-001")
        anbefaling = AnbefalingFields(styrke="A", rasjonale="Fordi det virker.")
        pub_date = datetime(2023, 1, 15, 10, 0, 0)
        review_date = datetime(2024, 6, 20, 14, 30, 0)
        item = ContentItem(
            id="full-001",
            title="Full item",
            body="Mye tekst.",
            content_type="anbefaling",
            path="/anbefalinger/full-001",
            nki_indicator_id="0003-0010-330",
            forst_publisert=pub_date,
            sist_faglig_oppdatert=review_date,
            role_tags=["lege", "sykepleier"],
            links=[link],
            has_text_content=True,
            anbefaling_fields=anbefaling,
        )
        assert item.id == "full-001"
        assert len(item.links) == 1
        assert item.target_groups == ["lege", "sykepleier"]
        assert item.anbefaling_fields.styrke == "A"
        assert item.nki_indicator_id == "0003-0010-330"
        assert item.has_text_content is True
        assert item.forst_publisert == pub_date
        assert item.sist_faglig_oppdatert == review_date

    def test_pdf_only_item_sets_is_pdf_only(self):
        item = ContentItem(
            id="pdf-001",
            title="PDF item",
            body="",
            content_type="rapport",
            document_url="https://helsedirektoratet.no/test.pdf",
        )
        assert item.has_text_content is False
        assert item.is_pdf_only is True


@pytest.mark.unit
class TestAnbefalingFields:
    def test_all_fields_optional(self):
        anbefaling = AnbefalingFields()
        assert anbefaling.praktisk is None
        assert anbefaling.rasjonale is None
        assert anbefaling.styrke is None

    def test_fields_stored_correctly(self):
        anbefaling = AnbefalingFields(
            styrke="B",
            rasjonale="Evidensbasert.",
            fordeler_ulemper="Mer fordel enn ulempe.",
        )
        assert anbefaling.styrke == "B"
        assert anbefaling.rasjonale == "Evidensbasert."
        assert anbefaling.fordeler_ulemper == "Mer fordel enn ulempe."
