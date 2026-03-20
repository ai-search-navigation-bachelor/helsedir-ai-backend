import pytest

from app.services.statistics.nki_matching import (
    extract_indicator_paths,
    find_indicator_match,
    normalize_match_text,
    plan_indicator_backfill,
)


@pytest.mark.unit
def test_extract_indicator_paths_reads_public_url_from_indicator_text():
    indicator = {
        "tekst": "Se indikator url: https://www.helsedirektoratet.no/statistikk/test-side?foo=bar",
    }

    assert extract_indicator_paths(indicator) == ["/statistikk/test-side"]


@pytest.mark.unit
def test_find_indicator_match_prefers_url_before_title():
    indicator = {
        "id": "0003-0010-330",
        "tittel": "Samme tittel",
        "tekst": "Se indikator url: https://www.helsedirektoratet.no/statistikk/unik-side",
    }
    content_rows = [
        {"id": "content-1", "tittel": "Samme tittel", "kort_tittel": None, "path": "/annen-side"},
        {"id": "content-2", "tittel": "Helt annen tittel", "kort_tittel": None, "path": "/statistikk/unik-side"},
    ]

    result = find_indicator_match(indicator, content_rows)

    assert result["strategy"] == "url"
    assert result["match"]["id"] == "content-2"


@pytest.mark.unit
def test_find_indicator_match_returns_ambiguous_when_normalized_title_hits_multiple_rows():
    indicator = {"id": "0003-0010-331", "tittel": "  Måling av  kvalitet "}
    content_rows = [
        {"id": "content-1", "tittel": "Maling av kvalitet", "kort_tittel": None, "path": "/a"},
        {"id": "content-2", "tittel": "Måling   av kvalitet", "kort_tittel": None, "path": "/b"},
    ]

    result = find_indicator_match(indicator, content_rows)

    assert result["match"] is None
    assert result["reason"] == "ambiguous_normalized_title"


@pytest.mark.unit
def test_plan_indicator_backfill_skips_existing_mapping_without_force():
    indicators = [
        {"id": "0003-0010-330", "tittel": "Malt tittel"},
        {"id": "0003-0010-331", "tittel": "Annen tittel"},
    ]
    content_rows = [
        {
            "id": "content-1",
            "tittel": "Malt tittel",
            "kort_tittel": None,
            "path": "/malt-tittel",
            "nki_indicator_id": "already-set",
        },
        {
            "id": "content-2",
            "tittel": "Annen tittel",
            "kort_tittel": None,
            "path": "/annen-tittel",
            "nki_indicator_id": None,
        },
    ]

    updates, skipped = plan_indicator_backfill(indicators, content_rows, force=False)

    assert updates == [
        {
            "content_id": "content-2",
            "indicator_id": "0003-0010-331",
            "strategy": "exact_title",
            "title": "Annen tittel",
        }
    ]
    assert any(item["reason"] == "existing_mapping" for item in skipped)


@pytest.mark.unit
def test_find_indicator_match_does_not_treat_same_row_title_and_short_title_as_ambiguous():
    indicator = {"id": "0003-0010-330", "tittel": "Lik tittel"}
    content_rows = [
        {
            "id": "content-1",
            "tittel": "Lik tittel",
            "kort_tittel": "Lik tittel",
            "path": "/lik-tittel",
        }
    ]

    result = find_indicator_match(indicator, content_rows)

    assert result["match"]["id"] == "content-1"
    assert result["strategy"] == "exact_title"


@pytest.mark.unit
def test_normalize_match_text_collapses_spacing_and_normalizes_case():
    assert normalize_match_text("  Kvalitet   På  Tvers ") == "kvalitet pa tvers"
