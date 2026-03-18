import pytest

from scripts.data.migration.backfill_generisk_normerende_enheter import (
    _build_synthetic_gne_links,
    _find_best_parent,
    _plan_backfill,
    _resolve_root_publication,
)


@pytest.mark.unit
def test_find_best_parent_prefers_nearest_path_prefix():
    veileder = {
        "id": "guide-1",
        "tittel": "Veileder",
        "info_type": "veileder-lov-forskrift",
        "path": "/veiledere/test",
        "links": [],
    }
    chapter = {
        "id": "chapter-1",
        "tittel": "Gjennomfore",
        "info_type": "kapittel",
        "path": "/veiledere/test/gjennomfore",
        "links": [],
    }

    parent = _find_best_parent(
        "/veiledere/test/gjennomfore/7a-underpunkt",
        {
            veileder["path"]: veileder,
            chapter["path"]: chapter,
        },
    )

    assert parent == chapter


@pytest.mark.unit
def test_resolve_root_publication_falls_back_to_top_ancestor():
    guide = {
        "id": "guide-1",
        "tittel": "Veileder",
        "info_type": "veileder-lov-forskrift",
        "path": "/veiledere/test",
        "links": [],
    }
    chapter = {
        "id": "chapter-1",
        "tittel": "Gjennomfore",
        "info_type": "kapittel",
        "path": "/veiledere/test/gjennomfore",
        "links": [
            {"rel": "forelder", "type": "veileder-lov-forskrift", "id": "guide-1", "tittel": "Veileder"}
        ],
    }

    root = _resolve_root_publication(
        chapter,
        {
            "guide-1": guide,
            "chapter-1": chapter,
        },
    )

    assert root == guide


@pytest.mark.unit
def test_build_synthetic_gne_links_replaces_self_referential_navigation():
    item = {
        "links": [
            {"rel": "root", "type": "generisk-normerende-enhet", "id": "gne-1", "tittel": "Seg selv"},
            {"rel": "publikasjon", "type": "generisk-normerende-enhet", "id": "gne-1", "tittel": "Seg selv"},
            {"rel": "relatert", "type": "ekstern", "href": "https://example.com"},
        ]
    }
    parent = {"id": "chapter-1", "tittel": "Kapittel", "info_type": "kapittel"}
    root = {"id": "guide-1", "tittel": "Veileder", "info_type": "veileder-lov-forskrift"}

    links = _build_synthetic_gne_links(item, parent, root)

    assert links == [
        {"rel": "relatert", "type": "ekstern", "href": "https://example.com"},
        {"rel": "forelder", "type": "kapittel", "tittel": "Kapittel", "id": "chapter-1"},
        {"rel": "root", "type": "veileder-lov-forskrift", "tittel": "Veileder", "id": "guide-1"},
        {"rel": "publikasjon", "type": "veileder-lov-forskrift", "tittel": "Veileder", "id": "guide-1"},
    ]


@pytest.mark.unit
def test_plan_backfill_links_gne_to_chapter_and_parent_back_to_gne():
    existing_rows = [
        {
            "id": "guide-1",
            "tittel": "Veileder",
            "info_type": "veileder-lov-forskrift",
            "path": "/veiledere/test",
            "links": [],
        },
        {
            "id": "chapter-1",
            "tittel": "Gjennomfore",
            "info_type": "kapittel",
            "path": "/veiledere/test/gjennomfore",
            "links": [
                {"rel": "forelder", "type": "veileder-lov-forskrift", "id": "guide-1", "tittel": "Veileder"}
            ],
        },
    ]
    gne_items = [
        {
            "id": "gne-1",
            "tittel": "7a. Underpunkt",
            "tekniskeData": {"infoType": "generisk-normerende-enhet"},
            "url": "https://www.helsedirektoratet.no/veiledere/test/gjennomfore/7a-underpunkt",
            "links": [
                {
                    "rel": "root",
                    "type": "generisk-normerende-enhet",
                    "href": "https://api.helsedirektoratet.no/innhold/generisk-normerende-enheter/gne-1",
                }
            ],
            "data": {"praktisk": "<p>Praktisk</p>", "rasjonale": "<p>Rasjonale</p>"},
            "tekst": "<p>Brødtekst</p>",
        }
    ]

    items_to_upsert, link_updates, skipped = _plan_backfill(existing_rows, gne_items)

    assert skipped == []
    assert len(items_to_upsert) == 1
    assert items_to_upsert[0]["path"] == "/veiledere/test/gjennomfore/7a-underpunkt"
    assert items_to_upsert[0]["info_type"] == "generisk-normerende-enhet"
    assert items_to_upsert[0]["links"] == [
        {"rel": "forelder", "type": "kapittel", "tittel": "Gjennomfore", "id": "chapter-1"},
        {"rel": "root", "type": "veileder-lov-forskrift", "tittel": "Veileder", "id": "guide-1"},
        {"rel": "publikasjon", "type": "veileder-lov-forskrift", "tittel": "Veileder", "id": "guide-1"},
    ]
    assert link_updates["chapter-1"][-1] == {
        "rel": "barn",
        "type": "generisk-normerende-enhet",
        "tittel": "7a. Underpunkt",
        "id": "gne-1",
    }
