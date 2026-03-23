import asyncio

import pytest

from app.entities.content import ContentItem
from app.exceptions.helsedir import HelseDirectorateAPIError
from app.services.statistics.nki_statistics_service import NKIStatisticsService


def _sample_indicator():
    return {
        "id": "0003-0010-330",
        "tittel": "Avtalevarighet på kommunenes fastleger",
        "tekst": "Beskrivelse av indikatoren",
        "attachments": [
            {
                "title": "Metodebeskrivelse",
                "fileUri": "/guillotine/helsedir/metode.pdf",
            }
        ],
    }


def _sample_dataset():
    return {
        "ColumnNames": [
            "MeasureName",
            "LocationName",
            "ParentName",
            "Value",
            "TimeFrom",
            "TimeTo",
            "PeriodType",
        ],
        "AttachmentDataRows": [
            [
                "Gjennomsnittlig avtalevarighet år",
                "Kommunegruppe 2",
                "Norge",
                "5,75",
                "2018-01-01T00:00:00",
                "2018-12-31T00:00:00",
                "year",
            ],
            [
                "Gjennomsnittlig avtalevarighet år",
                "Kommunegruppe 1",
                "Norge",
                "6.15",
                "2017-01-01T00:00:00",
                "2017-12-31T00:00:00",
                "year",
            ],
            [
                "Median avtalevarighet år",
                "Kommunegruppe 1",
                "Norge",
                "4",
                "2017-01-01T00:00:00",
                "2017-12-31T00:00:00",
                "year",
            ],
        ],
    }


@pytest.mark.unit
def test_get_statistics_for_content_normalizes_dataset_and_dimensions(mocker):
    service = NKIStatisticsService()
    content = ContentItem(
        id="content-1",
        title="Fastleger",
        body="",
        content_type="statistikk",
        nki_indicator_id="0003-0010-330",
    )

    detail_mock = mocker.patch(
        "app.services.statistics.nki_statistics_service.helsedir_api_service.get_nki_quality_indicator_by_id_async",
        new=mocker.AsyncMock(return_value=_sample_indicator()),
    )
    data_mock = mocker.patch(
        "app.services.statistics.nki_statistics_service.helsedir_api_service.get_nki_quality_indicator_data_async",
        new=mocker.AsyncMock(return_value=_sample_dataset()),
    )

    result = asyncio.run(service.get_statistics_for_content(content))

    assert result["has_statistics"] is True
    assert result["statistics_status"] == "available"
    assert result["content_id"] == "content-1"
    assert result["nki_indicator_id"] == "0003-0010-330"
    assert result["dimensions"] == {
        "measures": [
            "Gjennomsnittlig avtalevarighet år",
            "Median avtalevarighet år",
        ],
        "locations": ["Kommunegruppe 1", "Kommunegruppe 2"],
        "parent_locations": ["Norge"],
        "period_types": ["year"],
    }
    assert result["attachments"] == [
        {
            "title": "Metodebeskrivelse",
            "url": "https://www.helsedirektoratet.no/guillotine/helsedir/metode.pdf",
        }
    ]
    assert result["series"][0]["name"] == "Gjennomsnittlig avtalevarighet år"
    assert result["series"][0]["points"] == [
        {
            "x": "2017-12-31",
            "y": 6.15,
            "location": "Kommunegruppe 1",
            "parent_location": "Norge",
            "time_from": "2017-01-01",
            "time_to": "2017-12-31",
            "period_type": "year",
        },
        {
            "x": "2018-12-31",
            "y": 5.75,
            "location": "Kommunegruppe 2",
            "parent_location": "Norge",
            "time_from": "2018-01-01",
            "time_to": "2018-12-31",
            "period_type": "year",
        },
    ]

    second = asyncio.run(service.get_statistics_for_content(content))

    assert second["series"] == result["series"]
    detail_mock.assert_awaited_once()
    data_mock.assert_awaited_once()


@pytest.mark.unit
def test_get_statistics_for_content_returns_not_configured_when_indicator_id_missing():
    service = NKIStatisticsService()
    content = ContentItem(
        id="content-1",
        title="Fastleger",
        body="",
        content_type="statistikk",
    )

    result = asyncio.run(service.get_statistics_for_content(content))

    assert result["has_statistics"] is False
    assert result["statistics_status"] == "not_configured"


@pytest.mark.unit
def test_get_statistics_for_content_returns_unavailable_on_api_error(mocker):
    service = NKIStatisticsService()
    content = ContentItem(
        id="content-1",
        title="Fastleger",
        body="",
        content_type="statistikk",
        nki_indicator_id="0003-0010-330",
    )

    mocker.patch(
        "app.services.statistics.nki_statistics_service.helsedir_api_service.get_nki_quality_indicator_by_id_async",
        new=mocker.AsyncMock(side_effect=HelseDirectorateAPIError("boom")),
    )
    mocker.patch(
        "app.services.statistics.nki_statistics_service.helsedir_api_service.get_nki_quality_indicator_data_async",
        new=mocker.AsyncMock(return_value=None),
    )

    result = asyncio.run(service.get_statistics_for_content(content))

    assert result["has_statistics"] is False
    assert result["statistics_status"] == "unavailable"
