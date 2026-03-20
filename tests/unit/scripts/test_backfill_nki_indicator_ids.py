import pytest

from scripts.data.migration.backfill_nki_indicator_ids import _normalize_indicator_list


@pytest.mark.unit
def test_normalize_indicator_list_accepts_plain_list():
    assert _normalize_indicator_list([{"id": "1"}, {"id": "2"}]) == [{"id": "1"}, {"id": "2"}]


@pytest.mark.unit
def test_normalize_indicator_list_reads_wrapped_items():
    assert _normalize_indicator_list({"items": [{"id": "1"}]}) == [{"id": "1"}]
