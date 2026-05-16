"""Unit tests for backfill_nki_indicator_ids — indicator list normalization."""

import pytest

from scripts.data.migration.backfill_nki_indicator_ids import _normalize_indicator_list, save_matches_batch


@pytest.mark.unit
def test_normalize_indicator_list_accepts_plain_list():
    assert _normalize_indicator_list([{"id": "1"}, {"id": "2"}]) == [{"id": "1"}, {"id": "2"}]


@pytest.mark.unit
def test_normalize_indicator_list_reads_wrapped_items():
    assert _normalize_indicator_list({"items": [{"id": "1"}]}) == [{"id": "1"}]


@pytest.mark.unit
def test_save_matches_batch_rolls_back_and_reraises_on_error(mocker):
    cursor = mocker.MagicMock()
    cursor.executemany.side_effect = RuntimeError("boom")
    conn = mocker.MagicMock()
    conn.cursor.return_value = cursor
    mocker.patch(
        "scripts.data.migration.backfill_nki_indicator_ids.db_pool.get_connection",
        return_value=conn,
    )

    with pytest.raises(RuntimeError, match="boom"):
        save_matches_batch([{"indicator_id": "0003-0010-330", "content_id": "content-1"}])

    conn.rollback.assert_called_once()
