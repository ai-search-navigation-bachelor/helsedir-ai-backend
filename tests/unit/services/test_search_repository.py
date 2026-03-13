from unittest.mock import MagicMock

from app.services.repositories.search_repository import SearchRepository


def test_log_search_uses_current_search_logs_schema(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_search("sid", "kreft", role=None) is True
    assert cursor.execute.call_count == 1
    assert "session_id" not in cursor.execute.call_args.args[0]
    conn.commit.assert_called_once()


def test_log_search_results_uses_current_results_schema(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_search_results(
        "sid",
        [
            {
                "content_id": "abc",
                "position": 1,
                "score": 0.9,
                "semantic_score": 0.8,
                "bm25_score": 0.7,
                "rrf_score": 0.6,
                "type_match": 0.5,
                "role_match": 0.4,
                "maalgruppe_match": 0,
            }
        ],
    ) is True
    assert cursor.execute.call_count == 1
    assert "type_match" not in cursor.execute.call_args.args[0]
    assert "maalgruppe_match" not in cursor.execute.call_args.args[0]
    assert "role_match" in cursor.execute.call_args.args[0]
    conn.commit.assert_called_once()


def test_log_click_only_writes_click_log(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (1,)
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_click("sid", "abc") is True
    assert cursor.execute.call_count == 2
    assert "content_stats" not in "".join(call.args[0] for call in cursor.execute.call_args_list)
    conn.commit.assert_called_once()
