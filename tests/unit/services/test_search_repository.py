from unittest.mock import MagicMock

import mysql.connector

from app.services.repositories.search_repository import SearchRepository


def _mysql_error(message: str) -> mysql.connector.Error:
    err = mysql.connector.Error(msg=message)
    err.errno = 1054
    return err


def test_log_search_falls_back_when_session_id_column_is_missing(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute.side_effect = [
        _mysql_error("Unknown column 'session_id' in 'field list'"),
        None,
    ]
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_search("sid", "kreft", role=None) is True
    assert cursor.execute.call_count == 2
    assert "session_id" in cursor.execute.call_args_list[0].args[0]
    assert "INSERT INTO search_logs (search_id, query, role)" in cursor.execute.call_args_list[1].args[0]
    conn.commit.assert_called_once()


def test_log_search_results_falls_back_when_type_match_column_is_missing(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute.side_effect = [
        _mysql_error("Unknown column 'type_match' in 'field list'"),
        None,
        None,
    ]
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
    assert cursor.execute.call_count == 3
    assert "type_match" in cursor.execute.call_args_list[0].args[0]
    assert "semantic_score" in cursor.execute.call_args_list[1].args[0]
    conn.commit.assert_called_once()


def test_log_search_results_skips_content_stats_when_table_is_missing(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor

    missing_table = mysql.connector.Error(msg="Table 'helsedir_ai.content_stats' doesn't exist")
    missing_table.errno = 1146

    cursor.execute.side_effect = [
        None,
        missing_table,
    ]
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
            }
        ],
    ) is True
    conn.commit.assert_called_once()
