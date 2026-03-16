from unittest.mock import MagicMock

import mysql.connector

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


def test_log_search_returns_false_when_connection_is_missing(mocker):
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=None,
    )

    repo = SearchRepository()

    assert repo.log_search("sid", "kreft", role=None) is False


def test_log_search_returns_false_when_insert_fails(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute.side_effect = mysql.connector.Error(msg="boom")
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_search("sid", "kreft", role=None) is False
    conn.commit.assert_not_called()


def test_log_search_results_returns_false_when_insert_fails(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute.side_effect = mysql.connector.Error(msg="boom")
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_search_results("sid", [{"content_id": "abc", "position": 1}]) is False
    conn.commit.assert_not_called()


def test_log_search_results_empty_list_is_no_op(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_search_results("sid", []) is True
    cursor.execute.assert_not_called()
    conn.commit.assert_called_once()


def test_log_click_returns_false_when_insert_fails(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (1,)
    cursor.execute.side_effect = [None, mysql.connector.Error(msg="boom")]
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_click("sid", "abc") is False
    conn.commit.assert_not_called()


def test_log_click_without_logged_position_uses_none_position(mocker):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = None
    mocker.patch(
        "app.services.repositories.search_repository.db_pool.get_connection",
        return_value=conn,
    )

    repo = SearchRepository()

    assert repo.log_click("sid", "abc") is True
    assert cursor.execute.call_count == 2
    assert cursor.execute.call_args_list[1].args[1] == ("sid", "abc", None)
    conn.commit.assert_called_once()
