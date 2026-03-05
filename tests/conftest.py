"""
Root conftest.py — fixtures available to all tests.

Strategy:
- session-scope autouse fixture patches db_pool.get_connection to return None,
  preventing any real MySQL connection attempt across the entire test session.
- mock_content injects synthetic ContentItem list into content_service
  and restores it after each test to prevent state bleed.
- mock_database_service replaces database_service calls in controllers.
- client provides a session-scoped TestClient over the real FastAPI app.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.entities.content import ContentItem


# ---------------------------------------------------------------------------
# Prevent any real MySQL connection during the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def patch_db_pool():
    """
    Patch db_pool.get_connection at session scope before any test runs.
    autouse=True applies it automatically to every test.
    """
    with patch(
        "app.services.repositories.base.db_pool.get_connection",
        return_value=None,
    ):
        yield


# ---------------------------------------------------------------------------
# Synthetic content for search tests
# ---------------------------------------------------------------------------

SAMPLE_CONTENT = [
    ContentItem(
        id="001",
        title="Diabetes retningslinje",
        body="Tekst om diabetes behandling og forebygging.",
        content_type="retningslinje",
        path="/retningslinjer/diabetes",
    ),
    ContentItem(
        id="002",
        title="ADHD hos barn veileder",
        body="Veileder om ADHD diagnose og behandling av barn.",
        content_type="veileder",
        path="/veiledere/adhd",
    ),
    ContentItem(
        id="003",
        title="Hjerteinfarkt nasjonalt forløp",
        body="Nasjonalt forløp for hjerteinfarkt.",
        content_type="nasjonalt-forlop",
        path="/forlop/hjerteinfarkt",
    ),
    ContentItem(
        id="004",
        title="Psykisk helse temaside",
        body="",
        content_type="temaside",
        path="/temasider/psykisk-helse",
    ),
]


@pytest.fixture
def mock_content():
    """
    Replace content_service's in-memory lists with SAMPLE_CONTENT.
    Restores original state after each test.
    """
    from app.services.data.content_service import content_service

    orig_content = content_service.content[:]
    orig_by_id = dict(content_service.content_by_id)
    orig_by_path = dict(content_service.content_by_path)
    orig_types = set(content_service.searchable_types)

    content_service.content = list(SAMPLE_CONTENT)
    content_service.content_by_id = {item.id: item for item in SAMPLE_CONTENT}
    content_service.content_by_path = {
        item.path: item for item in SAMPLE_CONTENT if item.path
    }
    content_service.searchable_types = {item.content_type for item in SAMPLE_CONTENT}

    yield content_service

    content_service.content = orig_content
    content_service.content_by_id = orig_by_id
    content_service.content_by_path = orig_by_path
    content_service.searchable_types = orig_types


# ---------------------------------------------------------------------------
# Mock database_service for controller/route tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_database_service(mocker):
    """
    Replace database_service with a MagicMock so tests can run without a real
    database. The following methods are explicitly stubbed with sensible defaults:

    - log_search → True
    - get_search_by_id → {"query": "diabetes", "role": None}
    - log_search_results → True
    - log_click → True
    - get_logged_content_ids_for_search → set()
    - get_max_position_for_search → 0
    - is_connected → False

    All other DatabaseService methods are generic MagicMock defaults (return a
    new MagicMock on each call). Tests that need specific behaviour for those
    methods should configure mock_database_service.<method>.return_value directly.
    """
    mock_db = MagicMock()
    mock_db.log_search.return_value = True
    mock_db.get_search_by_id.return_value = {"query": "diabetes", "role": None}
    mock_db.log_search_results.return_value = True
    mock_db.log_click.return_value = True
    mock_db.get_logged_content_ids_for_search.return_value = set()
    mock_db.get_max_position_for_search.return_value = 0
    mock_db.is_connected.return_value = False

    mocker.patch("app.controllers.search_controller.database_service", mock_db)
    return mock_db


# ---------------------------------------------------------------------------
# FastAPI TestClient (shared across session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Return the FastAPI app instance."""
    from app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    """
    Synchronous TestClient over the full FastAPI app.
    Session-scoped: lifespan runs once (BM25 prebuild over empty content is safe).
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
