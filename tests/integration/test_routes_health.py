"""
Integration tests for the /health endpoint.
Uses the shared session-scoped TestClient.
"""

import pytest


@pytest.mark.integration
class TestHealthRoute:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_response_has_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data

    def test_health_content_type_is_json(self, client):
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_method_not_allowed_for_post(self, client):
        response = client.post("/health")
        assert response.status_code == 405
