from fastapi.testclient import TestClient

from axiom.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reports_service_name() -> None:
    response = client.get("/api/v1/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "axiom-api"
    assert "environment" in body
