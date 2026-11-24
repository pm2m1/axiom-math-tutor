from fastapi.testclient import TestClient

from axiom.main import app

client = TestClient(app)


def test_count_ncr_computed() -> None:
    response = client.post("/api/v1/count", json={"operation": "ncr", "n": 5, "r": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "computed"
    assert body["computed"] == "10"


def test_count_ncr_verified_correct() -> None:
    response = client.post(
        "/api/v1/count",
        json={"operation": "ncr", "n": 5, "r": 2, "proposed_answer": "10"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified_correct"


def test_count_probability() -> None:
    response = client.post(
        "/api/v1/count",
        json={"operation": "probability", "n": 10, "r": 2, "favorable": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "computed"
    assert body["computed"] == "1/15"


def test_count_rejects_r_out_of_bounds() -> None:
    response = client.post("/api/v1/count", json={"operation": "ncr", "n": 5, "r": 1000})

    assert response.status_code == 422


def test_count_rejects_negative_n() -> None:
    response = client.post("/api/v1/count", json={"operation": "ncr", "n": -1, "r": 0})

    assert response.status_code == 422


def test_count_makes_no_dependency_calls() -> None:
    import inspect

    from axiom.api.routes import count_combinatorics

    parameters = inspect.signature(count_combinatorics).parameters
    assert list(parameters) == ["request"]
    assert parameters["request"].default is inspect.Parameter.empty


def test_existing_routes_still_work_after_adding_count() -> None:
    assert client.get("/health").status_code == 200
    assert (
        client.post(
            "/api/v1/verify-derivative", json={"expression": "x^2", "proposed_derivative": "2*x"}
        ).status_code
        == 200
    )
