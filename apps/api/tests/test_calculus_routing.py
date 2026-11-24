from fastapi.testclient import TestClient

from axiom.main import app

client = TestClient(app)


def test_verify_derivative_correct() -> None:
    response = client.post(
        "/api/v1/verify-derivative",
        json={"expression": "x^2", "variable": "x", "proposed_derivative": "2*x"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_correct"
    assert body["operation"] == "derivative"
    assert body["deterministic"] is True
    assert body["computed"] == "2*x"


def test_verify_derivative_incorrect() -> None:
    response = client.post(
        "/api/v1/verify-derivative",
        json={"expression": "x^2", "proposed_derivative": "3*x"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified_incorrect"


def test_verify_derivative_trig() -> None:
    response = client.post(
        "/api/v1/verify-derivative",
        json={"expression": "sin(x)", "proposed_derivative": "cos(x)"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified_correct"


def test_verify_derivative_rejects_injection_attempt() -> None:
    response = client.post(
        "/api/v1/verify-derivative",
        json={"expression": '__import__("os")', "proposed_derivative": "1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert "Traceback" not in response.text


def test_verify_integral_correct_with_constant() -> None:
    response = client.post(
        "/api/v1/verify-integral",
        json={"expression": "2*x", "proposed_antiderivative": "x^2 + 7"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified_correct"


def test_verify_integral_incorrect() -> None:
    response = client.post(
        "/api/v1/verify-integral",
        json={"expression": "x^2", "proposed_antiderivative": "x^2"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified_incorrect"


def test_verify_derivative_rejects_empty_expression() -> None:
    response = client.post(
        "/api/v1/verify-derivative", json={"expression": "", "proposed_derivative": "1"}
    )

    assert response.status_code == 422


def test_verify_integral_rejects_multi_character_variable() -> None:
    response = client.post(
        "/api/v1/verify-integral",
        json={"expression": "2*x", "variable": "xy", "proposed_antiderivative": "x^2"},
    )

    assert response.status_code == 422


def test_calculus_routes_make_no_dependency_calls() -> None:
    import inspect

    from axiom.api.routes import verify_derivative, verify_integral

    for fn in (verify_derivative, verify_integral):
        parameters = inspect.signature(fn).parameters
        assert list(parameters) == ["request"]
        assert parameters["request"].default is inspect.Parameter.empty


def test_existing_routes_still_work_after_adding_calculus() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 200
    assert (
        client.post("/api/v1/verify", json={"equation": "2*x = 4", "solutions": ["2"]}).status_code
        == 200
    )
