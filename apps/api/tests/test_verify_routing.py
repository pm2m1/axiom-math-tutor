from fastapi.testclient import TestClient

from axiom.main import app

client = TestClient(app)


def test_verify_correct_quadratic() -> None:
    response = client.post(
        "/api/v1/verify",
        json={"equation": "x^2 - 5*x + 6 = 0", "variable": "x", "solutions": ["2", "3"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_correct"
    assert body["category"] == "quadratic"
    assert body["domain"] == "real"
    assert body["deterministic"] is True
    assert body["complete"] is True
    assert body["candidates"] == [
        {"value": "2", "valid": True},
        {"value": "3", "valid": True},
    ]


def test_verify_incorrect_linear() -> None:
    response = client.post(
        "/api/v1/verify", json={"equation": "2*x + 1 = 7", "variable": "x", "solutions": ["4"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_incorrect"
    assert body["candidates"] == [{"value": "4", "valid": False}]


def test_verify_incomplete_quadratic() -> None:
    response = client.post(
        "/api/v1/verify", json={"equation": "x^2 - 1 = 0", "variable": "x", "solutions": ["1"]}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "incomplete"


def test_verify_no_real_solutions() -> None:
    response = client.post(
        "/api/v1/verify", json={"equation": "x^2 + 1 = 0", "variable": "x", "solutions": []}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_correct"
    assert body["complete"] is True


def test_verify_unsupported_higher_degree() -> None:
    response = client.post(
        "/api/v1/verify", json={"equation": "x^3 - x = 0", "variable": "x", "solutions": ["0"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["reason"] == "unsupported_degree"


def test_verify_rejects_arbitrary_python_execution_attempt() -> None:
    response = client.post(
        "/api/v1/verify",
        json={"equation": '__import__("os") = 1', "variable": "x", "solutions": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    # The response never echoes a stack trace, exception repr, or
    # anything resembling Python's own error output.
    assert "Traceback" not in response.text
    assert "os.py" not in response.text


def test_verify_defaults_variable_to_x() -> None:
    response = client.post("/api/v1/verify", json={"equation": "2*x = 4", "solutions": ["2"]})

    assert response.status_code == 200
    assert response.json()["status"] == "verified_correct"


def test_verify_rejects_empty_equation() -> None:
    response = client.post("/api/v1/verify", json={"equation": "", "solutions": []})

    assert response.status_code == 422


def test_verify_rejects_overlong_equation() -> None:
    response = client.post(
        "/api/v1/verify", json={"equation": "x = 1" + " " * 300, "solutions": []}
    )

    assert response.status_code == 422


def test_verify_rejects_multi_character_variable() -> None:
    response = client.post(
        "/api/v1/verify", json={"equation": "2*x = 4", "variable": "xy", "solutions": []}
    )

    assert response.status_code == 422


def test_verify_rejects_too_many_solutions() -> None:
    response = client.post(
        "/api/v1/verify",
        json={"equation": "2*x = 4", "solutions": [str(i) for i in range(21)]},
    )

    assert response.status_code == 422


def test_verify_rejects_non_list_solutions() -> None:
    response = client.post("/api/v1/verify", json={"equation": "2*x = 4", "solutions": "2"})

    assert response.status_code == 422


def test_verify_makes_no_dependency_calls() -> None:
    """`/api/v1/verify` takes only its Pydantic request body — no
    `Depends(...)` parameter at all. No retrieval pipeline, judge, search
    client, generator, or feedback store is constructed to serve this
    route, so it cannot make a Meta or Tavily call by construction, not
    merely by test convention."""
    import inspect

    from axiom.api.routes import verify_equation

    parameters = inspect.signature(verify_equation).parameters
    assert list(parameters) == ["request"]
    assert parameters["request"].default is inspect.Parameter.empty


def test_existing_health_and_status_routes_still_work_after_adding_verify() -> None:
    # /api/v1/answer and /api/v1/feedback are deliberately not exercised
    # here: both have real `Depends(...)` singletons that FastAPI
    # constructs during request resolution even for a request that will
    # ultimately fail body validation (see tests/test_answer_routing.py),
    # so calling them without `app.dependency_overrides` would build the
    # real workflow/feedback store. Their own test files already cover
    # backward compatibility with the proper overrides in place.
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 200
