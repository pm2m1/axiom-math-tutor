from pathlib import Path

from fastapi.testclient import TestClient

from axiom.api.dependencies import get_mastery_store
from axiom.main import app
from axiom.mastery import MasteryStore

client = TestClient(app)


def _override_mastery_store(tmp_path: Path) -> MasteryStore:
    store = MasteryStore(tmp_path / "mastery.sqlite3")
    app.dependency_overrides[get_mastery_store] = lambda: store
    return store


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_next_problem_defaults_to_easy_linear_for_a_new_learner(tmp_path: Path) -> None:
    _override_mastery_store(tmp_path)

    response = client.post("/api/v1/practice/next", json={"learner_id": "learner-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["skill"] == "linear_equations"
    assert body["difficulty"] == "easy"
    assert "=" in body["equation"]
    assert body["variable"] == "x"


def test_next_problem_escalates_once_mastery_improves(tmp_path: Path) -> None:
    store = _override_mastery_store(tmp_path)
    for _ in range(5):
        store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)

    response = client.post("/api/v1/practice/next", json={"learner_id": "learner-1"})

    assert response.status_code == 200
    assert response.json()["skill"] == "quadratic_equations"


def test_submit_correct_answer_updates_mastery_upward(tmp_path: Path) -> None:
    _override_mastery_store(tmp_path)

    response = client.post(
        "/api/v1/practice/submit",
        json={
            "learner_id": "learner-1",
            "skill": "linear_equations",
            "equation": "2*x = 8",
            "solutions": ["4"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_correct"
    assert body["correct"] is True
    assert body["mastery_estimate"] > 0.5


def test_submit_incorrect_answer_updates_mastery_downward(tmp_path: Path) -> None:
    _override_mastery_store(tmp_path)

    response = client.post(
        "/api/v1/practice/submit",
        json={
            "learner_id": "learner-1",
            "skill": "linear_equations",
            "equation": "2*x = 8",
            "solutions": ["5"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_incorrect"
    assert body["correct"] is False
    assert body["mastery_estimate"] < 0.5


def test_submit_unsupported_equation_does_not_move_mastery(tmp_path: Path) -> None:
    store = _override_mastery_store(tmp_path)

    response = client.post(
        "/api/v1/practice/submit",
        json={
            "learner_id": "learner-1",
            "skill": "linear_equations",
            "equation": "x^3 = 8",
            "solutions": ["2"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["correct"] is None
    assert store.get_mastery("learner-1", "linear_equations") == 0.5


def test_mastery_summary_reports_both_skills_for_a_new_learner(tmp_path: Path) -> None:
    _override_mastery_store(tmp_path)

    response = client.post("/api/v1/mastery/summary", json={"learner_id": "learner-1"})

    assert response.status_code == 200
    body = response.json()
    skills = {s["skill"]: s["estimate"] for s in body["skills"]}
    assert skills == {"linear_equations": 0.5, "quadratic_equations": 0.5}


def test_mastery_summary_reflects_recorded_attempts(tmp_path: Path) -> None:
    store = _override_mastery_store(tmp_path)
    store.record_attempt("learner-1", "quadratic_equations", verified=True, correct=True)

    response = client.post("/api/v1/mastery/summary", json={"learner_id": "learner-1"})

    body = response.json()
    skills = {s["skill"]: s["estimate"] for s in body["skills"]}
    assert skills["quadratic_equations"] > 0.5
    assert skills["linear_equations"] == 0.5


def test_mastery_is_isolated_per_learner(tmp_path: Path) -> None:
    store = _override_mastery_store(tmp_path)
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)

    response = client.post("/api/v1/mastery/summary", json={"learner_id": "learner-2"})

    skills = {s["skill"]: s["estimate"] for s in response.json()["skills"]}
    assert skills["linear_equations"] == 0.5


def test_practice_rejects_empty_learner_id(tmp_path: Path) -> None:
    _override_mastery_store(tmp_path)

    response = client.post("/api/v1/practice/next", json={"learner_id": ""})

    assert response.status_code == 422


def test_submit_rejects_too_many_hints_used(tmp_path: Path) -> None:
    _override_mastery_store(tmp_path)

    response = client.post(
        "/api/v1/practice/submit",
        json={
            "learner_id": "learner-1",
            "skill": "linear_equations",
            "equation": "2*x = 8",
            "solutions": ["4"],
            "hints_used": 99,
        },
    )

    assert response.status_code == 422


def test_practice_routes_make_no_network_dependency_calls(tmp_path: Path) -> None:
    """The only `Depends(...)` on these routes is the local mastery
    store — no Meta/Tavily-backed dependency is ever constructed."""
    import inspect

    from axiom.api.routes import mastery_summary, next_practice_problem, submit_practice_answer

    for fn in (next_practice_problem, submit_practice_answer, mastery_summary):
        parameters = inspect.signature(fn).parameters
        assert set(parameters) == {"request", "mastery_store"}


def test_existing_routes_still_work_after_adding_practice() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 200
