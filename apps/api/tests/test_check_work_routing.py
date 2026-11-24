from fastapi.testclient import TestClient

from axiom.main import app

client = TestClient(app)


def test_check_work_all_steps_equivalent() -> None:
    response = client.post(
        "/api/v1/check-work",
        json={
            "equation": "2*x + 3 = 11",
            "variable": "x",
            "steps": ["2*x = 8", "x = 4"],
            "solutions": ["4"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "all_steps_equivalent"
    assert body["chain_preserves_solution_set"] is True
    assert body["first_invalid_step_index"] is None
    assert len(body["steps"]) == 2
    assert body["steps"][0] == {
        "index": 1,
        "equation": "2*x = 8",
        "category": "linear",
        "transition": "equivalent",
        "detail": "this step preserves the real solution set",
        "misconception": None,
    }
    assert body["final_answer"]["status"] == "verified_correct"


def test_check_work_identifies_first_invalid_step() -> None:
    response = client.post(
        "/api/v1/check-work",
        json={"equation": "2*x + 3 = 11", "steps": ["2*x = 8", "x = 5"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "invalid_step_found"
    assert body["first_invalid_step_index"] == 2
    assert body["steps"][0]["transition"] == "equivalent"
    assert body["steps"][1]["transition"] == "incompatible"
    assert body["final_answer"] is None


def test_check_work_distinguishes_narrowing_from_equivalence() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "x^2 = 1", "steps": ["x = 1"]})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "valid_but_incomplete"
    assert body["steps"][0]["transition"] == "narrowing"
    assert body["first_invalid_step_index"] is None
    assert body["chain_preserves_solution_set"] is False


def test_check_work_flags_partial_overlap_as_provably_invalid() -> None:
    # {1,2} -> {2,3}: shares "2" but loses "1" while gaining "3" — this
    # is not a legitimate branch or extension, so it must be reported as
    # the first provably invalid step, not merely "incomplete".
    response = client.post(
        "/api/v1/check-work",
        json={"equation": "(x-1)*(x-2) = 0", "steps": ["(x-2)*(x-3) = 0"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "invalid_step_found"
    assert body["steps"][0]["transition"] == "partial_overlap"
    assert body["first_invalid_step_index"] == 1
    assert body["chain_preserves_solution_set"] is False


def test_check_work_distinguishes_broadening() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "x = 1", "steps": ["x^2 = 1"]})

    assert response.status_code == 200
    assert response.json()["steps"][0]["transition"] == "broadening"


def test_check_work_correct_final_answer_despite_invalid_step() -> None:
    response = client.post(
        "/api/v1/check-work",
        json={"equation": "2*x + 3 = 11", "steps": ["2*x = 8", "x = 5"], "solutions": ["4"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "invalid_step_found"
    assert body["final_answer"]["status"] == "verified_correct"


def test_check_work_no_final_answer_submitted() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "2*x = 8", "steps": ["x = 4"]})

    assert response.status_code == 200
    assert response.json()["final_answer"] is None


def test_check_work_empty_solutions_claims_no_real_solutions() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "x^2 + 1 = 0", "steps": [], "solutions": []}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_answer"]["status"] == "verified_correct"
    assert body["final_answer"]["complete"] is True


def test_check_work_rejects_injection_attempt_as_a_step() -> None:
    response = client.post(
        "/api/v1/check-work",
        json={"equation": "2*x = 8", "steps": ['__import__("os") = 1', "x = 4"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["steps"][0]["transition"] == "unsupported"
    assert body["steps"][1]["transition"] == "unsupported"
    assert body["status"] == "verification_incomplete"
    assert "Traceback" not in response.text
    assert "os.py" not in response.text


def test_check_work_unsupported_original_equation() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "x^3 - x = 0", "steps": ["x = 0"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["steps"] == []


def test_check_work_defaults_variable_to_x() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "2*x = 4", "steps": ["x = 2"]})

    assert response.status_code == 200
    assert response.json()["status"] == "all_steps_equivalent"


def test_check_work_rejects_empty_equation() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "", "steps": []})

    assert response.status_code == 422


def test_check_work_rejects_too_many_steps() -> None:
    response = client.post(
        "/api/v1/check-work",
        json={"equation": "2*x = 8", "steps": [f"x = {i}" for i in range(16)]},
    )

    assert response.status_code == 422


def test_check_work_rejects_overlong_step() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "2*x = 8", "steps": ["x = 4" + " " * 300]}
    )

    assert response.status_code == 422


def test_check_work_rejects_multi_character_variable() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "2*x = 4", "variable": "xy", "steps": []}
    )

    assert response.status_code == 422


def test_check_work_rejects_too_many_solutions() -> None:
    response = client.post(
        "/api/v1/check-work",
        json={"equation": "2*x = 4", "steps": [], "solutions": [str(i) for i in range(21)]},
    )

    assert response.status_code == 422


def test_check_work_rejects_non_list_steps() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "2*x = 4", "steps": "x=2"})

    assert response.status_code == 422


def test_check_work_makes_no_dependency_calls() -> None:
    """`/api/v1/check-work`, like `/api/v1/verify`, takes only its
    Pydantic request body — no `Depends(...)` parameter, so it cannot
    make a Meta or Tavily call by construction."""
    import inspect

    from axiom.api.routes import check_work

    parameters = inspect.signature(check_work).parameters
    assert list(parameters) == ["request"]
    assert parameters["request"].default is inspect.Parameter.empty


def test_existing_routes_still_work_after_adding_check_work() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 200
    assert (
        client.post("/api/v1/verify", json={"equation": "2*x = 4", "solutions": ["2"]}).status_code
        == 200
    )


def test_check_work_reports_sign_error_misconception() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "2*x = 8", "steps": ["x = -4"]})

    assert response.status_code == 200
    body = response.json()
    assert body["steps"][0]["transition"] == "incompatible"
    assert body["steps"][0]["misconception"] == "sign_error"


def test_check_work_reports_lost_branch_misconception() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "x^2 = 1", "steps": ["x = 1"]})

    assert response.status_code == 200
    assert response.json()["steps"][0]["misconception"] == "lost_solution_branch"


def test_check_work_no_hints_by_default() -> None:
    response = client.post("/api/v1/check-work", json={"equation": "2*x = 8", "steps": []})

    assert response.status_code == 200
    assert response.json()["hints"] is None


def test_check_work_hints_sliced_to_requested_level() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "2*x = 8", "steps": [], "hint_level": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["hints"]) == 2
    assert "4" not in body["hints"][0]
    assert "4" not in body["hints"][1]


def test_check_work_level_four_hint_names_the_solution() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "2*x = 8", "steps": [], "hint_level": 4}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["hints"]) == 4
    assert "4" in body["hints"][3]


def test_check_work_rejects_hint_level_out_of_range() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "2*x = 8", "steps": [], "hint_level": 5}
    )

    assert response.status_code == 422


def test_check_work_hints_none_for_unsupported_equation() -> None:
    response = client.post(
        "/api/v1/check-work", json={"equation": "x^3 = 8", "steps": [], "hint_level": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["hints"] is None
