"""Real FastAPI response shapes for POST /api/v1/answer, via
`app.dependency_overrides` on `get_workflow` — exercises the actual route
handler and response schema, not just the graph in isolation (that's
tests/test_workflow.py). `get_feedback_store` is also overridden in every
test with a temp-path store, so normal test runs never touch the real
`.data/feedback.sqlite3` file."""

from pathlib import Path

from fastapi.testclient import TestClient

from axiom.api.dependencies import get_feedback_store, get_workflow
from axiom.feedback import FeedbackStore
from axiom.main import app
from axiom.retrieval.answer_generation import GeneratedAnswer, GenerationOutcome
from axiom.retrieval.evidence_judge import EvidenceJudgment, JudgeOutcome
from axiom.web_search import WebSearchOutcome
from axiom.workflow import build_workflow
from tests.test_workflow import _FakeGenerator, _FakeJudge, _FakeRetriever, _FakeSearchClient

client = TestClient(app)


def _override_feedback_store(tmp_path: Path) -> FeedbackStore:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    app.dependency_overrides[get_feedback_store] = lambda: store
    return store


def _sufficient(chunk_ids: list[str]) -> JudgeOutcome:
    judgment = EvidenceJudgment(
        verdict="sufficient",
        coverage="full",
        supporting_chunk_ids=chunk_ids,
        reason_code="complete_support",
        reason="test reason",
    )
    return JudgeOutcome(judgment, None, None, 0.1)


def _good_answer(cited_source_ids: list[str]) -> GenerationOutcome:
    answer = GeneratedAnswer(
        abstain=False,
        explanation="A clear explanation.",
        abstain_reason=None,
        cited_source_ids=cited_source_ids,
    )
    return GenerationOutcome(answer, None, None, 0.1)


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_existing_question_route_still_works_unmodified() -> None:
    response = client.post("/api/v1/question", json={"question": "What is x?"})
    assert response.status_code == 200
    assert response.json()["route"] == "unavailable"


def test_health_and_status_routes_unaffected() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 200


def test_answer_route_rejects_empty_question(tmp_path: Path) -> None:
    # FastAPI resolves a route's Depends() parameters before a body
    # validation failure is raised, so the feedback-store dependency
    # must still be overridden here — otherwise this 422 case would
    # silently construct the real singleton against the real DB path.
    _override_feedback_store(tmp_path)
    response = client.post("/api/v1/answer", json={"question": ""})
    assert response.status_code == 422


def test_answer_route_returns_kb_grounded_answer(tmp_path: Path) -> None:
    _override_feedback_store(tmp_path)
    retriever = _FakeRetriever(["a"])
    judge = _FakeJudge(_sufficient(["a"]))
    search = _FakeSearchClient(WebSearchOutcome([], None))
    generator = _FakeGenerator(_good_answer(["a"]))
    graph = build_workflow(retriever, judge, search, generator)
    app.dependency_overrides[get_workflow] = lambda: graph

    response = client.post(
        "/api/v1/answer", json={"question": "What is the symmetric property of equality?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "kb"
    assert body["answer"] is not None
    assert body["citations"][0]["source_type"] == "kb"
    assert body["answer_id"] is not None
    assert search.calls == 0


def test_answer_route_returns_refusal_with_no_raw_errors_leaked(tmp_path: Path) -> None:
    _override_feedback_store(tmp_path)
    retriever = _FakeRetriever(["a"])
    judge = _FakeJudge(JudgeOutcome(None, "internal provider secret", None, 0.1))
    search = _FakeSearchClient(WebSearchOutcome([], None))
    generator = _FakeGenerator(_good_answer(["a"]))
    graph = build_workflow(retriever, judge, search, generator)
    app.dependency_overrides[get_workflow] = lambda: graph

    response = client.post("/api/v1/answer", json={"question": "What is the quadratic formula?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["reason"] == "judge_error"
    assert body["answer_id"] is None
    assert "internal provider secret" not in response.text


def test_answer_route_refuses_off_topic_question(tmp_path: Path) -> None:
    _override_feedback_store(tmp_path)
    retriever = _FakeRetriever(["a"])
    judge = _FakeJudge(_sufficient(["a"]))
    search = _FakeSearchClient(WebSearchOutcome([], None))
    generator = _FakeGenerator(_good_answer(["a"]))
    graph = build_workflow(retriever, judge, search, generator)
    app.dependency_overrides[get_workflow] = lambda: graph

    response = client.post("/api/v1/answer", json={"question": "Tell me a joke"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["reason"] == "off_topic"
    assert body["answer_id"] is None
    assert retriever.calls == 0


# --- Phase 1L: feedback, attached to a real answer_id from /api/v1/answer ---


def test_feedback_can_be_submitted_for_a_real_answer(tmp_path: Path) -> None:
    _override_feedback_store(tmp_path)
    retriever = _FakeRetriever(["a"])
    judge = _FakeJudge(_sufficient(["a"]))
    search = _FakeSearchClient(WebSearchOutcome([], None))
    generator = _FakeGenerator(_good_answer(["a"]))
    graph = build_workflow(retriever, judge, search, generator)
    app.dependency_overrides[get_workflow] = lambda: graph

    answer = client.post(
        "/api/v1/answer", json={"question": "What is the symmetric property of equality?"}
    ).json()

    response = client.post(
        "/api/v1/feedback",
        json={"answer_id": answer["answer_id"], "rating": "up", "correction": "good answer"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_id"] == answer["answer_id"]
    assert body["rating"] == "up"
    assert body["answer_status"] == "kb"
    assert search.calls == 0
    assert generator.calls == 1  # feedback submission made no extra generation call


def test_feedback_for_unknown_answer_id_returns_404(tmp_path: Path) -> None:
    _override_feedback_store(tmp_path)

    response = client.post("/api/v1/feedback", json={"answer_id": "not-a-real-id", "rating": "up"})

    assert response.status_code == 404


def test_feedback_rejects_invalid_rating(tmp_path: Path) -> None:
    store = _override_feedback_store(tmp_path)
    answer_id = store.record_answer("kb")

    response = client.post("/api/v1/feedback", json={"answer_id": answer_id, "rating": "sideways"})

    assert response.status_code == 422


def test_feedback_rejects_overlong_correction(tmp_path: Path) -> None:
    store = _override_feedback_store(tmp_path)
    answer_id = store.record_answer("kb")

    response = client.post(
        "/api/v1/feedback",
        json={"answer_id": answer_id, "rating": "up", "correction": "x" * 1001},
    )

    assert response.status_code == 422


def test_feedback_rejects_empty_answer_id(tmp_path: Path) -> None:
    _override_feedback_store(tmp_path)

    response = client.post("/api/v1/feedback", json={"answer_id": "", "rating": "up"})

    assert response.status_code == 422


def test_answer_route_returns_computed_result_for_a_structured_command(tmp_path: Path) -> None:
    # The deterministic problem router (Phase 3) matches this before the
    # graph ever reaches retrieval/judge/search/generation — fakes that
    # would raise on any call still prove that via a clean 200 response.
    _override_feedback_store(tmp_path)
    judge = _FakeJudge(JudgeOutcome(None, "should never be called", None, 0.1))
    search = _FakeSearchClient(WebSearchOutcome([], None))
    generator = _FakeGenerator(_good_answer([]))
    graph = build_workflow(_FakeRetriever([]), judge, search, generator)
    app.dependency_overrides[get_workflow] = lambda: graph

    response = client.post("/api/v1/answer", json={"question": "differentiate x^2"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "computed"
    assert body["verification_method"] == "deterministic_sympy"
    assert body["solver_category"] == "derivative"
    assert "2*x" in body["answer"]
    assert body["answer_id"] is not None
    assert judge.calls == 0
    assert search.calls == 0
    assert generator.calls == 0


def test_answer_route_kb_answer_reports_evidence_grounded_verification_method(
    tmp_path: Path,
) -> None:
    _override_feedback_store(tmp_path)
    retriever = _FakeRetriever(["a"])
    judge = _FakeJudge(_sufficient(["a"]))
    search = _FakeSearchClient(WebSearchOutcome([], None))
    generator = _FakeGenerator(_good_answer(["a"]))
    graph = build_workflow(retriever, judge, search, generator)
    app.dependency_overrides[get_workflow] = lambda: graph

    response = client.post(
        "/api/v1/answer", json={"question": "What is the symmetric property of equality?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "kb"
    assert body["verification_method"] == "evidence_grounded_llm"
    assert body["solver_category"] is None
