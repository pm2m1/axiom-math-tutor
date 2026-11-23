"""Real FastAPI response shapes for POST /api/v1/answer, via
`app.dependency_overrides` on `get_workflow` — exercises the actual route
handler and response schema, not just the graph in isolation (that's
tests/test_workflow.py)."""

from fastapi.testclient import TestClient

from axiom.api.dependencies import get_workflow
from axiom.main import app
from axiom.retrieval.answer_generation import GeneratedAnswer, GenerationOutcome
from axiom.retrieval.evidence_judge import EvidenceJudgment, JudgeOutcome
from axiom.web_search import WebSearchOutcome
from axiom.workflow import build_workflow
from tests.test_workflow import _FakeGenerator, _FakeJudge, _FakeRetriever, _FakeSearchClient

client = TestClient(app)


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


def test_answer_route_rejects_empty_question() -> None:
    response = client.post("/api/v1/answer", json={"question": ""})
    assert response.status_code == 422


def test_answer_route_returns_kb_grounded_answer() -> None:
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
    assert search.calls == 0


def test_answer_route_returns_refusal_with_no_raw_errors_leaked() -> None:
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
    assert "internal provider secret" not in response.text


def test_answer_route_refuses_off_topic_question() -> None:
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
    assert retriever.calls == 0
