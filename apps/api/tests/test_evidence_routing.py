from fastapi.testclient import TestClient

from axiom.api.dependencies import (
    RetrievalPipeline,
    get_evidence_judge,
    get_retrieval_pipeline,
    get_web_search_client,
)
from axiom.knowledge.models import KnowledgeChunk
from axiom.main import app
from axiom.retrieval.evidence_judge import EvidenceJudgment, JudgeOutcome
from axiom.retrieval.models import RetrievalResult
from axiom.web_search import WebSearchOutcome, WebSource

client = TestClient(app)


def _chunk(chunk_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="Doc title",
        section="Section",
        subsection=None,
        text=f"text for {chunk_id}",
        source_url="https://example.org/doc",
        source_name="Test",
        license="CC BY-SA 4.0",
        topic="equations",
        subtopic=None,
    )


class _FakeRetriever:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._chunk_ids = chunk_ids

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=_chunk(cid), score=1.0, rank=i)
            for i, cid in enumerate(self._chunk_ids[:limit], start=1)
        ]


class _FakeJudge:
    def __init__(self, outcome: JudgeOutcome) -> None:
        self._outcome = outcome
        self.calls = 0

    def judge(self, query: str, chunk_ids: list[str], chunk_texts: list[str]) -> JudgeOutcome:
        self.calls += 1
        return self._outcome


class _FakeSearchClient:
    def __init__(self, outcome: WebSearchOutcome) -> None:
        self._outcome = outcome
        self.calls = 0

    def search(self, question: str) -> WebSearchOutcome:
        self.calls += 1
        return self._outcome


def _sufficient_outcome(chunk_ids: list[str]) -> JudgeOutcome:
    judgment = EvidenceJudgment(
        verdict="sufficient",
        coverage="full",
        supporting_chunk_ids=chunk_ids,
        reason_code="complete_support",
        reason="test reason",
    )
    return JudgeOutcome(judgment, None, None, 0.1)


def _insufficient_outcome() -> JudgeOutcome:
    judgment = EvidenceJudgment(
        verdict="insufficient",
        coverage="none",
        supporting_chunk_ids=[],
        reason_code="no_relevant_evidence",
        reason="test reason",
    )
    return JudgeOutcome(judgment, None, None, 0.1)


def _override(pipeline=None, judge=None, search_client=None) -> None:
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_evidence_judge] = lambda: judge
    app.dependency_overrides[get_web_search_client] = lambda: search_client


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_existing_routes_unaffected() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 200


def test_rejects_empty_question() -> None:
    response = client.post("/api/v1/question", json={"question": ""})
    assert response.status_code == 422


def test_rejects_overlong_question() -> None:
    response = client.post("/api/v1/question", json={"question": "x" * 501})
    assert response.status_code == 422


def test_kb_sufficient_returns_cited_sources_and_makes_no_search_call() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a", "b", "c"]))
    judge = _FakeJudge(_sufficient_outcome(["a", "b"]))
    search_client = _FakeSearchClient(WebSearchOutcome([], None))
    _override(pipeline=pipeline, judge=judge, search_client=search_client)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "kb"
    assert {s["chunk_id"] for s in body["kb_sources"]} == {"a", "b"}
    assert body["web_sources"] == []
    assert search_client.calls == 0


def test_kb_insufficient_triggers_exactly_one_search_call() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a", "b"]))
    judge = _FakeJudge(_insufficient_outcome())
    web_results = [WebSource(title="T", url="https://example.org", snippet="snippet")]
    search_client = _FakeSearchClient(WebSearchOutcome(web_results, None))
    _override(pipeline=pipeline, judge=judge, search_client=search_client)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "web_search"
    assert body["kb_sources"] == []
    assert len(body["web_sources"]) == 1
    assert body["web_sources"][0]["url"] == "https://example.org"
    assert search_client.calls == 1


def test_empty_search_results_handled_explicitly_not_as_error() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a"]))
    judge = _FakeJudge(_insufficient_outcome())
    search_client = _FakeSearchClient(WebSearchOutcome([], None))
    _override(pipeline=pipeline, judge=judge, search_client=search_client)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "web_search"
    assert body["web_sources"] == []
    assert body["detail"] == "no web search results found"
    assert search_client.calls == 1


def test_judge_failure_is_dependency_failure_and_never_searches() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a"]))
    judge = _FakeJudge(JudgeOutcome(None, "some provider error", None, 0.1))
    search_client = _FakeSearchClient(WebSearchOutcome([], None))
    _override(pipeline=pipeline, judge=judge, search_client=search_client)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "unavailable"
    assert body["reason"] == "judge_error"
    assert "some provider error" not in response.text
    assert search_client.calls == 0


def test_judge_not_configured_is_dependency_failure_and_never_searches() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a"]))
    search_client = _FakeSearchClient(WebSearchOutcome([], None))
    _override(pipeline=pipeline, judge=None, search_client=search_client)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "unavailable"
    assert body["reason"] == "judge_error"
    assert search_client.calls == 0


def test_mcp_failure_is_dependency_failure() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a"]))
    judge = _FakeJudge(_insufficient_outcome())
    search_client = _FakeSearchClient(WebSearchOutcome([], "mcp connection failed: secret=abc123"))
    _override(pipeline=pipeline, judge=judge, search_client=search_client)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "unavailable"
    assert body["reason"] == "search_error"
    assert "secret=abc123" not in response.text


def test_search_not_configured_is_dependency_failure() -> None:
    pipeline = RetrievalPipeline(hybrid=_FakeRetriever(["a"]))
    judge = _FakeJudge(_insufficient_outcome())
    _override(pipeline=pipeline, judge=judge, search_client=None)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "unavailable"
    assert body["reason"] == "search_error"


def test_retrieval_unavailable_is_dependency_failure() -> None:
    _override(pipeline=None, judge=_FakeJudge(_insufficient_outcome()), search_client=None)

    response = client.post("/api/v1/question", json={"question": "What is x?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "unavailable"
    assert body["reason"] == "retrieval_error"
