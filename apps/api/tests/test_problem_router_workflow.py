"""Full LangGraph execution of the new `classify_math_problem` node
(Phase 3) — exercises the real compiled graph, matching the style of
`tests/test_workflow.py`. A match must bypass retrieval/judge/search/
generation entirely (zero calls); anything unmatched must behave
identically to the pre-Phase-3 graph (covered by `test_workflow.py`,
already re-verified unchanged)."""

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.evidence_judge import EvidenceJudgment, JudgeOutcome
from axiom.retrieval.models import RetrievalResult
from axiom.web_search import WebSearchOutcome
from axiom.workflow import build_workflow, initial_state, response_from_state


def _chunk(chunk_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="Title",
        section="Section",
        subsection=None,
        text="Curriculum text.",
        source_url="https://example.org/doc",
        source_name="Test",
        license="CC BY-SA 4.0",
        topic="equations",
        subtopic=None,
    )


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        self.calls += 1
        return [RetrievalResult(chunk=_chunk("a"), score=1.0, rank=1)]


class _FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, query: str, chunk_ids: list[str], chunk_texts: list[str]):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise AssertionError("judge should never be called for a matched structured problem")


class _FakeSearchClient:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, question: str) -> WebSearchOutcome:
        self.calls += 1
        raise AssertionError("search should never be called for a matched structured problem")


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, evidence):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise AssertionError("generator should never be called for a matched structured problem")


def _graph_with_failing_fakes():  # type: ignore[no-untyped-def]
    """For asserting a match bypasses the RAG pipeline entirely — any
    call to judge/search/generate is itself a test failure."""
    retriever = _FakeRetriever()
    return build_workflow(retriever, _FakeJudge(), _FakeSearchClient(), _FakeGenerator()), retriever


class _InsufficientJudge:
    def judge(self, query: str, chunk_ids: list[str], chunk_texts: list[str]) -> JudgeOutcome:
        judgment = EvidenceJudgment(
            verdict="insufficient",
            coverage="none",
            supporting_chunk_ids=[],
            reason_code="no_relevant_evidence",
            reason="test reason",
        )
        return JudgeOutcome(judgment, None, None, 0.1)


class _EmptySearchClient:
    def search(self, question: str) -> WebSearchOutcome:
        return WebSearchOutcome([], None)


def _graph_with_passthrough_fakes():  # type: ignore[no-untyped-def]
    """For asserting an *unmatched* question reaches the normal RAG
    pipeline (not that it produces any particular answer) — real,
    non-raising fakes that resolve to a clean `"abstained"` terminus."""
    retriever = _FakeRetriever()
    return (
        build_workflow(retriever, _InsufficientJudge(), _EmptySearchClient(), _FakeGenerator()),
        retriever,
    )


def test_differentiate_command_is_computed_without_touching_the_rag_pipeline() -> None:
    graph, retriever = _graph_with_failing_fakes()

    result = graph.invoke(initial_state("differentiate x^2"))
    response = response_from_state(result)

    assert response.status == "computed"
    assert response.verification_method == "deterministic_sympy"
    assert response.solver_category == "derivative"
    assert response.answer is not None
    assert "2*x" in response.answer
    assert response.citations == []
    assert retriever.calls == 0


def test_integrate_command_is_computed() -> None:
    graph, _ = _graph_with_failing_fakes()

    result = graph.invoke(initial_state("integrate cos(x)"))
    response = response_from_state(result)

    assert response.status == "computed"
    assert response.solver_category == "integral"
    assert "sin(x)" in response.answer


def test_ncr_command_is_computed() -> None:
    graph, _ = _graph_with_failing_fakes()

    result = graph.invoke(initial_state("5 choose 2"))
    response = response_from_state(result)

    assert response.status == "computed"
    assert response.solver_category == "combinatorics"
    assert "10" in response.answer


def test_unparseable_structured_command_falls_through_to_the_rag_pipeline() -> None:
    # Matches the router's pattern syntactically, but the expression
    # itself is outside the restricted grammar — must NOT be treated as
    # computed; falls through to the existing RAG pipeline, proving it
    # did NOT short-circuit as "computed".
    graph, retriever = _graph_with_passthrough_fakes()

    result = graph.invoke(initial_state('differentiate __import__("os")'))
    response = response_from_state(result)

    assert response.status != "computed"
    assert retriever.calls == 1


def test_natural_language_question_is_unaffected_by_the_router() -> None:
    graph, retriever = _graph_with_passthrough_fakes()

    result = graph.invoke(initial_state("What is the derivative of sin(x)?"))
    response = response_from_state(result)

    assert response.status != "computed"
    assert retriever.calls == 1


def test_off_topic_rejection_still_happens_before_the_router_matters() -> None:
    graph, retriever = _graph_with_failing_fakes()

    result = graph.invoke(initial_state("What's a good recipe for cookies?"))
    response = response_from_state(result)

    assert response.status == "refused"
    assert retriever.calls == 0
