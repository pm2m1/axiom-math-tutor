"""Typed-state LangGraph workflow (Phase 1K) tying together the input
guardrail, the existing Phase 1J retrieval/judge/search pipeline
(unchanged), and the new Phase 1K answer generator/output validator into
one deterministic graph.

Every routing decision is a plain Python conditional reading explicit
state fields — the LLM is never asked to decide what to do next, only to
judge evidence sufficiency or generate an explanation from evidence it
was already given. There is no loop: each node runs at most once per
request, and every path terminates at `finalize_response`.

State is a single typed `WorkflowState` TypedDict (see docs/architecture.md's
"Agent state must be explicit and typed" principle, now applied) — every
value a later node depends on is a named field, never something
re-derived from accumulated history.
"""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from axiom.guardrails import classify_input
from axiom.models.answer import AnswerReason, AnswerResponse, Citation
from axiom.retrieval.answer_generation import (
    AnswerGenerator,
    EvidenceItem,
    GeneratedAnswer,
    validate_generated_answer,
)
from axiom.retrieval.base import Retriever
from axiom.retrieval.evidence_judge import JudgeClient
from axiom.retrieval.models import RetrievalResult
from axiom.web_search import WebSearchClient, WebSource


class WorkflowState(TypedDict, total=False):
    question: str

    input_status: Literal["accepted", "rejected_off_topic", "rejected_injection"]
    input_detail: str

    retrieval_ok: bool
    retrieved_chunk_ids: list[str]
    retrieved_chunk_texts: list[str]
    retrieved_chunks: list[RetrievalResult]

    judge_ok: bool
    judge_verdict: Literal["sufficient", "insufficient"]
    judge_cited_chunk_ids: list[str]

    route: Literal["kb", "web"]
    search_ok: bool
    search_empty: bool
    web_results: list[WebSource]

    evidence: list[EvidenceItem]

    generation_outcome: Literal["ok", "no_evidence", "error"]
    generation_error_detail: str
    generated_answer: GeneratedAnswer | None

    output_status: Literal["valid", "model_abstained", "invalid"]
    output_invalid_reason: str | None

    final_status: Literal["kb", "web", "refused", "abstained", "unavailable"]
    final_reason: AnswerReason | None
    final_detail: str | None
    final_answer: str | None
    final_citations: list[Citation]


def initial_state(question: str) -> WorkflowState:
    return {
        "question": question,
        "input_status": "accepted",
        "input_detail": "",
        "retrieval_ok": False,
        "retrieved_chunk_ids": [],
        "retrieved_chunk_texts": [],
        "retrieved_chunks": [],
        "judge_ok": False,
        "judge_verdict": "insufficient",
        "judge_cited_chunk_ids": [],
        "route": "kb",
        "search_ok": False,
        "search_empty": False,
        "web_results": [],
        "evidence": [],
        "generation_outcome": "error",
        "generation_error_detail": "",
        "generated_answer": None,
        "output_status": "invalid",
        "output_invalid_reason": None,
        "final_status": "unavailable",
        "final_reason": None,
        "final_detail": None,
        "final_answer": None,
        "final_citations": [],
    }


def response_from_state(state: WorkflowState) -> AnswerResponse:
    return AnswerResponse(
        status=state["final_status"],
        answer=state.get("final_answer"),
        citations=state.get("final_citations") or [],
        reason=state.get("final_reason"),
        detail=state.get("final_detail"),
    )


@dataclass
class _Final:
    status: Literal["kb", "web", "refused", "abstained", "unavailable"]
    reason: AnswerReason | None = None
    detail: str | None = None
    answer: str | None = None
    citations: list[Citation] | None = None

    def to_update(self) -> dict[str, object]:
        return {
            "final_status": self.status,
            "final_reason": self.reason,
            "final_detail": self.detail,
            "final_answer": self.answer,
            "final_citations": self.citations or [],
        }


def _build_citations(state: WorkflowState) -> list[Citation]:
    answer = state["generated_answer"]
    assert answer is not None
    cited_ids = set(answer.cited_source_ids)
    evidence = state.get("evidence", [])
    route = state.get("route")

    if route == "kb":
        chunk_by_id = {r.chunk.chunk_id: r.chunk for r in state.get("retrieved_chunks", [])}
        return [
            Citation(
                source_id=item.source_id,
                source_type="kb",
                title=chunk_by_id[item.source_id].title,
                url=chunk_by_id[item.source_id].source_url,
            )
            for item in evidence
            if item.source_id in cited_ids and item.source_id in chunk_by_id
        ]

    web_results = state.get("web_results", [])
    citations = []
    for idx, item in enumerate(evidence):
        if item.source_id in cited_ids and idx < len(web_results):
            result = web_results[idx]
            citations.append(
                Citation(
                    source_id=item.source_id,
                    source_type="web",
                    title=result.title,
                    url=result.url,
                )
            )
    return citations


def build_workflow(
    hybrid_retriever: Retriever | None,
    judge: JudgeClient | None,
    search_client: WebSearchClient | None,
    generator: AnswerGenerator | None,
) -> CompiledStateGraph[WorkflowState, Any, Any, Any]:
    """Each node closes over one injected dependency — `None` means that
    dependency is unavailable, handled explicitly in the node, never an
    unhandled exception propagating out of `.invoke()`. `hybrid_retriever`
    is the same `HybridRetriever` Phase 1J already validated (BM25 +
    Qdrant dense + RRF, top-5, unchanged)."""

    def validate_input_node(state: WorkflowState) -> dict[str, object]:
        classification = classify_input(state["question"])
        return {"input_status": classification.status, "input_detail": classification.detail}

    def route_after_validate(state: WorkflowState) -> str:
        return "accepted" if state["input_status"] == "accepted" else "rejected"

    def retrieve_evidence_node(state: WorkflowState) -> dict[str, object]:
        if hybrid_retriever is None:
            return {"retrieval_ok": False}
        try:
            hits = hybrid_retriever.search(state["question"], limit=5)
        except Exception:
            return {"retrieval_ok": False}
        return {
            "retrieval_ok": True,
            "retrieved_chunk_ids": [h.chunk.chunk_id for h in hits],
            "retrieved_chunk_texts": [h.chunk.text for h in hits],
            "retrieved_chunks": hits,
        }

    def route_after_retrieve(state: WorkflowState) -> str:
        return "ok" if state.get("retrieval_ok") else "error"

    def judge_evidence_node(state: WorkflowState) -> dict[str, object]:
        if judge is None:
            return {"judge_ok": False}
        outcome = judge.judge(
            state["question"], state["retrieved_chunk_ids"], state["retrieved_chunk_texts"]
        )
        if outcome.judgment is None:
            return {"judge_ok": False}
        is_sufficient = outcome.judgment.verdict == "sufficient"
        return {
            "judge_ok": True,
            "judge_verdict": outcome.judgment.verdict,
            "judge_cited_chunk_ids": outcome.judgment.supporting_chunk_ids,
            "route": "kb" if is_sufficient else "web",
        }

    def route_after_judge(state: WorkflowState) -> str:
        if not state.get("judge_ok"):
            return "error"
        return "sufficient" if state["judge_verdict"] == "sufficient" else "insufficient"

    def fetch_web_fallback_node(state: WorkflowState) -> dict[str, object]:
        if search_client is None:
            return {"search_ok": False}
        outcome = search_client.search(state["question"])
        if outcome.error is not None:
            return {"search_ok": False}
        return {
            "search_ok": True,
            "search_empty": len(outcome.results) == 0,
            "web_results": outcome.results,
        }

    def route_after_search(state: WorkflowState) -> str:
        if not state.get("search_ok"):
            return "error"
        return "empty" if state.get("search_empty") else "ok"

    def generate_answer_node(state: WorkflowState) -> dict[str, object]:
        route = state.get("route")
        if route == "kb":
            cited_ids = set(state.get("judge_cited_chunk_ids", []))
            chunk_by_id = {r.chunk.chunk_id: r.chunk for r in state.get("retrieved_chunks", [])}
            evidence = [
                EvidenceItem(source_id=cid, source_type="kb", text=chunk_by_id[cid].text)
                for cid in cited_ids
                if cid in chunk_by_id
            ]
        else:
            evidence = [
                EvidenceItem(source_id=f"web-{i}", source_type="web", text=r.snippet)
                for i, r in enumerate(state.get("web_results", []))
            ]

        if not evidence:
            return {"generation_outcome": "no_evidence"}
        if generator is None:
            return {
                "generation_outcome": "error",
                "generation_error_detail": "generator not configured",
            }

        outcome = generator.generate(state["question"], evidence)
        if outcome.answer is None:
            return {
                "generation_outcome": "error",
                "generation_error_detail": outcome.error or "generation failed",
            }
        return {
            "generation_outcome": "ok",
            "generated_answer": outcome.answer,
            "evidence": evidence,
        }

    def route_after_generate(state: WorkflowState) -> str:
        return "ok" if state.get("generation_outcome") == "ok" else "finalize"

    def validate_output_node(state: WorkflowState) -> dict[str, object]:
        answer = state["generated_answer"]
        assert answer is not None
        if answer.abstain:
            return {
                "output_status": "model_abstained",
                "output_invalid_reason": answer.abstain_reason,
            }
        evidence_ids = [item.source_id for item in state.get("evidence", [])]
        error = validate_generated_answer(answer, evidence_ids)
        if error:
            return {"output_status": "invalid", "output_invalid_reason": error}
        return {"output_status": "valid"}

    def finalize_response_node(state: WorkflowState) -> dict[str, object]:
        input_status = state.get("input_status")
        if input_status == "rejected_off_topic":
            return _Final("refused", "off_topic", state.get("input_detail")).to_update()
        if input_status == "rejected_injection":
            return _Final("refused", "prompt_injection", state.get("input_detail")).to_update()

        if not state.get("retrieval_ok"):
            return _Final(
                "unavailable", "retrieval_error", "retrieval is not available"
            ).to_update()
        if not state.get("judge_ok"):
            return _Final("unavailable", "judge_error", "evidence judge failed").to_update()

        if state.get("route") == "web":
            if not state.get("search_ok"):
                return _Final("unavailable", "search_error", "web search failed").to_update()
            if state.get("search_empty"):
                return _Final(
                    "abstained", "insufficient_evidence", "no web search results found"
                ).to_update()

        generation_outcome = state.get("generation_outcome")
        if generation_outcome == "no_evidence":
            return _Final(
                "abstained", "insufficient_evidence", "no usable evidence for generation"
            ).to_update()
        if generation_outcome == "error":
            return _Final("unavailable", "generation_error", "answer generation failed").to_update()

        output_status = state.get("output_status")
        answer = state.get("generated_answer")
        if output_status == "model_abstained":
            detail = answer.abstain_reason if answer else None
            return _Final("abstained", "insufficient_evidence", detail).to_update()
        if output_status == "invalid":
            return _Final(
                "abstained", "invalid_output", "generated answer failed validation"
            ).to_update()

        assert answer is not None
        route = state.get("route", "kb")
        return _Final(
            status=route, answer=answer.explanation, citations=_build_citations(state)
        ).to_update()

    builder = StateGraph(WorkflowState)
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("retrieve_evidence", retrieve_evidence_node)
    builder.add_node("judge_evidence", judge_evidence_node)
    builder.add_node("fetch_web_fallback", fetch_web_fallback_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("validate_output", validate_output_node)
    builder.add_node("finalize_response", finalize_response_node)

    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input",
        route_after_validate,
        {"accepted": "retrieve_evidence", "rejected": "finalize_response"},
    )
    builder.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieve,
        {"ok": "judge_evidence", "error": "finalize_response"},
    )
    builder.add_conditional_edges(
        "judge_evidence",
        route_after_judge,
        {
            "error": "finalize_response",
            "sufficient": "generate_answer",
            "insufficient": "fetch_web_fallback",
        },
    )
    builder.add_conditional_edges(
        "fetch_web_fallback",
        route_after_search,
        {"error": "finalize_response", "empty": "finalize_response", "ok": "generate_answer"},
    )
    builder.add_conditional_edges(
        "generate_answer",
        route_after_generate,
        {"ok": "validate_output", "finalize": "finalize_response"},
    )
    builder.add_edge("validate_output", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile()
