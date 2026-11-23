from fastapi import APIRouter, Depends

from axiom.api.dependencies import (
    RetrievalPipeline,
    get_evidence_judge,
    get_retrieval_pipeline,
    get_web_search_client,
)
from axiom.core.config import get_settings
from axiom.models.evidence_routing import (
    EvidenceRoutingResponse,
    KBSource,
    QuestionRequest,
)
from axiom.models.evidence_routing import WebSource as WebSourceModel
from axiom.models.status import HealthResponse, StatusResponse
from axiom.retrieval.evidence_judge import JudgeClient
from axiom.web_search import WebSearchClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/v1/status", response_model=StatusResponse)
def status() -> StatusResponse:
    settings = get_settings()
    return StatusResponse(status="ok", service="axiom-api", environment=settings.environment)


@router.post("/api/v1/question", response_model=EvidenceRoutingResponse)
def answer_question(
    request: QuestionRequest,
    pipeline: RetrievalPipeline | None = Depends(get_retrieval_pipeline),
    judge: JudgeClient | None = Depends(get_evidence_judge),
    search_client: WebSearchClient | None = Depends(get_web_search_client),
) -> EvidenceRoutingResponse:
    """Evidence-only routing (Phase 1J): the selected hybrid retriever's
    top-5 chunks are judged for sufficiency by the frozen Meta judge
    (Phase 1I, unchanged). Sufficient evidence returns its cited KB
    chunks; insufficient evidence triggers exactly one Tavily MCP web
    search. A judge failure is a dependency failure, not a web-search
    trigger — this does not generate an answer, only routes and cites."""
    if pipeline is None:
        return EvidenceRoutingResponse(
            route="unavailable", reason="retrieval_error", detail="retrieval is not available"
        )
    if judge is None:
        return EvidenceRoutingResponse(
            route="unavailable", reason="judge_error", detail="evidence judge is not configured"
        )

    try:
        hits = pipeline.hybrid.search(request.question, limit=5)
    except Exception:
        return EvidenceRoutingResponse(
            route="unavailable", reason="retrieval_error", detail="retrieval failed"
        )

    chunk_ids = [hit.chunk.chunk_id for hit in hits]
    chunk_texts = [hit.chunk.text for hit in hits]
    outcome = judge.judge(request.question, chunk_ids, chunk_texts)

    if outcome.judgment is None:
        return EvidenceRoutingResponse(
            route="unavailable", reason="judge_error", detail="evidence judge failed"
        )

    if outcome.judgment.verdict == "sufficient":
        chunks_by_id = {hit.chunk.chunk_id: hit.chunk for hit in hits}
        cited = [
            chunks_by_id[cid]
            for cid in outcome.judgment.supporting_chunk_ids
            if cid in chunks_by_id
        ]
        kb_sources = [
            KBSource(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section,
                subsection=chunk.subsection,
                source_url=chunk.source_url,
            )
            for chunk in cited
        ]
        return EvidenceRoutingResponse(route="kb", kb_sources=kb_sources)

    if search_client is None:
        return EvidenceRoutingResponse(
            route="unavailable", reason="search_error", detail="web search is not configured"
        )

    search_outcome = search_client.search(request.question)
    if search_outcome.error is not None:
        return EvidenceRoutingResponse(
            route="unavailable", reason="search_error", detail="web search failed"
        )

    web_sources = [
        WebSourceModel(title=r.title, url=r.url, snippet=r.snippet) for r in search_outcome.results
    ]
    detail = None if web_sources else "no web search results found"
    return EvidenceRoutingResponse(route="web_search", web_sources=web_sources, detail=detail)
