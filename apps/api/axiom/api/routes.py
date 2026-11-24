from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Depends, HTTPException

from axiom.api.dependencies import (
    RetrievalPipeline,
    get_evidence_judge,
    get_feedback_store,
    get_mastery_store,
    get_retrieval_pipeline,
    get_web_search_client,
    get_workflow,
)
from axiom.core.config import get_settings
from axiom.feedback import FeedbackStore, UnknownAnswerError
from axiom.mastery import MasteryStore
from axiom.models.answer import AnswerResponse
from axiom.models.evidence_routing import (
    EvidenceRoutingResponse,
    KBSource,
    QuestionRequest,
)
from axiom.models.evidence_routing import WebSource as WebSourceModel
from axiom.models.feedback import FeedbackRequest, FeedbackResponse
from axiom.models.practice import (
    MasterySummaryRequest,
    MasterySummaryResponse,
    NextProblemRequest,
    NextProblemResponse,
    SkillMastery,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)
from axiom.models.status import HealthResponse, StatusResponse
from axiom.models.verification import (
    CalculusResponse,
    CandidateResponse,
    CheckWorkRequest,
    CheckWorkResponse,
    CombinatoricsRequest,
    CombinatoricsResponse,
    DerivativeRequest,
    IntegralRequest,
    StepResult,
    VerifyRequest,
    VerifyResponse,
)
from axiom.practice import generate_practice_problem, select_next_problem
from axiom.retrieval.evidence_judge import JudgeClient
from axiom.verification.engine import check_combinatorics as run_check_combinatorics
from axiom.verification.engine import check_derivative as run_check_derivative
from axiom.verification.engine import check_integral as run_check_integral
from axiom.verification.engine import check_steps as run_check_steps
from axiom.verification.engine import hints_for_equation as run_hints_for_equation
from axiom.verification.engine import verify as run_verification
from axiom.web_search import WebSearchClient
from axiom.workflow import WorkflowState, initial_state, response_from_state

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

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


@router.post("/api/v1/answer", response_model=AnswerResponse)
def answer_with_generation(
    request: QuestionRequest,
    workflow: "CompiledStateGraph[WorkflowState, Any, Any, Any]" = Depends(get_workflow),
    feedback_store: FeedbackStore = Depends(get_feedback_store),
) -> AnswerResponse:
    """Guarded, evidence-grounded answer generation (Phase 1K) — the
    LangGraph workflow in `axiom/workflow.py`: input guardrail, the
    unchanged Phase 1J retrieval/judge/search pipeline, then Meta answer
    generation and output validation. See docs/agentic-workflow.md.
    Additive to `/api/v1/question`, which is unchanged and still routes
    without generating an answer.

    `answer_id` (Phase 1L, extended in Phase 3) is assigned here, outside
    the graph, only for an actual answer (`kb`/`web`/`computed`) —
    feedback can only attach to a request that produced one. See
    docs/feedback.md."""
    raw_result = workflow.invoke(initial_state(request.question))
    response = response_from_state(cast(WorkflowState, raw_result))
    if response.status in ("kb", "web", "computed"):
        response.answer_id = feedback_store.record_answer(response.status)
    return response


@router.post("/api/v1/feedback", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest, feedback_store: FeedbackStore = Depends(get_feedback_store)
) -> FeedbackResponse:
    """Persisted for later human review only (Phase 1L) — never read back
    into retrieval, the judge, or generation. See docs/feedback.md."""
    try:
        record = feedback_store.submit_feedback(
            request.answer_id, request.rating, request.correction
        )
    except UnknownAnswerError as exc:
        raise HTTPException(status_code=404, detail="unknown answer_id") from exc
    return FeedbackResponse(
        feedback_id=record.feedback_id,
        answer_id=record.answer_id,
        rating=record.rating,
        correction=record.correction,
        answer_status=record.answer_status,
        created_at=record.created_at,
    )


@router.post("/api/v1/verify", response_model=VerifyResponse)
def verify_equation(request: VerifyRequest) -> VerifyResponse:
    """Deterministic SymPy verification (Phase 2A) — no LLM, no network
    call, additive to every Phase 1 endpoint. See docs/verification.md
    for the supported grammar and the precise meaning of each status."""
    outcome = run_verification(request.equation, request.variable, request.solutions)
    return VerifyResponse(
        status=outcome.status,
        category=outcome.category,
        domain=outcome.domain,
        deterministic=outcome.deterministic,
        reason=outcome.reason,
        detail=outcome.detail,
        complete=outcome.complete,
        candidates=[CandidateResponse(value=c.value, valid=c.valid) for c in outcome.candidates],
    )


@router.post("/api/v1/check-work", response_model=CheckWorkResponse)
def check_work(request: CheckWorkRequest) -> CheckWorkResponse:
    """Deterministic SymPy step-by-step verification (Phase 2B) — no
    LLM, no network call, additive to every Phase 1/2A endpoint. Reuses
    `axiom/verification/parser.py` and `engine.py` exactly as Phase 2A
    built them; there is no second parser or solver. See
    docs/verification.md's "Check My Work" section."""
    outcome = run_check_steps(request.equation, request.variable, request.steps, request.solutions)
    final_answer = None
    if outcome.final_answer is not None:
        final_answer = VerifyResponse(
            status=outcome.final_answer.status,
            category=outcome.final_answer.category,
            domain=outcome.final_answer.domain,
            deterministic=outcome.final_answer.deterministic,
            reason=outcome.final_answer.reason,
            detail=outcome.final_answer.detail,
            complete=outcome.final_answer.complete,
            candidates=[
                CandidateResponse(value=c.value, valid=c.valid)
                for c in outcome.final_answer.candidates
            ],
        )
    hints = None
    if request.hint_level is not None:
        hints = run_hints_for_equation(request.equation, request.variable, request.hint_level)
    return CheckWorkResponse(
        status=outcome.status,
        steps=[
            StepResult(
                index=s.index,
                equation=s.equation,
                category=s.category,
                transition=s.transition,
                detail=s.detail,
                misconception=s.misconception,
            )
            for s in outcome.steps
        ],
        first_invalid_step_index=outcome.first_invalid_step_index,
        first_non_equivalent_step_index=outcome.first_non_equivalent_step_index,
        chain_preserves_solution_set=outcome.chain_preserves_solution_set,
        deterministic=outcome.deterministic,
        detail=outcome.detail,
        final_answer=final_answer,
        hints=hints,
    )


@router.post("/api/v1/verify-derivative", response_model=CalculusResponse)
def verify_derivative(request: DerivativeRequest) -> CalculusResponse:
    """Deterministic SymPy derivative verification (Phase 3) — no LLM,
    no network call. See docs/verification.md's "Calculus" section."""
    outcome = run_check_derivative(
        request.expression, request.variable, request.proposed_derivative
    )
    return CalculusResponse(
        status=outcome.status,
        operation=outcome.operation,
        computed=outcome.computed,
        deterministic=outcome.deterministic,
        reason=outcome.reason,
        detail=outcome.detail,
    )


@router.post("/api/v1/verify-integral", response_model=CalculusResponse)
def verify_integral(request: IntegralRequest) -> CalculusResponse:
    """Deterministic SymPy indefinite-integral verification (Phase 3) —
    no LLM, no network call. Definite integrals are out of scope. See
    docs/verification.md's "Calculus" section."""
    outcome = run_check_integral(
        request.expression, request.variable, request.proposed_antiderivative
    )
    return CalculusResponse(
        status=outcome.status,
        operation=outcome.operation,
        computed=outcome.computed,
        deterministic=outcome.deterministic,
        reason=outcome.reason,
        detail=outcome.detail,
    )


@router.post("/api/v1/count", response_model=CombinatoricsResponse)
def count_combinatorics(request: CombinatoricsRequest) -> CombinatoricsResponse:
    """Exact rational combinatorics/probability (Phase 3) — `nCr`,
    `nPr`, and a narrowly-defined "all drawn items favorable" probability
    pattern. No LLM, no network call. See docs/verification.md."""
    outcome = run_check_combinatorics(
        request.operation, request.n, request.r, request.favorable, request.proposed_answer
    )
    return CombinatoricsResponse(
        status=outcome.status,
        operation=outcome.operation,
        computed=outcome.computed,
        deterministic=outcome.deterministic,
        reason=outcome.reason,
        detail=outcome.detail,
    )


@router.post("/api/v1/practice/next", response_model=NextProblemResponse)
def next_practice_problem(
    request: NextProblemRequest, mastery_store: MasteryStore = Depends(get_mastery_store)
) -> NextProblemResponse:
    """Selects a skill/difficulty from the learner's current mastery
    (`axiom/practice.py::select_next_problem` — a documented heuristic,
    not a statistically validated adaptive algorithm), generates a
    problem from a small hand-designed template bank, and independently
    re-verifies it through the same engine `/api/v1/verify` uses before
    ever returning it. See docs/jee-solving.md."""
    linear_mastery = mastery_store.get_mastery(request.learner_id, "linear_equations")
    quadratic_mastery = mastery_store.get_mastery(request.learner_id, "quadratic_equations")
    skill, difficulty = select_next_problem(linear_mastery, quadratic_mastery)
    problem = generate_practice_problem(skill, difficulty)
    if problem is None:
        raise HTTPException(status_code=503, detail="practice problem generation failed")
    return NextProblemResponse(
        skill=problem.skill,
        difficulty=problem.difficulty,
        equation=problem.equation,
        variable=problem.variable,
    )


@router.post("/api/v1/practice/submit", response_model=SubmitAnswerResponse)
def submit_practice_answer(
    request: SubmitAnswerRequest, mastery_store: MasteryStore = Depends(get_mastery_store)
) -> SubmitAnswerResponse:
    """Grades with the exact same deterministic engine `/api/v1/verify`
    uses — an LLM is never asked whether a practice answer is correct.
    Mastery is updated only for a decisive `verified_correct`/
    `verified_incorrect` result; see `axiom/mastery.py` for the exact
    update rule and why every other outcome withholds the update."""
    outcome = run_verification(request.equation, request.variable, request.solutions)
    correct: bool | None
    if outcome.status == "verified_correct":
        correct = True
    elif outcome.status == "verified_incorrect":
        correct = False
    else:
        correct = None
    result = mastery_store.record_attempt(
        request.learner_id,
        request.skill,
        verified=correct is not None,
        correct=correct,
        hints_used=request.hints_used,
    )
    return SubmitAnswerResponse(
        status=outcome.status,
        detail=outcome.detail,
        correct=correct,
        mastery_estimate=result.estimate,
    )


@router.post("/api/v1/mastery/summary", response_model=MasterySummaryResponse)
def mastery_summary(
    request: MasterySummaryRequest, mastery_store: MasteryStore = Depends(get_mastery_store)
) -> MasterySummaryResponse:
    """Always reports both known practice skills, defaulting to the
    uninformative prior for one never attempted — a simpler, more
    predictable contract for the frontend than a sparse list."""
    skills = ["linear_equations", "quadratic_equations"]
    return MasterySummaryResponse(
        skills=[
            SkillMastery(skill=skill, estimate=mastery_store.get_mastery(request.learner_id, skill))
            for skill in skills
        ]
    )
