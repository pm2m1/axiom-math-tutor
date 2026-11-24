from typing import Literal

from pydantic import BaseModel

AnswerReason = Literal[
    "off_topic",
    "prompt_injection",
    "insufficient_evidence",
    "invalid_output",
    "retrieval_error",
    "judge_error",
    "search_error",
    "generation_error",
]


class Citation(BaseModel):
    source_id: str
    source_type: Literal["kb", "web"]
    title: str
    url: str


class AnswerResponse(BaseModel):
    """`status` discriminates the response:

    - `"kb"` / `"web"` — a grounded explanation, `answer` set, `citations`
      drawn only from the evidence actually used.
    - `"computed"` (Phase 3) — the question matched one of a small set of
      explicit, unambiguous computation commands (`axiom/problem_router.py`
      — e.g. "differentiate 2*x^2", "integrate cos(x)"), and was answered
      by exact SymPy computation, not by the LLM/retrieval path at all.
      `verification_method` is `"deterministic_sympy"` and `solver_category`
      names which solver answered it.
    - `"refused"` — the input guardrail rejected the request (off-topic or
      a prompt-injection attempt); nothing downstream was called.
    - `"abstained"` — evidence was looked for but is insufficient, or the
      model's own output failed validation; no unsupported answer is
      returned.
    - `"unavailable"` — a dependency failed (retrieval, judge, search, or
      generation). `detail` is always a short, generic, hand-written
      string — never a raw provider error, stack trace, or credential.

    `answer` is non-null only for `"kb"`/`"web"`/`"computed"`. For
    `"kb"`/`"web"` this is an LLM explanation grounded in cited evidence,
    not a formally verified derivation (`verification_method` is
    `"evidence_grounded_llm"` for these) — see docs/agentic-workflow.md.
    Only `"computed"` answers are the result of a deterministic
    mathematical check; the two are never conflated.

    `answer_id` (Phase 1L) is a server-issued identifier, set only for
    `"kb"`/`"web"`/`"computed"` — feedback can only be attached to a
    request that actually produced an answer. It is assigned by the
    route handler after the workflow runs, not part of `WorkflowState` —
    recording it is bookkeeping for feedback attribution, not a workflow
    decision."""

    status: Literal["kb", "web", "computed", "refused", "abstained", "unavailable"]
    answer: str | None = None
    citations: list[Citation] = []
    reason: AnswerReason | None = None
    detail: str | None = None
    answer_id: str | None = None
    verification_method: Literal["deterministic_sympy", "evidence_grounded_llm"] | None = None
    solver_category: Literal["derivative", "integral", "combinatorics"] | None = None
