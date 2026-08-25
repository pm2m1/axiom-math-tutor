# Architecture

This document describes the current architecture (Phase 0: essentially
none yet), the target architecture the roadmap builds toward, and the
reasoning behind the decisions that will shape it. Anything not listed under
"Current" does not exist in the codebase yet, regardless of how detailed its
description below is.

## Current

```text
Browser (React) --> FastAPI --> GET /health, GET /api/v1/status
```

The frontend renders a static shell (mode selector, question form) with no
backend call wired to it beyond the status check. The backend has no
persistence, no LLM integration, and no business logic beyond returning a
fixed status payload built from `Settings`.

## Target

```text
Student
   |
   v
AI Gateway (input/output guardrails)
   |
   v
Problem Analyzer (topic/skill/difficulty classification)
   |
   +--------------------+
   |                    |
   v                    v
Student Model      Knowledge Layer (VectorDB + web fallback)
   |                    |
   +---------+----------+
             |
             v
        Routing Engine
             |
   +---------+---------+---------+---------+
   |         |         |         |         |
   v         v         v         v         v
  RAG      SymPy     Python     Lean*    MCP/Web
   |         |         |         |         |
   +---------+---------+---------+---------+
             |
             v
         Math Solver
             |
             v
          Verifier
             |
             v
 Misconception Diagnoser
             |
             v
        Tutor Policy
             |
             v
          Student
             |
             v
      Human Feedback --> Evaluation Store
```

`*` Lean is a research direction, not a planned MVP component. It's included
in the diagram only so today's structure doesn't have to be redesigned if it
gets added later.

None of the boxes above `Math Solver` and below `Tutor Policy` exist yet.
The roadmap (`docs/roadmap.md`) sequences them.

## The core separation: solver, verifier, tutor

This is the one architectural principle everything else is built to
preserve, so it's worth stating plainly:

- **The solver** produces a candidate mathematical answer or evaluates a
  claim. It may be an LLM, a symbolic engine, or (usually) both.
- **The verifier** checks the solver's output against ground truth using a
  deterministic tool wherever one exists — SymPy for algebra and calculus
  today, potentially a proof assistant for claims that warrant it later. An
  LLM is not a verifier for anything a deterministic tool can check; "ask the
  model if it's right" is not verification.
- **The tutor policy** decides what the student is shown. It has access to
  the verified solution but is a separate decision: whether to reveal it in
  full, give a hint, or ask a Socratic question depends on hint level,
  student mastery, and the mode the student chose — not on how the solver
  phrased its output.

Concretely: the solver may determine `x = 3`. The tutor is not obligated to
say so. It might instead ask "what operation would remove the `+5` from both
sides?" The solver's job ends at producing a verified fact; the tutor's job
is deciding how much of that fact serves the student's learning right now.

This separation exists because collapsing solving and tutoring into one LLM
call (the common pattern in "AI tutor" demos) makes it structurally
impossible to reason about correctness and pedagogy independently — a
prompt tweak meant to make explanations gentler can silently change whether
the underlying math is even right. Keeping them as distinct components with
an explicit interface between them is what makes each one independently
testable.

## Deterministic vs. LLM responsibilities

Anything that can be computed deterministically should be, and the LLM's
role is reasoning, language, and judgment calls that don't have a closed-form
answer:

- **Deterministic**: symbolic simplification and equation solving (SymPy),
  numeric evaluation, checking a student's step against the previous one,
  retrieval scoring/ranking, mastery updates given an outcome.
- **LLM-mediated**: understanding a free-text question, generating candidate
  solution steps for the solver to verify, phrasing a hint, diagnosing which
  named misconception a wrong step is consistent with, and generating
  practice problems.

Where both are plausible (e.g., "is this equation equivalent to that one"),
prefer the deterministic tool and only fall back to an LLM judgment when the
deterministic check doesn't apply.

## State

Agent state must be explicit and typed, not implicit in prompt history. A
LangGraph-style workflow will carry a state object roughly of the shape:

```python
class TutorState(TypedDict):
    question: str
    student_attempt: str | None

    topic: str | None
    skill: str | None
    difficulty: float | None

    retrieved_context: list[RetrievedChunk]
    retrieval_confidence: float | None

    candidate_solution: str | None
    verified: bool | None

    misconception: str | None
    misconception_confidence: float | None

    mastery: float | None
    hint_level: int

    feedback: dict | None
```

This is a sketch, not a commitment to the exact fields — it will be refined
once retrieval and verification are actually implemented in Phase 1/2. The
commitment is to the principle: every value the workflow depends on should be
a named, typed field, not something re-derived by re-prompting an LLM with
accumulated chat history.

## Retrieval architecture (planned)

Knowledge-base-first retrieval against a vector store (Qdrant), with a web
search / MCP fallback used only when KB retrieval confidence is below
threshold. This ordering is a requirement, not a preference: an agent that
reaches for the web before checking its own curriculum content cannot be
trusted to prefer vetted material, and it's slower and more expensive for
no benefit on the common case.

## Human feedback philosophy

Human-in-the-loop feedback (student "was this helpful", teacher review of
flagged sessions) is planned as an input to evaluation and future prompt/
retrieval tuning, not as a mechanism for the system to silently retrain
itself. Two reasons:

1. **Unsupervised self-modification from feedback signal is unsafe for a
   tutoring product.** A thumbs-down on a hint that was correct but blunt,
   aggregated automatically, could push the system toward answers that feel
   nicer but are less accurate or less pedagogically sound. Any change in
   behavior driven by feedback should go through a reviewed step (even if
   that step is "a person looks at the aggregated evaluation report before
   changing a prompt or retraining"), not a fully automatic loop.
2. **Feedback in this domain is sparse and noisy.** A student is not a
   reliable judge of whether a hint was mathematically correct, and a
   teacher reviewing sessions is the higher-signal input. The evaluation
   store should distinguish the two rather than averaging them together.

## Evaluation (planned)

Phase 4 introduces evaluation along axes that get built out roughly in this
order: mathematical correctness (SymPy-verifiable ground truth, JEEBench),
retrieval quality (precision/recall against the curriculum KB), tutoring
quality (does a hint ladder actually move a simulated or real student
toward the answer without giving it away), then security/latency/cost as
operational constraints once there's a working system to measure. Fabricated
or placeholder benchmark numbers are not acceptable at any point — a metric
is only reported once it's been computed against a real evaluation set.
