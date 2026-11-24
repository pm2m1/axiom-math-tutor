# Architecture

This document describes the current architecture (Phase 0: essentially
none yet), the target architecture the roadmap builds toward, and the
reasoning behind the decisions that will shape it. Anything not listed under
"Current" does not exist in the codebase yet, regardless of how detailed its
description below is.

## Current

```text
Browser (React) --> FastAPI --> GET /health, GET /api/v1/status
                              -> POST /api/v1/question (Phase 1J, routing only)
                                   -> hybrid retrieval (BM25 + Qdrant dense + RRF)
                                   -> Meta evidence judge (muse-spark-1.3)
                                   -> sufficient: cited KB chunks
                                      insufficient: one Tavily MCP web search
                              -> POST /api/v1/answer (Phase 1K, LangGraph workflow)
                                   -> input guardrail (off-topic / injection rejection)
                                   -> same retrieval + judge + web fallback as above
                                   -> Meta answer generation, evidence-only, cited
                                   -> output validation (citations, no false
                                      verification claims)
                                   -> server-issued answer_id (Phase 1L)
                              -> POST /api/v1/feedback (Phase 1L)
                                   -> SQLite, write-only from the system's view
                              -> POST /api/v1/verify (Phase 2A)
                                   -> restricted AST parser (no eval/sympify-on-string)
                                   -> SymPy Poly/solveset, exact real-domain checking
                                   -> linear/quadratic only, no LLM involved
                              -> POST /api/v1/check-work (Phase 2B, + hints/misconceptions in Phase 3)
                                   -> same parser/solver, ordered student steps
                                   -> exact solution-set comparison per transition
                                   -> first provably-invalid step, independent of
                                      a separately-checked final answer
                              -> POST /api/v1/verify-derivative, /verify-integral,
                                 /count (Phase 3)
                                   -> same engine, extended calculus parsing mode
                                   -> exact diff/integrate/binomial, verified
                                      independently of the computed reference value
                              -> POST /api/v1/practice/{next,submit}, /mastery/summary (Phase 3)
                                   -> hand-designed, re-verified problem templates
                                   -> graded by the same engine as /verify
                                   -> SQLite mastery, write-gated on decisive outcomes
```

The React frontend now has three active modes. Tutor mode calls
`/api/v1/answer` for real and renders its `kb`/`web`/`computed`/
`refused`/`abstained`/`unavailable` outcomes distinctly (`"computed"`,
Phase 3, is a deterministic problem-router match — see
`docs/jee-solving.md`), with citations and a thumbs up/down +
optional-correction feedback control posting to `/api/v1/feedback`
(Phase 1L — see `docs/feedback.md`). Check My Work (Phase 2B/3 — see
`docs/verification.md` and `docs/jee-solving.md`) calls
`/api/v1/check-work` for real: ordered step inputs, an optional final
answer, a progressive hint ladder, misconception labels, and a results
view distinguishing equivalent/narrowing/broadening/partial-overlap/
incompatible/unsupported/indeterminate transitions, with the first
provably incorrect step flagged explicitly. Practice (Phase 3) calls
`/api/v1/practice/next`/`submit` and `/api/v1/mastery/summary` for
real. `/api/v1/question` (Phase 1J, see `docs/evidence-routing.md`)
still has no frontend caller; it exists for routing-only use. The
backend's persistence is Qdrant's vector store, the feedback SQLite
file, and now the mastery SQLite file.

`POST /api/v1/verify` and `POST /api/v1/check-work` (Phase 2A/2B, see
`docs/verification.md`) are the first real piece of the "Verifier" box
in the target diagram below — a deterministic SymPy engine for
single-variable linear/quadratic equations, with its own restricted
expression parser (no `eval`, no `sympy.sympify` on a string), now also
checking ordered reasoning chains, not just final candidate sets. Both
are deliberately standalone: nothing in the Phase 1K LangGraph workflow
calls either, and `/api/v1/answer`'s LLM-generated explanations are
still not "formally verified" merely because this exists alongside
them. No LLM is used anywhere in `check-work`'s error localization.

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

Several boxes now exist in narrow slices, none in the full form the
diagram implies. `Math Solver`/`Verifier`: equations (`/api/v1/verify`,
`/api/v1/check-work`), derivatives/integrals (`/verify-derivative`,
`/verify-integral`), and combinatorics (`/count`) — all exact SymPy
computation, Phase 2A/2B/3. `Problem Analyzer`/`Routing Engine`: a
*deliberately narrow* slice — `axiom/problem_router.py` (Phase 3)
recognizes a handful of explicit computation commands inside the
existing LangGraph workflow and routes only those to the solvers above;
it does not classify or route arbitrary JEE word problems, which is
real `Problem Analyzer`/`Routing Engine` scope this phase does not
claim. `Student Model`: `axiom/mastery.py` (Phase 3) — pseudonymous,
SQLite, two skills. `Misconception Diagnoser`: four labels derived from
Phase 2B's existing transitions (Phase 3), not a maintained taxonomy.
`Tutor Policy`: the 4-level hint ladder gating full-solution disclosure
behind an explicit request (Phase 3) — not mastery-gated hint
selection, and not applied to Tutor mode's LLM explanations at all.
See `docs/jee-solving.md` for exactly what each slice does and doesn't
cover. The roadmap (`docs/roadmap.md`) sequences what's still missing.

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

- **Deterministic**: symbolic simplification and equation solving
  (SymPy — `/api/v1/verify`, Phase 2A, for single-variable
  linear/quadratic equations today), numeric evaluation, checking a
  student's step against the previous one (`/api/v1/check-work`, Phase
  2B, via exact solution-set comparison), retrieval scoring/ranking,
  mastery updates given an outcome.
- **LLM-mediated**: understanding a free-text question, generating candidate
  solution steps for the solver to verify, phrasing a hint, diagnosing which
  named misconception a wrong step is consistent with, and generating
  practice problems.

Where both are plausible (e.g., "is this equation equivalent to that one"),
prefer the deterministic tool and only fall back to an LLM judgment when the
deterministic check doesn't apply.

## State

Agent state must be explicit and typed, not implicit in prompt history.
This is no longer just a sketch: `axiom/workflow.py::WorkflowState`
(Phase 1K — see `docs/agentic-workflow.md`) is a real `TypedDict` driving
a real LangGraph `StateGraph`, carrying guardrail classification,
retrieval hits, judge verdict, routing decision, the evidence actually
handed to generation, and the generated/validated answer, as named
fields — every value a later node depends on is explicit, never
re-derived from accumulated prompt history. Fields not yet needed
(student mastery, misconception, hint level — Phase 3) aren't speculated
into the state ahead of the phase that would use them.

## Retrieval architecture

Knowledge-base-first retrieval against a vector store (Qdrant), with a web
search / MCP fallback used only when KB retrieval confidence is below
threshold. This ordering is a requirement, not a preference: an agent that
reaches for the web before checking its own curriculum content cannot be
trusted to prefer vetted material, and it's slower and more expensive for
no benefit on the common case.

Qdrant now persists the dense retriever's vectors, either embedded locally
or via a real server (`compose.yaml`, Phase 1D/1E — see
`docs/retrieval-evaluation.md`). The KB-first-then-web ordering above is
implemented twice, deliberately sharing the same underlying pipeline: as
routing-only `POST /api/v1/question` (Phase 1J — see
`docs/evidence-routing.md`) and as the first stage of the full
`POST /api/v1/answer` LangGraph workflow (Phase 1K — see
`docs/agentic-workflow.md`), which adds an input guardrail and
evidence-grounded answer generation on top of the identical retrieval →
judge → fallback sequence. Neither call path reranks or reconfigures
retrieval itself. The frontend doesn't call either endpoint yet.

## Human feedback philosophy

Human-in-the-loop feedback (student "was this helpful", teacher review of
flagged sessions) is captured (Phase 1L — thumbs up/down and an optional
correction, persisted to SQLite, see `docs/feedback.md`) as an input to
future evaluation and prompt/retrieval tuning, not as a mechanism for the
system to silently retrain itself: nothing reads the feedback table back
into retrieval, the judge, generation, or any model parameter. Two
reasons the capture/use split is deliberate, not just not-built-yet:

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
