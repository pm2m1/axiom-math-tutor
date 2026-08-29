# Roadmap

Phases are sequential in intent but not strictly blocking — later phases may
start once their prerequisites within earlier phases exist, without waiting
for everything in an earlier phase to be "done." Each phase lists acceptance
criteria; a phase isn't complete until its criteria hold against real usage,
not just against a demo path.

## Phase 0 — Foundation

Repository structure, backend/frontend skeleton, local tooling.

- [x] Monorepo layout (`apps/api`, `apps/web`, `docs`)
- [x] FastAPI app with `/health` and `/api/v1/status`
- [x] React/TypeScript frontend shell (mode selector, question form, dev
      status indicator)
- [x] Backend tests (pytest), lint/format (ruff), type checking (mypy)
- [x] Frontend build (tsc + vite), lint (oxlint)

**Acceptance**: both apps run locally, tests and lint pass, no fabricated
functionality.

## Phase 1 — Assignment-complete agentic RAG

The minimum scope required by the originating assignment.

**Phase 1A — knowledge base and retrieval evaluation foundation (done):**

- [x] Narrow algebra corpus (equations, inequalities, quadratics), sourced
      and licensed deliberately — see `data/README.md`
- [x] Deterministic ingestion: manifest → fetch → chunk, with stable chunk
      IDs and provenance preserved through the pipeline
- [x] Hand-checked labeled retrieval evaluation set (50 queries)
- [x] BM25 lexical retrieval baseline with Hit@k/Recall@5/MRR
- [x] Confidence investigation: measured whether BM25's top score separates
      in-scope from out-of-scope queries (it doesn't cleanly — see
      `docs/retrieval-evaluation.md`) before picking a fallback threshold

**Phase 1B — dense retrieval experiment (done):**

- [x] Dense embedding retriever (`Alibaba-NLP/gte-modernbert-base`) behind
      the same `Retriever` protocol as BM25, evaluated against the identical
      Phase 1A corpus and query set — no in-memory Qdrant yet, that's the
      next infrastructure step, not this one
- [x] Side-by-side comparison, query-type breakdown, and paired failure
      analysis: dense outperforms BM25 on every metric (Hit@1 +0.17, MRR
      +0.12), concentrated on paraphrase/conceptual/definition queries
- [x] Repeated the confidence investigation for dense scores: the
      in-scope/out-of-scope overlap is proportionally similar to BM25's, not
      meaningfully cleaner — see `docs/retrieval-evaluation.md`

**Phase 1C — hybrid retrieval experiment (done):**

- [x] Reciprocal Rank Fusion of BM25 and dense rankings (`HybridRetriever`,
      same `Retriever` protocol), evaluated against the identical corpus and
      query set — still in-memory, no Qdrant
- [x] Beats dense — the stronger single baseline — on every metric (Hit@1
      +0.05, MRR +0.03) and recovers most of dense's regression on
      `application` queries, without a new dependency or a tuned constant
- [x] Repeated the confidence investigation for RRF scores: fused score has
      less usable dynamic range than either component's own score, not
      more — see `docs/retrieval-evaluation.md`

**Still open for Phase 1:**

- [ ] Input/output guardrails (reject non-math input, refuse to leak system
      prompts, basic prompt-injection resistance)
- [ ] Qdrant-backed vector retrieval (the Phase 1B/1C retrievers are
      in-memory; moving the best-measured one to Qdrant is a separate,
      deliberately deferred infrastructure change — see
      `docs/retrieval-evaluation.md`)
- [ ] KB-first retrieval policy with an empirically justified confidence
      threshold (not chosen arbitrarily — none of Phase 1A's, 1B's, or 1C's
      confidence investigations found one yet)
- [ ] Web search / MCP fallback triggered only below that threshold
- [ ] LangGraph workflow tying analyzer → retrieval → routing → response
      together with explicit typed state
- [ ] Human-in-the-loop feedback capture (thumbs up/down or equivalent,
      persisted, not fed back into the system automatically)

**Acceptance**: given a question the KB actually covers, the system answers
from the KB and can show what it retrieved; given a question it doesn't
cover, it falls back to web/MCP and says so; a deliberately off-topic or
adversarial input is refused by the guardrail rather than reaching the LLM
call unfiltered.

## Phase 2 — Verified mathematical reasoning

- [ ] Problem router: classify a question into a small set of solvable
      categories (e.g., algebraic equation, calculus, arithmetic word
      problem) rather than always going straight to an LLM
- [ ] SymPy-backed solver for the categories that support it
- [ ] Deterministic verifier that checks candidate solutions against SymPy
      rather than asking an LLM to grade itself
- [ ] Confidence scoring and an explicit abstention path (the system says
      "I can't verify this" instead of presenting an unverified guess as
      fact)
- [ ] Recorded reasoning traces (what was retrieved, what was computed, what
      was verified) attached to each answer for later evaluation and debugging

**Acceptance**: for problems in a supported category, the answer shown to
the student has been checked by SymPy, not just generated by an LLM; the
system can distinguish and label verified vs. unverified output.

## Phase 3 — Adaptive tutoring

- [ ] Explicit student model (per-skill mastery estimate, persisted)
- [ ] "Check my work": given a student's own steps, identify the first
      invalid step using the verifier, not a vibe-based LLM judgment
- [ ] Misconception diagnosis: map an invalid step to a named misconception
      from a maintained taxonomy, with a confidence score
- [ ] Hint ladder: tutor policy that reveals progressively more of the
      verified solution across turns, gated by hint level and mastery,
      never handing over the answer on the first ask in Tutor mode
- [ ] Practice/remediation: generate problems targeting a diagnosed skill
      gap, verified before being shown
- [ ] Prerequisite/curriculum graph (Neo4j) informing what to remediate
      first when multiple gaps are diagnosed

**Acceptance**: two students who make the same mathematical error but have
different mastery histories can receive different hints; the first invalid
step in a submitted solution is identified correctly against a hand-checked
test set, not just plausibly.

## Phase 4 — Evaluation and research

- [ ] JEEBench evaluation harness for mathematical correctness
- [ ] A custom tutoring-quality benchmark (does a hint ladder measurably
      help a held-out set of problems get solved without revealing the
      answer outright) — built once Phase 3 exists, not before
- [ ] Retrieval evaluation (precision/recall of KB retrieval against a
      labeled query set)
- [ ] Ablations isolating the effect of verification, retrieval, and hint
      ladder independently
- [ ] Adversarial testing of the guardrails from Phase 1
- [ ] DSPy-based prompt/retrieval optimization, only if the ablations above
      show it's worth the added complexity

**Acceptance**: every number reported here is computed from a real run
against a real evaluation set, with the harness and data checked into the
repository or otherwise reproducible — no benchmark result is reported
without one.

## Explicitly out of scope for now

- User accounts / authentication
- Multi-tenant deployment, production infrastructure, Docker/CI (will be
  added when there's a system worth deploying, not before)
- Lean/formal verification (research direction; revisit only if Phase 2's
  SymPy verifier proves insufficient for a class of problems that matters)
