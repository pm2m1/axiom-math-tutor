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

**Phase 1D — Qdrant-backed persistent dense retrieval (done):**

- [x] `QdrantDenseRetriever` (local persistent mode) behind the same
      `Retriever` protocol, storing the same embeddings from the same
      pinned model; BM25 and RRF fusion untouched — see
      `docs/retrieval-evaluation.md`
- [x] Parity confirmed: identical Hit@k/Recall@5/MRR to the in-memory
      retrievers, 50/50 query top-1 ranking agreement for both dense and
      hybrid
- [x] Persistence and idempotency demonstrated with the real 40-chunk
      corpus: indexing twice produces 40 points, not 80; a fresh process
      queries the collection in seconds instead of re-embedding for ~200s

**Phase 1E — Qdrant server parity (done):**

- [x] Docker Compose Qdrant server (`compose.yaml`, pinned image, named
      volume), selected via `AXIOM_QDRANT_URL`; embedded local mode remains
      the default and is unaffected — see `docs/retrieval-evaluation.md`
- [x] Parity confirmed against a real running server: identical
      Hit@k/Recall@5/MRR, 50/50 top-1 agreement and 1.00 top-5 overlap for
      both dense and hybrid
- [x] Persistence demonstrated across a container restart, not just a
      process restart; not an HNSW/ANN test — this corpus is too small to
      cross Qdrant's full-scan threshold in either deployment mode

**Phase 1F — Qdrant-native lexical retrieval and fusion (done):**

- [x] `QdrantBM25Retriever` (server-side native BM25, English defaults)
      evaluated independently and in hybrid against `rank_bm25` — similar
      aggregate lexical metrics but only 30/50 top-1 agreement, and it
      makes the hybrid worse (Hit@1 0.74→0.67) despite that — see
      `docs/retrieval-evaluation.md`
- [x] `QdrantNativeHybridRetriever` (native Query API RRF, k=61 — derived
      and tested to reproduce our application RRF's k=60 under Qdrant's
      zero-based rank convention) confirmed near-exact parity with
      application-side RRF (49/50 top-1 agreement)
- [x] Validated system unchanged: `rank_bm25` + Qdrant dense + application
      RRF remains what's recommended, since native BM25 regressed the
      hybrid; native RRF is a measured, available, but not yet adopted
      alternative for the fusion step alone

**Phase 1G — KB-sufficiency evaluation foundation (done):**

- [x] New 107-query sufficiency benchmark (`data/evaluation/
      algebra_sufficiency.jsonl`, human-verified against the actual 40
      chunks — not LLM-generated and trusted), separate from the Phase 1A
      retrieval benchmark, with a fixed development/test split
- [x] Three scalar threshold baselines (BM25/dense/RRF top score) plus one
      logistic regression over six retrieval-evidence features, thresholds
      selected on development only, evaluated on the locked test split
      exactly once — see `docs/kb-sufficiency.md`
- [x] Real result, not tuned to look good: the classifier separates
      supported from near-domain-unsupported meaningfully better than
      chance (85% near-domain rejection) but still has an 11% false-accept
      rate with an identifiable cause (chunks that name a concept without
      explaining it) — not yet safe to gate an automatic fallback decision

**Phase 1H — evidence-support cross-encoder evaluation (done):**

- [x] Relabeled Phase 1G's `test` split as the historical v1 test set
      (previously inspected, no longer an untouched holdout) and created a
      fresh 60-query holdout (`algebra_sufficiency_v2.jsonl`), hand-
      verified the same way as Phase 1G, evaluated only after model and
      feature design were frozen — see `docs/kb-sufficiency.md`
- [x] `cross-encoder/ms-marco-MiniLM-L6-v2` (pinned revision) scored
      against the selected hybrid retriever's existing top-5 evidence, no
      reranking, no LLM judge; three approaches compared — retrieval-only,
      cross-encoder-only, and a combined classifier
- [x] Real, honest negative result: on the fresh holdout the combined
      classifier's false-accept rate (0.16) is worse than retrieval-only
      (0.12), not better; the cross-encoder is fooled by the same topical
      closeness that fools retrieval on the "mentions but doesn't explain"
      failure class, and does not correct it

**Phase 1I — LLM evidence-sufficiency judge (done):**

- [x] LLM evidence-sufficiency judge implemented, frozen prompt
      `evidence-sufficiency-v1`, `EvidenceJudgment` schema, evidence-only
      policy against the same top-5 hybrid evidence — an earlier OpenAI
      implementation was prepared but never evaluated (API-key
      availability); Gemini (`gemini-3.6-flash`) was smoke-tested
      successfully but its free tier (5 RPM / 20 RPD, confirmed via AI
      Studio) could not complete a 58-call run; Meta's Model API
      (`muse-spark-1.3`, via the official `openai` SDK pointed at a
      different `base_url`) was added as a second provider — see
      `docs/kb-sufficiency.md` for the full disclosure
- [x] Fresh 58-query holdout (`algebra_sufficiency_v3.jsonl`), hand-
      verified the same way as Phase 1G/1H, created before every
      provider change and unmodified by any of them
- [x] Resumable, paced, provider-separated evaluation execution
      (`judge_execution.py`): requests paced ≥16s apart, every
      successfully-validated judgment cached locally (git-ignored),
      keyed by provider/model/prompt/thinking/evidence/schema so a
      change silently invalidates stale entries — judge semantics
      unchanged throughout
- [x] Execution order changed to evaluate fresh v3 first
      (`--evaluator llm-judge-v3`), deferring the 167 development/
      historical LLM calls — decided before any v3 result existed from
      either provider; the retrieval-only comparison baseline is fit on
      development entirely locally (no LLM calls)
- [x] Real result: all 58 v3 cases judged successfully by Meta
      `muse-spark-1.3` (0 errors). Accuracy 0.98, precision 1.00, recall
      0.96, F1 0.98, false-accept rate 0.00, false-reject rate 0.04
      (1/58, a conservative misclassification, not an accept) — a
      substantial improvement over the frozen retrieval-only baseline on
      the same 58 cases (accuracy 0.84, FAR 0.16, FRR 0.15) and over
      Phase 1H's cross-encoder. All 7 "related but insufficient" cases
      (the core failure class this phase targeted) were correctly
      rejected. See `docs/kb-sufficiency.md` for the full breakdown.

**Phase 1J — evidence routing and web fallback (done):**

- [x] `POST /api/v1/question` — the first live request path: hybrid
      retrieval (unchanged) → Meta evidence judge (unchanged, Phase 1I)
      → cited KB chunks if sufficient, one Tavily MCP web search if not.
      Evidence routing only, no answer generation — see
      `docs/evidence-routing.md`
- [x] Dependency failures (judge unavailable/failed, retrieval
      unavailable, MCP unavailable/failed) are explicit typed outcomes;
      a judge failure never triggers web search; no raw provider error,
      traceback, or credential ever reaches the response body
- [x] Retrieval pipeline, judge, and search client are process-scoped
      singletons (`axiom/api/dependencies.py`) — a request never
      re-embeds the corpus or reloads a model; live requests are
      deliberately uncached and never touch Phase 1I's benchmark
      `JudgeCache`/`RequestPacer`
- [x] Offline tests (`tests/test_evidence_routing.py`) cover all three
      routes plus every dependency-failure sub-case via
      `app.dependency_overrides` — no network, no API keys required for
      normal `pytest`

**Phase 1K — guarded agentic answer workflow (done):**

- [x] LangGraph `StateGraph` (`axiom/workflow.py`, typed `WorkflowState`)
      with seven nodes — validate_input, retrieve_evidence, judge_evidence,
      fetch_web_fallback, generate_answer, validate_output,
      finalize_response — every routing decision a plain Python
      conditional, no autonomous tool choice, no loops; reuses Phase 1J's
      retrieval/judge/search pipeline unchanged
- [x] Deterministic input guardrail (`axiom/guardrails.py`) rejecting
      off-topic and prompt-injection requests before any retrieval/judge/
      search/generation call — not model-based, by design (see
      `docs/agentic-workflow.md` for the tradeoff)
- [x] Evidence-grounded answer generation (`MetaAnswerGenerator`, same
      Meta endpoint as the Phase 1I judge, separate frozen prompt/schema)
      and output validation (citation checks, no claimed formal
      verification) as two distinct nodes/concerns
- [x] New additive `POST /api/v1/answer`; `POST /api/v1/question`
      (Phase 1J) unchanged
- [x] Offline tests across guardrail/generation/full-graph/route layers
      (see `docs/agentic-workflow.md` for counts) — no network, no API
      keys required for normal `pytest`
- [x] Live verification: one real KB-supported answer (0 Tavily calls),
      one real web-fallback answer (exactly 1 Tavily call, 3 real
      citations), one real off-topic refusal (0 Tavily calls) — all
      passed first attempt; see `docs/agentic-workflow.md`'s "Live
      verification" section

**Phase 1L — frontend integration, human feedback, end-to-end (done):**

- [x] React Tutor mode calls `/api/v1/answer` for real through a small
      typed API client (`apps/web/src/api/client.ts`); Check My Work and
      Practice are shown disabled with an honest "needs Phase 2/3" note,
      not silently routed to the same backend
- [x] All five `AnswerResponse` statuses rendered distinctly
      (`AnswerView.tsx`), with citations, loading state, duplicate-submit
      prevention (disabled button while a request is in flight), and a
      frontend-only `network-error` case kept separate from the
      backend's own `"unavailable"`
- [x] Server-issued `answer_id` (new optional `AnswerResponse` field,
      assigned in the route handler, outside `WorkflowState`) and
      `POST /api/v1/feedback` (thumbs up/down + optional bounded
      correction, SQLite, replace-not-duplicate on resubmission,
      survives a process restart) — see `docs/feedback.md`
- [x] `FeedbackControls.tsx`, remounted per answer via `key={answerId}`,
      posts to `/api/v1/feedback` and shows success/failure
- [x] Offline tests: 23 frontend (Vitest + Testing Library — request
      submission, loading/duplicate-submit, all five answer renderings,
      feedback success/failure) plus 18 new/updated backend
      (`test_feedback.py`, `test_answer_routing.py` additions); 222
      backend tests total, all passing, no network/API keys required
- [x] Live verification: one real KB-supported answer, one real
      web-fallback answer (exactly 1 Tavily call), one real off-topic
      refusal (0 provider calls), one real feedback submission verified
      on disk after killing and reopening the backend process — see
      `docs/feedback.md`'s "Live verification"
- [x] Diagnosing why Qdrant BM25 regresses the hybrid despite comparable
      standalone lexical metrics — deferred; not required for Phase 1
      acceptance, revisit only if retrieval quality becomes a blocker
- [x] Optionally running the historical v1/v2 sets against the LLM
      judge — deferred, not required for the v3 decision

**Phase 1 acceptance checklist:**

- [x] Given a question the KB actually covers, the system answers from
      the KB and shows what it retrieved (frontend and backend)
- [x] Given a question it doesn't cover, it falls back to web/MCP and
      says so, with real citations
- [x] A deliberately off-topic or adversarial input is refused by the
      guardrail before reaching retrieval, the judge, search, or
      generation
- [x] A dependency failure produces an explicit `"unavailable"` state,
      never a fabricated explanation
- [x] A student can rate an answer and leave an optional correction;
      feedback persists across a backend restart and is never fed back
      into prompts, retrieval, evidence labels, or model parameters
- [x] This is assignment-complete agentic RAG, not verified mathematics,
      adaptive tutoring, or a production system — none of those claims
      are made anywhere in the UI or docs

## Phase 2 — Verified mathematical reasoning

**Phase 2A — deterministic verification engine (done):**

- [x] `POST /api/v1/verify` — exact SymPy verification for
      single-variable linear and quadratic equations (rational
      coefficients, real-domain roots including simple irrationals) —
      additive, no LLM, no network call, no change to any Phase 1
      route or behavior. See `docs/verification.md`
- [x] Restricted expression parser (`axiom/verification/parser.py`): an
      `ast.parse`-based walker, never `eval`/`exec`/`sympy.sympify`-on-a-
      string/`parse_expr`; default-deny on attribute/subscript access,
      arbitrary calls, other names, variable-dependent denominators,
      and bounded on literal magnitude, exponent, expression length,
      AST node count, and AST depth
- [x] Small, independently testable engine functions (`classify`,
      `solve_real`, `is_decidable_set`, `check_candidate`,
      `compare_to_solution_set`) composed by `verify()`; candidate
      validity always checked against the original equation, never a
      transformed form
- [x] Five typed outcomes (`verified_correct`, `verified_incorrect`,
      `incomplete`, `unsupported`, `indeterminate`) distinguishing a
      correct complete set, an individually-valid-but-incomplete set,
      an incorrect/extraneous candidate, no real solutions (exactly
      verified, distinguished from an infinite-solution identity), an
      out-of-scope problem, and an unresolved symbolic result
- [x] Hand-checked test matrix (108 new tests: parser, engine
      unit-level, and `TestClient` route-level) covering linear and
      quadratic correctness, degenerate/identity/contradiction cases,
      higher-degree/multivariable/denominator/injection-attempt
      rejections, duplicate roots, and the indeterminate path
      (exercised directly, since the bounded grammar doesn't reach it
      through real input) — all offline, zero Meta/Tavily calls
- [x] Measured latency: ~3 ms/call average across the supported test
      matrix; documented as not a CPU/memory isolation guarantee (no
      process isolation in this milestone — see `docs/verification.md`'s
      "Known limitations")

**Phase 2B — step-by-step verification and Check My Work (done):**

- [x] `POST /api/v1/check-work` — ordered student equation steps checked
      against the original equation, reusing Phase 2A's restricted
      parser and `solve_real`/`is_decidable_set` exactly; no second
      parser or solver added. See `docs/verification.md`
- [x] `classify_transition()` — exact SymPy set comparison distinguishing
      `equivalent`, `narrowing` (legitimate single-branch choice),
      `broadening` (introduces solutions needing independent recheck),
      `partial_overlap`, and `incompatible`; never collapses a
      non-equivalent step into "arithmetic mistake" by default.
      `partial_overlap` (drops one valid solution while gaining a
      different, unverified one) is treated with the same severity as
      `incompatible` — both set `first_invalid_step_index` — since
      losing a genuine solution is never something a legitimate
      narrowing or broadening does on its own (caught and fixed before
      publication; see the Phase 2B audit, below)
- [x] First-error localization: `first_invalid_step_index` (the first
      *provably* wrong transition) kept distinct from
      `first_non_equivalent_step_index` (the first *confirmed*
      departure from exact equivalence, of any decidable kind) and from
      `chain_preserves_solution_set` (`true`/`false`/`null` — `null`
      when only continuity gaps were seen, never assumed `true`); a
      correct final answer never retroactively validates an earlier
      wrong step
- [x] Verification-continuity handling: a step that can't be
      parsed/classified/decided is reported `"unsupported"`/
      `"indeterminate"` honestly, the next step explicitly notes the
      lost handoff, and comparisons resume from the next decidable step
      rather than cascading failure forever or silently validating past
      the gap
- [x] Final-answer checking kept independent: `final_answer` is always
      computed against the *original* equation via the unchanged
      Phase 2A `verify()`, never against the student's last written
      step, and never collapsed into the step-reasoning result
- [x] React Check My Work activated (`CheckWorkForm.tsx`): equation
      input, add/remove step and final-answer inputs (with an explicit
      "no real solutions" action distinct from omitting a final
      answer), loading/duplicate-submission prevention, numbered
      per-step results with text labels for every transition kind (not
      color alone), and the first provably incorrect step flagged
      explicitly. Practice remains disabled; Tutor mode untouched
- [x] Hand-checked test matrix (60 new tests: `classify_transition`
      unit-level, full `check_steps()` scenarios, and `TestClient`
      route-level) covering valid chains, first-incorrect-step
      localization, narrowing/broadening/partial-overlap/repeated-root/
      no-real-solution/identity/contradiction cases, the four
      final-answer/step-correctness combinations, and
      injection/denominator/degree/malformed/continuity-loss rejections
      — 46 engine + 19 route-level (65 total, updated by the
      correctness audit below), all offline, zero Meta/Tavily calls
- [x] Measured latency: ~8–20 ms/call across representative valid and
      invalid step chains (up to 3–4 steps); same no-process-isolation
      caveat as Phase 2A, documented in `docs/verification.md`
- [x] **Pre-publication correctness audit**: `partial_overlap` was
      initially bucketed with `narrowing`/`broadening` as merely
      "incomplete," not provably wrong. Audited before publishing and
      corrected: losing a genuine solution is never something a
      legitimate narrowing or broadening does without cause, so
      `partial_overlap` now sets `first_invalid_step_index` the same as
      `incompatible` does (`axiom/verification/engine.py`'s
      `check_steps()`). Continuity handling (an unsupported middle step
      can't be retroactively certified by a later clean comparison; an
      earlier confirmed error is never cleared by later steps) was
      independently re-verified and found already correct — no change
      needed there. Fixed in a separate, narrowly-scoped commit
      (`fix: classify unsound step transitions`), not folded into the
      original Phase 2B commit.

**Still open for Phase 2:**

- [ ] Problem router: classify a question into a small set of solvable
      categories (e.g., algebraic equation, calculus, arithmetic word
      problem) rather than always going straight to an LLM
- [ ] Extending the SymPy solver/verifier beyond linear/quadratic
      (higher-degree polynomials, systems of equations, inequalities)
- [ ] Wiring either verification endpoint into the Phase 1K LangGraph
      workflow or into Tutor mode's explanations — both remain
      standalone from the agentic answer path
- [ ] Confidence scoring and an explicit abstention path (the system says
      "I can't verify this" instead of presenting an unverified guess as
      fact)
- [ ] Recorded reasoning traces (what was retrieved, what was computed, what
      was verified) attached to each answer for later evaluation and debugging
- [ ] Named misconception taxonomy and tutoring policy for diagnosed
      errors (Phase 3) — Check My Work currently reports only the exact
      mathematical transition, never a claimed cognitive diagnosis

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
