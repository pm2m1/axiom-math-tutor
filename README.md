# Axiom

Axiom is a research project toward a verified, mastery-aware AI tutor for
mathematics. It started from an academic assignment on agentic RAG systems;
the goal here is to build something closer to how a good human tutor works
than to how a general-purpose chatbot answers a question.

## The problem

Most "AI math tutor" demos are a chat interface in front of a general LLM.
That gets the arithmetic wrong often enough to matter, and it has no concept
of a student: it cannot tell you which step in a solution was invalid, what
misconception produced that step, or what the student has and hasn't
mastered over time. It also tends to hand over the full solution the moment
it's asked, which does not help anyone learn.

## Design idea

Axiom separates three concerns that are usually collapsed into a single LLM
call:

1. **Mathematical reasoning** — producing a candidate solution or checking a
   claim.
2. **Verification** — deciding, with a deterministic tool where one exists
   (SymPy today; possibly a proof assistant later) rather than another LLM
   guess, whether that claim is actually correct.
3. **Tutoring policy** — deciding how much of the verified solution the
   student should see right now.

A solver may know that `x = 3`. Whether the student sees `x = 3` immediately,
or is asked "what operation removes the `+5` from both sides?", is a
separate decision made by the tutoring policy, not a byproduct of how the
solver happened to phrase its output. See
[`docs/architecture.md`](docs/architecture.md) for the reasoning behind this
split and how it's expected to evolve.

## Status

Phase 1 (assignment-complete agentic RAG) is done: an algebra knowledge
base with measured BM25/dense/hybrid/Qdrant retrieval baselines, a Meta
LLM evidence-sufficiency judge, a Tavily web-search fallback, a LangGraph
workflow with an input guardrail and evidence-grounded answer generation,
a React frontend that calls it for real, and persisted human feedback.
Phase 2A/2B/3 add a deterministic SymPy verification engine — no LLM
involved — covering single-variable linear/quadratic equations,
derivatives, indefinite integrals, and combinatorics/probability.
Phase 3 adds a narrow, explicit-command router into Tutor mode
(`"differentiate x^2"`, not general word problems), a misconception/
hint ladder and real Practice mode built on it, and a benchmark harness
run against a **synthetic, clearly-labeled, non-authentic** dataset — a
genuine, rights-cleared JEE Main/Advanced dataset was not established
(see `docs/jee-solving.md`'s "Benchmark provenance" for exactly why).
None of this makes Tutor mode's LLM-generated explanations "verified."
See the table below, `docs/roadmap.md`'s acceptance checklists, and
`docs/verification.md` / `docs/jee-solving.md`.

| Area | Status |
| --- | --- |
| Backend (FastAPI), frontend (React/Vite) skeletons | Implemented |
| Algebra knowledge base, chunking, BM25/dense/hybrid/Qdrant retrieval baselines | Implemented |
| Guardrails, KB-first routing, web fallback | Implemented |
| LangGraph agentic workflow, evidence-grounded answer generation | Implemented |
| React frontend wired to `/api/v1/answer`; Tutor mode | Implemented |
| Human feedback (thumbs up/down, optional correction, SQLite) | Implemented |
| Deterministic SymPy verification: equations, derivatives, integrals, combinatorics | Implemented |
| Check My Work: step-by-step verification, first-error localization, hints, misconceptions | Implemented |
| Practice mode: real problem generation/grading, mastery-driven | Implemented |
| Deterministic command router into Tutor mode (explicit commands only) | Implemented |
| Student mastery model (SQLite, pseudonymous, two skills) | Implemented |
| Benchmark harness (metrics computation, frozen/dev split) | Implemented, synthetic data only |
| Genuine JEE Main/Advanced dataset | Blocked — see `docs/jee-solving.md` |
| End-to-end natural-language JEE word-problem solving | Not implemented |
| Verification wired into Tutor's LLM explanations, confidence/abstention | Planned |
| Lean-based formal verification | Research direction |

Retrieval quality against the current baseline is documented in
[`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md).

See [`docs/roadmap.md`](docs/roadmap.md) for the phased plan.

## Repository structure

```text
axiom-math-tutor/
├── apps/
│   ├── api/          FastAPI backend (Python, uv)
│   └── web/          React + TypeScript frontend (Vite)
├── data/
│   ├── manifests/     Source provenance (URL, license, attribution)
│   ├── corpus/        Normalized, chunked knowledge base (generated)
│   └── evaluation/    Labeled retrieval queries
└── docs/
    ├── architecture.md
    ├── roadmap.md
    ├── retrieval-evaluation.md
    ├── kb-sufficiency.md
    ├── evidence-routing.md
    ├── agentic-workflow.md
    ├── feedback.md
    ├── verification.md
    ├── jee-solving.md
    └── decisions/    Architecture decision records
```

## Local setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+.

### Backend

```bash
cd apps/api
uv sync
uv run uvicorn axiom.main:app --reload
```

The API serves `GET /health` and `GET /api/v1/status` on
`http://localhost:8000`, plus `POST /api/v1/question`, `POST /api/v1/answer`,
`POST /api/v1/feedback`, `POST /api/v1/verify`, `POST /api/v1/check-work`,
`POST /api/v1/verify-derivative`, `POST /api/v1/verify-integral`,
`POST /api/v1/count`, `POST /api/v1/practice/next`,
`POST /api/v1/practice/submit`, and `POST /api/v1/mastery/summary` — see
`docs/evidence-routing.md`, `docs/agentic-workflow.md`, `docs/feedback.md`,
`docs/verification.md`, and `docs/jee-solving.md`. Every verification/
practice/mastery endpoint needs no credentials and makes no network
call — pure local SymPy computation and SQLite.

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and expects the API at
`http://localhost:8000` (override with `VITE_API_BASE_URL`). A small status
indicator in the top bar shows whether it can reach the API. All three
modes are real: Tutor calls `/api/v1/answer`; Check My Work calls
`/api/v1/check-work` (steps, hints, misconception labels); Practice
calls `/api/v1/practice/next`/`submit` and `/api/v1/mastery/summary`,
with a pseudonymous `learner_id` stored in browser `localStorage` (not
an account — see `docs/jee-solving.md`).

### Environment variables

Copy `.env.example` to `.env` in `apps/api` if you want to override
defaults. Nothing is required to start either app or to exercise the
input guardrail and refusal paths. To get real `"kb"`/`"web"` answers
instead of `"unavailable"` responses, set these (see `.env.example` for
what each one is for — never commit real values):

- `MODEL_API_KEY` (and optionally `MODEL_API_BASE_URL`) — Meta's Model
  API, used by both the evidence judge and answer generation
- `TAVILY_API_KEY` — Tavily's hosted MCP server, used only when KB
  evidence is judged insufficient

Both are read directly from the process environment (not loaded from
`.env` automatically), so export them in the shell that starts
`uvicorn`. `AXIOM_FEEDBACK_DB_PATH` and `AXIOM_MASTERY_DB_PATH`
optionally override where the feedback/mastery SQLite files are written
(defaults: `.data/feedback.sqlite3` and `.data/mastery.sqlite3` at the
repo root).

### Inspecting the feedback/mastery databases

```bash
sqlite3 .data/feedback.sqlite3 "select * from feedback;"
sqlite3 .data/mastery.sqlite3 "select * from mastery;"
```

or, without the `sqlite3` CLI:

```bash
cd apps/api
uv run python -c "import sqlite3; c = sqlite3.connect('../../.data/feedback.sqlite3'); print(c.execute('select * from feedback').fetchall())"
uv run python -c "import sqlite3; c = sqlite3.connect('../../.data/mastery.sqlite3'); print(c.execute('select * from mastery').fetchall())"
```

See `docs/feedback.md` and `docs/jee-solving.md` for the schemas.

### Benchmark harness

```bash
cd apps/api
uv run python -m axiom.benchmark.harness --dataset dev
uv run python -m axiom.benchmark.harness --dataset frozen
```

Runs against a small, hand-checked, **synthetic** (not real JEE)
dataset — see `docs/jee-solving.md`'s "Benchmark provenance" before
citing any number this prints.

### Knowledge base and retrieval baseline

```bash
cd apps/api
uv run python -m axiom.knowledge.ingest    # fetch + chunk the algebra corpus
uv run python -m axiom.retrieval.evaluate --retriever compare  # BM25 vs. dense vs. hybrid
uv run python -m axiom.retrieval.index_qdrant   # one-time: embed corpus into Qdrant (embedded storage)
uv run python -m axiom.retrieval.evaluate --retriever parity   # local vs. embedded Qdrant
```

Qdrant defaults to embedded local storage; a real server is optional, via
Docker Compose:

```bash
docker compose up -d   # starts Qdrant on http://localhost:6333
cd apps/api
AXIOM_QDRANT_URL=http://localhost:6333 uv run python -m axiom.retrieval.index_qdrant
AXIOM_QDRANT_URL=http://localhost:6333 uv run python -m axiom.retrieval.evaluate --retriever server-parity
```

The chunked corpus and evaluation set are already committed under `data/`,
so `evaluate` works without running `ingest` first. See
[`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md) for what the
numbers mean.

## Tests and checks

```bash
# Backend
cd apps/api
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy axiom

# Frontend
cd apps/web
npm run build   # tsc --build && vite build
npm run lint
npm run test    # vitest, offline — mocks the API client, no real backend calls
```

## Roadmap (high level)

- **Phase 0 — Foundation**: repository, tooling, backend/frontend skeleton.
- **Phase 1 — Assignment-complete agentic RAG** *(done)*: guardrails,
  knowledge base, vector retrieval, web-search fallback, LangGraph
  workflow, a real frontend integration, and persisted human-in-the-loop
  feedback — see `docs/roadmap.md` for the full acceptance checklist.
- **Phase 2 — Verified mathematical reasoning**: problem routing,
  confidence/abstention, reasoning traces. *(Phase 2A/2B — a
  deterministic SymPy verification engine for single-variable
  linear/quadratic equations, surfaced via final-answer checking and a
  real Check My Work step-by-step checker — are done; see
  `docs/verification.md`.)*
- **Phase 3 — JEE-style reasoning, adaptive tutoring, benchmark
  foundation** *(done, with documented gaps)*: derivative/integral/
  combinatorics solvers, a narrow explicit-command router into Tutor
  mode, misconception labels + hint ladder, a real student mastery
  model, real Practice mode, and a benchmark harness — run against
  synthetic data only, since a genuine JEE dataset couldn't be legally
  established this phase. End-to-end natural-language JEE solving is
  **not** implemented. See `docs/jee-solving.md`.
- **Phase 4 — Evaluation and research**: a real JEEBench (or other
  rights-cleared) dataset run through the existing harness, a custom
  tutoring benchmark, retrieval/tutoring evaluation, ablations,
  adversarial testing.

Details and acceptance criteria are in [`docs/roadmap.md`](docs/roadmap.md).
