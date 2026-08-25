# Axiom

Axiom is a research project toward a verified, mastery-aware AI tutor for
mathematics. It started as a graduate assignment on agentic RAG systems; the
goal here is to build something closer to how a good human tutor works than
to how a general-purpose chatbot answers a question.

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

This repository currently contains a project skeleton: a FastAPI backend
with health/status endpoints, a React/TypeScript frontend that renders a
static tutor UI shell, and the documentation describing where this is going.
No retrieval, no LLM calls, no verification, and no student modeling exist
yet.

| Area | Status |
| --- | --- |
| Backend skeleton (FastAPI, health/status) | Implemented |
| Frontend skeleton (mode selector, question form) | Implemented |
| Agentic RAG (guardrails, VectorDB, web fallback, LangGraph, HITL) | Planned |
| SymPy-based verification, confidence/abstention | Planned |
| Student model, misconception diagnosis, hint ladder | Planned |
| JEEBench / MathTutorBench evaluation | Planned |
| Lean-based formal verification | Research direction |

See [`docs/roadmap.md`](docs/roadmap.md) for the phased plan.

## Repository structure

```text
axiom-math-tutor/
├── apps/
│   ├── api/          FastAPI backend (Python, uv)
│   └── web/          React + TypeScript frontend (Vite)
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   └── decisions/    Architecture decision records
└── scripts/
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
`http://localhost:8000`.

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and expects the API at
`http://localhost:8000` (override with `VITE_API_BASE_URL`). A small status
indicator in the top bar shows whether it can reach the API.

### Environment variables

Copy `.env.example` to `.env` in `apps/api` if you want to override defaults.
Nothing is required to run the current skeleton.

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
```

## Roadmap (high level)

- **Phase 0 — Foundation**: repository, tooling, backend/frontend skeleton.
  *(this is where the project is today)*
- **Phase 1 — Assignment-complete agentic RAG**: guardrails, knowledge base,
  vector retrieval, web-search fallback, LangGraph workflow, human-in-the-loop
  feedback.
- **Phase 2 — Verified mathematical reasoning**: problem routing, SymPy
  verification, confidence/abstention, reasoning traces.
- **Phase 3 — Adaptive tutoring**: student model, misconception diagnosis,
  hint ladder, targeted remediation, prerequisite graph.
- **Phase 4 — Evaluation and research**: JEEBench, a custom tutoring
  benchmark, retrieval/tutoring evaluation, ablations, adversarial testing.

Details and acceptance criteria are in [`docs/roadmap.md`](docs/roadmap.md).
