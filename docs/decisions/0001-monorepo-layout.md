# 0001 — Monorepo layout with per-app dependency manifests

## Status

Accepted

## Context

The project needs a Python backend (FastAPI) and a TypeScript frontend
(React/Vite), plus documentation and, later, additional services (evaluation
harnesses, curriculum data pipelines). These need to evolve together during
early development but have entirely disjoint dependency ecosystems.

## Decision

Single repository, apps under `apps/` (`apps/api`, `apps/web`), each with its
own dependency manifest (`apps/api/pyproject.toml`, `apps/web/package.json`)
rather than a root-level manifest managing both. Shared, non-app-specific
code would live under `packages/` if and when something is actually shared
between apps — no `packages/` directory is created today because nothing
needs it yet.

## Rationale

- A single repository keeps the backend and frontend in sync during a phase
  where they change together frequently (e.g., adding an endpoint and the
  UI that calls it in the same change).
- Per-app manifests avoid a root `pyproject.toml`/`package.json` that would
  either duplicate dependencies or introduce a workspace tool (uv workspaces,
  npm workspaces, Turborepo, etc.) before there's a second app of the same
  language to justify one. `apps/api` is the only Python project and
  `apps/web` the only Node project right now.
- Keeping `docs/` and `scripts/` at the root, outside `apps/`, reflects that
  they apply to the whole project rather than to one app.

## Consequences

- Running backend or frontend commands requires `cd`-ing into the relevant
  `apps/*` directory first; there is no root task runner today.
- If a second Python service is added later (e.g., a standalone evaluation
  harness), revisit whether a uv workspace is warranted at that point rather
  than assuming it now.
