# Evidence routing (Phase 1J)

`POST /api/v1/question` is the first live request path in this project.
It is evidence routing, not answer generation: given a question, it
returns either the KB chunks that support an answer or a bounded set of
web search results — never a generated answer, never a verified solution.
Answer generation, LangGraph orchestration, feedback capture, tutoring,
and any frontend UI for this endpoint are explicitly out of scope for
this phase.

## Request / response

```json
POST /api/v1/question
{"question": "What is the discriminant of a quadratic equation?"}
```

`question` is required, 1–500 characters (Pydantic `Field` constraints;
a violation is a normal FastAPI 422, not a custom error path).

```json
{
  "route": "kb",
  "kb_sources": [
    {"chunk_id": "quadratics-formula-01", "title": "Quadratic Formula",
     "section": "Discriminant", "subsection": null,
     "source_url": "https://en.wikibooks.org/wiki/Algebra/Chapter_9/Quadratic_Equation"}
  ],
  "web_sources": [],
  "reason": null,
  "detail": null
}
```

`route` discriminates the response: `"kb"` (evidence sufficient),
`"web_search"` (evidence insufficient, web fallback used), or
`"unavailable"` (a dependency failed — see below). Exactly one of
`kb_sources`/`web_sources` is populated except for `"unavailable"` or a
genuinely empty web search. `reason` and `detail` are only set for
`"unavailable"`; `detail` is always a short, hand-written, generic string
— never a raw provider error, stack trace, or credential.

## Pipeline

Unchanged from Phases 1F–1I, reused exactly as validated:

1. **Retrieval**: `rank_bm25.BM25Retriever` + `QdrantDenseRetriever` +
   application-side RRF (`HybridRetriever`, k=60, candidate depth=10) —
   the selected pipeline from `docs/retrieval-evaluation.md`. Top-5
   fused results, same depth as every sufficiency benchmark since
   Phase 1G.
2. **Evidence judge**: `MetaEvidenceJudge` (`muse-spark-1.3`, Phase 1I),
   unchanged — same prompt (`evidence-sufficiency-v1`), same schema
   (`EvidenceJudgment`), same `effort="low"` reasoning setting. The
   judge is given the question and the top-5 chunk texts only; it does
   not see this is a live request rather than a benchmark case.
3. **Routing decision**: `verdict == "sufficient"` → return the KB
   chunks the judge actually cited (`supporting_chunk_ids`), not all 5
   retrieved chunks. `verdict == "insufficient"` → call the web search
   fallback. A judge failure (construction error, API error, invalid
   structured response) is a dependency failure — it never falls
   through to web search.
4. **Web search fallback**: `TavilyMCPSearchClient`
   (`axiom/web_search.py`) — Tavily's official hosted MCP server
   (`https://mcp.tavily.com/mcp/`), reached via the official `mcp`
   Python SDK's streamable-HTTP client, `Authorization: Bearer
   $TAVILY_API_KEY`. Called with the question only — no gold labels, no
   KB context, no retrieval scores. Exactly one call per insufficient-
   evidence request; bounded to 3 results, 15s timeout. The search
   tool's exact name is discovered via `list_tools()` at connection time
   rather than hardcoded, since Tavily's own documentation is
   inconsistent about underscore vs. hyphen; result content is parsed
   defensively (`structured_content` first, text-content JSON as a
   fallback) since the exact result schema wasn't independently
   confirmed before implementation — see "Limitations" below.

## Dependency failures

Three sub-cases, each a distinct `reason`, never an unhandled 500 and
never a raw error in the response body:

| `reason` | Cause |
|---|---|
| `retrieval_error` | Retrieval pipeline failed to construct (e.g. Qdrant collection not indexed) or a search call raised |
| `judge_error` | `MODEL_API_KEY`/`MODEL_API_BASE_URL` missing or invalid, or the judge call itself failed/returned an invalid structured response |
| `search_error` | `TAVILY_API_KEY` missing, or the MCP connection/tool call failed |

A judge failure **never** triggers a web search — tested explicitly
(`test_judge_failure_is_dependency_failure_and_never_searches`,
`test_judge_not_configured_is_dependency_failure_and_never_searches`).

## Request-scoped vs. process-scoped state

`axiom/api/dependencies.py` builds the retrieval pipeline (corpus load,
embedding model, Qdrant client), the evidence judge, and the web-search
client once per process (`functools.lru_cache`) — a request never
re-embeds the corpus or reloads the dense model. None of this touches
Phase 1I's benchmark `JudgeCache` or `RequestPacer`
(`axiom/retrieval/judge_execution.py`): those exist only for the offline
58-query v3 benchmark and are never imported by the live route. A live
question is always one fresh, uncached judge call and (if needed) one
fresh, uncached search call — deliberately kept separate so a live
request never silently reuses a benchmark judgment, and running the
benchmark never affects live traffic.

## Setup

```bash
# .env (see .env.example)
MODEL_API_KEY=...          # Meta Model API, same credential as Phase 1I
MODEL_API_BASE_URL=https://api.meta.ai/v1   # optional, this is the default
TAVILY_API_KEY=...         # Tavily hosted MCP server
```

If `MODEL_API_KEY` is absent, every request returns
`{"route": "unavailable", "reason": "judge_error", ...}`. If
`TAVILY_API_KEY` is absent, a KB-sufficient question still works
normally; an insufficient-evidence question returns
`{"route": "unavailable", "reason": "search_error", ...}` instead of
reaching the web.

## Verified behavior

Offline (`tests/test_evidence_routing.py`, no network, no API keys,
`app.dependency_overrides`): KB-sufficient response and citation shape,
zero search calls on that path; insufficient-evidence response makes
exactly one search call; empty search results handled as a normal
`"web_search"` response (not an error); judge failure, judge
not-configured, MCP failure, and search not-configured all return
`"unavailable"` with the correct `reason` and never leak the underlying
error text; request validation (empty/overlong question → 422); existing
`/health` and `/api/v1/status` routes unaffected.

Live (real `MetaEvidenceJudge` + real `TavilyMCPSearchClient`, questions
outside the frozen v3 benchmark, run once this phase):

- *"Can you explain the symmetric property of equality with an
  example?"* → `route: "kb"`, cited `equations-properties-07` ("Properties
  of Equality"), **0** Tavily calls.
- *"What is Newton's law of universal gravitation?"* → `route:
  "web_search"`, **exactly 1** real Tavily MCP call, 3 real results
  returned (Wikipedia, Britannica, CK-12 — titles/URLs/snippets all
  populated).

Both checks passed on the first attempt; no fix was needed. To reproduce
manually:

```bash
cd apps/api
uv run uvicorn axiom.main:app --reload
# separately:
curl -X POST http://localhost:8000/api/v1/question \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you explain the symmetric property of equality with an example?"}'
curl -X POST http://localhost:8000/api/v1/question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Newton'"'"'s law of universal gravitation?"}'
```

The first should return `route: "kb"` with zero web calls; the second
should return `route: "web_search"` with exactly one real Tavily call.
Tavily's actual result schema (confirmed by this run): `title`, `url`,
and `content` fields per result — the defensive parsing described above
(falling back to `snippet` if `content` is absent) was exercised against
the real server's `content` field specifically.

## Limitations

- Tavily's result schema was confirmed by the live check above
  (`title`/`url`/`content` per result via `structured_content`); parsing
  still defends against a missing `content` field (falls back to
  `snippet`, tolerates absent fields as empty strings) and against a
  server that returns only text content instead of `structured_content`,
  neither of which this run exercised.
- The search tool name is discovered via `list_tools()`, not hardcoded —
  if the server exposes multiple tools containing "search", the first
  one found is used.
- No retries on web search failure (one call, as instructed); the judge
  retries transient API errors (Phase 1I, unchanged) but a live request
  pays that latency synchronously.
- No rate limiting or pacing on the live path — Phase 1I's `RequestPacer`
  is benchmark-only. A burst of live requests is not throttled here.
- No caching of live judge or search results — every request is answered
  fresh, by design (see "Request-scoped vs. process-scoped state"), so
  repeated identical questions cost a full judge call (and a search call,
  if insufficient) every time.
