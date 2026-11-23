# Human feedback (Phase 1L)

`POST /api/v1/feedback` lets a user rate a specific answer from
`POST /api/v1/answer` (thumbs up/down, optional free-text correction) and
persists it to a local SQLite database. This closes the last open item
from Phase 1K's "Still open for Phase 1" list. Feedback is write-only
from the system's perspective: nothing in `axiom/workflow.py`,
`evidence_judge.py`, or `answer_generation.py` reads it. See
`docs/architecture.md`'s "Human feedback philosophy" for why that's a
deliberate boundary, not a missing feature.

## Request / response

```json
POST /api/v1/feedback
{"answer_id": "354b8c07-c044-4d93-be83-286bc109ea2f", "rating": "up",
 "correction": "Clear step-by-step, correct use of (b/2a)^2."}
```

```json
{
  "feedback_id": "51f444b6-1be4-44c2-ba14-c0325f5bb313",
  "answer_id": "354b8c07-c044-4d93-be83-286bc109ea2f",
  "rating": "up",
  "correction": "Clear step-by-step, correct use of (b/2a)^2.",
  "answer_status": "kb",
  "created_at": "2026-11-23T16:50:37.825147+00:00"
}
```

`rating` is `"up"` or `"down"` (a plain FastAPI 422 for anything else).
`correction` is optional, max 1000 characters. `answer_id` must refer to
a real, previously-issued answer or the request is rejected with 404
(`UnknownAnswerError`) — feedback cannot be attached to a question that
never produced an answer.

## Server-issued answer IDs

`AnswerResponse` (Phase 1K's existing schema) gained one new optional
field: `answer_id: str | None = None`. It's assigned in the
`/api/v1/answer` route handler, *after* the LangGraph workflow runs —
`FeedbackStore.record_answer(status)` generates a `uuid4` and records it
alongside the answer's status, then the handler sets
`response.answer_id` before returning. It is `None` for `"refused"`,
`"abstained"`, and `"unavailable"` — there is no answer to attach
feedback to. This keeps the identifier entirely outside
`WorkflowState`/LangGraph: the frozen Phase 1K routing semantics were not
touched to add it. Existing clients that don't read the new field are
unaffected.

## Storage

`axiom/feedback.py::FeedbackStore` — stdlib `sqlite3`, no new dependency.
Database file defaults to `.data/feedback.sqlite3` at the repo root
(same `.gitignore`-covered directory Qdrant's embedded storage already
uses), overridable via `AXIOM_FEEDBACK_DB_PATH`.

```sql
CREATE TABLE answers (
  answer_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE feedback (
  feedback_id TEXT PRIMARY KEY,
  answer_id TEXT NOT NULL UNIQUE REFERENCES answers(answer_id),
  rating TEXT NOT NULL,
  correction TEXT,
  created_at TEXT NOT NULL
);
```

`PRAGMA foreign_keys = ON` enforces the reference at the database level.
Only the answer's `status` and the feedback content are stored — never
the student's question text. The project's instructions are explicit
that feedback should persist the minimum necessary for research
attribution, not full conversation history; a question is free-text and
could contain anything a student typed, so it's deliberately left out
unless a future phase gives a specific, documented reason to add it.

Resubmitting feedback for the same `answer_id` **replaces** the previous
row rather than creating a duplicate (`INSERT ... ON CONFLICT(answer_id)
DO UPDATE`, backed by the `UNIQUE` constraint) — a student changing
their mind about a rating overwrites, it doesn't accumulate.

Feedback survives a backend restart: it's a plain file on disk, opened
fresh by `FeedbackStore.__init__` on every process start, not an
in-memory structure. Verified directly — see "Live verification" below.

## Dependency injection

`axiom/api/dependencies.py::get_feedback_store()` — one more
`@lru_cache` singleton, same pattern as every other Phase 1J/1K
dependency. `answer_with_generation()` and `submit_feedback()` both take
it via `Depends(...)`; tests override it with a `FeedbackStore` pointed
at a `tmp_path` database via `app.dependency_overrides`, so normal
`pytest` runs never touch the real `.data/feedback.sqlite3`.

## Tests

`tests/test_feedback.py` (7): answer-ID issuance, submit-and-retrieve,
restart persistence (closing and reopening the store), resubmission
replaces rather than duplicates, unknown-answer-id rejection, no
feedback vs. feedback present, and that feedback for one answer never
leaks into another's. `tests/test_answer_routing.py` adds 5 more:
feedback submitted against a real answer from a real route call (and
asserts this makes **zero** additional calls to the fake search/
generator clients), unknown-answer-id → 404, invalid rating → 422,
overlong correction → 422, empty `answer_id` → 422. All offline — no
network, no API keys.

## Frontend integration

`apps/web/src/components/FeedbackControls.tsx`, mounted with
`key={answerId}` by `AnswerView` so a new answer always gets a fresh,
reset control — rate, optionally add text, submit, see a clear
success/failure state. Only rendered for `"kb"`/`"web"` answers (the
only ones with an `answer_id`); never shown for refusals, abstentions,
unavailable responses, or network errors.

## Known limitations

- **Feedback is never read back into the system.** No prompt, retrieval
  ranking, evidence label, model parameter, or student model is touched
  by anything in this file or the route that calls it. This is a
  deliberate boundary (see `docs/architecture.md`), not an oversight to
  fix later.
- **No authentication.** Anyone who can call the API can submit
  feedback for any `answer_id` they've seen. Building a generic auth
  system was explicitly out of scope for this phase.
- **No deletion/export endpoint.** Inspecting or removing stored
  feedback is a direct SQLite operation (see the README's "Inspecting
  the feedback database" section), not an API capability — this phase's
  scope is capture, not an admin surface.

## Live verification

One real feedback submission, tied to a real `/api/v1/answer` response
from this phase's live demo (see `docs/agentic-workflow.md`-style
verification in the main Phase 1L report): submitted against a real KB
answer's `answer_id`, then the backend process was killed
(`taskkill /F`) and the SQLite file reopened in a separate Python
process, confirming the row was present on disk — not just held in the
now-dead process's memory.
