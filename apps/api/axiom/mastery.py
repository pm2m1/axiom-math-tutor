"""Minimal local student mastery model (Phase 3) — SQLite via stdlib
`sqlite3`, reusing exactly the pattern `axiom/feedback.py` already
established (no new dependency, one store class, one schema).

Mastery is updated *only* from a deterministic SymPy verification
outcome (Check My Work's final-answer check, or Practice mode's
grading) that resolved decisively to correct or incorrect — never from:

- an LLM-generated Tutor-mode answer (no deterministic check backs it);
- a thumbs-up/down feedback rating (that is a subjective opinion about
  helpfulness, not evidence the student solved anything, and Phase 1L's
  feedback store is explicitly documented as never read back into any
  decision — mastery included);
- a page view, a hint request, or any other passive signal;
- an `"incomplete"`/`"unsupported"`/`"indeterminate"` verification
  result — these are not decisive evidence of correct or incorrect
  performance, only of an incomplete or out-of-scope submission.

`learner_id` is an opaque string the *client* generates (e.g. a random
id stored in browser `localStorage`) and sends with each request. There
is no authentication, no password, no server-issued identity, and no
guarantee the same `learner_id` represents the same person across
devices, browsers, or sessions — this is a pseudonymous local bucket
for a single-user local deployment, not an account system. Building a
real auth system was explicitly out of scope for this phase.

Update rule (a simple, explainable exponential moving average):

    new_estimate = old_estimate + LEARNING_RATE * (outcome - old_estimate)

where `outcome` is `1.0` for a verified-correct attempt and `0.0` for a
verified-incorrect one, `LEARNING_RATE = 0.3`, and `old_estimate`
starts at `DEFAULT_MASTERY_ESTIMATE = 0.5` (an uninformative prior) for
a skill never attempted before. The estimate is clamped to `[0, 1]`.
Every attempt is logged to the `attempts` table regardless of whether
it updates the estimate, for later audit/history — only the `mastery`
table's estimate is gated on decisive, verified evidence.
"""

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LEARNING_RATE = 0.3
DEFAULT_MASTERY_ESTIMATE = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = _REPO_ROOT / ".data" / "mastery.sqlite3"


def default_db_path() -> Path:
    override = os.environ.get("AXIOM_MASTERY_DB_PATH")
    return Path(override) if override else DEFAULT_DB_PATH


@dataclass
class MasteryEstimate:
    learner_id: str
    skill: str
    estimate: float
    updated_at: str


class MasteryStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    learner_id TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    correct INTEGER,
                    hints_used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mastery (
                    learner_id TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    estimate REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (learner_id, skill)
                )
                """
            )

    def get_mastery(self, learner_id: str, skill: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT estimate FROM mastery WHERE learner_id = ? AND skill = ?",
                (learner_id, skill),
            ).fetchone()
        return float(row[0]) if row else DEFAULT_MASTERY_ESTIMATE

    def get_all_mastery(self, learner_id: str) -> list[MasteryEstimate]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT learner_id, skill, estimate, updated_at FROM mastery WHERE learner_id = ?"
                " ORDER BY skill",
                (learner_id,),
            ).fetchall()
        return [MasteryEstimate(*row) for row in rows]

    def record_attempt(
        self,
        learner_id: str,
        skill: str,
        verified: bool,
        correct: bool | None,
        hints_used: int = 0,
    ) -> MasteryEstimate:
        """Always logs the attempt. Only updates the mastery estimate
        when `verified` is `True` and `correct` is not `None` — a
        decisive, deterministic outcome. Returns the (possibly
        unchanged) current estimate either way."""
        attempt_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO attempts "
                "(attempt_id, learner_id, skill, verified, correct, hints_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt_id,
                    learner_id,
                    skill,
                    int(verified),
                    None if correct is None else int(correct),
                    hints_used,
                    created_at,
                ),
            )

        current = self.get_mastery(learner_id, skill)
        if not (verified and correct is not None):
            return MasteryEstimate(learner_id, skill, current, created_at)

        outcome = 1.0 if correct else 0.0
        new_estimate = current + LEARNING_RATE * (outcome - current)
        new_estimate = max(0.0, min(1.0, new_estimate))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mastery (learner_id, skill, estimate, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(learner_id, skill) DO UPDATE SET
                    estimate = excluded.estimate,
                    updated_at = excluded.updated_at
                """,
                (learner_id, skill, new_estimate, created_at),
            )
        return MasteryEstimate(learner_id, skill, new_estimate, created_at)
