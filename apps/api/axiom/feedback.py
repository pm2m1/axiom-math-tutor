"""Persistent human feedback (Phase 1L) for answers produced by
`POST /api/v1/answer`. A small SQLite store via the stdlib `sqlite3`
module — no new dependency.

Only what's needed for attribution and later review is persisted: the
answer's server-issued id, its route/status, and the feedback rating and
optional correction text. The question text and generated explanation
are deliberately NOT stored here — this is a feedback log, not a
transcript store.

Feedback is write-only from the workflow's perspective: nothing in this
module is read by `axiom/workflow.py`, `evidence_judge.py`, or
`answer_generation.py`. A human reviewing feedback later is a separate,
deliberate step (see docs/architecture.md's "Human feedback philosophy")
— this phase does not wire feedback into any automatic retraining or
prompt change.
"""

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Rating = Literal["up", "down"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = _REPO_ROOT / ".data" / "feedback.sqlite3"


def default_db_path() -> Path:
    override = os.environ.get("AXIOM_FEEDBACK_DB_PATH")
    return Path(override) if override else DEFAULT_DB_PATH


class UnknownAnswerError(ValueError):
    """Raised when feedback references an `answer_id` this process never
    issued — never silently accepted as if it were valid."""


@dataclass
class AnswerRecord:
    answer_id: str
    status: str
    created_at: str


@dataclass
class FeedbackRecord:
    feedback_id: str
    answer_id: str
    rating: Rating
    correction: str | None
    answer_status: str
    created_at: str


class FeedbackStore:
    """Each method opens and closes its own connection — this is a local
    file written a few times per request at most, not a connection-pooled
    service; `PRAGMA foreign_keys=ON` makes `feedback.answer_id` an
    enforced reference to `answers.answer_id`, so `UnknownAnswerError` is
    backed by a real constraint, not just an application-level check."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    answer_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    answer_id TEXT NOT NULL UNIQUE REFERENCES answers(answer_id),
                    rating TEXT NOT NULL,
                    correction TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record_answer(self, status: str) -> str:
        answer_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO answers (answer_id, status, created_at) VALUES (?, ?, ?)",
                (answer_id, status, created_at),
            )
        return answer_id

    def get_answer(self, answer_id: str) -> AnswerRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT answer_id, status, created_at FROM answers WHERE answer_id = ?",
                (answer_id,),
            ).fetchone()
        return AnswerRecord(*row) if row else None

    def submit_feedback(
        self, answer_id: str, rating: Rating, correction: str | None
    ) -> FeedbackRecord:
        """One feedback row per `answer_id`: resubmitting replaces the
        previous rating/correction/id rather than inserting a second row
        — a user revising their own feedback is the expected case, not a
        duplicate to reject."""
        answer = self.get_answer(answer_id)
        if answer is None:
            raise UnknownAnswerError(answer_id)
        feedback_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (feedback_id, answer_id, rating, correction, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(answer_id) DO UPDATE SET
                    feedback_id = excluded.feedback_id,
                    rating = excluded.rating,
                    correction = excluded.correction,
                    created_at = excluded.created_at
                """,
                (feedback_id, answer_id, rating, correction, created_at),
            )
        return FeedbackRecord(feedback_id, answer_id, rating, correction, answer.status, created_at)

    def get_feedback(self, answer_id: str) -> FeedbackRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT f.feedback_id, f.answer_id, f.rating, f.correction, a.status, f.created_at
                FROM feedback f JOIN answers a ON f.answer_id = a.answer_id
                WHERE f.answer_id = ?
                """,
                (answer_id,),
            ).fetchone()
        return FeedbackRecord(*row) if row else None
