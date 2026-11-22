"""Execution reliability for the Phase 1I LLM judge — request pacing and a
resumable local cache — with no effect on judge semantics (model, prompt,
thinking level, schema, and decision logic are unchanged; see
evidence_judge.py). This exists because a single evaluation session
cannot make the ~225 calls a full four-dataset benchmark needs within
this project's observed Gemini free-tier limits (5 RPM, 20 RPD); see
docs/kb-sufficiency.md.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom.retrieval.evidence_judge import EvidenceJudgment, JudgeOutcome, JudgeUsage

_REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_PATH = _REPO_ROOT / ".data" / "evaluation-cache" / "phase1i_judge_cache.jsonl"

# Keeps real request starts below the 5-requests-per-minute project limit
# with margin, targeting ~4/minute.
PACING_INTERVAL_SECONDS = 16.0

_SCHEMA_FINGERPRINT = json.dumps(EvidenceJudgment.model_json_schema(), sort_keys=True)


def cache_key(
    *,
    provider: str,
    model: str,
    prompt_version: str,
    thinking_level: str,
    query: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> str:
    """A cached judgment is only ever reused when every one of these
    matches exactly, including the provider, the evidence content itself
    (not just chunk IDs), and the output schema's own fingerprint — so a
    provider, model, prompt, thinking-level, retrieval, or schema change
    silently invalidates old entries rather than risking a stale reuse.
    Different providers never collide: a Gemini and a Meta run share one
    cache file but never share entries."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "thinking_level": thinking_level,
            "query": query,
            "chunk_ids": chunk_ids,
            "chunk_texts": chunk_texts,
            "schema": _SCHEMA_FINGERPRINT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PreparedCase:
    """One benchmark case with its evidence and cache key already
    computed — retrieval is local and free, so this is done for every
    case up front; only the Gemini call itself is paced and cached."""

    query: str
    chunk_ids: list[str]
    chunk_texts: list[str]
    key: str


class JudgeCache:
    """Append-only JSONL cache of successfully-validated judgments,
    keyed by `cache_key`. Never holds a failed, invalid, or quota-limited
    outcome — see `run_paced_judgments`."""

    def __init__(self, path: Path = CACHE_PATH) -> None:
        self._path = path
        self._entries: dict[str, JudgeOutcome] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self._entries[row["key"]] = _outcome_from_row(row)

    def get(self, key: str) -> JudgeOutcome | None:
        return self._entries.get(key)

    def put(self, key: str, outcome: JudgeOutcome) -> None:
        assert outcome.judgment is not None, "only successful judgments are cached"
        self._entries[key] = outcome
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_row_from_outcome(key, outcome)) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


def _row_from_outcome(key: str, outcome: JudgeOutcome) -> dict[str, Any]:
    assert outcome.judgment is not None
    return {
        "key": key,
        "judgment": outcome.judgment.model_dump(),
        "usage": outcome.usage.model_dump() if outcome.usage else None,
        "latency_seconds": outcome.latency_seconds,
        "model_version": outcome.model_version,
    }


def _outcome_from_row(row: dict[str, Any]) -> JudgeOutcome:
    judgment = EvidenceJudgment.model_validate(row["judgment"])
    usage = JudgeUsage.model_validate(row["usage"]) if row.get("usage") else None
    return JudgeOutcome(
        judgment=judgment,
        error=None,
        usage=usage,
        latency_seconds=row.get("latency_seconds", 0.0),
        model_version=row.get("model_version"),
        quota_exhausted=False,
    )


class RequestPacer:
    """Sleeps as needed so successful request starts are spaced at least
    `interval_seconds` apart. Real runs use the wall clock; tests inject
    a fake `sleep`/`now` so nothing actually blocks."""

    def __init__(
        self,
        interval_seconds: float = PACING_INTERVAL_SECONDS,
        *,
        sleep: Any = time.sleep,
        now: Any = time.monotonic,
    ) -> None:
        self._interval = interval_seconds
        self._sleep = sleep
        self._now = now
        self._last_start: float | None = None

    def wait(self) -> None:
        if self._last_start is not None:
            remaining = self._interval - (self._now() - self._last_start)
            if remaining > 0:
                self._sleep(remaining)
        self._last_start = self._now()


def run_paced_judgments(
    prepared: list[PreparedCase],
    judge: Any,
    cache: JudgeCache,
    pacer: RequestPacer,
) -> tuple[list[JudgeOutcome], bool]:
    """Runs one judge call per case not already cached, paced between
    calls. Returns (outcomes, quota_exhausted). Stops immediately on a
    quota-exhausted outcome, preserving every outcome — including
    already-cached ones — completed so far; the caller decides whether to
    continue to another dataset."""
    outcomes = []
    for case in prepared:
        cached = cache.get(case.key)
        if cached is not None:
            outcomes.append(cached)
            continue
        pacer.wait()
        outcome = judge.judge(case.query, case.chunk_ids, case.chunk_texts)
        outcomes.append(outcome)
        if outcome.judgment is not None:
            cache.put(case.key, outcome)
        if outcome.quota_exhausted:
            return outcomes, True
    return outcomes, False
