from pathlib import Path

import pytest

from axiom.retrieval.evidence_judge import EvidenceJudgment, JudgeOutcome
from axiom.retrieval.judge_execution import (
    JudgeCache,
    PreparedCase,
    RequestPacer,
    cache_key,
    run_paced_judgments,
)


def _judgment(verdict: str = "sufficient") -> EvidenceJudgment:
    return EvidenceJudgment(
        verdict=verdict,  # type: ignore[arg-type]
        coverage="full" if verdict == "sufficient" else "none",
        supporting_chunk_ids=["a"],
        reason_code="complete_support",
        reason="test reason",
    )


def _base_key_kwargs() -> dict[str, object]:
    return dict(
        provider="gemini",
        model="gemini-3.6-flash",
        prompt_version="evidence-sufficiency-v1",
        thinking_level="low",
        query="What is x?",
        chunk_ids=["a", "b"],
        chunk_texts=["text a", "text b"],
    )


# --- cache_key sensitivity ---


def test_cache_key_changes_when_provider_changes() -> None:
    base = cache_key(**_base_key_kwargs())
    changed = cache_key(**{**_base_key_kwargs(), "provider": "meta"})
    assert base != changed


def test_cache_key_changes_when_model_changes() -> None:
    base = cache_key(**_base_key_kwargs())
    changed = cache_key(**{**_base_key_kwargs(), "model": "gemini-2.5-flash"})
    assert base != changed


def test_cache_key_changes_when_prompt_version_changes() -> None:
    base = cache_key(**_base_key_kwargs())
    changed = cache_key(**{**_base_key_kwargs(), "prompt_version": "evidence-sufficiency-v2"})
    assert base != changed


def test_cache_key_changes_when_thinking_level_changes() -> None:
    base = cache_key(**_base_key_kwargs())
    changed = cache_key(**{**_base_key_kwargs(), "thinking_level": "medium"})
    assert base != changed


def test_cache_key_changes_when_evidence_text_changes() -> None:
    base = cache_key(**_base_key_kwargs())
    changed = cache_key(**{**_base_key_kwargs(), "chunk_texts": ["text a", "different text"]})
    assert base != changed


def test_cache_key_changes_when_chunk_ids_change() -> None:
    base = cache_key(**_base_key_kwargs())
    changed = cache_key(**{**_base_key_kwargs(), "chunk_ids": ["a", "c"]})
    assert base != changed


def test_cache_key_stable_for_identical_inputs() -> None:
    assert cache_key(**_base_key_kwargs()) == cache_key(**_base_key_kwargs())


# --- JudgeCache ---


def test_judge_cache_reuses_valid_cached_judgment(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    outcome = JudgeOutcome(_judgment(), None, None, 1.0, "gemini-3.6-flash")
    cache.put("key-1", outcome)

    reused = cache.get("key-1")

    assert reused is not None
    assert reused.judgment is not None
    assert reused.judgment.verdict == "sufficient"


def test_judge_cache_does_not_return_entry_for_different_key(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    cache.put("key-1", JudgeOutcome(_judgment(), None, None, 1.0))

    assert cache.get("key-2") is None


def test_judge_cache_rejects_caching_a_failed_outcome(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    failed = JudgeOutcome(None, "some error", None, 1.0)

    with pytest.raises(AssertionError):
        cache.put("key-1", failed)


def test_judge_cache_persists_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    first = JudgeCache(path=path)
    first.put("key-1", JudgeOutcome(_judgment(), None, None, 1.0, "gemini-3.6-flash"))

    reloaded = JudgeCache(path=path)

    assert len(reloaded) == 1
    cached = reloaded.get("key-1")
    assert cached is not None
    assert cached.judgment is not None and cached.judgment.verdict == "sufficient"


def test_judge_cache_starts_empty_when_no_file_exists(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "does-not-exist.jsonl")
    assert len(cache) == 0
    assert cache.get("anything") is None


# --- RequestPacer, with a fake clock/sleeper (no real waiting) ---


def test_pacer_does_not_wait_before_first_request() -> None:
    sleeps: list[float] = []
    pacer = RequestPacer(10.0, sleep=sleeps.append, now=lambda: 100.0)

    pacer.wait()

    assert sleeps == []


def test_pacer_waits_remaining_interval_between_requests() -> None:
    sleeps: list[float] = []
    clock = {"t": 0.0}
    pacer = RequestPacer(10.0, sleep=sleeps.append, now=lambda: clock["t"])

    pacer.wait()
    clock["t"] = 3.0
    pacer.wait()

    assert sleeps == [pytest.approx(7.0)]


def test_pacer_does_not_wait_if_interval_already_elapsed() -> None:
    sleeps: list[float] = []
    clock = {"t": 0.0}
    pacer = RequestPacer(10.0, sleep=sleeps.append, now=lambda: clock["t"])

    pacer.wait()
    clock["t"] = 15.0
    pacer.wait()

    assert sleeps == []


# --- run_paced_judgments: resumable execution ---


class _FakeJudge:
    def __init__(self, outcomes: list[JudgeOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def judge(self, query: str, chunk_ids: list[str], chunk_texts: list[str]) -> JudgeOutcome:
        self.calls += 1
        return self._outcomes.pop(0)


def _prepared(n: int) -> list[PreparedCase]:
    return [
        PreparedCase(query=f"q{i}", chunk_ids=["a"], chunk_texts=["text"], key=f"key-{i}")
        for i in range(n)
    ]


def test_run_paced_judgments_skips_already_cached_cases(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    cases = _prepared(2)
    cache.put(cases[0].key, JudgeOutcome(_judgment(), None, None, 1.0))
    judge = _FakeJudge([JudgeOutcome(_judgment(), None, None, 1.0)])
    pacer = RequestPacer(0.0, sleep=lambda _: None, now=lambda: 0.0)

    outcomes, quota_exhausted = run_paced_judgments(cases, judge, cache, pacer)

    assert quota_exhausted is False
    assert len(outcomes) == 2
    assert judge.calls == 1  # only the uncached case triggered a real call


def test_run_paced_judgments_does_not_cache_failed_outcomes(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    cases = _prepared(1)
    judge = _FakeJudge([JudgeOutcome(None, "transient error", None, 1.0)])
    pacer = RequestPacer(0.0, sleep=lambda _: None, now=lambda: 0.0)

    run_paced_judgments(cases, judge, cache, pacer)

    assert len(cache) == 0


def test_run_paced_judgments_stops_on_quota_exhausted_preserving_cache(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    cases = _prepared(3)
    judge = _FakeJudge(
        [
            JudgeOutcome(_judgment(), None, None, 1.0),
            JudgeOutcome(None, "quota", None, 1.0, quota_exhausted=True),
        ]
    )
    pacer = RequestPacer(0.0, sleep=lambda _: None, now=lambda: 0.0)

    outcomes, quota_exhausted = run_paced_judgments(cases, judge, cache, pacer)

    assert quota_exhausted is True
    assert len(outcomes) == 2  # stopped before the third case
    assert len(cache) == 1  # the one successful judgment before quota hit is preserved
