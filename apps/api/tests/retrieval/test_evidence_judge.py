from dataclasses import dataclass

import pytest

from axiom.retrieval.evidence_judge import (
    EvidenceJudgment,
    GeminiEvidenceJudge,
    JudgeOutcome,
    MetaCredentialError,
    MetaEvidenceJudge,
    decide,
    format_evidence,
    validate_judgment,
)


def _judgment(
    verdict: str = "sufficient",
    coverage: str = "full",
    supporting_chunk_ids: list[str] | None = None,
    reason_code: str = "complete_support",
) -> EvidenceJudgment:
    return EvidenceJudgment(
        verdict=verdict,  # type: ignore[arg-type]
        coverage=coverage,  # type: ignore[arg-type]
        supporting_chunk_ids=supporting_chunk_ids or ["a"],
        reason_code=reason_code,  # type: ignore[arg-type]
        reason="test reason",
    )


def test_format_evidence_includes_chunk_ids_and_text() -> None:
    text = format_evidence(["a", "b"], ["first passage", "second passage"])
    assert "[chunk_id=a]" in text
    assert "first passage" in text
    assert "[chunk_id=b]" in text
    assert "second passage" in text


def test_validate_judgment_accepts_consistent_sufficient() -> None:
    judgment = _judgment(verdict="sufficient", coverage="full", supporting_chunk_ids=["a"])
    assert validate_judgment(judgment, ["a", "b"]) is None


def test_validate_judgment_accepts_consistent_insufficient() -> None:
    judgment = _judgment(verdict="insufficient", coverage="partial", supporting_chunk_ids=[])
    assert validate_judgment(judgment, ["a", "b"]) is None


def test_validate_judgment_rejects_sufficient_with_partial_coverage() -> None:
    judgment = _judgment(verdict="sufficient", coverage="partial")
    error = validate_judgment(judgment, ["a"])
    assert error is not None
    assert "contradictory" in error


def test_validate_judgment_rejects_insufficient_with_full_coverage() -> None:
    judgment = _judgment(verdict="insufficient", coverage="full")
    error = validate_judgment(judgment, ["a"])
    assert error is not None


def test_validate_judgment_rejects_invented_chunk_id() -> None:
    judgment = _judgment(supporting_chunk_ids=["a", "not-a-real-chunk"])
    error = validate_judgment(judgment, ["a", "b"])
    assert error is not None
    assert "invented" in error


def test_decide_maps_sufficient_to_true() -> None:
    outcome = JudgeOutcome(_judgment(verdict="sufficient"), None, None, 0.0)
    assert decide(outcome) is True


def test_decide_maps_insufficient_to_false() -> None:
    outcome = JudgeOutcome(_judgment(verdict="insufficient", coverage="none"), None, None, 0.0)
    assert decide(outcome) is False


def test_decide_maps_error_to_none() -> None:
    outcome = JudgeOutcome(None, "some API error", None, 0.0)
    assert decide(outcome) is None


# --- GeminiEvidenceJudge's response/retry handling, without any real API call ---


@dataclass
class _FakeUsageMetadata:
    prompt_token_count: int
    candidates_token_count: int
    thoughts_token_count: int
    total_token_count: int


@dataclass
class _FakeResponse:
    parsed: EvidenceJudgment | None
    usage_metadata: _FakeUsageMetadata | None
    model_version: str | None = "gemini-test-version"


def _api_error(code: int) -> Exception:
    from google.genai import errors

    return errors.APIError(code=code, response_json={"error": {"message": "fake"}})


class _FakeModels:
    """Returns responses/raises errors from a fixed script, one per call,
    so retry behavior is exercised deterministically."""

    def __init__(self, script: list[_FakeResponse | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    def generate_content(self, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _judge_with_fake_client(
    script: list[_FakeResponse | Exception], *, max_retries: int = 2
) -> tuple[GeminiEvidenceJudge, _FakeModels]:
    judge = object.__new__(GeminiEvidenceJudge)
    judge._max_retries = max_retries  # type: ignore[attr-defined]
    models = _FakeModels(script)

    class _FakeClient:
        def __init__(self) -> None:
            self.models = models

    judge._client = _FakeClient()  # type: ignore[attr-defined]
    return judge, models


def test_gemini_judge_returns_judgment_on_success() -> None:
    usage = _FakeUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=5,
        thoughts_token_count=20,
        total_token_count=35,
    )
    response = _FakeResponse(parsed=_judgment(), usage_metadata=usage)
    judge, models = _judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is not None
    assert outcome.error is None
    assert outcome.usage is not None
    assert outcome.usage.total_tokens == 35
    assert outcome.model_version == "gemini-test-version"
    assert models.calls == 1


def test_gemini_judge_records_permanent_error_without_retrying() -> None:
    judge, models = _judge_with_fake_client([_api_error(400)])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.quota_exhausted is False
    assert models.calls == 1


def test_gemini_judge_retries_on_429_then_succeeds() -> None:
    response = _FakeResponse(parsed=_judgment(), usage_metadata=None)
    judge, models = _judge_with_fake_client([_api_error(429), response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is not None
    assert models.calls == 2


def test_gemini_judge_marks_quota_exhausted_after_retries_fail() -> None:
    judge, models = _judge_with_fake_client(
        [_api_error(429), _api_error(429), _api_error(429)], max_retries=2
    )

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.quota_exhausted is True
    assert models.calls == 3


def test_gemini_judge_records_missing_parsed_output_as_error() -> None:
    response = _FakeResponse(parsed=None, usage_metadata=None)
    judge, _ = _judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.error is not None


def test_gemini_judge_records_invalid_judgment_as_error_not_a_verdict() -> None:
    invalid = _judgment(supporting_chunk_ids=["not-supplied"])
    response = _FakeResponse(parsed=invalid, usage_metadata=None)
    judge, _ = _judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.error is not None


# --- MetaEvidenceJudge credential checks (constructor, no network) ---


def test_meta_judge_rejects_unexpected_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "LLM_123_secret")
    monkeypatch.setenv("MODEL_API_BASE_URL", "https://not-meta.example.com/v1")

    with pytest.raises(MetaCredentialError):
        MetaEvidenceJudge()


def test_meta_judge_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_API_BASE_URL", "https://api.meta.ai/v1")

    with pytest.raises(MetaCredentialError):
        MetaEvidenceJudge()


# --- MetaEvidenceJudge's response/retry handling, without any real API call ---


@dataclass
class _FakeOutputTokensDetails:
    reasoning_tokens: int


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    output_tokens_details: _FakeOutputTokensDetails | None


@dataclass
class _FakeMetaResponse:
    output_parsed: EvidenceJudgment | None
    usage: _FakeUsage | None
    model: str | None = "muse-spark-test-version"


def _rate_limit_error() -> Exception:
    import httpx2
    import openai

    request = httpx2.Request("POST", "https://api.meta.ai/v1/responses")
    response = httpx2.Response(429, request=request, json={"error": {"message": "fake"}})
    return openai.RateLimitError("fake", response=response, body={"error": {"message": "fake"}})


def _billing_error() -> Exception:
    import httpx2
    import openai

    request = httpx2.Request("POST", "https://api.meta.ai/v1/responses")
    response = httpx2.Response(402, request=request, json={"error": {"message": "fake"}})
    return openai.APIStatusError("fake", response=response, body={"error": {"message": "fake"}})


class _FakeResponses:
    """Returns responses/raises errors from a fixed script, one per call,
    so retry behavior is exercised deterministically."""

    def __init__(self, script: list[_FakeMetaResponse | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    def parse(self, **kwargs: object) -> _FakeMetaResponse:
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _meta_judge_with_fake_client(
    script: list[_FakeMetaResponse | Exception], *, max_retries: int = 2
) -> tuple[MetaEvidenceJudge, _FakeResponses]:
    judge = object.__new__(MetaEvidenceJudge)
    judge._max_retries = max_retries  # type: ignore[attr-defined]
    responses = _FakeResponses(script)

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = responses

    judge._client = _FakeClient()  # type: ignore[attr-defined]
    return judge, responses


def test_meta_judge_returns_judgment_on_success() -> None:
    usage = _FakeUsage(
        input_tokens=200, output_tokens=60, total_tokens=260, output_tokens_details=None
    )
    response = _FakeMetaResponse(output_parsed=_judgment(), usage=usage)
    judge, responses = _meta_judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is not None
    assert outcome.error is None
    assert outcome.usage is not None
    assert outcome.usage.total_tokens == 260
    assert outcome.model_version == "muse-spark-test-version"
    assert responses.calls == 1


def test_meta_judge_reports_reasoning_tokens_as_thoughts_tokens() -> None:
    usage = _FakeUsage(
        input_tokens=200,
        output_tokens=60,
        total_tokens=260,
        output_tokens_details=_FakeOutputTokensDetails(reasoning_tokens=40),
    )
    response = _FakeMetaResponse(output_parsed=_judgment(), usage=usage)
    judge, _ = _meta_judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.usage is not None
    assert outcome.usage.thoughts_tokens == 40


def test_meta_judge_retries_on_rate_limit_then_succeeds() -> None:
    response = _FakeMetaResponse(output_parsed=_judgment(), usage=None)
    judge, responses = _meta_judge_with_fake_client([_rate_limit_error(), response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is not None
    assert responses.calls == 2


def test_meta_judge_marks_stoppable_after_rate_limit_retries_exhausted() -> None:
    judge, responses = _meta_judge_with_fake_client(
        [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()], max_retries=2
    )

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.quota_exhausted is True
    assert responses.calls == 3


def test_meta_judge_treats_billing_error_as_stoppable_without_retrying() -> None:
    judge, responses = _meta_judge_with_fake_client([_billing_error()])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.quota_exhausted is True
    assert responses.calls == 1  # billing is not retried, unlike rate limits


def test_meta_judge_records_missing_parsed_output_as_error() -> None:
    response = _FakeMetaResponse(output_parsed=None, usage=None)
    judge, _ = _meta_judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.error is not None


def test_meta_judge_records_invalid_judgment_as_error_not_a_verdict() -> None:
    invalid = _judgment(supporting_chunk_ids=["not-supplied"])
    response = _FakeMetaResponse(output_parsed=invalid, usage=None)
    judge, _ = _meta_judge_with_fake_client([response])

    outcome = judge.judge("query", ["a"], ["evidence text"])

    assert outcome.judgment is None
    assert outcome.error is not None
