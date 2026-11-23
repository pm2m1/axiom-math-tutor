from dataclasses import dataclass

from axiom.retrieval.answer_generation import (
    EvidenceItem,
    GeneratedAnswer,
    MetaAnswerGenerator,
    format_evidence_for_generation,
    validate_generated_answer,
)


def _answer(
    abstain: bool = False,
    explanation: str = "An explanation.",
    cited_source_ids: list[str] | None = None,
    abstain_reason: str | None = None,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        abstain=abstain,
        explanation=explanation,
        abstain_reason=abstain_reason,
        cited_source_ids=cited_source_ids if cited_source_ids is not None else ["a"],
    )


def test_format_evidence_includes_source_id_type_and_text() -> None:
    evidence = [EvidenceItem(source_id="a", source_type="kb", text="chunk text")]
    text = format_evidence_for_generation(evidence)
    assert "[source_id=a source_type=kb]" in text
    assert "chunk text" in text


# --- validate_generated_answer ---


def test_validate_accepts_well_formed_answer() -> None:
    answer = _answer(cited_source_ids=["a", "b"])
    assert validate_generated_answer(answer, ["a", "b", "c"]) is None


def test_validate_accepts_abstention_with_no_citations() -> None:
    answer = _answer(abstain=True, explanation="", cited_source_ids=[], abstain_reason="not enough")
    assert validate_generated_answer(answer, ["a"]) is None


def test_validate_rejects_empty_explanation_when_not_abstaining() -> None:
    answer = _answer(explanation="   ")
    error = validate_generated_answer(answer, ["a"])
    assert error is not None
    assert "empty" in error


def test_validate_rejects_no_citations_when_not_abstaining() -> None:
    answer = _answer(cited_source_ids=[])
    error = validate_generated_answer(answer, ["a"])
    assert error is not None
    assert "cites no sources" in error


def test_validate_rejects_invented_citation() -> None:
    answer = _answer(cited_source_ids=["a", "not-real"])
    error = validate_generated_answer(answer, ["a"])
    assert error is not None
    assert "invented" in error


def test_validate_rejects_claimed_sympy_verification() -> None:
    answer = _answer(explanation="This step is verified by sympy to be correct.")
    error = validate_generated_answer(answer, ["a"])
    assert error is not None
    assert "verification" in error


def test_validate_rejects_claimed_formal_proof() -> None:
    answer = _answer(explanation="This has been formally verified and is proven correct.")
    error = validate_generated_answer(answer, ["a"])
    assert error is not None


# --- MetaAnswerGenerator's response/retry handling, without any real API call ---


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class _FakeResponse:
    output_parsed: GeneratedAnswer | None
    usage: _FakeUsage | None
    model: str | None = "muse-spark-test-version"


def _api_error(status_code: int) -> Exception:
    import httpx2
    import openai

    request = httpx2.Request("POST", "https://api.meta.ai/v1/responses")
    response = httpx2.Response(status_code, request=request, json={"error": {"message": "fake"}})
    if status_code == 429:
        return openai.RateLimitError("fake", response=response, body={"error": {"message": "fake"}})
    return openai.APIStatusError("fake", response=response, body={"error": {"message": "fake"}})


class _FakeResponses:
    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls = 0

    def parse(self, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, _FakeResponse)
        return item


def _generator_with_fake_client(
    script: list[object], *, max_retries: int = 2
) -> tuple[MetaAnswerGenerator, _FakeResponses]:
    generator = object.__new__(MetaAnswerGenerator)
    generator._max_retries = max_retries  # type: ignore[attr-defined]
    responses = _FakeResponses(script)

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = responses

    generator._client = _FakeClient()  # type: ignore[attr-defined]
    return generator, responses


def test_generator_returns_answer_on_success() -> None:
    usage = _FakeUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    response = _FakeResponse(output_parsed=_answer(), usage=usage)
    generator, responses = _generator_with_fake_client([response])

    outcome = generator.generate("question", [EvidenceItem("a", "kb", "text")])

    assert outcome.answer is not None
    assert outcome.error is None
    assert outcome.usage is not None
    assert outcome.usage.total_tokens == 150
    assert outcome.model_version == "muse-spark-test-version"
    assert responses.calls == 1


def test_generator_retries_on_rate_limit_then_succeeds() -> None:
    response = _FakeResponse(output_parsed=_answer(), usage=None)
    generator, responses = _generator_with_fake_client([_api_error(429), response])

    outcome = generator.generate("question", [EvidenceItem("a", "kb", "text")])

    assert outcome.answer is not None
    assert responses.calls == 2


def test_generator_records_permanent_error_without_retrying() -> None:
    generator, responses = _generator_with_fake_client([_api_error(400)])

    outcome = generator.generate("question", [EvidenceItem("a", "kb", "text")])

    assert outcome.answer is None
    assert outcome.error is not None
    assert responses.calls == 1


def test_generator_records_missing_parsed_output_as_error() -> None:
    response = _FakeResponse(output_parsed=None, usage=None)
    generator, _ = _generator_with_fake_client([response])

    outcome = generator.generate("question", [EvidenceItem("a", "kb", "text")])

    assert outcome.answer is None
    assert outcome.error is not None


def test_generator_does_not_validate_citations_itself() -> None:
    """Citation validation is `validate_generated_answer`'s job, called
    explicitly by the workflow's output-validation node — the generator
    itself returns whatever structurally valid answer the API gave it,
    even one with a bad citation, so the two concerns stay separable."""
    invalid = _answer(cited_source_ids=["not-real"])
    response = _FakeResponse(output_parsed=invalid, usage=None)
    generator, _ = _generator_with_fake_client([response])

    outcome = generator.generate("question", [EvidenceItem("a", "kb", "text")])

    assert outcome.answer is not None
    assert outcome.answer.cited_source_ids == ["not-real"]
