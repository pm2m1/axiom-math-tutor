import pytest

from axiom.retrieval.evidence_support import extract_evidence_support


class FakeScorer:
    """Maps known passages to fixed scores so aggregation is checkable by
    hand, without loading a real cross-encoder."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, query: str, passages: list[str]) -> list[float]:
        assert len(passages) == len(self._scores)
        return self._scores


def test_extract_evidence_support_computes_top1_top2_and_margin() -> None:
    scorer = FakeScorer([8.6, 3.2, 1.0])

    features = extract_evidence_support(scorer, "query", ["a", "b", "c"])

    assert features.ce_top1 == 8.6
    assert features.ce_top2 == 3.2
    assert features.ce_margin == pytest.approx(5.4)


def test_extract_evidence_support_mean_top3_averages_first_three_scores() -> None:
    scorer = FakeScorer([9.0, 6.0, 3.0, 0.0, -1.0])

    features = extract_evidence_support(scorer, "query", ["a", "b", "c", "d", "e"])

    assert features.ce_mean_top3 == pytest.approx(6.0)


def test_extract_evidence_support_mean_top3_uses_fewer_than_three_if_available() -> None:
    scorer = FakeScorer([4.0, 2.0])

    features = extract_evidence_support(scorer, "query", ["a", "b"])

    assert features.ce_mean_top3 == pytest.approx(3.0)


def test_extract_evidence_support_handles_single_passage() -> None:
    scorer = FakeScorer([5.0])

    features = extract_evidence_support(scorer, "query", ["a"])

    assert features.ce_top1 == 5.0
    assert features.ce_top2 == 0.0
    assert features.ce_margin == 0.0
    assert features.ce_mean_top3 == pytest.approx(5.0)


def test_extract_evidence_support_handles_no_passages() -> None:
    scorer = FakeScorer([])

    features = extract_evidence_support(scorer, "query", [])

    assert features.ce_top1 == 0.0
    assert features.ce_top2 == 0.0
    assert features.ce_margin == 0.0
    assert features.ce_mean_top3 == 0.0


def test_extract_evidence_support_vector_returns_top1_and_margin_only() -> None:
    scorer = FakeScorer([8.6, 3.2, 1.0])

    features = extract_evidence_support(scorer, "query", ["a", "b", "c"])

    assert features.vector() == [features.ce_top1, features.ce_margin]
