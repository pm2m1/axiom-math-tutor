import pytest

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.models import RetrievalResult
from axiom.retrieval.sufficiency import (
    RetrievalEvidence,
    SufficiencyCase,
    confusion_matrix,
    decide,
    extract_features,
    gather_evidence,
    load_sufficiency_cases,
    select_threshold,
)


def _chunk(chunk_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="Doc",
        section=None,
        subsection=None,
        text="text",
        source_url="https://example.org/doc",
        source_name="Test",
        license="CC BY-SA 4.0",
        topic="equations",
        subtopic=None,
    )


def _hit(chunk_id: str, score: float, rank: int) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id), score=score, rank=rank)


# --- dataset parsing / labeling invariants against the real committed file ---


def test_load_sufficiency_cases_parses_real_file() -> None:
    cases = load_sufficiency_cases()

    assert len(cases) >= 100
    assert {c.id for c in cases} == {c.id for c in cases}  # loads without error
    assert len({c.id for c in cases}) == len(cases)


def test_all_cases_have_a_valid_split() -> None:
    cases = load_sufficiency_cases()
    assert {c.split for c in cases} <= {"development", "test"}


def test_all_splits_present_for_every_class() -> None:
    cases = load_sufficiency_cases()
    for split in ("development", "test"):
        split_cases = [c for c in cases if c.split == split]
        assert any(c.supported for c in split_cases)
        assert any(not c.supported and c.negative_type == "near_domain" for c in split_cases)
        assert any(not c.supported and c.negative_type == "easy_ood" for c in split_cases)


def test_supported_cases_have_support_chunk_ids() -> None:
    cases = load_sufficiency_cases()
    for case in cases:
        if case.supported:
            assert case.support_chunk_ids, case.id


def test_unsupported_cases_have_no_support_chunk_ids() -> None:
    cases = load_sufficiency_cases()
    for case in cases:
        if not case.supported:
            assert case.support_chunk_ids == [], case.id


def test_unsupported_cases_have_a_negative_type() -> None:
    cases = load_sufficiency_cases()
    for case in cases:
        if not case.supported:
            assert case.negative_type in ("near_domain", "easy_ood"), case.id


def test_supported_cases_have_no_negative_type() -> None:
    cases = load_sufficiency_cases()
    for case in cases:
        if case.supported:
            assert case.negative_type is None, case.id


def test_no_duplicate_query_text_across_dataset() -> None:
    cases = load_sufficiency_cases()
    queries = [c.query for c in cases]
    assert len(set(queries)) == len(queries)


def test_support_chunk_ids_reference_real_corpus_chunks() -> None:
    from axiom.retrieval.evaluate import load_chunks

    chunk_ids = {c.chunk_id for c in load_chunks()}
    cases = load_sufficiency_cases()
    for case in cases:
        for cid in case.support_chunk_ids:
            assert cid in chunk_ids, f"{case.id} references unknown chunk {cid}"


def test_sufficiency_case_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        SufficiencyCase.model_validate_json('{"id": "x", "query": "q"}')


class FakeRetriever:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self._hits = hits

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        return self._hits[:limit]


def test_gather_evidence_uses_bm25_dense_and_hybrid_retrievers() -> None:
    bm25 = FakeRetriever([_hit("a", 9.0, 1), _hit("b", 5.0, 2)])
    dense = FakeRetriever([_hit("a", 0.8, 1)])
    hybrid = FakeRetriever([_hit("a", 0.03, 1)])

    evidence = gather_evidence(bm25, dense, hybrid, "query", depth=5)

    assert evidence.bm25_hits_ids == ["a", "b"]
    assert evidence.bm25_scores == [9.0, 5.0]
    assert evidence.dense_hits_ids == ["a"]
    assert evidence.hybrid_hits_ids == ["a"]
    assert evidence.hybrid_scores == [0.03]
    assert evidence.hybrid_texts == ["text"]


def test_gather_evidence_respects_depth_for_hybrid_retriever() -> None:
    bm25 = FakeRetriever([])
    dense = FakeRetriever([])
    hybrid = FakeRetriever([_hit("a", 0.9, 1), _hit("b", 0.5, 2), _hit("c", 0.1, 3)])

    evidence = gather_evidence(bm25, dense, hybrid, "query", depth=2)

    assert evidence.hybrid_hits_ids == ["a", "b"]


def test_gather_evidence_handles_empty_hybrid_result() -> None:
    bm25 = FakeRetriever([])
    dense = FakeRetriever([])
    hybrid = FakeRetriever([])

    evidence = gather_evidence(bm25, dense, hybrid, "query")

    assert evidence.hybrid_hits_ids == []
    assert evidence.hybrid_texts == []


# --- feature extraction, with fake retrieval evidence ---


def _evidence(
    bm25_ids: list[str],
    bm25_scores: list[float],
    dense_ids: list[str],
    dense_scores: list[float],
    hybrid_scores: list[float] | None = None,
) -> RetrievalEvidence:
    hybrid_scores = hybrid_scores or []
    return RetrievalEvidence(
        bm25_hits_ids=bm25_ids,
        bm25_scores=bm25_scores,
        dense_hits_ids=dense_ids,
        dense_scores=dense_scores,
        hybrid_hits_ids=["a"] * len(hybrid_scores),
        hybrid_scores=hybrid_scores,
        hybrid_texts=["text"] * len(hybrid_scores),
    )


def test_extract_features_computes_top1_and_margin() -> None:
    evidence = _evidence(["a", "b", "c"], [9.0, 5.0, 1.0], ["a", "d", "e"], [0.8, 0.3, 0.1], [0.03])

    features = extract_features(evidence)

    assert features.bm25_top1 == 9.0
    assert features.bm25_margin == 4.0
    assert features.dense_top1 == 0.8
    assert features.dense_margin == pytest.approx(0.5)
    assert features.rrf_top1 == 0.03


def test_extract_features_top1_agreement_true_when_same_chunk() -> None:
    evidence = _evidence(["a", "b"], [9.0, 5.0], ["a", "c"], [0.8, 0.3], [0.03])

    assert extract_features(evidence).top1_agreement == 1.0


def test_extract_features_top1_agreement_false_when_different_chunk() -> None:
    evidence = _evidence(["a"], [9.0], ["b"], [0.8])

    assert extract_features(evidence).top1_agreement == 0.0


def test_extract_features_top5_overlap_is_jaccard() -> None:
    evidence = _evidence(["a", "b", "c"], [9.0, 5.0, 1.0], ["a", "b", "d"], [0.8, 0.5, 0.1])

    # intersection {a, b} = 2, union {a, b, c, d} = 4
    assert extract_features(evidence).top5_overlap == pytest.approx(0.5)


def test_extract_features_handles_empty_hits_without_error() -> None:
    evidence = _evidence([], [], [], [])

    features = extract_features(evidence)

    assert features.bm25_top1 == 0.0
    assert features.bm25_margin == 0.0
    assert features.top1_agreement == 0.0
    assert features.top5_overlap == 0.0
    assert features.rrf_top1 == 0.0


def test_extract_features_handles_single_hit_margin() -> None:
    evidence = _evidence(["a"], [3.0], [], [])

    assert extract_features(evidence).bm25_margin == 0.0


# --- confusion matrix / rates ---


def test_confusion_matrix_standard_metrics() -> None:
    labels = [True, True, True, False, False]
    predictions = [True, True, False, True, False]
    # tp=2 (first two), fn=1 (third), fp=1 (fourth), tn=1 (fifth)
    matrix = confusion_matrix(labels, predictions)

    assert (
        matrix.true_positive,
        matrix.false_positive,
        matrix.true_negative,
        matrix.false_negative,
    ) == (
        2,
        1,
        1,
        1,
    )
    assert matrix.accuracy == pytest.approx(3 / 5)
    assert matrix.precision == pytest.approx(2 / 3)
    assert matrix.recall == pytest.approx(2 / 3)


def test_confusion_matrix_false_accept_rate() -> None:
    # 2 unsupported cases, 1 wrongly accepted (false positive)
    labels = [False, False, True]
    predictions = [True, False, True]
    matrix = confusion_matrix(labels, predictions)

    assert matrix.false_accept_rate == pytest.approx(1 / 2)


def test_confusion_matrix_false_reject_rate() -> None:
    # 2 supported cases, 1 wrongly rejected (false negative)
    labels = [True, True, False]
    predictions = [True, False, False]
    matrix = confusion_matrix(labels, predictions)

    assert matrix.false_reject_rate == pytest.approx(1 / 2)


def test_confusion_matrix_handles_no_positive_predictions() -> None:
    labels = [True, False]
    predictions = [False, False]
    matrix = confusion_matrix(labels, predictions)

    assert matrix.precision == 0.0
    assert matrix.f1 == 0.0


# --- threshold selection ---


def test_select_threshold_meets_precision_target_when_achievable() -> None:
    labels = [True, True, True, False]
    scores = [3.0, 2.0, 1.0, 0.0]

    selection = select_threshold(labels, scores, min_precision=1.0)

    assert selection.met_precision_target is True
    assert selection.threshold == 1.0
    assert selection.dev_recall == pytest.approx(1.0)


def test_select_threshold_reports_best_effort_when_target_unreachable() -> None:
    labels = [True, False, True, False]
    scores = [3.0, 3.0, 1.0, 1.0]  # tied scores make perfect precision impossible

    selection = select_threshold(labels, scores, min_precision=1.0)

    assert selection.met_precision_target is False


def test_select_threshold_prefers_higher_recall_among_qualifying_thresholds() -> None:
    labels = [True, True, False]
    scores = [2.0, 1.0, 0.0]

    selection = select_threshold(labels, scores, min_precision=1.0)

    # both t=2.0 and t=1.0 give precision 1.0; t=1.0 gives higher recall
    assert selection.threshold == 1.0
    assert selection.dev_recall == pytest.approx(1.0)


# --- decision ---


def test_decide_accepts_when_score_meets_threshold() -> None:
    decision = decide(0.8, 0.5, "bm25-threshold")
    assert decision.use_knowledge_base is True
    assert decision.score == 0.8
    assert decision.strategy == "bm25-threshold"


def test_decide_rejects_when_score_below_threshold() -> None:
    decision = decide(0.3, 0.5, "bm25-threshold")
    assert decision.use_knowledge_base is False
