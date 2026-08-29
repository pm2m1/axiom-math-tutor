import pytest

from axiom.retrieval.evaluate import (
    EvalCase,
    QueryResult,
    breakdown_by_query_type,
    compute_metrics,
    paired_top1_failures,
    score_distribution,
)


def _case(case_id: str, query_type: str, relevant: list[str], *, in_scope: bool = True) -> EvalCase:
    return EvalCase(
        id=case_id,
        query=f"query {case_id}",
        relevant_chunk_ids=relevant,
        topic="equations",
        query_type=query_type,
        in_scope=in_scope,
    )


# Four in-scope cases (two "definition", two "paraphrase") plus one
# out-of-scope case, with hand-computable rankings for two retrievers.
_CASES = [
    _case("c1", "definition", ["x1"]),
    _case("c2", "definition", ["x2"]),
    _case("c3", "paraphrase", ["x3"]),
    _case("c4", "paraphrase", ["x4"]),
    _case("c5", "definition", [], in_scope=False),
]

_BM25_RESULTS = [
    QueryResult(_CASES[0], ["x1", "y", "z"], 5.0),
    QueryResult(_CASES[1], ["y", "x2", "z"], 4.0),
    QueryResult(_CASES[2], ["y", "z", "x3"], 3.0),
    QueryResult(_CASES[3], ["x4", "y", "z"], 6.0),
    QueryResult(_CASES[4], ["y", "z"], 2.0),
]

_DENSE_RESULTS = [
    QueryResult(_CASES[0], ["y", "x1", "z"], 0.9),
    QueryResult(_CASES[1], ["x2", "y", "z"], 0.8),
    QueryResult(_CASES[2], ["x3", "y", "z"], 0.7),
    QueryResult(_CASES[3], ["y", "x4", "z"], 0.6),
    QueryResult(_CASES[4], ["y", "z"], 0.3),
]


def test_compute_metrics_matches_hand_calculation() -> None:
    metrics = compute_metrics(_BM25_RESULTS)

    assert metrics.hit_at_1 == pytest.approx(0.5)
    assert metrics.hit_at_3 == pytest.approx(1.0)
    assert metrics.hit_at_5 == pytest.approx(1.0)
    assert metrics.recall_at_5 == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx((1 + 0.5 + 1 / 3 + 1) / 4)


def test_breakdown_by_query_type_groups_and_averages() -> None:
    breakdown = breakdown_by_query_type(_DENSE_RESULTS)

    assert set(breakdown) == {"definition", "paraphrase"}
    n, hit1, mrr = breakdown["definition"]
    assert n == 2
    assert hit1 == pytest.approx(0.5)
    assert mrr == pytest.approx((0.5 + 1.0) / 2)


def test_breakdown_excludes_out_of_scope_cases() -> None:
    breakdown = breakdown_by_query_type(_BM25_RESULTS)
    n, _, _ = breakdown["definition"]
    assert n == 2  # c1, c2 — not the out-of-scope c5


def test_paired_top1_failures_counts_agreement_and_disagreement() -> None:
    failures = paired_top1_failures(_BM25_RESULTS, _DENSE_RESULTS)

    assert failures.both_correct == 0
    assert failures.bm25_only == 2
    assert failures.dense_only == 2
    assert failures.both_wrong == 0
    assert len(failures.disagreements) == 4


def test_score_distribution_splits_by_scope() -> None:
    in_scope = score_distribution(_BM25_RESULTS, in_scope=True)
    out_of_scope = score_distribution(_BM25_RESULTS, in_scope=False)

    assert "n=4" in in_scope
    assert "mean=4.50" in in_scope
    assert "n=1" in out_of_scope
