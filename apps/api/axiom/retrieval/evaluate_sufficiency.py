"""KB-sufficiency evaluation (Phase 1G/1H).

    uv run python -m axiom.retrieval.evaluate_sufficiency
    uv run python -m axiom.retrieval.evaluate_sufficiency --evaluator combined

The default `retrieval` evaluator reproduces Phase 1G's original report:
three scalar retrieval-score thresholds plus a logistic regression over six
retrieval-evidence features, all selected on the development split, then
evaluated on the Phase 1G test split.

`combined` (Phase 1H) additionally scores the same selected hybrid
retriever's top-5 evidence with a query-passage cross-encoder, and reports
three approaches — retrieval-only, cross-encoder-only, and a combined
classifier — across three datasets: development, the Phase 1G test split
(now called out explicitly as previously-inspected, not an untouched
holdout — see docs/kb-sufficiency.md), and a fresh Phase 1H holdout
(data/evaluation/algebra_sufficiency_v2.jsonl) evaluated exactly once
after the model/feature/threshold design was frozen.
"""

import argparse
import io
import sys

from axiom.retrieval.base import Retriever
from axiom.retrieval.bm25 import BM25Retriever
from axiom.retrieval.dense import SentenceTransformerEmbedder
from axiom.retrieval.evaluate import load_chunks
from axiom.retrieval.evidence_support import (
    CrossEncoderScorer,
    PairScorer,
    extract_evidence_support,
)
from axiom.retrieval.hybrid import HybridRetriever
from axiom.retrieval.qdrant_store import QdrantDenseRetriever, create_client
from axiom.retrieval.sufficiency import (
    SUFFICIENCY_V2_PATH,
    ConfusionMatrix,
    LogisticSufficiencyModel,
    SufficiencyCase,
    SufficiencyFeatures,
    confusion_matrix,
    extract_features,
    gather_evidence,
    load_sufficiency_cases,
    select_threshold,
)


def _print_matrix(label: str, matrix: ConfusionMatrix) -> None:
    print(
        f"{label:<28} accuracy={matrix.accuracy:.2f} precision={matrix.precision:.2f} "
        f"recall={matrix.recall:.2f} f1={matrix.f1:.2f} "
        f"false_accept={matrix.false_accept_rate:.2f} false_reject={matrix.false_reject_rate:.2f}"
    )


def _hard_negative_rejection_rates(
    cases: list[SufficiencyCase], predictions: list[bool]
) -> tuple[float, float]:
    """Rejection rate (predicted USE_KB=False) among near-domain and among
    easy-OOD unsupported cases specifically."""

    def rejection_rate(negative_type: str) -> float:
        subset = [
            not pred
            for case, pred in zip(cases, predictions, strict=True)
            if not case.supported and case.negative_type == negative_type
        ]
        return sum(subset) / len(subset) if subset else 0.0

    return rejection_rate("near_domain"), rejection_rate("easy_ood")


def _report_approach(
    label: str,
    dev_scores: list[float],
    dev_labels: list[bool],
    eval_sets: list[tuple[str, list[float], list[bool], list[SufficiencyCase]]],
) -> None:
    selection = select_threshold(dev_labels, dev_scores)
    print(
        f"{label} — threshold={selection.threshold:.4f} "
        f"met_precision_target={selection.met_precision_target} "
        f"dev_precision={selection.dev_precision:.2f} dev_recall={selection.dev_recall:.2f}"
    )
    for name, scores, labels, cases in eval_sets:
        predictions = [score >= selection.threshold for score in scores]
        matrix = confusion_matrix(labels, predictions)
        near_rate, ood_rate = _hard_negative_rejection_rates(cases, predictions)
        _print_matrix(f"  [{name}]", matrix)
        print(
            f"  [{name}] confusion tp={matrix.true_positive} fp={matrix.false_positive} "
            f"tn={matrix.true_negative} fn={matrix.false_negative}"
        )
        print(f"  [{name}] hard negatives near_domain={near_rate:.2f} easy_ood={ood_rate:.2f}")


def _compute_features(
    cases: list[SufficiencyCase],
    bm25: Retriever,
    dense: Retriever,
    hybrid: Retriever,
    ce_scorer: PairScorer,
) -> tuple[list[SufficiencyFeatures], list[list[float]]]:
    retrieval_features = []
    ce_vectors = []
    for case in cases:
        evidence = gather_evidence(bm25, dense, hybrid, case.query)
        retrieval_features.append(extract_features(evidence))
        ce = extract_evidence_support(ce_scorer, case.query, evidence.hybrid_texts)
        ce_vectors.append(ce.vector())
    return retrieval_features, ce_vectors


def _run_retrieval_only(cases: list[SufficiencyCase]) -> None:
    dev_cases = [c for c in cases if c.split == "development"]
    test_cases = [c for c in cases if c.split == "test"]

    print("Dataset")
    print("-------")
    print(f"development  {len(dev_cases)}")
    print(f"test         {len(test_cases)}")
    print()

    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    embedder = SentenceTransformerEmbedder()
    qdrant_client = create_client()
    dense = QdrantDenseRetriever(qdrant_client, embedder)
    hybrid = HybridRetriever(bm25, dense)

    dev_features = [
        extract_features(gather_evidence(bm25, dense, hybrid, c.query)) for c in dev_cases
    ]
    test_features = [
        extract_features(gather_evidence(bm25, dense, hybrid, c.query)) for c in test_cases
    ]
    qdrant_client.close()

    dev_labels = [c.supported for c in dev_cases]
    test_labels = [c.supported for c in test_cases]

    print("Baselines (threshold selected on development only)")
    print("----------------------------------------------------")
    dev_bm25 = [f.bm25_top1 for f in dev_features]
    dev_dense = [f.dense_top1 for f in dev_features]
    dev_rrf = [f.rrf_top1 for f in dev_features]
    test_bm25 = [f.bm25_top1 for f in test_features]
    test_dense = [f.dense_top1 for f in test_features]
    test_rrf = [f.rrf_top1 for f in test_features]
    scalar_baselines = [
        ("BM25 threshold", dev_bm25, test_bm25),
        ("Dense threshold", dev_dense, test_dense),
        ("RRF threshold", dev_rrf, test_rrf),
    ]
    for label, dev_scores, test_scores in scalar_baselines:
        selection = select_threshold(dev_labels, dev_scores)
        predictions = [score >= selection.threshold for score in test_scores]
        print(
            f"{label:<16} threshold={selection.threshold:.4f} "
            f"met_precision_target={selection.met_precision_target} "
            f"dev_precision={selection.dev_precision:.2f} dev_recall={selection.dev_recall:.2f}"
        )
        _print_matrix(f"  {label} (test)", confusion_matrix(test_labels, predictions))
        near_rate, ood_rate = _hard_negative_rejection_rates(test_cases, predictions)
        print(f"  {label} hard negatives: near_domain={near_rate:.2f} easy_ood={ood_rate:.2f}")
    print()

    print("Selected multi-feature baseline (logistic regression)")
    print("-------------------------------------------------------")
    model = LogisticSufficiencyModel()
    model.fit([f.vector() for f in dev_features], dev_labels)
    dev_scores = model.decision_scores([f.vector() for f in dev_features])
    test_scores = model.decision_scores([f.vector() for f in test_features])
    selection = select_threshold(dev_labels, dev_scores)
    logistic_predictions = [score >= selection.threshold for score in test_scores]
    print(
        f"threshold={selection.threshold:.4f} "
        f"met_precision_target={selection.met_precision_target} "
        f"dev_precision={selection.dev_precision:.2f} dev_recall={selection.dev_recall:.2f}"
    )
    print()

    print("Locked test — logistic regression")
    print("------------------------------------")
    matrix = confusion_matrix(test_labels, logistic_predictions)
    print(f"accuracy         {matrix.accuracy:.2f}")
    print(f"precision        {matrix.precision:.2f}")
    print(f"recall           {matrix.recall:.2f}")
    print(f"f1               {matrix.f1:.2f}")
    print(f"false_accept_rate {matrix.false_accept_rate:.2f}")
    print(f"false_reject_rate {matrix.false_reject_rate:.2f}")
    print(
        f"confusion matrix  tp={matrix.true_positive} fp={matrix.false_positive} "
        f"tn={matrix.true_negative} fn={matrix.false_negative}"
    )
    print()

    print("Hard negatives (locked test, logistic regression)")
    print("-----------------------------------------------------")
    near_rate, ood_rate = _hard_negative_rejection_rates(test_cases, logistic_predictions)
    print(f"near-domain unsupported rejection rate  {near_rate:.2f}")
    print(f"easy OOD unsupported rejection rate     {ood_rate:.2f}")
    print()

    print("Misclassified test queries (logistic regression)")
    print("------------------------------------------------")
    diagnostics = zip(test_cases, test_features, logistic_predictions, strict=True)
    for case, features, prediction in diagnostics:
        if prediction != case.supported:
            print(f"  {case.id}: {case.query!r}")
            print(
                f"      true_supported={case.supported} predicted_use_kb={prediction} "
                f"negative_type={case.negative_type}"
            )
            print(f"      bm25_top1={features.bm25_top1:.3f} dense_top1={features.dense_top1:.3f}")


def _run_combined(cases: list[SufficiencyCase]) -> None:
    dev_cases = [c for c in cases if c.split == "development"]
    historical_test_cases = [c for c in cases if c.split == "test"]

    holdout_cases: list[SufficiencyCase] = []
    if SUFFICIENCY_V2_PATH.exists():
        holdout_cases = load_sufficiency_cases(SUFFICIENCY_V2_PATH)

    print("Dataset")
    print("-------")
    print(f"development                {len(dev_cases)}")
    print(f"historical Phase 1G test   {len(historical_test_cases)}  (previously inspected)")
    print(f"fresh Phase 1H holdout     {len(holdout_cases)}")
    print()

    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    embedder = SentenceTransformerEmbedder()
    qdrant_client = create_client()
    dense = QdrantDenseRetriever(qdrant_client, embedder)
    hybrid = HybridRetriever(bm25, dense)
    ce_scorer = CrossEncoderScorer()

    dev_retrieval, dev_ce = _compute_features(dev_cases, bm25, dense, hybrid, ce_scorer)
    hist_retrieval, hist_ce = _compute_features(
        historical_test_cases, bm25, dense, hybrid, ce_scorer
    )
    if holdout_cases:
        holdout_retrieval, holdout_ce = _compute_features(
            holdout_cases, bm25, dense, hybrid, ce_scorer
        )
    else:
        holdout_retrieval, holdout_ce = [], []
    qdrant_client.close()

    dev_labels = [c.supported for c in dev_cases]
    hist_labels = [c.supported for c in historical_test_cases]
    holdout_labels = [c.supported for c in holdout_cases]

    def eval_sets(
        hist_scores: list[float], holdout_scores: list[float]
    ) -> list[tuple[str, list[float], list[bool], list[SufficiencyCase]]]:
        sets = [("historical v1 test", hist_scores, hist_labels, historical_test_cases)]
        if holdout_cases:
            sets.append(("fresh holdout", holdout_scores, holdout_labels, holdout_cases))
        return sets

    print("=== Retrieval-only (Phase 1G) ===")
    dev_retrieval_vectors = [f.vector() for f in dev_retrieval]
    hist_retrieval_vectors = [f.vector() for f in hist_retrieval]
    holdout_retrieval_vectors = [f.vector() for f in holdout_retrieval]
    retrieval_model = LogisticSufficiencyModel()
    retrieval_model.fit(dev_retrieval_vectors, dev_labels)
    _report_approach(
        "retrieval-only logistic",
        retrieval_model.decision_scores(dev_retrieval_vectors),
        dev_labels,
        eval_sets(
            retrieval_model.decision_scores(hist_retrieval_vectors),
            retrieval_model.decision_scores(holdout_retrieval_vectors) if holdout_cases else [],
        ),
    )
    print()

    print("=== Cross-encoder-only ===")
    dev_ce_top1 = [v[0] for v in dev_ce]
    hist_ce_top1 = [v[0] for v in hist_ce]
    holdout_ce_top1 = [v[0] for v in holdout_ce]
    _report_approach(
        "ce_top1 threshold", dev_ce_top1, dev_labels, eval_sets(hist_ce_top1, holdout_ce_top1)
    )
    print()

    print("=== Combined (retrieval + cross-encoder) ===")
    dev_combined = [r + c for r, c in zip(dev_retrieval_vectors, dev_ce, strict=True)]
    hist_combined = [r + c for r, c in zip(hist_retrieval_vectors, hist_ce, strict=True)]
    holdout_combined = [r + c for r, c in zip(holdout_retrieval_vectors, holdout_ce, strict=True)]
    combined_model = LogisticSufficiencyModel()
    combined_model.fit(dev_combined, dev_labels)
    holdout_combined_scores = (
        combined_model.decision_scores(holdout_combined) if holdout_cases else []
    )
    _report_approach(
        "combined logistic",
        combined_model.decision_scores(dev_combined),
        dev_labels,
        eval_sets(combined_model.decision_scores(hist_combined), holdout_combined_scores),
    )


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", choices=["retrieval", "combined"], default="retrieval")
    args = parser.parse_args()

    cases = load_sufficiency_cases()
    if args.evaluator == "retrieval":
        _run_retrieval_only(cases)
    else:
        _run_combined(cases)


if __name__ == "__main__":
    main()
