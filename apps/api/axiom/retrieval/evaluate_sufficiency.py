"""KB-sufficiency evaluation (Phase 1G).

    uv run python -m axiom.retrieval.evaluate_sufficiency

Loads the committed sufficiency benchmark (data/evaluation/
algebra_sufficiency.jsonl) and the selected retrieval pipeline
(rank_bm25.BM25Retriever + Qdrant dense + application-side RRF, all
unchanged from Phase 1F) to measure whether retrieval evidence alone can
tell a genuinely-supported algebra question from one the corpus can't
actually answer. Three scalar threshold baselines and one small logistic
regression baseline are all selected on the development split only, then
evaluated on the locked test split exactly once. See docs/kb-sufficiency.md.
"""

import io
import sys

from axiom.retrieval.bm25 import BM25Retriever
from axiom.retrieval.dense import SentenceTransformerEmbedder
from axiom.retrieval.evaluate import load_chunks
from axiom.retrieval.hybrid import HybridRetriever
from axiom.retrieval.qdrant_store import QdrantDenseRetriever, create_client
from axiom.retrieval.sufficiency import (
    ConfusionMatrix,
    LogisticSufficiencyModel,
    SufficiencyCase,
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
    easy-OOD unsupported test cases specifically."""

    def rejection_rate(negative_type: str) -> float:
        subset = [
            not pred
            for case, pred in zip(cases, predictions, strict=True)
            if not case.supported and case.negative_type == negative_type
        ]
        return sum(subset) / len(subset) if subset else 0.0

    return rejection_rate("near_domain"), rejection_rate("easy_ood")


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = load_sufficiency_cases()
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
    model.fit(dev_features, dev_labels)
    dev_scores = model.decision_scores(dev_features)
    test_scores = model.decision_scores(test_features)
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


if __name__ == "__main__":
    main()
