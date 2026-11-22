"""KB-sufficiency evaluation (Phase 1G/1H/1I).

    uv run python -m axiom.retrieval.evaluate_sufficiency
    uv run python -m axiom.retrieval.evaluate_sufficiency --evaluator combined
    uv run python -m axiom.retrieval.evaluate_sufficiency --evaluator llm-judge
    uv run python -m axiom.retrieval.evaluate_sufficiency --evaluator llm-judge-v3
    uv run python -m axiom.retrieval.evaluate_sufficiency --evaluator llm-judge-v3 --provider meta

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

`llm-judge` (Phase 1I) asks a small structured-output LLM whether the same
top-5 evidence alone is sufficient, and compares it against the Phase 1G
retrieval-only classifier across development, both historical sets (v1
test, v2 holdout), and a fresh Phase 1I holdout
(data/evaluation/algebra_sufficiency_v3.jsonl). Requires GEMINI_API_KEY;
without it, prints what would run and stops rather than fabricating
results. On the Gemini Developer API free tier, a persistent quota
condition (429/RESOURCE_EXHAUSTED surviving retries) stops the run
immediately and reports how many judgments completed, rather than
continuing to the next dataset.

`llm-judge-v3` runs only the fresh v3 holdout against one frozen judge
(`--provider gemini`, default, or `--provider meta`), comparing it
against the retrieval-only baseline (fit and threshold-selected on
development entirely locally — no LLM calls either way). This is the
primary Phase 1I execution: the judge takes no training or threshold
fitting, so the development/historical runs aren't needed to determine
its configuration, and skipping them conserves scarce free-tier quota for
the one generalization estimate that matters. Meta's Model API
(`muse-spark-1.3`) was added as a second provider after Gemini's
free-tier daily quota was exhausted partway through v3 — before any
aggregate v3 result existed. See docs/kb-sufficiency.md for the full
disclosure. Gemini and Meta share one cache file but never share cache
entries.
"""

import argparse
import io
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from axiom.retrieval.base import Retriever
from axiom.retrieval.bm25 import BM25Retriever
from axiom.retrieval.dense import SentenceTransformerEmbedder
from axiom.retrieval.evaluate import load_chunks
from axiom.retrieval.evidence_judge import (
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    META_JUDGE_MODEL,
    META_REASONING_EFFORT,
    THINKING_LEVEL,
    GeminiEvidenceJudge,
    JudgeClient,
    JudgeOutcome,
    MetaEvidenceJudge,
    decide,
)
from axiom.retrieval.evidence_support import (
    CrossEncoderScorer,
    PairScorer,
    extract_evidence_support,
)
from axiom.retrieval.hybrid import HybridRetriever
from axiom.retrieval.judge_execution import (
    JudgeCache,
    PreparedCase,
    RequestPacer,
    cache_key,
    run_paced_judgments,
)
from axiom.retrieval.qdrant_store import QdrantDenseRetriever, create_client
from axiom.retrieval.sufficiency import (
    SUFFICIENCY_V2_PATH,
    SUFFICIENCY_V3_PATH,
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


def _evidence_and_judgments(
    name: str,
    cases: list[SufficiencyCase],
    bm25: Retriever,
    dense: Retriever,
    hybrid: Retriever,
    judge: JudgeClient,
    cache: JudgeCache,
    pacer: RequestPacer,
) -> tuple[list[SufficiencyFeatures], list[JudgeOutcome], bool]:
    """Returns (features, outcomes, quota_exhausted). Retrieval/features
    are computed for every case (local, no API cost); the judge call
    itself is skipped for any case already in `cache` under the exact
    current model/prompt/thinking/evidence identity, and paced between
    real calls — see judge_execution.py. On a persistent quota condition,
    stops immediately with only the cases completed so far — the caller
    must stop the whole run rather than continue to the next dataset."""
    features = []
    prepared = []
    for case in cases:
        evidence = gather_evidence(bm25, dense, hybrid, case.query, depth=5)
        features.append(extract_features(evidence))
        key = cache_key(
            provider="gemini",
            model=JUDGE_MODEL,
            prompt_version=JUDGE_PROMPT_VERSION,
            thinking_level=THINKING_LEVEL,
            query=case.query,
            chunk_ids=evidence.hybrid_hits_ids,
            chunk_texts=evidence.hybrid_texts,
        )
        prepared.append(
            PreparedCase(
                query=case.query,
                chunk_ids=evidence.hybrid_hits_ids,
                chunk_texts=evidence.hybrid_texts,
                key=key,
            )
        )

    cached_count = sum(1 for p in prepared if cache.get(p.key) is not None)
    print(f"{name}: cached={cached_count} remaining={len(prepared) - cached_count}")
    outcomes, quota_exhausted = run_paced_judgments(prepared, judge, cache, pacer)
    return features, outcomes, quota_exhausted


def _report_judge_dataset(
    name: str, cases: list[SufficiencyCase], outcomes: list[JudgeOutcome]
) -> None:
    kept_cases = []
    predictions = []
    for case, outcome in zip(cases, outcomes, strict=True):
        prediction = decide(outcome)
        if prediction is not None:
            kept_cases.append(case)
            predictions.append(prediction)

    errors = len(cases) - len(kept_cases)
    labels = [c.supported for c in kept_cases]
    matrix = confusion_matrix(labels, predictions)
    near_rate, ood_rate = _hard_negative_rejection_rates(kept_cases, predictions)
    _print_matrix(f"  [{name}]", matrix)
    print(
        f"  [{name}] confusion tp={matrix.true_positive} fp={matrix.false_positive} "
        f"tn={matrix.true_negative} fn={matrix.false_negative}"
    )
    print(f"  [{name}] hard negatives near_domain={near_rate:.2f} easy_ood={ood_rate:.2f}")
    print(f"  [{name}] evaluation errors {errors}/{len(cases)}")
    reason_counts = Counter(o.judgment.reason_code for o in outcomes if o.judgment is not None)
    print(f"  [{name}] reason codes {dict(reason_counts)}")


def _report_usage(outcomes: list[JudgeOutcome], cost_note: str) -> None:
    usages = [o.usage for o in outcomes if o.usage is not None]
    latencies = [o.latency_seconds for o in outcomes]
    errors = sum(1 for o in outcomes if o.judgment is None)
    total_prompt = sum(u.prompt_tokens for u in usages)
    total_candidates = sum(u.candidates_tokens for u in usages)
    total_thoughts = sum(u.thoughts_tokens for u in usages)
    total_tokens = sum(u.total_tokens for u in usages)
    n = len(outcomes)

    print("Usage / cost")
    print("------------")
    print(f"requests={n} errors={errors}")
    print(
        f"prompt_tokens={total_prompt} candidates_tokens={total_candidates} "
        f"thoughts_tokens={total_thoughts} total_tokens={total_tokens}"
    )
    print(cost_note)
    print()

    print("Latency (seconds)")
    print("-----------------")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(
            f"mean={statistics.mean(latencies):.3f} median={statistics.median(latencies):.3f} "
            f"p95={p95:.3f} total={sum(latencies):.1f}"
        )


def _run_llm_judge(cases: list[SufficiencyCase]) -> None:
    dev_cases = [c for c in cases if c.split == "development"]
    hist_v1_cases = [c for c in cases if c.split == "test"]
    hist_v2_cases = (
        load_sufficiency_cases(SUFFICIENCY_V2_PATH) if SUFFICIENCY_V2_PATH.exists() else []
    )
    v3_cases = load_sufficiency_cases(SUFFICIENCY_V3_PATH) if SUFFICIENCY_V3_PATH.exists() else []

    print("Dataset")
    print("-------")
    print(f"development                     {len(dev_cases)}")
    print(f"historical Phase 1G v1 test      {len(hist_v1_cases)}  (previously inspected)")
    print(f"historical Phase 1H v2 holdout   {len(hist_v2_cases)}  (previously inspected)")
    print(f"fresh Phase 1I v3 holdout        {len(v3_cases)}")
    print()

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set.")
        print("The Phase 1I judge integration is implemented, but the real")
        print("evaluation cannot proceed without an authorized API key.")
        print("Stopping without fabricating results.")
        return

    print(f"model={JUDGE_MODEL} prompt={JUDGE_PROMPT_VERSION} thinking_level={THINKING_LEVEL}")
    print()

    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    embedder = SentenceTransformerEmbedder()
    qdrant_client = create_client()
    dense = QdrantDenseRetriever(qdrant_client, embedder)
    hybrid = HybridRetriever(bm25, dense)
    judge: JudgeClient = GeminiEvidenceJudge()
    cache = JudgeCache()
    pacer = RequestPacer()

    named_datasets = [
        ("development", dev_cases),
        ("historical v1 test", hist_v1_cases),
        ("historical v2 holdout", hist_v2_cases),
        ("fresh v3 holdout", v3_cases),
    ]
    judge_results: dict[str, tuple[list[SufficiencyFeatures], list[JudgeOutcome]]] = {}
    for name, dataset_cases in named_datasets:
        if not dataset_cases:
            continue
        features, outcomes, quota_exhausted = _evidence_and_judgments(
            name, dataset_cases, bm25, dense, hybrid, judge, cache, pacer
        )
        judge_results[name] = (features, outcomes)
        if quota_exhausted:
            qdrant_client.close()
            print(f"Gemini quota exhausted (429) during '{name}', after retries.")
            print(f"Completed {len(outcomes)}/{len(dataset_cases)} judgments in '{name}' this run.")
            print(f"Cache now holds {len(cache)} total valid judgments (resumable next run).")
            print("Stopping without committing an incomplete Phase 1I result.")
            return
    qdrant_client.close()

    dev_features, dev_outcomes = judge_results["development"]
    dev_labels = [c.supported for c in dev_cases]

    print("=== Retrieval-only (Phase 1G) ===")
    retrieval_model = LogisticSufficiencyModel()
    dev_vectors = [f.vector() for f in dev_features]
    retrieval_model.fit(dev_vectors, dev_labels)
    retrieval_eval_sets = [
        (
            name,
            retrieval_model.decision_scores([f.vector() for f in judge_results[name][0]]),
            [c.supported for c in dataset_cases],
            dataset_cases,
        )
        for name, dataset_cases in named_datasets[1:]
        if name in judge_results
    ]
    _report_approach(
        "retrieval-only logistic",
        retrieval_model.decision_scores(dev_vectors),
        dev_labels,
        retrieval_eval_sets,
    )
    print()

    print("=== LLM evidence judge ===")
    for name, dataset_cases in named_datasets:
        if name in judge_results:
            _report_judge_dataset(name, dataset_cases, judge_results[name][1])
    print()

    all_outcomes = [o for _, outcomes in judge_results.values() for o in outcomes]
    _report_usage(
        all_outcomes, "cost_usd=0.00 (Gemini Developer API free tier, no billing enabled)"
    )


@dataclass
class _ProviderConfig:
    model: str
    thinking_level: str
    key_env: str
    judge_factory: Callable[[], JudgeClient]
    cost_note: str


_PROVIDERS: dict[str, _ProviderConfig] = {
    "gemini": _ProviderConfig(
        JUDGE_MODEL,
        THINKING_LEVEL,
        "GEMINI_API_KEY",
        GeminiEvidenceJudge,
        "cost_usd=0.00 (Gemini Developer API free tier, no billing enabled)",
    ),
    "meta": _ProviderConfig(
        META_JUDGE_MODEL,
        META_REASONING_EFFORT,
        "MODEL_API_KEY",
        MetaEvidenceJudge,
        "covered by Meta Model API free credits; $0.00 out-of-pocket this run "
        "(no per-token rate computed — not independently verified)",
    ),
}


def _run_llm_judge_v3(cases: list[SufficiencyCase], *, provider: str = "gemini") -> None:
    """Primary Phase 1I execution: the fresh v3 holdout only, against
    exactly one provider per run. The judge is a frozen zero-shot
    evaluator (no training, no threshold fitting), so the development/
    historical runs aren't required to determine it — this order was
    chosen to conserve scarce free-tier quota for the one generalization
    estimate that matters, decided before any v3 result existed. The
    retrieval-only comparison baseline is fit and threshold-selected on
    development entirely locally (no LLM calls, either provider). Gemini
    and a second provider (Meta's Model API, added after Gemini's
    free-tier daily quota was exhausted) share one cache file but never
    share cache entries — see docs/kb-sufficiency.md."""
    config = _PROVIDERS[provider]
    model = config.model
    thinking_level = config.thinking_level
    key_env = config.key_env
    judge_factory = config.judge_factory

    dev_cases = [c for c in cases if c.split == "development"]
    v3_cases = load_sufficiency_cases(SUFFICIENCY_V3_PATH) if SUFFICIENCY_V3_PATH.exists() else []

    if not v3_cases:
        print("No fresh v3 holdout found.")
        return

    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    embedder = SentenceTransformerEmbedder()
    qdrant_client = create_client()
    dense = QdrantDenseRetriever(qdrant_client, embedder)
    hybrid = HybridRetriever(bm25, dense)

    dev_features = [
        extract_features(gather_evidence(bm25, dense, hybrid, c.query)) for c in dev_cases
    ]
    v3_features = [
        extract_features(gather_evidence(bm25, dense, hybrid, c.query)) for c in v3_cases
    ]

    cache = JudgeCache()
    prepared = []
    for case in v3_cases:
        evidence = gather_evidence(bm25, dense, hybrid, case.query, depth=5)
        key = cache_key(
            provider=provider,
            model=model,
            prompt_version=JUDGE_PROMPT_VERSION,
            thinking_level=thinking_level,
            query=case.query,
            chunk_ids=evidence.hybrid_hits_ids,
            chunk_texts=evidence.hybrid_texts,
        )
        prepared.append(
            PreparedCase(
                query=case.query,
                chunk_ids=evidence.hybrid_hits_ids,
                chunk_texts=evidence.hybrid_texts,
                key=key,
            )
        )

    cached_count = sum(1 for p in prepared if cache.get(p.key) is not None)
    print(f"fresh v3 holdout: {len(v3_cases)}")
    print(f"provider={provider} model={model} prompt={JUDGE_PROMPT_VERSION}")
    print(f"thinking={thinking_level}")
    print(f"Cached: {cached_count}  Remaining: {len(prepared) - cached_count}")

    if cached_count == len(prepared):
        qdrant_client.close()
        outcomes: list[JudgeOutcome] = []
        for p in prepared:
            cached = cache.get(p.key)
            assert cached is not None
            outcomes.append(cached)
    else:
        if not os.environ.get(key_env):
            qdrant_client.close()
            print(f"{key_env} is not set. Stopping without fabricating results.")
            return
        judge: JudgeClient = judge_factory()
        pacer = RequestPacer()
        outcomes, quota_exhausted = run_paced_judgments(prepared, judge, cache, pacer)
        qdrant_client.close()
        completed = sum(1 for p in prepared if cache.get(p.key) is not None)
        if quota_exhausted or completed < len(v3_cases):
            print(f"Cached v3: {completed} / {len(v3_cases)}")
            print(f"Remaining: {len(v3_cases) - completed}")
            print("Stopping without committing an incomplete Phase 1I result.")
            return

    print()
    print("=== Retrieval-only (Phase 1G), fresh v3 holdout ===")
    dev_labels = [c.supported for c in dev_cases]
    retrieval_model = LogisticSufficiencyModel()
    dev_vectors = [f.vector() for f in dev_features]
    retrieval_model.fit(dev_vectors, dev_labels)
    v3_labels = [c.supported for c in v3_cases]
    v3_scores = retrieval_model.decision_scores([f.vector() for f in v3_features])
    _report_approach(
        "retrieval-only logistic",
        retrieval_model.decision_scores(dev_vectors),
        dev_labels,
        [("fresh v3 holdout", v3_scores, v3_labels, v3_cases)],
    )
    print()

    print(f"=== LLM evidence judge ({model}), fresh v3 holdout ===")
    _report_judge_dataset("fresh v3 holdout", v3_cases, outcomes)
    print()

    _report_usage(outcomes, config.cost_note)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluator",
        choices=["retrieval", "combined", "llm-judge", "llm-judge-v3"],
        default="retrieval",
    )
    parser.add_argument(
        "--provider",
        choices=list(_PROVIDERS),
        default="gemini",
        help="LLM provider for --evaluator llm-judge-v3 only.",
    )
    args = parser.parse_args()

    cases = load_sufficiency_cases()
    if args.evaluator == "retrieval":
        _run_retrieval_only(cases)
    elif args.evaluator == "combined":
        _run_combined(cases)
    elif args.evaluator == "llm-judge":
        _run_llm_judge(cases)
    else:
        _run_llm_judge_v3(cases, provider=args.provider)


if __name__ == "__main__":
    main()
