"""Reproducible retrieval evaluation.

    uv run python -m axiom.retrieval.evaluate --retriever bm25
    uv run python -m axiom.retrieval.evaluate --retriever dense
    uv run python -m axiom.retrieval.evaluate --retriever compare

Loads the committed corpus and labeled query set and runs it through one or
both retrievers. `compare` re-runs both (no hardcoded Phase 1A numbers) and
adds a query-type breakdown and a paired top-1 failure analysis. See
docs/retrieval-evaluation.md for what these numbers mean and their
limitations — in particular the out-of-scope sample is only 8 queries, too
few to support a confident KB-sufficiency threshold.
"""

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.base import Retriever
from axiom.retrieval.bm25 import BM25Retriever
from axiom.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder
from axiom.retrieval.metrics import hit_at_k, recall_at_k, reciprocal_rank

_REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_PATH = _REPO_ROOT / "data" / "corpus" / "algebra_chunks.jsonl"
EVAL_PATH = _REPO_ROOT / "data" / "evaluation" / "algebra_retrieval.jsonl"


class EvalCase(BaseModel):
    id: str
    query: str
    relevant_chunk_ids: list[str]
    topic: str
    query_type: str
    in_scope: bool


def load_chunks(path: Path = CORPUS_PATH) -> list[KnowledgeChunk]:
    with path.open(encoding="utf-8") as f:
        return [KnowledgeChunk.model_validate_json(line) for line in f if line.strip()]


def load_eval_cases(path: Path = EVAL_PATH) -> list[EvalCase]:
    with path.open(encoding="utf-8") as f:
        return [EvalCase.model_validate_json(line) for line in f if line.strip()]


@dataclass
class QueryResult:
    case: EvalCase
    ranked_ids: list[str]
    top_score: float | None


def run_queries(retriever: Retriever, cases: list[EvalCase]) -> list[QueryResult]:
    results = []
    for case in cases:
        hits = retriever.search(case.query, limit=5)
        results.append(
            QueryResult(
                case=case,
                ranked_ids=[hit.chunk.chunk_id for hit in hits],
                top_score=hits[0].score if hits else None,
            )
        )
    return results


@dataclass
class Metrics:
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_5: float
    mrr: float


def compute_metrics(results: list[QueryResult]) -> Metrics:
    in_scope = [r for r in results if r.case.in_scope]
    n = len(in_scope)
    hits1 = hits3 = hits5 = 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for r in in_scope:
        relevant = set(r.case.relevant_chunk_ids)
        hits1 += hit_at_k(r.ranked_ids, relevant, 1)
        hits3 += hit_at_k(r.ranked_ids, relevant, 3)
        hits5 += hit_at_k(r.ranked_ids, relevant, 5)
        recalls.append(recall_at_k(r.ranked_ids, relevant, 5))
        reciprocal_ranks.append(reciprocal_rank(r.ranked_ids, relevant))
    return Metrics(
        hit_at_1=hits1 / n,
        hit_at_3=hits3 / n,
        hit_at_5=hits5 / n,
        recall_at_5=statistics.mean(recalls),
        mrr=statistics.mean(reciprocal_ranks),
    )


def score_distribution(results: list[QueryResult], *, in_scope: bool) -> str:
    values = [
        r.top_score for r in results if r.case.in_scope == in_scope and r.top_score is not None
    ]
    if not values:
        return "n=0"
    return (
        f"n={len(values)}  min={min(values):.2f}  median={statistics.median(values):.2f}  "
        f"max={max(values):.2f}  mean={statistics.mean(values):.2f}"
    )


def breakdown_by_query_type(results: list[QueryResult]) -> dict[str, tuple[int, float, float]]:
    """Hit@1 rate and mean reciprocal rank per query_type, in-scope queries only."""
    by_type: dict[str, list[QueryResult]] = {}
    for r in results:
        if r.case.in_scope:
            by_type.setdefault(r.case.query_type, []).append(r)

    out = {}
    for query_type, group in sorted(by_type.items()):
        hit1 = sum(hit_at_k(r.ranked_ids, set(r.case.relevant_chunk_ids), 1) for r in group) / len(
            group
        )
        mrr = statistics.mean(
            reciprocal_rank(r.ranked_ids, set(r.case.relevant_chunk_ids)) for r in group
        )
        out[query_type] = (len(group), hit1, mrr)
    return out


@dataclass
class PairedFailures:
    both_correct: int
    bm25_only: int
    dense_only: int
    both_wrong: int
    disagreements: list[tuple[QueryResult, QueryResult, str]]


def paired_top1_failures(
    bm25_results: list[QueryResult], dense_results: list[QueryResult]
) -> PairedFailures:
    dense_by_id = {r.case.id: r for r in dense_results}
    both_correct = bm25_only = dense_only = both_wrong = 0
    disagreements: list[tuple[QueryResult, QueryResult, str]] = []

    for bm25_result in bm25_results:
        if not bm25_result.case.in_scope:
            continue
        dense_result = dense_by_id[bm25_result.case.id]
        relevant = set(bm25_result.case.relevant_chunk_ids)
        bm25_ok = hit_at_k(bm25_result.ranked_ids, relevant, 1)
        dense_ok = hit_at_k(dense_result.ranked_ids, relevant, 1)

        if bm25_ok and dense_ok:
            both_correct += 1
        elif bm25_ok:
            bm25_only += 1
            disagreements.append((bm25_result, dense_result, "bm25_only"))
        elif dense_ok:
            dense_only += 1
            disagreements.append((bm25_result, dense_result, "dense_only"))
        else:
            both_wrong += 1

    return PairedFailures(both_correct, bm25_only, dense_only, both_wrong, disagreements)


def _print_metrics(name: str, metrics: Metrics, results: list[QueryResult]) -> None:
    print(f"Retrieval evaluation — {name}")
    print("-" * (24 + len(name)))
    print(f"Hit@1          {metrics.hit_at_1:.2f}")
    print(f"Hit@3          {metrics.hit_at_3:.2f}")
    print(f"Hit@5          {metrics.hit_at_5:.2f}")
    print(f"Recall@5       {metrics.recall_at_5:.2f}")
    print(f"MRR            {metrics.mrr:.2f}")
    print()
    print("Top-1 score distribution, in-scope queries:")
    print("  " + score_distribution(results, in_scope=True))
    print("Top-1 score distribution, out-of-scope queries:")
    print("  " + score_distribution(results, in_scope=False))


@dataclass
class Timing:
    bm25_embed_seconds: float
    dense_embed_seconds: float
    bm25_query_seconds: float
    dense_query_seconds: float


def _print_comparison(
    bm25_results: list[QueryResult],
    dense_results: list[QueryResult],
    timing: Timing,
) -> None:
    bm25_metrics = compute_metrics(bm25_results)
    dense_metrics = compute_metrics(dense_results)

    print("Retriever comparison")
    print("--------------------")
    print(f"{'Metric':<12}{'BM25':>8}{'Dense':>8}{'Delta':>8}")
    for label, bm25_value, dense_value in [
        ("Hit@1", bm25_metrics.hit_at_1, dense_metrics.hit_at_1),
        ("Hit@3", bm25_metrics.hit_at_3, dense_metrics.hit_at_3),
        ("Hit@5", bm25_metrics.hit_at_5, dense_metrics.hit_at_5),
        ("Recall@5", bm25_metrics.recall_at_5, dense_metrics.recall_at_5),
        ("MRR", bm25_metrics.mrr, dense_metrics.mrr),
    ]:
        print(f"{label:<12}{bm25_value:>8.2f}{dense_value:>8.2f}{dense_value - bm25_value:>+8.2f}")

    print()
    print("By query type (n, Hit@1, MRR):")
    bm25_by_type = breakdown_by_query_type(bm25_results)
    dense_by_type = breakdown_by_query_type(dense_results)
    for query_type in sorted(bm25_by_type):
        n, bm25_hit1, bm25_mrr = bm25_by_type[query_type]
        _, dense_hit1, dense_mrr = dense_by_type[query_type]
        print(
            f"  {query_type:<12} n={n:<3} "
            f"bm25 hit@1={bm25_hit1:.2f} mrr={bm25_mrr:.2f}  "
            f"dense hit@1={dense_hit1:.2f} mrr={dense_mrr:.2f}"
        )

    print()
    failures = paired_top1_failures(bm25_results, dense_results)
    print("Paired top-1 outcome (in-scope queries):")
    print(f"  both correct   {failures.both_correct}")
    print(f"  bm25 only      {failures.bm25_only}")
    print(f"  dense only     {failures.dense_only}")
    print(f"  both wrong     {failures.both_wrong}")
    if failures.disagreements:
        print()
        print("Disagreements:")
        for bm25_result, dense_result, kind in failures.disagreements:
            case = bm25_result.case
            print(f"  [{kind}] {case.id}: {case.query!r}")
            print(f"      expected       {case.relevant_chunk_ids}")
            print(f"      bm25 top-1     {bm25_result.ranked_ids[:1]}")
            print(f"      dense top-1    {dense_result.ranked_ids[:1]}")

    print()
    print("Top-1 score distribution, in-scope queries:")
    print(f"  bm25   {score_distribution(bm25_results, in_scope=True)}")
    print(f"  dense  {score_distribution(dense_results, in_scope=True)}")
    print("Top-1 score distribution, out-of-scope queries:")
    print(f"  bm25   {score_distribution(bm25_results, in_scope=False)}")
    print(f"  dense  {score_distribution(dense_results, in_scope=False)}")
    print()
    print("BM25 and dense scores are on different, non-comparable scales —")
    print("the two distributions above are not meant to be read against each")
    print("other, only in-scope vs. out-of-scope within the same retriever.")
    print()
    n = len(bm25_results)
    print(
        f"Corpus embedding time — bm25: {timing.bm25_embed_seconds:.2f}s  "
        f"dense: {timing.dense_embed_seconds:.2f}s"
    )
    print(
        f"Query retrieval time, {n} queries — "
        f"bm25: {timing.bm25_query_seconds:.2f}s total "
        f"({timing.bm25_query_seconds / n * 1000:.1f}ms/query)  "
        f"dense: {timing.dense_query_seconds:.2f}s total "
        f"({timing.dense_query_seconds / n * 1000:.1f}ms/query)"
    )
    print(
        "These are single local-machine measurements, CPU-only, no batching "
        "across queries — descriptive, not a production latency claim."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=["bm25", "dense", "compare"], default="bm25")
    args = parser.parse_args()

    chunks = load_chunks()
    cases = load_eval_cases()

    if args.retriever == "compare":
        t0 = time.perf_counter()
        bm25 = BM25Retriever(chunks)
        bm25_embed_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense = DenseRetriever(chunks, SentenceTransformerEmbedder())
        dense_embed_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        bm25_results = run_queries(bm25, cases)
        bm25_query_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_results = run_queries(dense, cases)
        dense_query_seconds = time.perf_counter() - t0

        timing = Timing(
            bm25_embed_seconds, dense_embed_seconds, bm25_query_seconds, dense_query_seconds
        )
        _print_comparison(bm25_results, dense_results, timing)
        return

    retriever: Retriever
    if args.retriever == "bm25":
        retriever = BM25Retriever(chunks)
    else:
        retriever = DenseRetriever(chunks, SentenceTransformerEmbedder())

    results = run_queries(retriever, cases)
    _print_metrics(args.retriever, compute_metrics(results), results)


if __name__ == "__main__":
    main()
