"""Reproducible retrieval evaluation.

    uv run python -m axiom.retrieval.evaluate --retriever bm25
    uv run python -m axiom.retrieval.evaluate --retriever dense
    uv run python -m axiom.retrieval.evaluate --retriever hybrid
    uv run python -m axiom.retrieval.evaluate --retriever qdrant-dense
    uv run python -m axiom.retrieval.evaluate --retriever qdrant-hybrid
    uv run python -m axiom.retrieval.evaluate --retriever compare
    uv run python -m axiom.retrieval.evaluate --retriever parity
    uv run python -m axiom.retrieval.evaluate --retriever server-parity

Loads the committed corpus and labeled query set and runs it through one or
more retrievers. `qdrant-dense`/`qdrant-hybrid` and `qdrant-dense`'s
underlying client target embedded local storage by default, or a Qdrant
server if `AXIOM_QDRANT_URL` is set — see compose.yaml. `compare` re-runs
bm25/dense/hybrid (Phase 1B/1C); `parity` additionally re-runs Qdrant
(always embedded) and checks it ranks the same way as its local NumPy
counterpart (Phase 1D); `server-parity` compares embedded Qdrant against a
real Qdrant server, both dense and hybrid, and requires `AXIOM_QDRANT_URL`
(Phase 1E). No numbers here are hardcoded from earlier phases. See
docs/retrieval-evaluation.md for what these numbers mean and their
limitations — in particular the out-of-scope sample is only 8 queries, too
few to support a confident KB-sufficiency threshold.
"""

import argparse
import io
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.base import Retriever
from axiom.retrieval.bm25 import BM25Retriever
from axiom.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder
from axiom.retrieval.hybrid import CANDIDATE_DEPTH, RRF_K, HybridRetriever
from axiom.retrieval.metrics import hit_at_k, recall_at_k, reciprocal_rank
from axiom.retrieval.qdrant_store import (
    QdrantDenseRetriever,
    create_client,
    create_local_client,
    create_server_client,
)

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


@dataclass
class HybridAnalysis:
    fixes_dense_failure: int
    breaks_dense_success: int
    correct_where_both_failed: int
    wrong_where_component_correct: int
    bm25_only_recovered: int
    bm25_only_total: int
    dense_only_preserved: int
    dense_only_total: int
    disagreements: list[tuple[QueryResult, QueryResult, QueryResult, str]]


def hybrid_component_analysis(
    bm25_results: list[QueryResult],
    dense_results: list[QueryResult],
    hybrid_results: list[QueryResult],
) -> HybridAnalysis:
    """How hybrid's top-1 outcome compares to its two components, and
    specifically whether it recovers Phase 1B's 2 bm25-only successes
    without losing its 9 dense-only ones."""
    dense_by_id = {r.case.id: r for r in dense_results}
    hybrid_by_id = {r.case.id: r for r in hybrid_results}
    pairwise = paired_top1_failures(bm25_results, dense_results)
    bm25_only_ids = {b.case.id for b, _, kind in pairwise.disagreements if kind == "bm25_only"}
    dense_only_ids = {b.case.id for b, _, kind in pairwise.disagreements if kind == "dense_only"}

    fixes = breaks = correct_where_both_failed = wrong_where_component_correct = 0
    disagreements: list[tuple[QueryResult, QueryResult, QueryResult, str]] = []

    for bm25_result in bm25_results:
        if not bm25_result.case.in_scope:
            continue
        dense_result = dense_by_id[bm25_result.case.id]
        hybrid_result = hybrid_by_id[bm25_result.case.id]
        relevant = set(bm25_result.case.relevant_chunk_ids)
        bm25_ok = hit_at_k(bm25_result.ranked_ids, relevant, 1)
        dense_ok = hit_at_k(dense_result.ranked_ids, relevant, 1)
        hybrid_ok = hit_at_k(hybrid_result.ranked_ids, relevant, 1)

        if not dense_ok and hybrid_ok:
            fixes += 1
        if dense_ok and not hybrid_ok:
            breaks += 1
        if not bm25_ok and not dense_ok and hybrid_ok:
            correct_where_both_failed += 1
        if (bm25_ok or dense_ok) and not hybrid_ok:
            wrong_where_component_correct += 1

        if hybrid_ok != dense_ok:
            kind = "hybrid_fixes_dense" if hybrid_ok else "hybrid_breaks_dense"
            disagreements.append((bm25_result, dense_result, hybrid_result, kind))

    def _recovered(case_id: str) -> bool:
        hybrid_result = hybrid_by_id[case_id]
        relevant = set(hybrid_result.case.relevant_chunk_ids)
        return hit_at_k(hybrid_result.ranked_ids, relevant, 1)

    return HybridAnalysis(
        fixes_dense_failure=fixes,
        breaks_dense_success=breaks,
        correct_where_both_failed=correct_where_both_failed,
        wrong_where_component_correct=wrong_where_component_correct,
        bm25_only_recovered=sum(1 for cid in bm25_only_ids if _recovered(cid)),
        bm25_only_total=len(bm25_only_ids),
        dense_only_preserved=sum(1 for cid in dense_only_ids if _recovered(cid)),
        dense_only_total=len(dense_only_ids),
        disagreements=disagreements,
    )


def explain_hybrid(query: str, bm25: Retriever, dense: Retriever) -> None:
    """Print each candidate's RRF contribution for one query, at
    CANDIDATE_DEPTH rather than the final result limit. Debugging aid for
    the comparison output, not a public API."""
    bm25_hits = {h.chunk.chunk_id: h.rank for h in bm25.search(query, limit=CANDIDATE_DEPTH)}
    dense_hits = {h.chunk.chunk_id: h.rank for h in dense.search(query, limit=CANDIDATE_DEPTH)}

    rows = []
    for chunk_id in set(bm25_hits) | set(dense_hits):
        bm25_contrib = 1.0 / (RRF_K + bm25_hits[chunk_id]) if chunk_id in bm25_hits else 0.0
        dense_contrib = 1.0 / (RRF_K + dense_hits[chunk_id]) if chunk_id in dense_hits else 0.0
        rows.append(
            (
                chunk_id,
                bm25_hits.get(chunk_id),
                dense_hits.get(chunk_id),
                bm25_contrib,
                dense_contrib,
            )
        )
    rows.sort(key=lambda r: (-(r[3] + r[4]), r[0]))

    print(f"  RRF breakdown for {query!r}:")
    for chunk_id, bm25_rank, dense_rank, bm25_c, dense_c in rows[:5]:
        print(
            f"    {chunk_id:<32} bm25_rank={bm25_rank!s:<4} dense_rank={dense_rank!s:<4} "
            f"bm25={bm25_c:.4f} dense={dense_c:.4f} total={bm25_c + dense_c:.4f}"
        )


def ranking_agreement(
    a: list[QueryResult], b: list[QueryResult]
) -> tuple[int, float, list[tuple[QueryResult, QueryResult]]]:
    """How closely two retrievers rank the same queries: top-1 agreement
    count, mean top-5 Jaccard overlap, and the cases where top-1 differs.
    Used to check ranking parity between a local and a Qdrant-backed
    retriever — aggregate metrics can be identical while individual queries
    still rank differently."""
    b_by_id = {r.case.id: r for r in b}
    top1_matches = 0
    overlaps: list[float] = []
    differing: list[tuple[QueryResult, QueryResult]] = []

    for a_result in a:
        b_result = b_by_id[a_result.case.id]
        if a_result.ranked_ids[:1] == b_result.ranked_ids[:1]:
            top1_matches += 1
        else:
            differing.append((a_result, b_result))

        a_top5, b_top5 = set(a_result.ranked_ids[:5]), set(b_result.ranked_ids[:5])
        union = a_top5 | b_top5
        overlaps.append(len(a_top5 & b_top5) / len(union) if union else 1.0)

    return top1_matches, statistics.mean(overlaps), differing


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
    hybrid_query_seconds: float


def _print_comparison(
    bm25_results: list[QueryResult],
    dense_results: list[QueryResult],
    hybrid_results: list[QueryResult],
    bm25: Retriever,
    dense: Retriever,
    timing: Timing,
) -> None:
    bm25_metrics = compute_metrics(bm25_results)
    dense_metrics = compute_metrics(dense_results)
    hybrid_metrics = compute_metrics(hybrid_results)

    print("Retriever comparison")
    print("--------------------")
    print(f"{'Metric':<12}{'BM25':>8}{'Dense':>8}{'Hybrid':>8}{'Hyb-Dns':>9}{'Hyb-BM25':>10}")
    for label, bm25_v, dense_v, hybrid_v in [
        ("Hit@1", bm25_metrics.hit_at_1, dense_metrics.hit_at_1, hybrid_metrics.hit_at_1),
        ("Hit@3", bm25_metrics.hit_at_3, dense_metrics.hit_at_3, hybrid_metrics.hit_at_3),
        ("Hit@5", bm25_metrics.hit_at_5, dense_metrics.hit_at_5, hybrid_metrics.hit_at_5),
        (
            "Recall@5",
            bm25_metrics.recall_at_5,
            dense_metrics.recall_at_5,
            hybrid_metrics.recall_at_5,
        ),
        ("MRR", bm25_metrics.mrr, dense_metrics.mrr, hybrid_metrics.mrr),
    ]:
        print(
            f"{label:<12}{bm25_v:>8.2f}{dense_v:>8.2f}{hybrid_v:>8.2f}"
            f"{hybrid_v - dense_v:>+9.2f}{hybrid_v - bm25_v:>+10.2f}"
        )

    print()
    print("By query type (n, Hit@1, MRR):")
    bm25_by_type = breakdown_by_query_type(bm25_results)
    dense_by_type = breakdown_by_query_type(dense_results)
    hybrid_by_type = breakdown_by_query_type(hybrid_results)
    for query_type in sorted(bm25_by_type):
        n, bm25_hit1, bm25_mrr = bm25_by_type[query_type]
        _, dense_hit1, dense_mrr = dense_by_type[query_type]
        _, hybrid_hit1, hybrid_mrr = hybrid_by_type[query_type]
        print(
            f"  {query_type:<12} n={n:<3} "
            f"bm25 hit@1={bm25_hit1:.2f} mrr={bm25_mrr:.2f}  "
            f"dense hit@1={dense_hit1:.2f} mrr={dense_mrr:.2f}  "
            f"hybrid hit@1={hybrid_hit1:.2f} mrr={hybrid_mrr:.2f}"
        )

    print()
    analysis = hybrid_component_analysis(bm25_results, dense_results, hybrid_results)
    print("Hybrid vs. dense (in-scope top-1):")
    print(f"  fixes a dense failure             {analysis.fixes_dense_failure}")
    print(f"  breaks a dense success            {analysis.breaks_dense_success}")
    print(f"  correct where both failed         {analysis.correct_where_both_failed}")
    print(f"  wrong where a component was right {analysis.wrong_where_component_correct}")
    print(
        f"  bm25-only successes recovered     "
        f"{analysis.bm25_only_recovered}/{analysis.bm25_only_total}"
    )
    print(
        f"  dense-only successes preserved    "
        f"{analysis.dense_only_preserved}/{analysis.dense_only_total}"
    )
    if analysis.disagreements:
        print()
        print("Hybrid/dense disagreements:")
        for bm25_result, dense_result, hybrid_result, kind in analysis.disagreements:
            case = bm25_result.case
            print(f"  [{kind}] {case.id}: {case.query!r}")
            print(f"      expected       {case.relevant_chunk_ids}")
            print(f"      bm25 top-1     {bm25_result.ranked_ids[:1]}")
            print(f"      dense top-1    {dense_result.ranked_ids[:1]}")
            print(f"      hybrid top-1   {hybrid_result.ranked_ids[:1]}")

        print()
        print("RRF breakdown, first fix and first regression:")
        for bm25_result, _, _, kind in analysis.disagreements:
            if kind == "hybrid_fixes_dense":
                explain_hybrid(bm25_result.case.query, bm25, dense)
                break
        for bm25_result, _, _, kind in analysis.disagreements:
            if kind == "hybrid_breaks_dense":
                explain_hybrid(bm25_result.case.query, bm25, dense)
                break

    print()
    print("Top-1 score distribution, in-scope queries:")
    print(f"  bm25    {score_distribution(bm25_results, in_scope=True)}")
    print(f"  dense   {score_distribution(dense_results, in_scope=True)}")
    print(f"  hybrid  {score_distribution(hybrid_results, in_scope=True)}")
    print("Top-1 score distribution, out-of-scope queries:")
    print(f"  bm25    {score_distribution(bm25_results, in_scope=False)}")
    print(f"  dense   {score_distribution(dense_results, in_scope=False)}")
    print(f"  hybrid  {score_distribution(hybrid_results, in_scope=False)}")
    print()
    print("Scores are on different, non-comparable scales across retrievers.")
    print("Hybrid's is an RRF fusion score, a function of rank and candidate")
    print("depth, not a calibrated relevance probability — none of these")
    print("should become a KB-sufficiency threshold from this data alone.")
    print()
    n = len(bm25_results)
    print(
        f"Corpus embedding time — bm25: {timing.bm25_embed_seconds:.2f}s  "
        f"dense: {timing.dense_embed_seconds:.2f}s  "
        "(hybrid reuses both retrievers and embeds nothing itself)"
    )
    print(
        f"Query retrieval time, {n} queries — "
        f"bm25: {timing.bm25_query_seconds:.2f}s total "
        f"({timing.bm25_query_seconds / n * 1000:.1f}ms/query)  "
        f"dense: {timing.dense_query_seconds:.2f}s total "
        f"({timing.dense_query_seconds / n * 1000:.1f}ms/query)  "
        f"hybrid: {timing.hybrid_query_seconds:.2f}s total "
        f"({timing.hybrid_query_seconds / n * 1000:.1f}ms/query)"
    )
    print(
        "These are single local-machine measurements, CPU-only, no batching "
        "across queries — descriptive, not a production latency claim."
    )


@dataclass
class ParityTiming:
    model_load_seconds: float
    corpus_embed_seconds: float
    qdrant_open_seconds: float
    dense_query_seconds: float
    qdrant_dense_query_seconds: float


def _print_parity(
    bm25_results: list[QueryResult],
    dense_results: list[QueryResult],
    qdrant_dense_results: list[QueryResult],
    hybrid_results: list[QueryResult],
    qdrant_hybrid_results: list[QueryResult],
    timing: ParityTiming,
) -> None:
    print("Qdrant parity comparison")
    print("------------------------")
    print(f"{'Retriever':<15}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'Recall@5':>10}{'MRR':>8}")
    for label, results in [
        ("BM25", bm25_results),
        ("Dense local", dense_results),
        ("Dense Qdrant", qdrant_dense_results),
        ("Hybrid local", hybrid_results),
        ("Hybrid Qdrant", qdrant_hybrid_results),
    ]:
        m = compute_metrics(results)
        print(
            f"{label:<15}{m.hit_at_1:>8.2f}{m.hit_at_3:>8.2f}{m.hit_at_5:>8.2f}"
            f"{m.recall_at_5:>10.2f}{m.mrr:>8.2f}"
        )

    n = len(dense_results)

    print()
    top1, overlap, differing = ranking_agreement(dense_results, qdrant_dense_results)
    print(f"Dense local vs. Qdrant dense — top-1 agreement: {top1}/{n}")
    print(f"  mean top-5 overlap (Jaccard): {overlap:.2f}")
    if differing:
        print("  top-1 differs:")
        for local_r, qdrant_r in differing:
            print(
                f"    {local_r.case.id}: local={local_r.ranked_ids[:1]} "
                f"qdrant={qdrant_r.ranked_ids[:1]}"
            )

    print()
    top1, overlap, differing = ranking_agreement(hybrid_results, qdrant_hybrid_results)
    print(f"Hybrid local vs. Qdrant hybrid — top-1 agreement: {top1}/{n}")
    print(f"  mean top-5 overlap (Jaccard): {overlap:.2f}")
    if differing:
        print("  top-1 differs:")
        for local_r, qdrant_r in differing:
            print(
                f"    {local_r.case.id}: local={local_r.ranked_ids[:1]} "
                f"qdrant={qdrant_r.ranked_ids[:1]}"
            )

    print()
    print(
        "Model load time (shared by dense local and Qdrant dense): "
        f"{timing.model_load_seconds:.2f}s"
    )
    print(
        f"Corpus embedding time, local dense only: {timing.corpus_embed_seconds:.2f}s "
        "(Qdrant dense does not repeat this — it queries the collection built by index_qdrant.py)"
    )
    print(f"Qdrant collection open time: {timing.qdrant_open_seconds:.2f}s")
    print(
        f"Query retrieval time, {n} queries — "
        f"dense local: {timing.dense_query_seconds:.2f}s total  "
        f"dense qdrant: {timing.qdrant_dense_query_seconds:.2f}s total"
    )
    print(
        "These are single local-machine measurements, CPU-only — descriptive, "
        "not a production latency claim."
    )


@dataclass
class ServerParityTiming:
    embedded_open_seconds: float
    server_connect_seconds: float
    embedded_query_seconds: float
    server_query_seconds: float


def _print_server_parity(
    bm25_results: list[QueryResult],
    embedded_dense_results: list[QueryResult],
    server_dense_results: list[QueryResult],
    embedded_hybrid_results: list[QueryResult],
    server_hybrid_results: list[QueryResult],
    timing: ServerParityTiming,
) -> None:
    """Embedded local-mode Qdrant vs. a real Qdrant server — Phase 1E. Not
    an HNSW/ANN benchmark: this 40-point corpus is far too small for that,
    and both deployments use Qdrant's normal defaults (no forced
    full_scan_threshold or HNSW parameters). See docs/retrieval-evaluation.md."""
    print("Qdrant server parity comparison")
    print("--------------------------------")
    print(f"{'Retriever':<16}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'Recall@5':>10}{'MRR':>8}")
    for label, results in [
        ("BM25", bm25_results),
        ("Dense embedded", embedded_dense_results),
        ("Dense server", server_dense_results),
        ("Hybrid embedded", embedded_hybrid_results),
        ("Hybrid server", server_hybrid_results),
    ]:
        m = compute_metrics(results)
        print(
            f"{label:<16}{m.hit_at_1:>8.2f}{m.hit_at_3:>8.2f}{m.hit_at_5:>8.2f}"
            f"{m.recall_at_5:>10.2f}{m.mrr:>8.2f}"
        )

    n = len(embedded_dense_results)

    print()
    top1, overlap, differing = ranking_agreement(embedded_dense_results, server_dense_results)
    print(f"Embedded dense vs. server dense — top-1 agreement: {top1}/{n}")
    print(f"  mean top-5 overlap (Jaccard): {overlap:.2f}")
    if differing:
        print("  top-1 differs:")
        for embedded_r, server_r in differing:
            print(
                f"    {embedded_r.case.id}: embedded={embedded_r.ranked_ids[:1]} "
                f"server={server_r.ranked_ids[:1]}"
            )

    print()
    top1, overlap, differing = ranking_agreement(embedded_hybrid_results, server_hybrid_results)
    print(f"Embedded hybrid vs. server hybrid — top-1 agreement: {top1}/{n}")
    print(f"  mean top-5 overlap (Jaccard): {overlap:.2f}")
    if differing:
        print("  top-1 differs:")
        for embedded_r, server_r in differing:
            print(
                f"    {embedded_r.case.id}: embedded={embedded_r.ranked_ids[:1]} "
                f"server={server_r.ranked_ids[:1]}"
            )

    print()
    print(f"Embedded collection open time: {timing.embedded_open_seconds:.2f}s")
    print(f"Server connect + readiness check: {timing.server_connect_seconds:.2f}s")
    print(
        f"Query retrieval time, {n} queries — "
        f"embedded: {timing.embedded_query_seconds:.2f}s total  "
        f"server: {timing.server_query_seconds:.2f}s total"
    )
    print(
        "These are single local-machine measurements against a local Docker "
        "container, not a production or network-latency claim. Server mode "
        "is not expected to be faster than embedded mode on 40 points — the "
        "point of this phase is operational architecture, not speed."
    )


def main() -> None:
    # Some queries contain math symbols (e.g. ≥) that aren't representable
    # in Windows' default console/redirect codepage (cp1252); write UTF-8
    # regardless of terminal so `evaluate.py --retriever compare > file`
    # doesn't crash partway through.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        choices=[
            "bm25",
            "dense",
            "hybrid",
            "qdrant-dense",
            "qdrant-hybrid",
            "compare",
            "parity",
            "server-parity",
        ],
        default="bm25",
    )
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

        hybrid = HybridRetriever(bm25, dense)

        t0 = time.perf_counter()
        bm25_results = run_queries(bm25, cases)
        bm25_query_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_results = run_queries(dense, cases)
        dense_query_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        hybrid_results = run_queries(hybrid, cases)
        hybrid_query_seconds = time.perf_counter() - t0

        timing = Timing(
            bm25_embed_seconds,
            dense_embed_seconds,
            bm25_query_seconds,
            dense_query_seconds,
            hybrid_query_seconds,
        )
        _print_comparison(bm25_results, dense_results, hybrid_results, bm25, dense, timing)
        return

    if args.retriever == "parity":
        bm25 = BM25Retriever(chunks)

        t0 = time.perf_counter()
        embedder = SentenceTransformerEmbedder()
        model_load_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense = DenseRetriever(chunks, embedder)
        corpus_embed_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        qdrant_client = create_local_client()
        qdrant_open_seconds = time.perf_counter() - t0
        qdrant_dense = QdrantDenseRetriever(qdrant_client, embedder)

        local_hybrid = HybridRetriever(bm25, dense)
        qdrant_hybrid = HybridRetriever(bm25, qdrant_dense)

        bm25_results = run_queries(bm25, cases)

        t0 = time.perf_counter()
        dense_results = run_queries(dense, cases)
        dense_query_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        qdrant_dense_results = run_queries(qdrant_dense, cases)
        qdrant_dense_query_seconds = time.perf_counter() - t0

        hybrid_results = run_queries(local_hybrid, cases)
        qdrant_hybrid_results = run_queries(qdrant_hybrid, cases)
        qdrant_client.close()

        parity_timing = ParityTiming(
            model_load_seconds,
            corpus_embed_seconds,
            qdrant_open_seconds,
            dense_query_seconds,
            qdrant_dense_query_seconds,
        )
        _print_parity(
            bm25_results,
            dense_results,
            qdrant_dense_results,
            hybrid_results,
            qdrant_hybrid_results,
            parity_timing,
        )
        return

    if args.retriever == "server-parity":
        url = os.environ.get("AXIOM_QDRANT_URL")
        if not url:
            parser.error("--retriever server-parity requires AXIOM_QDRANT_URL to be set")

        bm25 = BM25Retriever(chunks)
        embedder = SentenceTransformerEmbedder()

        t0 = time.perf_counter()
        embedded_client = create_local_client()
        embedded_open_seconds = time.perf_counter() - t0
        embedded_dense = QdrantDenseRetriever(embedded_client, embedder)

        t0 = time.perf_counter()
        server_client = create_server_client(url)
        server_connect_seconds = time.perf_counter() - t0
        server_dense = QdrantDenseRetriever(server_client, embedder)

        embedded_hybrid = HybridRetriever(bm25, embedded_dense)
        server_hybrid = HybridRetriever(bm25, server_dense)

        bm25_results = run_queries(bm25, cases)

        t0 = time.perf_counter()
        embedded_dense_results = run_queries(embedded_dense, cases)
        embedded_query_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        server_dense_results = run_queries(server_dense, cases)
        server_query_seconds = time.perf_counter() - t0

        embedded_hybrid_results = run_queries(embedded_hybrid, cases)
        server_hybrid_results = run_queries(server_hybrid, cases)

        embedded_client.close()
        server_client.close()

        server_parity_timing = ServerParityTiming(
            embedded_open_seconds,
            server_connect_seconds,
            embedded_query_seconds,
            server_query_seconds,
        )
        _print_server_parity(
            bm25_results,
            embedded_dense_results,
            server_dense_results,
            embedded_hybrid_results,
            server_hybrid_results,
            server_parity_timing,
        )
        return

    retriever: Retriever
    if args.retriever == "bm25":
        retriever = BM25Retriever(chunks)
    elif args.retriever == "dense":
        retriever = DenseRetriever(chunks, SentenceTransformerEmbedder())
    elif args.retriever == "qdrant-dense":
        qdrant_client = create_client()
        retriever = QdrantDenseRetriever(qdrant_client, SentenceTransformerEmbedder())
    elif args.retriever == "qdrant-hybrid":
        bm25 = BM25Retriever(chunks)
        qdrant_client = create_client()
        qdrant_dense = QdrantDenseRetriever(qdrant_client, SentenceTransformerEmbedder())
        retriever = HybridRetriever(bm25, qdrant_dense)
    else:
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, SentenceTransformerEmbedder())
        retriever = HybridRetriever(bm25, dense)

    results = run_queries(retriever, cases)
    _print_metrics(args.retriever, compute_metrics(results), results)


if __name__ == "__main__":
    main()
