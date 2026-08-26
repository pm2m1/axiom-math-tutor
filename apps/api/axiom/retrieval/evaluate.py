"""Reproducible retrieval evaluation over the BM25 baseline.

    uv run python -m axiom.retrieval.evaluate

Loads the committed corpus and labeled query set, runs every query through
BM25Retriever, and reports Hit@k/Recall@5/MRR plus the top-score spread for
in-scope vs. out-of-scope queries. See docs/retrieval-evaluation.md for what
these numbers currently mean.
"""

import statistics
from pathlib import Path

from pydantic import BaseModel

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.bm25 import BM25Retriever
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


def _distribution(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)}  min={min(values):.2f}  median={statistics.median(values):.2f}  "
        f"max={max(values):.2f}  mean={statistics.mean(values):.2f}"
    )


def run() -> None:
    chunks = load_chunks()
    cases = load_eval_cases()
    retriever = BM25Retriever(chunks)

    in_scope = [c for c in cases if c.in_scope]
    out_of_scope = [c for c in cases if not c.in_scope]

    hits1 = hits3 = hits5 = 0
    recalls5: list[float] = []
    reciprocal_ranks: list[float] = []
    in_scope_top_scores: list[float] = []
    out_of_scope_top_scores: list[float] = []

    for case in in_scope:
        results = retriever.search(case.query, limit=5)
        ranked_ids = [r.chunk.chunk_id for r in results]
        relevant = set(case.relevant_chunk_ids)
        hits1 += hit_at_k(ranked_ids, relevant, 1)
        hits3 += hit_at_k(ranked_ids, relevant, 3)
        hits5 += hit_at_k(ranked_ids, relevant, 5)
        recalls5.append(recall_at_k(ranked_ids, relevant, 5))
        reciprocal_ranks.append(reciprocal_rank(ranked_ids, relevant))
        if results:
            in_scope_top_scores.append(results[0].score)

    for case in out_of_scope:
        results = retriever.search(case.query, limit=5)
        if results:
            out_of_scope_top_scores.append(results[0].score)

    n = len(in_scope)
    print("Retrieval evaluation")
    print("--------------------")
    print(f"queries        {len(cases)}")
    print(f"in-scope       {len(in_scope)}")
    print(f"out-of-scope   {len(out_of_scope)}")
    print()
    print(f"Hit@1          {hits1 / n:.2f}")
    print(f"Hit@3          {hits3 / n:.2f}")
    print(f"Hit@5          {hits5 / n:.2f}")
    print(f"Recall@5       {statistics.mean(recalls5):.2f}")
    print(f"MRR            {statistics.mean(reciprocal_ranks):.2f}")
    print()
    print("Top-1 score distribution, in-scope queries:")
    print("  " + _distribution(in_scope_top_scores))
    print("Top-1 score distribution, out-of-scope queries:")
    print("  " + _distribution(out_of_scope_top_scores))


if __name__ == "__main__":
    run()
