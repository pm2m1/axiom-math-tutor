"""Retrieval metrics over one query's ranked results.

Relevance labels in the evaluation set are binary (a chunk either supports
the query or it wasn't judged to), so nDCG would carry no more information
than Recall@k and MRR already do here — it's left out rather than added for
appearances.
"""


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> bool:
    return any(cid in relevant_ids for cid in ranked_ids[:k])


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
