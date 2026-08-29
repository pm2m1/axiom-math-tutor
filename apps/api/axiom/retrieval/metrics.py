"""Retrieval metrics over one query's ranked results.

nDCG is deliberately not implemented here. Hit@k, Recall@5, and MRR were
enough to characterize the Phase 1A/1B baselines; that's a scope decision,
not a claim that nDCG is uninformative under binary relevance — some of our
queries do have more than one labeled relevant chunk, and nDCG would still
distinguish ranking those higher from ranking them lower. It's worth adding
if a future comparison needs that distinction.
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
