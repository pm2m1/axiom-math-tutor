"""Reciprocal Rank Fusion hybrid retrieval baseline. Combines BM25 and dense
rankings by rank position rather than by score, since the two retrievers'
scores are not on comparable scales (unbounded BM25 vs. cosine similarity in
[-1, 1]). See docs/retrieval-evaluation.md (Phase 1C) for why RRF and these
constants.
"""

from axiom.retrieval.base import Retriever
from axiom.retrieval.models import RetrievalResult

# Cormack et al., 2009 ("Reciprocal Rank Fusion outperforms Condorcet and
# individual Rank Learning Methods") use k=60; there's no benchmark-specific
# reason to deviate from it here.
RRF_K = 60

# Each component retriever contributes its top 10 candidates before fusion —
# a quarter of the 40-chunk corpus, enough overlap for fusion to matter
# without retrieving the whole corpus every query.
CANDIDATE_DEPTH = 10


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]], *, k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked result lists into one ranking: a chunk's score is the sum
    of 1 / (k + rank) over every list it appears in. Returns (chunk_id,
    score) pairs sorted by score descending, chunk_id ascending as a
    deterministic tiebreak — two chunks landing on the same fused score
    should not depend on dict iteration order."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for hit in results:
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + hit.rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    """Fuses an existing BM25Retriever and DenseRetriever with RRF. Takes
    already-constructed retrievers rather than building them itself, so it
    doesn't duplicate corpus embedding or tokenization."""

    def __init__(self, bm25: Retriever, dense: Retriever, *, depth: int = CANDIDATE_DEPTH) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._depth = depth

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        bm25_hits = self._bm25.search(query, limit=self._depth)
        dense_hits = self._dense.search(query, limit=self._depth)
        chunks = {hit.chunk.chunk_id: hit.chunk for hit in bm25_hits + dense_hits}

        fused = reciprocal_rank_fusion([bm25_hits, dense_hits])
        return [
            RetrievalResult(chunk=chunks[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(fused[:limit], start=1)
        ]
