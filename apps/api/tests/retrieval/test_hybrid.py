import pytest

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.base import Retriever
from axiom.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from axiom.retrieval.models import RetrievalResult


def _chunk(chunk_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="Doc",
        section=None,
        subsection=None,
        text="text",
        source_url="https://example.org/doc",
        source_name="Test",
        license="CC BY-SA 4.0",
        topic="equations",
        subtopic=None,
    )


def _hit(chunk_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id), score=0.0, rank=rank)


class FakeRetriever:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self._hits = hits
        self.last_limit: int | None = None

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        self.last_limit = limit
        return self._hits[:limit]


def test_reciprocal_rank_fusion_sums_contributions_across_lists() -> None:
    list_a = [_hit("a", 1), _hit("b", 2)]
    list_b = [_hit("b", 1), _hit("a", 2)]

    fused = dict(reciprocal_rank_fusion([list_a, list_b], k=1))

    assert fused["a"] == pytest.approx(1 / 2 + 1 / 3)
    assert fused["b"] == pytest.approx(1 / 3 + 1 / 2)


def test_reciprocal_rank_fusion_breaks_ties_by_chunk_id() -> None:
    tied = [_hit("b", 1), _hit("a", 1)]

    fused = reciprocal_rank_fusion([tied])

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]


def test_hybrid_retriever_ranks_chunks_found_by_both_ahead_of_either_alone() -> None:
    bm25 = FakeRetriever([_hit("a", 1), _hit("b", 2)])
    dense = FakeRetriever([_hit("b", 1), _hit("c", 2)])
    retriever = HybridRetriever(bm25, dense)

    results = retriever.search("query", limit=3)

    assert [r.chunk.chunk_id for r in results] == ["b", "a", "c"]
    assert results[0].rank == 1


def test_hybrid_retriever_includes_bm25_only_candidate() -> None:
    bm25 = FakeRetriever([_hit("a", 1)])
    dense = FakeRetriever([])
    retriever = HybridRetriever(bm25, dense)

    results = retriever.search("query", limit=5)

    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_hybrid_retriever_includes_dense_only_candidate() -> None:
    bm25 = FakeRetriever([])
    dense = FakeRetriever([_hit("a", 1)])
    retriever = HybridRetriever(bm25, dense)

    results = retriever.search("query", limit=5)

    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_hybrid_retriever_deduplicates_candidate_found_by_both() -> None:
    bm25 = FakeRetriever([_hit("a", 1)])
    dense = FakeRetriever([_hit("a", 1)])
    retriever = HybridRetriever(bm25, dense)

    results = retriever.search("query", limit=5)

    assert len(results) == 1


def test_hybrid_retriever_respects_limit() -> None:
    bm25 = FakeRetriever([_hit("a", 1), _hit("b", 2), _hit("c", 3)])
    dense = FakeRetriever([])
    retriever = HybridRetriever(bm25, dense)

    results = retriever.search("query", limit=2)

    assert len(results) == 2


def test_hybrid_retriever_requests_candidate_depth_not_final_limit() -> None:
    bm25 = FakeRetriever([_hit("a", 1)])
    dense = FakeRetriever([_hit("a", 1)])
    retriever = HybridRetriever(bm25, dense, depth=7)

    retriever.search("query", limit=1)

    assert bm25.last_limit == 7
    assert dense.last_limit == 7


def test_hybrid_retriever_empty_when_no_candidates() -> None:
    retriever = HybridRetriever(FakeRetriever([]), FakeRetriever([]))

    assert retriever.search("query", limit=5) == []


def test_hybrid_retriever_preserves_chunk_provenance() -> None:
    bm25 = FakeRetriever([_hit("a", 1)])
    dense = FakeRetriever([])
    retriever = HybridRetriever(bm25, dense)

    result = retriever.search("query", limit=1)[0]

    assert result.chunk.source_url == "https://example.org/doc"
    assert result.chunk.license == "CC BY-SA 4.0"


def test_hybrid_retriever_satisfies_retriever_protocol() -> None:
    retriever = HybridRetriever(FakeRetriever([]), FakeRetriever([]))
    assert isinstance(retriever, Retriever)


def test_hybrid_retriever_works_with_evaluation_harness() -> None:
    from axiom.retrieval.evaluate import EvalCase, run_queries

    bm25 = FakeRetriever([_hit("a", 1)])
    dense = FakeRetriever([_hit("a", 1)])
    retriever = HybridRetriever(bm25, dense)
    case = EvalCase(
        id="t1",
        query="q",
        relevant_chunk_ids=["a"],
        topic="equations",
        query_type="definition",
        in_scope=True,
    )

    results = run_queries(retriever, [case])

    assert results[0].ranked_ids == ["a"]
