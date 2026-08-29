import numpy as np

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.base import Retriever
from axiom.retrieval.dense import DenseRetriever


class FakeEmbedder:
    """Maps known strings to hand-picked unit vectors so ranking behavior is
    checkable by hand, without loading a real model."""

    _VECTORS = {
        "about discriminants": [1.0, 0.0],
        "about inequalities": [0.0, 1.0],
        "what is the discriminant": [0.9, 0.1],
    }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self._VECTORS[text] for text in texts])


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="Doc",
        section=None,
        subsection=None,
        text=text,
        source_url="https://example.org/doc",
        source_name="Test",
        license="CC BY-SA 4.0",
        topic="equations",
        subtopic=None,
    )


def test_search_ranks_by_cosine_similarity() -> None:
    chunks = [_chunk("a", "about discriminants"), _chunk("b", "about inequalities")]
    retriever = DenseRetriever(chunks, FakeEmbedder())

    results = retriever.search("what is the discriminant", limit=2)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_search_respects_limit() -> None:
    chunks = [_chunk("a", "about discriminants"), _chunk("b", "about inequalities")]
    retriever = DenseRetriever(chunks, FakeEmbedder())

    results = retriever.search("what is the discriminant", limit=1)

    assert len(results) == 1


def test_search_preserves_chunk_provenance() -> None:
    chunks = [_chunk("a", "about discriminants")]
    retriever = DenseRetriever(chunks, FakeEmbedder())

    result = retriever.search("about discriminants", limit=1)[0]

    assert result.chunk.source_url == "https://example.org/doc"
    assert result.chunk.license == "CC BY-SA 4.0"


def test_dense_retriever_satisfies_retriever_protocol() -> None:
    retriever = DenseRetriever([_chunk("a", "about discriminants")], FakeEmbedder())
    assert isinstance(retriever, Retriever)
