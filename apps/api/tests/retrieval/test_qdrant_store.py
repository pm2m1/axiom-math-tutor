from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.base import Retriever
from axiom.retrieval.qdrant_store import (
    QdrantDenseRetriever,
    chunk_to_payload,
    ensure_collection,
    index_chunks,
    payload_to_chunk,
    point_id_for_chunk,
)


class FakeEmbedder:
    """Same fixture pattern as test_dense.py: known strings map to
    hand-picked vectors so ranking is checkable without a real model."""

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


@pytest.fixture
def client(tmp_path: Path) -> Iterator[QdrantClient]:
    """A local-mode client backed by a per-test temporary directory — no
    Docker, no network, no shared state with a developer's real collection."""
    qdrant_client = QdrantClient(path=str(tmp_path / "qdrant"))
    yield qdrant_client
    qdrant_client.close()


def test_point_id_for_chunk_is_deterministic() -> None:
    assert point_id_for_chunk("equations-properties-02") == point_id_for_chunk(
        "equations-properties-02"
    )


def test_point_id_for_chunk_differs_across_chunk_ids() -> None:
    assert point_id_for_chunk("a") != point_id_for_chunk("b")


def test_chunk_to_payload_roundtrips() -> None:
    chunk = _chunk("a", "about discriminants")
    assert payload_to_chunk(chunk_to_payload(chunk)) == chunk


def test_index_chunks_creates_collection_and_upserts(client: QdrantClient) -> None:
    chunks = [_chunk("a", "about discriminants"), _chunk("b", "about inequalities")]

    result = index_chunks(client, chunks, FakeEmbedder(), vector_size=2)

    assert result.collection_created is True
    assert result.point_count == 2


def test_index_chunks_is_idempotent(client: QdrantClient) -> None:
    chunks = [_chunk("a", "about discriminants"), _chunk("b", "about inequalities")]

    index_chunks(client, chunks, FakeEmbedder(), vector_size=2)
    second = index_chunks(client, chunks, FakeEmbedder(), vector_size=2)

    assert second.collection_created is False
    assert second.point_count == 2


def test_ensure_collection_rejects_incompatible_dimension(client: QdrantClient) -> None:
    ensure_collection(client, vector_size=2)

    with pytest.raises(ValueError):
        ensure_collection(client, vector_size=5)


def test_ensure_collection_recreate_rebuilds_with_new_dimension(client: QdrantClient) -> None:
    ensure_collection(client, vector_size=2)

    created = ensure_collection(client, vector_size=5, recreate=True)

    assert created is True
    vectors = client.get_collection("axiom_algebra").config.params.vectors
    assert vectors.size == 5  # type: ignore[union-attr]


def test_qdrant_dense_retriever_ranks_and_preserves_provenance(client: QdrantClient) -> None:
    chunks = [_chunk("a", "about discriminants"), _chunk("b", "about inequalities")]
    index_chunks(client, chunks, FakeEmbedder(), vector_size=2)
    retriever = QdrantDenseRetriever(client, FakeEmbedder())

    results = retriever.search("what is the discriminant", limit=2)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].rank == 1
    assert results[0].chunk.source_url == "https://example.org/doc"
    assert results[0].chunk.license == "CC BY-SA 4.0"


def test_qdrant_dense_retriever_respects_limit(client: QdrantClient) -> None:
    chunks = [_chunk("a", "about discriminants"), _chunk("b", "about inequalities")]
    index_chunks(client, chunks, FakeEmbedder(), vector_size=2)
    retriever = QdrantDenseRetriever(client, FakeEmbedder())

    results = retriever.search("what is the discriminant", limit=1)

    assert len(results) == 1


def test_qdrant_dense_retriever_satisfies_retriever_protocol(client: QdrantClient) -> None:
    index_chunks(client, [_chunk("a", "about discriminants")], FakeEmbedder(), vector_size=2)

    assert isinstance(QdrantDenseRetriever(client, FakeEmbedder()), Retriever)


def test_qdrant_dense_retriever_requires_existing_collection(client: QdrantClient) -> None:
    with pytest.raises(ValueError):
        QdrantDenseRetriever(client, FakeEmbedder())
