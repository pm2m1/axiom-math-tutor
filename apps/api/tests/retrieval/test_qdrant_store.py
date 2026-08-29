from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.base import Retriever
from axiom.retrieval.hybrid import RRF_K
from axiom.retrieval.qdrant_store import (
    BM25_VECTOR_NAME,
    NATIVE_RRF_K,
    QdrantBM25Retriever,
    QdrantDenseRetriever,
    QdrantNativeHybridRetriever,
    _to_retrieval_results,
    chunk_to_payload,
    create_client,
    create_local_client,
    create_server_client,
    default_storage_path,
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


def test_default_storage_path_respects_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXIOM_QDRANT_PATH", "/tmp/somewhere")
    assert default_storage_path() == Path("/tmp/somewhere")


def test_default_storage_path_falls_back_to_repo_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AXIOM_QDRANT_PATH", raising=False)
    assert default_storage_path().parts[-2:] == (".data", "qdrant")


def test_create_client_uses_local_mode_when_url_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AXIOM_QDRANT_URL", raising=False)
    monkeypatch.setenv("AXIOM_QDRANT_PATH", str(tmp_path / "qdrant"))

    client = create_client()
    try:
        assert client.collection_exists("axiom_algebra") is False
    finally:
        client.close()


def test_create_client_fails_clearly_when_server_url_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AXIOM_QDRANT_URL", "http://localhost:1")

    with pytest.raises(ConnectionError, match="not reachable"):
        create_client()


def test_create_server_client_does_not_fall_back_to_local_mode(tmp_path: Path) -> None:
    """A misconfigured/unreachable server must raise, not silently return a
    working local client — that would make a "server" run misleading."""
    with pytest.raises(ConnectionError):
        create_server_client("http://localhost:1")


def test_create_local_client_ignores_server_url_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AXIOM_QDRANT_URL", "http://localhost:1")
    monkeypatch.setenv("AXIOM_QDRANT_PATH", str(tmp_path / "qdrant"))

    client = create_local_client()
    try:
        assert client.collection_exists("axiom_algebra") is False
    finally:
        client.close()


def test_ensure_collection_creates_bm25_sparse_vector_with_idf_modifier(
    client: QdrantClient,
) -> None:
    ensure_collection(client, vector_size=2)

    sparse = client.get_collection("axiom_algebra").config.params.sparse_vectors
    assert sparse is not None
    assert sparse[BM25_VECTOR_NAME].modifier == qmodels.Modifier.IDF


def test_ensure_collection_rejects_collection_missing_bm25_sparse_vector(
    client: QdrantClient,
) -> None:
    # Simulates a collection built before Phase 1F: dense vector only, no
    # sparse config at all.
    client.create_collection(
        "axiom_algebra",
        vectors_config=qmodels.VectorParams(size=2, distance=qmodels.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="bm25"):
        ensure_collection(client, vector_size=2)


def test_to_retrieval_results_preserves_rank_score_and_provenance() -> None:
    chunk = _chunk("a", "about discriminants")
    points = [
        qmodels.ScoredPoint(id=1, version=0, score=0.9, payload=chunk_to_payload(chunk)),
    ]

    results = _to_retrieval_results(points)

    assert results[0].rank == 1
    assert results[0].score == 0.9
    assert results[0].chunk.chunk_id == "a"
    assert results[0].chunk.source_url == "https://example.org/doc"


def test_to_retrieval_results_numbers_ranks_from_one_in_order() -> None:
    points = [
        qmodels.ScoredPoint(id=1, version=0, score=0.9, payload=chunk_to_payload(_chunk("a", "x"))),
        qmodels.ScoredPoint(id=2, version=0, score=0.5, payload=chunk_to_payload(_chunk("b", "y"))),
    ]

    results = _to_retrieval_results(points)

    assert [r.rank for r in results] == [1, 2]
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]


def test_to_retrieval_results_raises_on_missing_payload() -> None:
    points = [qmodels.ScoredPoint(id=1, version=0, score=0.9, payload=None)]

    with pytest.raises(ValueError, match="payload"):
        _to_retrieval_results(points)


def test_qdrant_bm25_retriever_requires_existing_collection(client: QdrantClient) -> None:
    with pytest.raises(ValueError):
        QdrantBM25Retriever(client)


def test_qdrant_bm25_retriever_satisfies_retriever_protocol(client: QdrantClient) -> None:
    ensure_collection(client, vector_size=2)
    assert isinstance(QdrantBM25Retriever(client), Retriever)


def test_native_hybrid_retriever_requires_existing_collection(client: QdrantClient) -> None:
    with pytest.raises(ValueError):
        QdrantNativeHybridRetriever(client, FakeEmbedder())


def test_native_hybrid_retriever_satisfies_retriever_protocol(client: QdrantClient) -> None:
    ensure_collection(client, vector_size=2)
    assert isinstance(QdrantNativeHybridRetriever(client, FakeEmbedder()), Retriever)


class _RecordingClient:
    """Stands in for QdrantClient to inspect the request
    QdrantNativeHybridRetriever builds, without hitting a real server —
    this checks the request shape, not Qdrant's fusion behavior itself
    (verified separately against a real server)."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def query_points(self, collection_name: str, **kwargs: Any) -> qmodels.QueryResponse:
        self.last_kwargs = kwargs
        return qmodels.QueryResponse(points=[])


def test_native_hybrid_retriever_requests_k61_and_candidate_depth() -> None:
    client = _RecordingClient()
    retriever = QdrantNativeHybridRetriever(client, FakeEmbedder())  # type: ignore[arg-type]

    retriever.search("about discriminants", limit=5)

    query = client.last_kwargs["query"]
    assert isinstance(query, qmodels.RrfQuery)
    assert query.rrf.k == NATIVE_RRF_K
    assert all(prefetch.limit == 10 for prefetch in client.last_kwargs["prefetch"])


def test_native_rrf_k61_matches_application_rrf_k60_under_rank_conventions() -> None:
    """Qdrant's Rrf(k=...) uses zero-based rank positions internally; our
    application RRF (hybrid.py) uses one-based rank. This is the
    deterministic parity check behind NATIVE_RRF_K=61 — verified
    analytically here, and empirically against a real server during
    Phase 1F (see docs/retrieval-evaluation.md)."""

    def application_rrf(one_based_rank: int) -> float:
        return 1 / (RRF_K + one_based_rank)

    def qdrant_rrf(zero_based_position: int) -> float:
        return 1 / (zero_based_position + NATIVE_RRF_K)

    for rank in range(1, 11):
        assert application_rrf(rank) == pytest.approx(qdrant_rrf(rank - 1))
