"""Persistent dense retrieval backed by Qdrant. See docs/retrieval-evaluation.md
(Phase 1D/1E) for why this was introduced: the Phase 1B DenseRetriever
re-embeds all 40 chunks on every process start (~200-300s on this
CPU-only machine); Qdrant lets a corpus be embedded once and queried many
times.

`create_client()` supports two deployment modes without duplicating any
retrieval code: embedded/local persistent storage (`QdrantClient(path=...)`,
the default) and a real Qdrant server (`QdrantClient(url=...)`, selected by
setting `AXIOM_QDRANT_URL` — see `compose.yaml`). Only the dense
storage/search changes either way: BM25 (bm25.py) and RRF (hybrid.py) are
untouched, so QdrantDenseRetriever is a drop-in Retriever, not a new
algorithm.
"""

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.dense import Embedder
from axiom.retrieval.models import RetrievalResult

COLLECTION_NAME = "axiom_algebra"
VECTOR_SIZE = 768
DISTANCE = qmodels.Distance.COSINE

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Qdrant point IDs must be an unsigned integer or a UUID, not an arbitrary
# string like our chunk IDs (e.g. "equations-properties-02"). Points are
# addressed by a UUID deterministically derived from the chunk ID via
# uuid5, so re-indexing the same chunk always maps to the same point; the
# chunk ID itself stays the canonical identity, in the payload.
_POINT_ID_NAMESPACE = uuid.UUID("a91e5f4e-8f76-4c1e-9c2c-6e8d9a2b6f10")


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def default_storage_path() -> Path:
    override = os.environ.get("AXIOM_QDRANT_PATH")
    return Path(override) if override else _REPO_ROOT / ".data" / "qdrant"


def create_local_client() -> QdrantClient:
    storage_path = default_storage_path()
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def create_server_client(url: str) -> QdrantClient:
    """Connectivity is verified immediately rather than left for the first
    real query — a misconfigured or unreachable server should fail loudly
    here, not silently behave like an empty collection later."""
    client = QdrantClient(url=url)
    try:
        client.get_collections()
    except ResponseHandlingException as exc:
        raise ConnectionError(f"Qdrant server at {url!r} is not reachable: {exc}") from exc
    return client


def create_client() -> QdrantClient:
    """Server mode if `AXIOM_QDRANT_URL` is set, otherwise embedded local
    persistent storage. Server mode never falls back to local mode
    silently on failure, which would make a "server" run misleading."""
    url = os.environ.get("AXIOM_QDRANT_URL")
    return create_server_client(url) if url else create_local_client()


def chunk_to_payload(chunk: KnowledgeChunk) -> dict[str, Any]:
    return chunk.model_dump()


def payload_to_chunk(payload: dict[str, Any]) -> KnowledgeChunk:
    return KnowledgeChunk.model_validate(payload)


def ensure_collection(
    client: QdrantClient, *, vector_size: int = VECTOR_SIZE, recreate: bool = False
) -> bool:
    """Create the collection if it doesn't exist. If it does, validate its
    vector config matches rather than silently reusing or deleting it.
    Returns True if the collection was (re)created."""
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and not recreate:
        vectors = client.get_collection(COLLECTION_NAME).config.params.vectors
        if not isinstance(vectors, qmodels.VectorParams):
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} has a named/multi-vector "
                "config; expected a single unnamed dense vector."
            )
        if vectors.size != vector_size or vectors.distance != DISTANCE:
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} exists with size={vectors.size} "
                f"distance={vectors.distance}, expected size={vector_size} "
                f"distance={DISTANCE}. Pass --recreate to rebuild it."
            )
        return False

    if exists:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=DISTANCE),
    )
    return True


@dataclass
class IndexResult:
    point_count: int
    collection_created: bool
    embed_seconds: float
    upsert_seconds: float


def index_chunks(
    client: QdrantClient,
    chunks: list[KnowledgeChunk],
    embedder: Embedder,
    *,
    vector_size: int = VECTOR_SIZE,
    recreate: bool = False,
) -> IndexResult:
    """Embed and upsert every chunk. Point IDs are deterministic (see
    point_id_for_chunk), so running this twice against the same corpus
    updates the same 40 points rather than duplicating them."""
    created = ensure_collection(client, vector_size=vector_size, recreate=recreate)

    t0 = time.perf_counter()
    vectors = embedder.encode([chunk.text for chunk in chunks])
    embed_seconds = time.perf_counter() - t0

    points = [
        qmodels.PointStruct(
            id=point_id_for_chunk(chunk.chunk_id),
            vector=vectors[i].tolist(),
            payload=chunk_to_payload(chunk),
        )
        for i, chunk in enumerate(chunks)
    ]

    t0 = time.perf_counter()
    client.upsert(COLLECTION_NAME, points=points)
    upsert_seconds = time.perf_counter() - t0

    point_count = client.count(COLLECTION_NAME).count
    return IndexResult(point_count, created, embed_seconds, upsert_seconds)


class QdrantDenseRetriever:
    """Dense retrieval against an already-indexed Qdrant collection. Embeds
    only the query at search time — corpus embedding happened once, during
    indexing (see index_qdrant.py), not on every process start."""

    def __init__(
        self, client: QdrantClient, embedder: Embedder, *, collection: str = COLLECTION_NAME
    ) -> None:
        if not client.collection_exists(collection):
            raise ValueError(
                f"Collection {collection!r} does not exist. Run "
                "`uv run python -m axiom.retrieval.index_qdrant` first."
            )
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        query_vector = self._embedder.encode([query])[0]
        response = self._client.query_points(
            self._collection, query=query_vector.tolist(), limit=limit, with_payload=True
        )
        results = []
        for rank, point in enumerate(response.points, start=1):
            if point.payload is None:
                raise ValueError(f"point {point.id} in {self._collection!r} has no payload")
            results.append(
                RetrievalResult(chunk=payload_to_chunk(point.payload), score=point.score, rank=rank)
            )
        return results
