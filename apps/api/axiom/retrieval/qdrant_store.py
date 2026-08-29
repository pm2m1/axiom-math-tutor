"""Persistent retrieval backed by Qdrant. See docs/retrieval-evaluation.md
(Phase 1D/1E/1F) for why this was introduced: the Phase 1B DenseRetriever
re-embeds all 40 chunks on every process start (~200-300s on this
CPU-only machine); Qdrant lets a corpus be embedded once and queried many
times. Phase 1F adds Qdrant's own server-side BM25 and native RRF fusion
alongside the dense retriever, on the same collection.

`create_client()` supports two deployment modes without duplicating any
retrieval code: embedded/local persistent storage (`QdrantClient(path=...)`,
the default) and a real Qdrant server (`QdrantClient(url=...)`, selected by
setting `AXIOM_QDRANT_URL` — see `compose.yaml`). BM25 and native RRF need
a real server; embedded/local mode only supports dense retrieval (see
BM25_VECTOR_NAME below).
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
from axiom.retrieval.hybrid import CANDIDATE_DEPTH
from axiom.retrieval.models import RetrievalResult

COLLECTION_NAME = "axiom_algebra"
VECTOR_SIZE = 768
DISTANCE = qmodels.Distance.COSINE

# Qdrant's own server-side BM25 (Phase 1F) — a different lexical
# implementation from axiom.retrieval.bm25's rank_bm25.BM25Okapi, with its
# own tokenizer/stemmer/stopword defaults. Requires a real Qdrant server:
# embedded/local mode has no built-in tokenizer for this and would need the
# optional fastembed dependency instead, which this project doesn't add —
# see docs/retrieval-evaluation.md (Phase 1F).
BM25_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"

# Reproduces our application-side RRF's k=60 (axiom/retrieval/hybrid.py)
# exactly: Qdrant's Rrf.k uses zero-based rank positions internally, so its
# k must be one higher than ours to land on the same formula. Verified
# empirically against a real server with hand-computable ranks, not
# assumed — see test_qdrant_store.py and docs/retrieval-evaluation.md.
NATIVE_RRF_K = 61

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
    Returns True if the collection was (re)created.

    Every point carries both an unnamed dense vector and a named `bm25`
    sparse vector (IDF modifier) — the two can coexist without the dense
    vector needing a name of its own, confirmed against a real server.
    Qdrant can't add a vector field to an existing collection, only
    validate or replace it entirely, so a collection built before Phase 1F
    (dense only) fails the check below rather than silently missing BM25;
    pass recreate=True to rebuild it."""
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and not recreate:
        params = client.get_collection(COLLECTION_NAME).config.params
        vectors = params.vectors
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
        if not params.sparse_vectors or BM25_VECTOR_NAME not in params.sparse_vectors:
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} exists without a {BM25_VECTOR_NAME!r} "
                "sparse vector (added in Phase 1F). Pass --recreate to rebuild it."
            )
        return False

    if exists:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=DISTANCE),
        sparse_vectors_config={
            BM25_VECTOR_NAME: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
        },
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


def index_bm25(client: QdrantClient, chunks: list[KnowledgeChunk]) -> float:
    """Add each chunk's server-computed BM25 sparse vector to its
    already-indexed point, via Qdrant's native text inference — real
    tokenization/stemming done by the server, not a value we construct.
    `update_vectors` keeps the existing dense vector and payload
    untouched. Requires a real server (see BM25_VECTOR_NAME above); run
    index_chunks first so the points already exist. Returns elapsed
    seconds."""
    t0 = time.perf_counter()
    client.update_vectors(
        COLLECTION_NAME,
        points=[
            qmodels.PointVectors(
                id=point_id_for_chunk(chunk.chunk_id),
                vector={BM25_VECTOR_NAME: qmodels.Document(text=chunk.text, model=BM25_MODEL)},
            )
            for chunk in chunks
        ],
    )
    return time.perf_counter() - t0


def _to_retrieval_results(points: list[qmodels.ScoredPoint]) -> list[RetrievalResult]:
    results = []
    for rank, point in enumerate(points, start=1):
        if point.payload is None:
            raise ValueError(f"point {point.id} has no payload")
        results.append(
            RetrievalResult(chunk=payload_to_chunk(point.payload), score=point.score, rank=rank)
        )
    return results


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
        return _to_retrieval_results(response.points)


class QdrantBM25Retriever:
    """Lexical retrieval via Qdrant's own server-side BM25 — a different
    implementation from axiom.retrieval.bm25.BM25Retriever, with Qdrant's
    own tokenizer/stemmer/stopword defaults (English, Snowball stemmer,
    default stopwords, lowercased, no ASCII folding), not ours. See
    docs/retrieval-evaluation.md (Phase 1F) for the measured difference."""

    def __init__(self, client: QdrantClient, *, collection: str = COLLECTION_NAME) -> None:
        if not client.collection_exists(collection):
            raise ValueError(
                f"Collection {collection!r} does not exist. Run "
                "`uv run python -m axiom.retrieval.index_qdrant` first."
            )
        self._client = client
        self._collection = collection

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        response = self._client.query_points(
            self._collection,
            query=qmodels.Document(text=query, model=BM25_MODEL),
            using=BM25_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        return _to_retrieval_results(response.points)


class QdrantNativeHybridRetriever:
    """BM25 + dense fusion computed entirely server-side via Qdrant's Query
    API (prefetch + Rrf), rather than the application-side RRF in
    hybrid.py. k=NATIVE_RRF_K (61) reproduces hybrid.py's k=60 exactly —
    see that constant's docstring for why 61, not 60."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        *,
        collection: str = COLLECTION_NAME,
        depth: int = CANDIDATE_DEPTH,
        k: int = NATIVE_RRF_K,
    ) -> None:
        if not client.collection_exists(collection):
            raise ValueError(
                f"Collection {collection!r} does not exist. Run "
                "`uv run python -m axiom.retrieval.index_qdrant` first."
            )
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._depth = depth
        self._k = k

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        query_vector = self._embedder.encode([query])[0]
        dense_prefetch = qmodels.Prefetch(query=query_vector.tolist(), limit=self._depth)
        bm25_prefetch = qmodels.Prefetch(
            query=qmodels.Document(text=query, model=BM25_MODEL),
            using=BM25_VECTOR_NAME,
            limit=self._depth,
        )
        response = self._client.query_points(
            self._collection,
            prefetch=[dense_prefetch, bm25_prefetch],
            query=qmodels.RrfQuery(rrf=qmodels.Rrf(k=self._k)),
            limit=limit,
            with_payload=True,
        )
        return _to_retrieval_results(response.points)
