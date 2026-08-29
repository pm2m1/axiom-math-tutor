"""Dense embedding retrieval baseline. See docs/retrieval-evaluation.md
(Phase 1B) for why this model was picked and what the comparison against
BM25 found.

`Embedder` is factored out of `DenseRetriever` so tests can supply a small
fake instead of loading the real ~150M-parameter model — normal `pytest`
runs never touch the network or the sentence-transformers package's model
download path.
"""

from typing import Protocol

import numpy as np

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.models import RetrievalResult

DENSE_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
# Pinned so a future re-run of the evaluation uses the exact weights this
# baseline was measured against, not whatever "main" resolves to later.
DENSE_MODEL_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """gte-modernbert-base's own sentence-transformers config declares no
    query/document prompt (`default_prompt_name: null`), unlike e.g. E5 or
    BGE models, so queries and chunks are encoded identically here rather
    than with an asymmetric prefix. Its config also declares
    `similarity_fn_name: cosine`; encoding with `normalize_embeddings=True`
    makes cosine similarity equivalent to a plain dot product."""

    def __init__(
        self, model_id: str = DENSE_MODEL_ID, revision: str = DENSE_MODEL_REVISION
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id, revision=revision)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


class DenseRetriever:
    def __init__(self, chunks: list[KnowledgeChunk], embedder: Embedder) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._chunk_embeddings = embedder.encode([chunk.text for chunk in chunks])

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        query_embedding = self._embedder.encode([query])[0]
        scores = self._chunk_embeddings @ query_embedding
        ranked = np.argsort(scores)[::-1][:limit]
        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ranked, start=1)
        ]
