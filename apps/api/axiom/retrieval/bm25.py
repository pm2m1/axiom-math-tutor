"""Lexical retrieval baseline. This exists to measure whether retrieval
works at all before adopting a vector store — see docs/retrieval-evaluation.md."""

import re

from rank_bm25 import BM25Okapi

from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.models import RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ranked, start=1)
        ]
