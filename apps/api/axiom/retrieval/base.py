from typing import Protocol, runtime_checkable

from axiom.retrieval.models import RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Ranked search over the knowledge base. BM25Retriever and
    DenseRetriever both implement this so the evaluation harness in
    evaluate.py doesn't special-case either one, and a future Qdrant-backed
    retriever can slot in the same way."""

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]: ...
