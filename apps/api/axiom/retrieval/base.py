from typing import Protocol

from axiom.retrieval.models import RetrievalResult


class Retriever(Protocol):
    """Ranked search over the knowledge base. A future Qdrant-backed
    retriever implements the same contract so the evaluation harness in
    evaluate.py doesn't change when the backend does."""

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]: ...
