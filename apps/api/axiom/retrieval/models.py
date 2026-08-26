from pydantic import BaseModel

from axiom.knowledge.models import KnowledgeChunk


class RetrievalResult(BaseModel):
    """One ranked hit. `score` is only meaningful relative to other results
    from the same retriever and query set — BM25 scores are not bounded
    probabilities, and a future dense retriever will have its own scale.
    See docs/retrieval-evaluation.md."""

    chunk: KnowledgeChunk
    score: float
    rank: int
