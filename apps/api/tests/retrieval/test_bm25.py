from axiom.knowledge.models import KnowledgeChunk
from axiom.retrieval.bm25 import BM25Retriever, tokenize


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


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Solve 3x+1=4, please.") == ["solve", "3x", "1", "4", "please"]


def test_search_ranks_matching_chunk_first() -> None:
    chunks = [
        _chunk("a", "The discriminant of a quadratic equation is b squared minus 4ac."),
        _chunk("b", "Solving inequalities involves flipping the sign when dividing by a negative."),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.search("what is the discriminant", limit=2)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_search_respects_limit() -> None:
    chunks = [_chunk(str(i), f"chunk number {i} about equations") for i in range(10)]
    retriever = BM25Retriever(chunks)

    results = retriever.search("equations", limit=3)

    assert len(results) == 3
