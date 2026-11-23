"""Singleton-per-process dependencies for the evidence-routing endpoint
(Phase 1J). Built once and reused across requests via `lru_cache`, so a
request never re-embeds the corpus, reopens the Qdrant collection, or
reconstructs an LLM/MCP client. Construction failures (missing
credentials, an unindexed collection) are caught here and surfaced as
`None`, never as an unhandled exception from a route.
"""

from dataclasses import dataclass
from functools import lru_cache

from axiom.retrieval.base import Retriever
from axiom.retrieval.bm25 import BM25Retriever
from axiom.retrieval.dense import SentenceTransformerEmbedder
from axiom.retrieval.evaluate import load_chunks
from axiom.retrieval.evidence_judge import JudgeClient, MetaCredentialError, MetaEvidenceJudge
from axiom.retrieval.hybrid import HybridRetriever
from axiom.retrieval.qdrant_store import QdrantDenseRetriever, create_client
from axiom.web_search import TavilyMCPSearchClient, WebSearchClient


@dataclass
class RetrievalPipeline:
    hybrid: Retriever


@lru_cache
def get_retrieval_pipeline() -> RetrievalPipeline | None:
    try:
        chunks = load_chunks()
        bm25 = BM25Retriever(chunks)
        embedder = SentenceTransformerEmbedder()
        qdrant_client = create_client()
        dense = QdrantDenseRetriever(qdrant_client, embedder)
        return RetrievalPipeline(hybrid=HybridRetriever(bm25, dense))
    except Exception:
        return None


@lru_cache
def get_evidence_judge() -> JudgeClient | None:
    try:
        return MetaEvidenceJudge()
    except MetaCredentialError:
        return None


@lru_cache
def get_web_search_client() -> WebSearchClient | None:
    try:
        return TavilyMCPSearchClient()
    except ValueError:
        return None
