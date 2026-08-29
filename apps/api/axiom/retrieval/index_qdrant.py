"""One-time indexing: embed the committed algebra corpus and upsert it into
a persistent Qdrant collection, so the Qdrant retrievers in evaluate.py
don't re-embed the corpus on every run. See docs/retrieval-evaluation.md
(Phase 1D/1E/1F).

    uv run python -m axiom.retrieval.index_qdrant
    uv run python -m axiom.retrieval.index_qdrant --recreate

Targets embedded local storage by default, or a Qdrant server if
`AXIOM_QDRANT_URL` is set (see compose.yaml):

    AXIOM_QDRANT_URL=http://localhost:6333 uv run python -m axiom.retrieval.index_qdrant

Also adds each chunk's Qdrant-native BM25 sparse vector (Phase 1F). This
step needs a real server — embedded/local mode has no built-in tokenizer
for it and raises a clear ImportError (it would need the optional
fastembed dependency, not added here) if run without AXIOM_QDRANT_URL set.
"""

import argparse
import os
import time

from axiom.retrieval.dense import SentenceTransformerEmbedder
from axiom.retrieval.evaluate import load_chunks
from axiom.retrieval.qdrant_store import (
    create_client,
    default_storage_path,
    index_bm25,
    index_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    chunks = load_chunks()
    client = create_client()
    target = os.environ.get("AXIOM_QDRANT_URL") or str(default_storage_path())

    t0 = time.perf_counter()
    result = index_chunks(client, chunks, SentenceTransformerEmbedder(), recreate=args.recreate)
    dense_seconds = time.perf_counter() - t0

    bm25_seconds = index_bm25(client, chunks)
    total_seconds = dense_seconds + bm25_seconds
    client.close()

    print(f"Target:             {target}")
    print(f"Collection created: {result.collection_created}")
    print(f"Points indexed:     {result.point_count}")
    print(f"Dense embedding:    {result.embed_seconds:.2f}s")
    print(f"Dense upsert:       {result.upsert_seconds:.2f}s")
    print(f"BM25 indexing:      {bm25_seconds:.2f}s")
    print(f"Total time:         {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
