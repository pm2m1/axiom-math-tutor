"""One-time indexing: embed the committed algebra corpus and upsert it into
a persistent Qdrant collection, so the Qdrant retrievers in evaluate.py
don't re-embed the corpus on every run. See docs/retrieval-evaluation.md
(Phase 1D/1E).

    uv run python -m axiom.retrieval.index_qdrant
    uv run python -m axiom.retrieval.index_qdrant --recreate

Targets embedded local storage by default, or a Qdrant server if
`AXIOM_QDRANT_URL` is set (see compose.yaml):

    AXIOM_QDRANT_URL=http://localhost:6333 uv run python -m axiom.retrieval.index_qdrant
"""

import argparse
import os
import time

from axiom.retrieval.dense import SentenceTransformerEmbedder
from axiom.retrieval.evaluate import load_chunks
from axiom.retrieval.qdrant_store import create_client, default_storage_path, index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    chunks = load_chunks()
    client = create_client()
    target = os.environ.get("AXIOM_QDRANT_URL") or str(default_storage_path())

    t0 = time.perf_counter()
    result = index_chunks(client, chunks, SentenceTransformerEmbedder(), recreate=args.recreate)
    total_seconds = time.perf_counter() - t0
    client.close()

    print(f"Target:             {target}")
    print(f"Collection created: {result.collection_created}")
    print(f"Points indexed:     {result.point_count}")
    print(f"Embedding time:     {result.embed_seconds:.2f}s")
    print(f"Upsert time:        {result.upsert_seconds:.2f}s")
    print(f"Total time:         {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
