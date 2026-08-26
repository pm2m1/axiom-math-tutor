"""Fetch manifest sources and write the normalized chunk corpus.

    uv run python -m axiom.knowledge.ingest [--refresh]

Network access is isolated to this module. `chunking.py` and everything in
`axiom.retrieval` work from the resulting JSONL file, so tests never need a
live connection. Fetched pages are cached under `.data/raw/` (git-ignored) so
re-running ingestion doesn't re-fetch Wikibooks on every invocation; pass
--refresh to bypass the cache.
"""

import sys
from pathlib import Path

import httpx

from axiom.knowledge.chunking import build_chunks
from axiom.knowledge.manifest import load_manifest

# apps/api/axiom/knowledge/ingest.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = _REPO_ROOT / "data" / "manifests" / "algebra_sources.json"
CORPUS_PATH = _REPO_ROOT / "data" / "corpus" / "algebra_chunks.jsonl"
RAW_CACHE_DIR = _REPO_ROOT / ".data" / "raw"

USER_AGENT = (
    "axiom-research-corpus/0.1 (educational project; +https://github.com/pm2m1/axiom-math-tutor)"
)


def _fetch(document_id: str, url: str, *, refresh: bool) -> str:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_CACHE_DIR / f"{document_id}.html"
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def run(*, refresh: bool = False) -> int:
    documents = load_manifest(MANIFEST_PATH)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with CORPUS_PATH.open("w", encoding="utf-8") as out:
        for document in documents:
            html = _fetch(document.document_id, document.source_url, refresh=refresh)
            for chunk in build_chunks(document, html):
                out.write(chunk.model_dump_json())
                out.write("\n")
                total += 1
    return total


if __name__ == "__main__":
    count = run(refresh="--refresh" in sys.argv[1:])
    print(f"wrote {count} chunks to {CORPUS_PATH.relative_to(_REPO_ROOT)}")
