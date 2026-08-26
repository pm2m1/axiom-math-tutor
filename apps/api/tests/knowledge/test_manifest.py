import json
from pathlib import Path

import pytest

from axiom.knowledge.manifest import load_manifest

_VALID = {
    "sources": [
        {
            "document_id": "doc-1",
            "title": "Doc One",
            "source_url": "https://en.wikibooks.org/wiki/Algebra/Doc_One",
            "source_name": "Wikibooks: Algebra",
            "license": "CC BY-SA 4.0",
            "attribution": "Wikibooks contributors, sample",
            "retrieved": "2026-08-26",
            "topic": "equations",
            "subtopic": None,
        }
    ]
}


def test_load_manifest_parses_sources(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(_VALID), encoding="utf-8")

    documents = load_manifest(manifest_path)

    assert len(documents) == 1
    assert documents[0].document_id == "doc-1"
    assert documents[0].license == "CC BY-SA 4.0"


def test_load_manifest_rejects_missing_field(tmp_path: Path) -> None:
    broken = {"sources": [{k: v for k, v in _VALID["sources"][0].items() if k != "license"}]}
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError):
        load_manifest(manifest_path)
