import json
from pathlib import Path

from axiom.knowledge.models import SourceDocument


def load_manifest(path: Path) -> list[SourceDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SourceDocument.model_validate(entry) for entry in data["sources"]]
