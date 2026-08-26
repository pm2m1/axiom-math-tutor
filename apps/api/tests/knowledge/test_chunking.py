from pathlib import Path

from axiom.knowledge.chunking import build_chunks, parse_sections
from axiom.knowledge.models import SourceDocument

FIXTURE = (Path(__file__).parent / "fixtures" / "sample_page.html").read_text(encoding="utf-8")


def _document(document_id: str = "sample") -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        title="Sample Page",
        source_url="https://en.wikibooks.org/wiki/Algebra/Sample",
        source_name="Wikibooks: Algebra",
        license="CC BY-SA 4.0",
        attribution="Wikibooks contributors, sample",
        retrieved="2026-08-26",
        topic="equations",
        subtopic=None,
    )


def test_drops_front_matter_and_short_stub() -> None:
    sections = parse_sections(FIXTURE)
    headings = [(section, subsection) for section, subsection, _ in sections]
    assert headings == [("First Section", None), ("Second Section", "A Subsection")]


def test_inlines_math_and_strips_editsection() -> None:
    sections = parse_sections(FIXTURE)
    _, _, text = sections[0]
    assert "x+2=5" in text
    assert "edit" not in text


def test_drops_navigation_table_and_quiz_widget() -> None:
    sections = parse_sections(FIXTURE)
    all_text = " ".join(text for _, _, text in sections)
    assert "chapter nav" not in all_text
    assert "Interactive quiz" not in all_text


def test_build_chunks_carries_provenance() -> None:
    chunks = build_chunks(_document(), FIXTURE)
    assert len(chunks) == 2
    first = chunks[0]
    assert first.source_url == "https://en.wikibooks.org/wiki/Algebra/Sample"
    assert first.license == "CC BY-SA 4.0"
    assert first.topic == "equations"
    assert first.section == "First Section"


def test_chunk_ids_are_deterministic() -> None:
    first_run = [c.chunk_id for c in build_chunks(_document(), FIXTURE)]
    second_run = [c.chunk_id for c in build_chunks(_document(), FIXTURE)]
    assert first_run == second_run == ["sample-00", "sample-01"]
