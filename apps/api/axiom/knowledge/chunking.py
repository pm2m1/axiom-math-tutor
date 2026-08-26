"""Turn one source page's HTML into an ordered list of KnowledgeChunk records.

The MediaWiki output we ingest precedes each heading with a sibling
`<div class="mw-heading mw-heading2|mw-heading3">` rather than nesting content
inside the heading itself. So chunking walks the direct children of
`div.mw-parser-output` in document order, tracks the current h2/h3, and closes
a chunk whenever a new heading (or the end of the page) is reached. This keeps
each chunk aligned with one section or subsection, so a formula and the
paragraph explaining it stay together rather than landing in separate chunks.
"""

import re

from bs4 import BeautifulSoup, Tag

from axiom.knowledge.models import KnowledgeChunk, SourceDocument

_MATH_DISPLAYSTYLE_PREFIX = "{\\displaystyle"
_MIN_CHUNK_CHARS = 40
_CONTENT_TAGS = ("p", "ul", "ol", "table")


def _math_alttext(math_tag: Tag) -> str:
    """MediaWiki renders formulas as an <img> with a MathML fallback; both
    carry the original LaTeX in `alttext`. Using that keeps formulas as text
    instead of dropping them or leaving a bare image alt attribute."""
    text = str(math_tag.get("alttext", "")).strip()
    if text.startswith(_MATH_DISPLAYSTYLE_PREFIX) and text.endswith("}"):
        text = text[len(_MATH_DISPLAYSTYLE_PREFIX) : -1]
    return text.strip()


def _block_text(tag: Tag) -> str:
    fragment = BeautifulSoup(str(tag), "html.parser")
    for editsection in fragment.find_all(class_="mw-editsection"):
        editsection.decompose()
    for math_element in fragment.find_all(class_="mwe-math-element"):
        math_tag = math_element.find("math")
        math_element.replace_with(_math_alttext(math_tag) if math_tag else "")
    return re.sub(r"\s+", " ", fragment.get_text(" ", strip=True))


def parse_sections(html: str) -> list[tuple[str | None, str | None, str]]:
    """Return (section, subsection, text) tuples in document order.

    Sections under `_MIN_CHUNK_CHARS` are dropped after whitespace
    normalization — in practice these are empty stubs or interactive quiz
    widgets with no static text, not real content.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output")
    if content is None:
        return []

    sections: list[tuple[str | None, str | None, str]] = []
    section: str | None = None
    subsection: str | None = None
    parts: list[str] = []

    def flush() -> None:
        text = " ".join(part for part in parts if part).strip()
        if len(text) >= _MIN_CHUNK_CHARS:
            sections.append((section, subsection, text))
        parts.clear()

    seen_heading = False
    for child in content.find_all(True, recursive=False):
        raw_classes: str | list[str] = child.get("class") or []
        classes = raw_classes if isinstance(raw_classes, list) else [raw_classes]
        if child.name == "div" and "mw-heading" in classes:
            flush()
            seen_heading = True
            heading = child.find(["h2", "h3", "h4"])
            heading_text = heading.get_text(strip=True) if heading else None
            if "mw-heading2" in classes:
                section, subsection = heading_text, None
            else:
                subsection = heading_text
            continue
        # "noprint" catches the prev/next chapter navigation table; "quiz"
        # catches the interactive end-of-section quiz widgets.
        if "noprint" in classes or "quiz" in classes:
            continue
        # Every page in this source opens with a chapter/section breadcrumb
        # table before its first heading; the markup for it isn't consistent
        # enough to match by class, but no page puts real content in a table
        # ahead of its first heading, so it's safe to drop unconditionally.
        if child.name == "table" and not seen_heading:
            continue
        if child.name in _CONTENT_TAGS:
            block = _block_text(child)
            if block:
                parts.append(block)

    flush()
    return sections


def build_chunks(document: SourceDocument, html: str) -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id=f"{document.document_id}-{index:02d}",
            document_id=document.document_id,
            title=document.title,
            section=section,
            subsection=subsection,
            text=text,
            source_url=document.source_url,
            source_name=document.source_name,
            license=document.license,
            topic=document.topic,
            subtopic=document.subtopic,
        )
        for index, (section, subsection, text) in enumerate(parse_sections(html))
    ]
