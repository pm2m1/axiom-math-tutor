from pydantic import BaseModel


class SourceDocument(BaseModel):
    """One source page declared in a manifest, before chunking.

    Carries the provenance a manifest is responsible for recording. Not tied
    to any storage or retrieval backend.
    """

    document_id: str
    title: str
    source_url: str
    source_name: str
    license: str
    attribution: str
    retrieved: str
    topic: str
    subtopic: str | None = None


class KnowledgeChunk(BaseModel):
    """A retrievable unit of curriculum content, with enough provenance to
    cite it back to the student."""

    chunk_id: str
    document_id: str
    title: str
    section: str | None
    subsection: str | None
    text: str
    source_url: str
    source_name: str
    license: str
    topic: str
    subtopic: str | None = None
