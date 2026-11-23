from typing import Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class KBSource(BaseModel):
    chunk_id: str
    title: str
    section: str | None
    subsection: str | None
    source_url: str


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str


class EvidenceRoutingResponse(BaseModel):
    """`route` discriminates which of `kb_sources`/`web_sources` is
    meaningful; exactly one is non-empty except when `route` is
    `"unavailable"` or the web search legitimately found nothing.
    `detail` is always a short, generic, hand-written string — never a
    raw provider error, credential, or traceback."""

    route: Literal["kb", "web_search", "unavailable"]
    kb_sources: list[KBSource] = []
    web_sources: list[WebSource] = []
    reason: Literal["judge_error", "search_error", "retrieval_error"] | None = None
    detail: str | None = None
