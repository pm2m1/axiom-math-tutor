"""Web-search fallback via Tavily's official hosted MCP server (Phase
1J), used only when the KB evidence judge finds the top-5 retrieved
evidence insufficient. Each call opens a fresh MCP connection — no
session is kept between requests, and nothing here touches the Phase 1I
benchmark's `JudgeCache`; live requests and benchmark execution stay
separate.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
MAX_WEB_RESULTS = 3
SEARCH_TIMEOUT_SECONDS = 15.0


@dataclass
class WebSource:
    title: str
    url: str
    snippet: str


@dataclass
class WebSearchOutcome:
    """`results` may legitimately be empty — a real zero-result search,
    not a failure. `error` is set only when the search itself could not
    be completed (MCP connection/tool-call failure)."""

    results: list[WebSource]
    error: str | None


class WebSearchClient(Protocol):
    def search(self, question: str) -> WebSearchOutcome: ...


def _find_search_tool(tools: list[Any]) -> str | None:
    for tool in tools:
        name = getattr(tool, "name", "")
        if "search" in name.lower():
            return name
    return None


def _entries_from_structured(structured: Any) -> list[dict[str, Any]]:
    if isinstance(structured, list):
        return [entry for entry in structured if isinstance(entry, dict)]
    if isinstance(structured, dict):
        results = structured.get("results")
        if isinstance(results, list):
            return [entry for entry in results if isinstance(entry, dict)]
    return []


def _entries_from_text_content(content: list[Any]) -> list[dict[str, Any]]:
    entries = []
    for item in content:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        entries.extend(_entries_from_structured(parsed))
    return entries


def _to_web_sources(entries: list[dict[str, Any]], limit: int) -> list[WebSource]:
    return [
        WebSource(
            title=str(entry.get("title", "")),
            url=str(entry.get("url", "")),
            snippet=str(entry.get("content") or entry.get("snippet") or ""),
        )
        for entry in entries[:limit]
    ]


class TavilyMCPSearchClient:
    """One search = one fresh MCP connection: initialize, discover the
    search tool via `list_tools()` (Tavily's own docs disagree on the
    tool name's exact spelling, so it isn't hardcoded), call it with the
    question only, and parse whichever shape the result carries —
    `structured_content` first, text content as a fallback."""

    def __init__(self, *, max_results: int = MAX_WEB_RESULTS) -> None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set.")
        self._api_key = api_key
        self._max_results = max_results

    def search(self, question: str) -> WebSearchOutcome:
        try:
            return asyncio.run(self._search_async(question))
        except Exception as exc:
            return WebSearchOutcome([], str(exc))

    async def _search_async(self, question: str) -> WebSearchOutcome:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.types import CallToolResult

        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx2.AsyncClient(
            headers=headers, timeout=SEARCH_TIMEOUT_SECONDS
        ) as http_client:
            async with streamable_http_client(TAVILY_MCP_URL, http_client=http_client) as (
                read,
                write,
            ):
                async with ClientSession(
                    read, write, read_timeout_seconds=SEARCH_TIMEOUT_SECONDS
                ) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    tool_name = _find_search_tool(listed.tools)
                    if tool_name is None:
                        return WebSearchOutcome([], "no search tool found on MCP server")

                    result = await session.call_tool(
                        tool_name, {"query": question, "max_results": self._max_results}
                    )
                    if not isinstance(result, CallToolResult):
                        return WebSearchOutcome([], "unexpected MCP result type")
                    if result.is_error:
                        return WebSearchOutcome([], "MCP search tool reported an error")

                    entries = _entries_from_structured(result.structured_content)
                    if not entries:
                        entries = _entries_from_text_content(result.content)
                    return WebSearchOutcome(_to_web_sources(entries, self._max_results), None)
