"""Web-search tool for external definitions/business context.

Providers (``SEARCH_PROVIDER``):

* ``tavily``    - Tavily REST API (needs ``TAVILY_API_KEY``)
* ``llm``       - the LLM provider's own grounded search (Claude web search / Gemini Google Search)
* ``duckduckgo``- key-free fallback via the ``ddgs`` package (snippets only)
* ``auto``      - tavily if a key exists, else llm, else duckduckgo

Every query is scrubbed of identity/financial numbers before leaving the process.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from ..common.llm import LLMClient, LLMError, SearchAnswer, SearchSource
from ..common.privacy import scrub_pii

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Search failed; message is safe to show."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str) -> SearchAnswer: ...


class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self._key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self._key:
            raise SearchError("TAVILY_API_KEY is not set.")

    def search(self, query: str) -> SearchAnswer:
        import httpx

        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": 5, "include_answer": True},
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            raise SearchError("Could not reach Tavily (network error).") from exc
        if resp.status_code != 200:
            raise SearchError(f"Tavily error (HTTP {resp.status_code}).")
        data = resp.json()
        sources = [
            SearchSource(title=r.get("title") or r["url"], url=r["url"], snippet=(r.get("content") or "")[:300])
            for r in data.get("results", [])
            if r.get("url")
        ]
        return SearchAnswer(query=query, answer=data.get("answer") or "", sources=sources, provider="tavily")


class LLMNativeSearch:
    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def search(self, query: str) -> SearchAnswer:
        try:
            return self._llm.web_search(query)
        except LLMError as exc:
            raise SearchError(str(exc)) from exc


class DuckDuckGoSearch:
    """Key-free fallback: returns real result snippets; the agent summarises them with citations."""

    name = "duckduckgo"

    def search(self, query: str) -> SearchAnswer:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise SearchError("Key-free search needs the 'ddgs' package: pip install ddgs") from exc
        try:
            results = list(DDGS().text(query, max_results=5))
        except Exception as exc:  # ddgs raises several transport-specific exceptions
            raise SearchError(f"DuckDuckGo search failed: {type(exc).__name__}") from exc
        sources = [
            SearchSource(title=r.get("title") or r.get("href", ""), url=r["href"], snippet=(r.get("body") or "")[:300])
            for r in results
            if r.get("href")
        ]
        return SearchAnswer(query=query, answer="", sources=sources, provider="duckduckgo")


def build_search_provider(mode: str, llm: LLMClient | None) -> SearchProvider:
    """Pick a provider according to ``SEARCH_PROVIDER`` and what is configured."""
    mode = (mode or "auto").lower()
    if mode == "tavily":
        return TavilySearch()
    if mode == "llm":
        if llm is None:
            raise SearchError("SEARCH_PROVIDER=llm needs a configured LLM provider.")
        return LLMNativeSearch(llm)
    if mode == "duckduckgo":
        return DuckDuckGoSearch()
    if os.environ.get("TAVILY_API_KEY"):
        return TavilySearch()
    if llm is not None and getattr(llm, "provider", "") in {"anthropic", "gemini"}:
        return LLMNativeSearch(llm)
    return DuckDuckGoSearch()


def safe_search(provider: SearchProvider, query: str) -> SearchAnswer:
    """Scrub PII from the query, run the search and drop sources without a usable URL."""
    clean = scrub_pii(query).strip()
    if not clean:
        raise SearchError("The search query was empty after removing sensitive values.")
    answer = provider.search(clean)
    answer.sources = [s for s in answer.sources if s.url.startswith(("http://", "https://"))]
    return answer


def format_sources_markdown(sources: list[SearchSource], limit: int = 6) -> str:
    """Numbered markdown list of clickable sources."""
    if not sources:
        return "_No web sources were returned._"
    lines = []
    for i, src in enumerate(sources[:limit], start=1):
        title = (src.title or src.url).replace("[", "(").replace("]", ")").strip()
        lines.append(f"{i}. [{title}]({src.url})")
    return "\n".join(lines)
