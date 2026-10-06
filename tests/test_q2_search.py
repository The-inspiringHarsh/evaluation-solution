"""Search providers: source formatting, URL filtering and provider selection."""

import pytest

from src.common.llm import SearchAnswer, SearchSource
from src.question2_inventory.search import (
    DuckDuckGoSearch,
    FallbackSearch,
    LLMNativeSearch,
    SearchError,
    build_search_provider,
    format_sources_markdown,
    safe_search,
)


class _Static:
    name = "static"

    def __init__(self, answer):
        self.answer = answer
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.answer


def test_format_sources_markdown_is_clickable():
    md = format_sources_markdown([SearchSource("Title [x]", "https://example.com/a"), SearchSource("B", "https://b.org")])
    assert md.splitlines()[0] == "1. [Title (x)](https://example.com/a)"
    assert md.splitlines()[1] == "2. [B](https://b.org)"


def test_format_sources_empty():
    assert "No web sources" in format_sources_markdown([])


def test_safe_search_drops_non_http_sources_and_scrubs():
    provider = _Static(SearchAnswer("q", "a", [SearchSource("ok", "https://ok.com"), SearchSource("bad", "javascript:alert(1)")]))
    ans = safe_search(provider, "account 123456789012 meaning")
    assert [s.url for s in ans.sources] == ["https://ok.com"]
    assert "123456789012" not in provider.queries[0]


def test_safe_search_rejects_empty_query():
    with pytest.raises(SearchError):
        safe_search(_Static(SearchAnswer("q", "a")), "   ")


def test_provider_selection(monkeypatch, fake_llm_factory):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    llm = fake_llm_factory()
    llm.provider = "anthropic"
    auto = build_search_provider("auto", llm)
    assert isinstance(auto, FallbackSearch) and auto.name == "llm, then duckduckgo"
    assert isinstance(auto._providers[0], LLMNativeSearch) and isinstance(auto._providers[1], DuckDuckGoSearch)
    assert isinstance(build_search_provider("auto", None), DuckDuckGoSearch)
    with pytest.raises(SearchError):
        build_search_provider("tavily", None)


class _Failing:
    name = "broken"

    def search(self, query):
        raise SearchError("grounding not available on this tier")


def test_fallback_search_uses_next_provider_and_reports_who_answered():
    good = _Static(SearchAnswer("q", "", [SearchSource("Investopedia", "https://example.org/turnover")], provider="duckduckgo"))
    ans = FallbackSearch([_Failing(), good]).search("inventory turnover")
    assert ans.provider == "duckduckgo" and good.queries == ["inventory turnover"]


def test_fallback_search_raises_when_every_provider_fails():
    with pytest.raises(SearchError, match="broken: grounding not available"):
        FallbackSearch([_Failing(), _Failing()]).search("q")


def test_fallback_search_keeps_an_unsourced_answer_as_last_resort():
    unsourced = _Static(SearchAnswer("q", "Turnover is COGS / average inventory.", []))
    assert FallbackSearch([unsourced, _Failing()]).search("q").answer.startswith("Turnover")
