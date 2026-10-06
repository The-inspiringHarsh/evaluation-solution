"""Provider adapters with the SDK/HTTP layer mocked: JSON parsing and web-search source extraction."""

from types import SimpleNamespace as NS

import pytest

from src.common.llm import AnthropicClient, CachedLLM, ImagePart, LLMError, TextPart, _parse_json_text


def _client(monkeypatch, responses):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    client = AnthropicClient("claude-opus-5-5")
    queue = list(responses)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return queue.pop(0)

    client._client = NS(beta=NS(messages=NS(create=create)), messages=NS(create=create))
    return client, calls


def test_complete_json_sends_schema_and_images(monkeypatch):
    msg = NS(stop_reason="end_turn", content=[NS(type="text", text='{"a": 1}')])
    client, calls = _client(monkeypatch, [msg])
    out = client.complete_json(system="s", parts=[ImagePart(b"\x89PNG", "image/png", "Page 1:"), TextPart("go")], schema={"type": "object"})
    assert out == {"a": 1}
    sent = calls[0]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["effort"] == "medium"
    blocks = sent["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image", "text"]
    assert sent["fallbacks"] == "default"


def test_refusal_and_truncation_raise(monkeypatch):
    client, _ = _client(monkeypatch, [NS(stop_reason="refusal", content=[]), NS(stop_reason="max_tokens", content=[])])
    with pytest.raises(LLMError, match="declined"):
        client.complete_json(system="s", parts=[TextPart("x")], schema={})
    with pytest.raises(LLMError, match="truncated"):
        client.complete_json(system="s", parts=[TextPart("x")], schema={})


def test_web_search_collects_cited_sources(monkeypatch):
    result_block = NS(
        type="web_search_tool_result",
        content=[
            NS(url="https://a.example/turnover", title="A"),
            NS(url="https://b.example/other", title="B"),
        ],
    )
    text_block = NS(
        type="text",
        text="Turnover is COGS / average inventory.",
        citations=[
            NS(url="https://b.example/other", title="B", cited_text="COGS divided by average inventory"),
        ],
    )
    client, calls = _client(monkeypatch, [NS(stop_reason="end_turn", content=[result_block, text_block])])
    ans = client.web_search("inventory turnover")
    assert ans.answer.startswith("Turnover")
    assert ans.sources[0].url == "https://b.example/other"  # cited first
    assert {s.url for s in ans.sources} == {"https://a.example/turnover", "https://b.example/other"}
    assert calls[0]["tools"][0]["type"] == "web_search_20260209"


def test_web_search_tool_error_is_not_raised(monkeypatch):
    err = NS(type="web_search_tool_result", content=NS(error_code="max_uses_exceeded"))
    client, _ = _client(monkeypatch, [NS(stop_reason="end_turn", content=[err, NS(type="text", text="no data", citations=None)])])
    ans = client.web_search("q")
    assert ans.sources == [] and ans.answer == "no data"


def test_parse_json_text():
    assert _parse_json_text('```json\n{"x": 2}\n```') == {"x": 2}
    with pytest.raises(LLMError):
        _parse_json_text("not json")


def test_cached_llm_reuses_responses(tmp_path):
    class Inner:
        provider, model = "p", "m"
        calls = 0

        def complete_json(self, **kw):
            Inner.calls += 1
            return {"v": Inner.calls}

    cached = CachedLLM(Inner(), str(tmp_path))
    a = cached.complete_json(system="s", parts=[TextPart("q")], schema={})
    b = cached.complete_json(system="s", parts=[TextPart("q")], schema={})
    c = cached.complete_json(system="s", parts=[TextPart("other")], schema={})
    assert a == b == {"v": 1} and c == {"v": 2}
