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


# --------------------------------------------------------------------------- Gemini (REST, mocked)

import httpx  # noqa: E402

from src.common.config import get_settings  # noqa: E402
from src.common.llm import GeminiClient, _Pacer  # noqa: E402

_OK = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "plan", "thought": True}, {"text": '{"ok": true}'}]}}]}


def _gemini(monkeypatch, responses, model="gemini-3.6-flash", **kwargs):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-not-real")
    client = GeminiClient(model, **kwargs)
    queue = list(responses)
    calls, sleeps = [], []

    def post(url, json, headers, timeout):
        calls.append((url, json))
        status, payload = queue.pop(0)
        return NS(status_code=status, json=lambda: payload)

    client._httpx = NS(post=post, HTTPError=httpx.HTTPError)
    client._sleep = sleeps.append
    return client, calls, sleeps


def _quota_error(*quota_ids, retry="7s"):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": q} for q in quota_ids]}]
    if retry:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
    return 429, {"error": {"code": 429, "message": "You exceeded your current quota.\n* details", "details": details}}


def test_gemini_retries_overload_then_succeeds(monkeypatch):
    client, calls, sleeps = _gemini(monkeypatch, [(503, {"error": {"message": "high demand"}}), (200, _OK)])
    assert client.complete_json(system="s", parts=[TextPart("x")], schema={"type": "object"}) == {"ok": True}
    assert len(calls) == 2 and len(sleeps) == 1
    assert client.calls_by_model == {"gemini-3.6-flash": 1}


def test_gemini_honours_server_retry_delay_for_per_minute_quota(monkeypatch):
    client, calls, sleeps = _gemini(monkeypatch, [_quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), (200, _OK)])
    client.complete_json(system="s", parts=[TextPart("x")], schema={"type": "object"})
    assert sleeps == [7.0] and len(calls) == 2


def test_gemini_daily_quota_moves_to_fallback_model(monkeypatch):
    responses = [_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), (200, _OK)]
    client, calls, sleeps = _gemini(monkeypatch, responses, fallback_models=["gemini-3.8-flash"])
    client.complete_json(system="s", parts=[TextPart("x")], schema={"type": "object"})
    assert "gemini-3.6-flash:" in calls[0][0] and "gemini-3.8-flash:" in calls[1][0]
    assert sleeps == []  # a per-day quota is not waited out
    assert client.calls_by_model == {"gemini-3.8-flash": 1}
    only, _, _ = _gemini(monkeypatch, [_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(LLMError, match="quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier limit"):
        only.complete_json(system="s", parts=[TextPart("x")], schema={"type": "object"})


def test_gemini_bad_request_fails_fast_without_leaking_the_key(monkeypatch):
    error = {"error": {"message": "Bad schema near test-gemini-key-not-real"}}
    client, calls, _ = _gemini(monkeypatch, [(400, error), (200, _OK)])
    with pytest.raises(LLMError) as info:
        client.complete_json(system="s", parts=[TextPart("x")], schema={"type": "object"})
    assert "HTTP 400" in str(info.value) and "test-gemini-key-not-real" not in str(info.value)
    assert len(calls) == 1


def test_gemini_gives_up_after_max_retries(monkeypatch):
    client, calls, sleeps = _gemini(monkeypatch, [(503, {})] * 3, max_retries=2)
    with pytest.raises(LLMError, match="HTTP 503"):
        client.complete_text(system="s", messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 3 and len(sleeps) == 2


def test_gemini_request_shape_and_truncation(monkeypatch):
    schema = {"type": "object", "properties": {"v": {"type": ["string", "null"]}}, "required": ["v"], "additionalProperties": False}
    truncated = {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": '{"v": "12'}]}}]}
    client, calls, _ = _gemini(monkeypatch, [(200, _OK), (200, truncated)])
    client.complete_json(system="s", parts=[ImagePart(b"\x89PNG", "image/png", "Page 1:"), TextPart("go")], schema=schema)
    config = calls[0][1]["generationConfig"]
    assert config["responseSchema"]["properties"]["v"] == {"type": "string", "nullable": True}
    assert "additionalProperties" not in config["responseSchema"]
    assert "temperature" not in config  # Gemini 3+ keeps its default temperature
    assert [list(p) for p in calls[0][1]["contents"][0]["parts"]] == [["text"], ["inline_data"], ["text"]]
    with pytest.raises(LLMError, match="truncated"):
        client.complete_json(system="s", parts=[TextPart("x")], schema=schema)
    legacy, legacy_calls, _ = _gemini(monkeypatch, [(200, _OK)], model="gemini-2.0-flash")
    legacy.complete_json(system="s", parts=[TextPart("x")], schema=schema)
    assert legacy_calls[0][1]["generationConfig"]["temperature"] == 0


def test_gemini_web_search_reads_grounding_sources(monkeypatch):
    data = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": "Turnover is COGS / average inventory."}]},
                "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://a.example/t", "title": "a.example"}}]},
            }
        ]
    }
    client, calls, _ = _gemini(monkeypatch, [(200, data)])
    ans = client.web_search("inventory turnover")
    assert ans.answer.startswith("Turnover") and [s.url for s in ans.sources] == ["https://a.example/t"]
    assert calls[0][1]["tools"] == [{"google_search": {}}]


def test_pacer_spaces_request_starts():
    pacer, slept = _Pacer(rpm=60), []
    pacer.wait(slept.append)
    pacer.wait(slept.append)
    assert slept and 0.9 < slept[-1] <= 1.0
    unlimited, none = _Pacer(rpm=0), []
    unlimited.wait(none.append)
    assert none == []


def test_settings_read_gemini_pacing_and_fallbacks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", " gemini-3.8-flash , ,gemini-3.5-flash")
    monkeypatch.setenv("LLM_MAX_RPM", "8")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    s = get_settings()
    assert s.llm_fallback_models == ("gemini-3.8-flash", "gemini-3.5-flash")
    assert s.llm_max_rpm == 8.0 and s.llm_max_retries == 3 and s.llm_model == "gemini-3.6-flash"
