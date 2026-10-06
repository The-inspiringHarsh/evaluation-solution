"""Thin provider interface for vision-capable LLMs.

Every provider exposes the same three calls:

* ``complete_json`` - images/text in, schema-constrained JSON out (used by both questions)
* ``complete_text`` - plain text answer for summaries
* ``web_search``   - grounded search with source URLs (only where the provider offers it)

Provider SDKs are imported lazily so the app starts without any of them configured.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, Union

from .config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised for any provider failure; message is safe to show to users (no secrets)."""


class LLMNotConfigured(LLMError):
    """Raised when the configured provider has no API key."""


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/png"
    label: str = ""


Part = Union[TextPart, ImagePart]


@dataclass
class SearchSource:
    title: str
    url: str
    snippet: str = ""


@dataclass
class SearchAnswer:
    """Answer text grounded in web results, with the sources that support it."""

    query: str
    answer: str
    sources: list[SearchSource] = field(default_factory=list)
    provider: str = ""


class LLMClient(Protocol):
    provider: str
    model: str

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]: ...

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str: ...

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer: ...


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned malformed JSON.") from exc
    if not isinstance(value, dict):
        raise LLMError("The model returned JSON that is not an object.")
    return value


# --------------------------------------------------------------------------- Anthropic


class AnthropicClient:
    """Claude via the official ``anthropic`` SDK (structured outputs + server-side web search)."""

    provider = "anthropic"

    def __init__(self, model: str, effort: str = "medium") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is not installed. Run: pip install anthropic") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=3, timeout=300.0)
        self.model = model
        self.effort = effort
        # Server-side refusal fallback (routes a declined request to a fallback model).
        self._use_fallbacks = os.getenv("LLM_FALLBACKS", "default").strip().lower() != "off"
        self._stats_lock = threading.Lock()
        self.calls_by_model: dict[str, int] = {}

    def _create(self, **kwargs: Any) -> Any:
        anthropic = self._anthropic
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("output_config", {})
        kwargs["output_config"].setdefault("effort", self.effort)
        try:
            if self._use_fallbacks:
                msg = self._client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                msg = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (authentication error).") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached; retry shortly.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Anthropic rejected the request: {getattr(exc, 'message', 'bad request')}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error (HTTP {exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API (network error).") from exc
        served_by = getattr(msg, "model", None)
        served_by = served_by if isinstance(served_by, str) and served_by else kwargs["model"]
        with self._stats_lock:
            self.calls_by_model[served_by] = self.calls_by_model.get(served_by, 0) + 1
        if msg.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        return msg

    @staticmethod
    def _content(parts: Sequence[Part]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, ImagePart):
                if part.label:
                    blocks.append({"type": "text", "text": part.label})
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": part.media_type, "data": _b64(part.data)}})
            else:
                blocks.append({"type": "text", "text": part.text})
        return blocks

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
        msg = self._create(
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": self._content(parts)}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if msg.stop_reason == "max_tokens":
            raise LLMError("The model response was truncated (max_tokens).")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _parse_json_text(text)

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        msg = self._create(max_tokens=max_tokens, system=system, messages=list(messages))
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer:
        system = (
            "You answer business/definition questions using web search. Cite sources. "
            "Be concise (under 150 words). Do not speculate beyond the sources."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}]
        answer_chunks: list[str] = []
        sources: dict[str, SearchSource] = {}
        for _ in range(4):  # handle pause_turn continuations
            msg = self._create(max_tokens=4000, system=system, messages=messages, tools=tools)
            for block in msg.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    answer_chunks.append(block.text)
                    for cit in getattr(block, "citations", None) or []:
                        url = getattr(cit, "url", None)
                        if url:
                            src = sources.setdefault(url, SearchSource(title=getattr(cit, "title", "") or url, url=url))
                            if not src.snippet:
                                src.snippet = (getattr(cit, "cited_text", "") or "")[:300]
                elif btype == "web_search_tool_result":
                    content = getattr(block, "content", None)
                    if isinstance(content, list):
                        for item in content:
                            url = getattr(item, "url", None)
                            if url and url not in sources:
                                sources[url] = SearchSource(title=getattr(item, "title", "") or url, url=url)
                    else:
                        code = getattr(content, "error_code", "unknown")
                        logger.warning("web_search tool error: %s", code)
            if msg.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": msg.content}]
        cited_first = sorted(sources.values(), key=lambda s: (s.snippet == "",))
        return SearchAnswer(query=query, answer="".join(answer_chunks).strip(), sources=cited_first, provider="anthropic_web_search")


# --------------------------------------------------------------------------- Gemini (REST)


def _schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert our JSON schema (``type: [T, "null"]``) into Gemini's OpenAPI subset."""
    out = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            t = node.get("type")
            if isinstance(t, list):
                non_null = [x for x in t if x != "null"]
                node["type"] = non_null[0] if non_null else "string"
                if "null" in t:
                    node["nullable"] = True
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        return node

    return walk(out)


_GEMINI_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _gemini_major_version(model: str) -> int | None:
    match = re.match(r"gemini-(\d+)", model)
    return int(match.group(1)) if match else None


def _duration_seconds(text: str) -> float | None:
    """Parse a protobuf Duration such as ``"37s"`` or ``"0.5s"``."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", text or "")
    return float(match.group(1)) if match else None


@dataclass
class _GeminiError:
    status: int
    message: str = ""
    retry_after: float | None = None
    daily_quota: bool = False


def _gemini_error(resp: Any, key: str = "") -> _GeminiError:
    """Pull the first line of the API message, ``RetryInfo.retryDelay`` and whether a per-day quota was hit."""
    info = _GeminiError(resp.status_code)
    try:
        err = resp.json().get("error", {})
    except (ValueError, AttributeError):
        return info
    message = str(err.get("message") or "").strip()
    if message:
        info.message = message.splitlines()[0][:200]
        if key:
            info.message = info.message.replace(key, "<key>")
    for detail in err.get("details") or []:
        kind = str(detail.get("@type", ""))
        if kind.endswith("RetryInfo"):
            info.retry_after = _duration_seconds(str(detail.get("retryDelay", "")))
        elif kind.endswith("QuotaFailure"):
            violations = detail.get("violations") or []
            info.daily_quota = any("PerDay" in str(v.get("quotaId", "")) for v in violations)
            limits = [f"{v.get('quotaId')} limit {v.get('quotaValue', '?')}" for v in violations if v.get("quotaId")]
            if limits:
                info.message = "quota exceeded: " + "; ".join(limits)
    return info


class _Pacer:
    """Spaces request starts so one process sends at most ``rpm`` requests a minute, across all threads."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self, sleep: Any = time.sleep) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            sleep(start - now)


_PACERS: dict[str, _Pacer] = {}
_PACERS_LOCK = threading.Lock()


def _pacer_for(model: str, rpm: float) -> _Pacer:
    """One pacer per model, shared by every client in the process (quotas are per model)."""
    with _PACERS_LOCK:
        pacer = _PACERS.get(model)
        if pacer is None or pacer.interval != (60.0 / rpm if rpm > 0 else 0.0):
            pacer = _PACERS[model] = _Pacer(rpm)
        return pacer


class GeminiClient:
    """Google Gemini via its REST API (JSON mode + Google Search grounding).

    Transient failures (HTTP 429 per-minute quota, 5xx "high demand", network errors) are retried with
    exponential backoff, honouring the server's ``retryDelay``. ``max_rpm`` paces requests per model so
    free-tier per-minute limits are not hit in the first place. ``fallback_models`` are tried in order when
    a model is overloaded, unavailable or out of daily quota; :attr:`calls_by_model` records which model
    answered each call so outputs can report it.
    """

    provider = "gemini"
    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, fallback_models: Sequence[str] = (), max_rpm: float = 0.0, max_retries: int = 5) -> None:
        import httpx

        self._httpx = httpx
        self.model = model
        self._models = [model, *[m for m in fallback_models if m and m != model]]
        self._key = os.environ.get("GEMINI_API_KEY", "")
        self._rpm = max_rpm
        self._max_retries = max(0, max_retries)
        self._sleep = time.sleep
        self._stats_lock = threading.Lock()
        self.calls_by_model: dict[str, int] = {}

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(60.0, 2.0 * 2**attempt) * (0.75 + random.random() / 2)

    def _post(self, body: dict[str, Any], retries: int | None = None, fallbacks: bool = True) -> dict[str, Any]:
        problem = "Gemini API error."
        max_retries = self._max_retries if retries is None else min(retries, self._max_retries)
        models = self._models if fallbacks else self._models[:1]
        for model in models:
            url = f"{self._BASE}/{model}:generateContent"
            for attempt in range(max_retries + 1):
                _pacer_for(model, self._rpm).wait(self._sleep)
                try:
                    resp = self._httpx.post(url, json=body, headers={"x-goog-api-key": self._key}, timeout=300.0)
                except self._httpx.HTTPError:
                    problem = "Could not reach the Gemini API (network error)."
                    delay = self._backoff(attempt)
                else:
                    if resp.status_code == 200:
                        with self._stats_lock:
                            self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
                        return resp.json()
                    err = _gemini_error(resp, self._key)
                    problem = f"Gemini API error (HTTP {err.status}" + (f": {err.message}" if err.message else "") + ")."
                    if err.status == 404 or (err.status == 429 and err.daily_quota):
                        break  # this model cannot serve today: try the next one
                    if err.status not in _GEMINI_RETRY_STATUS:
                        raise LLMError(problem)
                    delay = err.retry_after if err.retry_after is not None else self._backoff(attempt)
                if attempt < max_retries:
                    logger.warning("%s Retrying %s (attempt %d) in %.0fs.", problem, model, attempt + 2, delay)
                    self._sleep(min(delay, 120.0))
            if model != models[-1]:
                logger.warning("%s Switching from %s to the next fallback model.", problem, model)
        raise LLMError(problem)

    @staticmethod
    def _parts(parts: Sequence[Part]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, ImagePart):
                if part.label:
                    out.append({"text": part.label})
                out.append({"inline_data": {"mime_type": part.media_type, "data": _b64(part.data)}})
            else:
                out.append({"text": part.text})
        return out

    @staticmethod
    def _text(data: dict[str, Any]) -> str:
        try:
            candidate = data["candidates"][0]
        except (KeyError, IndexError) as exc:
            raise LLMError("Gemini returned no candidates (possibly blocked).") from exc
        reason = candidate.get("finishReason", "")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if reason == "MAX_TOKENS":
            raise LLMError("The model response was truncated (max tokens).")
        if not text and reason not in ("", "STOP"):
            raise LLMError(f"Gemini returned no text (finish reason {reason}).")
        return text

    def _generation_config(self, **config: Any) -> dict[str, Any]:
        # Gemini 3+ is tuned for the default temperature (lower values can cause looping), so only
        # older models get a fixed temperature.
        temperature = config.pop("temperature")
        if (_gemini_major_version(self.model) or 3) < 3:
            config["temperature"] = temperature
        return config

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": self._parts(parts)}],
            "generationConfig": self._generation_config(
                responseMimeType="application/json",
                responseSchema=_schema_for_gemini(schema),
                maxOutputTokens=max(max_tokens, 16000),  # thinking tokens count against this limit
                temperature=0,
            ),
        }
        return _parse_json_text(self._text(self._post(body)))

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": self._generation_config(maxOutputTokens=max(max_tokens, 8000), temperature=0.2),
        }
        return self._text(self._post(body)).strip()

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer:
        body = {
            "contents": [{"role": "user", "parts": [{"text": query + "\nAnswer concisely (under 150 words)."}]}],
            "tools": [{"google_search": {}}],
        }
        # Grounding quota is separate from generation quota (and absent on some tiers), so fail fast here and
        # let the search layer fall back to another provider instead of cycling through models.
        data = self._post(body, retries=1, fallbacks=False)
        answer = self._text(data).strip()
        sources: list[SearchSource] = []
        meta = (data.get("candidates") or [{}])[0].get("groundingMetadata", {})
        for chunk in meta.get("groundingChunks", []):
            web = chunk.get("web") or {}
            if web.get("uri"):
                sources.append(SearchSource(title=web.get("title", web["uri"]), url=web["uri"]))
        return SearchAnswer(query=query, answer=answer, sources=sources, provider="gemini_google_search")


# --------------------------------------------------------------------------- OpenAI (REST)


class OpenAIClient:
    """OpenAI chat completions via REST (strict JSON schema). No built-in web search here."""

    provider = "openai"
    _URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str) -> None:
        import httpx

        self._httpx = httpx
        self.model = model
        self._key = os.environ.get("OPENAI_API_KEY", "")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._httpx.post(self._URL, json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=300.0)
        except self._httpx.HTTPError as exc:
            raise LLMError("Could not reach the OpenAI API (network error).") from exc
        if resp.status_code != 200:
            raise LLMError(f"OpenAI API error (HTTP {resp.status_code}).")
        return resp.json()

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, ImagePart):
                if part.label:
                    content.append({"type": "text", "text": part.label})
                url = f"data:{part.media_type};base64,{_b64(part.data)}"
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
            else:
                content.append({"type": "text", "text": part.text})
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}},
        }
        data = self._post(body)
        return _parse_json_text(data["choices"][0]["message"]["content"] or "")

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        return (self._post(body)["choices"][0]["message"]["content"] or "").strip()

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer:
        raise LLMError("The OpenAI provider has no built-in web search here; configure TAVILY_API_KEY.")


def get_llm_client(settings: Settings) -> LLMClient:
    """Build the configured client or raise :class:`LLMNotConfigured` with setup guidance."""
    if not settings.llm_available:
        raise LLMNotConfigured(settings.setup_hint())
    if settings.llm_provider == "anthropic":
        return AnthropicClient(settings.llm_model, settings.llm_effort)
    if settings.llm_provider == "gemini":
        return GeminiClient(
            settings.llm_model,
            fallback_models=settings.llm_fallback_models,
            max_rpm=settings.llm_max_rpm,
            max_retries=settings.llm_max_retries,
        )
    return OpenAIClient(settings.llm_model)


# --------------------------------------------------------------------------- response cache


class CachedLLM:
    """Disk cache for ``complete_json`` so re-runs (tests, UI demos) do not re-bill identical requests.

    The cache holds extracted values, so it lives under ``.cache/`` which is git-ignored.
    """

    def __init__(self, inner: LLMClient, cache_dir: str) -> None:
        import hashlib
        from pathlib import Path

        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hash = hashlib.sha256
        self.provider = inner.provider
        self.model = inner.model
        self.cache_hits = 0

    def _key(self, system: str, parts: Sequence[Part], schema: dict[str, Any]) -> str:
        h = self._hash()
        h.update(f"{self.provider}|{self.model}|{system}|{json.dumps(schema, sort_keys=True)}".encode())
        for part in parts:
            h.update(part.data if isinstance(part, ImagePart) else part.text.encode())
            if isinstance(part, ImagePart):
                h.update(part.label.encode())
        return h.hexdigest()

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
        path = self._dir / f"{self._key(system, parts, schema)}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        value = self._inner.complete_json(system=system, parts=parts, schema=schema, max_tokens=max_tokens)
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    @property
    def calls_by_model(self) -> dict[str, int]:
        """Live (uncached) calls per model that actually answered, as counted by the wrapped client."""
        return dict(getattr(self._inner, "calls_by_model", {}))

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        return self._inner.complete_text(system=system, messages=messages, max_tokens=max_tokens)

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer:
        return self._inner.web_search(query, max_uses=max_uses)


def get_cached_llm_client(settings: Settings) -> LLMClient:
    """Configured client wrapped in the disk cache unless ``LLM_CACHE=off``."""
    from .config import PROJECT_ROOT

    client = get_llm_client(settings)
    if os.getenv("LLM_CACHE", "on").strip().lower() == "off":
        return client
    return CachedLLM(client, str(PROJECT_ROOT / ".cache" / "llm"))
