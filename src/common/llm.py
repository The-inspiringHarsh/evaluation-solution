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


_SECRETISH = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|tvly-[A-Za-z0-9_\-]{8,})")


def _safe_detail(message: str, limit: int = 200) -> str:
    """First line of a provider error message, truncated, with anything key-shaped removed."""
    line = (str(message).strip().splitlines() or [""])[0]
    return _SECRETISH.sub("<redacted>", line)[:limit]


def _parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        # Some OpenAI-compatible models wrap the object in prose; accept the first complete JSON object.
        start = text.find("{")
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else (None, 0)
        except json.JSONDecodeError:
            value = None
        if value is None:
            raise LLMError("The model returned malformed JSON.") from exc
    if not isinstance(value, dict):
        raise LLMError("The model returned JSON that is not an object.")
    return value


class _LastCall:
    """Per-thread record of which model answered this thread's latest call and whether it came from a cache.

    Calls run in worker threads and the Streamlit client is shared, so a per-thread value is the only way to
    attribute a response to the model that produced it without changing the provider interface.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def set(self, model: str | None, cached: bool = False) -> None:
        self._local.value = (model, cached)

    def get(self) -> tuple[str | None, bool]:
        return getattr(self._local, "value", (None, False))


def _last_call(client: Any) -> tuple[str | None, bool]:
    """``(model, from_cache)`` for the calling thread's latest successful call on ``client``."""
    value = getattr(client, "last_call", None)
    return value if isinstance(value, tuple) else (None, False)


# --------------------------------------------------------------------------- Anthropic


class AnthropicClient:
    """Claude via the official ``anthropic`` SDK (structured outputs + server-side web search)."""

    provider = "anthropic"

    def __init__(self, model: str, effort: str = "medium", timeout_s: float = 180.0, max_retries: int = 3) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is not installed. Run: pip install anthropic") from exc
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise LLMNotConfigured("ANTHROPIC_API_KEY is not set.")
        self._anthropic = anthropic
        # The key is passed explicitly so only ANTHROPIC_API_KEY is ever used (never another auth variable).
        # The SDK retries 408/409/429/5xx and connection errors with backoff, up to ``max_retries`` times.
        self._client = anthropic.Anthropic(api_key=key, max_retries=max(0, min(max_retries, 5)), timeout=timeout_s)
        self.model = model
        self.effort = effort
        # Server-side refusal fallback (routes a declined request to a fallback model).
        self._use_fallbacks = os.getenv("LLM_FALLBACKS", "default").strip().lower() != "off"
        self._stats_lock = threading.Lock()
        self.calls_by_model: dict[str, int] = {}
        self._last = _LastCall()

    @property
    def last_call(self) -> tuple[str | None, bool]:
        return self._last.get()

    def _create(self, **kwargs: Any) -> Any:
        anthropic = self._anthropic
        self._last.set(None)
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("output_config", {})
        kwargs["output_config"].setdefault("effort", self.effort)
        try:
            if self._use_fallbacks:
                msg = self._client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                msg = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic rejected the API key (HTTP 401). Check ANTHROPIC_API_KEY in .env, then restart the app."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"This Anthropic key is not allowed to use model '{kwargs['model']}' (HTTP 403).") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(
                f"Anthropic model '{kwargs['model']}' was not found for this key (HTTP 404). "
                "Set ANTHROPIC_MODEL in .env to a model your account can use, then restart."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached after retries; wait a minute and try again.") from exc
        except anthropic.BadRequestError as exc:
            detail = _safe_detail(getattr(exc, "message", "") or "bad request")
            raise LLMError(f"Anthropic rejected the request (HTTP 400): {detail}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 529:
                raise LLMError("Anthropic is overloaded right now (HTTP 529); try again shortly.") from exc
            raise LLMError(f"Anthropic API error (HTTP {exc.status_code}).") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("The Anthropic API did not answer in time (timeout); try again.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API (network error). Check your internet connection.") from exc
        served_by = getattr(msg, "model", None)
        served_by = served_by if isinstance(served_by, str) and served_by else kwargs["model"]
        with self._stats_lock:
            self.calls_by_model[served_by] = self.calls_by_model.get(served_by, 0) + 1
        self._last.set(served_by)
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


_GEMINI_NEW_KEY = "create a Gemini key at https://aistudio.google.com/apikey and put it in GEMINI_API_KEY in .env"
_GEMINI_KEY_REASONS = {
    "API_KEY_SERVICE_BLOCKED": "this key is not allowed to call the Gemini API (its API restrictions block it); "
    + _GEMINI_NEW_KEY
    + ", or allow 'Generative Language API' for the key in Google Cloud Console",
    "API_KEY_INVALID": "the key is not valid; " + _GEMINI_NEW_KEY,
    "SERVICE_DISABLED": "the Generative Language API is disabled for this key's project; " + _GEMINI_NEW_KEY,
}


def _gemini_error(resp: Any, key: str = "") -> _GeminiError:
    """Pull the first line of the API message, ``RetryInfo.retryDelay`` and whether a per-day quota was hit.

    Tolerates bodies that do not follow Google's error shape (e.g. from a proxy): anything unexpected
    simply leaves the defaults, so the caller still raises a normal :class:`LLMError`.
    """
    info = _GeminiError(resp.status_code)
    try:
        body = resp.json()
    except ValueError:
        return info
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return info
    message = str(err.get("message") or "")
    if key:
        message = message.replace(key, "<key>")  # before truncating, so a key at the cut is never half-kept
    message = message.strip()
    if message:
        info.message = message.splitlines()[0][:200]
    details = err.get("details")
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        kind = str(detail.get("@type", ""))
        if kind.endswith("ErrorInfo") and detail.get("reason") in _GEMINI_KEY_REASONS:
            info.message = _GEMINI_KEY_REASONS[str(detail.get("reason"))]
        elif kind.endswith("RetryInfo"):
            info.retry_after = _duration_seconds(str(detail.get("retryDelay", "")))
        elif kind.endswith("QuotaFailure"):
            raw = detail.get("violations")
            violations = [v for v in raw if isinstance(v, dict)] if isinstance(raw, list) else []
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
    a model is overloaded, unavailable or out of daily quota; :attr:`last_call` names the model that answered
    (see :class:`UsageTracker`) and :attr:`calls_by_model` totals live calls per model for this client.
    """

    provider = "gemini"
    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self, model: str, fallback_models: Sequence[str] = (), max_rpm: float = 0.0, max_retries: int = 5, timeout_s: float = 180.0
    ) -> None:
        import httpx

        self._timeout = timeout_s
        self._httpx = httpx
        self.model = model
        self._models = [model, *[m for m in fallback_models if m and m != model]]
        self._key = os.environ.get("GEMINI_API_KEY", "")
        self._rpm = max_rpm
        self._max_retries = max(0, max_retries)
        self._sleep = time.sleep
        self._stats_lock = threading.Lock()
        self.calls_by_model: dict[str, int] = {}
        self._last = _LastCall()

    @property
    def last_call(self) -> tuple[str | None, bool]:
        return self._last.get()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(60.0, 2.0 * 2**attempt) * (0.75 + random.random() / 2)

    @staticmethod
    def _body_for(model: str, body: dict[str, Any], temperature: float | None) -> dict[str, Any]:
        # Gemini 3+ is tuned for the default temperature (lower values can cause looping), so only older
        # models get a fixed one. Decided per model because fallbacks may be a different generation.
        if temperature is None or (_gemini_major_version(model) or 3) >= 3:
            return body
        return {**body, "generationConfig": {**body.get("generationConfig", {}), "temperature": temperature}}

    def _post(
        self, body: dict[str, Any], retries: int | None = None, fallbacks: bool = True, temperature: float | None = None
    ) -> dict[str, Any]:
        problem = "Gemini API error."
        self._last.set(None)
        max_retries = self._max_retries if retries is None else min(retries, self._max_retries)
        models = self._models if fallbacks else self._models[:1]
        for model in models:
            url = f"{self._BASE}/{model}:generateContent"
            model_body = self._body_for(model, body, temperature)
            for attempt in range(max_retries + 1):
                _pacer_for(model, self._rpm).wait(self._sleep)
                try:
                    resp = self._httpx.post(url, json=model_body, headers={"x-goog-api-key": self._key}, timeout=self._timeout)
                except self._httpx.HTTPError:
                    problem = "Could not reach the Gemini API (network error)."
                    delay = self._backoff(attempt)
                else:
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except ValueError as exc:
                            raise LLMError("Gemini returned a response that is not JSON.") from exc
                        with self._stats_lock:
                            self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
                        self._last.set(model)
                        return data
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

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": self._parts(parts)}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _schema_for_gemini(schema),
                "maxOutputTokens": max(max_tokens, 16000),  # thinking tokens count against this limit
            },
        }
        return _parse_json_text(self._text(self._post(body, temperature=0)))

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max(max_tokens, 8000)},
        }
        return self._text(self._post(body, temperature=0.2)).strip()

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
    """OpenAI chat completions via REST (strict JSON schema). No built-in web search here.

    ``OPENAI_BASE_URL`` points it at an OpenAI-compatible gateway (for example AI Pipe:
    ``https://aipipe.org/openai/v1``); the key in ``OPENAI_API_KEY`` is sent as a bearer token.
    """

    provider = "openai"
    _DEFAULT_BASE = "https://api.openai.com/v1"

    _RETRY_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(self, model: str, timeout_s: float = 180.0, max_retries: int = 2) -> None:
        import httpx

        self._httpx = httpx
        self._timeout = timeout_s
        self._max_retries = max(0, min(max_retries, 5))
        self._sleep = time.sleep
        self.model = model
        base = os.environ.get("OPENAI_BASE_URL", "").strip() or self._DEFAULT_BASE
        self._URL = base.rstrip("/") + "/chat/completions"
        self._key = os.environ.get("OPENAI_API_KEY", "")
        self._stats_lock = threading.Lock()
        self.calls_by_model: dict[str, int] = {}
        self._last = _LastCall()

    @property
    def last_call(self) -> tuple[str | None, bool]:
        return self._last.get()

    @staticmethod
    def _status_message(status: int, model: str) -> str:
        if status == 401:
            return "The OpenAI-compatible API rejected the key (HTTP 401). Check OPENAI_API_KEY in .env."
        if status == 403:
            return f"This key may not use model '{model}' (HTTP 403)."
        if status == 404:
            return f"Model '{model}' or the endpoint was not found (HTTP 404). Check OPENAI_MODEL and OPENAI_BASE_URL."
        if status == 429:
            return "OpenAI-compatible API rate limit or quota reached after retries (HTTP 429); try again later."
        return f"OpenAI API error (HTTP {status})."

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        self._last.set(None)
        timeout_error = getattr(self._httpx, "TimeoutException", ())
        for attempt in range(self._max_retries + 1):
            last_attempt = attempt == self._max_retries
            try:
                resp = self._httpx.post(
                    self._URL, json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=self._timeout
                )
            except timeout_error as exc:
                if last_attempt:
                    raise LLMError("The OpenAI-compatible API did not answer in time (timeout); try again.") from exc
            except self._httpx.HTTPError as exc:
                if last_attempt:
                    raise LLMError("Could not reach the OpenAI-compatible API (network error).") from exc
            else:
                if resp.status_code == 200:
                    break
                if resp.status_code not in self._RETRY_STATUS or last_attempt:
                    raise LLMError(self._status_message(resp.status_code, self.model))
            self._sleep(min(30.0, 2.0 * 2**attempt))
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError("The OpenAI-compatible API returned a response that is not JSON.") from exc
        served_by = data.get("model") if isinstance(data, dict) else None
        served_by = served_by if isinstance(served_by, str) and served_by else self.model
        with self._stats_lock:
            self.calls_by_model[served_by] = self.calls_by_model.get(served_by, 0) + 1
        self._last.set(served_by)
        return data

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
        message = data["choices"][0]["message"]
        # Reasoning models on some gateways put the answer in reasoning_content and leave content empty.
        return _parse_json_text(message.get("content") or message.get("reasoning_content") or "")

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
        return AnthropicClient(
            settings.llm_model, settings.llm_effort, timeout_s=settings.llm_timeout_s, max_retries=min(settings.llm_max_retries, 3)
        )
    if settings.llm_provider == "gemini":
        return GeminiClient(
            settings.llm_model,
            fallback_models=settings.llm_fallback_models,
            max_rpm=settings.llm_max_rpm,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
        )
    return OpenAIClient(settings.llm_model, timeout_s=settings.llm_timeout_s, max_retries=min(settings.llm_max_retries, 3))


# --------------------------------------------------------------------------- response cache


_CACHE_FORMAT = 2
MODEL_NOT_RECORDED = "model not recorded"


class CachedLLM:
    """Disk cache for ``complete_json`` so re-runs (tests, UI demos) do not re-bill identical requests.

    Each entry stores the model that answered (a fallback model may answer under the main model's key), so
    reused responses stay attributable. Entries written before that was recorded replay with no model.
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
        self._last = _LastCall()

    @property
    def last_call(self) -> tuple[str | None, bool]:
        return self._last.get()

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
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict) and stored.get("_cache_format") == _CACHE_FORMAT:
                self._last.set(stored.get("model"), cached=True)
                return stored["value"]
            self._last.set(None, cached=True)  # legacy entry: the bare value, model unknown
            return stored
        value = self._inner.complete_json(system=system, parts=parts, schema=schema, max_tokens=max_tokens)
        model, _ = _last_call(self._inner)
        path.write_text(json.dumps({"_cache_format": _CACHE_FORMAT, "model": model, "value": value}), encoding="utf-8")
        self._last.set(model)
        return value

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        text = self._inner.complete_text(system=system, messages=messages, max_tokens=max_tokens)
        self._last.set(_last_call(self._inner)[0])
        return text

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer:
        answer = self._inner.web_search(query, max_uses=max_uses)
        self._last.set(_last_call(self._inner)[0])
        return answer


class UsageTracker:
    """Wraps a client for one batch or session and counts which model answered each call.

    Live calls and cached responses are counted separately, by the model that produced them, plus calls that
    failed. The counts live on this wrapper, so wrapping the shared Streamlit client per batch keeps other
    sessions' calls out of a batch's numbers.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self._lock = threading.Lock()
        self._live: dict[str, int] = {}
        self._cached: dict[str, int] = {}
        self._failed = 0

    def _run(self, call: Any) -> Any:
        try:
            result = call()
        except LLMError:
            with self._lock:
                self._failed += 1
            raise
        model, cached = _last_call(self._inner)
        counts = self._cached if cached else self._live
        name = model or MODEL_NOT_RECORDED
        with self._lock:
            counts[name] = counts.get(name, 0) + 1
        return result

    def complete_json(self, *, system: str, parts: Sequence[Part], schema: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
        return self._run(lambda: self._inner.complete_json(system=system, parts=parts, schema=schema, max_tokens=max_tokens))

    def complete_text(self, *, system: str, messages: Sequence[dict[str, str]], max_tokens: int = 2000) -> str:
        return self._run(lambda: self._inner.complete_text(system=system, messages=messages, max_tokens=max_tokens))

    def web_search(self, query: str, *, max_uses: int = 3) -> SearchAnswer:
        return self._run(lambda: self._inner.web_search(query, max_uses=max_uses))

    def summary(self) -> dict[str, Any]:
        """``live_calls_by_model``, ``cached_responses_by_model`` and ``failed_calls`` for the outputs."""
        with self._lock:
            return {
                "live_calls_by_model": dict(self._live),
                "cached_responses_by_model": dict(self._cached),
                "failed_calls": self._failed,
            }


def get_cached_llm_client(settings: Settings) -> LLMClient:
    """Configured client wrapped in the disk cache unless ``LLM_CACHE=off``."""
    from .config import PROJECT_ROOT

    client = get_llm_client(settings)
    if os.getenv("LLM_CACHE", "on").strip().lower() == "off":
        return client
    return CachedLLM(client, str(PROJECT_ROOT / ".cache" / "llm"))
