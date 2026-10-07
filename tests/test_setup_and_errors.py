"""Environment loading, provider/model configuration, safe provider errors and OCR discovery (all offline)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import pytest

from src.common import config
from src.common.config import get_settings, load_dotenv_if_present
from src.common.llm import AnthropicClient, LLMError, LLMNotConfigured, OpenAIClient, TextPart, _safe_detail, get_llm_client

FAKE_KEY = "sk-ant-" + "test-" + "0" * 16  # shape only, never a real key; built up so the secret scan stays clean

_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_MODEL",
    "GEMINI_MODEL",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "LLM_TIMEOUT_SECONDS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """No provider variables from the real shell or the real project `.env`."""
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv_if_present", lambda: False)
    return monkeypatch


# --------------------------------------------------------------------------- .env loading


def test_dotenv_is_loaded_from_the_project_root_not_the_working_directory(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\nANTHROPIC_MODEL=claude-sonnet-5-5\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    other = tmp_path / "somewhere_else"
    other.mkdir()
    monkeypatch.chdir(other)
    assert load_dotenv_if_present(env) is True
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == FAKE_KEY
    assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet-5-5"


def test_default_env_path_is_the_project_root():
    assert config.ENV_PATH == config.PROJECT_ROOT / ".env"
    assert (config.PROJECT_ROOT / "app.py").is_file()


def test_empty_shell_variable_does_not_hide_the_dotenv_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    load_dotenv_if_present(env)
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == FAKE_KEY


def test_real_environment_value_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_MODEL", "from-shell")
    load_dotenv_if_present(env)
    import os

    assert os.environ["ANTHROPIC_MODEL"] == "from-shell"


def test_missing_dotenv_is_reported_not_raised(tmp_path):
    assert load_dotenv_if_present(tmp_path / "absent.env") is False


def test_env_files_are_git_ignored():
    ignore = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore and "!.env.example" in ignore


def test_env_example_holds_placeholders_only():
    example = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if line.split("=", 1)[0] in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
            assert line.split("=", 1)[1].strip() == "", line.split("=", 1)[0]
    assert "ANTHROPIC_MODEL=" in example


# --------------------------------------------------------------------------- keys and models


def test_missing_key_message_names_the_exact_variable_and_file(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    settings = get_settings()
    assert not settings.llm_available
    with pytest.raises(LLMNotConfigured) as info:
        get_llm_client(settings)
    assert "ANTHROPIC_API_KEY=<your key>" in str(info.value)
    assert ".env" in str(info.value)


def test_key_is_detected_and_never_part_of_settings_text(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    settings = get_settings()
    assert settings.llm_available
    assert FAKE_KEY not in repr(settings) and FAKE_KEY not in settings.setup_hint()


def test_key_for_another_provider_triggers_a_warning(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "AIza" + "NotAnAnthropicKey" + "0" * 18)
    settings = get_settings()
    assert any("sk-ant-" in w for w in settings.config_warnings)
    assert not any("AIza" in w for w in settings.config_warnings)


def test_anthropic_model_variable_wins(clean_env):
    assert get_settings().llm_model == "claude-opus-5-5"
    assert get_settings().model_source == "default"
    clean_env.setenv("LLM_MODEL", "claude-sonnet-5-5")
    assert (get_settings().llm_model, get_settings().model_source) == ("claude-sonnet-5-5", "LLM_MODEL")
    clean_env.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    assert (get_settings().llm_model, get_settings().model_source) == ("claude-opus-5", "ANTHROPIC_MODEL")


def test_provider_specific_model_variable_is_per_provider(clean_env):
    clean_env.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    assert get_settings().llm_model == "gpt-4.1-mini"


def test_default_anthropic_model_is_known_to_the_installed_sdk(clean_env):
    known = config._known_anthropic_models()
    assert known, "anthropic SDK model list unavailable"
    assert config.DEFAULT_MODELS["anthropic"] in known
    assert not get_settings().config_warnings


def test_unknown_anthropic_model_gets_a_warning(clean_env):
    clean_env.setenv("ANTHROPIC_MODEL", "claude-does-not-exist")
    assert any("ANTHROPIC_MODEL" in w for w in get_settings().config_warnings)


def test_timeout_setting(clean_env):
    assert get_settings().llm_timeout_s == 180.0
    clean_env.setenv("LLM_TIMEOUT_SECONDS", "45")
    assert get_settings().llm_timeout_s == 45.0
    clean_env.setenv("LLM_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError):
        get_settings()


def test_anthropic_client_uses_only_the_anthropic_key(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = get_llm_client(get_settings())
    assert isinstance(client, AnthropicClient)
    assert client._client.api_key == FAKE_KEY
    assert client._client.timeout == 180.0
    assert client._client.max_retries == 3


# --------------------------------------------------------------------------- provider errors


def _anthropic_with_error(monkeypatch, exc):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = AnthropicClient("claude-opus-5-5")

    def create(**kwargs):
        raise exc

    client._client = NS(beta=NS(messages=NS(create=create)), messages=NS(create=create))
    return client


def _status_error(cls, status):
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, json={"error": {"message": "boom"}})
    assert issubclass(cls, anthropic.APIStatusError)
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    ("name", "status", "expected"),
    [
        ("AuthenticationError", 401, "rejected the API key"),
        ("PermissionDeniedError", 403, "not allowed"),
        ("NotFoundError", 404, "ANTHROPIC_MODEL"),
        ("RateLimitError", 429, "rate limit"),
        ("InternalServerError", 529, "overloaded"),
        ("InternalServerError", 500, "HTTP 500"),
    ],
)
def test_anthropic_status_errors_are_friendly(monkeypatch, name, status, expected):
    import anthropic

    client = _anthropic_with_error(monkeypatch, _status_error(getattr(anthropic, name), status))
    with pytest.raises(LLMError, match=expected) as info:
        client.complete_text(system="s", messages=[{"role": "user", "content": "hi"}])
    assert FAKE_KEY not in str(info.value)


def test_anthropic_timeout_and_network_errors(monkeypatch):
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = _anthropic_with_error(monkeypatch, anthropic.APITimeoutError(request=request))
    with pytest.raises(LLMError, match="timeout"):
        client.complete_json(system="s", parts=[TextPart("x")], schema={})
    client = _anthropic_with_error(monkeypatch, anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError, match="network"):
        client.complete_json(system="s", parts=[TextPart("x")], schema={})


def test_safe_detail_strips_key_shaped_text():
    out = _safe_detail(f"invalid header {FAKE_KEY}\nsecond line with content")
    assert FAKE_KEY not in out and "second line" not in out and "<redacted>" in out


def _openai(monkeypatch, responses):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = OpenAIClient("gpt-4.1", max_retries=2)
    queue, sleeps = list(responses), []

    def post(*args, **kwargs):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client._httpx = NS(post=post, HTTPError=httpx.HTTPError, TimeoutException=httpx.TimeoutException)
    client._sleep = sleeps.append
    return client, sleeps


def _ok():
    return NS(status_code=200, json=lambda: {"model": "gpt-4.1", "choices": [{"message": {"content": "fine"}}]})


def test_openai_retries_rate_limit_then_succeeds(monkeypatch):
    client, sleeps = _openai(monkeypatch, [NS(status_code=429), NS(status_code=503), _ok()])
    assert client.complete_text(system="s", messages=[{"role": "user", "content": "q"}]) == "fine"
    assert len(sleeps) == 2


def test_openai_does_not_retry_auth_errors(monkeypatch):
    client, sleeps = _openai(monkeypatch, [NS(status_code=401)])
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        client.complete_text(system="s", messages=[{"role": "user", "content": "q"}])
    assert sleeps == []


def test_openai_timeout_after_retries(monkeypatch):
    timeouts = [httpx.ReadTimeout("slow") for _ in range(3)]
    client, sleeps = _openai(monkeypatch, timeouts)
    with pytest.raises(LLMError, match="timeout"):
        client.complete_text(system="s", messages=[{"role": "user", "content": "q"}])
    assert len(sleeps) == 2


# --------------------------------------------------------------------------- OCR discovery


def test_tesseract_cmd_override_and_fallback(monkeypatch, tmp_path):
    from src.question3_documents import ocr

    fake = tmp_path / "tesseract.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("TESSERACT_CMD", str(fake))
    assert ocr.find_tesseract() == str(fake)
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing.exe"))
    assert ocr.find_tesseract() is None
    monkeypatch.delenv("TESSERACT_CMD")
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr.os, "name", "nt")
    monkeypatch.setattr(ocr, "_WINDOWS_TESSERACT_PATHS", (str(fake),))
    assert ocr.find_tesseract() == str(fake)


def test_pipeline_runs_without_tesseract(monkeypatch):
    """OCR fallback: with Tesseract missing, OCR returns empty results instead of failing."""
    from PIL import Image

    from src.question3_documents import ocr

    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    page = ocr.ocr_page(Image.new("RGB", (200, 100), "white"))
    assert page.words == [] and page.text == ""


def test_gemini_blocked_key_gets_actionable_message():
    from src.common.llm import _gemini_error

    body = {
        "error": {
            "code": 403,
            "message": "Requests to this API generativelanguage.googleapis.com method ... are blocked.",
            "status": "PERMISSION_DENIED",
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_SERVICE_BLOCKED"}],
        }
    }
    info = _gemini_error(NS(status_code=403, json=lambda: body))
    assert "aistudio.google.com/apikey" in info.message and "GEMINI_API_KEY" in info.message
