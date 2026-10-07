"""Central configuration loaded from environment variables (and an optional .env file).

No secret is ever logged or rendered; only whether a key is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_PROVIDERS = ("anthropic", "gemini", "openai")

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-4.1",
}

_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Provider-specific model variables win over the generic LLM_MODEL.
_MODEL_VARS = {
    "anthropic": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
}

# Cheap shape checks that catch a key pasted into the wrong line; they never reveal the key itself.
_KEY_PREFIXES = {"anthropic": "sk-ant-"}

ENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv_if_present(env_path: Path | None = None) -> bool:
    """Load `.env` from the project root (not the working directory); return whether it was found.

    A variable already set to a non-empty value in the real environment wins. A variable that is set but
    empty (for example ``$env:ANTHROPIC_API_KEY = ""`` left over in a shell) is filled from `.env`, because an
    empty value would otherwise hide the key and the app would report it as missing.
    """
    env_path = env_path or ENV_PATH
    if not env_path.is_file():
        return False
    try:
        from dotenv import dotenv_values
    except ImportError:  # python-dotenv is optional
        return False
    for name, value in dotenv_values(env_path).items():
        if value is not None and not os.environ.get(name, "").strip():
            os.environ[name] = value.strip()
    return True


def _known_anthropic_models() -> frozenset[str]:
    """Model IDs the installed ``anthropic`` SDK knows about (empty when the SDK is missing)."""
    try:
        from typing import get_args

        from anthropic.types import Model
    except ImportError:
        return frozenset()
    names: set[str] = set()
    stack = list(get_args(Model))
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            names.add(item)
        else:
            stack.extend(get_args(item))
    return frozenset(names)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Construct with :func:`get_settings`."""

    llm_provider: str = "anthropic"
    llm_model: str = DEFAULT_MODELS["anthropic"]
    llm_effort: str = "medium"
    search_provider: str = "auto"
    review_threshold: float = 0.85
    exec_timeout_s: float = 10.0
    max_output_chars: int = 20_000
    pdf_dpi: int = 200
    llm_fallback_models: tuple[str, ...] = ()
    llm_max_rpm: float = 0.0
    llm_max_retries: int = 5
    crop_reads: str = "per_field"
    llm_timeout_s: float = 180.0
    model_source: str = "default"
    dotenv_found: bool = False
    api_keys_present: dict[str, bool] = field(default_factory=dict)
    config_warnings: tuple[str, ...] = ()

    @property
    def llm_key_var(self) -> str:
        return _KEY_VARS[self.llm_provider]

    @property
    def llm_available(self) -> bool:
        return self.api_keys_present.get(self.llm_provider, False)

    @property
    def tavily_available(self) -> bool:
        return self.api_keys_present.get("tavily", False)

    @property
    def llm_model_var(self) -> str:
        return _MODEL_VARS[self.llm_provider]

    def setup_hint(self) -> str:
        """Human-readable guidance shown when the vision/LLM key is missing."""
        where = f"`{ENV_PATH}`" if self.dotenv_found else f"a new `.env` at `{ENV_PATH}` (copy `.env.example`)"
        return (
            f"No API key found for LLM provider '{self.llm_provider}'. "
            f"Add the line `{self.llm_key_var}=<your key>` to {where} "
            "(or export it in your shell), then restart the app. "
            "Set `LLM_PROVIDER` to one of: " + ", ".join(SUPPORTED_PROVIDERS) + "."
        )


def get_settings() -> Settings:
    """Read settings from the environment (after loading `.env` if present)."""
    dotenv_found = bool(load_dotenv_if_present())
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {SUPPORTED_PROVIDERS}, got {provider!r}")
    model_var = _MODEL_VARS[provider]
    if os.getenv(model_var, "").strip():
        model, model_source = os.environ[model_var].strip(), model_var
    elif os.getenv("LLM_MODEL", "").strip():
        model, model_source = os.environ["LLM_MODEL"].strip(), "LLM_MODEL"
    else:
        model, model_source = DEFAULT_MODELS[provider], "default"
    threshold = _float_env("REVIEW_THRESHOLD", 0.85)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("REVIEW_THRESHOLD must be in (0, 1].")
    crop_reads = os.getenv("CROP_READS", "per_field").strip().lower() or "per_field"
    if crop_reads not in ("per_field", "batched"):
        raise ValueError(f"CROP_READS must be 'per_field' or 'batched', got {crop_reads!r}")
    keys = {name: bool(os.getenv(var, "").strip()) for name, var in _KEY_VARS.items()}
    keys["tavily"] = bool(os.getenv("TAVILY_API_KEY", "").strip())
    warnings: list[str] = []
    key = os.getenv(_KEY_VARS[provider], "").strip()
    prefix = _KEY_PREFIXES.get(provider)
    if key and prefix and not key.startswith(prefix):
        warnings.append(f"{_KEY_VARS[provider]} does not start with '{prefix}', so it may be a key for another provider.")
    if provider == "anthropic":
        known = _known_anthropic_models()
        if known and model not in known:
            warnings.append(
                f"Model '{model}' is not in the installed anthropic SDK's model list; "
                f"if requests fail with 'model not found', set {model_var} to a supported ID."
            )
    timeout = _float_env("LLM_TIMEOUT_SECONDS", 180.0)
    if timeout <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be positive.")
    return Settings(
        llm_provider=provider,
        llm_model=model,
        llm_effort=os.getenv("LLM_EFFORT", "medium").strip() or "medium",
        search_provider=os.getenv("SEARCH_PROVIDER", "auto").strip().lower() or "auto",
        review_threshold=threshold,
        exec_timeout_s=_float_env("EXEC_TIMEOUT_SECONDS", 10.0),
        max_output_chars=int(_float_env("MAX_OUTPUT_CHARS", 20_000)),
        pdf_dpi=int(_float_env("PDF_DPI", 200)),
        llm_fallback_models=tuple(m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()),
        llm_max_rpm=_float_env("LLM_MAX_RPM", 0.0),
        llm_max_retries=int(_float_env("LLM_MAX_RETRIES", 5)),
        crop_reads=crop_reads,
        llm_timeout_s=timeout,
        model_source=model_source,
        dotenv_found=dotenv_found,
        api_keys_present=keys,
        config_warnings=tuple(warnings),
    )
