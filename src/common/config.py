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
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-4.1",
}

_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def load_dotenv_if_present() -> None:
    """Load `.env` from the project root without overriding real environment variables."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(env_path, override=False)


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
    api_keys_present: dict[str, bool] = field(default_factory=dict)

    @property
    def llm_key_var(self) -> str:
        return _KEY_VARS[self.llm_provider]

    @property
    def llm_available(self) -> bool:
        return self.api_keys_present.get(self.llm_provider, False)

    @property
    def tavily_available(self) -> bool:
        return self.api_keys_present.get("tavily", False)

    def setup_hint(self) -> str:
        """Human-readable guidance shown when the vision/LLM key is missing."""
        return (
            f"No API key found for LLM provider '{self.llm_provider}'. "
            f"Copy `.env.example` to `.env` and set `{self.llm_key_var}=...` "
            "(or export it in your shell), then restart the app. "
            "Set `LLM_PROVIDER` to one of: " + ", ".join(SUPPORTED_PROVIDERS) + "."
        )


def get_settings() -> Settings:
    """Read settings from the environment (after loading `.env` if present)."""
    load_dotenv_if_present()
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {SUPPORTED_PROVIDERS}, got {provider!r}")
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    threshold = _float_env("REVIEW_THRESHOLD", 0.85)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("REVIEW_THRESHOLD must be in (0, 1].")
    keys = {name: bool(os.getenv(var, "").strip()) for name, var in _KEY_VARS.items()}
    keys["tavily"] = bool(os.getenv("TAVILY_API_KEY", "").strip())
    return Settings(
        llm_provider=provider,
        llm_model=model,
        llm_effort=os.getenv("LLM_EFFORT", "medium").strip() or "medium",
        search_provider=os.getenv("SEARCH_PROVIDER", "auto").strip().lower() or "auto",
        review_threshold=threshold,
        exec_timeout_s=_float_env("EXEC_TIMEOUT_SECONDS", 10.0),
        max_output_chars=int(_float_env("MAX_OUTPUT_CHARS", 20_000)),
        pdf_dpi=int(_float_env("PDF_DPI", 200)),
        api_keys_present=keys,
    )
