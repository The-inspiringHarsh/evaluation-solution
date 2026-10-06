"""Masking, PII scrubbing and configuration handling."""

import pytest

from src.common.config import get_settings
from src.common.llm import LLMNotConfigured, _schema_for_gemini, get_llm_client
from src.common.privacy import contains_pii, mask_field, mask_numbers_in_text, mask_value, scrub_pii


def test_mask_value_keeps_last_four_and_separators():
    assert mask_value("1234 5678 9012") == "•••• •••• 9012"
    assert mask_value("SBIN0227112") == "•••••••7112"
    assert mask_value("abc") == "•••"


def test_mask_field_only_sensitive():
    assert mask_field("ifsc_code", "SBIN0227112").endswith("7112")
    assert mask_field("place", "West Bihar") == "West Bihar"


def test_mask_numbers_in_text_hides_identifiers_but_keeps_prose():
    reason = "independent reads disagree: vision_crop='HDFC0001234' | vision_full_page='HDFCO001234'"
    masked = mask_numbers_in_text(reason)
    assert "HDFC0001234" not in masked and "HDFCO001234" not in masked
    assert masked.startswith("independent reads disagree: vision_crop='")
    assert "1234'" in masked  # last four characters stay visible for the reviewer
    assert "0012345678" not in mask_numbers_in_text("account 0012345678 read twice")
    assert "ABCDE1234F" not in mask_numbers_in_text("PAN ABCDE1234F")
    assert "2345 6789 0123" not in mask_numbers_in_text("Aadhaar 2345 6789 0123")
    keep = "confidence 0.72 below threshold 0.85; date 12/03/1985; amount 5,000; place Mumbai"
    assert mask_numbers_in_text(keep) == keep
    assert mask_numbers_in_text(None) == ""


def test_scrub_pii():
    text = "PAN ABCDE1234F account 31004258912 aadhaar 1234 5678 9012 mail a@b.com"
    out = scrub_pii(text)
    assert not contains_pii(out)
    assert "ABCDE1234F" not in out and "31004258912" not in out


def test_missing_key_gives_setup_hint(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr("src.common.config.load_dotenv_if_present", lambda: None)
    settings = get_settings()
    assert not settings.llm_available
    with pytest.raises(LLMNotConfigured, match="ANTHROPIC_API_KEY"):
        get_llm_client(settings)


def test_invalid_settings(monkeypatch):
    monkeypatch.setattr("src.common.config.load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        get_settings()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("REVIEW_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        get_settings()


def test_threshold_from_env(monkeypatch):
    monkeypatch.setattr("src.common.config.load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.9")
    assert get_settings().review_threshold == 0.9


def test_gemini_schema_conversion():
    out = _schema_for_gemini({"type": "object", "additionalProperties": False, "properties": {"v": {"type": ["string", "null"]}}})
    assert "additionalProperties" not in out
    assert out["properties"]["v"] == {"type": "string", "nullable": True}
