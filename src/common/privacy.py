"""Masking and PII-scrubbing helpers used by the UI, logs and the search tool."""

from __future__ import annotations

import re

# Fields whose values are identity or financial numbers and must be masked in the UI.
SENSITIVE_FIELDS = {
    "aadhaar_number",
    "pan_number",
    "dl_number",
    "passport_number",
    "mrz_line_2",
    "bank_account_number",
    "ifsc_code",
    "tin_pan",
    "policy_number",
    "application_number",
}

_PII_PATTERNS = [
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # Aadhaar-like
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE),  # PAN
    re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),  # IFSC
    re.compile(r"\b\d{9,18}\b"),  # account / policy numbers
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # email
    re.compile(r"\+?\d[\d\s-]{8,}\d"),  # phone
]


def mask_value(value: str | None, visible: int = 4) -> str:
    """Mask all but the last ``visible`` alphanumeric characters, keeping separators."""
    if value is None:
        return ""
    text = str(value)
    alnum_positions = [i for i, ch in enumerate(text) if ch.isalnum()]
    if len(alnum_positions) <= visible:
        return "•" * len(text)
    keep = set(alnum_positions[-visible:])
    return "".join(ch if (i in keep or not ch.isalnum()) else "•" for i, ch in enumerate(text))


def mask_field(field_name: str, value: str | None) -> str:
    """Mask a value only when the field is sensitive."""
    if value is None:
        return ""
    return mask_value(value) if field_name in SENSITIVE_FIELDS else str(value)


_AADHAAR_SPACED = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")
_ID_LIKE_RUN = re.compile(r"[A-Za-z0-9]{6,}")


def mask_numbers_in_text(text: str | None) -> str:
    """Mask identifier-like tokens inside free text (review reasons, evidence, hints).

    A token is masked when it is a spaced 12-digit group or a run of 6+ letters/digits that
    contains at least two digits (account, PAN, IFSC, DL, passport, policy numbers, MRZ chunks).
    Words, short numbers, amounts with separators and dates such as 12/03/1985 stay readable.
    """
    if not text:
        return ""
    out = _AADHAAR_SPACED.sub(lambda m: mask_value(m.group()), str(text))
    return _ID_LIKE_RUN.sub(lambda m: mask_value(m.group()) if sum(c.isdigit() for c in m.group()) >= 2 else m.group(), out)


def scrub_pii(text: str) -> str:
    """Remove identity/financial numbers, emails and phones from text sent to external search."""
    out = text
    for pattern in _PII_PATTERNS:
        out = pattern.sub("[redacted]", out)
    return out


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)
