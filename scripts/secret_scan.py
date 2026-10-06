"""Scan the files Git tracks (or would track) for secrets and for artefacts that must not be committed.

Usage:  python scripts/secret_scan.py        (exit code 1 when anything is found)

Run it after `git add` and before every commit/push. It prints file names and the kind of finding,
never the matched text.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Tavily API key": re.compile(r"tvly-[A-Za-z0-9_\-]{20,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "hard-coded credential": re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][^'\"\s]{16,}['\"]"),
}

FORBIDDEN_PATHS = [
    (re.compile(r"(^|/)\.env$|\.env$"), "environment file"),
    (re.compile(r"(^|/)\.cache/"), "cache (may hold extracted values)"),
    (re.compile(r"(^|/)(crops|debug)/"), "temporary crops/debug output"),
    (re.compile(r"\.log$"), "log file"),
    (re.compile(r"^data/documents/(?!README\.md$)"), "supplied identity document"),
]
ALLOWED_ENV_FILES = {".env.example"}


def tracked_files() -> list[str]:
    """Files in the index plus untracked files that are not ignored (i.e. what `git add -A` would commit)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return sorted(set(out.stdout.splitlines()))


def main() -> int:
    findings: list[str] = []
    files = tracked_files()
    for rel in files:
        if Path(rel).name not in ALLOWED_ENV_FILES:
            for pattern, kind in FORBIDDEN_PATHS:
                if pattern.search(rel):
                    findings.append(f"{rel}: {kind} must not be committed")
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary file (images, workbook)
        for kind, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{rel}: possible {kind}")
    print(f"Scanned {len(files)} files.")
    for f in findings:
        print(f"  FOUND {f}")
    if findings:
        return 1
    print("No secrets or forbidden artefacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
