"""Batch-process the supplied documents and write every Question 3 artefact to outputs/question3/.

Usage:  python scripts/run_question3.py [--input data/documents] [--out outputs/question3]
Needs an LLM API key (see .env.example). Responses are cached in .cache/llm (git-ignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import get_settings  # noqa: E402
from src.common.llm import LLMNotConfigured, get_cached_llm_client  # noqa: E402
from src.question3_documents.pipeline import process_files  # noqa: E402
from src.question3_documents.report import summary_rows, write_outputs  # noqa: E402

SUPPORTED = {".png", ".jpg", ".jpeg", ".pdf"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "documents"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "question3"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    try:
        llm = get_cached_llm_client(settings)
    except LLMNotConfigured as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    in_dir = Path(args.input)
    files = [(p.name, p.read_bytes()) for p in sorted(in_dir.glob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"ERROR: no PNG/JPG/PDF files in {in_dir}. Copy the supplied documents there first.", file=sys.stderr)
        return 2
    print(f"Processing {len(files)} files with {settings.llm_provider}:{settings.llm_model} (threshold {settings.review_threshold})")
    start = time.time()
    batch = process_files(files, llm, settings.review_threshold, settings.pdf_dpi, progress=lambda m, f: print(f"  [{f:4.0%}] {m}"))
    paths = write_outputs(batch.documents, args.out, settings.review_threshold, batch.file_errors)
    print(f"Done in {time.time() - start:.0f}s: {len(batch.documents)} logical documents, {len(batch.file_errors)} file errors")
    for row in summary_rows(batch.documents):
        print(
            f"  {row['source_file'][:45]:45} p{row['pages']:5} {row['document_type']:34} cls={row['classification_confidence']:.2f} "
            f"accepted={row['fields_auto_accepted']}/{row['fields_total']}"
        )
    for name, path in paths.items():
        print(f"  wrote {name}: {path.relative_to(ROOT)}")
    for name, err in batch.file_errors.items():
        print(f"  ERROR {name}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
