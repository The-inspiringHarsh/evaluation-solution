"""Batch-process the supplied documents and write every Question 3 artefact to outputs/question3/.

Usage:  python scripts/run_question3.py [--input data/documents] [--out outputs/question3]
Needs an LLM API key (see .env.example). Responses are cached in .cache/llm (git-ignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
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
    print(
        f"Processing {len(files)} files with {settings.llm_provider}:{settings.llm_model} "
        f"(threshold {settings.review_threshold}, crop reads {settings.crop_reads})"
    )
    start = time.time()
    batch = process_files(
        files,
        llm,
        settings.review_threshold,
        settings.pdf_dpi,
        progress=lambda m, f: print(f"  [{f:4.0%}] {m}"),
        crop_reads=settings.crop_reads,
    )
    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "fallback_models": list(settings.llm_fallback_models),
        "crop_reads": settings.crop_reads,
        "live_calls_by_model": getattr(llm, "calls_by_model", {}),
        "cached_responses_reused": getattr(llm, "cache_hits", 0),
        "files": len(files),
        "duration_seconds": round(time.time() - start),
    }
    paths = write_outputs(batch.documents, args.out, settings.review_threshold, batch.file_errors, run_info)
    print(f"Done in {time.time() - start:.0f}s: {len(batch.documents)} logical documents, {len(batch.file_errors)} file errors")
    print(f"  live calls by model: {run_info['live_calls_by_model']}; cached responses reused: {run_info['cached_responses_reused']}")
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
