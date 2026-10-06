"""Serialisation of results: per-document JSON, combined JSON, flagging report and summary CSV."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import DocumentResult

FLAG_COLUMNS = [
    "source_file",
    "document_id",
    "page",
    "predicted_document_type",
    "field_name",
    "raw_candidate",
    "confidence",
    "threshold",
    "review_reason",
    "suggested_reviewer_action",
]
SUMMARY_COLUMNS = [
    "document_id",
    "source_file",
    "pages",
    "document_type",
    "classification_confidence",
    "fields_total",
    "fields_auto_accepted",
    "fields_for_review",
    "fields_missing",
    "mean_field_confidence",
    "document_review_required",
]


def document_json(doc: DocumentResult) -> dict[str, Any]:
    """The documented JSON shape (extra diagnostic keys kept under each field)."""
    return json.loads(doc.model_dump_json())


def suggested_action(reason: str, field_name: str) -> str:
    r = reason.lower()
    if "unsupported" in r or "not one of the ten" in r:
        return "Confirm the document type; route to the correct workflow or reject it."
    if "not found" in r or "missing" in r or "not extracted" in r:
        return "Locate the field on the page and key it in manually, or request a clearer scan."
    if "disagree" in r:
        return "Compare the listed candidates against the highlighted region and enter the correct value."
    if "mrz" in r:
        return "Re-read the MRZ and visual zone; a mismatch can indicate a misread or an altered document."
    if "validation" in r or "checksum" in r:
        return "Check the value against the image; if read correctly, the source document itself fails the format rule."
    if "normalis" in r or "ambiguous" in r:
        return "Confirm the ambiguous characters (0/O, 1/I/L, 5/S, 8/B) against the image."
    if "classifier" in r or "classification" in r:
        return "Confirm the predicted document type."
    return f"Visually verify '{field_name}' against the source image and approve or correct it."


def flag_rows(documents: Iterable[DocumentResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in documents:
        dtype = doc.classification.document_type.value
        if doc.classification.review_required:
            reasons = [w for w in doc.warnings if "classifier" in w or "unsupported" in w or "not one of" in w]
            reason = "; ".join(reasons) or f"classification confidence {doc.classification.confidence:.2f} below threshold"
            rows.append(
                {
                    "source_file": doc.source_file,
                    "document_id": doc.document_id,
                    "page": ",".join(str(p) for p in doc.pages),
                    "predicted_document_type": dtype,
                    "field_name": "(document classification)",
                    "raw_candidate": doc.classification.evidence,
                    "confidence": doc.classification.confidence,
                    "threshold": doc.threshold,
                    "review_reason": reason,
                    "suggested_reviewer_action": suggested_action(reason, "document type"),
                }
            )
        for name, f in doc.fields.items():
            if not f.review_required:
                continue
            reason = "; ".join(f.review_reasons) or "below threshold"
            candidate = f.raw_value if f.raw_value is not None else (" | ".join(f.candidates) if f.candidates else "")
            rows.append(
                {
                    "source_file": doc.source_file,
                    "document_id": doc.document_id,
                    "page": str(f.page) if f.page else "",
                    "predicted_document_type": dtype,
                    "field_name": name,
                    "raw_candidate": candidate,
                    "confidence": f.confidence,
                    "threshold": doc.threshold,
                    "review_reason": reason,
                    "suggested_reviewer_action": suggested_action(f.review_reasons[0] if f.review_reasons else reason, name),
                }
            )
    return rows


def summary_rows(documents: Iterable[DocumentResult]) -> list[dict[str, Any]]:
    rows = []
    for doc in documents:
        fields = list(doc.fields.values())
        confs = [f.confidence for f in fields]
        rows.append(
            {
                "document_id": doc.document_id,
                "source_file": doc.source_file,
                "pages": ",".join(str(p) for p in doc.pages),
                "document_type": doc.classification.document_type.value,
                "classification_confidence": doc.classification.confidence,
                "fields_total": len(fields),
                "fields_auto_accepted": sum(1 for f in fields if not f.review_required),
                "fields_for_review": sum(1 for f in fields if f.review_required),
                "fields_missing": sum(1 for f in fields if f.raw_value is None and not f.candidates),
                "mean_field_confidence": round(sum(confs) / len(confs), 3) if confs else None,
                "document_review_required": doc.document_review_required,
            }
        )
    return rows


def to_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def batch_json(documents: list[DocumentResult], threshold: float, file_errors: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "threshold_rationale": (
            "0.85: identity and financial fields are high-risk; a false acceptance (wrong account number, IFSC or "
            "TIN) costs far more than a manual check, so anything without strong multi-pass agreement and a passing "
            "validator is sent to a human."
        ),
        "documents": [document_json(d) for d in documents],
        "file_errors": file_errors or {},
    }


def flagging_json(flags: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """The review queue as JSON (same rows as flagging_report.csv)."""
    return {"threshold": threshold, "flag_count": len(flags), "flags": flags}


def write_outputs(
    documents: list[DocumentResult], out_dir: str | Path, threshold: float, file_errors: dict[str, str] | None = None
) -> dict[str, Path]:
    """Write every required Question 3 artefact under ``out_dir``."""
    out = Path(out_dir)
    (out / "json").mkdir(parents=True, exist_ok=True)
    for stale in (out / "json").glob("*.json"):
        stale.unlink()
    paths: dict[str, Path] = {}
    for doc in documents:
        path = out / "json" / f"{doc.document_id}.json"
        path.write_text(json.dumps(document_json(doc), indent=2, ensure_ascii=False), encoding="utf-8")
    flags = flag_rows(documents)
    paths["all_results"] = out / "all_results.json"
    paths["all_results"].write_text(
        json.dumps(batch_json(documents, threshold, file_errors), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["flagging_csv"] = out / "flagging_report.csv"
    paths["flagging_csv"].write_text(to_csv(flags, FLAG_COLUMNS), encoding="utf-8")
    paths["flagging_json"] = out / "flagging_report.json"
    paths["flagging_json"].write_text(json.dumps(flagging_json(flags, threshold), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary_csv"] = out / "extraction_summary.csv"
    paths["summary_csv"].write_text(to_csv(summary_rows(documents), SUMMARY_COLUMNS), encoding="utf-8")
    return paths
