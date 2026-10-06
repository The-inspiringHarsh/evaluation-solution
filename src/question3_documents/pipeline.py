"""Batch pipeline: ingest -> OCR -> classify/group -> extract -> cross-checks -> DocumentResult list."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from ..common.llm import LLMClient, LLMError
from .extract import LogicalDocument, classify_source, extract_document
from .imaging import IngestError, SourceDocument, load_source
from .ocr import PageOCR, keyword_vote, ocr_page, tesseract_available
from .schemas import DOC_SPECS, Classification, DocumentResult, DocumentType, FieldResult
from .validators import mrz_date_to_iso, parse_mrz_line2, validate_mrz2, words_to_number

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class BatchResult:
    documents: list[DocumentResult] = field(default_factory=list)
    file_errors: dict[str, str] = field(default_factory=dict)
    threshold: float = 0.85
    llm_used: bool = False


def document_id(source: SourceDocument, pages: list[int]) -> str:
    return f"{source.sha1[:10]}-p{'-'.join(str(p) for p in pages)}"


def _noop(_msg: str, _frac: float) -> None:
    return None


def process_files(
    files: list[tuple[str, bytes]],
    llm: Optional[LLMClient],
    threshold: float = 0.85,
    pdf_dpi: int = 200,
    progress: ProgressFn = _noop,
    doc_workers: int = 3,
    crop_reads: str = "per_field",
) -> BatchResult:
    """Process a batch of (filename, bytes). Never raises for per-file problems; they are reported."""
    batch = BatchResult(threshold=threshold, llm_used=llm is not None)
    total = max(1, len(files))

    def handle(index_and_file: tuple[int, tuple[str, bytes]]) -> tuple[str, list[DocumentResult], Optional[str]]:
        index, (name, data) = index_and_file
        progress(f"Processing {name}", index / total)
        try:
            return name, process_one(name, data, llm, threshold, pdf_dpi, crop_reads), None
        except IngestError as exc:
            return name, [], str(exc)
        except LLMError as exc:
            return name, [], f"vision model error: {exc}"

    with ThreadPoolExecutor(max_workers=doc_workers) as pool:
        for name, docs, error in pool.map(handle, enumerate(files)):
            if error:
                batch.file_errors[name] = error
            batch.documents.extend(docs)
    add_cross_document_hints(batch.documents)
    progress("Done", 1.0)
    return batch


def process_one(
    name: str, data: bytes, llm: Optional[LLMClient], threshold: float, pdf_dpi: int = 200, crop_reads: str = "per_field"
) -> list[DocumentResult]:
    source = load_source(name, data, pdf_dpi=pdf_dpi)
    page_ocr: dict[int, PageOCR] = {p.page_number: ocr_page(p.processed) for p in source.pages}
    preprocessing = [
        {"page": p.page_number, "rotation_applied": p.rotation_applied, "skew_corrected_deg": p.skew_corrected_deg, **p.quality}
        for p in source.pages
    ]
    if llm is None:
        return [_offline_result(source, page_ocr, threshold, preprocessing)]

    logical = classify_source(llm, source, page_ocr, threshold)
    results = []
    for doc in logical:
        results.append(_build_result(llm, doc, page_ocr, threshold, preprocessing, crop_reads))
    return results


def _classification(doc: LogicalDocument) -> Classification:
    return Classification(
        document_type=doc.doc_type,
        confidence=doc.classification_confidence,
        review_required=doc.classification_review,
        evidence=doc.evidence,
        model_confidence=doc.model_confidence,
        keyword_vote=doc.keyword_type.value if doc.keyword_type else None,
    )


def _build_result(
    llm: LLMClient,
    doc: LogicalDocument,
    page_ocr: dict[int, PageOCR],
    threshold: float,
    preprocessing: list[dict],
    crop_reads: str = "per_field",
) -> DocumentResult:
    warnings = list(doc.classification_notes)
    result = DocumentResult(
        document_id=document_id(doc.source, doc.pages),
        source_file=doc.source.name,
        pages=doc.pages,
        classification=_classification(doc),
        threshold=threshold,
        preprocessing=[p for p in preprocessing if p["page"] in doc.pages],
    )
    if doc.doc_type == DocumentType.UNKNOWN:
        warnings.append("unsupported document: no fields extracted; routed to the human-review queue")
        result.warnings = warnings
        result.document_review_required = True
        return result

    fields, extract_warnings = extract_document(llm, doc, page_ocr, threshold, crop_reads=crop_reads)
    warnings.extend(extract_warnings)
    spec = DOC_SPECS[doc.doc_type]
    result.fields = {f.key: fields[f.key] for f in spec.required_fields}
    result.auxiliary_fields = {f.key: fields[f.key] for f in spec.fields if f.auxiliary}
    warnings.extend(cross_field_checks(doc.doc_type, result.fields, result.auxiliary_fields, threshold))
    if not tesseract_available():
        warnings.append("Tesseract not installed: OCR agreement and label anchoring were unavailable")
    result.warnings = warnings
    result.document_review_required = result.classification.review_required or any(f.review_required for f in result.fields.values())
    return result


def _offline_result(source: SourceDocument, page_ocr: dict[int, PageOCR], threshold: float, preprocessing: list[dict]) -> DocumentResult:
    """Without a vision model: OCR keyword classification only; every field is honestly missing."""
    text = " ".join(o.text for o in page_ocr.values())
    kw_type, _ = keyword_vote(text)
    doc_type = kw_type or DocumentType.UNKNOWN
    pages = [p.page_number for p in source.pages]
    result = DocumentResult(
        document_id=document_id(source, pages),
        source_file=source.name,
        pages=pages,
        classification=Classification(
            document_type=doc_type,
            confidence=0.4 if kw_type else 0.0,
            review_required=True,
            evidence="OCR keyword vote only (vision model not configured)",
            keyword_vote=kw_type.value if kw_type else None,
        ),
        threshold=threshold,
        preprocessing=preprocessing,
        warnings=["vision model not configured: classification is OCR-keyword only and no fields were extracted"],
    )
    if kw_type:
        for spec in DOC_SPECS[kw_type].required_fields:
            result.fields[spec.key] = FieldResult(
                confidence=0.0,
                review_required=True,
                handwritten=spec.kind != "printed",
                review_reasons=["vision model not configured; field not extracted"],
            )
    return result


def _cap(field_result: FieldResult, cap: float, reason: str, threshold: float) -> None:
    field_result.review_reasons.append(reason)
    if field_result.confidence > cap:
        field_result.confidence = cap
    field_result.review_required = True  # any cross-field conflict needs a human


def cross_field_checks(doc_type: DocumentType, fields: dict[str, FieldResult], aux: dict[str, FieldResult], threshold: float) -> list[str]:
    """Within-document consistency rules. Agreement is noted; conflicts lower confidence and force review."""
    notes: list[str] = []
    today = date.today().isoformat()
    if doc_type == DocumentType.PASSPORT:
        mrz = fields.get("mrz_line_2")
        if mrz and mrz.normalized_value and len(mrz.normalized_value) == 44:
            parts = parse_mrz_line2(mrz.normalized_value)
            checks = [
                ("passport_number", parts["document_number"].replace("<", "")),
                ("date_of_birth", mrz_date_to_iso(parts["birth_date"], prefer_past=True)),
                ("date_of_expiry", mrz_date_to_iso(parts["expiry_date"], prefer_past=False)),
            ]
            mrz_ok = validate_mrz2(mrz.normalized_value).passed
            for key, mrz_value in checks:
                viz = fields.get(key)
                if not viz or not viz.normalized_value or not mrz_value:
                    continue
                if viz.normalized_value == mrz_value:
                    notes.append(f"{key} agrees with the MRZ" + (" (MRZ check digits valid)" if mrz_ok else ""))
                else:
                    _cap(viz, 0.6, f"conflicts with MRZ value {mrz_value}", threshold)
                    notes.append(f"{key} conflicts with the MRZ")
        exp, dob = fields.get("date_of_expiry"), fields.get("date_of_birth")
        if exp and dob and exp.normalized_value and dob.normalized_value and exp.normalized_value <= dob.normalized_value:
            _cap(exp, 0.6, "expiry date is not after date of birth", threshold)
    if doc_type == DocumentType.DRIVING_LICENCE:
        doi, valid = fields.get("date_of_issue"), fields.get("valid_till")
        if doi and valid and doi.normalized_value and valid.normalized_value:
            if valid.normalized_value <= doi.normalized_value:
                _cap(valid, 0.6, "valid-till date is not after the issue date", threshold)
                notes.append("issue/validity dates are out of order")
            else:
                notes.append("issue date precedes validity date")
            if doi.normalized_value > today:
                _cap(doi, 0.6, "issue date is in the future", threshold)
    if doc_type == DocumentType.NACH_MANDATE:
        figures, words = fields.get("amount_in_figures"), aux.get("amount_in_words")
        if figures and figures.normalized_value and words and words.raw_value:
            parsed = words_to_number(words.raw_value)
            if parsed is None:
                notes.append("amount in words could not be parsed for cross-check")
            elif str(parsed) == figures.normalized_value.split(".")[0]:
                notes.append("amount in figures matches amount in words")
            else:
                _cap(figures, 0.6, f"amount in figures differs from amount in words ({parsed})", threshold)
                notes.append("amount in figures conflicts with amount in words")
    for name, f in fields.items():
        if (
            name.startswith("date")
            and f.normalized_value
            and doc_type not in {DocumentType.PASSPORT, DocumentType.DRIVING_LICENCE}
            and f.normalized_value > today
        ):
            _cap(f, 0.6, "date is in the future", threshold)
    return notes


_GROUPS = {
    "application_number": "application/policy number",
    "policy_number": "application/policy number",
    "place": "place",
    "place_of_birth": "place",
}


def add_cross_document_hints(documents: list[DocumentResult]) -> None:
    """For flagged fields, note matching values seen elsewhere in the batch. Hints only - never fills a value."""
    seen: dict[str, list[tuple[str, str]]] = {}
    for doc in documents:
        for name, f in doc.fields.items():
            group = _GROUPS.get(name)
            if group and f.normalized_value and not f.review_required:
                seen.setdefault(group, []).append((f.normalized_value, doc.source_file))
    for doc in documents:
        for name, f in doc.fields.items():
            group = _GROUPS.get(name)
            if not group or not f.review_required:
                continue
            others = [(v, src) for v, src in seen.get(group, []) if src != doc.source_file]
            if not others:
                continue
            values = sorted({v for v, _ in others})
            f.review_reasons.append(
                f"review hint only (not used to fill): {group} read confidently elsewhere in this batch as {', '.join(values[:2])}"
            )
