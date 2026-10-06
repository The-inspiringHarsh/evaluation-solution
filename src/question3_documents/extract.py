"""Classification and multi-pass field extraction.

Printed fields:     vision full page + vision crop + OCR crop + validator
Handwritten fields: vision full page + vision crop (colour, enhanced) + vision crop (binarised,
                    character-by-character prompt) + OCR crop (low weight) + validator
Checkbox fields:    vision full page + two independent crop reads of the option group

Crop reads never see what another pass returned, so their agreement is real evidence.

``crop_reads="batched"`` sends all colour crops of a document in one request and all binarised crops in a
second one (about four requests per document instead of one per crop) for rate-limited keys. Each crop is
labelled with its field and read only for that field, so the passes stay independent of each other.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

from ..common.llm import ImagePart, LLMClient, LLMError, TextPart
from .confidence import Evidence, legibility_from_reads, needs_review, score
from .imaging import PageImage, SourceDocument, VISION_MAX_SIDE, clamp_box, crop_binarized, crop_color, encode_jpeg, encode_png
from .ocr import PageOCR, anchor_region, extract_value_after_label, keyword_vote, ocr_crop
from .schemas import DOC_SPECS, SUPPORTED_TYPES, DocSpec, DocumentType, FieldResult, FieldSpec
from .validators import comparison_key, validate

logger = logging.getLogger(__name__)

LEGIBILITY_ENUM = ["clear", "partial", "poor", "absent"]

# --------------------------------------------------------------------------- classification

CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pages": {"type": "array", "items": {"type": "integer"}},
                    "document_type": {"type": "string", "enum": [t.value for t in DocumentType]},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["pages", "document_type", "confidence", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["documents"],
    "additionalProperties": False,
}


def _classify_system() -> str:
    lines = [f"- {s.doc_type.value}: {s.display}. {s.cues}" for s in DOC_SPECS.values()]
    return (
        "You classify scanned Indian identity and insurance documents by their visible CONTENT "
        "(titles, printed labels, layout).\n"
        "Supported types:\n" + "\n".join(lines) + "\n"
        "- unknown_or_other: anything else, including other insurance forms (e.g. a proposal/application "
        "declaration or an assignment request form) even if they share the same insurer branding.\n"
        "Group consecutive pages that belong to the SAME form into one logical document (continuation pages of a "
        "form are one document). Every page must appear in exactly one document.\n"
        "confidence: your probability (0-1) that the type is correct. evidence: the printed title or key label you "
        "relied on (max 12 words). Do not use file names; none are given."
    )


@dataclass
class LogicalDocument:
    source: SourceDocument
    pages: list[int]
    doc_type: DocumentType
    model_confidence: float
    evidence: str
    keyword_type: Optional[DocumentType] = None
    classification_confidence: float = 0.0
    classification_review: bool = True
    classification_notes: list[str] = field(default_factory=list)


def classify_source(llm: LLMClient, source: SourceDocument, page_ocr: dict[int, PageOCR], threshold: float) -> list[LogicalDocument]:
    parts: list[Any] = []
    for page in source.pages:
        parts.append(ImagePart(encode_jpeg(page.processed, max_side=1200), "image/jpeg", label=f"Page {page.page_number}:"))
    parts.append(TextPart(f"This file has {len(source.pages)} page(s). Classify and group them."))
    data = llm.complete_json(system=_classify_system(), parts=parts, schema=CLASSIFY_SCHEMA, max_tokens=3000)
    docs = _sanitize_groups(data.get("documents", []), [p.page_number for p in source.pages])
    results = []
    for group in docs:
        text = " ".join(page_ocr[p].text for p in group["pages"] if p in page_ocr)
        kw_type, _ = keyword_vote(text)
        doc = LogicalDocument(
            source=source,
            pages=group["pages"],
            doc_type=DocumentType(group["document_type"]),
            model_confidence=max(0.0, min(1.0, float(group["confidence"]))),
            evidence=group["evidence"][:200],
            keyword_type=kw_type,
        )
        _score_classification(doc, threshold)
        results.append(doc)
    return results


def _sanitize_groups(groups: list[dict[str, Any]], all_pages: list[int]) -> list[dict[str, Any]]:
    """Ensure each page is used exactly once; leftover pages become their own unknown document."""
    seen: set[int] = set()
    clean = []
    for g in groups:
        pages = [p for p in dict.fromkeys(g.get("pages", [])) if p in all_pages and p not in seen]
        if not pages:
            continue
        seen.update(pages)
        clean.append({**g, "pages": sorted(pages)})
    for p in all_pages:
        if p not in seen:
            clean.append(
                {
                    "pages": [p],
                    "document_type": DocumentType.UNKNOWN.value,
                    "confidence": 0.0,
                    "evidence": "page not assigned by classifier",
                }
            )
    return sorted(clean, key=lambda g: g["pages"][0])


def _score_classification(doc: LogicalDocument, threshold: float) -> None:
    """confidence = 0.6 * model probability + 0.4 * independent keyword agreement."""
    if doc.keyword_type is None:
        kw_component = 0.5
        doc.classification_notes.append("OCR keyword classifier found no decisive printed keywords")
    elif doc.keyword_type == doc.doc_type:
        kw_component = 1.0
    else:
        kw_component = 0.0
        doc.classification_notes.append(
            f"OCR keyword classifier suggests '{doc.keyword_type.value}' but the vision model says '{doc.doc_type.value}'"
        )
    doc.classification_confidence = round(0.6 * doc.model_confidence + 0.4 * kw_component, 3)
    doc.classification_review = doc.doc_type == DocumentType.UNKNOWN or needs_review(doc.classification_confidence, threshold)
    if doc.doc_type == DocumentType.UNKNOWN:
        doc.classification_notes.append("document type is not one of the ten supported types")


# --------------------------------------------------------------------------- full-page extraction


def _full_page_schema(spec: DocSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": [f.key for f in spec.fields]},
                        "value": {"type": ["string", "null"]},
                        "page": {"type": ["integer", "null"]},
                        "legibility": {"type": "string", "enum": LEGIBILITY_ENUM},
                        "handwritten": {"type": "boolean"},
                        "evidence": {"type": "string"},
                        "bbox": {"type": ["array", "null"], "items": {"type": "number"}},
                        "ambiguous_characters": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["field", "value", "page", "legibility", "handwritten", "evidence", "bbox", "ambiguous_characters"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


_TRANSCRIBE_RULES = (
    "Transcription rules:\n"
    "- Copy exactly what is written/printed. Do not correct spelling, do not complete partial values, do not "
    "reformat dates, do not infer from other documents or from what would be 'typical'.\n"
    "- If a value is blank, illegible or not present, return null with legibility 'absent' or 'poor'.\n"
    "- Watch for 0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G confusions; list every character you are unsure of in "
    "ambiguous_characters (e.g. '5 or S at position 7').\n"
    "- evidence: the printed label (and page) next to the value, max 12 words. No reasoning.\n"
)


def _field_lines(spec: DocSpec) -> str:
    lines = []
    for f in spec.fields:
        extra = f" Options: {', '.join(f.options)}. Return the ticked option text exactly as listed." if f.options else ""
        lines.append(f"- {f.key}: {f.label} - {f.description}.{extra}")
    return "\n".join(lines)


def extract_full_page(llm: LLMClient, doc: LogicalDocument, spec: DocSpec) -> dict[str, dict[str, Any]]:
    parts: list[Any] = []
    for p in doc.pages:
        page = doc.source.pages[p - 1]
        parts.append(ImagePart(encode_png(page.processed, max_side=VISION_MAX_SIDE), "image/png", label=f"Page {p}:"))
    system = (
        f"You extract fields from a {spec.display}.\n"
        + _TRANSCRIBE_RULES
        + "- bbox: the value's bounding box on its page as [x0, y0, x1, y1] in 0-1000 normalised coordinates "
        "(0,0 = top-left), or null if absent.\n- handwritten: true if the value is handwritten/ticked by hand."
    )
    prompt = f"Return one entry for every field below.\n{_field_lines(spec)}"
    data = llm.complete_json(system=system, parts=[*parts, TextPart(prompt)], schema=_full_page_schema(spec), max_tokens=6000)
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("fields", []):
        out.setdefault(item["field"], item)
    return out


# --------------------------------------------------------------------------- crop reads

CROP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {"type": ["string", "null"]},
        "legibility": {"type": "string", "enum": LEGIBILITY_ENUM},
        "label_visible": {"type": "boolean"},
        "ambiguous_characters": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
    },
    "required": ["value", "legibility", "label_visible", "ambiguous_characters", "evidence"],
    "additionalProperties": False,
}


@dataclass
class Read:
    method: str
    value: Optional[str]
    legibility: str = ""
    ambiguous: list[str] = field(default_factory=list)
    label_visible: Optional[bool] = None
    evidence: str = ""
    error: Optional[str] = None


def _crop_prompt(spec: FieldSpec, doc_spec: DocSpec, variant: str) -> tuple[str, str]:
    system = (
        f"You read ONE field from a cropped region of a {doc_spec.display}. The crop was cut around the printed "
        f"label {spec.anchors[0]!r} (if visible) and may contain neighbouring fields - read only the requested one.\n" + _TRANSCRIBE_RULES
    )
    return system, _crop_task(spec, variant)


def _crop_task(spec: FieldSpec, variant: str) -> str:
    kind = "handwritten" if spec.kind != "printed" else "printed"
    if spec.options:
        task = (
            f"Which option is ticked/marked for '{spec.label}'? Options: {', '.join(spec.options)}. "
            "Return the ticked option text exactly as listed (or null if none is clearly ticked). A struck-through "
            "option is NOT selected."
        )
    elif variant == "binarized":
        task = (
            f"This is a high-contrast black-and-white version of the crop. Read the {kind} value of '{spec.label}' "
            f"({spec.description}) CHARACTER BY CHARACTER, left to right, then return the full value."
        )
    else:
        task = f"Read the {kind} value of '{spec.label}' ({spec.description})."
    return task


def _read_from(method: str, data: dict[str, Any]) -> Read:
    value = data.get("value")
    value = value.strip() if isinstance(value, str) and value.strip() else None
    return Read(
        method,
        value,
        data.get("legibility", ""),
        list(data.get("ambiguous_characters") or []),
        data.get("label_visible"),
        data.get("evidence", ""),
    )


def crop_read(llm: LLMClient, image: Image.Image, spec: FieldSpec, doc_spec: DocSpec, method: str, variant: str) -> Read:
    system, task = _crop_prompt(spec, doc_spec, variant)
    try:
        data = llm.complete_json(
            system=system,
            parts=[ImagePart(encode_png(image, max_side=VISION_MAX_SIDE), "image/png"), TextPart(task)],
            schema=CROP_SCHEMA,
            max_tokens=2000,
        )
    except LLMError as exc:
        return Read(method, None, error=str(exc))
    return _read_from(method, data)


def batch_crop_reads(
    llm: LLMClient, items: list[tuple[FieldSpec, Image.Image]], doc_spec: DocSpec, method: str, variant: str
) -> dict[str, Read]:
    """Read several fields in one request, one labelled crop per field. Returns a Read per field key."""
    if not items:
        return {}
    system = (
        f"You read several fields from cropped regions of a {doc_spec.display}. Each crop is labelled with the one "
        "field it was cut for and may contain neighbouring fields. Read each field ONLY from its own crop; never "
        "use another crop to fill or correct a value.\n" + _TRANSCRIBE_RULES
    )
    parts: list[Any] = []
    tasks: list[str] = []
    for spec, image in items:
        parts.append(ImagePart(encode_png(image, max_side=VISION_MAX_SIDE), "image/png", f"Crop for field '{spec.key}':"))
        tasks.append(f"- {spec.key} (crop cut around the printed label {spec.anchors[0]!r}): {_crop_task(spec, variant)}")
    parts.append(TextPart("Return one result per field key, each read from that field's crop:\n" + "\n".join(tasks)))
    keys = [spec.key for spec, _ in items]
    schema = {"type": "object", "properties": {k: CROP_SCHEMA for k in keys}, "required": keys, "additionalProperties": False}
    try:
        data = llm.complete_json(system=system, parts=parts, schema=schema, max_tokens=1500 * len(items))
    except LLMError as exc:
        return {k: Read(method, None, error=str(exc)) for k in keys}
    return {
        k: _read_from(method, data[k]) if isinstance(data.get(k), dict) else Read(method, None, error="field missing from batched response")
        for k in keys
    }


# --------------------------------------------------------------------------- regions


def _bbox_to_pixels(bbox: Any, size: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        return None
    w, h = size
    return int(x0 / 1000 * w), int(y0 / 1000 * h), int(x1 / 1000 * w), int(y1 / 1000 * h)


def _pad(box: tuple[int, int, int, int], size: tuple[int, int], spec: FieldSpec) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad_x = int(0.03 * w + 0.15 * bw)
    pad_y = int(0.012 * h + 0.6 * bh)
    if spec.validator == "mrz2":
        return clamp_box((0, y0 - pad_y, w, y1 + pad_y), size)
    return clamp_box((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), size)


def _near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], size: tuple[int, int]) -> bool:
    w, h = size
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return dx <= 0.25 * w and dy <= 0.04 * h


def choose_region(
    page: PageImage, spec: FieldSpec, full_read: Optional[dict[str, Any]], ocr: Optional[PageOCR]
) -> tuple[Optional[tuple[int, int, int, int]], float, str]:
    """Return (crop box, proximity score, region description).

    Proximity: 1.0 when the model's value box sits next to the OCR-located printed label, 0.8 when only the
    label anchor is available, 0.5 when only the model box is available, 0.0 when no region exists.
    """
    size = page.size
    model_box = _bbox_to_pixels(full_read.get("bbox"), size) if full_read else None
    anchor, label_box = anchor_region(ocr, spec, size) if ocr else (None, None)
    if model_box and label_box and _near(model_box, label_box, size):
        union = (
            min(model_box[0], label_box[0]),
            min(model_box[1], label_box[1]),
            max(model_box[2], label_box[2]),
            max(model_box[3], label_box[3]),
        )
        return _pad(union, size, spec), 1.0, "model value box adjacent to OCR-located label"
    if model_box:
        return _pad(model_box, size, spec), 0.5, "model value box (label not confirmed by OCR)"
    if anchor:
        return clamp_box(anchor, size), 0.8, "region derived from OCR-located printed label"
    return None, 0.0, "no region located"


# --------------------------------------------------------------------------- per-field orchestration


def _field_region(
    doc: LogicalDocument, spec: FieldSpec, full_read: Optional[dict[str, Any]], page_ocr: dict[int, PageOCR]
) -> tuple[PageImage, Optional[tuple[int, int, int, int]], float, str]:
    """Page and crop region for one field (deterministic, so batched and per-field reads use the same crop)."""
    page_no = (full_read or {}).get("page") or doc.pages[0]
    if page_no not in doc.pages:
        page_no = doc.pages[0]
    page = doc.source.pages[page_no - 1]
    region, proximity, region_desc = choose_region(page, spec, full_read, page_ocr.get(page_no))
    return page, region, proximity, region_desc


def _batched_crop_reads(
    llm: LLMClient, doc: LogicalDocument, spec: DocSpec, full: dict[str, dict[str, Any]], page_ocr: dict[int, PageOCR]
) -> dict[str, dict[str, Read]]:
    """All colour crops in one request, all binarised crops (non-printed fields) in another."""
    color: list[tuple[FieldSpec, Image.Image]] = []
    binarized: list[tuple[FieldSpec, Image.Image]] = []
    for f in spec.fields:
        page, region, _, _ = _field_region(doc, f, full.get(f.key), page_ocr)
        if region is None:
            continue
        color.append((f, crop_color(page, region)))
        if f.kind != "printed":
            binarized.append((f, crop_binarized(page, region)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        color_job = pool.submit(batch_crop_reads, llm, color, spec, "vision_crop", "color")
        binarized_job = pool.submit(batch_crop_reads, llm, binarized, spec, "vision_crop_binarized", "binarized")
        color_reads, binarized_reads = color_job.result(), binarized_job.result()
    out: dict[str, dict[str, Read]] = {}
    for key, read in color_reads.items():
        out.setdefault(key, {})["color"] = read
    for key, read in binarized_reads.items():
        out.setdefault(key, {})["binarized"] = read
    return out


def _ocr_support(key: Optional[str], ocr_text: Optional[str]) -> Optional[float]:
    if not key or not ocr_text:
        return None
    hay = re.sub(r"[\s.,]", "", ocr_text).upper()
    if not hay:
        return None
    if key in hay:
        return 1.0
    n = len(key)
    best = 0.0
    for i in range(0, max(1, len(hay) - n + 1)):
        best = max(best, difflib.SequenceMatcher(None, hay[i : i + n], key).ratio())
    return round(best, 3)


def resolve_field(
    llm: LLMClient,
    doc: LogicalDocument,
    doc_spec: DocSpec,
    spec: FieldSpec,
    full_read: Optional[dict[str, Any]],
    page_ocr: dict[int, PageOCR],
    threshold: float,
    prefetched: Optional[dict[str, Read]] = None,
    failed_passes: Optional[list[tuple[str, str, str]]] = None,
) -> FieldResult:
    page, region, proximity, region_desc = _field_region(doc, spec, full_read, page_ocr)
    page_no = page.page_number
    prefetched = prefetched or {}

    reads: list[Read] = []
    if full_read is not None:
        v = full_read.get("value")
        reads.append(
            Read(
                "vision_full_page",
                v.strip() if isinstance(v, str) and v.strip() else None,
                full_read.get("legibility", ""),
                list(full_read.get("ambiguous_characters") or []),
                None,
                full_read.get("evidence", ""),
            )
        )
    ocr_text: Optional[str] = None
    if region is not None:
        color = crop_color(page, region)
        reads.append(prefetched.get("color") or crop_read(llm, color, spec, doc_spec, "vision_crop", "color"))
        if spec.kind != "printed":
            reads.append(
                prefetched.get("binarized")
                or crop_read(llm, crop_binarized(page, region), spec, doc_spec, "vision_crop_binarized", "binarized")
            )
        ocr_text = ocr_crop(color if spec.kind == "printed" else crop_binarized(page, region), spec)

    errors = [r.error for r in reads if r.error]
    if failed_passes is not None:
        failed_passes.extend((spec.key, r.method, r.error) for r in reads if r.error)
    vision = [r for r in reads if r.error is None]
    keys = {id(r): comparison_key(spec.validator, r.value, spec.options) for r in vision}
    nonnull = [r for r in vision if keys[id(r)]]
    counts = Counter(keys[id(r)] for r in nonnull)

    result = FieldResult(page=page_no, handwritten=spec.kind != "printed", region=list(region) if region else None)
    result.method = [r.method for r in vision]
    if ocr_text is not None:
        result.method.append("ocr_tesseract")
    reasons = [f"a vision pass failed: {e}" for e in errors]

    if not nonnull:
        ev = Evidence(missing=True, handwritten=result.handwritten)
        result.confidence = score(ev)
        result.review_required = True
        result.review_reasons = reasons + ["value not found or unreadable in any pass"]
        result.evidence = _evidence_text(vision, region_desc, page_no)
        result.confidence_components = ev.components()
        return result

    top_key, top_count = counts.most_common(1)[0]
    tied = len([k for k, c in counts.items() if c == top_count]) > 1
    disagreement = len(counts) > 1 and (top_count < 2 or tied)
    preferred_order = (
        ["vision_crop", "vision_crop_binarized", "vision_full_page"] if result.handwritten else ["vision_full_page", "vision_crop"]
    )
    agreeing = sorted((r for r in nonnull if keys[id(r)] == top_key), key=lambda r: preferred_order.index(r.method))
    chosen = agreeing[0]
    result.candidates = list(dict.fromkeys(r.value for r in nonnull if r.value))

    validation = validate(spec.validator, chosen.value or "", spec.options)
    ambiguous_count = max((len(r.ambiguous) for r in agreeing), default=0)
    label_seen = any(r.label_visible for r in vision if r.label_visible is not None)
    if proximity == 0.5 and label_seen:
        proximity = 0.8
    key_for_ocr = top_key
    if spec.validator == "mrz2" and ocr_text:
        ocr_text = extract_value_after_label(ocr_text, spec) or ocr_text

    ev = Evidence(
        agreement=top_count / max(1, len(vision)) if not disagreement else 0.0,
        validation=None if validation.passed is None else (1.0 if validation.passed else 0.0),
        legibility=legibility_from_reads([r.legibility for r in agreeing], ambiguous_count),
        ocr=_ocr_support(key_for_ocr, ocr_text),
        proximity=proximity,
        image_quality=page.quality.get("quality_score"),
        handwritten=result.handwritten,
        n_vision_reads=len(vision),
        majority=not disagreement,
        some_read_empty=len(nonnull) < len(vision),
        coerced=validation.coerced,
    )
    if "validator" not in result.method and validation.passed is not None:
        result.method.append("validator")
    conf = score(ev)

    if disagreement:
        result.raw_value = None
        result.normalized_value = None
        reasons.append("independent reads disagree: " + " | ".join(f"{r.method}={r.value!r}" for r in nonnull))
    else:
        result.raw_value = chosen.value
        result.normalized_value = validation.normalized
    if validation.passed is False:
        reasons.extend(f"validation: {n}" for n in validation.notes)
    elif validation.coerced:
        reasons.extend(f"normalisation: {n}" for n in validation.notes)
    if ev.some_read_empty:
        reasons.append("some passes returned no value: " + ", ".join(r.method for r in vision if not keys[id(r)]))
    if ambiguous_count:
        reasons.append("ambiguous characters reported: " + "; ".join(agreeing[0].ambiguous[:4]))
    if proximity < 1.0:
        reasons.append(f"label proximity not fully confirmed ({region_desc})")
    reasons.extend(ev.caps_applied)

    result.confidence = conf
    result.confidence_components = ev.components()
    result.review_required = needs_review(conf, threshold) or disagreement or validation.passed is False
    result.review_reasons = list(dict.fromkeys(reasons)) if result.review_required else [r for r in reasons if r.startswith("validation")]
    if result.review_required and not result.review_reasons:
        result.review_reasons = [f"confidence {conf:.2f} below threshold {threshold:.2f}"]
    elif result.review_required and conf < threshold:
        result.review_reasons.insert(0, f"confidence {conf:.2f} below threshold {threshold:.2f}")
    result.evidence = _evidence_text(agreeing, region_desc, page_no)
    return result


def _evidence_text(reads: list[Read], region_desc: str, page_no: int) -> str:
    label = next((r.evidence for r in reads if r.evidence), "")
    return f"p{page_no}: {label[:120]}" + (f" [{region_desc}]" if region_desc else "")


def extract_document(
    llm: LLMClient,
    doc: LogicalDocument,
    page_ocr: dict[int, PageOCR],
    threshold: float,
    workers: int = 4,
    crop_reads: str = "per_field",
) -> tuple[dict[str, FieldResult], list[str]]:
    """Run all passes for one logical document. Returns (fields, warnings)."""
    spec = DOC_SPECS[doc.doc_type]
    warnings: list[str] = []
    try:
        full = extract_full_page(llm, doc, spec)
    except LLMError as exc:
        full = {}
        warnings.append(f"full-page extraction failed: {exc}")
    prefetched = _batched_crop_reads(llm, doc, spec, full, page_ocr) if crop_reads == "batched" else {}
    failed: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            f.key: pool.submit(resolve_field, llm, doc, spec, f, full.get(f.key), page_ocr, threshold, prefetched.get(f.key), failed)
            for f in spec.fields
        }
        fields = {k: fut.result() for k, fut in futures.items()}
    # A failed pass lowers the evidence for a field but is not by itself a review reason, so record it here
    # where it is visible even when the remaining passes still agree.
    for method, error in sorted({(m, e) for _, m, e in failed}):
        keys = sorted(k for k, m, e in failed if (m, e) == (method, error))
        warnings.append(f"{method} pass failed for {', '.join(keys)}: {error}")
    return fields, warnings


__all__ = [
    "LogicalDocument",
    "SUPPORTED_TYPES",
    "batch_crop_reads",
    "classify_source",
    "extract_document",
    "resolve_field",
    "choose_region",
]
