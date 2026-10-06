"""Optional Tesseract OCR: word boxes, printed-label anchoring, keyword classification and crop reads.

Every function degrades gracefully (returns empty results) when Tesseract is not installed.
"""

from __future__ import annotations

import difflib
import functools
import logging
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .schemas import DOC_SPECS, DocumentType, FieldSpec

logger = logging.getLogger(__name__)

# Tesseract spawns OpenMP threads per process; we already parallelise across fields.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
OCR_TIMEOUT_S = 30


@functools.lru_cache(maxsize=1)
def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Word:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    line_key: tuple[int, int, int]

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height


@dataclass
class PageOCR:
    words: list[Word]
    text: str

    def lines(self) -> list[list[Word]]:
        grouped: dict[tuple[int, int, int], list[Word]] = {}
        for w in self.words:
            grouped.setdefault(w.line_key, []).append(w)
        return [sorted(ws, key=lambda w: w.left) for ws in grouped.values()]


def ocr_page(img: Image.Image) -> PageOCR:
    """Word-level OCR of a full page (grayscale, upscaled if small)."""
    if not tesseract_available():
        return PageOCR([], "")
    import pytesseract

    gray = img.convert("L")
    scale = 1.0
    if max(gray.size) < 1800:
        scale = 1800 / max(gray.size)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 3", timeout=OCR_TIMEOUT_S)
    except (pytesseract.TesseractError, RuntimeError) as exc:  # RuntimeError = timeout
        logger.warning("tesseract failed: %s", type(exc).__name__)
        return PageOCR([], "")
    words = []
    for i, text in enumerate(data["text"]):
        if not text or not text.strip():
            continue
        words.append(
            Word(
                text=text.strip(),
                left=int(data["left"][i] / scale),
                top=int(data["top"][i] / scale),
                width=int(data["width"][i] / scale),
                height=int(data["height"][i] / scale),
                conf=float(data["conf"][i]),
                line_key=(data["block_num"][i], data["par_num"][i], data["line_num"][i]),
            )
        )
    text = " ".join(w.text for w in words)
    return PageOCR(words, text)


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9<]", "", token.lower())


def find_label(ocr: PageOCR, label: str, min_ratio: float = 0.78) -> Optional[tuple[int, int, int, int]]:
    """Locate a printed label phrase; return the union box of the best-matching word run."""
    target = [_norm(t) for t in label.split() if _norm(t)]
    if not target or not ocr.words:
        return None
    target_str = " ".join(target)
    best: tuple[float, Optional[tuple[int, int, int, int]]] = (0.0, None)
    for line in ocr.lines():
        tokens = [_norm(w.text) for w in line]
        n = len(target)
        for size in {max(1, n - 1), n, n + 1}:
            for start in range(0, max(1, len(line) - size + 1)):
                window = tokens[start : start + size]
                if not window:
                    continue
                ratio = difflib.SequenceMatcher(None, " ".join(window), target_str).ratio()
                if ratio > best[0]:
                    ws = line[start : start + size]
                    box = (min(w.left for w in ws), min(w.top for w in ws), max(w.box[2] for w in ws), max(w.box[3] for w in ws))
                    best = (ratio, box)
    return best[1] if best[0] >= min_ratio else None


def anchor_region(
    ocr: PageOCR, spec: FieldSpec, page_size: tuple[int, int]
) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
    """Derive the value region from the printed label. Returns (region, label_box)."""
    w, h = page_size
    for label in spec.anchors:
        box = find_label(ocr, label)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        lh = max(12, y1 - y0)
        if spec.region == "right":
            region = (x0 - lh, y0 - int(1.8 * lh), min(w, x1 + int(0.45 * w)), y1 + int(1.8 * lh))
        elif spec.region == "right_wide":
            region = (x0 - lh, y0 - int(1.8 * lh), w, y1 + int(1.8 * lh))
        elif spec.region == "below":
            region = (x0 - 2 * lh, y0 - lh, min(w, x1 + int(0.35 * w)), y1 + int(5 * lh))
        else:  # "row"
            region = (0, y0 - int(1.5 * lh), w, y1 + int(2.5 * lh))
        return region, box
    return None, None


@functools.lru_cache(maxsize=256)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Whole-word/phrase match, so 'ecs' does not fire inside 'specs'."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])")


def keyword_vote(text: str) -> tuple[Optional[DocumentType], dict[str, int]]:
    """Independent content-based classifier: count printed keyword hits per document type."""
    lowered = text.lower()
    scores: dict[str, int] = {}
    for doc_type, spec in DOC_SPECS.items():
        scores[doc_type.value] = sum(1 for kw in spec.keywords if _keyword_pattern(kw).search(lowered))
    best_type, best = max(scores.items(), key=lambda kv: kv[1])
    ranked = sorted(scores.values(), reverse=True)
    runner_up = ranked[1] if len(ranked) > 1 else 0
    if best >= 2 and best > runner_up:
        return DocumentType(best_type), scores
    return None, scores


def ocr_crop(img: Image.Image, spec: FieldSpec) -> Optional[str]:
    """Single-block OCR of a crop. MRZ uses a character whitelist."""
    if not tesseract_available():
        return None
    import pytesseract

    config = "--psm 6"
    if spec.validator == "mrz2":
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    try:
        text = pytesseract.image_to_string(img.convert("L"), config=config, timeout=OCR_TIMEOUT_S)
    except (pytesseract.TesseractError, RuntimeError):  # RuntimeError = timeout
        return None
    text = text.strip()
    return text or None


def extract_value_after_label(ocr_text: str, spec: FieldSpec) -> Optional[str]:
    """From OCR'd crop text, keep what follows the label on the same line (printed values)."""
    if not ocr_text:
        return None
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if spec.validator == "mrz2":
        candidates = [re.sub(r"\s", "", ln) for ln in lines if ln.count("<") >= 3]
        mrz2 = [c for c in candidates if not c.startswith("P<")]
        return mrz2[-1] if mrz2 else None
    for ln in lines:
        for label in spec.anchors:
            idx = ln.lower().find(label.lower().split()[0])
            if idx >= 0:
                rest = ln[idx + len(label) :].strip(" :.-|")
                if rest:
                    return rest
    return lines[0] if len(lines) == 1 else None
