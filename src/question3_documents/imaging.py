"""Ingestion (PNG/JPEG/PDF -> page images), orientation/skew correction, quality metrics and crops.

Originals are only ever read; every transformation produces a new in-memory image.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # tesseract: one thread per process

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
VISION_MAX_SIDE = 1568  # long-edge size sent to the vision model for full pages


class IngestError(ValueError):
    """Unsupported or unreadable input file."""


@dataclass
class PageImage:
    page_number: int  # 1-based
    original: Image.Image  # as decoded (EXIF-transposed), never modified
    processed: Image.Image  # orientation + deskew + contrast enhanced (RGB)
    rotation_applied: int = 0
    skew_corrected_deg: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.processed.size


@dataclass
class SourceDocument:
    name: str
    sha1: str
    kind: str  # "image" | "pdf"
    pages: list[PageImage]


def file_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()  # noqa: S324 - used as a stable id, not for security


def load_source(name: str, data: bytes, pdf_dpi: int = 200) -> SourceDocument:
    """Decode an uploaded file into processed page images."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"Unsupported file type '{ext}'. Supported: PNG, JPG/JPEG, PDF.")
    if not data:
        raise IngestError(f"{name} is empty.")
    if ext == ".pdf":
        images = pdf_to_images(data, dpi=pdf_dpi)
        kind = "pdf"
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:  # PIL raises several decoder-specific errors
            raise IngestError(f"{name} could not be decoded as an image.") from exc
        images = [ImageOps.exif_transpose(img)]
        kind = "image"
    pages = [preprocess_page(i + 1, im) for i, im in enumerate(images)]
    return SourceDocument(name=name, sha1=file_sha1(data), kind=kind, pages=pages)


def load_path(path: str | Path, pdf_dpi: int = 200) -> SourceDocument:
    path = Path(path)
    return load_source(path.name, path.read_bytes(), pdf_dpi=pdf_dpi)


def pdf_to_images(data: bytes, dpi: int = 200) -> list[Image.Image]:
    try:
        import pymupdf
    except ImportError:  # older PyMuPDF releases
        import fitz as pymupdf  # type: ignore[no-redef]
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises its own error types
        raise IngestError("The PDF could not be opened.") from exc
    if doc.page_count == 0:
        raise IngestError("The PDF has no pages.")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return images


# --------------------------------------------------------------------------- preprocessing


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img.copy()
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def detect_orientation(img: Image.Image) -> tuple[int, float]:
    """Return (clockwise rotation in degrees, confidence) from Tesseract OSD, or (0, 0) if unavailable."""
    try:
        import pytesseract

        gray = img.convert("L")
        if max(gray.size) < 1000:
            scale = 1000 / max(gray.size)
            gray = gray.resize((int(gray.width * scale), int(gray.height * scale)))
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT, config="--psm 0", timeout=30)
        return int(osd.get("rotate", 0)), float(osd.get("orientation_conf", 0.0))
    except Exception as exc:  # tesseract missing or OSD failed on sparse text
        logger.debug("orientation detection unavailable: %s", type(exc).__name__)
        return 0, 0.0


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate small text skew (degrees, |angle| <= 10) from long near-horizontal line segments."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(100, gray.shape[1] // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=150, minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 10:
            angles.append(angle)
    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def quality_metrics(gray: np.ndarray) -> dict[str, float]:
    """Sharpness (Laplacian variance), contrast and resolution -> a 0..1 quality score."""
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    res = float(min(gray.shape))
    s_sharp = min(1.0, sharp / 300.0)
    s_contrast = min(1.0, contrast / 60.0)
    s_res = min(1.0, res / 1000.0)
    score = round(0.45 * s_sharp + 0.30 * s_contrast + 0.25 * s_res, 3)
    return {"sharpness": round(sharp, 1), "contrast": round(contrast, 1), "min_side_px": res, "quality_score": score}


def enhance(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel plus light denoising (keeps colour ink distinguishable)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    out = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    return cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)


def rotate_bound(arr: np.ndarray, angle: float) -> np.ndarray:
    h, w = arr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_page(page_number: int, original: Image.Image) -> PageImage:
    rgb_img = to_rgb(original)
    rotation, conf = detect_orientation(rgb_img)
    if rotation and conf >= 2.0:
        rgb_img = rgb_img.rotate(-rotation, expand=True)
    else:
        rotation = 0
    arr = np.array(rgb_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    skew = estimate_skew(gray)
    if abs(skew) >= 0.3:
        arr = rotate_bound(arr, skew)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        skew = 0.0
    quality = quality_metrics(gray)
    processed = Image.fromarray(enhance(arr))
    return PageImage(page_number, original, processed, rotation, round(skew, 2), quality)


# --------------------------------------------------------------------------- encodings & crops


def encode_png(img: Image.Image, max_side: Optional[int] = None) -> bytes:
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_jpeg(img: Image.Image, max_side: Optional[int] = None, quality: int = 90) -> bytes:
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def clamp_box(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, int(x0)), min(w, int(x1))))
    y0, y1 = sorted((max(0, int(y0)), min(h, int(y1))))
    if x1 - x0 < 8:
        x1 = min(w, x0 + 8)
    if y1 - y0 < 8:
        y1 = min(h, y0 + 8)
    return x0, y0, x1, y1


def crop_color(page: PageImage, box: tuple[int, int, int, int], min_width: int = 1100) -> Image.Image:
    """Colour crop, upscaled so thin handwriting strokes survive model downsampling."""
    crop = page.processed.crop(box)
    if crop.width < min_width:
        scale = min(4.0, min_width / max(1, crop.width))
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    return crop


def crop_binarized(page: PageImage, box: tuple[int, int, int, int], min_width: int = 1100) -> Image.Image:
    """Independent preprocessing path: grayscale, unsharp mask, adaptive threshold, upscale."""
    gray = cv2.cvtColor(
        np.array(to_rgb(page.original if page.rotation_applied == 0 and page.skew_corrected_deg == 0 else page.processed)),
        cv2.COLOR_RGB2GRAY,
    )
    crop = gray[box[1] : box[3], box[0] : box[2]]
    if crop.size == 0:
        crop = gray
    scale = min(4.0, max(1.0, min_width / max(1, crop.shape[1])))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(crop, (0, 0), 2)
    sharp = cv2.addWeighted(crop, 1.6, blur, -0.6, 0)
    binar = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(binar)
