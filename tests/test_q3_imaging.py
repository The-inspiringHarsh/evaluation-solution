"""Ingestion of images and PDFs, multi-page handling and preprocessing (originals untouched)."""

import io

import numpy as np
import pytest
from PIL import Image

from src.question3_documents.imaging import IngestError, crop_binarized, crop_color, load_source, pdf_to_images
from tests.conftest import make_text_image, pdf_bytes, png_bytes


def test_png_ingestion(pan_like_png):
    src = load_source("card.png", pan_like_png)
    assert src.kind == "image" and len(src.pages) == 1
    page = src.pages[0]
    assert page.processed.mode == "RGB"
    assert 0 <= page.quality["quality_score"] <= 1


def test_jpeg_and_rgba_ingestion():
    img = make_text_image(["hello"]).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert load_source("a.png", buf.getvalue()).pages[0].processed.mode == "RGB"
    buf = io.BytesIO()
    make_text_image(["hello"]).save(buf, format="JPEG")
    assert load_source("a.jpeg", buf.getvalue()).kind == "image"


def test_multi_page_pdf():
    data = pdf_bytes([["Page one title"], ["Page two continuation"]])
    src = load_source("form.pdf", data, pdf_dpi=100)
    assert src.kind == "pdf"
    assert [p.page_number for p in src.pages] == [1, 2]
    assert len(pdf_to_images(data, dpi=72)) == 2


def test_unsupported_and_corrupt_files():
    with pytest.raises(IngestError):
        load_source("notes.txt", b"hello")
    with pytest.raises(IngestError):
        load_source("broken.png", b"not an image")
    with pytest.raises(IngestError):
        load_source("broken.pdf", b"%PDF-garbage")
    with pytest.raises(IngestError):
        load_source("empty.png", b"")


def test_original_is_not_modified(pan_like_png):
    src = load_source("card.png", pan_like_png)
    original = Image.open(io.BytesIO(pan_like_png)).convert("RGB")
    assert np.array_equal(np.array(src.pages[0].original.convert("RGB")), np.array(original))


def test_sideways_page_is_rotated_upright():
    pytest.importorskip("pytesseract")
    from src.question3_documents.ocr import tesseract_available

    if not tesseract_available():
        pytest.skip("tesseract not installed")
    lines = ["INCOME TAX DEPARTMENT GOVT OF INDIA"] * 3 + ["Permanent Account Number Card for testing orientation"] * 6
    upright = make_text_image(lines, size=(1600, 1000))
    rotated = upright.rotate(90, expand=True)
    page = load_source("r.png", png_bytes(rotated)).pages[0]
    assert page.rotation_applied in (90, 270)
    assert page.processed.width > page.processed.height


def test_crops(pan_like_png):
    page = load_source("card.png", pan_like_png).pages[0]
    c = crop_color(page, (50, 50, 400, 150))
    b = crop_binarized(page, (50, 50, 400, 150))
    assert c.width >= 1000 and b.mode == "L"
    assert set(np.unique(np.array(b))) <= {0, 255}
