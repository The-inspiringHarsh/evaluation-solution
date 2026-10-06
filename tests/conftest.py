"""Shared fixtures: real workbook, scripted fake LLMs and synthetic document images."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.llm import SearchAnswer, SearchSource  # noqa: E402
from src.question2_inventory.loader import load_inventory  # noqa: E402

WORKBOOK = ROOT / "data" / "inventory" / "Inventory-Records-Sample-Data.xlsx"


@pytest.fixture(scope="session")
def inventory():
    return load_inventory(WORKBOOK)


class FakeLLM:
    """Scripted stand-in for an LLM provider (no network)."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, json_handler: Callable[[str, list, dict], dict] | None = None, text: str = "summary") -> None:
        self.json_handler = json_handler
        self.text = text
        self.json_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []
        self.search_calls: list[str] = []

    def complete_json(self, *, system, parts, schema, max_tokens=4000):
        self.json_calls.append({"system": system, "parts": parts, "schema": schema})
        assert self.json_handler is not None, "unexpected complete_json call"
        return self.json_handler(system, list(parts), schema)

    def complete_text(self, *, system, messages, max_tokens=2000):
        self.text_calls.append({"system": system, "messages": messages})
        return self.text

    def web_search(self, query, *, max_uses=3):
        self.search_calls.append(query)
        return SearchAnswer(
            query=query,
            answer="Inventory turnover is cost of goods sold divided by average inventory.",
            sources=[
                SearchSource(
                    "Investopedia: Inventory Turnover",
                    "https://www.investopedia.com/terms/i/inventoryturnover.asp",
                    "COGS / average inventory",
                )
            ],
            provider="fake",
        )


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def make_text_image(lines: list[str], size=(1400, 900)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black", font=_font(44))
        y += 90
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pdf_bytes(pages: list[list[str]]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=16)
            y += 28
    return doc.tobytes()


@pytest.fixture
def pan_like_png() -> bytes:
    return png_bytes(
        make_text_image(
            [
                "INCOME TAX DEPARTMENT",
                "Permanent Account Number Card",
                "ABCPK1234Q",
                "Name",
                "TEST PERSON",
                "Father's Name",
                "TEST FATHER",
                "Date of Birth 01/02/1990",
            ]
        )
    )
