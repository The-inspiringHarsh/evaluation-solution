"""Live smoke test against the supplied files. Runs only with RUN_LIVE_TESTS=1 and a configured key."""

import os
from pathlib import Path

import pytest

from src.common.config import get_settings
from src.common.llm import get_cached_llm_client
from src.question2_inventory.agent import InventoryAgent
from src.question2_inventory.loader import load_inventory
from src.question2_inventory.search import build_search_provider
from src.question3_documents.pipeline import process_files
from src.question3_documents.schemas import DocumentType

ROOT = Path(__file__).resolve().parents[1]
settings = get_settings()
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not settings.llm_available,
    reason="set RUN_LIVE_TESTS=1 and an LLM API key to run live integration tests",
)


def test_live_inventory_count():
    agent = InventoryAgent(
        load_inventory(ROOT / "data/inventory/Inventory-Records-Sample-Data.xlsx"), get_cached_llm_client(settings), None
    )
    turn = agent.ask("How many products are in the inventory?")
    assert turn.execution is not None and turn.execution.ok
    assert int(turn.execution.result if not hasattr(turn.execution.result, "iloc") else turn.execution.result.iloc[0]) == 46


def _require_supplied_documents() -> None:
    if not any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"} for p in (ROOT / "data/documents").glob("*")):
        pytest.skip("copy the supplied documents into data/documents/ (they are git-ignored)")


def test_live_pdf_is_unsupported():
    _require_supplied_documents()
    path = ROOT / "data/documents/Assignment Ashok.pdf"
    batch = process_files([(path.name, path.read_bytes())], get_cached_llm_client(settings), settings.review_threshold)
    assert batch.documents
    assert all(d.classification.document_type == DocumentType.UNKNOWN for d in batch.documents)
    assert all(d.document_review_required for d in batch.documents)


def test_live_search_question_cites_clickable_sources():
    llm = get_cached_llm_client(settings)
    agent = InventoryAgent(
        load_inventory(ROOT / "data/inventory/Inventory-Records-Sample-Data.xlsx"),
        llm,
        build_search_provider(settings.search_provider, llm),
    )
    turn = agent.ask("What does inventory turnover mean, and does this workbook contain enough information to calculate it?")
    assert turn.search is not None and turn.search.sources
    assert all(s.url.startswith(("http://", "https://")) for s in turn.search.sources)
    assert "From the workbook" in turn.answer and "Sources (web)" in turn.answer


def test_live_sample_images_cover_the_ten_types_with_scored_fields():
    """Coverage of the supplied set only (no per-file expectations), plus field-level invariants."""
    _require_supplied_documents()
    files = [(p.name, p.read_bytes()) for p in sorted((ROOT / "data/documents").iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    batch = process_files(files, get_cached_llm_client(settings), settings.review_threshold)
    assert not batch.file_errors
    supported = set(DocumentType) - {DocumentType.UNKNOWN}
    assert supported <= {d.classification.document_type for d in batch.documents}
    for doc in batch.documents:
        for f in doc.fields.values():
            assert 0.0 <= f.confidence <= 1.0
            if f.normalized_value is None:
                assert f.review_required
            if f.raw_value is None and not f.candidates:
                assert f.confidence == 0.0
            if not f.review_required:
                assert f.confidence >= settings.review_threshold
