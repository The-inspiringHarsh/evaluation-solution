"""Capture the six evidence screenshots from a running app with Playwright.

Usage:
    streamlit run app.py --server.headless true &       # with an LLM key configured
    python scripts/run_question3.py                     # so "Load last saved batch results" has data
    python scripts/capture_screenshots.py [--url http://localhost:8501] [--chromium /path/to/chrome]

Masking stays ON (default) so no identity or financial number is captured.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"


def wait_idle(page: Page, timeout_ms: int = 240_000) -> None:
    """Wait until Streamlit stops showing its running indicator."""
    page.wait_for_timeout(1500)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=timeout_ms)
    page.wait_for_timeout(1200)


def ask(page: Page, question: str) -> None:
    box = page.get_by_test_id("stChatInputTextArea")
    box.fill(question)
    box.press("Enter")
    wait_idle(page)


def shot(page: Page, out: Path, name: str, focus=None) -> None:
    """Viewport screenshot (sidebar included), after scrolling ``focus`` into view."""
    if focus is not None:
        focus.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
    path = out / name
    page.screenshot(path=str(path))
    print(f"saved {path}")


def inventory_shots(page: Page, out: Path) -> None:
    # 1. numerical answer + generated code
    ask(page, "Which five products have the highest stock on hand?")
    page.get_by_text("Generated pandas code").last.click()
    page.wait_for_timeout(800)
    shot(page, out, "01_inventory_numeric_code.png", page.get_by_test_id("stChatMessage").last)

    # 2. web search with citations (fresh conversation)
    page.get_by_role("button", name="Reset conversation").click()
    wait_idle(page)
    ask(page, "What does inventory turnover mean, and does this workbook contain enough information to calculate it?")
    shot(page, out, "02_inventory_web_search.png", page.get_by_test_id("stChatMessage").last)


def document_shots(page: Page, out: Path) -> None:
    # 3-6. document pipeline, using the batch saved by scripts/run_question3.py
    page.get_by_text("Document Pipeline (Q3)").click()
    wait_idle(page)
    page.get_by_role("button", name="Load last saved batch results").click()
    wait_idle(page)
    shot(page, out, "03_documents_classification.png")

    page.get_by_role("tab", name="Field results").click()
    page.get_by_text("Show handwritten-field documents only").click()
    wait_idle(page)
    expanders = page.get_by_test_id("stExpander")
    for i in range(min(2, expanders.count())):
        expanders.nth(i).locator("summary").click()
        page.wait_for_timeout(600)
    shot(page, out, "04_handwritten_fields.png", expanders.first if expanders.count() else None)

    page.get_by_role("tab", name="Human-review queue").click()
    page.wait_for_timeout(1500)
    shot(page, out, "05_review_queue.png")

    page.get_by_role("tab", name="Downloads").click()
    page.wait_for_timeout(1500)
    page.get_by_text("Preview: all_results.json").click()  # masked JSON preview
    page.wait_for_timeout(1500)
    shot(page, out, "06_downloads.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--chromium", default=os.getenv("CHROMIUM_PATH"))
    parser.add_argument("--out", default=str(OUT), help="output folder (default: screenshots/)")
    parser.add_argument("--sections", default="inventory,documents", help="comma-separated: inventory, documents")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(args.url)
        wait_idle(page)
        if "inventory" in sections:
            inventory_shots(page, out)
        if "documents" in sections:
            document_shots(page, out)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
