"""InsightExtract AI - one Streamlit app for Question 2 (inventory agent) and Question 3 (document pipeline).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from PIL import ImageDraw, ImageFilter

from src.common.config import DATA_DIR, OUTPUT_DIR, get_settings
from src.common.llm import LLMError, LLMNotConfigured, get_cached_llm_client
from src.common.privacy import SENSITIVE_FIELDS, mask_field, mask_numbers_in_text
from src.question2_inventory.agent import InventoryAgent
from src.question2_inventory.loader import WorkbookError, load_inventory, profile, stock_consistency
from src.question2_inventory.sandbox import run_code
from src.question2_inventory.search import SearchError, build_search_provider
from src.question3_documents.imaging import IngestError, load_source
from src.question3_documents.pipeline import add_cross_document_hints, process_one
from src.question3_documents.report import (
    FLAG_COLUMNS,
    SUMMARY_COLUMNS,
    batch_json,
    document_json,
    flag_rows,
    flagging_json,
    summary_rows,
    to_csv,
)
from src.question3_documents.schemas import DocumentResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="InsightExtract AI", page_icon="🧾", layout="wide")

SETTINGS = get_settings()
DEFAULT_WORKBOOK = DATA_DIR / "inventory" / "Inventory-Records-Sample-Data.xlsx"
SAMPLE_DOC_DIR = DATA_DIR / "documents"
SAVED_RESULTS = OUTPUT_DIR / "question3" / "all_results.json"


# --------------------------------------------------------------------------- shared helpers


@st.cache_resource(show_spinner=False)
def llm_client():
    """Build the LLM client once per process; returns (client, error_message)."""
    try:
        return get_cached_llm_client(SETTINGS), None
    except LLMNotConfigured as exc:
        return None, str(exc)
    except LLMError as exc:
        return None, str(exc)


def sidebar() -> str:
    st.sidebar.title("InsightExtract AI")
    page = st.sidebar.radio("Navigate", ["Inventory Agent (Q2)", "Document Pipeline (Q3)", "Setup & About"], label_visibility="collapsed")
    st.sidebar.divider()
    client, err = llm_client()
    st.sidebar.caption("Configuration")
    st.sidebar.write(f"**LLM provider:** {SETTINGS.llm_provider}  \n**Model:** `{SETTINGS.llm_model}`")
    if client is None:
        st.sidebar.error("LLM key not configured")
    else:
        st.sidebar.success("LLM key configured")
    st.sidebar.write(f"**Review threshold:** {SETTINGS.review_threshold:.2f}")
    st.sidebar.caption("Keys are read from the environment / `.env` and are never displayed.")
    return page


# --------------------------------------------------------------------------- Question 2


@st.cache_resource(show_spinner=False)
def _load_workbook(data: bytes | None, name: str):
    if data is None:
        return load_inventory(DEFAULT_WORKBOOK)
    tmp = Path(st.session_state.get("_tmpdir", ".cache")) / "uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / Path(name).name
    path.write_bytes(data)
    return load_inventory(path)


def render_chart(chart: dict[str, Any]) -> None:
    data: pd.DataFrame = chart["data"]
    x, y, kind = chart["x"], chart["y"], chart["type"]
    title = chart.get("title") or ""
    base = alt.Chart(data).properties(title=title, height=380)
    if kind == "bar":
        c = base.mark_bar().encode(x=alt.X(f"{x}:N", sort="-y"), y=alt.Y(f"{y}:Q"), tooltip=[x, y])
    elif kind == "barh":
        c = base.mark_bar().encode(y=alt.Y(f"{x}:N", sort="-x"), x=alt.X(f"{y}:Q"), tooltip=[x, y])
    elif kind == "line":
        c = base.mark_line(point=True).encode(x=f"{x}", y=f"{y}:Q", tooltip=[x, y])
    elif kind == "pie":
        c = base.mark_arc().encode(theta=f"{y}:Q", color=f"{x}:N", tooltip=[x, y])
    else:
        c = base.mark_circle(size=80).encode(x=f"{x}:Q", y=f"{y}:Q", tooltip=list(data.columns[:4]))
    st.altair_chart(c, width="stretch")


def render_turn(turn) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        tags = []
        if turn.execution is not None:
            tags.append("📊 workbook data")
        if turn.search is not None:
            tags.append("🌐 web context")
        if tags:
            st.caption(" · ".join(tags))
        st.markdown(turn.answer)
        if turn.execution is not None and turn.execution.ok:
            res = turn.execution.result
            if isinstance(res, pd.DataFrame) and not res.empty:
                st.dataframe(res, hide_index=True)
            elif isinstance(res, pd.Series) and not res.empty:
                st.dataframe(res.to_frame())
            if turn.chart is not None:
                render_chart(turn.chart)
            for w in turn.execution.warnings:
                st.caption(f"⚠️ {w}")
        if turn.execution is not None and not turn.execution.ok:
            st.error(turn.execution.error)
        if turn.code:
            with st.expander("Generated pandas code" + (" (auto-repaired once)" if turn.repaired else "")):
                st.code(turn.code, language="python")
        if turn.search_error:
            st.warning(f"Web search unavailable: {turn.search_error}")


def inventory_page() -> None:
    st.header("Inventory Agent")
    st.caption(
        "Question 2 · ask questions about the inventory workbook; answers come from sandboxed pandas code, with web search for external context."
    )

    upload = st.file_uploader("Use a different workbook (optional, .xlsx)", type=["xlsx"], key="wb_upload")
    try:
        data = _load_workbook(upload.getvalue() if upload else None, upload.name if upload else DEFAULT_WORKBOOK.name)
    except (WorkbookError, ValueError) as exc:
        st.error(f"Could not load the workbook: {exc}")
        return
    prof = profile(data)
    cols = st.columns(5)
    cols[0].metric("Product records", prof["records"])
    cols[1].metric("Header row (Excel)", prof["header_row"])
    cols[2].metric("Total Hand-In-Stock", f"{prof.get('total_hand_in_stock', 0):,}")
    cols[3].metric("Total cost value (USD)", f"{prof.get('total_cost_price_usd', 0):,.0f}")
    cols[4].metric("Stock inconsistencies", prof.get("stock_inconsistencies"))
    with st.expander("Workbook status, column mapping and data preview"):
        st.write(
            f"Sheet **{data.sheet_name}** · header detected on row **{data.header_row_excel}**, data from row **{data.first_data_row_excel}** · file `{Path(data.source_path).name}`"
        )
        st.dataframe(
            pd.DataFrame({"normalised column": list(data.column_map), "original header": list(data.column_map.values())}), hide_index=True
        )
        st.dataframe(data.df.head(10), hide_index=True)
        inc = stock_consistency(data.df)
        st.caption("Rows where Opening + Stock in − Units sold ≠ stated Hand-In-Stock (source values are not modified):")
        st.dataframe(inc, hide_index=True)

    client, err = llm_client()
    top = st.columns([6, 1])
    if top[1].button("Reset conversation", width="stretch"):
        st.session_state.pop("agent", None)
        st.rerun()

    if client is None:
        st.warning("The chat agent needs an LLM key. " + (err or ""))
        sandbox_playground(data)
        return

    agent: InventoryAgent | None = st.session_state.get("agent")
    if agent is None or agent.data is not data:
        try:
            provider = build_search_provider(SETTINGS.search_provider, client)
        except SearchError as exc:
            provider = None
            st.info(f"Web search disabled: {exc}")
        agent = InventoryAgent(
            data=data, llm=client, search_provider=provider, timeout_s=SETTINGS.exec_timeout_s, max_output_chars=SETTINGS.max_output_chars
        )
        st.session_state["agent"] = agent
    st.caption(f"Search provider: **{agent.search_provider.name if agent.search_provider else 'none'}**")

    for turn in agent.history:
        render_turn(turn)
    question = st.chat_input("Ask about the inventory, e.g. 'Which five products have the highest stock on hand?'")
    if question:
        with st.spinner("Planning, running sandboxed code and summarising…"):
            turn = agent.ask(question)
        render_turn(turn)
    with st.expander("Sandbox playground (run your own pandas code safely)"):
        sandbox_playground(data, nested=True)


def sandbox_playground(data, nested: bool = False) -> None:
    if not nested:
        st.subheader("Sandbox playground")
        st.caption("Without an LLM key you can still query the workbook with your own pandas code in the restricted sandbox.")
    code = st.text_area(
        "Code (assign the answer to `result`)",
        value='result = df.nlargest(5, "hand_in_stock")[["product_name", "hand_in_stock"]]',
        key=f"pg_{nested}",
    )
    if st.button("Run in sandbox", key=f"pg_btn_{nested}"):
        res = run_code(code, data.df, SETTINGS.exec_timeout_s, SETTINGS.max_output_chars)
        if res.ok:
            st.write(res.result if not isinstance(res.result, pd.DataFrame) else None)
            if isinstance(res.result, pd.DataFrame):
                st.dataframe(res.result, hide_index=True)
            if res.chart:
                render_chart(res.chart)
        else:
            st.error(res.error)


# --------------------------------------------------------------------------- Question 3


def _table_height(n_rows: int, max_rows: int = 20) -> int:
    """Pixel height that shows up to ``max_rows`` rows without an inner scrollbar."""
    return 35 * (min(max(n_rows, 1), max_rows) + 1) + 3


def confidence_badge(conf: float, review: bool) -> str:
    if review:
        return f"🟠 {conf:.2f}" if conf >= 0.5 else f"🔴 {conf:.2f}"
    return f"🟢 {conf:.2f}"


def field_table(doc: DocumentResult, show_full: bool) -> pd.DataFrame:
    rows = []
    for name, f in doc.fields.items():
        shown = f.normalized_value if f.normalized_value is not None else f.raw_value
        if not show_full:
            shown = mask_field(name, shown)
        rows.append(
            {
                "field": name,
                "value": shown if shown else "— (null)",
                "confidence": confidence_badge(f.confidence, f.review_required),
                "handwritten": "✍️" if f.handwritten else "",
                "methods": ", ".join(f.method),
                "review": "REVIEW" if f.review_required else "auto-accept",
                "reasons": ("; ".join(f.review_reasons) if show_full else mask_numbers_in_text("; ".join(f.review_reasons)))[:300],
            }
        )
    return pd.DataFrame(rows)


def _load_saved() -> list[DocumentResult]:
    data = json.loads(SAVED_RESULTS.read_text(encoding="utf-8"))
    return [DocumentResult.model_validate(d) for d in data["documents"]]


def document_page() -> None:
    st.header("Document Pipeline")
    st.caption(
        "Question 3 · classify identity & insurance documents, extract fields with per-field confidence, and route uncertain results to human review."
    )
    client, err = llm_client()
    if client is None:
        st.warning(
            "Vision extraction needs an LLM key. "
            + (err or "")
            + " Without it, the pipeline only runs OCR keyword classification and marks every field as missing."
        )

    c1, c2 = st.columns([3, 2])
    uploads = c1.file_uploader("Upload documents (PNG, JPG/JPEG, PDF)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
    use_samples = c2.checkbox("Use the supplied sample documents", value=not uploads)
    run = c2.button("Process batch", type="primary", width="stretch")
    if SAVED_RESULTS.exists() and c2.button("Load last saved batch results", width="stretch"):
        st.session_state["q3_docs"] = _load_saved()
        st.session_state["q3_sources"] = {}
        st.session_state["q3_errors"] = {}

    if run:
        files: list[tuple[str, bytes]] = [(u.name, u.getvalue()) for u in uploads or []]
        if use_samples:
            files += [
                (p.name, p.read_bytes())
                for p in sorted(SAMPLE_DOC_DIR.glob("*"))
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}
            ]
        if not files and use_samples:
            st.error(
                "No sample documents found in `data/documents/`. Copy the supplied PNG/JPG/PDF files there "
                "(they are kept out of Git on purpose) or upload them above."
            )
        elif not files:
            st.error("Upload at least one file or tick 'Use the supplied sample documents'.")
        else:
            docs: list[DocumentResult] = []
            errors: dict[str, str] = {}
            sources: dict[str, bytes] = {}
            bar = st.progress(0.0, text="Starting…")
            for i, (name, data) in enumerate(files):
                bar.progress(i / len(files), text=f"Processing {name} ({i + 1}/{len(files)})")
                try:
                    docs.extend(process_one(name, data, client, SETTINGS.review_threshold, SETTINGS.pdf_dpi))
                    sources[name] = data
                except IngestError as exc:
                    errors[name] = str(exc)
                except LLMError as exc:
                    errors[name] = f"vision model error: {exc}"
            add_cross_document_hints(docs)
            bar.progress(1.0, text=f"Processed {len(files)} file(s) into {len(docs)} logical document(s)")
            st.session_state.update(q3_docs=docs, q3_errors=errors, q3_sources=sources)

    docs = st.session_state.get("q3_docs")
    if not docs:
        st.info("Choose files and press **Process batch**, or load the last saved results.")
        return
    for name, msg in (st.session_state.get("q3_errors") or {}).items():
        st.error(f"{name}: {msg}")
    show_full = st.toggle("Show unmasked values (local evaluation only)", value=False)

    tab_summary, tab_fields, tab_review, tab_download = st.tabs(
        ["Classification summary", "Field results", "Human-review queue", "Downloads"]
    )
    with tab_summary:
        summary = pd.DataFrame(summary_rows(docs))
        m = st.columns(4)
        m[0].metric("Logical documents", len(docs))
        m[1].metric("Supported types found", int((summary["document_type"] != "unknown_or_other").sum()))
        m[2].metric("Documents needing review", int(summary["document_review_required"].sum()))
        m[3].metric("Fields auto-accepted", f"{int(summary['fields_auto_accepted'].sum())}/{int(summary['fields_total'].sum())}")
        view = summary[
            [
                "source_file",
                "pages",
                "document_type",
                "classification_confidence",
                "fields_total",
                "fields_auto_accepted",
                "fields_for_review",
                "mean_field_confidence",
                "document_review_required",
            ]
        ]
        st.dataframe(
            view,
            hide_index=True,
            height=_table_height(len(view)),
            column_config={
                "source_file": st.column_config.TextColumn("source file", width="medium"),
                "document_type": st.column_config.TextColumn("predicted type", width="medium"),
                "classification_confidence": st.column_config.NumberColumn("type conf.", format="%.2f"),
                "fields_total": st.column_config.NumberColumn("fields"),
                "fields_auto_accepted": st.column_config.NumberColumn("auto-accepted"),
                "fields_for_review": st.column_config.NumberColumn("for review"),
                "mean_field_confidence": st.column_config.NumberColumn("mean field conf.", format="%.2f"),
                "document_review_required": st.column_config.CheckboxColumn("needs review"),
            },
        )
    with tab_fields:
        only_hw = st.checkbox("Show handwritten-field documents only", value=False)
        for doc in docs:
            if only_hw and not any(f.handwritten for f in doc.fields.values()):
                continue
            title = (
                f"{doc.source_file} · pages {doc.pages} · {doc.classification.document_type.value} ({doc.classification.confidence:.2f})"
            )
            with st.expander(("🔎 " if doc.document_review_required else "✅ ") + title, expanded=False):
                if doc.fields:
                    st.dataframe(field_table(doc, show_full), hide_index=True, height=_table_height(len(doc.fields)))
                for w in doc.warnings:
                    st.caption(f"• {w if show_full else mask_numbers_in_text(w)}")
                preview(doc, show_full)
    with tab_review:
        flags = pd.DataFrame(flag_rows(docs), columns=FLAG_COLUMNS)
        if not show_full and not flags.empty:
            flags["raw_candidate"] = [
                mask_field(f, v) if f in SENSITIVE_FIELDS else mask_numbers_in_text(v)
                for f, v in zip(flags["field_name"], flags["raw_candidate"].astype(str))
            ]
            flags["review_reason"] = [mask_numbers_in_text(r) for r in flags["review_reason"].astype(str)]
        st.write(f"**{len(flags)}** items below the {SETTINGS.review_threshold:.2f} threshold or otherwise flagged.")
        st.dataframe(
            flags,
            hide_index=True,
            height=_table_height(len(flags), max_rows=18),
            column_order=[
                "source_file",
                "field_name",
                "raw_candidate",
                "confidence",
                "review_reason",
                "suggested_reviewer_action",
                "predicted_document_type",
                "page",
                "threshold",
                "document_id",
            ],
            column_config={
                "source_file": st.column_config.TextColumn("source file", width="medium"),
                "field_name": st.column_config.TextColumn("field", width="medium"),
                "raw_candidate": st.column_config.TextColumn("raw candidate", width="small"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
                "review_reason": st.column_config.TextColumn("review reason", width="large"),
                "suggested_reviewer_action": st.column_config.TextColumn("suggested action", width="large"),
                "predicted_document_type": st.column_config.TextColumn("predicted type", width="medium"),
            },
        )
    with tab_download:
        st.caption("Downloads contain unmasked values for local evaluation. Handle them as sensitive data.")
        payload = json.dumps(batch_json(docs, SETTINGS.review_threshold, st.session_state.get("q3_errors")), indent=2, ensure_ascii=False)
        flags_all = flag_rows(docs)
        d = st.columns(5)
        d[0].download_button("all_results.json", payload, "all_results.json", "application/json", width="stretch")
        d[1].download_button("flagging_report.csv", to_csv(flags_all, FLAG_COLUMNS), "flagging_report.csv", "text/csv", width="stretch")
        d[2].download_button(
            "flagging_report.json",
            json.dumps(flagging_json(flags_all, SETTINGS.review_threshold), indent=2, ensure_ascii=False),
            "flagging_report.json",
            "application/json",
            width="stretch",
        )
        d[3].download_button(
            "extraction_summary.csv", to_csv(summary_rows(docs), SUMMARY_COLUMNS), "extraction_summary.csv", "text/csv", width="stretch"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for doc in docs:
                zf.writestr(f"json/{doc.document_id}.json", json.dumps(document_json(doc), indent=2, ensure_ascii=False))
        d[4].download_button("per-document JSON (.zip)", buf.getvalue(), "documents_json.zip", "application/zip", width="stretch")
        with st.expander("Preview: all_results.json" + (" (unmasked)" if show_full else " (masked)")):
            st.json(_masked_json(json.loads(payload)) if not show_full else json.loads(payload), expanded=3)


def _masked_json(data: Any) -> Any:
    """Masked copy for on-screen preview: identifiers in values, candidates, evidence and reasons."""
    for doc in data.get("documents", []):
        doc["warnings"] = [mask_numbers_in_text(w) for w in doc.get("warnings", [])]
        for group in ("fields", "auxiliary_fields"):
            for name, f in (doc.get(group) or {}).items():
                mask = (lambda v, n=name: mask_field(n, v)) if name in SENSITIVE_FIELDS else mask_numbers_in_text
                for key in ("raw_value", "normalized_value"):
                    f[key] = mask(f.get(key)) if f.get(key) else f.get(key)
                f["candidates"] = [mask(c) for c in f.get("candidates", [])]
                f["evidence"] = mask_numbers_in_text(f.get("evidence"))
                f["review_reasons"] = [mask_numbers_in_text(r) for r in f.get("review_reasons", [])]
    return data


def _blur_sensitive_regions(img, doc: DocumentResult, page: int) -> None:
    """Blur (in place) the located regions of sensitive fields, with a margin for approximate boxes."""
    for name, f in doc.fields.items():
        if name not in SENSITIVE_FIELDS or not f.region or f.page != page:
            continue
        x0, y0, x1, y1 = f.region
        pad = max(6, int(0.25 * (y1 - y0)))
        box = (max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))
        if box[2] > box[0] and box[3] > box[1]:
            img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(radius=12)), box[:2])


def preview(doc: DocumentResult, show_full: bool) -> None:
    sources: dict[str, bytes] = st.session_state.get("q3_sources") or {}
    data = sources.get(doc.source_file)
    if data is None:
        path = SAMPLE_DOC_DIR / doc.source_file
        data = path.read_bytes() if path.exists() else None
    if data is None:
        return
    if not st.checkbox("Show document preview with extraction regions", key=f"pv_{doc.document_id}"):
        return
    src = load_source(doc.source_file, data, pdf_dpi=SETTINGS.pdf_dpi)
    for p in doc.pages:
        img = src.pages[p - 1].processed.copy()
        if not show_full:
            _blur_sensitive_regions(img, doc, p)
        draw = ImageDraw.Draw(img)
        for name, f in doc.fields.items():
            if f.region and f.page == p:
                colour = (220, 120, 0) if f.review_required else (0, 150, 60)
                draw.rectangle(f.region, outline=colour, width=4)
                draw.text((f.region[0] + 4, max(0, f.region[1] - 14)), name, fill=colour)
        if not show_full:
            st.caption(
                "Regions of identity/financial-number fields are blurred. The same number may also be printed elsewhere "
                "on the page, so keep previews off when sharing your screen."
            )
        st.image(img, caption=f"{doc.source_file} · page {p} (processed)", width="stretch")


# --------------------------------------------------------------------------- about


def about_page() -> None:
    st.header("Setup & About")
    st.markdown(
        f"""
**Numbering note.** This submission maps **Question 2 → inventory Excel conversational agent** and
**Question 3 → document classification & field extraction**, as instructed (the Word file's numbering differs).

**Configuration status**
- LLM provider: `{SETTINGS.llm_provider}` · model `{SETTINGS.llm_model}` · key present: **{SETTINGS.llm_available}**
- Search provider setting: `{SETTINGS.search_provider}` (Tavily key present: **{SETTINGS.tavily_available}**)
- Review threshold: **{SETTINGS.review_threshold}** (env `REVIEW_THRESHOLD`)

**Setup**: copy `.env.example` to `.env`, set the key for your provider, restart `streamlit run app.py`.
See README.md for full instructions, methodology and limitations.
"""
    )


def main() -> None:
    page = sidebar()
    if page.startswith("Inventory"):
        inventory_page()
    elif page.startswith("Document"):
        document_page()
    else:
        about_page()


main()
