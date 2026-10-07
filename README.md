# InsightExtract AI – Inventory Agent & Document Intelligence

One Streamlit application with two independent solutions for the DesiCrew evaluation:

* **Question 2 – Inventory conversational agent.** Chat with `Inventory-Records-Sample-Data.xlsx`.
  The agent writes pandas code, runs it in a locked-down sandbox, uses web search for external
  definitions, and answers in plain English, keeping workbook facts and web context apart.
* **Question 3 – Document classification & field extraction.** Batch-process identity and insurance
  documents (PNG/JPG/PDF), classify each logical document, extract the required fields with a
  dedicated handwriting strategy, score every field, and route anything uncertain to a human-review
  queue with JSON and CSV reports.

> **Numbering note.** The assignment Word file numbers the tasks differently. As instructed for this
> submission, **Question 2 = inventory Excel conversational agent** and **Question 3 = document
> classification and field-extraction pipeline**. The separate "document-aware support assistant"
> task in the Word file is intentionally not implemented.

---

## Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Setup (Windows, macOS, Linux)](#setup)
4. [Environment variables](#environment-variables)
5. [Running the app, scripts and tests](#running)
6. [Inputs and outputs](#inputs-and-outputs)
7. [Question 3 confidence methodology](#confidence-methodology)
8. [Handwriting approach](#handwriting-approach)
9. [Security and privacy](#security-and-privacy)
10. [Results](#results)
11. [Known limitations](#known-limitations)
12. [Troubleshooting](#troubleshooting)
13. [Screenshots](#screenshots)

---

## Architecture

```
app.py                         Streamlit UI (Inventory Agent · Document Pipeline · Setup & About)
src/common/
  config.py                    settings from env/.env (provider, model, threshold, limits)
  llm.py                       thin provider interface: Anthropic (default, SDK), Gemini (REST), OpenAI (REST)
                               complete_json (schema-constrained) · complete_text · web_search · disk cache
  privacy.py                   masking for the UI, PII scrubbing for search queries
src/question2_inventory/
  loader.py                    header-row detection, column normalisation + original-name map, profile,
                               stock-consistency check
  sandbox.py                   AST allow/deny validation + separate process with timeout, CPU/memory limits,
                               output caps
  search.py                    Tavily · LLM-native search (Claude web search / Gemini Google Search) · DuckDuckGo
  agent.py                     plan (JSON) → sandboxed code and/or web search → plain-English summary
src/question3_documents/
  schemas.py                   10 document types, field specs (labels, kind, validator), Pydantic output models
  imaging.py                   PDF→images, orientation (Tesseract OSD), deskew, CLAHE, quality metrics, crops
  ocr.py                       Tesseract words, printed-label anchoring, keyword classifier, crop OCR
  extract.py                   classification + page grouping, full-page pass, crop passes, consensus
  validators.py                Aadhaar/Verhoeff, PAN, IFSC, account, amount, dates, DL, passport, MRZ, checkbox
  confidence.py                weighted evidence score + hard caps
  pipeline.py                  batch orchestration, cross-field checks, cross-document hints
  report.py                    per-document JSON, all_results.json, flagging report, summary CSV
scripts/                       run_question2_demo.py · run_question3.py · capture_screenshots.py
tests/                         pytest suite (LLM calls mocked) + opt-in live integration tests
```

**Why this shape.** Both questions use direct provider SDK/REST calls with JSON-schema-constrained
outputs instead of an agent framework: the control flow (plan → execute → summarise; classify →
read → re-read → score) is explicit Python, which keeps it testable with a scripted fake LLM.

### Question 2 flow

1. `loader.load_inventory` scans the sheet for the first row with ≥3 text cells followed by a numeric
   row (header = Excel row 6, data from row 7), normalises headers (`"Hand-In-\nStock"` →
   `hand_in_stock`) and keeps a map back to the original labels.
2. The **planner** call receives the column schema, a profile, the product list and the last five turns,
   and returns `{intent, code, search_query, message}` with intent `data`, `search`,
   `data_and_search`, `clarify` or `out_of_scope`.
3. Code runs in the **sandbox**; if it fails, the sanitised error is sent back for one repair attempt.
4. Searches go through `safe_search` (PII scrubbed, non-HTTP URLs dropped).
5. The **summary** call may only use numbers present in the execution result and must split the
   answer into "From the workbook" and "External context (web)" with `[n]` citations; the numbered,
   clickable source list is appended by code, not by the model.

### Question 3 flow

See [outputs/question3/methodology.md](outputs/question3/methodology.md) for the full description.
In short: ingest → orientation/deskew/contrast → OCR → classify & group pages (vision model + OCR
keyword cross-check) → full-page read → crop reads (two independent ones for handwriting) → OCR of
the crop → validators → consensus & confidence → cross-field checks → reports.

---

## Features

**Inventory agent (Q2)**
- Automatic workbook loading with header detection and column mapping; upload another workbook in the UI.
- Workbook profile (records, header row, totals, inconsistency count) and a data/column-map preview.
- Chat with session history, follow-ups, clarifying questions, graceful handling of unknown products,
  empty results and out-of-scope or unsafe requests.
- Generated pandas code in a collapsible section; tables and Altair charts (bar, horizontal bar, line, pie, scatter).
- Web search with clickable, numbered sources, clearly separated from workbook facts.
- Stated `Hand-In-Stock` values are never altered; the 12 arithmetic discrepancies are reported, not "fixed".
- A sandbox playground that works even without an LLM key.
- Reset-conversation button.

**Document pipeline (Q3)**
- Multi-file upload (PNG, JPG/JPEG, PDF) or one click to process the supplied samples.
- PDF rasterisation, orientation detection, deskew, contrast enhancement; originals untouched.
- Content-based classification (file names are never shown to the model) with multi-page grouping.
- Field extraction for all 10 document types; per-field confidence, method trail, evidence, review reasons.
- Handwriting: targeted crops, two independent crop reads with different preprocessing, OCR comparison.
- Validators for every high-risk format, MRZ check digits and cross-field consistency.
- Human-review queue with suggested reviewer actions; unsupported documents always queued.
- Masked sensitive values by default (toggle to reveal locally); document preview with extraction regions.
- Downloads: `all_results.json`, flagging CSV, summary CSV, per-document JSON zip.

---

## Setup

Python **3.11 or newer** (developed and tested on 3.13). Tesseract is optional but recommended
(orientation detection, label anchoring, OCR agreement, keyword classification).

The supplied Question 3 documents are identity documents and are not stored in Git. After cloning, copy
the twelve supplied files (ten images and the two Ashok PDFs) into `data/documents/`; the list is in
[data/documents/README.md](data/documents/README.md). The inventory workbook is already in `data/inventory/`.

### Linux (Debian/Ubuntu)

```bash
git clone <your-repo-url> evaluation-solution && cd evaluation-solution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y tesseract-ocr          # optional, recommended
cp .env.example .env                           # then edit .env and set your key
```

### macOS

```bash
git clone <your-repo-url> evaluation-solution && cd evaluation-solution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract                         # optional, recommended
cp .env.example .env
```

### Windows (PowerShell)

```powershell
git clone <your-repo-url> evaluation-solution; cd evaluation-solution
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# optional: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH
Copy-Item .env.example .env
```

On Windows the sandbox relies on the wall-clock timeout (POSIX CPU/memory limits are unavailable).

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `gemini` or `openai` |
| `LLM_MODEL` | provider default | `claude-opus-5-5`, `gemini-3.6-flash`, `gpt-4.1` (Gemini Pro models need a paid-tier key) |
| `LLM_EFFORT` | `medium` | Claude effort level (`low`…`max`) |
| `LLM_FALLBACKS` | `default` | Claude server-side refusal fallback; `off` to disable |
| `LLM_FALLBACK_MODELS` | – | Gemini: comma-separated models tried when the main one is overloaded or out of daily quota |
| `LLM_MAX_RPM` | `0` | Gemini: requests per minute per model (0 = unpaced; about 8 suits a free-tier key) |
| `LLM_MAX_RETRIES` | `5` | Gemini: retries for HTTP 429/5xx/network errors, with backoff and the server's `retryDelay` |
| `LLM_CACHE` | `on` | cache structured responses in `.cache/llm` (git-ignored) |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` | – | key for the chosen provider |
| `SEARCH_PROVIDER` | `auto` | `auto`, `tavily`, `llm`, `duckduckgo` |
| `TAVILY_API_KEY` | – | optional Tavily key |
| `REVIEW_THRESHOLD` | `0.85` | field auto-accept threshold |
| `CROP_READS` | `per_field` | Question 3: `per_field` (one request per crop) or `batched` (about 4 requests per document, for rate-limited keys) |
| `EXEC_TIMEOUT_SECONDS` | `10` | sandbox wall-clock timeout |
| `MAX_OUTPUT_CHARS` | `20000` | sandbox output cap |
| `PDF_DPI` | `200` | PDF rasterisation resolution |

`SEARCH_PROVIDER=auto` uses Tavily when a key is set, otherwise the LLM's own grounded search
(Claude web search or Gemini Google Search) and falls back to DuckDuckGo (key-free, via `ddgs`) when
that fails, for example on a free-tier Gemini key, which does not include Google Search grounding.
Each answer records which provider actually answered.

The app starts without any key: the Inventory page then offers the sandbox playground and the
Document page runs OCR-keyword classification only, marking every field as missing, with setup
guidance shown in both places. It never fabricates AI output.

---

## Running

```bash
streamlit run app.py                         # UI at http://localhost:8501
python scripts/run_question2_demo.py         # writes outputs/question2/demo_transcript.md (+ chart PNG)
python scripts/run_question3.py              # processes data/documents → outputs/question3/
python -m pytest                             # unit tests (all external calls mocked)
RUN_LIVE_TESTS=1 python -m pytest tests/test_integration_live.py   # live smoke test (needs a key)
python scripts/capture_screenshots.py        # Playwright screenshots of the running app
```

---

## Inputs and outputs

| Path | Content |
|---|---|
| `data/inventory/Inventory-Records-Sample-Data.xlsx` | Q2 workbook (copy of the supplied file) |
| `data/documents/` | Q3 input folder: copy the supplied documents here (git-ignored, see the privacy note below) |
| `outputs/question2/demo_transcript.md` | live transcript of the demonstration questions |
| `outputs/question3/json/<document-id>.json` | one JSON per logical document |
| `outputs/question3/all_results.json` | all documents + threshold and rationale |
| `outputs/question3/flagging_report.csv` / `.json` | human-review queue |
| `outputs/question3/extraction_summary.csv` | per-document summary |
| `outputs/question3/methodology.md` | classification, handwriting, confidence and threshold method |
| `outputs/question3/failure_cases.md` | observed failure cases and limitations |

To process your own files, put them in `data/documents/` (or pass `--input <folder>` to
`scripts/run_question3.py`), or upload them in the UI.

---

## Confidence methodology

Each field gets a deterministic score from six evidence components: agreement between independent
vision reads (0.35), format/checksum validation (0.20), legibility minus ambiguous characters (0.15),
OCR support (0.10; 0.05 for handwriting), proximity of the value to its printed label (0.10) and page
image quality (0.10). Missing components are left out and the weights renormalised. Hard caps then
apply: disagreement without a majority → value withheld and ≤ 0.40; a pass found nothing → ≤ 0.75;
failed validation → ≤ 0.70; confusable characters normalised → ≤ 0.84; handwriting with fewer than
two reads → ≤ 0.80; cross-field conflict → ≤ 0.60; missing → 0.0. Every field's JSON includes its
`confidence_components`.

**Threshold 0.85.** These are high-risk identity and financial fields: a wrong account number, IFSC
or TIN that is auto-accepted costs far more than a reviewer's glance, so false acceptance is treated
as the worse error. Under the scoring above, 0.85 is reachable only when independent reads agree, the
value is legible, sits next to its label and passes its rule. Change it with `REVIEW_THRESHOLD`.

Full details: [outputs/question3/methodology.md](outputs/question3/methodology.md).

## Handwriting approach

Handwritten fields (all fields on the six insurance forms) are read differently from printed fields:

1. The full page is preprocessed (deskew, CLAHE contrast, light denoise) without touching the original.
2. The full-page vision pass returns each value's bounding box; Tesseract independently locates the
   printed label (e.g. "IFSC", "Place of birth"), and the crop is built from both.
3. Two **independent** crop reads run on differently preprocessed images: a colour, contrast-enhanced
   upscale, and a binarised, sharpened upscale with a character-by-character prompt. Neither sees
   another pass's answer. With `CROP_READS=batched` (used for the committed outputs, because the
   free-tier Gemini key allowed 20 requests per model per day) all colour crops of a form go in one
   request and all binarised crops in another, each crop labelled with its field.
4. Tesseract OCR of the crop is compared too, at half weight because it is weak on handwriting.
5. Reads are compared with format-aware keys (IFSC/PAN/account/date rules), confusable characters
   (0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G) are normalised by position and flagged, and disagreement lowers
   confidence. With no majority the value becomes `null` with all candidates listed for the reviewer.
6. Checkboxes are read as "which printed option is ticked" twice; struck-through options don't count,
   and anything other than exactly one selection is flagged.
7. Values are never copied from another document; matches elsewhere in the batch appear only as
   review hints.

## Security and privacy

- **Sandbox:** AST validation blocks imports, dunder access, `eval`/`exec`/`open`/`getattr`/`globals`,
  file/IO/network methods (`to_csv`, `read_*`, `load`, `savefig`…), `str.format`, `query`/`eval` string
  evaluation, try/except, classes and module aliasing. `pd`/`np` attributes are allow-listed. Code runs
  in a separate process (forkserver) with restricted builtins, a wall-clock timeout, CPU-time and
  address-space limits, a 200-row/20 000-character output cap and sanitised error messages (no paths
  or tracebacks).
- **Secrets:** keys come only from the environment or `.env` (git-ignored); the UI shows only whether a
  key is present. No key appears in code, logs, screenshots or Git history.
- **PII:** identity and financial numbers are masked in the UI by default; search queries are scrubbed
  of Aadhaar/PAN/IFSC/account/phone/email patterns; logs record error types, not document content.
- **Local evaluation artefacts:** the JSON/CSV outputs under `outputs/question3/` contain full values,
  because the evaluation compares them with an answer key. Treat them as sensitive.
- **Source documents:** the supplied identity documents look like sample/demo documents but are still
  treated as sensitive. `data/documents/` is git-ignored, so they never enter Git history. The evaluation
  outputs in `outputs/question3/` do contain full extracted values: push the repository to a **private**
  GitHub repository, or remove `outputs/question3/json/`, `all_results.json` and the CSV/JSON reports before
  making anything public.
- **Caches:** `.cache/` (LLM response cache, temporary uploads) is git-ignored.

## Results

See [SUBMISSION.md](SUBMISSION.md) for the test results, the sample-processing results and the
screenshot index.

## Known limitations

- Extraction quality depends on the configured vision model; outputs were produced with the model
  named in `SUBMISSION.md` and will differ with other providers/models.
- The 0.85 threshold and the component weights are reasoned defaults, not calibrated on a labelled set;
  with an answer key they should be tuned (e.g. choose the threshold that gives a target precision).
- Bounding boxes from the vision model are approximate; label anchoring needs Tesseract. Without
  Tesseract, crops fall back to the model's boxes and confidence loses the OCR/proximity evidence.
- Tesseract OCR is used as weak evidence for handwriting; it often fails on cursive text.
- Only Indian document formats are validated (Indian passport number pattern, Indian DL format).
- The keyword classifier only knows English printed phrases.
- The sandbox's CPU/memory limits are POSIX-only; on Windows only the timeout applies.
- Web search needs network access to the chosen provider; DuckDuckGo results are snippets only.
  Free-tier Gemini keys have no Google Search grounding, so with such a key search relies on the
  DuckDuckGo fallback, which needs ordinary internet access.
- The committed outputs were produced with a free-tier Gemini key: Flash models only, 20 requests per
  model per day and 5 per minute (failed "high demand" requests count too). The run therefore used
  batched crop reads and spread calls over several Flash models; `all_results.json` → `run` lists how
  many calls each model answered. A paid key with `CROP_READS=per_field` and a Pro model gives more
  independent reads per field.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No API key found for LLM provider" | create `.env` from `.env.example`, set the key, restart Streamlit |
| `TesseractNotFoundError` / OCR warnings | install Tesseract and make sure `tesseract` is on PATH |
| Web search unavailable | set `TAVILY_API_KEY`, or use Anthropic/Gemini (`SEARCH_PROVIDER=llm`), or `pip install ddgs` |
| "Blocked by safety check" | the generated code used a forbidden construct; rephrase the question |
| Sandbox "CPU-time or memory limit exceeded" | the query was too heavy; narrow it |
| PDF errors | the PDF is encrypted or corrupt; export it again or convert pages to PNG |
| Stale results after changing prompts | set `LLM_CACHE=off` or delete `.cache/llm` |
| Gemini `HTTP 429: quota exceeded: …PerDay…` | the free tier allows 20 requests per model per day: set `LLM_FALLBACK_MODELS`, `CROP_READS=batched` and `LLM_MAX_RPM=4.5`, wait for the daily reset, or enable billing; successful responses are cached, so a re-run only repeats the failed calls |
| Gemini `HTTP 503` (high demand) | transient; the client retries with backoff and then tries the fallback models |

## Screenshots

| # | Screenshot |
|---|---|
| 1 | [Inventory chat – numerical answer with generated pandas code](screenshots/01_inventory_numeric_code.png) |
| 2 | [Inventory chat – web search with clickable citations](screenshots/02_inventory_web_search.png) |
| 3 | [Document batch – classification summary](screenshots/03_documents_classification.png) |
| 4 | [Handwritten-field extraction with confidence scores](screenshots/04_handwritten_fields.png) |
| 5 | [Human-review queue](screenshots/05_review_queue.png) |
| 6 | [Downloads – JSON and reports](screenshots/06_downloads.png) |
