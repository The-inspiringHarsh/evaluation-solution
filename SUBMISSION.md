# Submission: DesiCrew evaluation (Questions 2 and 3)

**GitHub repository:** `https://github.com/<your-github-user>/evaluation-solution` (placeholder: replace after pushing; see [Publishing](#publishing))

**Numbering.** Question 2 = inventory Excel conversational agent. Question 3 = document classification and
field-extraction pipeline. The Word file numbers the tasks differently; this mapping was specified for
this submission, and the separate document-aware support assistant is intentionally not implemented.

## Status at a glance

| Area | Status |
|---|---|
| Q2 inventory agent: loader, sandbox, search, agent, UI | Implemented; unit-tested with a mocked LLM |
| Q2 live demo transcript (`outputs/question2/demo_transcript.md`) | Done, live on the free Gemini tier, 7 Oct 2026 (11 questions) |
| Q2 web search | Implemented (Claude, Gemini grounding, Tavily, DuckDuckGo with fallback); **not demonstrated live**: the free Gemini tier has no Google Search grounding and the build environment blocks DuckDuckGo, so the app shows "web search unavailable" |
| Q3 pipeline: ingest, classify, extract, score, report, UI | Implemented; unit-tested with a scripted fake vision model |
| Q3 outputs on the 12 supplied files (`outputs/question3/`) | Done, live: 12 logical documents, 0 file errors, 36 of 44 fields auto-accepted, 10 items queued for review |
| Live integration tests (4) | 3 passed, 1 failed (web search, for the environment reason above) |
| Screenshots (6) | Done, captured from the running app with masking on |
| Automated tests (mocked) | 168 passed, 4 live tests skipped |

All results below come from real runs. Gemini free tier: Flash models only, 20 requests per model per day,
so Question 3 used batched crop reads and Question 2 also fell back to Flash-Lite models (the transcript
footer lists which model answered each call).

## Question 2 checklist: inventory conversational agent

| # | Requirement | Where | Evidence |
|---|---|---|---|
| 1 | Load the workbook automatically | `src/question2_inventory/loader.py` (`load_inventory`), loaded on app start | `test_header_row_detected_on_row_6` |
| 2 | Detect the real header row | `detect_header_row`: first row with ≥3 text cells followed by a numeric row | header = Excel row 6, data from row 7 (test) |
| 3 | Normalise column names, keep the original mapping | `normalize_column_name`, `InventoryData.column_map` (shown in the UI) | `test_column_normalisation_keeps_original_mapping`, `test_normalize_column_name` |
| 4 | Generate pandas code for analytical questions | planner call in `agent.py` (JSON schema: intent, code, search query, message) | `test_numeric_question_runs_code_and_summarises` |
| 5 | Execute that code safely | `sandbox.py`: AST allow/deny lists, restricted builtins, separate process, timeout, CPU/memory limits, output caps, sanitised errors | 26 tests in `test_q2_sandbox.py` (imports, dunders, `eval`/`exec`/`open`, file/network methods, timeout, memory limit, truncation) |
| 6 | Show the generated code in a collapsible section | "Generated pandas code" expander in `app.py` | screenshot 1 |
| 7 | Use web search for external definitions or business context | planner intents `search` / `data_and_search`; `search.py` (Claude web search, Gemini Google Search, Tavily, DuckDuckGo) | `test_search_question_uses_search_and_cites` |
| 8 | Clickable source URLs | numbered `[title](url)` list appended by code, non-HTTP URLs dropped | `test_format_sources_markdown_is_clickable`, `test_safe_search_drops_non_http_sources_and_scrubs` |
| 9 | Concise plain-English summary | summary call restricted to numbers in the execution result | `test_numeric_question_runs_code_and_summarises` |
| 10 | Chat history within the session | `InventoryAgent.history` in `st.session_state`; last five turns go to the planner | `test_follow_up_sees_history` |
| 11 | Follow-ups, ambiguity, unknown products, invalid requests, empty results | intents `clarify` / `out_of_scope`; empty results reported as such; failed code repaired once | `test_clarification_and_out_of_scope`, `test_unknown_product_gives_empty_result`, `test_failed_code_is_repaired_once`, `test_unsafe_generated_code_is_blocked_not_run`, `test_empty_question` |
| 12 | Numbers, filters, ranking, comparisons, aggregation, consistency checks, charts | sandboxed pandas plus a validated chart spec (bar, horizontal bar, line, pie, scatter) rendered with Altair | `test_chart_spec_validated`; demo transcript question 6 (`chart_q6.png`) |
| 13 | Separate workbook facts from web context | answer sections "From the workbook" and "External context (web)" with `[n]` citations | `test_search_question_uses_search_and_cites` |
| – | Never "fix" the stated Hand-In-Stock | `stock_consistency` adds expected/difference columns on a copy | `test_stock_inconsistencies_reported_not_fixed` |
| – | Reset conversation button, workbook profile, tables, charts | `app.py` Inventory page | screenshots 1–2 |
| – | PII never sent to web search | `safe_search` scrubs Aadhaar/PAN/IFSC/account/phone/email patterns | `test_search_query_is_scrubbed_of_pii` |

**Workbook smoke values** (all asserted by `test_workbook_smoke_totals` and
`test_stock_inconsistencies_reported_not_fixed` against the supplied workbook):

| Check | Expected | Loaded |
|---|---|---|
| Header row / first data row | 6 / 7 | 6 / 7 |
| Product records | 46 | 46 |
| Total Hand-In-Stock | 2,004 | 2,004 |
| Total Cost Price Total (USD) | 359,760 | 359,760 |
| Highest stock on hand | Smartphone, 80 | Smartphone, 80 |
| Laptop and Smartphone inventory value | USD 72,000 each | USD 72,000 each |
| Rows where Opening + Stock In − Units Sold ≠ Hand-In-Stock | 12 | 12 (reported, not altered) |

## Question 3 checklist: document classification and extraction

| # | Requirement | Where | Evidence |
|---|---|---|---|
| 1 | Accept PNG, JPG/JPEG and PDF | `imaging.load_source` | `test_png_ingestion`, `test_jpeg_and_rgba_ingestion`, `test_unsupported_and_corrupt_files` |
| 2 | PDFs to page images | PyMuPDF at `PDF_DPI` (200) | `test_multi_page_pdf` |
| 3 | Orientation detection and preprocessing | Tesseract OSD rotation, Hough-line deskew, CLAHE, light denoise, quality metrics; originals untouched | `test_sideways_page_is_rotated_upright`, `test_original_is_not_modified` |
| 4 | Classify each logical document by content | vision classifier sees page images only (no file names), groups pages; OCR keyword vote cross-checks | `test_classification_schema_lists_every_type`, `test_classifier_page_groups_are_sanitised` |
| 5 | Extract the required fields for all ten types | `schemas.DOC_SPECS`, `extract.py` | `test_every_supported_document_schema`, `test_printed_document_fields_are_accepted` |
| 6 | Confidence for every field | `confidence.score`: weighted evidence plus hard caps, components saved per field | 11 tests in `test_q3_confidence.py` |
| 7 | Flag missing, conflicting, unsupported or low-confidence results | `review_required` + `review_reasons`; unsupported documents always queued | `test_missing_field_has_zero_confidence_and_review`, `test_handwritten_disagreement_withholds_value`, `test_unsupported_multipage_pdf_is_one_unknown_document` |
| 8 | Evidence and review reasons | `evidence` (label, page, region), `method`, `candidates`, `review_reasons` | `test_json_serialisation_and_reports` |
| 9 | Structured JSON and a review report | `report.write_outputs` | `test_json_serialisation_and_reports`, `test_csv_helper` |
| 10 | Multi-page PDFs are not split into unrelated documents | classifier page grouping, sanitised so every page is used exactly once | `test_unsupported_multipage_pdf_is_one_unknown_document` |
| – | Handwriting strategy: targeted crops, ≥2 independent reads, OCR comparison, `null` instead of guesses, raw + normalised values, no cross-document copying | `extract.resolve_field`, `pipeline.add_cross_document_hints` | `test_handwritten_disagreement_withholds_value`, `test_handwritten_needs_two_reads` |
| – | Validators: Aadhaar Verhoeff, PAN, IFSC, DL, passport, MRZ check digits, dates, account (leading zeros), amount, checkbox | `validators.py` | 30 tests in `test_q3_validators.py` |
| – | Validation supports confidence, never overwrites the observed value | raw value kept; failure caps confidence at 0.70 and flags | `test_validation_failure_is_preserved_and_flagged` |
| – | Threshold 0.85, configurable | `REVIEW_THRESHOLD` | `test_threshold_behaviour`, `test_threshold_is_configurable` |
| – | The two Ashok PDFs go to `unknown_or_other` and the review queue | classifier prompt treats other insurer forms as unsupported | live test `test_live_pdf_is_unsupported` (passed); both PDFs classified `unknown_or_other` in the outputs |
| – | UI: upload, batch status, classification summary, field results with confidence, review queue, masking, preview, downloads | `app.py` Document page | screenshots 3–6 |

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
sudo apt-get install -y tesseract-ocr                     # optional, recommended (macOS: brew install tesseract)
cp .env.example .env                                      # then set ANTHROPIC_API_KEY (or GEMINI_API_KEY / OPENAI_API_KEY)

streamlit run app.py                                      # UI at http://localhost:8501
python scripts/run_question2_demo.py                      # writes outputs/question2/demo_transcript.md
python scripts/run_question3.py                           # writes outputs/question3/*
python -m pytest                                          # mocked unit tests
RUN_LIVE_TESTS=1 python -m pytest tests/test_integration_live.py   # live smoke tests (needs a key)
python scripts/capture_screenshots.py                     # Playwright screenshots of the running app
```

## Test results

Run on 7 Oct 2026 in a Linux container (Python 3.13.16, Tesseract 5.3.4):

```
$ python -m pytest -rs
166 passed, 4 skipped
SKIPPED [4] tests/test_integration_live.py: set RUN_LIVE_TESTS=1 and an LLM API key to run live integration tests

$ ruff check . && ruff format --check .
All checks passed!
```

Live integration tests (`RUN_LIVE_TESTS=1`, free Gemini key, 7 Oct 2026):

```
PASSED test_live_inventory_count
PASSED test_live_pdf_is_unsupported
PASSED test_live_sample_images_cover_the_ten_types_with_scored_fields
FAILED test_live_search_question_cites_clickable_sources
  search provider llm failed: Gemini API error (HTTP 429 ...)       # free tier: no Google Search grounding
  search provider duckduckgo failed: DuckDuckGo search failed       # blocked by the build environment's network
```

The search test is expected to pass with a Claude or paid Gemini key, a Tavily key, or DuckDuckGo on an
ordinary internet connection.

| Test module | Tests | Covers |
|---|---|---|
| `test_q2_loader.py` | 10 | header detection, column normalisation, smoke totals, inconsistencies |
| `test_q2_sandbox.py` | 26 | blocked constructs, allowed analytics, timeout, memory limit, sanitised errors, truncation, chart specs |
| `test_q2_agent.py` | 11 | code generation and repair, unsafe code, search and citations, unsourced search text, clarification, unknown product, follow-ups |
| `test_q2_search.py` | 8 | clickable source formatting, PII scrubbing, provider selection, search fallback |
| `test_q3_validators.py` | 30 | Aadhaar/Verhoeff, PAN, IFSC, account, amount, dates, DL, passport, MRZ, checkbox, confusables |
| `test_q3_confidence.py` | 11 | weights, caps, missing = 0, threshold behaviour, determinism |
| `test_q3_imaging.py` | 7 | PNG/JPEG/RGBA/PDF ingestion, corrupt files, orientation, originals untouched, crops |
| `test_q3_pipeline.py` | 26 | every schema, missing fields, disagreement, unsupported PDF, page grouping, batched crop reads, failed passes, box order, JSON and reports |
| `test_privacy_config.py` | 8 | masking, PII scrubbing, settings and setup hints |
| `test_llm_providers.py` | 29 | structured-output requests, refusals, truncation, web-search citations, cache, Gemini retries, quotas, fallbacks, per-model temperature, key redaction, model attribution |
| `test_integration_live.py` | 4 | live: inventory count, search citations, ten-type coverage of the sample images, Ashok PDF unsupported |

## Sample-processing results

**Question 2** (`outputs/question2/demo_transcript.md`, one continuous session):

| # | Question | Result |
|---|---|---|
| 1 | How many products are in the inventory? | 46 (`df["product_id"].nunique()`) |
| 2 | Five products with the highest stock | Smartphone 80, then four more, with code and table |
| 3 | Total current inventory value | USD 359,760 |
| 4 | Stock-calculation inconsistencies | the 12 rows, reported and not altered |
| 5 | Fewer than 30 units in stock | filtered table |
| 6 | Chart of the ten most valuable products | bar chart (`chart_q6.png`) |
| 7 | Inventory turnover meaning and feasibility | workbook part answered; web search unavailable (see above) |
| 8 | Follow-up: lowest stock among those ten | Gaming Monitor, 29 units (uses chat history) |
| 9 | Hoverboards (unknown product) | "nothing matched", no invented number |
| 10 | "Which is the best one?" (ambiguous) | asks which metric to use |
| 11 | Delete the workbook / show environment variables | refused as out of scope |

**Question 3** (`outputs/question3/extraction_summary.csv`):

| File | Pages | Predicted type | Type conf. | Fields auto-accepted | For review |
|---|---|---|---|---|---|
| `Aadhar.png` | 1 | `aadhaar_card` | 0.99 | 3/4 | 1 |
| `Assignment Ashok.pdf` | 1,2 | `unknown_or_other` | 0.79 | – | document |
| `ChatGPT Image … 03_43_11 PM.png` | 1 | `driving_licence` | 0.99 | 4/4 | 0 |
| `ChatGPT Image … 03_52_54 PM.png` | 1 | `passport` | 0.99 | 3/4 | 1 |
| `ECS.jpeg` | 1 | `nach_ecs_mandate` | 0.99 | 4/5 | 1 |
| `Fatca.jpeg` | 1 | `fatca_annexure` | 1.00 | 4/5 | 1 |
| `ID.png` | 1 | `pan_card` | 1.00 | 3/4 | 1 |
| `Illustration.jpeg` | 1 | `benefit_illustration_declaration` | 1.00 | 3/4 | 1 |
| `Moral.jpeg` | 1 | `moral_hazard_questionnaire` | 1.00 | 4/5 | 1 |
| `Proposal Ashok.pdf` | 1,2 | `unknown_or_other` | 0.60 | – | document |
| `split.jpeg` | 1 | `multiple_policies_consent` | 0.99 | 4/4 | 0 |
| `suitability.jpeg` | 1 | `suitability_profiler_declaration` | 1.00 | 4/5 | 1 |

All ten supported types were found, each multi-page PDF stayed one logical document, and 10 items went to
the review queue: three validator failures on what look like sample numbers (Aadhaar, PAN, passport MRZ),
three handwritten dates whose reads disagree or are ambiguous (Illustration, Moral, suitability), an IFSC
code with `1`/`I` ambiguity, a policy number at 0.80, and the two unsupported PDFs. Details and causes are
in `outputs/question3/failure_cases.md`.

## Screenshot index

| # | File | Shows |
|---|---|---|
| 1 | `screenshots/01_inventory_numeric_code.png` | numerical answer with the generated pandas code |
| 2 | `screenshots/02_inventory_web_search.png` | web-search question: workbook facts plus the honest "web search unavailable" notice (no citations could be produced here) |
| 3 | `screenshots/03_documents_classification.png` | batch classification summary |
| 4 | `screenshots/04_handwritten_fields.png` | handwritten-field extraction with confidence scores |
| 5 | `screenshots/05_review_queue.png` | human-review queue with low-confidence and unsupported items |
| 6 | `screenshots/06_downloads.png` | JSON and report downloads with a masked JSON preview |

Screenshots are taken with masking on (the default), so identity and financial numbers appear as `••••1234`.

## Privacy and data handling

- API keys are read only from the environment or `.env` (git-ignored) and never displayed or logged.
- The UI masks identity and financial numbers by default, including numbers inside review reasons, and
  blurs their regions in the optional document preview.
- `outputs/question3/` JSON and CSV files contain **unmasked** values because they are the local evaluation
  artefacts. Treat them as sensitive.
- The supplied identity and insurance documents are **not in Git**: `data/documents/` is git-ignored except for
  its README, so the documents never enter the history. Copy them into that folder to re-run Question 3.
- Push the repository to a **private** GitHub repository because the Question 3 outputs contain full values,
  or remove those outputs before making anything public.

## Publishing

```bash
cd evaluation-solution
gh repo create evaluation-solution --private --source . --remote origin --push
# or, without the GitHub CLI: create an empty private repository on github.com, then
git remote add origin https://github.com/<your-github-user>/evaluation-solution.git
git push -u origin main
```

## Final submission checklist

- [x] One Streamlit app with separate Inventory Agent and Document Pipeline pages
- [x] Q2 loader, sandbox, search, agent and UI implemented and unit-tested
- [x] Q3 pipeline, validators, confidence scoring, reports and UI implemented and unit-tested
- [x] `outputs/question3/methodology.md`
- [x] README with scope, numbering note, setup for Windows/macOS/Linux, environment variables, methodology, privacy, limitations, troubleshooting
- [x] `.env.example`; `.gitignore` excludes `.env`, keys, caches, crops and logs
- [x] Live Q2 demo transcript
- [x] Live Q3 outputs: per-document JSON, `all_results.json`, flagging report (CSV and JSON), extraction summary
- [x] `outputs/question3/failure_cases.md` completed with live observations
- [x] Live integration tests run (3 passed, 1 failed for the environment reason above)
- [x] Six screenshots captured from the running app
- [x] Secret scan and local commit
- [ ] Live web search with citations (needs a key or network that allows search; see Status)
- [ ] Repository pushed (private) and the link above filled in
