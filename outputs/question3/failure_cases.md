# Question 3: observed failure cases

This file lists failures actually observed while running the pipeline on the supplied files, what each
one does to the output, and how the pipeline contains it. General limitations are in the README.

## Runs behind this list

| Run | Date | What ran |
|---|---|---|
| Offline (no vision model) | 6 Oct 2026 | preprocessing, Tesseract OCR and the keyword classifier on all 12 supplied files (55 s) |
| Live, attempt 1 | 6 Oct 2026, about 18:50 UTC | per-field crop reads on `gemini-3.6-flash`; stopped after 3 files when the free-tier daily quota (20 requests per model) ran out |
| Live, run 1 | 6 Oct 2026, 19:04–19:38 UTC | batched crop reads, main model `gemini-3.8-flash` with Flash fallbacks; 27 live calls (3.8-flash 2, 3.7-flash 5, 3.5-flash 9, 3-flash-preview 11); 10 of 12 files, `split.jpeg` and `suitability.jpeg` failed on quota |
| Live, gap fill 1 | 7 Oct 2026, about 01:55 UTC | same settings; reused the 27 cached responses and made 7 live calls (3.6-flash 2, 3.5-flash 2, 3-flash-preview 2, 3.7-flash 1) |
| Live, gap fill 2 | 7 Oct 2026, 07:01–07:10 UTC | same settings after the box-order fix (case 4); 6 live calls (3.8-flash 2, 3.6-flash 2, 3.7-flash 1, 3.5-flash 1) for `Illustration.jpeg`, `split.jpeg` and `suitability.jpeg`; 12 of 12 files, 0 file errors |
| Final write | 7 Oct 2026, 07:12 UTC | same settings, all 38 responses from the cache (0 live calls) after a fix to the wording of ambiguity review reasons; this is the run block in `all_results.json`. Its 32 "model not recorded" responses are the ones made by run 1 and gap fill 1, before the cache stored the answering model |

## Observed without the vision model

### 1. Keyword classifier calls the Proposal PDF a NACH/ECS mandate

* **File:** `Proposal Ashok.pdf` (2 pages). The keyword vote over both pages picked `nach_ecs_mandate`.
* **Cause:** page 2 is a set of proposal declarations, but its signature line says "Signature should match
  with signature of ECS/SI mandate", so the printed words "ECS" and "mandate" both hit the NACH keyword list.
* **Effect if used alone:** an unsupported form would be routed to NACH extraction.
* **Containment:** the keyword vote never decides the type. The vision classifier decides; the vote only
  contributes 40 % of the classification confidence and adds a disagreement note
  ("OCR keyword classifier suggests 'nach_ecs_mandate' but the vision model says ..."), which appears in the
  review queue. `unknown_or_other` documents are always queued for review.

### 2. Substring keyword matching produced a spurious hit (fixed)

* **File:** `Moral.jpeg` scored a NACH keyword because "ecs" matched inside a longer word.
* **Fix:** keywords are now matched as whole words/phrases (`ocr._keyword_pattern`). After the fix the ten
  images each received a keyword vote for a different supported type, and `Assignment Ashok.pdf` had no
  decisive keywords.

### 3. Weak passport keyword on insurance forms

* **Files:** `Illustration.jpeg`, `Moral.jpeg`, `suitability.jpeg` and both Ashok PDFs. Their insurer
  footer says "DO NOT prefix any country code", which matches the passport keyword "country code".
* **Effect:** one spurious passport point per form. It never changed a vote (a vote needs at least two hits
  and a strict lead), but it shows why OCR keywords are only a cross-check.

## Observed in the live runs

Values below are masked; the full values are in the local JSON outputs.

### 4. The vision model sometimes returns boxes as [y0, x0, y1, x1] (fixed)

* **File:** `Illustration.jpeg`. The prompt asks for `[x0, y0, x1, y1]`, but Gemini at times answered in its
  native `[ymin, xmin, ymax, xmax]` order. The crops for application number, date and place were cut from
  the wrong part of the page, both crop passes returned nothing, and three fields fell to 0.62–0.65.
  After the fix, application number, policyholder name and place are read consistently and accepted
  (0.87–0.96).
* **Fix:** `extract._model_box` reads each box both ways and keeps the reading that lies next to the
  OCR-located label, or, without a label, the one with a plausible field shape. A swapped reading is noted
  in the field's evidence ("box given as [y0, x0, y1, x1]"). Two tests cover it, including a guard that
  label proximity never turns a wide value box into a thin sliver (an early version of the fix did that on
  the driving licence and FATCA forms).

### 5. Free-tier quota failures

* **What happened:** the free tier allows 20 requests per model per day, and failed "high demand" (503)
  requests count too. Per-field crop reads need about 100 requests for this batch, so attempt 1 stopped
  after three files. Run 1 switched to batched crop reads but still ran out before `split.jpeg` and
  `suitability.jpeg`.
* **Effect:** a failed pass is never treated as an empty read. Gap fill 2 completed the two missing files,
  so the final outputs have no file errors and no failed passes. The document gets a warning
  ("vision_crop pass failed for …: HTTP 429 …"), the field records the failure as a review reason, and a
  field with no successful read scores 0.0 and is queued.
* **Containment:** retries with backoff, per-model pacing, fallback models and a response cache so a
  re-run with the same settings repeats only the requests that failed or changed.

### 6. Validators reject values that are read clearly (sample data)

These values agree across every read, but fail the official format rules, so they are capped at 0.70 and
queued rather than "corrected":

| File | Field | Validator result |
|---|---|---|
| `Aadhar.png` | `aadhaar_number` | starts with 0 or 1, and the Verhoeff checksum fails |
| `ID.png` (PAN) | `pan_number` | 4th character `D` is not a valid holder-type code |
| Passport image | `mrz_line_2` | birth-date and expiry check digits do not match |

The passport's number, birth date and expiry still agree with the MRZ fields, so only the MRZ line is
queued. All three look like sample documents built with placeholder numbers.

### 7. Handwritten reads that disagree

* `Moral.jpeg` → `date`: the full-page read, the colour crop and the binarised crop gave three different
  dates (day 24, 25 or 26; year 2024 or 2026). With no majority the value is withheld (`null`, 0.40) and
  all three candidates are shown to the reviewer.
* `Illustration.jpeg` → `date`: the full-page read gave 28/04/2024 and the binarised crop 20/07/2016, and
  the colour crop returned nothing. Withheld (`null`, 0.40).
* `suitability.jpeg` → `date`: the reads gave `26/04/26` and `26/01/16`, with `6`/`4` reported as ambiguous.
  The majority value is kept at 0.79 and queued.
* `ECS.jpeg` → `ifsc_code`: two reads differ in one character, and the model flagged `1`/`I` ambiguity at
  three positions. The majority value is kept at 0.75 and queued.
* `Fatca.jpeg` → `policy_number`: the reads agree but OCR could not confirm the label next to the value, so
  it stays at 0.80, just under the threshold. The same number was read confidently on another form in the
  batch; that is shown as a review hint only and never copied in.

### 8. Keyword and vision classifiers disagree on the Proposal PDF

As predicted in case 1, the keyword vote said `nach_ecs_mandate` and the vision model said
`unknown_or_other`. The vision model decides, the disagreement lowers classification confidence to 0.60,
and the document is queued as unsupported. `Assignment Ashok.pdf` has no decisive keywords and is also
classified `unknown_or_other`.

### 9. Summary of the final outputs

12 logical documents from 12 files, all ten supported types found, both PDFs kept as single
`unknown_or_other` documents, 36 of 44 fields auto-accepted, 10 items in the review queue, 0 file errors.
No value was copied between documents and no withheld value was guessed.
