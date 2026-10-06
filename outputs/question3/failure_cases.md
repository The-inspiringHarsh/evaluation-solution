# Question 3: observed failure cases

This file lists failures actually observed while running the pipeline on the supplied files, what each
one does to the output, and how the pipeline contains it. General limitations are in the README.

## Runs behind this list

| Run | Date | What ran |
|---|---|---|
| Offline (no vision model) | 6 Oct 2026 | preprocessing, Tesseract OCR and the keyword classifier on all 12 supplied files (55 s) |
| Live (vision model) | pending | full pipeline via `scripts/run_question3.py`; needs an LLM API key |

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

## Observed in the live run

Pending the live run. This section will record misclassifications, fields whose independent reads
disagreed, values withheld as `null`, validator failures on visually clear values, and any provider
errors, each with the affected file, field and the reason shown in the review queue.
