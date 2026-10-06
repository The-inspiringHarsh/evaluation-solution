# Question 3 – Methodology

This note explains how the document pipeline classifies documents, reads fields (with a separate
strategy for handwriting), scores confidence, and decides what goes to human review. Everything
described here is implemented in `src/question3_documents/`.

## 1. Pipeline overview

```
file (PNG/JPG/PDF)
  └─ imaging.load_source        PDF → page images (PyMuPDF, 200 dpi); EXIF transpose
       └─ preprocess_page       Tesseract OSD orientation fix → Hough-line deskew (|θ| ≤ 10°)
                                 → CLAHE contrast + light denoise → quality metrics
  └─ ocr.ocr_page               Tesseract word boxes (used for label anchoring + keyword vote)
  └─ extract.classify_source    vision model groups pages into logical documents and classifies
                                 them; independent OCR keyword vote cross-checks the type
  └─ extract.extract_document   per logical document:
        pass 1  vision_full_page          all fields + value bounding boxes
        region  choose_region             value box ∩ OCR-located printed label → crop box
        pass 2  vision_crop               colour, contrast-enhanced, upscaled crop
        pass 3  vision_crop_binarized     (handwritten / checkbox only) adaptive-threshold crop,
                                          character-by-character prompt
        pass 4  ocr_tesseract             OCR of the crop
        rules   validators                format / checksum / calendar checks
  └─ pipeline.cross_field_checks MRZ ↔ visual zone, issue < expiry, amount figures ↔ words
  └─ pipeline.add_cross_document_hints   hints only, never fills a value
  └─ report.write_outputs        per-document JSON, all_results.json, flagging report, summary
```

The original files are never modified: every transformation creates a new in-memory image.

## 2. Classification

* **Content only.** The classifier receives page images labelled "Page 1", "Page 2"… and never the
  file name. It is told the ten supported types (with visual cues) and that any other form — even
  with the same insurer branding — is `unknown_or_other`.
* **Multi-page grouping.** The model returns logical documents as page groups. Groups are sanitised
  (duplicates/out-of-range pages dropped, unassigned pages become their own `unknown_or_other`
  document), so a 2-page PDF form becomes one logical document with `pages: [1, 2]`.
* **Independent cross-check.** A Tesseract keyword vote counts printed phrases per type
  (e.g. "permanent account number", "285ba", "suitability profiler").
* **Classification confidence** = `0.6 × model probability + 0.4 × keyword agreement`, where keyword
  agreement is 1.0 (same type), 0.5 (no decisive keywords) or 0.0 (different type).
  `unknown_or_other` is always routed to review regardless of confidence.

## 3. Printed vs handwritten fields

| | Printed fields (ID cards) | Handwritten fields (insurance forms) |
|---|---|---|
| Vision reads | full page + colour crop | full page + colour crop + **binarised crop** (character-by-character prompt) |
| Crop preprocessing | CLAHE, upscale to ≥1100 px wide | colour path: CLAHE + upscale; second path: grayscale → unsharp mask → adaptive threshold → upscale |
| OCR weight in score | 0.10 | 0.05 (Tesseract is unreliable on handwriting) |
| Minimum reads before a value can be auto-accepted | 1 | 2 (otherwise capped at 0.80) |
| Checkbox fields | – | the crop prompt lists the printed options; struck-through options are explicitly "not selected"; exactly one option must be selected |

**Locating the region.** The full-page pass returns a bounding box for each value. Tesseract locates
the printed label next to it (fuzzy phrase match on OCR word boxes). When the two are adjacent the
crop covers both and proximity evidence is 1.0; a model box alone scores 0.5 (0.8 if a crop pass
confirms the label is visible); a label anchor alone scores 0.8.

**Independence.** Crop passes never see what the full-page pass read, so agreement is genuine
evidence rather than the model repeating itself.

**No guessing.** Prompts require exact transcription, `null` for blank/illegible values, and a list
of ambiguous characters. When reads disagree without a majority, the field's `raw_value` and
`normalized_value` are set to `null`, every candidate is preserved in `candidates`, and the field is
sent to review. Values from other documents are never copied in; cross-document matches appear only
as review hints.

**Confusable characters.** Validators know the expected character class at each position
(IFSC `AAAA0XXXXXX`, PAN `AAAAA9999A`, digits-only account/policy numbers, Indian passport `A9999999`)
and normalise 0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G accordingly. The raw value is kept, the change is
listed in `review_reasons`, and confidence is capped at 0.84 so the field is reviewed at the default
threshold.

## 4. Validation rules

| Field | Rule |
|---|---|
| Aadhaar | 12 digits, first digit 2–9, Verhoeff checksum |
| PAN / TIN | `AAAAA9999A`, 4th character a valid holder-type code (A B C F G H J K L P T) |
| IFSC | 4 letters + `0` + 6 alphanumerics |
| Bank account | digits only, 9–18 long, kept as a string (leading zeros preserved) |
| Amount | numeric after removing ₹/commas; cross-checked against amount in words on NACH forms |
| Dates | real calendar date; DOB/issue dates not in the future; form dates within 2000–next year |
| Driving licence | `SS RR YYYY NNNNNNN` with a valid state code and plausible year; issue < valid-till |
| Passport | `A9999999`; expiry after DOB; passport number/DOB/expiry agree with the MRZ |
| MRZ line 2 | exactly 44 characters; ICAO 9303 check digits for number, DOB, expiry, personal number and composite |
| Checkbox | exactly one printed option selected |

Validation **supports confidence; it never overwrites the observed value**. A value that is visually
clear but fails a rule (e.g. a sample Aadhaar number that fails Verhoeff) is kept and flagged.

## 5. Confidence score

For each field the following evidence components are computed (each 0–1):

| Component | Weight | Definition |
|---|---|---|
| agreement | 0.35 | share of successful vision reads whose format-aware comparison key equals the consensus |
| validation | 0.20 | 1 pass / 0 fail / omitted when no rule applies |
| legibility | 0.15 | mean model legibility (clear 1.0, partial 0.6, poor 0.25) − 0.15 per ambiguous character |
| ocr | 0.10 (0.05 handwritten) | best fuzzy match of the consensus inside the crop's OCR text |
| proximity | 0.10 | value region adjacent to the OCR-located printed label (see §3) |
| image_quality | 0.10 | 0.45 × sharpness + 0.30 × contrast + 0.25 × resolution, each normalised to 0–1 |

`confidence = Σ wᵢ·cᵢ / Σ wᵢ` over the components that are available, then hard caps apply:

| Condition | Cap |
|---|---|
| reads disagree with no majority (value withheld) | 0.40 |
| a read returned nothing while others returned a value | 0.75 |
| validation failed | 0.70 |
| confusable characters were normalised | 0.84 |
| handwritten field with fewer than two vision reads | 0.80 |
| cross-field conflict (MRZ, date order, amount words) | 0.60 |
| value missing in every read | 0.00 |

The score is deterministic for a given set of reads, and each JSON field records its
`confidence_components` so a reviewer can see why a value scored as it did.

## 6. Review threshold: 0.85

A field is auto-accepted only when `confidence ≥ 0.85` **and** reads agree **and** no validation or
cross-field rule failed. The threshold is configurable (`REVIEW_THRESHOLD`).

Why 0.85: these are identity and financial fields used for KYC and bank mandates. Accepting a wrong
account number, IFSC or TIN means a failed debit, a compliance breach or money moving to the wrong
account, while a false alarm costs a reviewer a few seconds. With the weights above, 0.85 can only be
reached when independent reads agree, the value is legible, sits next to its label and passes its
format rule. A single failure (a disagreeing read, a failed checksum, an ambiguous character)
pushes the field below the line. Missing values score 0.0, and unsupported documents are always
flagged.

## 7. Outputs

* `json/<document-id>.json`: one file per logical document (`document-id` = first 10 hex chars of
  the file's SHA-1 + page list, so it is stable across runs).
* `all_results.json`: every document plus the threshold and its rationale.
* `flagging_report.csv` / `.json`: one row per flagged field or document, with source file, page,
  predicted type, field, raw candidate, confidence, threshold, reason and a suggested reviewer action.
* `extraction_summary.csv`: per-document counts of accepted, flagged and missing fields.

The JSON and CSV files contain **unmasked values** because they are the local evaluation artefacts.
The UI masks identity and financial numbers by default.
