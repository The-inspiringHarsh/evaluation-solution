"""Reproducible field-level confidence scoring.

score = weighted mean of the available evidence components (each in [0, 1]):

=================  ========  =============================================================
component          weight    meaning
=================  ========  =============================================================
agreement          0.35      share of independent vision reads that agree with the consensus
validation         0.20      1 = format/checksum rule passed, 0 = failed (absent if no rule)
legibility         0.15      model-reported legibility, minus 0.15 per ambiguous character
ocr                0.10      Tesseract support for the consensus (0.05 for handwriting)
proximity          0.10      value region sits next to the expected printed label
image_quality      0.10      sharpness/contrast/resolution score of the page
=================  ========  =============================================================

Hard caps then apply (the lowest applicable cap wins):

* no majority between reads                -> value withheld (null), score <= 0.40
* a read returned nothing while others did -> <= 0.75
* validation failed                        -> <= 0.70
* confusable characters were coerced        -> <= 0.84 (forces review at the 0.85 default)
* handwritten field with < 2 vision reads   -> <= 0.80
* missing value                             -> 0.00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

WEIGHTS = {"agreement": 0.35, "validation": 0.20, "legibility": 0.15, "ocr": 0.10, "proximity": 0.10, "image_quality": 0.10}
HANDWRITTEN_OCR_WEIGHT = 0.05
LEGIBILITY_SCORES = {"clear": 1.0, "partial": 0.6, "poor": 0.25, "absent": 0.0}


@dataclass
class Evidence:
    agreement: Optional[float] = None
    validation: Optional[float] = None
    legibility: Optional[float] = None
    ocr: Optional[float] = None
    proximity: Optional[float] = None
    image_quality: Optional[float] = None
    handwritten: bool = False
    n_vision_reads: int = 0
    majority: bool = True
    some_read_empty: bool = False
    coerced: bool = False
    missing: bool = False
    caps_applied: list[str] = field(default_factory=list)

    def components(self) -> dict[str, Optional[float]]:
        return {k: (None if getattr(self, k) is None else round(getattr(self, k), 3)) for k in WEIGHTS}


def score(ev: Evidence) -> float:
    """Combine evidence into one confidence value in [0, 1] (deterministic)."""
    if ev.missing:
        ev.caps_applied.append("missing value -> 0.0")
        return 0.0
    total, weight_sum = 0.0, 0.0
    for name, weight in WEIGHTS.items():
        value = getattr(ev, name)
        if value is None:
            continue
        if name == "ocr" and ev.handwritten:
            weight = HANDWRITTEN_OCR_WEIGHT
        total += weight * max(0.0, min(1.0, value))
        weight_sum += weight
    result = total / weight_sum if weight_sum else 0.0

    caps: list[tuple[float, str]] = []
    if not ev.majority:
        caps.append((0.40, "independent reads disagree (no majority)"))
    if ev.some_read_empty:
        caps.append((0.75, "at least one independent read found no value"))
    if ev.validation == 0.0:
        caps.append((0.70, "format/checksum validation failed"))
    if ev.coerced:
        caps.append((0.84, "confusable characters were normalised"))
    if ev.handwritten and ev.n_vision_reads < 2:
        caps.append((0.80, "fewer than two vision reads for a handwritten field"))
    for cap, reason in caps:
        if result > cap:
            result = cap
            ev.caps_applied.append(f"{reason} -> capped at {cap}")
    return round(result, 3)


def legibility_from_reads(labels: list[str], ambiguous_chars: int) -> Optional[float]:
    values = [LEGIBILITY_SCORES.get(lbl, 0.5) for lbl in labels if lbl]
    if not values:
        return None
    return max(0.0, sum(values) / len(values) - 0.15 * ambiguous_chars)


def needs_review(confidence: float, threshold: float) -> bool:
    return confidence < threshold
