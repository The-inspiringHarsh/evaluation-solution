"""Confidence scoring and threshold behaviour."""

from src.question3_documents.confidence import Evidence, legibility_from_reads, needs_review, score


def _strong(**kw):
    base = dict(agreement=1.0, validation=1.0, legibility=1.0, ocr=1.0, proximity=1.0, image_quality=1.0, n_vision_reads=3)
    base.update(kw)
    return Evidence(**base)


def test_perfect_evidence_scores_one():
    assert score(_strong()) == 1.0


def test_missing_is_zero():
    assert score(Evidence(missing=True)) == 0.0


def test_absent_components_are_renormalised():
    assert score(Evidence(agreement=1.0, legibility=1.0, n_vision_reads=2)) == 1.0


def test_validation_failure_caps():
    assert score(_strong(validation=0.0)) <= 0.70


def test_disagreement_caps_hard():
    ev = _strong(agreement=0.0, majority=False)
    assert score(ev) <= 0.40
    assert any("disagree" in c for c in ev.caps_applied)


def test_coercion_forces_review_at_default_threshold():
    assert needs_review(score(_strong(coerced=True)), 0.85)


def test_handwritten_needs_two_reads():
    assert score(_strong(handwritten=True, n_vision_reads=1)) <= 0.80
    assert score(_strong(handwritten=True, n_vision_reads=3)) == 1.0


def test_handwritten_ocr_has_lower_weight():
    printed = score(_strong(ocr=0.0))
    handwritten = score(_strong(ocr=0.0, handwritten=True))
    assert handwritten > printed


def test_score_is_deterministic_and_monotonic():
    a = score(_strong(legibility=0.6))
    b = score(_strong(legibility=0.6))
    c = score(_strong(legibility=0.25))
    assert a == b and c < a


def test_legibility_penalises_ambiguous_characters():
    assert legibility_from_reads(["clear", "clear"], 0) == 1.0
    assert legibility_from_reads(["clear"], 2) == 0.7
    assert legibility_from_reads([], 0) is None


def test_threshold_behaviour():
    assert needs_review(0.849, 0.85) and not needs_review(0.85, 0.85)
    assert not needs_review(0.80, 0.75)
