"""Tests for MultiEvidenceCalibrator v2 weighted logic."""

from src.core.ocr.calibration import EvidenceWeights, MultiEvidenceCalibrator


def test_calibrate_all_evidence():
    cal = MultiEvidenceCalibrator(EvidenceWeights())
    val = cal.calibrate(
        raw_confidence=0.8,
        completeness=0.75,
        item_mean=0.7,
        fallback_recent=0.1,
        parse_error_rate=0.05,
    )
    assert val is not None
    # Expect weighted blend near but below raw due to penalty terms
    assert 0.72 <= val <= 0.82


def test_calibrate_missing_raw():
    cal = MultiEvidenceCalibrator()
    val = cal.calibrate(
        raw_confidence=None,
        completeness=0.6,
        item_mean=0.5,
    )
    assert val is not None
    assert 0.5 <= val <= 0.6


def test_adaptive_reweight():
    cal = MultiEvidenceCalibrator()
    orig_raw = cal.weights.w_raw
    cal.adaptive_reweight(observed_brier=0.35)
    assert cal.weights.w_raw <= orig_raw  # raw weight decreases on poor Brier
    cal.adaptive_reweight(observed_brier=0.10)
    assert cal.weights.w_raw >= orig_raw - 0.05  # partially rebounds
