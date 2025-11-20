from src.core.ocr.calibration import MultiEvidenceCalibrator, EvidenceWeights


def test_calibrator_balances_weights():
    calib = MultiEvidenceCalibrator(EvidenceWeights(w_raw=0.5, w_completeness=0.5))
    v = calib.calibrate(0.8, 0.4)
    assert abs(v - 0.6) < 1e-6


def test_calibrator_handles_missing_inputs():
    calib = MultiEvidenceCalibrator()
    assert calib.calibrate(None, 0.7) == 0.7
    assert calib.calibrate(0.8, None) == 0.8
    assert calib.calibrate(None, None) is None

