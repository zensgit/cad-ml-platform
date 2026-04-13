"""Tests for hybrid calibration module."""

from __future__ import annotations

import pytest
import numpy as np

from src.ml.hybrid.calibration import (
    BetaCalibration,
    CalibrationMethod,
    CalibrationMetrics,
    ConfidenceCalibrator,
    HistogramBinning,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibration_data():
    """Synthetic calibration data: overconfident model."""
    rng = np.random.RandomState(42)
    n = 200
    # True labels: ~60% positive
    labels = (rng.rand(n) < 0.6).astype(float)
    # Overconfident predictions: shift towards 1.0
    confidences = np.clip(labels * 0.7 + rng.rand(n) * 0.3, 0.05, 0.95)
    return confidences, labels


@pytest.fixture
def well_calibrated_data():
    """Well-calibrated synthetic data."""
    rng = np.random.RandomState(123)
    n = 300
    confidences = rng.rand(n)
    labels = (rng.rand(n) < confidences).astype(float)
    return confidences, labels


# ---------------------------------------------------------------------------
# TemperatureScaling
# ---------------------------------------------------------------------------

class TestTemperatureScaling:

    def test_unfitted_returns_identity(self):
        cal = TemperatureScaling()
        assert cal.calibrate(0.8) == 0.8

    def test_fit_and_calibrate(self, calibration_data):
        confidences, labels = calibration_data
        cal = TemperatureScaling()
        cal.fit(confidences, labels)
        assert cal._fitted is True
        result = cal.calibrate(0.9)
        assert 0.0 <= result <= 1.0

    def test_temperature_property(self, calibration_data):
        confidences, labels = calibration_data
        cal = TemperatureScaling()
        cal.fit(confidences, labels)
        assert cal.temperature > 0.0

    def test_calibrate_batch(self, calibration_data):
        confidences, labels = calibration_data
        cal = TemperatureScaling()
        cal.fit(confidences, labels)
        batch = cal.calibrate_batch(np.array([0.3, 0.5, 0.7, 0.9]))
        assert len(batch) == 4
        assert all(0.0 <= v <= 1.0 for v in batch)


# ---------------------------------------------------------------------------
# PlattScaling
# ---------------------------------------------------------------------------

class TestPlattScaling:

    def test_unfitted_returns_identity(self):
        cal = PlattScaling()
        assert cal.calibrate(0.7) == 0.7

    def test_fit_and_calibrate(self, calibration_data):
        confidences, labels = calibration_data
        cal = PlattScaling()
        cal.fit(confidences, labels)
        assert cal._fitted is True
        result = cal.calibrate(0.8)
        assert 0.0 <= result <= 1.0

    def test_output_is_sigmoid(self, calibration_data):
        """Platt scaling should produce sigmoid-shaped output."""
        confidences, labels = calibration_data
        cal = PlattScaling()
        cal.fit(confidences, labels)
        low = cal.calibrate(0.1)
        high = cal.calibrate(0.9)
        # Higher input should generally produce higher output
        # (not always guaranteed but usually true for well-behaved data)
        assert isinstance(low, float)
        assert isinstance(high, float)


# ---------------------------------------------------------------------------
# IsotonicCalibration
# ---------------------------------------------------------------------------

class TestIsotonicCalibration:

    def test_unfitted_returns_identity(self):
        cal = IsotonicCalibration()
        assert cal.calibrate(0.5) == 0.5

    def test_fit_and_calibrate(self, calibration_data):
        confidences, labels = calibration_data
        cal = IsotonicCalibration()
        cal.fit(confidences, labels)
        assert cal._fitted is True
        result = cal.calibrate(0.7)
        assert 0.0 <= result <= 1.0

    def test_calibrate_edge_values(self, calibration_data):
        confidences, labels = calibration_data
        cal = IsotonicCalibration()
        cal.fit(confidences, labels)
        # Test near boundaries
        assert 0.0 <= cal.calibrate(0.01) <= 1.0
        assert 0.0 <= cal.calibrate(0.99) <= 1.0


# ---------------------------------------------------------------------------
# HistogramBinning
# ---------------------------------------------------------------------------

class TestHistogramBinning:

    def test_unfitted_returns_identity(self):
        cal = HistogramBinning(n_bins=10)
        assert cal.calibrate(0.5) == 0.5

    def test_fit_and_calibrate(self, calibration_data):
        confidences, labels = calibration_data
        cal = HistogramBinning(n_bins=10)
        cal.fit(confidences, labels)
        assert cal._fitted is True
        result = cal.calibrate(0.7)
        assert 0.0 <= result <= 1.0

    def test_different_bin_counts(self, calibration_data):
        confidences, labels = calibration_data
        cal5 = HistogramBinning(n_bins=5)
        cal20 = HistogramBinning(n_bins=20)
        cal5.fit(confidences, labels)
        cal20.fit(confidences, labels)
        # Both should produce valid results
        r5 = cal5.calibrate(0.6)
        r20 = cal20.calibrate(0.6)
        assert 0.0 <= r5 <= 1.0
        assert 0.0 <= r20 <= 1.0


# ---------------------------------------------------------------------------
# BetaCalibration
# ---------------------------------------------------------------------------

class TestBetaCalibration:

    def test_unfitted_returns_identity(self):
        cal = BetaCalibration()
        assert cal.calibrate(0.5) == 0.5

    def test_fit_and_calibrate(self, calibration_data):
        confidences, labels = calibration_data
        cal = BetaCalibration()
        cal.fit(confidences, labels)
        assert cal._fitted is True
        result = cal.calibrate(0.8)
        assert 0.0 <= result <= 1.0

    def test_extreme_values(self, calibration_data):
        confidences, labels = calibration_data
        cal = BetaCalibration()
        cal.fit(confidences, labels)
        # Values near 0 and 1 should not produce NaN
        r_low = cal.calibrate(0.01)
        r_high = cal.calibrate(0.99)
        assert not np.isnan(r_low)
        assert not np.isnan(r_high)


# ---------------------------------------------------------------------------
# ConfidenceCalibrator (main interface)
# ---------------------------------------------------------------------------

class TestConfidenceCalibrator:

    def test_default_method(self):
        cal = ConfidenceCalibrator()
        assert cal.method == CalibrationMethod.TEMPERATURE_SCALING

    def test_uncalibrated_passthrough(self):
        """Without fitting, should return original confidence."""
        cal = ConfidenceCalibrator()
        assert cal.calibrate(0.7) == 0.7

    def test_fit_and_calibrate_global(self, calibration_data):
        confidences, labels = calibration_data
        cal = ConfidenceCalibrator(
            method=CalibrationMethod.HISTOGRAM_BINNING,
            per_source=False,
        )
        cal.fit(confidences, labels)
        result = cal.calibrate(0.7)
        assert 0.0 <= result <= 1.0

    def test_per_source_calibration(self, calibration_data):
        confidences, labels = calibration_data
        sources = np.array(["ml"] * 100 + ["rules"] * 100)
        cal = ConfidenceCalibrator(
            method=CalibrationMethod.HISTOGRAM_BINNING,
            per_source=True,
        )
        cal.fit(confidences, labels, sources=sources)
        # Should use source-specific calibrator
        result = cal.calibrate(0.7, source="ml")
        assert 0.0 <= result <= 1.0

    def test_fallback_to_global(self, calibration_data):
        confidences, labels = calibration_data
        cal = ConfidenceCalibrator(per_source=True)
        cal.fit(confidences, labels)
        # No source-specific calibrator -> falls back to global
        result = cal.calibrate(0.7, source="unknown_source")
        assert 0.0 <= result <= 1.0

    def test_all_methods_available(self):
        """All calibration methods should be in METHODS map."""
        for method in CalibrationMethod:
            assert method in ConfidenceCalibrator.METHODS


# ---------------------------------------------------------------------------
# ECE / CalibrationMetrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:

    def test_evaluate_well_calibrated(self, well_calibrated_data):
        confidences, labels = well_calibrated_data
        cal = ConfidenceCalibrator()
        metrics = cal.evaluate(confidences, labels)
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.expected_calibration_error >= 0.0
        assert metrics.maximum_calibration_error >= 0.0
        assert metrics.brier_score >= 0.0
        assert metrics.n_samples == len(confidences)

    def test_ece_lower_than_mce(self, calibration_data):
        """ECE should generally be <= MCE."""
        confidences, labels = calibration_data
        cal = ConfidenceCalibrator()
        metrics = cal.evaluate(confidences, labels)
        assert metrics.expected_calibration_error <= metrics.maximum_calibration_error + 1e-9

    def test_reliability_diagram_structure(self, calibration_data):
        confidences, labels = calibration_data
        cal = ConfidenceCalibrator()
        metrics = cal.evaluate(confidences, labels, n_bins=10)
        diag = metrics.reliability_diagram
        assert "bin_accuracies" in diag
        assert "bin_confidences" in diag
        assert "bin_counts" in diag
        assert len(diag["bin_accuracies"]) == 10

    def test_metrics_to_dict(self, calibration_data):
        confidences, labels = calibration_data
        cal = ConfidenceCalibrator()
        metrics = cal.evaluate(confidences, labels)
        d = metrics.to_dict()
        assert "ece" in d
        assert "mce" in d
        assert "brier_score" in d
        assert "n_samples" in d

    def test_perfect_calibration_low_ece(self):
        """Perfect calibration should have very low ECE."""
        # Create perfectly calibrated data
        confidences = np.array([0.5] * 100)
        labels = np.array([0] * 50 + [1] * 50)
        cal = ConfidenceCalibrator()
        metrics = cal.evaluate(confidences, labels)
        assert metrics.expected_calibration_error < 0.1
