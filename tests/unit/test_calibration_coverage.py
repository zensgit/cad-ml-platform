"""Tests for src/core/ocr/calibration.py to improve coverage.

Covers:
- EvidenceWeights dataclass
- MultiEvidenceCalibrator class
- calibrate method
- adaptive_reweight method
"""

from __future__ import annotations

import pytest


class TestEvidenceWeightsDataclass:
    """Tests for EvidenceWeights dataclass."""

    def test_default_weights(self):
        """Test EvidenceWeights has correct default values."""
        from src.core.ocr.calibration import EvidenceWeights

        weights = EvidenceWeights()

        assert weights.w_raw == 0.5
        assert weights.w_completeness == 0.25
        assert weights.w_item_mean == 0.15
        assert weights.w_fallback_recent == 0.05
        assert weights.w_parse_error == 0.05

    def test_custom_weights(self):
        """Test EvidenceWeights with custom values."""
        from src.core.ocr.calibration import EvidenceWeights

        weights = EvidenceWeights(
            w_raw=0.6,
            w_completeness=0.2,
            w_item_mean=0.1,
            w_fallback_recent=0.05,
            w_parse_error=0.05,
        )

        assert weights.w_raw == 0.6
        assert weights.w_completeness == 0.2

    def test_weights_sum_to_one(self):
        """Test default weights sum to 1.0."""
        from src.core.ocr.calibration import EvidenceWeights

        weights = EvidenceWeights()
        total = (
            weights.w_raw
            + weights.w_completeness
            + weights.w_item_mean
            + weights.w_fallback_recent
            + weights.w_parse_error
        )

        assert total == 1.0


class TestMultiEvidenceCalibratorInit:
    """Tests for MultiEvidenceCalibrator initialization."""

    def test_default_weights(self):
        """Test calibrator uses default weights."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        assert calibrator.weights.w_raw == 0.5

    def test_custom_weights(self):
        """Test calibrator with custom weights."""
        from src.core.ocr.calibration import EvidenceWeights, MultiEvidenceCalibrator

        custom_weights = EvidenceWeights(w_raw=0.7)
        calibrator = MultiEvidenceCalibrator(weights=custom_weights)

        assert calibrator.weights.w_raw == 0.7


class TestCalibrateMethod:
    """Tests for calibrate method."""

    def test_calibrate_with_raw_only(self):
        """Test calibrate with only raw confidence."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(raw_confidence=0.8, completeness=None)

        assert result is not None
        assert 0 <= result <= 1
        assert result == 0.8  # Only raw confidence, full weight

    def test_calibrate_with_completeness(self):
        """Test calibrate with raw and completeness."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(raw_confidence=0.8, completeness=0.9)

        assert result is not None
        # Weighted average: (0.8 * 0.5 + 0.9 * 0.25) / (0.5 + 0.25)
        expected = (0.8 * 0.5 + 0.9 * 0.25) / 0.75
        assert abs(result - expected) < 0.001

    def test_calibrate_with_all_evidence(self):
        """Test calibrate with all evidence types."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.8,
            completeness=0.9,
            item_mean=0.85,
            fallback_recent=0.1,  # Penalized
            parse_error_rate=0.05,  # Penalized
        )

        assert result is not None
        assert 0 <= result <= 1

    def test_calibrate_with_no_evidence(self):
        """Test calibrate with no evidence returns None."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(raw_confidence=None, completeness=None, item_mean=None)

        assert result is None

    def test_calibrate_with_none_values(self):
        """Test calibrate skips None values."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.8,
            completeness=None,  # Skipped
            item_mean=0.85,
            fallback_recent=None,  # Skipped
            parse_error_rate=None,  # Skipped
        )

        assert result is not None
        assert 0 <= result <= 1

    def test_calibrate_fallback_penalty(self):
        """Test high fallback_recent penalizes score."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Low fallback rate
        result_low = calibrator.calibrate(
            raw_confidence=0.8, completeness=None, fallback_recent=0.1
        )

        # High fallback rate
        result_high = calibrator.calibrate(
            raw_confidence=0.8, completeness=None, fallback_recent=0.9
        )

        # Higher fallback rate should result in lower score
        assert result_high < result_low

    def test_calibrate_parse_error_penalty(self):
        """Test high parse_error_rate penalizes score."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Low error rate
        result_low = calibrator.calibrate(
            raw_confidence=0.8, completeness=None, parse_error_rate=0.1
        )

        # High error rate
        result_high = calibrator.calibrate(
            raw_confidence=0.8, completeness=None, parse_error_rate=0.9
        )

        # Higher error rate should result in lower score
        assert result_high < result_low

    def test_calibrate_clamps_to_range(self):
        """Test calibrate clamps result to [0, 1]."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Very high values
        result_high = calibrator.calibrate(raw_confidence=1.5, completeness=None)
        assert result_high <= 1.0

        # Very low values
        result_low = calibrator.calibrate(raw_confidence=-0.5, completeness=None)
        assert result_low >= 0.0

    def test_calibrate_penalty_clamps_input(self):
        """Test penalty calculation clamps input to [0, 1]."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Fallback > 1 should be clamped
        result = calibrator.calibrate(raw_confidence=0.8, completeness=None, fallback_recent=1.5)
        assert result is not None

        # Fallback < 0 should be clamped
        result = calibrator.calibrate(raw_confidence=0.8, completeness=None, fallback_recent=-0.5)
        assert result is not None


class TestAdaptiveReweight:
    """Tests for adaptive_reweight method."""

    def test_poor_brier_increases_completeness(self):
        """Test poor Brier score increases completeness weight."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        initial_completeness = calibrator.weights.w_completeness

        calibrator.adaptive_reweight(observed_brier=0.35)

        assert calibrator.weights.w_completeness > initial_completeness

    def test_poor_brier_increases_item_mean(self):
        """Test poor Brier score increases item_mean weight."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        initial_item_mean = calibrator.weights.w_item_mean

        calibrator.adaptive_reweight(observed_brier=0.35)

        assert calibrator.weights.w_item_mean > initial_item_mean

    def test_poor_brier_decreases_raw(self):
        """Test poor Brier score decreases raw weight."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        initial_raw = calibrator.weights.w_raw

        calibrator.adaptive_reweight(observed_brier=0.35)

        assert calibrator.weights.w_raw < initial_raw

    def test_good_brier_increases_raw(self):
        """Test good Brier score increases raw weight."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        initial_raw = calibrator.weights.w_raw

        calibrator.adaptive_reweight(observed_brier=0.1)

        assert calibrator.weights.w_raw > initial_raw

    def test_good_brier_decreases_completeness(self):
        """Test good Brier score decreases completeness weight."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        initial_completeness = calibrator.weights.w_completeness

        calibrator.adaptive_reweight(observed_brier=0.1)

        assert calibrator.weights.w_completeness < initial_completeness

    def test_moderate_brier_no_change(self):
        """Test moderate Brier score doesn't change weights."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        initial_raw = calibrator.weights.w_raw
        initial_completeness = calibrator.weights.w_completeness

        calibrator.adaptive_reweight(observed_brier=0.2)  # Between 0.15 and 0.3

        assert calibrator.weights.w_raw == initial_raw
        assert calibrator.weights.w_completeness == initial_completeness

    def test_weight_caps_on_poor_brier(self):
        """Test weights are capped after repeated poor Brier scores."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Apply poor Brier multiple times
        for _ in range(20):
            calibrator.adaptive_reweight(observed_brier=0.5)

        # Weights should be capped
        assert calibrator.weights.w_completeness <= 0.35
        assert calibrator.weights.w_item_mean <= 0.25
        assert calibrator.weights.w_raw >= 0.4

    def test_weight_caps_on_good_brier(self):
        """Test weights are capped after repeated good Brier scores."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Apply good Brier multiple times
        for _ in range(20):
            calibrator.adaptive_reweight(observed_brier=0.05)

        # Weights should be capped
        assert calibrator.weights.w_raw <= 0.6
        assert calibrator.weights.w_completeness >= 0.2
        assert calibrator.weights.w_item_mean >= 0.1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_calibrate_with_zero_values(self):
        """Test calibrate with zero values."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(raw_confidence=0.0, completeness=0.0, item_mean=0.0)

        assert result == 0.0

    def test_calibrate_with_one_values(self):
        """Test calibrate with max values."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=1.0,
            completeness=1.0,
            item_mean=1.0,
            fallback_recent=0.0,  # No penalty
            parse_error_rate=0.0,  # No penalty
        )

        assert result == 1.0

    def test_brier_at_threshold(self):
        """Test adaptive_reweight at threshold boundaries."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        # Test at exactly 0.3 (boundary)
        calibrator1 = MultiEvidenceCalibrator()
        initial_raw = calibrator1.weights.w_raw
        calibrator1.adaptive_reweight(observed_brier=0.3)
        assert calibrator1.weights.w_raw == initial_raw  # No change

        # Test at exactly 0.15 (boundary)
        calibrator2 = MultiEvidenceCalibrator()
        initial_raw = calibrator2.weights.w_raw
        calibrator2.adaptive_reweight(observed_brier=0.15)
        assert calibrator2.weights.w_raw == initial_raw  # No change

    def test_single_evidence_normalization(self):
        """Test calibrate normalizes with single evidence."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Only completeness
        result = calibrator.calibrate(raw_confidence=None, completeness=0.9)

        assert result == 0.9  # Full weight to single evidence


class TestCalibratorIntegration:
    """Integration tests for calibrator workflow."""

    def test_full_calibration_workflow(self):
        """Test full calibration workflow."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Initial calibration
        score1 = calibrator.calibrate(raw_confidence=0.8, completeness=0.9, item_mean=0.85)

        # Simulate poor performance, adjust weights
        calibrator.adaptive_reweight(observed_brier=0.35)

        # Recalibrate with same inputs
        score2 = calibrator.calibrate(raw_confidence=0.8, completeness=0.9, item_mean=0.85)

        # Scores should differ due to weight adjustment
        assert score1 != score2

    def test_calibrator_state_persistence(self):
        """Test calibrator maintains state across calls."""
        from src.core.ocr.calibration import MultiEvidenceCalibrator

        calibrator = MultiEvidenceCalibrator()

        # Adjust weights
        calibrator.adaptive_reweight(observed_brier=0.35)

        new_raw = calibrator.weights.w_raw

        # Call calibrate, weights should remain
        calibrator.calibrate(raw_confidence=0.8, completeness=None)

        assert calibrator.weights.w_raw == new_raw
