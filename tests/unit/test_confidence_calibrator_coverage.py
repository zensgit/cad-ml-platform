"""Tests for src/core/assembly/confidence_calibrator.py to improve coverage.

Covers:
- PlattScaling fit and calibrate
- IsotonicCalibration fit and calibrate
- DSEvidenceFusion combine_evidence
- LogOddsWeighting combine_confidence
- ConfidenceCalibrationSystem train, calibrate_and_fuse, save/load
- CalibratedConfidence dataclass
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest


class TestPlattScaling:
    """Tests for PlattScaling class."""

    def test_platt_scaling_initialization(self):
        """Test PlattScaling initializes with LogisticRegression."""
        from src.core.assembly.confidence_calibrator import PlattScaling

        ps = PlattScaling()
        assert ps.fitted is False

    def test_platt_scaling_calibrate_unfitted_returns_raw(self):
        """Test unfitted calibrator returns raw confidence."""
        from src.core.assembly.confidence_calibrator import PlattScaling

        ps = PlattScaling()
        result = ps.calibrate(0.7)
        assert result == 0.7

    def test_platt_scaling_fit_sets_fitted_flag(self):
        """Test fit sets fitted flag to True."""
        from src.core.assembly.confidence_calibrator import PlattScaling, SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        ps = PlattScaling()
        confidence_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        true_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        ps.fit(confidence_scores, true_labels)
        assert ps.fitted is True

    def test_platt_scaling_fit_and_calibrate(self):
        """Test fit then calibrate returns valid probability."""
        from src.core.assembly.confidence_calibrator import PlattScaling, SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        ps = PlattScaling()
        confidence_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        true_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        ps.fit(confidence_scores, true_labels)

        calibrated = ps.calibrate(0.75)
        assert 0.0 <= calibrated <= 1.0

    def test_platt_scaling_handles_extreme_confidence(self):
        """Test calibrate handles extreme confidence values."""
        from src.core.assembly.confidence_calibrator import PlattScaling, SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        ps = PlattScaling()
        confidence_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        true_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        ps.fit(confidence_scores, true_labels)

        # Test very low and very high values
        calibrated_low = ps.calibrate(0.01)
        calibrated_high = ps.calibrate(0.99)

        assert 0.0 <= calibrated_low <= 1.0
        assert 0.0 <= calibrated_high <= 1.0


class TestIsotonicCalibration:
    """Tests for IsotonicCalibration class."""

    def test_isotonic_initialization(self):
        """Test IsotonicCalibration initializes correctly."""
        from src.core.assembly.confidence_calibrator import IsotonicCalibration

        ic = IsotonicCalibration()
        assert ic.fitted is False

    def test_isotonic_calibrate_unfitted_returns_raw(self):
        """Test unfitted calibrator returns raw confidence."""
        from src.core.assembly.confidence_calibrator import IsotonicCalibration

        ic = IsotonicCalibration()
        result = ic.calibrate(0.8)
        assert result == 0.8

    def test_isotonic_fit_sets_fitted_flag(self):
        """Test fit sets fitted flag to True."""
        from src.core.assembly.confidence_calibrator import IsotonicCalibration, SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        ic = IsotonicCalibration()
        confidence_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        true_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

        ic.fit(confidence_scores, true_labels)
        assert ic.fitted is True

    def test_isotonic_fit_and_calibrate(self):
        """Test fit then calibrate returns valid probability."""
        from src.core.assembly.confidence_calibrator import IsotonicCalibration, SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        ic = IsotonicCalibration()
        confidence_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        true_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

        ic.fit(confidence_scores, true_labels)

        calibrated = ic.calibrate(0.65)
        assert 0.0 <= calibrated <= 1.0


class TestDSEvidenceFusion:
    """Tests for DSEvidenceFusion class."""

    def test_combine_evidence_empty_list(self):
        """Test combine_evidence with empty list."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        result = DSEvidenceFusion.combine_evidence([])

        assert result["confidence"] == 0.0
        assert result["uncertainty"] == 1.0
        assert result["conflict"] == 0.0
        assert result["per_source_weights"] == {}

    def test_combine_evidence_single_source(self):
        """Test combine_evidence with single evidence."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        evidence = [{"source": "geometric", "confidence": 0.8}]
        result = DSEvidenceFusion.combine_evidence(evidence)

        assert result["confidence"] == 0.8
        assert result["per_source_weights"]["geometric"] == 1.0

    def test_combine_evidence_multiple_sources(self):
        """Test combine_evidence with multiple evidence sources."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        evidence = [
            {"source": "geometric", "confidence": 0.9},
            {"source": "textual", "confidence": 0.7},
            {"source": "rule_based", "confidence": 0.8},
        ]
        result = DSEvidenceFusion.combine_evidence(evidence)

        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "uncertainty" in result
        assert "conflict" in result
        assert len(result["per_source_weights"]) == 3

    def test_combine_evidence_with_explicit_uncertainty(self):
        """Test combine_evidence with explicit uncertainty values."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        evidence = [
            {"source": "a", "confidence": 0.8, "uncertainty": 0.2},
            {"source": "b", "confidence": 0.6, "uncertainty": 0.4},
        ]
        result = DSEvidenceFusion.combine_evidence(evidence)

        assert "confidence" in result
        assert "uncertainty" in result

    def test_combine_evidence_conflicting_sources(self):
        """Test combine_evidence with conflicting evidence."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        # High confidence from one source, uncertainty from another
        evidence = [
            {"source": "a", "confidence": 0.95, "uncertainty": 0.05},
            {"source": "b", "confidence": 0.05, "uncertainty": 0.05},
        ]
        result = DSEvidenceFusion.combine_evidence(evidence)

        # Result should reflect conflict
        assert result["conflict"] >= 0.0

    def test_combine_evidence_weight_calculation(self):
        """Test weight calculation is proportional to confidence."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        evidence = [
            {"source": "high", "confidence": 0.8},
            {"source": "low", "confidence": 0.2},
        ]
        result = DSEvidenceFusion.combine_evidence(evidence)

        # Higher confidence source should have higher weight
        assert result["per_source_weights"]["high"] > result["per_source_weights"]["low"]


class TestLogOddsWeighting:
    """Tests for LogOddsWeighting class."""

    def test_combine_confidence_empty_list(self):
        """Test combine_confidence with empty list returns 0.5."""
        from src.core.assembly.confidence_calibrator import LogOddsWeighting

        result = LogOddsWeighting.combine_confidence([])
        assert result == 0.5

    def test_combine_confidence_single_value(self):
        """Test combine_confidence with single value."""
        from src.core.assembly.confidence_calibrator import LogOddsWeighting

        result = LogOddsWeighting.combine_confidence([(0.8, 1.0)])
        assert 0.0 <= result <= 1.0
        # Should be close to 0.8 with weight 1.0
        assert abs(result - 0.8) < 0.01

    def test_combine_confidence_equal_weights(self):
        """Test combine_confidence with equal weights."""
        from src.core.assembly.confidence_calibrator import LogOddsWeighting

        result = LogOddsWeighting.combine_confidence([
            (0.7, 1.0),
            (0.9, 1.0),
        ])
        # Average should be between 0.7 and 0.9
        assert 0.7 <= result <= 0.9

    def test_combine_confidence_different_weights(self):
        """Test combine_confidence with different weights."""
        from src.core.assembly.confidence_calibrator import LogOddsWeighting

        # Higher weight for 0.9 should pull result towards 0.9
        result = LogOddsWeighting.combine_confidence([
            (0.5, 1.0),
            (0.9, 3.0),
        ])
        assert result > 0.7

    def test_combine_confidence_extreme_values(self):
        """Test combine_confidence handles extreme values."""
        from src.core.assembly.confidence_calibrator import LogOddsWeighting

        # Very low and very high values should be clipped
        result = LogOddsWeighting.combine_confidence([
            (0.01, 1.0),
            (0.99, 1.0),
        ])
        assert 0.0 <= result <= 1.0

    def test_combine_confidence_zero_weights(self):
        """Test combine_confidence with zero total weight."""
        from src.core.assembly.confidence_calibrator import LogOddsWeighting

        result = LogOddsWeighting.combine_confidence([
            (0.8, 0.0),
            (0.6, 0.0),
        ])
        assert result == 0.5


class TestCalibratedConfidenceDataclass:
    """Tests for CalibratedConfidence dataclass."""

    def test_calibrated_confidence_creation(self):
        """Test CalibratedConfidence dataclass creation."""
        from src.core.assembly.confidence_calibrator import CalibratedConfidence

        cc = CalibratedConfidence(
            raw_confidence=0.7,
            calibrated_confidence=0.75,
            per_source_weights={"a": 0.5, "b": 0.5},
            calibration_method="isotonic_ds",
            uncertainty=0.25,
        )

        assert cc.raw_confidence == 0.7
        assert cc.calibrated_confidence == 0.75
        assert cc.per_source_weights == {"a": 0.5, "b": 0.5}
        assert cc.calibration_method == "isotonic_ds"
        assert cc.uncertainty == 0.25


class TestConfidenceCalibrationSystem:
    """Tests for ConfidenceCalibrationSystem class."""

    def test_system_init_isotonic(self):
        """Test system initialization with isotonic method."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        system = ConfidenceCalibrationSystem(method="isotonic")
        assert system.method == "isotonic"

    def test_system_init_platt(self):
        """Test system initialization with platt method."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        system = ConfidenceCalibrationSystem(method="platt")
        assert system.method == "platt"

    def test_calibrate_and_fuse_empty_evidence(self):
        """Test calibrate_and_fuse with empty evidence list."""
        from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem

        system = ConfidenceCalibrationSystem(method="isotonic")
        result = system.calibrate_and_fuse([], fusion_method="ds")

        assert result.raw_confidence == 0.0
        assert result.calibrated_confidence == 0.0
        assert result.uncertainty == 1.0
        assert result.per_source_weights == {}

    def test_calibrate_and_fuse_ds_method(self):
        """Test calibrate_and_fuse with DS fusion method."""
        from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem

        system = ConfidenceCalibrationSystem(method="isotonic")
        evidence = [
            {"source": "geometric", "confidence": 0.9},
            {"source": "textual", "confidence": 0.7},
        ]
        result = system.calibrate_and_fuse(evidence, fusion_method="ds")

        assert 0.0 <= result.calibrated_confidence <= 1.0
        assert "ds" in result.calibration_method
        assert len(result.per_source_weights) == 2

    def test_calibrate_and_fuse_log_odds_method(self):
        """Test calibrate_and_fuse with log_odds fusion method."""
        from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem

        system = ConfidenceCalibrationSystem(method="isotonic")
        evidence = [
            {"source": "geometric", "confidence": 0.9},
            {"source": "textual", "confidence": 0.7},
        ]
        result = system.calibrate_and_fuse(evidence, fusion_method="log_odds")

        assert 0.0 <= result.calibrated_confidence <= 1.0
        assert "log_odds" in result.calibration_method
        assert len(result.per_source_weights) == 2

    def test_calibrate_and_fuse_evidence_without_source(self):
        """Test calibrate_and_fuse with evidence missing source field."""
        from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem

        system = ConfidenceCalibrationSystem(method="isotonic")
        evidence = [
            {"confidence": 0.8},  # No source field
        ]
        result = system.calibrate_and_fuse(evidence, fusion_method="ds")

        # Should handle missing source gracefully with "unknown"
        assert result.calibrated_confidence >= 0.0

    def test_train_calibrator_requires_sklearn(self):
        """Test train_calibrator raises error without sklearn."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn is available, skipping no-sklearn test")

        system = ConfidenceCalibrationSystem(method="isotonic")
        evaluation_data = [
            {"predicted_confidence": 0.9, "is_correct": 1},
            {"predicted_confidence": 0.5, "is_correct": 0},
        ]

        with pytest.raises(ImportError):
            system.train_calibrator(evaluation_data)

    def test_train_calibrator_with_sklearn(self):
        """Test train_calibrator works with sklearn."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            system = ConfidenceCalibrationSystem(method="isotonic")
            system.model_path = Path(tmpdir)

            evaluation_data = [
                {"predicted_confidence": 0.95, "is_correct": 1},
                {"predicted_confidence": 0.85, "is_correct": 1},
                {"predicted_confidence": 0.75, "is_correct": 1},
                {"predicted_confidence": 0.65, "is_correct": 1},
                {"predicted_confidence": 0.55, "is_correct": 0},
                {"predicted_confidence": 0.45, "is_correct": 0},
                {"predicted_confidence": 0.35, "is_correct": 0},
                {"predicted_confidence": 0.25, "is_correct": 0},
                {"predicted_confidence": 0.15, "is_correct": 0},
            ]

            metrics = system.train_calibrator(evaluation_data)

            assert "brier_score_before" in metrics
            assert "brier_score_after" in metrics
            assert "expected_calibration_error" in metrics
            assert "improvement" in metrics

    def test_save_and_load_calibrator(self):
        """Test save and load calibrator functionality."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            system = ConfidenceCalibrationSystem(method="isotonic")
            system.model_path = Path(tmpdir)

            # Train and save
            evaluation_data = [
                {"predicted_confidence": 0.9, "is_correct": 1},
                {"predicted_confidence": 0.8, "is_correct": 1},
                {"predicted_confidence": 0.7, "is_correct": 1},
                {"predicted_confidence": 0.3, "is_correct": 0},
                {"predicted_confidence": 0.2, "is_correct": 0},
                {"predicted_confidence": 0.1, "is_correct": 0},
            ]
            system.train_calibrator(evaluation_data)

            # Create new system and load
            system2 = ConfidenceCalibrationSystem(method="isotonic")
            system2.model_path = Path(tmpdir)
            loaded = system2.load_calibrator()

            assert loaded is True
            assert system2.calibrator is not None

    def test_load_calibrator_nonexistent_file(self):
        """Test load_calibrator returns False for nonexistent file."""
        from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            system = ConfidenceCalibrationSystem(method="isotonic")
            system.model_path = Path(tmpdir)

            loaded = system.load_calibrator()
            assert loaded is False


class TestCalibrationMetrics:
    """Tests for calibration metrics calculation."""

    def test_brier_score_calculation(self):
        """Test Brier score calculation logic."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            system = ConfidenceCalibrationSystem(method="isotonic")
            system.model_path = Path(tmpdir)

            # Perfect calibration: predictions match outcomes
            evaluation_data = [
                {"predicted_confidence": 1.0, "is_correct": 1},
                {"predicted_confidence": 0.0, "is_correct": 0},
            ]

            metrics = system.train_calibrator(evaluation_data)

            # Brier score should be close to 0 for perfect calibration
            assert metrics["brier_score_before"] < 0.1

    def test_ece_calculation_bins(self):
        """Test ECE calculation with binning logic."""
        from src.core.assembly.confidence_calibrator import (
            ConfidenceCalibrationSystem,
            SKLEARN_AVAILABLE,
        )

        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            system = ConfidenceCalibrationSystem(method="isotonic")
            system.model_path = Path(tmpdir)

            # Data spanning multiple bins
            evaluation_data = [
                {"predicted_confidence": 0.1, "is_correct": 0},
                {"predicted_confidence": 0.2, "is_correct": 0},
                {"predicted_confidence": 0.3, "is_correct": 0},
                {"predicted_confidence": 0.4, "is_correct": 0},
                {"predicted_confidence": 0.5, "is_correct": 0},
                {"predicted_confidence": 0.6, "is_correct": 1},
                {"predicted_confidence": 0.7, "is_correct": 1},
                {"predicted_confidence": 0.8, "is_correct": 1},
                {"predicted_confidence": 0.9, "is_correct": 1},
            ]

            metrics = system.train_calibrator(evaluation_data)

            assert "expected_calibration_error" in metrics
            assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


class TestSKLearnStubClasses:
    """Tests for sklearn stub classes when sklearn is not available."""

    def test_sklearn_available_flag_exists(self):
        """Test SKLEARN_AVAILABLE flag exists."""
        from src.core.assembly.confidence_calibrator import SKLEARN_AVAILABLE

        assert isinstance(SKLEARN_AVAILABLE, bool)


class TestDSFusionEdgeCases:
    """Tests for DS fusion edge cases."""

    def test_combine_evidence_high_conflict(self):
        """Test combine_evidence handles high conflict gracefully."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        # Create high conflict scenario
        evidence = [
            {"source": "a", "confidence": 0.99, "uncertainty": 0.01},
            {"source": "b", "confidence": 0.01, "uncertainty": 0.01},
        ]
        result = DSEvidenceFusion.combine_evidence(evidence)

        # Should not crash and return valid result
        assert "confidence" in result
        assert "conflict" in result

    def test_combine_evidence_zero_total_confidence(self):
        """Test combine_evidence with zero total confidence."""
        from src.core.assembly.confidence_calibrator import DSEvidenceFusion

        evidence = [
            {"source": "a", "confidence": 0.0, "uncertainty": 1.0},
            {"source": "b", "confidence": 0.0, "uncertainty": 1.0},
        ]
        result = DSEvidenceFusion.combine_evidence(evidence)

        # per_source_weights should handle division by zero
        assert result["per_source_weights"] == {}


class TestCalibrationSystemWithoutSKLearn:
    """Tests for calibration system behavior without sklearn."""

    def test_system_without_sklearn_uses_raw_confidence(self):
        """Test system uses raw confidence when sklearn unavailable."""
        from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem

        system = ConfidenceCalibrationSystem(method="isotonic")

        # If calibrator is None (sklearn not available), raw confidence is used
        if system.calibrator is None:
            evidence = [{"source": "test", "confidence": 0.8}]
            result = system.calibrate_and_fuse(evidence, fusion_method="ds")

            # Without calibration, confidence should be close to raw value
            assert abs(result.calibrated_confidence - 0.8) < 0.1
