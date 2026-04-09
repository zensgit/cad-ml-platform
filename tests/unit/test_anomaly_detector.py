"""Tests for the anomaly detector and auto-remediation modules."""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from src.ml.monitoring.anomaly_detector import AnomalyResult, MetricsAnomalyDetector
from src.ml.monitoring.auto_remediation import AutoRemediation, RemediationResult


# ---------------------------------------------------------------------------
# Anomaly detector tests
# ---------------------------------------------------------------------------

class TestMetricsAnomalyDetector:
    """Tests for MetricsAnomalyDetector."""

    def test_fit_and_detect_normal(self):
        """Fit with normal data and detect a normal value -- should not flag anomaly."""
        detector = MetricsAnomalyDetector()
        rng = np.random.RandomState(42)
        data = rng.normal(loc=100, scale=5, size=500)

        detector.fit("test_metric", data)
        result = detector.detect("test_metric", 101.0)

        assert isinstance(result, AnomalyResult)
        assert not result.is_anomaly
        assert result.metric_name == "test_metric"
        assert result.current_value == 101.0
        assert result.severity in ("NONE", "LOW")

    def test_detect_anomaly(self):
        """Fit with normal data (mean=100, std=5) then detect extreme outlier (500)."""
        detector = MetricsAnomalyDetector()
        rng = np.random.RandomState(42)
        data = rng.normal(loc=100, scale=5, size=500)

        detector.fit("test_metric", data)
        result = detector.detect("test_metric", 500.0)

        assert result.is_anomaly
        assert result.anomaly_score > 0.5
        assert result.severity in ("MEDIUM", "HIGH", "CRITICAL")
        assert result.metric_name == "test_metric"
        assert result.current_value == 500.0

    def test_severity_levels(self):
        """Verify _score_to_severity maps scores to correct labels."""
        detector = MetricsAnomalyDetector()

        assert detector._score_to_severity(0.0) == "NONE"
        assert detector._score_to_severity(0.1) == "NONE"
        assert detector._score_to_severity(0.3) == "LOW"
        assert detector._score_to_severity(0.45) == "LOW"
        assert detector._score_to_severity(0.5) == "MEDIUM"
        assert detector._score_to_severity(0.65) == "MEDIUM"
        assert detector._score_to_severity(0.7) == "HIGH"
        assert detector._score_to_severity(0.85) == "HIGH"
        assert detector._score_to_severity(0.9) == "CRITICAL"
        assert detector._score_to_severity(1.0) == "CRITICAL"

    def test_batch_detection(self):
        """Detect multiple metrics at once."""
        detector = MetricsAnomalyDetector()
        rng = np.random.RandomState(42)

        detector.fit("metric_a", rng.normal(50, 2, 300))
        detector.fit("metric_b", rng.normal(200, 10, 300))

        results = detector.detect_batch({"metric_a": 51.0, "metric_b": 205.0})

        assert len(results) == 2
        assert "metric_a" in results
        assert "metric_b" in results
        assert isinstance(results["metric_a"], AnomalyResult)
        assert isinstance(results["metric_b"], AnomalyResult)

    def test_detect_without_fit(self):
        """Detecting on an untrained metric returns is_anomaly=False with reason."""
        detector = MetricsAnomalyDetector()
        result = detector.detect("unknown_metric", 42.0)

        assert result.is_anomaly is False
        assert result.anomaly_score == 0.0
        assert result.severity == "NONE"
        assert result.details.get("reason") == "no_model_trained"

    def test_save_load_models(self, tmp_path):
        """Save models, create new detector, load, and verify detection works."""
        detector1 = MetricsAnomalyDetector()
        rng = np.random.RandomState(42)
        data = rng.normal(loc=100, scale=5, size=500)
        detector1.fit("saved_metric", data)

        model_path = str(tmp_path / "models.joblib")
        detector1.save_models(model_path)

        # Fresh detector -- no models yet
        detector2 = MetricsAnomalyDetector()
        assert detector2.get_status()["models_trained"] == 0

        detector2.load_models(model_path)
        assert detector2.get_status()["models_trained"] == 1
        assert "saved_metric" in detector2.get_status()["metrics_tracked"]

        # Detection should still work after load
        result_normal = detector2.detect("saved_metric", 101.0)
        result_anomaly = detector2.detect("saved_metric", 500.0)

        assert not result_normal.is_anomaly
        assert result_anomaly.is_anomaly

    def test_get_status(self):
        """get_status reports correct model counts."""
        detector = MetricsAnomalyDetector()
        status = detector.get_status()
        assert status["models_trained"] == 0
        assert status["metrics_tracked"] == []
        assert status["sklearn_available"] is True

        rng = np.random.RandomState(0)
        detector.fit("m1", rng.normal(0, 1, 100))
        detector.fit("m2", rng.normal(0, 1, 100))

        status = detector.get_status()
        assert status["models_trained"] == 2
        assert status["metrics_tracked"] == ["m1", "m2"]

    def test_anomaly_result_to_dict(self):
        """AnomalyResult.to_dict produces all expected keys."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_score=0.85,
            severity="HIGH",
            metric_name="test",
            current_value=999.0,
            threshold=0.5,
        )
        d = result.to_dict()
        assert d["is_anomaly"] is True
        assert d["anomaly_score"] == 0.85
        assert d["severity"] == "HIGH"
        assert d["metric_name"] == "test"
        assert d["current_value"] == 999.0
        assert d["threshold"] == 0.5


# ---------------------------------------------------------------------------
# Auto-remediation tests
# ---------------------------------------------------------------------------

class TestAutoRemediation:
    """Tests for AutoRemediation."""

    @staticmethod
    def _make_anomaly(
        metric_name: str = "test_metric",
        is_anomaly: bool = True,
        severity: str = "HIGH",
        score: float = 0.85,
    ) -> AnomalyResult:
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            severity=severity,
            metric_name=metric_name,
            current_value=999.0,
            threshold=0.5,
        )

    def test_rate_limit_blocks_after_max_actions(self):
        """Trigger the same rule 4 times; 4th should be rate-limited (max=3)."""
        remediation = AutoRemediation(rate_limit_window=3600.0)
        anomaly = self._make_anomaly(
            metric_name="classification_accuracy", severity="HIGH"
        )

        results = []
        for _ in range(4):
            r = asyncio.new_event_loop().run_until_complete(
                remediation.evaluate_and_act(anomaly)
            )
            results.append(r)

        # First 3 should execute
        for r in results[:3]:
            assert r.executed is True, f"Expected executed, got reason={r.reason}"

        # 4th should be blocked
        assert results[3].executed is False
        assert results[3].reason == "rate_limited"

    def test_action_history_recorded(self):
        """Verify that every evaluation is recorded in action history."""
        remediation = AutoRemediation(rate_limit_window=3600.0)

        a1 = self._make_anomaly(metric_name="classification_accuracy", severity="HIGH")
        a2 = self._make_anomaly(metric_name="drift_score", severity="HIGH")

        asyncio.new_event_loop().run_until_complete(remediation.evaluate_and_act(a1))
        asyncio.new_event_loop().run_until_complete(remediation.evaluate_and_act(a2))

        history = remediation.get_action_history()
        assert len(history) == 2
        assert history[0]["action"] == "rollback_model"
        assert history[1]["action"] == "refresh_baseline"

    def test_non_anomaly_skipped(self):
        """Non-anomalous result produces no action."""
        remediation = AutoRemediation()
        anomaly = self._make_anomaly(is_anomaly=False)

        result = asyncio.new_event_loop().run_until_complete(
            remediation.evaluate_and_act(anomaly)
        )

        assert result.executed is False
        assert result.reason == "not_anomalous"

    def test_no_matching_rule(self):
        """Anomaly on an unrecognised metric produces no action."""
        remediation = AutoRemediation()
        anomaly = self._make_anomaly(metric_name="unknown_xyz", severity="CRITICAL")

        result = asyncio.new_event_loop().run_until_complete(
            remediation.evaluate_and_act(anomaly)
        )

        assert result.executed is False
        assert result.reason == "no_matching_rule"

    def test_severity_below_threshold_skipped(self):
        """Anomaly with LOW severity on a rule requiring HIGH should not trigger."""
        remediation = AutoRemediation()
        anomaly = self._make_anomaly(
            metric_name="classification_accuracy", severity="LOW"
        )

        result = asyncio.new_event_loop().run_until_complete(
            remediation.evaluate_and_act(anomaly)
        )

        assert result.executed is False
        assert result.reason == "no_matching_rule"

    def test_remediation_result_to_dict(self):
        """RemediationResult.to_dict contains expected fields."""
        result = RemediationResult(
            action="test_action",
            executed=True,
            reason="ok",
            details={"key": "value"},
        )
        d = result.to_dict()
        assert d["action"] == "test_action"
        assert d["executed"] is True
        assert d["reason"] == "ok"
        assert "timestamp" in d
        assert d["details"] == {"key": "value"}
