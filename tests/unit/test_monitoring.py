"""Unit tests for src/ml/monitoring.py (B5.3 PredictionMonitor).

Covers:
- Sliding window behaviour and size limit
- Statistical properties: low_conf_rate, text_hit_rate, avg_confidence, etc.
- Drift detection check_drift() return value
- Alert de-duplication via cooldown
- summary() dict structure and types
- reset() clears state
"""

from __future__ import annotations

import time

import pytest

from src.ml.monitoring.prediction_monitor import PredictionMonitor, PredictionRecord


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def monitor():
    """Fresh monitor with small window for fast tests."""
    m = PredictionMonitor(
        window_size=10,
        low_conf_threshold=0.60,
        drift_alert_rate=0.10,
        text_hit_alert_rate=0.05,
        alert_cooldown_sec=300,
        min_window_for_alerts=5,
    )
    return m


def _fill(m: PredictionMonitor, n: int, confidence: float = 0.9, text_hit: bool = True):
    """Helper: record n identical predictions."""
    for _ in range(n):
        m.record(
            predicted_class="法兰",
            top1_confidence=confidence,
            confidence_margin=confidence - 0.1,
            text_hit=text_hit,
            filename_used=True,
            latency_ms=20.0,
        )


# ── Basics ────────────────────────────────────────────────────────────────────

class TestWindowBehaviour:
    def test_empty_monitor_n_zero(self, monitor):
        assert monitor.n == 0

    def test_record_increments_n(self, monitor):
        _fill(monitor, 3)
        assert monitor.n == 3

    def test_window_capped_at_max(self, monitor):
        _fill(monitor, 20)  # window_size=10
        assert monitor.n == 10

    def test_sliding_window_drops_oldest(self, monitor):
        # Fill window with class "A", then overflow with class "B"
        for _ in range(10):
            monitor.record("A", 0.9)
        for _ in range(5):
            monitor.record("B", 0.9)
        # Window now has 5 B's and 5 A's (oldest 5 A's dropped)
        dist = dict(monitor.class_distribution)
        assert dist["B"] == pytest.approx(0.5, abs=0.01)
        assert dist["A"] == pytest.approx(0.5, abs=0.01)


# ── Statistical properties ────────────────────────────────────────────────────

class TestStatistics:
    def test_low_conf_rate_all_high(self, monitor):
        _fill(monitor, 5, confidence=0.9)
        assert monitor.low_conf_rate == pytest.approx(0.0, abs=0.01)

    def test_low_conf_rate_all_low(self, monitor):
        _fill(monitor, 5, confidence=0.3)
        assert monitor.low_conf_rate == pytest.approx(1.0, abs=0.01)

    def test_low_conf_rate_mixed(self, monitor):
        _fill(monitor, 5, confidence=0.9)   # high conf
        _fill(monitor, 5, confidence=0.2)   # low conf
        assert monitor.low_conf_rate == pytest.approx(0.5, abs=0.01)

    def test_text_hit_rate_none(self, monitor):
        _fill(monitor, 5, text_hit=False)
        assert monitor.text_hit_rate == pytest.approx(0.0, abs=0.01)

    def test_text_hit_rate_all(self, monitor):
        _fill(monitor, 5, text_hit=True)
        assert monitor.text_hit_rate == pytest.approx(1.0, abs=0.01)

    def test_avg_confidence(self, monitor):
        for c in [0.8, 0.6, 0.4]:
            monitor.record("法兰", c)
        assert monitor.avg_confidence == pytest.approx(0.6, abs=0.01)

    def test_avg_latency(self, monitor):
        for ms in [10.0, 20.0, 30.0]:
            monitor.record("轴类", 0.9, latency_ms=ms)
        assert monitor.avg_latency_ms == pytest.approx(20.0, abs=0.1)

    def test_p95_latency_single_sample(self, monitor):
        monitor.record("法兰", 0.9, latency_ms=42.0)
        assert monitor.p95_latency_ms == pytest.approx(42.0, abs=0.1)

    def test_class_distribution_sums_to_one(self, monitor):
        for cls in ["法兰", "轴类", "箱体", "换热器", "罐体"]:
            monitor.record(cls, 0.9)
        total = sum(frac for _, frac in monitor.class_distribution)
        assert total == pytest.approx(1.0, abs=0.001)

    def test_empty_stats_return_zero(self, monitor):
        assert monitor.low_conf_rate == 0.0
        assert monitor.text_hit_rate == 0.0
        assert monitor.avg_confidence == 0.0
        assert monitor.avg_latency_ms == 0.0
        assert monitor.p95_latency_ms == 0.0
        assert monitor.class_distribution == []


# ── Drift detection ───────────────────────────────────────────────────────────

class TestDriftDetection:
    def test_no_drift_when_window_too_small(self, monitor):
        # min_window_for_alerts=5; 4 records should not trigger
        _fill(monitor, 4, confidence=0.1)
        assert monitor.check_drift() is False

    def test_no_drift_high_confidence(self, monitor):
        _fill(monitor, 6, confidence=0.95)
        assert monitor.check_drift() is False

    def test_drift_detected_low_conf(self, monitor):
        # low_conf_rate = 1.0 > drift_alert_rate 0.10
        _fill(monitor, 6, confidence=0.1)
        assert monitor.check_drift() is True

    def test_drift_detected_text_signal_loss(self, monitor):
        # text_hit_rate = 0.0 < text_hit_alert_rate 0.05
        _fill(monitor, 6, confidence=0.95, text_hit=False)
        assert monitor.check_drift() is True

    def test_no_drift_mixed_passes_thresholds(self, monitor):
        # low_conf_rate = 1/6 ≈ 0.17 > 0.10 → drift
        _fill(monitor, 5, confidence=0.95)
        monitor.record("法兰", 0.1)  # 1/6 low-conf
        # 0.167 > 0.10: expect drift
        assert monitor.check_drift() is True


# ── Alert cooldown ────────────────────────────────────────────────────────────

class TestAlertCooldown:
    def test_first_alert_sets_timestamp(self, monitor):
        """First call to _can_alert records the timestamp."""
        assert "low_conf" not in monitor._last_alert_time
        assert monitor._can_alert("low_conf") is True
        assert "low_conf" in monitor._last_alert_time

    def test_second_alert_within_cooldown_suppressed(self, monitor):
        """Second _can_alert call within cooldown returns False."""
        assert monitor._can_alert("low_conf") is True    # first: fires
        assert monitor._can_alert("low_conf") is False   # second: suppressed

    def test_alert_fires_after_cooldown_expires(self, monitor):
        """Alert fires again after cooldown period has passed."""
        monitor.alert_cooldown_sec = 0  # instant expiry
        assert monitor._can_alert("low_conf") is True
        assert monitor._can_alert("low_conf") is True    # fires again

    def test_different_keys_independent(self, monitor):
        """Different alert keys have independent cooldowns."""
        assert monitor._can_alert("low_conf") is True
        assert monitor._can_alert("text_hit_loss") is True   # different key
        assert monitor._can_alert("low_conf") is False       # suppressed

    def test_check_drift_respects_min_window(self, monitor):
        """check_drift() returns False when window is below minimum."""
        _fill(monitor, 4, confidence=0.1)  # below min_window_for_alerts=5
        assert monitor.check_drift() is False


# ── Summary dict ─────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_keys_present(self, monitor):
        _fill(monitor, 5)
        s = monitor.summary()
        required_keys = {
            "window_size", "n", "avg_confidence", "avg_margin",
            "low_conf_rate", "low_conf_threshold", "text_hit_rate",
            "filename_used_rate", "avg_latency_ms", "p95_latency_ms",
            "top5_classes", "drift_detected",
        }
        assert required_keys <= set(s.keys())

    def test_summary_types_json_serialisable(self, monitor):
        import json
        _fill(monitor, 5)
        json.dumps(monitor.summary())  # must not raise

    def test_summary_empty_monitor(self, monitor):
        s = monitor.summary()
        assert s["n"] == 0
        assert s["avg_confidence"] == 0.0
        assert s["top5_classes"] == []

    def test_summary_top5_classes_capped(self, monitor):
        for cls in ["A", "B", "C", "D", "E", "F", "G"]:
            monitor.record(cls, 0.9)
        s = monitor.summary()
        assert len(s["top5_classes"]) <= 5


# ── Reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_records(self, monitor):
        _fill(monitor, 5)
        monitor.reset()
        assert monitor.n == 0

    def test_reset_clears_alert_timestamps(self, monitor):
        _fill(monitor, 6, confidence=0.1)
        monitor._check_drift()  # sets alert timestamp
        monitor.reset()
        assert monitor._last_alert_time == {}

    def test_can_record_after_reset(self, monitor):
        _fill(monitor, 5)
        monitor.reset()
        _fill(monitor, 3)
        assert monitor.n == 3
