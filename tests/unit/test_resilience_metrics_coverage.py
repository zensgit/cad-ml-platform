"""Tests for src/core/resilience/metrics.py to improve coverage.

Covers:
- MetricPoint dataclass
- MetricSummary dataclass
- ResilienceMetrics class
- Event recording methods
- Summary and export methods
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest


class TestMetricPointDataclass:
    """Tests for MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test MetricPoint can be created."""
        from src.core.resilience.metrics import MetricPoint

        now = datetime.now()
        point = MetricPoint(timestamp=now, value=1.5)

        assert point.timestamp == now
        assert point.value == 1.5
        assert point.labels == {}

    def test_metric_point_with_labels(self):
        """Test MetricPoint with labels."""
        from src.core.resilience.metrics import MetricPoint

        now = datetime.now()
        labels = {"name": "test", "status": "ok"}
        point = MetricPoint(timestamp=now, value=2.0, labels=labels)

        assert point.labels == labels


class TestMetricSummaryDataclass:
    """Tests for MetricSummary dataclass."""

    def test_metric_summary_creation(self):
        """Test MetricSummary can be created."""
        from src.core.resilience.metrics import MetricSummary

        summary = MetricSummary(name="test_metric")

        assert summary.name == "test_metric"
        assert summary.count == 0
        assert summary.sum == 0.0
        assert summary.min == float('inf')
        assert summary.max == float('-inf')
        assert summary.avg == 0.0
        assert summary.p50 == 0.0
        assert summary.p95 == 0.0
        assert summary.p99 == 0.0

    def test_metric_summary_with_values(self):
        """Test MetricSummary with custom values."""
        from src.core.resilience.metrics import MetricSummary

        summary = MetricSummary(
            name="test",
            count=100,
            sum=500.0,
            min=1.0,
            max=10.0,
            avg=5.0,
            p50=4.5,
            p95=9.0,
            p99=9.9
        )

        assert summary.count == 100
        assert summary.avg == 5.0


class TestResilienceMetricsInit:
    """Tests for ResilienceMetrics initialization."""

    def test_default_window_size(self):
        """Test default window size is 300 seconds."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        assert metrics.window_size == 300

    def test_custom_window_size(self):
        """Test custom window size."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics(window_size=600)

        assert metrics.window_size == 600

    def test_initial_state(self):
        """Test initial state is empty."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        assert len(metrics.counters) == 0
        assert len(metrics.histograms) == 0


class TestRecordCircuitBreakerEvent:
    """Tests for record_circuit_breaker_event method."""

    def test_record_basic_event(self):
        """Test recording basic circuit breaker event."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "state_change",
            "circuit_breaker": "test_cb",
            "state": "open"
        }

        metrics.record_circuit_breaker_event(event)

        assert "test_cb" in metrics.circuit_breaker_metrics
        assert len(metrics.circuit_breaker_metrics["test_cb"]) == 1

    def test_record_event_with_duration(self):
        """Test recording event with duration."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "success",
            "circuit_breaker": "test_cb",
            "state": "closed",
            "duration": 0.5
        }

        metrics.record_circuit_breaker_event(event)

        assert "circuit_breaker_duration_seconds" in metrics.histograms
        assert 0.5 in metrics.histograms["circuit_breaker_duration_seconds"]

    def test_record_event_increments_counter(self):
        """Test counter is incremented."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "rejected",
            "circuit_breaker": "test_cb",
            "state": "open"
        }

        metrics.record_circuit_breaker_event(event)
        metrics.record_circuit_breaker_event(event)

        # Counter key is tuple of (metric_name, labels_tuple)
        counter_count = sum(1 for k, v in metrics.counters.items() if "rejected" in k[0])
        assert counter_count >= 1


class TestRecordRateLimiterEvent:
    """Tests for record_rate_limiter_event method."""

    def test_record_allowed_event(self):
        """Test recording allowed rate limiter event."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "allowed",
            "rate_limiter": "api_limiter",
            "identifier": "user_123"
        }

        metrics.record_rate_limiter_event(event)

        assert "api_limiter" in metrics.rate_limiter_metrics
        assert len(metrics.rate_limiter_metrics["api_limiter"]) == 1

    def test_record_rejected_event(self):
        """Test recording rejected rate limiter event."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "rejected",
            "rate_limiter": "api_limiter"
        }

        metrics.record_rate_limiter_event(event)

        # Check default identifier
        recorded = metrics.rate_limiter_metrics["api_limiter"][0]
        assert recorded["identifier"] == "default"


class TestRecordRetryEvent:
    """Tests for record_retry_event method."""

    def test_record_retry_attempt(self):
        """Test recording retry attempt."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "retry",
            "retry_policy": "default_retry",
            "attempt": 2,
            "delay": 1.5
        }

        metrics.record_retry_event(event)

        assert "default_retry" in metrics.retry_metrics
        assert "retry_delay_seconds" in metrics.histograms
        assert 1.5 in metrics.histograms["retry_delay_seconds"]

    def test_record_success_event(self):
        """Test recording retry success."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "success",
            "retry_policy": "default_retry",
            "attempt": 1
        }

        metrics.record_retry_event(event)

        recorded = metrics.retry_metrics["default_retry"][0]
        assert recorded["event"] == "success"

    def test_record_exhausted_event(self):
        """Test recording retry exhausted event."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "exhausted",
            "retry_policy": "default_retry"
        }

        metrics.record_retry_event(event)

        recorded = metrics.retry_metrics["default_retry"][0]
        assert recorded["event"] == "exhausted"


class TestRecordBulkheadEvent:
    """Tests for record_bulkhead_event method."""

    def test_record_success_event(self):
        """Test recording bulkhead success event."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "success",
            "bulkhead": "api_bulkhead",
            "active_calls": 5,
            "duration": 0.3
        }

        metrics.record_bulkhead_event(event)

        assert "api_bulkhead" in metrics.bulkhead_metrics
        assert "bulkhead_execution_seconds" in metrics.histograms

    def test_record_rejected_event(self):
        """Test recording bulkhead rejection."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        event = {
            "event": "rejected",
            "bulkhead": "api_bulkhead",
            "active_calls": 10
        }

        metrics.record_bulkhead_event(event)

        recorded = metrics.bulkhead_metrics["api_bulkhead"][0]
        assert recorded["active_calls"] == 10


class TestGetSummary:
    """Tests for get_summary method."""

    def test_empty_summary(self):
        """Test summary with no events."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        summary = metrics.get_summary()

        assert "timestamp" in summary
        assert "window_size" in summary
        assert summary["circuit_breakers"] == {}
        assert summary["rate_limiters"] == {}

    def test_summary_with_events(self):
        """Test summary with recorded events."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        # Record some events
        metrics.record_circuit_breaker_event({
            "event": "success",
            "circuit_breaker": "cb1",
            "state": "closed"
        })
        metrics.record_rate_limiter_event({
            "event": "allowed",
            "rate_limiter": "rl1"
        })

        summary = metrics.get_summary()

        assert "cb1" in summary["circuit_breakers"]
        assert "rl1" in summary["rate_limiters"]


class TestSummarizeCircuitBreakers:
    """Tests for _summarize_circuit_breakers method."""

    def test_summarize_multiple_events(self):
        """Test summarizing multiple circuit breaker events."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        # Record various events
        for event_type in ["success", "success", "failure", "rejected", "state_change"]:
            metrics.record_circuit_breaker_event({
                "event": event_type,
                "circuit_breaker": "test_cb",
                "state": "closed"
            })

        summary = metrics.get_summary()
        cb_summary = summary["circuit_breakers"]["test_cb"]

        assert cb_summary["total_events"] == 5
        assert cb_summary["successes"] == 2
        assert cb_summary["failures"] == 1
        assert cb_summary["rejections"] == 1
        assert cb_summary["state_changes"] == 1


class TestSummarizeRateLimiters:
    """Tests for _summarize_rate_limiters method."""

    def test_calculate_rejection_rate(self):
        """Test rejection rate calculation."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        # 3 allowed, 1 rejected = 25% rejection rate
        for _ in range(3):
            metrics.record_rate_limiter_event({
                "event": "allowed",
                "rate_limiter": "test_rl"
            })
        metrics.record_rate_limiter_event({
            "event": "rejected",
            "rate_limiter": "test_rl"
        })

        summary = metrics.get_summary()
        rl_summary = summary["rate_limiters"]["test_rl"]

        assert rl_summary["total_requests"] == 4
        assert rl_summary["allowed"] == 3
        assert rl_summary["rejected"] == 1
        assert rl_summary["rejection_rate"] == 0.25


class TestSummarizeRetryPolicies:
    """Tests for _summarize_retry_policies method."""

    def test_summarize_retry_delays(self):
        """Test summarizing retry delays."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        metrics.record_retry_event({
            "event": "retry",
            "retry_policy": "test_policy",
            "delay": 1.0
        })
        metrics.record_retry_event({
            "event": "retry",
            "retry_policy": "test_policy",
            "delay": 2.0
        })
        metrics.record_retry_event({
            "event": "success",
            "retry_policy": "test_policy"
        })

        summary = metrics.get_summary()
        retry_summary = summary["retry_policies"]["test_policy"]

        assert retry_summary["retries"] == 2
        assert retry_summary["successes"] == 1
        assert retry_summary["avg_delay"] == 1.5
        assert retry_summary["total_delay"] == 3.0


class TestSummarizeBulkheads:
    """Tests for _summarize_bulkheads method."""

    def test_summarize_active_calls(self):
        """Test summarizing active calls."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        for i in range(5):
            metrics.record_bulkhead_event({
                "event": "success",
                "bulkhead": "test_bh",
                "active_calls": i + 1,
                "duration": 0.1 * (i + 1)
            })

        summary = metrics.get_summary()
        bh_summary = summary["bulkheads"]["test_bh"]

        assert bh_summary["total_calls"] == 5
        assert bh_summary["successes"] == 5
        assert bh_summary["max_active_calls"] == 5
        assert bh_summary["avg_active_calls"] == 3.0  # (1+2+3+4+5)/5


class TestGetCounterSummary:
    """Tests for _get_counter_summary method."""

    def test_counter_key_format(self):
        """Test counter key formatting."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        metrics.record_circuit_breaker_event({
            "event": "success",
            "circuit_breaker": "cb1",
            "state": "closed"
        })

        summary = metrics.get_summary()
        counter_summary = summary["counters"]

        # Check at least one counter exists with proper format
        assert len(counter_summary) > 0
        for key in counter_summary:
            assert "{" in key
            assert "}" in key


class TestGetHistogramSummary:
    """Tests for _get_histogram_summary method."""

    def test_histogram_statistics(self):
        """Test histogram statistics calculation."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        # Add values to histogram
        for i in range(10):
            metrics.record_circuit_breaker_event({
                "event": "success",
                "circuit_breaker": "cb1",
                "state": "closed",
                "duration": float(i + 1)
            })

        summary = metrics.get_summary()
        hist_summary = summary["histograms"]["circuit_breaker_duration_seconds"]

        assert hist_summary["count"] == 10
        assert hist_summary["min"] == 1.0
        assert hist_summary["max"] == 10.0
        assert hist_summary["sum"] == 55.0
        assert hist_summary["avg"] == 5.5


class TestPercentile:
    """Tests for _percentile method."""

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        p50 = metrics._percentile(values, 0.50)
        p95 = metrics._percentile(values, 0.95)
        p99 = metrics._percentile(values, 0.99)

        # index = int(10 * 0.50) = 5, values[5] = 6
        assert p50 == 6
        assert p95 == 10  # index = int(10 * 0.95) = 9, values[9] = 10
        assert p99 == 10  # index = int(10 * 0.99) = 9, values[9] = 10

    def test_percentile_empty_list(self):
        """Test percentile with empty list."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        result = metrics._percentile([], 0.50)

        assert result == 0

    def test_percentile_single_value(self):
        """Test percentile with single value."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        result = metrics._percentile([5.0], 0.99)

        assert result == 5.0


class TestExportPrometheus:
    """Tests for export_prometheus method."""

    def test_export_empty(self):
        """Test exporting with no data."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        output = metrics.export_prometheus()

        assert "# HELP" in output
        assert "# TYPE" in output

    def test_export_with_counters(self):
        """Test exporting counters."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        metrics.record_circuit_breaker_event({
            "event": "success",
            "circuit_breaker": "cb1",
            "state": "closed"
        })

        output = metrics.export_prometheus()

        assert "circuit_breaker_success_total" in output

    def test_export_with_histograms(self):
        """Test exporting histograms."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        metrics.record_circuit_breaker_event({
            "event": "success",
            "circuit_breaker": "cb1",
            "state": "closed",
            "duration": 1.5
        })

        output = metrics.export_prometheus()

        assert "circuit_breaker_duration_seconds_bucket" in output
        assert "circuit_breaker_duration_seconds_count" in output
        assert "circuit_breaker_duration_seconds_sum" in output


class TestClearOldMetrics:
    """Tests for clear_old_metrics method."""

    def test_clear_old_events(self):
        """Test clearing old events."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics(window_size=60)

        # Record an event
        metrics.record_circuit_breaker_event({
            "event": "success",
            "circuit_breaker": "cb1",
            "state": "closed"
        })

        # Manually set old timestamp
        old_time = datetime.now() - timedelta(seconds=200)
        metrics.circuit_breaker_metrics["cb1"][0]["timestamp"] = old_time

        metrics.clear_old_metrics()

        # Old event should be cleared
        assert len(metrics.circuit_breaker_metrics["cb1"]) == 0


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_all(self):
        """Test reset clears all metrics."""
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()

        # Add various metrics
        metrics.record_circuit_breaker_event({
            "event": "success", "circuit_breaker": "cb1", "state": "closed"
        })
        metrics.record_rate_limiter_event({
            "event": "allowed", "rate_limiter": "rl1"
        })
        metrics.record_retry_event({
            "event": "retry", "retry_policy": "rp1"
        })
        metrics.record_bulkhead_event({
            "event": "success", "bulkhead": "bh1"
        })

        metrics.reset()

        assert len(metrics.circuit_breaker_metrics) == 0
        assert len(metrics.rate_limiter_metrics) == 0
        assert len(metrics.retry_metrics) == 0
        assert len(metrics.bulkhead_metrics) == 0
        assert len(metrics.counters) == 0
        assert len(metrics.histograms) == 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Test concurrent event recording."""
        import threading
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        errors = []

        def record_events():
            try:
                for i in range(100):
                    metrics.record_circuit_breaker_event({
                        "event": "success",
                        "circuit_breaker": f"cb_{threading.current_thread().name}",
                        "state": "closed"
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_summary(self):
        """Test concurrent summary generation."""
        import threading
        from src.core.resilience.metrics import ResilienceMetrics

        metrics = ResilienceMetrics()
        errors = []

        # Pre-populate with some data
        for i in range(10):
            metrics.record_circuit_breaker_event({
                "event": "success",
                "circuit_breaker": "cb1",
                "state": "closed"
            })

        def get_summaries():
            try:
                for _ in range(50):
                    metrics.get_summary()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_summaries) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
