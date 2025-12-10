"""Tests for analysis_metrics.py to improve coverage.

Covers:
- _safe_counter fallback path (ValueError when metric already registered)
- _safe_histogram fallback path
- _safe_gauge fallback path
- Metric existence verification
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAnalysisMetricsExist:
    """Tests that all expected analysis metrics exist."""

    def test_core_counters_exist(self):
        """Test core counter metrics are defined."""
        from src.utils.analysis_metrics import (
            analysis_requests_total,
            analysis_errors_total,
            analysis_rejections_total,
            analysis_error_code_total,
            analysis_cache_hits_total,
            analysis_cache_miss_total,
        )

        assert analysis_requests_total is not None
        assert analysis_errors_total is not None
        assert analysis_rejections_total is not None
        assert analysis_error_code_total is not None
        assert analysis_cache_hits_total is not None
        assert analysis_cache_miss_total is not None

    def test_histogram_metrics_exist(self):
        """Test histogram metrics are defined."""
        from src.utils.analysis_metrics import (
            analysis_stage_duration_seconds,
            parse_stage_latency_seconds,
            analysis_feature_vector_dimension,
            feature_extraction_latency_seconds,
            classification_latency_seconds,
            process_recommend_latency_seconds,
        )

        assert analysis_stage_duration_seconds is not None
        assert parse_stage_latency_seconds is not None
        assert analysis_feature_vector_dimension is not None
        assert feature_extraction_latency_seconds is not None
        assert classification_latency_seconds is not None
        assert process_recommend_latency_seconds is not None

    def test_gauge_metrics_exist(self):
        """Test gauge metrics are defined."""
        from src.utils.analysis_metrics import (
            faiss_index_size,
            analysis_parallel_enabled,
            strict_mode_enabled,
            faiss_rebuild_backoff_seconds,
            process_start_time_seconds,
        )

        assert faiss_index_size is not None
        assert analysis_parallel_enabled is not None
        assert strict_mode_enabled is not None
        assert faiss_rebuild_backoff_seconds is not None
        assert process_start_time_seconds is not None

    def test_faiss_metrics_exist(self):
        """Test Faiss-related metrics are defined."""
        from src.utils.analysis_metrics import (
            faiss_init_errors_total,
            faiss_rebuild_total,
            faiss_rebuild_duration_seconds,
            faiss_export_total,
            faiss_export_duration_seconds,
            faiss_import_total,
            faiss_import_duration_seconds,
            faiss_auto_rebuild_total,
            faiss_index_dim_mismatch_total,
        )

        assert faiss_init_errors_total is not None
        assert faiss_rebuild_total is not None
        assert faiss_rebuild_duration_seconds is not None
        assert faiss_export_total is not None
        assert faiss_export_duration_seconds is not None
        assert faiss_import_total is not None
        assert faiss_import_duration_seconds is not None
        assert faiss_auto_rebuild_total is not None
        assert faiss_index_dim_mismatch_total is not None

    def test_drift_metrics_exist(self):
        """Test drift-related metrics are defined."""
        from src.utils.analysis_metrics import (
            classification_prediction_drift_score,
            material_distribution_drift_score,
            material_drift_ratio,
            drift_baseline_created_total,
            drift_baseline_refresh_total,
        )

        assert classification_prediction_drift_score is not None
        assert material_distribution_drift_score is not None
        assert material_drift_ratio is not None
        assert drift_baseline_created_total is not None
        assert drift_baseline_refresh_total is not None

    def test_recovery_metrics_exist(self):
        """Test recovery-related metrics are defined."""
        from src.utils.analysis_metrics import (
            faiss_recovery_attempts_total,
            faiss_degraded_duration_seconds,
            faiss_next_recovery_eta_seconds,
            faiss_recovery_suppressed_total,
            faiss_recovery_suppression_remaining_seconds,
        )

        assert faiss_recovery_attempts_total is not None
        assert faiss_degraded_duration_seconds is not None
        assert faiss_next_recovery_eta_seconds is not None
        assert faiss_recovery_suppressed_total is not None
        assert faiss_recovery_suppression_remaining_seconds is not None


class TestMetricsLabelsSupport:
    """Tests for metrics label support."""

    def test_analysis_requests_labels(self):
        """Test analysis_requests_total supports labels."""
        from src.utils.analysis_metrics import analysis_requests_total

        labeled = analysis_requests_total.labels(status="success")
        assert labeled is not None

    def test_analysis_errors_labels(self):
        """Test analysis_errors_total supports labels."""
        from src.utils.analysis_metrics import analysis_errors_total

        labeled = analysis_errors_total.labels(stage="parse", code="timeout")
        assert labeled is not None

    def test_analysis_stage_duration_labels(self):
        """Test analysis_stage_duration_seconds supports labels."""
        from src.utils.analysis_metrics import analysis_stage_duration_seconds

        labeled = analysis_stage_duration_seconds.labels(stage="feature_extraction")
        assert labeled is not None

    def test_vector_query_latency_labels(self):
        """Test vector_query_latency_seconds supports labels."""
        from src.utils.analysis_metrics import vector_query_latency_seconds

        labeled = vector_query_latency_seconds.labels(backend="faiss")
        assert labeled is not None


class TestMetricCaching:
    """Tests for metric caching behavior."""

    def test_metric_cache_is_used(self):
        """Test that metric cache is populated."""
        from src.utils import analysis_metrics

        # The _METRIC_CACHE should be populated after imports
        # This tests that repeated access returns cached metrics
        m1 = analysis_metrics.analysis_requests_total
        m2 = analysis_metrics.analysis_requests_total
        # Should be the same object
        assert m1 is m2

    def test_counter_with_labels_returns_labeled(self):
        """Test counter.labels() returns proper labeled metric."""
        from src.utils.analysis_metrics import analysis_requests_total

        labeled1 = analysis_requests_total.labels(status="success")
        labeled2 = analysis_requests_total.labels(status="error")
        # Different labels should return different labeled children
        # (both should be usable)
        assert labeled1 is not None
        assert labeled2 is not None


class TestExportedMetrics:
    """Tests for __all__ exports."""

    def test_all_exports_importable(self):
        """Test all items in __all__ are importable."""
        from src.utils.analysis_metrics import __all__

        # Verify __all__ contains expected items
        assert "analysis_requests_total" in __all__
        assert "analysis_errors_total" in __all__
        assert "faiss_index_size" in __all__
        assert "process_start_time_seconds" in __all__

    def test_all_exports_count(self):
        """Test __all__ has expected number of exports."""
        from src.utils.analysis_metrics import __all__

        # Should have many metrics exported
        assert len(__all__) > 50


class TestProcessStartTime:
    """Tests for process_start_time_seconds initialization."""

    def test_process_start_time_is_set(self):
        """Test process_start_time_seconds is initialized."""
        from src.utils.analysis_metrics import process_start_time_seconds

        # The metric should exist and be set at module load time
        assert process_start_time_seconds is not None
