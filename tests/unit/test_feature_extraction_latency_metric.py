import pytest


def test_v4_latency_metric_exported_and_observable():
    # Import metric from analysis_metrics and ensure it supports version label and observe
    from src.utils.analysis_metrics import feature_extraction_latency_seconds

    # Should not raise when labeling with version=v4
    m = feature_extraction_latency_seconds.labels(version="v4")
    # Observing a tiny latency value should succeed
    m.observe(0.001)

    # If we reach here without exceptions, the metric is properly configured
    assert True
