def test_core_metrics_exported():
    from src.utils import analysis_metrics

    required = {
        'feature_extraction_latency_seconds',
        'similarity_degraded_total',
        'faiss_recovery_attempts_total',
        'faiss_degraded_duration_seconds',
        'vector_migrate_dimension_delta',
    }
    exported = set(getattr(analysis_metrics, '__all__', []))
    missing = required - exported
    assert not missing, f"Missing required metrics in __all__: {missing}"

