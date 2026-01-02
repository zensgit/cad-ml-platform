def test_recovery_state_backend_metric_label():
    from src.core import similarity
    from src.utils.analysis_metrics import faiss_recovery_state_backend

    # Label should be set based on env (default 'file')
    # This test asserts calling labels does not raise and we can set a value.
    g = faiss_recovery_state_backend.labels(backend="file")
    g.set(1)
    assert True
