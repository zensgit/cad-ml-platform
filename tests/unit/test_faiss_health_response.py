import pytest

try:
    import httpcore  # type: ignore

    if not hasattr(httpcore, "UnsupportedProtocol"):
        raise ImportError("httpcore too old for httpx TestClient")
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    import pytest

    pytest.skip(
        "Incompatible httpcore/httpx; skipping Faiss health response test", allow_module_level=True
    )


def test_faiss_health_response_keys():
    from src.main import app

    client = TestClient(app)
    resp = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
    assert resp.status_code in (200, 401, 403)
    if resp.status_code == 200:
        data = resp.json()
        assert "next_recovery_eta" in data
        assert "manual_recovery_in_progress" in data
        assert "degradation_history_count" in data
        assert "degraded" in data
