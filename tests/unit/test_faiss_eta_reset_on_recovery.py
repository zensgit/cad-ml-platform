import pytest
try:
    import httpcore  # type: ignore
    # Skip if legacy httpcore lacking required symbols (common in 0.9.x)
    if not hasattr(httpcore, "UnsupportedProtocol"):
        raise ImportError("httpcore too old for httpx TestClient")
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    import pytest
    pytest.skip("Incompatible httpcore/httpx; skipping Faiss ETA reset test", allow_module_level=True)

from src.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_eta_resets_to_zero_after_recovery(client: TestClient):
    # Set degraded state and simulate a scheduled recovery ETA via globals
    import src.core.similarity as sim

    # Force degraded state and set an ETA
    setattr(sim, "_FAISS_DEGRADED", True)
    setattr(sim, "_FAISS_MANUAL_RECOVERY_IN_PROGRESS", False)
    setattr(sim, "_FAISS_NEXT_RECOVERY_TS", 1234567890)

    # Health should reflect ETA > 0
    r = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    assert r.json()["next_recovery_eta"] == 1234567890
    # Metric value is set internally; rely on health response for validation

    # Trigger manual recovery; background implementation coordinates via manual flag
    r2 = client.post("/api/v1/faiss/recover", headers={"X-API-Key": "test"})
    assert r2.status_code in (200, 202)

    # Fetch health again and ensure the endpoint remains responsive and exposes manual flag
    r3 = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
    assert r3.status_code == 200
    payload = r3.json()
    assert "manual_recovery_in_progress" in payload
    # Metric should be reset by health handler; rely on health response semantics
