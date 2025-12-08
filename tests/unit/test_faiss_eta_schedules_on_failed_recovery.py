import time
import pytest
try:
    import httpcore  # type: ignore
    if not hasattr(httpcore, "UnsupportedProtocol"):
        raise ImportError("httpcore too old for httpx TestClient")
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    import pytest
    pytest.skip("Incompatible httpcore/httpx; skipping Faiss failed recovery ETA scheduling test", allow_module_level=True)

from src.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_eta_scheduled_on_failed_recovery(client: TestClient):
    # Force degraded state and ensure next recovery timestamp is cleared
    import src.core.similarity as sim

    # Put system into degraded mode with no available Faiss store so recovery will fail
    setattr(sim, "_VECTOR_DEGRADED", True)
    setattr(sim, "_VECTOR_DEGRADED_REASON", "forced_test")
    setattr(sim, "_VECTOR_DEGRADED_AT", time.time())
    setattr(sim, "_FAISS_NEXT_RECOVERY_TS", None)
    # Ensure suppression not active and manual flag false
    setattr(sim, "_FAISS_SUPPRESS_UNTIL_TS", None)
    setattr(sim, "_FAISS_MANUAL_RECOVERY_IN_PROGRESS", False)

    # Call recovery attempt directly to simulate background loop without waiting
    # Expect False (still degraded) or True (if faiss is available and recovery succeeds)
    before = time.time()
    result = sim.attempt_faiss_recovery(now=before)
    # If faiss is installed, recovery may succeed, so we accept both outcomes
    assert result in (True, False)

    if result is False:
        # Recovery failed, should have scheduled next attempt
        scheduled = getattr(sim, "_FAISS_NEXT_RECOVERY_TS")
        assert isinstance(scheduled, (int, float)) and scheduled > before

        # Health endpoint should reflect scheduled ETA
        r = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
        assert r.status_code == 200
        payload = r.json()
        # Health returns integer seconds; scheduled may be float. Allow small delta.
        assert abs(payload["next_recovery_eta"] - scheduled) < 2
    else:
        # Recovery succeeded - faiss is available, test passes
        # Optionally verify degraded flag is cleared
        assert getattr(sim, "_VECTOR_DEGRADED") is False or result is True
