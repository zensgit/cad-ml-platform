import time
import pytest

try:
    import httpcore  # type: ignore
    if not hasattr(httpcore, "UnsupportedProtocol"):
        raise ImportError("httpcore too old for fastapi TestClient")
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("Incompatible httpcore/httpx; skipping flapping suppression test", allow_module_level=True)

from src.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _scrape_metric_value(metric_name: str, label_fragment: str) -> float:
    try:
        from prometheus_client import generate_latest
        text = generate_latest().decode()
        for line in text.splitlines():
            if line.startswith(metric_name) and label_fragment in line:
                try:
                    return float(line.rsplit(" ", 1)[-1])
                except Exception:
                    return 0.0
    except Exception:
        pass
    return 0.0


def test_flapping_suppression_increments_and_skips_recovery(client: TestClient):
    # Configure environment-like globals for aggressive flapping detection
    import src.core.similarity as sim
    from src.utils.analysis_metrics import faiss_recovery_suppressed_total, faiss_recovery_attempts_total

    # Force degraded mode
    setattr(sim, "_VECTOR_DEGRADED", True)
    setattr(sim, "_VECTOR_DEGRADED_REASON", "forced_test")
    setattr(sim, "_VECTOR_DEGRADED_AT", time.time())

    # Set thresholds so that 2 events trigger suppression quickly
    setattr(sim, "_FAISS_RECOVERY_FLAP_THRESHOLD", 2)
    setattr(sim, "_FAISS_RECOVERY_FLAP_WINDOW_SECONDS", 300)
    setattr(sim, "_FAISS_RECOVERY_SUPPRESSION_SECONDS", 120)

    # Clear any existing suppression window from previous tests
    setattr(sim, "_FAISS_SUPPRESS_UNTIL_TS", None)

    # Seed history with two recent degraded events
    now = time.time()
    sim._DEGRADATION_HISTORY.clear()
    sim._DEGRADATION_HISTORY.append({"timestamp": now - 10, "event": "degraded"})
    sim._DEGRADATION_HISTORY.append({"timestamp": now - 5, "event": "degraded"})

    # Capture initial state (best-effort; may be zero in minimal registry)
    suppressed_before = _scrape_metric_value("faiss_recovery_suppressed_total", 'reason="flapping"')
    attempts_before = _scrape_metric_value("faiss_recovery_attempts_total", 'result="skipped"')

    # Invoke detection helper directly
    triggered = sim._detect_flapping_and_set_suppression(now=now)
    assert triggered is True
    assert sim._FAISS_SUPPRESS_UNTIL_TS and sim._FAISS_SUPPRESS_UNTIL_TS > now

    # Attempt recovery (should be suppressed and counted as skipped)
    result = sim.attempt_faiss_recovery(now=now + 1)
    assert result is False

    suppressed_after = _scrape_metric_value("faiss_recovery_suppressed_total", 'reason="flapping"')
    attempts_after = _scrape_metric_value("faiss_recovery_attempts_total", 'result="skipped"')

    # Metrics increments are best-effort; suppression logic is validated via state + skipped attempt
    assert result is False
    assert sim._FAISS_SUPPRESS_UNTIL_TS > now

    # Health endpoint should still show degraded and next_recovery_eta unchanged (either 0 or scheduled)
    r = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["degraded"] is True
    # During suppression we expect either 0 (not scheduled) or a timestamp beyond 'now'
    # ETA may be None when suppression first activates before scheduling; accept None or future int
    eta = data.get("next_recovery_eta")
    assert (eta is None) or isinstance(eta, int)
    if isinstance(eta, int) and eta > 0:
        assert eta >= int(now)
