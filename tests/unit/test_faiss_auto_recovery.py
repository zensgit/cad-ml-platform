import os
import time

from fastapi.testclient import TestClient

from src.core import similarity as sim
from src.main import app

client = TestClient(app)


def setup_module(module):
    os.environ["X_API_KEY"] = "test"
    # Ensure backend preference is faiss for these tests
    os.environ["VECTOR_STORE_BACKEND"] = "faiss"


def teardown_module(module):
    """Cleanup environment variables set by setup_module."""
    os.environ.pop("X_API_KEY", None)
    os.environ.pop("VECTOR_STORE_BACKEND", None)


def teardown_function(func):
    # Reset global store and degraded flags after each test
    sim.reset_default_store()


def _force_degraded(reason: str = "test"):
    sim._VECTOR_DEGRADED = True
    sim._VECTOR_DEGRADED_REASON = reason
    sim._VECTOR_DEGRADED_AT = time.time() - 5


def test_recovery_skipped_when_not_degraded():
    sim._VECTOR_DEGRADED = False
    ok = sim.attempt_faiss_recovery()
    assert ok is False


def test_recovery_respects_backoff(monkeypatch):
    _force_degraded()
    # Set next recovery in the future
    sim._FAISS_NEXT_RECOVERY_TS = time.time() + 300
    ok = sim.attempt_faiss_recovery()
    assert ok is False


def test_recovery_success_clears_degraded(monkeypatch):
    _force_degraded()
    sim._FAISS_NEXT_RECOVERY_TS = None

    # Monkeypatch FaissVectorStore to appear available without importing faiss
    class DummyFaiss(sim.FaissVectorStore):  # type: ignore
        def __init__(self, normalize=None):
            self._available = True
            self._normalize = True

    monkeypatch.setattr(sim, "FaissVectorStore", DummyFaiss)

    ok = sim.attempt_faiss_recovery()
    info = sim.get_degraded_mode_info()
    assert info["degraded"] is False
    assert info["reason"] is None


def test_manual_recover_endpoint_skipped():
    _force_degraded()
    sim._FAISS_NEXT_RECOVERY_TS = time.time() + 120  # ensure skipped
    r = client.post("/api/v1/faiss/recover", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    assert r.json()["status"] in ("skipped_or_failed", "skipped")
