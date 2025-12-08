import time
from fastapi.testclient import TestClient
from src.main import app
from src.ml import classifier
from src.ml.classifier import get_model_info


client = TestClient(app)


def test_model_health_uptime_absent():
    # Reset model state to ensure no model is loaded from prior tests
    classifier._MODEL = None
    classifier._MODEL_HASH = None
    classifier._MODEL_LOADED_AT = None
    classifier._MODEL_LAST_ERROR = None

    r = client.get("/api/v1/health/model", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "absent"
    assert data["uptime_seconds"] is None


def test_model_health_uptime_loaded(monkeypatch, tmp_path):
    # Forge minimal pickle model with predict method
    import pickle, os
    # Use a top-level dummy class to avoid local class pickle issue
    from src.ml.dummy_model_holder import DummyModel  # type: ignore
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(DummyModel()))
    monkeypatch.setenv("CLASSIFICATION_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CLASSIFICATION_MODEL_VERSION", "vTest")
    # Disable opcode scan to allow loading this simple pickle
    monkeypatch.setenv("MODEL_OPCODE_SCAN", "0")
    # Force reload
    from src.ml.classifier import reload_model, load_model
    resp = reload_model(str(model_path), expected_version="vTest", force=True)
    assert resp["status"] == "success"
    load_model()
    time.sleep(0.01)
    r = client.get("/api/v1/health/model", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] is not None and data["uptime_seconds"] > 0
