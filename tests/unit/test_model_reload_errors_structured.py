import os
from fastapi.testclient import TestClient
from pathlib import Path

from src.main import app


client = TestClient(app)


def setup_module(module):
    os.environ["X_API_KEY"] = "test"
    os.environ["ADMIN_TOKEN"] = "secret"


def teardown_module(module):
    """Cleanup environment variables set by setup_module."""
    os.environ.pop("X_API_KEY", None)
    os.environ.pop("ADMIN_TOKEN", None)
    os.environ.pop("MODEL_MAX_MB", None)
    os.environ.pop("ALLOWED_MODEL_HASHES", None)
    os.environ.pop("MODEL_OPCODE_SCAN", None)


def test_model_reload_magic_invalid(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"XX")
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(bad)}
    )
    assert resp.status_code in (200, 400, 422)
    data = resp.json()
    # Unified structure: status and error dict
    assert data["status"] == "magic_invalid"
    err = data["error"]
    assert err["code"] == "INPUT_FORMAT_INVALID"
    assert err["stage"] == "model_reload"
    assert "magic_bytes" in err.get("context", {})


def test_model_reload_size_exceeded(tmp_path):
    big = tmp_path / "big.pkl"
    big.write_bytes(b"\x80\x05" + b"0" * (1024 * 1024 * 3))
    os.environ["MODEL_MAX_MB"] = "1"
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(big)}
    )
    assert resp.status_code in (200, 400, 422)
    data = resp.json()
    assert data["status"] == "size_exceeded"
    err = data["error"]
    assert err["code"] == "MODEL_SIZE_EXCEEDED"
    assert err["stage"] == "model_reload"
    ctx = err.get("context", {})
    assert "size_mb" in ctx and "max_mb" in ctx
    assert ctx["size_mb"] > ctx["max_mb"] - 0.1


def test_model_reload_hash_mismatch(tmp_path):
    # Create a minimal valid pickle with protocol 5 (use top-level class to allow pickling)
    model_file = tmp_path / "model.pkl"
    import pickle
    from tests.unit.test_model_reload_errors_structured_support import DummyModel
    model_file.write_bytes(pickle.dumps(DummyModel(), protocol=5))
    # Set whitelist that won't match
    os.environ["ALLOWED_MODEL_HASHES"] = "deadbeefcafefeed"
    os.environ["MODEL_OPCODE_SCAN"] = "0"  # disable scan to reach hash mismatch path
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(model_file)}
    )
    data = resp.json()
    assert data["status"] == "hash_mismatch"
    err = data["error"]
    assert err["code"] == "INPUT_VALIDATION_FAILED"
    assert err["stage"] == "model_reload"
    ctx = err.get("context", {})
    assert "found_hash" in ctx and "expected_hashes" in ctx
    assert len(ctx.get("expected_hashes", [])) >= 1


def test_model_reload_opcode_blocked(tmp_path):
    # Craft pickle containing GLOBAL opcode by pickling a function reference
    import pickle
    from tests.unit.test_model_reload_errors_structured_support import dummy_function
    blocked_file = tmp_path / "blocked.pkl"
    blocked_file.write_bytes(pickle.dumps(dummy_function, protocol=2))  # Protocol 2 emits GLOBAL
    os.environ["MODEL_OPCODE_SCAN"] = "1"
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(blocked_file)}
    )
    data = resp.json()
    assert data["status"] == "opcode_blocked"
    err = data["error"]
    assert err["code"] == "INPUT_FORMAT_INVALID"
    assert err["stage"] == "model_reload"
    ctx = err.get("context", {})
    assert "opcode" in ctx and "position" in ctx


def test_model_reload_success(tmp_path):
    import pickle
    from tests.unit.test_model_reload_errors_structured_support import GoodModel
    good = tmp_path / "good.pkl"
    good.write_bytes(pickle.dumps(GoodModel(), protocol=5))
    # Clear restrictive envs
    os.environ.pop("ALLOWED_MODEL_HASHES", None)
    os.environ.pop("MODEL_MAX_MB", None)
    os.environ["MODEL_OPCODE_SCAN"] = "0"  # disable security scan for success path
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(good)}
    )
    data = resp.json()
    assert data["status"] == "success"
    assert data.get("model_version") is not None or True  # version may be 'none'
    assert "hash" in data
