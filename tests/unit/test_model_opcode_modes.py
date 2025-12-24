import os
import pickle
from pathlib import Path

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def setup_module(module):
    os.environ["X_API_KEY"] = "test"
    os.environ["ADMIN_TOKEN"] = "secret"


def teardown_module(module):
    """Cleanup environment variables set by setup_module."""
    os.environ.pop("X_API_KEY", None)
    os.environ.pop("ADMIN_TOKEN", None)
    os.environ.pop("MODEL_OPCODE_MODE", None)
    os.environ.pop("MODEL_OPCODE_SCAN", None)


def _write_model(path: Path, obj) -> None:
    path.write_bytes(pickle.dumps(obj, protocol=2))  # protocol 2 to include GLOBAL for function


class PredictOK:
    def predict(self, xs):
        return [0] * len(xs)


def unsafe_function():  # function pickling triggers GLOBAL opcode
    return 1


def test_opcode_blacklist_blocks_global(tmp_path):
    os.environ["MODEL_OPCODE_MODE"] = "blacklist"
    os.environ["MODEL_OPCODE_SCAN"] = "1"
    target = tmp_path / "blk.pkl"
    _write_model(target, unsafe_function)
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(target)},
    )
    data = resp.json()
    assert data["status"] == "opcode_blocked"
    assert data["error"]["code"] == "INPUT_FORMAT_INVALID"
    assert data["error"]["stage"] == "model_reload"


def test_opcode_audit_does_not_block(tmp_path):
    os.environ["MODEL_OPCODE_MODE"] = "audit"
    os.environ["MODEL_OPCODE_SCAN"] = "1"
    target = tmp_path / "audit.pkl"
    _write_model(target, unsafe_function)
    # Expect success because audit mode does not block globals
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(target), "force": True},
    )
    data = resp.json()
    # In audit mode we still require predict; unsafe_function lacks it so treat as load error
    if data["status"] != "success":
        # Should fail due to missing predict, not opcode block
        assert (
            data["status"] == "error"
            or data["status"].startswith("rollback")
            or data["status"] == "opcode_scan_error"
        )


def test_opcode_whitelist_blocks_global(tmp_path):
    os.environ["MODEL_OPCODE_MODE"] = "whitelist"
    os.environ["MODEL_OPCODE_SCAN"] = "1"
    target = tmp_path / "wl.pkl"
    _write_model(target, unsafe_function)
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(target)},
    )
    data = resp.json()
    assert data["status"] == "opcode_blocked"
    assert data["error"]["code"] == "INPUT_FORMAT_INVALID"


def test_opcode_audit_counts_increment(tmp_path):
    os.environ["MODEL_OPCODE_MODE"] = "audit"
    os.environ["MODEL_OPCODE_SCAN"] = "1"
    target = tmp_path / "audit2.pkl"
    _write_model(target, unsafe_function)
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(target), "force": True},
    )
    # Query audit endpoint
    audit_resp = client.get(
        "/api/v1/model/opcode-audit",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
    )
    audit = audit_resp.json()
    assert "GLOBAL" in audit["opcodes"] or len(audit["opcodes"]) > 0
    assert audit["total_samples"] >= 1
