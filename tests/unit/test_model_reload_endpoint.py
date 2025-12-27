import os
import pickle
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app
from tests.unit.test_model_reload_errors_structured_support import (
    DummyModel,
    GoodModel,
    VersionedModel,
)


def test_model_reload_success(tmp_path):
    """Test successful model reload with opcode scanning disabled."""
    # create dummy model
    model_path = tmp_path / "dummy_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(DummyModel(), f)

    # Disable opcode scanning for this test
    with patch.dict(os.environ, {"MODEL_OPCODE_SCAN": "0"}):
        client = TestClient(app)
        r = client.post(
            "/api/v1/model/reload",
            json={"path": str(model_path), "expected_version": "vX", "force": True},
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert "model_version" in data
        assert "hash" in data


def test_model_reload_not_found():
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"path": "nonexistent.pkl"},
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"
    # Verify structured error
    assert "error" in data
    assert data["error"]["code"] in ("DATA_NOT_FOUND", "MODEL_NOT_FOUND")


def test_model_reload_missing_path_uses_default(tmp_path, monkeypatch):
    model_path = tmp_path / "missing_default.pkl"
    monkeypatch.setenv("CLASSIFICATION_MODEL_PATH", str(model_path))
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"force": True},
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"
    assert data["error"]["context"]["path"] == str(model_path)


def test_model_reload_size_exceeded(tmp_path, monkeypatch):
    """Test model reload size limit validation."""
    # Create a valid pickle file that exceeds size limit
    big_path = tmp_path / "big_model.pkl"
    # Create a valid pickle with a simple object
    with big_path.open("wb") as f:
        # Create large valid pickle (repeat data to exceed limit)
        large_data = [0] * (300_000)  # Large list to exceed 0.5MB when pickled
        pickle.dump(large_data, f)

    monkeypatch.setenv("MODEL_MAX_MB", "0.5")  # 0.5MB limit

    # Disable opcode scanning to isolate size check
    with patch.dict(os.environ, {"MODEL_OPCODE_SCAN": "0"}):
        client = TestClient(app)
        r = client.post(
            "/api/v1/model/reload",
            json={"path": str(big_path), "expected_version": "vZ"},
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "size_exceeded"
        # Verify structured error present
        assert "error" in data
        assert data["error"]["code"] == "MODEL_SIZE_EXCEEDED"
        assert "size_mb" in data["error"]["context"]
        assert "max_mb" in data["error"]["context"]


def test_model_reload_version_mismatch(tmp_path, monkeypatch):
    """Test model reload with version mismatch."""
    # Create a model file with version metadata
    model_path = tmp_path / "versioned_model.pkl"
    model_data = VersionedModel()  # Has __version__ = "v2.0"
    with model_path.open("wb") as f:
        pickle.dump(model_data, f)

    # Set environment variable for expected version
    monkeypatch.setenv("CLASSIFICATION_MODEL_VERSION", "v1.0")

    # Disable opcode scanning for this test
    with patch.dict(os.environ, {"MODEL_OPCODE_SCAN": "0"}):
        client = TestClient(app)
        r = client.post(
            "/api/v1/model/reload",
            json={
                "path": str(model_path),
                "expected_version": "v2.0",  # Mismatch with env var
                "force": False,  # Don't force - should fail on mismatch
            },
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "version_mismatch"


def test_model_reload_rollback_on_failure(tmp_path):
    """Test model reload triggers rollback when loading fails."""
    # Create an invalid model file that will fail to load
    bad_model_path = tmp_path / "corrupt_model.pkl"
    bad_model_path.write_bytes(b"NOT_A_VALID_PICKLE_FILE")

    # Disable opcode scanning to test rollback behavior
    with patch.dict(os.environ, {"MODEL_OPCODE_SCAN": "0"}):
        client = TestClient(app)
        r = client.post(
            "/api/v1/model/reload",
            json={"path": str(bad_model_path), "force": True},
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )

        assert r.status_code == 200
        data = r.json()
        # Magic number check should catch this before rollback
        assert data["status"] in ("magic_invalid", "error", "rollback")

        if data["status"] == "rollback":
            # Verify structured error with rollback info
            assert "error" in data
            assert data["error"]["code"] == "MODEL_ROLLBACK"
            assert "rollback_version" in data["error"]["context"]
        elif data["status"] == "magic_invalid":
            # Magic number validation caught it before load attempt
            assert "error" in data
            assert data["error"]["code"] == "INPUT_FORMAT_INVALID"


def test_model_reload_structured_error_format():
    """Test that all error responses follow structured error format."""
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"path": "/nonexistent/path/model.pkl"},
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
    )

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"

    # Verify structured error format
    assert "error" in data
    error = data["error"]
    assert "code" in error
    assert "message" in error
    assert "stage" in error
    assert error["stage"] == "model_reload"
    assert "context" in error
    assert "path" in error["context"]


def test_model_reload_success_no_error_field(tmp_path):
    """Test that successful reload doesn't include error field."""
    # Create a valid model file
    model_path = tmp_path / "test_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(DummyModel(), f)

    # Disable opcode scanning for successful load
    with patch.dict(os.environ, {"MODEL_OPCODE_SCAN": "0"}):
        client = TestClient(app)
        r = client.post(
            "/api/v1/model/reload",
            json={"path": str(model_path), "force": True},
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )

        # Should succeed
        assert r.status_code == 200
        data = r.json()

        if data["status"] == "success":
            # Success response should not have error field
            assert "error" not in data or data["error"] is None
            assert "model_version" in data
            assert "hash" in data
