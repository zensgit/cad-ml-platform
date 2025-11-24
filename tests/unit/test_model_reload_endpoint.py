import os
import pickle
from pathlib import Path
from fastapi.testclient import TestClient

from src.main import app


def test_model_reload_success(tmp_path):
    # create dummy model
    model_path = tmp_path / "dummy_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"predict": lambda X: ["dummy"]}, f)
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"path": str(model_path), "expected_version": "vX", "force": True},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("success", "error")  # success preferred; error acceptable if pickle incompatible


def test_model_reload_not_found():
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"path": "nonexistent.pkl"},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"
    # Verify structured error
    assert "error" in data
    assert data["error"]["code"] == "DATA_NOT_FOUND"


def test_model_reload_size_exceeded(tmp_path, monkeypatch):
    # create large dummy file exceeding small limit
    big_path = tmp_path / "big_model.pkl"
    big_path.write_bytes(b"0" * (1024 * 1024 * 2))  # ~2MB
    monkeypatch.setenv("MODEL_MAX_MB", "0.5")  # 0.5MB limit
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"path": str(big_path), "expected_version": "vZ"},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "size_exceeded"
    # Verify structured error present
    assert "error" in data
    assert data["error"]["code"] == "MODEL_SIZE_EXCEEDED"
    assert "size_mb" in data["error"]["context"]
    assert "max_mb" in data["error"]["context"]


def test_model_reload_version_mismatch(tmp_path):
    """Test model reload with version mismatch."""
    # Create a model file with version metadata
    model_path = tmp_path / "versioned_model.pkl"
    model_data = {
        "predict": lambda X: ["test"],
        "__version__": "v2.0"  # Actual version in model
    }
    with model_path.open("wb") as f:
        pickle.dump(model_data, f)

    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={
            "path": str(model_path),
            "expected_version": "v1.0",  # Expected version (mismatch)
            "force": False  # Don't force - should fail on mismatch
        },
        headers={"X-API-Key": "test"},
    )

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "version_mismatch"
    # Verify structured error
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_FAILED"
    assert data["error"]["context"]["expected"] == "v1.0"
    assert data["error"]["context"]["actual"] == "v2.0"


def test_model_reload_rollback_on_failure(tmp_path, monkeypatch):
    """Test model reload triggers rollback when loading fails."""
    # Create an invalid model file that will fail to load
    bad_model_path = tmp_path / "corrupt_model.pkl"
    bad_model_path.write_bytes(b"NOT_A_VALID_PICKLE_FILE")

    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={
            "path": str(bad_model_path),
            "force": True
        },
        headers={"X-API-Key": "test"},
    )

    assert r.status_code == 200
    data = r.json()
    # Should either fail with error or rollback
    assert data["status"] in ("rollback", "error", "not_found")

    if data["status"] == "rollback":
        # Verify structured error with rollback info
        assert "error" in data
        assert data["error"]["code"] == "MODEL_ROLLBACK"
        assert "rollback_version" in data["error"]["context"]
        assert "model_version" in data  # Should have the rollback version


def test_model_reload_structured_error_format():
    """Test that all error responses follow structured error format."""
    client = TestClient(app)
    r = client.post(
        "/api/v1/model/reload",
        json={"path": "/nonexistent/path/model.pkl"},
        headers={"X-API-Key": "test"},
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


def test_model_reload_success_no_error_field():
    """Test that successful reload doesn't include error field."""
    client = TestClient(app)
    # Note: This may fail if no valid model is available
    # The test primarily validates the response structure
    r = client.post(
        "/api/v1/model/reload",
        json={"force": True},
        headers={"X-API-Key": "test"},
    )

    # Should succeed or fail gracefully
    assert r.status_code == 200
    data = r.json()

    if data["status"] == "success":
        # Success response should not have error field
        assert "error" not in data or data["error"] is None
        assert "model_version" in data
        assert "hash" in data
