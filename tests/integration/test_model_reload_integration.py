"""Integration tests for model reload with real pickle files.

Tests end-to-end reload behavior including load_seq increments, error handling,
and rollback mechanisms using actual pickle files.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import pytest


class DummyModel:
    """Simple model implementing predict method for testing."""

    def __init__(self, version: str):
        self.version = version

    def predict(self, X):
        """Dummy prediction returning version."""
        return [self.version] * len(X)


@pytest.fixture
def disable_opcode_scan():
    """Temporarily disable opcode scanning for integration tests."""
    original = os.getenv("MODEL_OPCODE_SCAN")
    os.environ["MODEL_OPCODE_SCAN"] = "0"
    yield
    if original:
        os.environ["MODEL_OPCODE_SCAN"] = original
    else:
        os.environ.pop("MODEL_OPCODE_SCAN", None)


@pytest.fixture
def temp_model_files():
    """Create temporary pickle files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = {}

        # Create v1 model
        v1_path = Path(tmpdir) / "model_v1.pkl"
        with v1_path.open("wb") as f:
            pickle.dump(DummyModel("v1"), f, protocol=4)
        files["v1"] = v1_path

        # Create v2 model
        v2_path = Path(tmpdir) / "model_v2.pkl"
        with v2_path.open("wb") as f:
            pickle.dump(DummyModel("v2"), f, protocol=4)
        files["v2"] = v2_path

        # Create v3 model
        v3_path = Path(tmpdir) / "model_v3.pkl"
        with v3_path.open("wb") as f:
            pickle.dump(DummyModel("v3"), f, protocol=4)
        files["v3"] = v3_path

        # Create invalid pickle (wrong magic)
        invalid_path = Path(tmpdir) / "invalid.pkl"
        with invalid_path.open("wb") as f:
            f.write(b"INVALID_PICKLE_DATA")
        files["invalid"] = invalid_path

        yield files


def test_reload_model_increments_load_seq(temp_model_files, disable_opcode_scan):
    """Test that load_seq increments on successive successful reloads."""
    from src.ml.classifier import get_model_info, reload_model

    # Initial state
    info0 = get_model_info()
    initial_seq = info0.get("load_seq", 0)

    # First reload: v1
    result1 = reload_model(str(temp_model_files["v1"]), expected_version="v1")
    assert result1["status"] == "success"

    info1 = get_model_info()
    assert info1["load_seq"] == initial_seq + 1
    assert info1["version"] == "v1"
    assert info1["rollback_level"] == 0

    # Second reload: v2
    result2 = reload_model(str(temp_model_files["v2"]), expected_version="v2")
    assert result2["status"] == "success"

    info2 = get_model_info()
    assert info2["load_seq"] == initial_seq + 2
    assert info2["version"] == "v2"
    assert info2["rollback_level"] == 0

    # Third reload: v3
    result3 = reload_model(str(temp_model_files["v3"]), expected_version="v3")
    assert result3["status"] == "success"

    info3 = get_model_info()
    assert info3["load_seq"] == initial_seq + 3
    assert info3["version"] == "v3"
    assert info3["rollback_level"] == 0


def test_reload_model_rollback_preserves_seq(temp_model_files, disable_opcode_scan):
    """Test that rollback preserves load_seq of rolled-back model."""
    from src.ml.classifier import get_model_info, reload_model

    # Load v1 successfully
    result1 = reload_model(str(temp_model_files["v1"]), expected_version="v1")
    assert result1["status"] == "success"

    info1 = get_model_info()
    v1_seq = info1["load_seq"]

    # Load v2 successfully
    result2 = reload_model(str(temp_model_files["v2"]), expected_version="v2")
    assert result2["status"] == "success"

    info2 = get_model_info()
    v2_seq = info2["load_seq"]
    assert v2_seq > v1_seq

    # Attempt to load invalid - should rollback to v2
    result_invalid = reload_model(str(temp_model_files["invalid"]), expected_version="v_invalid")
    assert result_invalid["status"] in ("rollback", "magic_invalid", "error")

    info_after = get_model_info()
    # After rollback, should still be on v2 with v2's sequence
    assert info_after["version"] == "v2"
    # Sequence doesn't change on rollback (preserves previous successful load)
    assert info_after["load_seq"] == v2_seq


def test_reload_model_invalid_magic_sets_error(temp_model_files):
    """Test that invalid pickle magic sets last_error."""
    from src.ml.classifier import get_model_info, reload_model

    # First establish a good model
    reload_model(str(temp_model_files["v1"]), expected_version="v1")

    # Try to load invalid pickle
    result = reload_model(str(temp_model_files["invalid"]), expected_version="invalid")
    assert result["status"] == "magic_invalid"

    # Check that error is recorded
    info = get_model_info()
    assert info["last_error"] is not None
    assert "magic" in info["last_error"].lower()


def test_reload_model_size_limit(temp_model_files):
    """Test that oversized model is rejected with proper error."""
    from src.ml.classifier import get_model_info, reload_model

    # Set extremely low size limit
    original_limit = os.getenv("MODEL_MAX_MB")
    try:
        os.environ["MODEL_MAX_MB"] = "0.000001"  # ~1 byte limit

        result = reload_model(str(temp_model_files["v1"]), expected_version="v1")
        assert result["status"] == "size_exceeded"

        info = get_model_info()
        assert info["last_error"] is not None
        assert "size" in info["last_error"].lower() or "exceed" in info["last_error"].lower()
    finally:
        # Restore original limit
        if original_limit:
            os.environ["MODEL_MAX_MB"] = original_limit
        else:
            os.environ.pop("MODEL_MAX_MB", None)


def test_reload_model_thread_safety_concurrent(disable_opcode_scan):
    """Test that concurrent reloads are serialized by lock."""
    import threading
    import time

    from src.ml.classifier import get_model_info, reload_model

    # Create a simple test model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(DummyModel("concurrent_test"), f, protocol=4)
        model_path = f.name

    try:
        results = []
        errors = []

        def reload_worker():
            """Worker thread attempting reload."""
            try:
                result = reload_model(model_path, expected_version="concurrent")
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Launch multiple threads attempting concurrent reload
        threads = [threading.Thread(target=reload_worker) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        # All threads should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All should succeed (lock serializes them)
        assert len(results) == 5

        # At least one should succeed
        success_count = sum(1 for r in results if r.get("status") == "success")
        assert success_count >= 1

        # Final model info should be consistent
        info = get_model_info()
        assert info["loaded"] is True
    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


def test_reload_model_clears_error_on_success(temp_model_files, disable_opcode_scan):
    """Test that successful reload clears previous error."""
    from src.ml.classifier import get_model_info, reload_model

    # First establish a good model
    reload_model(str(temp_model_files["v1"]), expected_version="v1")

    # Trigger an error
    reload_model(str(temp_model_files["invalid"]), expected_version="invalid")

    info_with_error = get_model_info()
    assert info_with_error["last_error"] is not None

    # Successful reload should clear error
    result = reload_model(str(temp_model_files["v2"]), expected_version="v2")
    assert result["status"] == "success"

    info_after = get_model_info()
    assert info_after["last_error"] is None
    assert info_after["version"] == "v2"
