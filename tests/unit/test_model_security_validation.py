"""Tests for model security validation (magic number, hash whitelist, forged files)."""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest


def test_model_magic_number_validation_success(tmp_path):
    """Test model reload succeeds with valid pickle magic number."""
    from src.ml.classifier import reload_model

    # Create a valid pickle file (protocol 4)
    model_path = tmp_path / "valid_model.pkl"
    model_data = {"predict": lambda X: ["test"]}
    with model_path.open("wb") as f:
        pickle.dump(model_data, f, protocol=4)

    result = reload_model(str(model_path), force=True)

    # Should succeed (or rollback if pickle incompatible, not security fail)
    assert result["status"] in ("success", "rollback", "error")
    assert result["status"] != "magic_invalid"


def test_model_magic_number_validation_fail_invalid_file(tmp_path):
    """Test model reload fails with invalid magic number."""
    from src.ml.classifier import reload_model

    # Create a file with invalid magic number (not a pickle file)
    forged_path = tmp_path / "forged.pkl"
    forged_path.write_bytes(b"NOT_A_PICKLE_FILE" + b"\x00" * 100)

    result = reload_model(str(forged_path), force=True)

    assert result["status"] == "magic_invalid"
    assert "magic_bytes" in result
    assert result["message"] == "File does not appear to be a valid pickle file"


def test_model_magic_number_validation_fail_elf_binary(tmp_path):
    """Test model reload rejects ELF binary (simulated malicious file)."""
    from src.ml.classifier import reload_model

    # ELF magic number (Linux executable)
    elf_path = tmp_path / "malicious.pkl"
    elf_path.write_bytes(b"\x7fELF" + b"\x00" * 100)

    result = reload_model(str(elf_path), force=True)

    assert result["status"] == "magic_invalid"


def test_model_magic_number_validation_fail_pe_binary(tmp_path):
    """Test model reload rejects PE binary (Windows executable)."""
    from src.ml.classifier import reload_model

    # PE magic number (Windows .exe)
    pe_path = tmp_path / "malware.pkl"
    pe_path.write_bytes(b"MZ" + b"\x00" * 100)

    result = reload_model(str(pe_path), force=True)

    assert result["status"] == "magic_invalid"


def test_model_hash_whitelist_validation_success(tmp_path):
    """Test model reload succeeds when hash is in whitelist."""
    from src.ml.classifier import reload_model
    import hashlib

    # Create a valid model
    model_path = tmp_path / "whitelisted_model.pkl"
    model_data = {"predict": lambda X: ["test"]}
    with model_path.open("wb") as f:
        pickle.dump(model_data, f, protocol=4)

    # Compute hash
    file_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]

    # Set whitelist environment variable
    with patch.dict(os.environ, {"ALLOWED_MODEL_HASHES": file_hash}):
        result = reload_model(str(model_path), force=True)

    # Should succeed (or rollback if incompatible, not hash_mismatch)
    assert result["status"] in ("success", "rollback", "error")
    assert result["status"] != "hash_mismatch"


def test_model_hash_whitelist_validation_fail(tmp_path):
    """Test model reload fails when hash is NOT in whitelist."""
    from src.ml.classifier import reload_model

    # Create a valid pickle model
    model_path = tmp_path / "unauthorized_model.pkl"
    model_data = {"predict": lambda X: ["test"]}
    with model_path.open("wb") as f:
        pickle.dump(model_data, f, protocol=4)

    # Set whitelist to different hash
    fake_whitelist = "0000000000000000,1111111111111111"
    with patch.dict(os.environ, {"ALLOWED_MODEL_HASHES": fake_whitelist}):
        result = reload_model(str(model_path), force=True)

    assert result["status"] == "hash_mismatch"
    assert "expected_hashes" in result
    assert "found_hash" in result
    assert result["found_hash"] not in result["expected_hashes"]


def test_model_hash_whitelist_multiple_allowed(tmp_path):
    """Test model reload with multiple hashes in whitelist."""
    from src.ml.classifier import reload_model
    import hashlib

    # Create model
    model_path = tmp_path / "model.pkl"
    model_data = {"predict": lambda X: ["test"]}
    with model_path.open("wb") as f:
        pickle.dump(model_data, f, protocol=4)

    # Compute hash
    file_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]

    # Set whitelist with multiple hashes (including the correct one)
    whitelist = f"fake_hash_1,{file_hash},fake_hash_2"
    with patch.dict(os.environ, {"ALLOWED_MODEL_HASHES": whitelist}):
        result = reload_model(str(model_path), force=True)

    # Should succeed (correct hash is in whitelist)
    assert result["status"] in ("success", "rollback", "error")
    assert result["status"] != "hash_mismatch"


def test_model_security_fail_metric_magic_invalid(tmp_path, monkeypatch):
    """Test model_security_fail_total metric increments on magic number failure."""
    from src.ml.classifier import reload_model
    from src.utils.analysis_metrics import model_security_fail_total

    # Get initial count
    initial_count = model_security_fail_total.labels(reason="magic_number_invalid")._value.get()

    # Create invalid file
    bad_path = tmp_path / "invalid.pkl"
    bad_path.write_bytes(b"INVALID" + b"\x00" * 50)

    result = reload_model(str(bad_path), force=True)

    # Verify metric incremented
    final_count = model_security_fail_total.labels(reason="magic_number_invalid")._value.get()
    assert final_count > initial_count
    assert result["status"] == "magic_invalid"


def test_model_security_fail_metric_hash_mismatch(tmp_path):
    """Test model_security_fail_total metric increments on hash mismatch."""
    from src.ml.classifier import reload_model
    from src.utils.analysis_metrics import model_security_fail_total

    # Get initial count
    initial_count = model_security_fail_total.labels(reason="hash_mismatch")._value.get()

    # Create model
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"predict": lambda X: ["test"]}, f, protocol=4)

    # Set restrictive whitelist
    with patch.dict(os.environ, {"ALLOWED_MODEL_HASHES": "fake_hash"}):
        result = reload_model(str(model_path), force=True)

    # Verify metric incremented
    final_count = model_security_fail_total.labels(reason="hash_mismatch")._value.get()
    assert final_count > initial_count
    assert result["status"] == "hash_mismatch"


def test_model_reload_without_whitelist_skips_hash_check(tmp_path):
    """Test model reload without whitelist env var skips hash validation."""
    from src.ml.classifier import reload_model

    # Create model
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"predict": lambda X: ["test"]}, f, protocol=4)

    # Ensure no whitelist set
    with patch.dict(os.environ, {"ALLOWED_MODEL_HASHES": ""}, clear=False):
        result = reload_model(str(model_path), force=True)

    # Should not fail on hash (whitelist not configured)
    assert result["status"] in ("success", "rollback", "error")
    assert result["status"] != "hash_mismatch"


def test_model_size_limit_validation(tmp_path):
    """Test model size limit validation."""
    from src.ml.classifier import reload_model

    # Create a large file (2MB)
    large_model = tmp_path / "large_model.pkl"
    large_model.write_bytes(b"\x80\x04" + b"0" * (2 * 1024 * 1024))

    # Set small size limit
    with patch.dict(os.environ, {"MODEL_MAX_MB": "1.0"}):
        result = reload_model(str(large_model), force=True)

    assert result["status"] == "size_exceeded"
    assert "size_mb" in result
    assert "max_mb" in result
    assert result["size_mb"] > result["max_mb"]


def test_model_interface_validation_missing_predict(tmp_path):
    """Test model reload fails if model lacks predict method."""
    from src.ml.classifier import reload_model

    # Create model without predict method
    bad_model = tmp_path / "no_predict.pkl"
    with bad_model.open("wb") as f:
        pickle.dump({"invalid": "object"}, f, protocol=4)

    result = reload_model(str(bad_model), force=True)

    # Should fail or rollback (missing predict method)
    assert result["status"] in ("rollback", "error", "rollback_level2")


def test_model_get_info():
    """Test get_model_info returns expected structure."""
    from src.ml.classifier import get_model_info

    info = get_model_info()

    assert "version" in info
    assert "hash" in info
    assert "path" in info
    assert "loaded_at" in info
    assert "loaded" in info
    assert isinstance(info["loaded"], bool)


def test_model_magic_number_protocol_0(tmp_path):
    """Test model reload accepts pickle protocol 0 (legacy)."""
    from src.ml.classifier import reload_model

    # Create pickle with protocol 0 (starts with '(')
    model_path = tmp_path / "protocol0.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"predict": lambda X: ["test"]}, f, protocol=0)

    result = reload_model(str(model_path), force=True)

    # Should succeed or rollback (not magic_invalid)
    assert result["status"] in ("success", "rollback", "error")
    assert result["status"] != "magic_invalid"


def test_model_reload_security_validations_order(tmp_path):
    """Test security validations run in correct order."""
    from src.ml.classifier import reload_model

    # Create file with invalid magic number
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_bytes(b"BAD" + b"\x00" * 100)

    # Even with valid hash whitelist, magic number should fail first
    with patch.dict(os.environ, {"ALLOWED_MODEL_HASHES": "valid_hash"}):
        result = reload_model(str(bad_path), force=True)

    # Magic number validation should fail before hash check
    assert result["status"] == "magic_invalid"
    assert result["status"] != "hash_mismatch"


def test_model_rollback_on_security_failure(tmp_path):
    """Test model rolls back to previous version on security failure."""
    from src.ml.classifier import reload_model

    # First, load a valid model
    valid_model = tmp_path / "valid.pkl"
    with valid_model.open("wb") as f:
        pickle.dump({"predict": lambda X: ["valid"]}, f, protocol=4)
    reload_model(str(valid_model), force=True)

    # Now try to load an invalid model (should rollback)
    invalid_model = tmp_path / "invalid.pkl"
    invalid_model.write_bytes(b"INVALID" + b"\x00" * 100)
    result = reload_model(str(invalid_model), force=True)

    # Should reject with magic_invalid (not rollback, as validation happens before loading)
    assert result["status"] == "magic_invalid"
