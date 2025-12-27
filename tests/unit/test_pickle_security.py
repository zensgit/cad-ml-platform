"""Tests for pickle security scanner with audit, blocklist, and whitelist modes.

Verifies that:
1. Audit mode logs but never blocks
2. Blocklist mode blocks dangerous opcodes
3. Whitelist mode only allows safe opcodes
4. Security configuration is properly exposed
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.pickle_security import (
    DANGEROUS_OPCODES,
    SAFE_OPCODES,
    OpcodeMode,
    audit_pickle_directory,
    get_opcode_mode_from_env,
    get_security_config,
    scan_pickle_opcodes,
    validate_pickle_file,
)


class TestOpcodeMode:
    """Test OpcodeMode enum."""

    def test_audit_mode_value(self):
        """Test audit mode string value."""
        assert OpcodeMode.AUDIT.value == "audit"

    def test_blocklist_mode_value(self):
        """Test blocklist mode string value."""
        assert OpcodeMode.BLOCKLIST.value == "blocklist"

    def test_whitelist_mode_value(self):
        """Test whitelist mode string value."""
        assert OpcodeMode.WHITELIST.value == "whitelist"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert OpcodeMode("audit") == OpcodeMode.AUDIT
        assert OpcodeMode("blocklist") == OpcodeMode.BLOCKLIST
        assert OpcodeMode("whitelist") == OpcodeMode.WHITELIST


class TestSafePickleScanning:
    """Test scanning safe pickle data."""

    @pytest.fixture
    def safe_pickle_data(self):
        """Create safe pickle data with only basic types."""
        data = {
            "name": "test",
            "value": 42,
            "items": [1, 2, 3],
            "nested": {"a": True, "b": False},
        }
        return pickle.dumps(data, protocol=4)

    def test_safe_data_passes_audit_mode(self, safe_pickle_data):
        """Test safe data passes audit mode."""
        result = scan_pickle_opcodes(safe_pickle_data, mode=OpcodeMode.AUDIT)

        assert result["safe"] is True
        assert result["mode"] == "audit"
        assert len(result["opcodes"]) > 0
        assert result["blocked_reason"] is None

    def test_safe_data_passes_blocklist_mode(self, safe_pickle_data):
        """Test safe data passes blocklist mode."""
        result = scan_pickle_opcodes(safe_pickle_data, mode=OpcodeMode.BLOCKLIST)

        assert result["safe"] is True
        assert result["mode"] == "blocklist"
        assert result["dangerous"] == []
        assert result["blocked_reason"] is None

    def test_safe_data_passes_whitelist_mode(self, safe_pickle_data):
        """Test safe data passes whitelist mode."""
        result = scan_pickle_opcodes(safe_pickle_data, mode=OpcodeMode.WHITELIST)

        assert result["safe"] is True
        assert result["mode"] == "whitelist"
        assert result["disallowed"] == []
        assert result["blocked_reason"] is None

    def test_opcode_counts_recorded(self, safe_pickle_data):
        """Test opcode counts are recorded."""
        result = scan_pickle_opcodes(safe_pickle_data)

        assert len(result["opcode_counts"]) > 0
        assert sum(result["opcode_counts"].values()) == len(result["opcodes"])

    def test_positions_included_when_requested(self, safe_pickle_data):
        """Test position info included when requested."""
        result = scan_pickle_opcodes(safe_pickle_data, include_positions=True)

        assert "positions" in result
        assert len(result["positions"]) > 0
        assert all("opcode" in p for p in result["positions"])
        assert all("position" in p for p in result["positions"])


class TestDangerousPickleScanning:
    """Test scanning dangerous pickle data."""

    @pytest.fixture
    def dangerous_pickle_data(self):
        """Create pickle data with dangerous opcodes (simulated).

        Note: Actually creating dangerous pickles is complex; we create
        a simple pickle and verify the scanning logic works.
        """

        # This creates a pickle with REDUCE opcode (via __reduce__)
        class DangerousClass:
            def __reduce__(self):
                return (print, ("malicious",))

        obj = DangerousClass()
        return pickle.dumps(obj, protocol=4)

    def test_audit_mode_never_blocks(self, dangerous_pickle_data):
        """Test audit mode logs but never blocks."""
        result = scan_pickle_opcodes(dangerous_pickle_data, mode=OpcodeMode.AUDIT)

        # Audit mode always returns safe=True (observation only)
        assert result["safe"] is True
        assert result["mode"] == "audit"
        # Opcodes are still recorded
        assert len(result["opcodes"]) > 0

    def test_blocklist_mode_blocks_dangerous(self, dangerous_pickle_data):
        """Test blocklist mode blocks dangerous opcodes."""
        result = scan_pickle_opcodes(dangerous_pickle_data, mode=OpcodeMode.BLOCKLIST)

        # Should detect REDUCE and/or GLOBAL opcodes
        assert result["safe"] is False or len(result["dangerous"]) > 0
        assert result["mode"] == "blocklist"

    def test_whitelist_mode_blocks_unlisted(self, dangerous_pickle_data):
        """Test whitelist mode blocks unlisted opcodes."""
        result = scan_pickle_opcodes(dangerous_pickle_data, mode=OpcodeMode.WHITELIST)

        # Whitelist is very restrictive
        # Dangerous pickles typically use opcodes not in the safe list
        assert result["safe"] is False or len(result["disallowed"]) > 0


class TestDangerousOpcodesList:
    """Test dangerous opcodes constants."""

    def test_dangerous_opcodes_contains_reduce(self):
        """Test REDUCE is in dangerous list."""
        assert "REDUCE" in DANGEROUS_OPCODES

    def test_dangerous_opcodes_contains_global(self):
        """Test GLOBAL is in dangerous list."""
        assert "GLOBAL" in DANGEROUS_OPCODES

    def test_dangerous_opcodes_contains_build(self):
        """Test BUILD is in dangerous list."""
        assert "BUILD" in DANGEROUS_OPCODES

    def test_dangerous_opcodes_contains_inst(self):
        """Test INST is in dangerous list."""
        assert "INST" in DANGEROUS_OPCODES

    def test_safe_opcodes_not_empty(self):
        """Test safe opcodes list is not empty."""
        assert len(SAFE_OPCODES) > 0

    def test_no_overlap_dangerous_safe(self):
        """Test no overlap between dangerous and safe lists."""
        overlap = DANGEROUS_OPCODES & SAFE_OPCODES
        assert len(overlap) == 0


class TestFileValidation:
    """Test pickle file validation."""

    @pytest.fixture
    def temp_pickle_file(self):
        """Create a temporary safe pickle file."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            data = {"test": "data", "value": 123}
            pickle.dump(data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    def test_validate_existing_file(self, temp_pickle_file):
        """Test validating an existing pickle file."""
        result = validate_pickle_file(temp_pickle_file)

        assert result["exists"] is True
        assert result["safe"] is True
        assert "size_mb" in result
        assert result["size_mb"] >= 0  # Small files may round to 0

    def test_validate_nonexistent_file(self):
        """Test validating a nonexistent file."""
        result = validate_pickle_file("/nonexistent/path/file.pkl")

        assert result["exists"] is False
        assert result["safe"] is False
        assert "error" in result

    def test_validate_with_mode_override(self, temp_pickle_file):
        """Test validating with explicit mode."""
        result = validate_pickle_file(temp_pickle_file, mode=OpcodeMode.WHITELIST)

        assert result["mode"] == "whitelist"
        # Safe pickle should pass whitelist too
        assert result["safe"] is True

    def test_validate_oversized_file(self, temp_pickle_file):
        """Test validation rejects oversized files."""
        with patch.dict(os.environ, {"MODEL_MAX_MB": "0.000001"}):
            result = validate_pickle_file(temp_pickle_file)

            # Very small limit should reject even tiny files
            assert result["safe"] is False
            assert "exceeds limit" in result.get("error", "")


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""

    def test_get_mode_default(self):
        """Test default mode is blocklist."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MODEL_OPCODE_MODE", None)
            mode = get_opcode_mode_from_env()
            assert mode == OpcodeMode.BLOCKLIST

    def test_get_mode_from_env(self):
        """Test getting mode from environment."""
        with patch.dict(os.environ, {"MODEL_OPCODE_MODE": "audit"}):
            mode = get_opcode_mode_from_env()
            assert mode == OpcodeMode.AUDIT

        with patch.dict(os.environ, {"MODEL_OPCODE_MODE": "whitelist"}):
            mode = get_opcode_mode_from_env()
            assert mode == OpcodeMode.WHITELIST

    def test_invalid_mode_falls_back_to_blocklist(self):
        """Test invalid mode falls back to blocklist."""
        with patch.dict(os.environ, {"MODEL_OPCODE_MODE": "invalid"}):
            mode = get_opcode_mode_from_env()
            assert mode == OpcodeMode.BLOCKLIST


class TestSecurityConfig:
    """Test security configuration retrieval."""

    def test_get_security_config_structure(self):
        """Test security config has expected fields."""
        config = get_security_config()

        assert "opcode_mode" in config
        assert "opcode_scan_enabled" in config
        assert "opcode_strict" in config
        assert "magic_number_check" in config
        assert "max_model_size_mb" in config
        assert "hash_whitelist_enabled" in config
        assert "dangerous_opcodes" in config
        assert "safe_opcodes_count" in config

    def test_dangerous_opcodes_in_config(self):
        """Test dangerous opcodes list in config."""
        config = get_security_config()

        assert len(config["dangerous_opcodes"]) > 0
        assert "REDUCE" in config["dangerous_opcodes"]

    def test_safe_opcodes_count_positive(self):
        """Test safe opcodes count is positive."""
        config = get_security_config()

        assert config["safe_opcodes_count"] > 0


class TestDirectoryAudit:
    """Test directory audit functionality."""

    @pytest.fixture
    def temp_pickle_dir(self):
        """Create a temporary directory with pickle files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a few safe pickle files
            for i in range(3):
                path = Path(tmpdir) / f"model_{i}.pkl"
                with open(path, "wb") as f:
                    pickle.dump({"id": i, "data": list(range(i * 10))}, f)

            yield tmpdir

    def test_audit_directory(self, temp_pickle_dir):
        """Test auditing a directory of pickle files."""
        result = audit_pickle_directory(temp_pickle_dir)

        assert result["files_scanned"] == 3
        assert result["files_safe"] == 3
        assert result["files_unsafe"] == 0
        assert len(result["file_results"]) == 3

    def test_audit_directory_aggregates_opcodes(self, temp_pickle_dir):
        """Test audit aggregates opcode counts across files."""
        result = audit_pickle_directory(temp_pickle_dir)

        assert len(result["all_opcodes"]) > 0
        # Each file contributes opcodes, so counts should be > individual file

    def test_audit_empty_directory(self):
        """Test auditing an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = audit_pickle_directory(tmpdir)

            assert result["files_scanned"] == 0
            assert result["files_safe"] == 0
            assert result["files_unsafe"] == 0


class TestInputSources:
    """Test different input source types."""

    def test_scan_from_bytes(self):
        """Test scanning from bytes."""
        data = pickle.dumps({"test": 123})
        result = scan_pickle_opcodes(data)

        assert result["safe"] is True

    def test_scan_from_path_string(self):
        """Test scanning from path string."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"test": 456}, f)
            path = f.name

        try:
            result = scan_pickle_opcodes(path)
            assert result["safe"] is True
        finally:
            os.unlink(path)

    def test_scan_from_path_object(self):
        """Test scanning from Path object."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"test": 789}, f)
            path = Path(f.name)

        try:
            result = scan_pickle_opcodes(path)
            assert result["safe"] is True
        finally:
            path.unlink()

    def test_scan_from_file_object(self):
        """Test scanning from file-like object."""
        data = pickle.dumps({"test": "file_object"})
        file_obj = io.BytesIO(data)

        result = scan_pickle_opcodes(file_obj)
        assert result["safe"] is True


class TestErrorHandling:
    """Test error handling in scanning."""

    def test_scan_invalid_pickle_data(self):
        """Test scanning invalid pickle data."""
        result = scan_pickle_opcodes(b"not a pickle")

        assert result["safe"] is False
        assert "scan_error" in result or result["blocked_reason"] is not None

    def test_scan_truncated_pickle(self):
        """Test scanning truncated pickle data."""
        valid_data = pickle.dumps({"test": "data"})
        truncated = valid_data[: len(valid_data) // 2]

        result = scan_pickle_opcodes(truncated)

        # Should handle gracefully
        assert isinstance(result["safe"], bool)
        assert "opcodes" in result


class TestModeBehavior:
    """Test mode-specific behavior in detail."""

    @pytest.fixture
    def mixed_pickle_data(self):
        """Create pickle data that may contain various opcodes."""
        # Simple data structure that won't trigger dangerous opcodes
        data = [1, 2, 3, "test", {"nested": True}]
        return pickle.dumps(data, protocol=4)

    def test_audit_mode_records_all_opcodes(self, mixed_pickle_data):
        """Test audit mode records all opcodes without blocking."""
        result = scan_pickle_opcodes(mixed_pickle_data, mode=OpcodeMode.AUDIT)

        # Audit mode is always safe (observation only)
        assert result["safe"] is True
        # But still records opcodes
        assert len(result["opcodes"]) > 0
        assert len(result["opcode_counts"]) > 0

    def test_blocklist_mode_only_checks_dangerous(self, mixed_pickle_data):
        """Test blocklist mode only blocks dangerous opcodes."""
        result = scan_pickle_opcodes(mixed_pickle_data, mode=OpcodeMode.BLOCKLIST)

        # Safe data should pass
        assert result["safe"] is True
        # Dangerous list should be empty for safe data
        assert result["dangerous"] == []
        # Disallowed list is not used in blocklist mode
        assert result["disallowed"] == []

    def test_whitelist_mode_is_strictest(self, mixed_pickle_data):
        """Test whitelist mode is the strictest."""
        result = scan_pickle_opcodes(mixed_pickle_data, mode=OpcodeMode.WHITELIST)

        # Simple data should still pass whitelist
        assert result["safe"] is True
        # Any disallowed would be recorded
        # (for simple data, should be empty)


class TestMetricsIntegration:
    """Test that scan results can be used with metrics."""

    def test_result_structure_for_metrics(self):
        """Test result structure is suitable for metrics recording."""
        data = pickle.dumps({"test": True})
        result = scan_pickle_opcodes(data)

        # These fields should be usable for Prometheus metrics
        assert isinstance(result["safe"], bool)
        assert isinstance(result["mode"], str)
        assert isinstance(result["opcode_counts"], dict)

        # Counts should be numeric
        for opcode, count in result["opcode_counts"].items():
            assert isinstance(opcode, str)
            assert isinstance(count, int)
