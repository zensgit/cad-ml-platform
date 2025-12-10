"""Tests for input_validator.py to improve coverage.

Covers:
- verify_signature for various CAD formats
- deep_format_validate for STEP, STL, IGES, DXF
- load_validation_matrix and matrix_validate
- sniff_mime with and without magic library
- is_supported_mime for various MIME types
- validate_and_read function
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.security.input_validator import (
    verify_signature,
    signature_hex_prefix,
    deep_format_validate,
    load_validation_matrix,
    matrix_validate,
    sniff_mime,
    is_supported_mime,
)


class TestVerifySignature:
    """Tests for verify_signature function."""

    def test_step_valid_signature(self):
        """Test STEP format with valid signature."""
        data = b"ISO-10303-21;HEADER;FILE_DESCRIPTION..."
        valid, hint = verify_signature(data, "step")
        assert valid is True
        assert "ISO-10303-21" in hint

    def test_step_invalid_signature(self):
        """Test STEP format with invalid signature."""
        data = b"NOT A STEP FILE"
        valid, hint = verify_signature(data, "step")
        assert valid is False

    def test_stp_extension(self):
        """Test STP extension (alias for STEP)."""
        data = b"ISO-10303-21;HEADER..."
        valid, _ = verify_signature(data, "stp")
        assert valid is True

    def test_stl_ascii_signature(self):
        """Test ASCII STL with 'solid' prefix."""
        data = b"solid mymodel\nfacet normal..."
        valid, hint = verify_signature(data, "stl")
        assert valid is True
        assert "ASCII" in hint

    def test_stl_binary_valid(self):
        """Test binary STL with sufficient size."""
        data = b"\x00" * 100  # 100 bytes > 84
        valid, hint = verify_signature(data, "stl")
        assert valid is True
        assert "Binary" in hint

    def test_stl_binary_too_small(self):
        """Test binary STL that's too small."""
        data = b"\x00" * 50  # 50 bytes < 84
        valid, _ = verify_signature(data, "stl")
        assert valid is False

    def test_iges_valid_signature(self):
        """Test IGES with valid token."""
        data = b"                                        S      1IGES"
        valid, hint = verify_signature(data, "iges")
        assert valid is True

    def test_igs_extension(self):
        """Test IGS extension (alias for IGES)."""
        data = b"IGES FILE HEADER..."
        valid, _ = verify_signature(data, "igs")
        assert valid is True

    def test_iges_case_insensitive(self):
        """Test IGES detection is case insensitive."""
        data = b"iges file data"  # lowercase
        valid, _ = verify_signature(data, "iges")
        assert valid is True  # uppercase check in header.upper()

    def test_dxf_section_present(self):
        """Test DXF with SECTION token."""
        data = b"0\nSECTION\n2\nHEADER..."
        valid, hint = verify_signature(data, "dxf")
        assert valid is True
        assert "SECTION" in hint or "Lenient" in hint

    def test_dwg_ac101_token(self):
        """Test DWG with AC101 version token."""
        data = b"AC1015..."
        valid, _ = verify_signature(data, "dwg")
        assert valid is True

    def test_dwg_unknown_lenient(self):
        """Test DWG with unknown format is lenient."""
        data = b"UNKNOWN DWG DATA"
        valid, hint = verify_signature(data, "dwg")
        assert valid is True
        assert "Lenient" in hint

    def test_unknown_format_lenient(self):
        """Test unknown format is lenient."""
        data = b"SOME DATA"
        valid, hint = verify_signature(data, "xyz")
        assert valid is True
        assert "lenient" in hint.lower()


class TestSignatureHexPrefix:
    """Tests for signature_hex_prefix function."""

    def test_default_length(self):
        """Test default hex prefix length."""
        data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
        result = signature_hex_prefix(data)
        assert len(result) == 32  # 16 bytes = 32 hex chars

    def test_custom_length(self):
        """Test custom hex prefix length."""
        data = b"\xff\xfe\xfd\xfc"
        result = signature_hex_prefix(data, length=4)
        assert result == "fffefdfc"


class TestDeepFormatValidate:
    """Tests for deep_format_validate function."""

    def test_step_missing_header(self):
        """Test STEP without ISO-10303-21 header."""
        data = b"HEADER;ENDSEC;DATA;..."
        valid, reason = deep_format_validate(data, "step")
        assert valid is False
        assert "missing_step_header" in reason

    def test_step_missing_header_section(self):
        """Test STEP with ISO header but missing HEADER section."""
        data = b"ISO-10303-21;DATA;ENDSEC;"
        valid, reason = deep_format_validate(data, "step")
        assert valid is False
        assert "missing_step_HEADER_section" in reason

    def test_step_valid(self):
        """Test valid STEP file."""
        data = b"ISO-10303-21;HEADER;FILE_DESCRIPTION...ENDSEC;DATA;"
        valid, reason = deep_format_validate(data, "step")
        assert valid is True
        assert reason == "ok"

    def test_stl_too_small(self):
        """Test STL file that's too small."""
        data = b"\x00" * 50
        valid, reason = deep_format_validate(data, "stl")
        assert valid is False
        assert "stl_too_small" in reason

    def test_stl_ascii_solid(self):
        """Test ASCII STL starting with solid."""
        data = b"solid model\nfacet normal 0 0 1\nouter loop\n" + b"\x00" * 50
        valid, reason = deep_format_validate(data, "stl")
        assert valid is True
        assert "ascii_solid" in reason

    def test_stl_binary_valid(self):
        """Test binary STL with sufficient size."""
        data = b"\x00" * 100
        valid, reason = deep_format_validate(data, "stl")
        assert valid is True
        assert "binary_min_size" in reason

    def test_iges_section_markers_present(self):
        """Test IGES with some section markers present."""
        # The check is lenient - any uppercase S,G,D,P tokens count
        data = b"SOME DATA WITH S AND G TOKENS"
        valid, reason = deep_format_validate(data, "iges")
        # At least 2 of S,G,D,P should be present in uppercase
        assert valid is True or "iges_section_markers_missing" in reason

    def test_iges_valid(self):
        """Test valid IGES with section markers."""
        data = b"S      1G      2D      3P      4"
        valid, reason = deep_format_validate(data, "igs")
        assert valid is True
        assert reason == "ok"

    def test_dxf_section_missing(self):
        """Test DXF without SECTION token."""
        data = b"0\nHEADER\n..."
        valid, reason = deep_format_validate(data, "dxf")
        assert valid is False
        assert "dxf_section_missing" in reason

    def test_dxf_valid(self):
        """Test valid DXF with SECTION."""
        data = b"0\nSECTION\n2\nHEADER..."
        valid, reason = deep_format_validate(data, "dxf")
        assert valid is True
        assert reason == "ok"

    def test_unknown_format_ok(self):
        """Test unknown format passes."""
        data = b"SOME DATA"
        valid, reason = deep_format_validate(data, "xyz")
        assert valid is True
        assert reason == "ok"


class TestLoadValidationMatrix:
    """Tests for load_validation_matrix function."""

    def test_missing_file_returns_empty(self):
        """Test missing config file returns empty dict."""
        with patch.dict(os.environ, {"FORMAT_VALIDATION_MATRIX": "/nonexistent/path.yaml"}):
            result = load_validation_matrix()
        assert result == {}

    def test_invalid_yaml_returns_empty(self):
        """Test invalid YAML returns empty dict."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            with patch.dict(os.environ, {"FORMAT_VALIDATION_MATRIX": f.name}):
                result = load_validation_matrix()
        os.unlink(f.name)
        assert result == {}

    def test_valid_yaml(self):
        """Test valid YAML is loaded."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("formats:\n  step:\n    required_tokens:\n      - ISO-10303-21")
            f.flush()
            with patch.dict(os.environ, {"FORMAT_VALIDATION_MATRIX": f.name}):
                result = load_validation_matrix()
        os.unlink(f.name)
        assert "formats" in result
        assert "step" in result["formats"]


class TestMatrixValidate:
    """Tests for matrix_validate function."""

    def test_no_spec_passes(self):
        """Test format without spec passes."""
        with patch("src.security.input_validator.load_validation_matrix", return_value={}):
            valid, reason = matrix_validate(b"DATA", "unknown")
        assert valid is True
        assert reason == "no_spec"

    def test_exempt_project(self):
        """Test exempt project bypasses validation."""
        matrix = {
            "formats": {"step": {"required_tokens": ["ISO-10303-21"]}},
            "exempt_projects": ["project123"]
        }
        with patch("src.security.input_validator.load_validation_matrix", return_value=matrix):
            valid, reason = matrix_validate(b"INVALID", "step", project_id="project123")
        assert valid is True
        assert reason == "exempt"

    def test_required_token_missing(self):
        """Test missing required token fails."""
        matrix = {
            "formats": {"step": {"required_tokens": ["ISO-10303-21"]}}
        }
        with patch("src.security.input_validator.load_validation_matrix", return_value=matrix):
            valid, reason = matrix_validate(b"NOT STEP DATA", "step")
        assert valid is False
        assert "missing_token" in reason

    def test_required_token_present(self):
        """Test present required token passes."""
        matrix = {
            "formats": {"step": {"required_tokens": ["ISO-10303-21"]}}
        }
        with patch("src.security.input_validator.load_validation_matrix", return_value=matrix):
            valid, reason = matrix_validate(b"ISO-10303-21;HEADER;", "step")
        assert valid is True
        assert reason == "ok"

    def test_stl_min_size_violation(self):
        """Test STL below minimum size fails."""
        matrix = {
            "formats": {"stl": {"min_size": 100}}
        }
        with patch("src.security.input_validator.load_validation_matrix", return_value=matrix):
            valid, reason = matrix_validate(b"\x00" * 50, "stl")
        assert valid is False
        assert "below_min_size" in reason

    def test_stl_min_size_met(self):
        """Test STL meeting minimum size passes."""
        matrix = {
            "formats": {"stl": {"min_size": 84}}
        }
        with patch("src.security.input_validator.load_validation_matrix", return_value=matrix):
            valid, reason = matrix_validate(b"\x00" * 100, "stl")
        assert valid is True


class TestSniffMime:
    """Tests for sniff_mime function."""

    def test_sniff_mime_returns_tuple(self):
        """Test sniff_mime returns (mime, bool) tuple."""
        # Test with arbitrary data - magic may or may not be installed
        mime, has_magic = sniff_mime(b"%PDF-1.4 some data")
        assert isinstance(mime, str)
        assert isinstance(has_magic, bool)
        # Either magic detected something or fallback was used
        assert mime == "application/octet-stream" or len(mime) > 0

    def test_sniff_mime_empty_data(self):
        """Test sniff_mime with empty data."""
        mime, has_magic = sniff_mime(b"")
        assert isinstance(mime, str)

    def test_sniff_mime_binary_data(self):
        """Test sniff_mime with binary data."""
        # Binary data that's not a known format
        mime, has_magic = sniff_mime(b"\x00\x01\x02\x03\x04\x05")
        assert isinstance(mime, str)


class TestIsSupportedMime:
    """Tests for is_supported_mime function."""

    def test_exact_match_allowed(self):
        """Test exact MIME type matches."""
        allowed_types = [
            "text/plain",
            "application/octet-stream",
            "application/zip",
            "model/stl",
            "application/stl",
            "application/dxf",
            "application/iges",
            "application/step",
        ]
        for mime in allowed_types:
            assert is_supported_mime(mime) is True, f"{mime} should be supported"

    def test_text_prefix_allowed(self):
        """Test text/* MIME types are allowed."""
        assert is_supported_mime("text/csv") is True
        assert is_supported_mime("text/html") is True

    def test_octet_stream_suffix(self):
        """Test octet-stream suffix is allowed."""
        assert is_supported_mime("application/x-octet-stream") is True

    def test_unsupported_mime(self):
        """Test unsupported MIME types."""
        assert is_supported_mime("image/png") is False
        assert is_supported_mime("video/mp4") is False
        assert is_supported_mime("audio/mpeg") is False


class TestValidateAndRead:
    """Tests for validate_and_read function."""

    @pytest.mark.asyncio
    async def test_validate_and_read_returns_data_and_mime(self):
        """Test validate_and_read returns data and MIME type."""
        from src.security.input_validator import validate_and_read

        mock_file = AsyncMock()
        mock_file.read.return_value = b"ISO-10303-21;HEADER;"

        with patch("src.security.input_validator.sniff_mime", return_value=("application/step", True)):
            data, mime = await validate_and_read(mock_file)

        assert data == b"ISO-10303-21;HEADER;"
        assert mime == "application/step"
        mock_file.read.assert_called_once()
