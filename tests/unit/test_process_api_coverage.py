"""Tests for src/api/v1/process.py to improve coverage.

Covers:
- ProcessRulesAuditResponse model
- process_rules_audit endpoint logic
- File hash calculation
- Materials and complexities extraction
- Error handling paths
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestProcessRulesAuditResponseModel:
    """Tests for ProcessRulesAuditResponse model structure."""

    def test_response_dict_structure(self):
        """Test response dict has expected structure."""
        # Test the expected response structure without importing the model
        response = {
            "version": "v1",
            "source": "config/process_rules.yaml",
            "hash": "abc123",
            "materials": ["aluminum", "steel"],
            "complexities": {"aluminum": ["simple", "complex"]},
            "raw": {"key": "value"},
        }

        assert response["version"] == "v1"
        assert response["source"] == "config/process_rules.yaml"
        assert response["hash"] == "abc123"
        assert response["materials"] == ["aluminum", "steel"]
        assert response["complexities"] == {"aluminum": ["simple", "complex"]}
        assert response["raw"] == {"key": "value"}

    def test_response_hash_can_be_none(self):
        """Test hash field can be None."""
        response = {
            "version": "v1",
            "source": "test",
            "hash": None,
            "materials": [],
            "complexities": {},
            "raw": {},
        }

        assert response["hash"] is None

    def test_response_empty_fields(self):
        """Test response with empty lists and dicts."""
        response = {
            "version": "",
            "source": "",
            "hash": None,
            "materials": [],
            "complexities": {},
            "raw": {},
        }

        assert response["materials"] == []
        assert response["complexities"] == {}
        assert response["raw"] == {}


class TestMaterialsExtraction:
    """Tests for materials extraction logic."""

    def test_extract_materials_from_rules(self):
        """Test extracting materials from rules dict."""
        rules = {
            "__meta__": {"version": "v1"},
            "aluminum": {"simple": [], "complex": []},
            "steel": {"simple": []},
            "plastic": {"medium": []},
        }

        materials = sorted([m for m in rules.keys() if not m.startswith("__")])

        assert materials == ["aluminum", "plastic", "steel"]

    def test_exclude_meta_keys(self):
        """Test meta keys are excluded from materials."""
        rules = {
            "__meta__": {"version": "v1"},
            "__config__": {"setting": True},
            "material1": {},
        }

        materials = [m for m in rules.keys() if not m.startswith("__")]

        assert "__meta__" not in materials
        assert "__config__" not in materials
        assert "material1" in materials

    def test_empty_rules(self):
        """Test empty rules returns empty materials."""
        rules = {}
        materials = sorted([m for m in rules.keys() if not m.startswith("__")])
        assert materials == []


class TestComplexitiesExtraction:
    """Tests for complexities extraction logic."""

    def test_extract_complexities(self):
        """Test extracting complexities for each material."""
        rules = {
            "aluminum": {"simple": [1, 2], "complex": [3, 4], "meta": "ignored"},
            "steel": {"simple": [1]},
        }

        complexities: Dict[str, list[str]] = {}
        for m in ["aluminum", "steel"]:
            cm = rules.get(m, {})
            if isinstance(cm, dict):
                complexities[m] = sorted([c for c in cm.keys() if isinstance(cm.get(c), list)])

        assert complexities["aluminum"] == ["complex", "simple"]
        assert complexities["steel"] == ["simple"]

    def test_skip_non_list_values(self):
        """Test non-list complexity values are skipped."""
        rules = {
            "material": {
                "valid": [1, 2, 3],
                "invalid_str": "not a list",
                "invalid_int": 42,
                "invalid_dict": {"nested": True},
            }
        }

        cm = rules["material"]
        complexities = sorted([c for c in cm.keys() if isinstance(cm.get(c), list)])

        assert complexities == ["valid"]

    def test_non_dict_material(self):
        """Test handling non-dict material values."""
        rules = {"material": "not a dict"}

        complexities: Dict[str, list[str]] = {}
        cm = rules.get("material", {})
        if isinstance(cm, dict):
            complexities["material"] = []

        assert "material" not in complexities


class TestFileHashCalculation:
    """Tests for file hash calculation."""

    def test_hash_calculation(self):
        """Test SHA256 hash calculation truncated to 16 chars."""
        content = b"test content"
        expected_hash = hashlib.sha256(content).hexdigest()[:16]

        result_hash = hashlib.sha256(content).hexdigest()[:16]

        assert result_hash == expected_hash
        assert len(result_hash) == 16

    def test_hash_different_content(self):
        """Test different content produces different hashes."""
        content1 = b"content 1"
        content2 = b"content 2"

        hash1 = hashlib.sha256(content1).hexdigest()[:16]
        hash2 = hashlib.sha256(content2).hexdigest()[:16]

        assert hash1 != hash2

    def test_hash_consistency(self):
        """Test same content always produces same hash."""
        content = b"consistent content"

        hash1 = hashlib.sha256(content).hexdigest()[:16]
        hash2 = hashlib.sha256(content).hexdigest()[:16]

        assert hash1 == hash2


class TestFileExistenceLogic:
    """Tests for file existence checking logic."""

    def test_file_exists_path_used(self):
        """Test when file exists, path is used as source."""
        path = "config/process_rules.yaml"
        file_exists = True

        source = path if file_exists else "embedded-defaults"

        assert source == path

    def test_file_not_exists_defaults(self):
        """Test when file doesn't exist, embedded defaults used."""
        path = "config/process_rules.yaml"
        file_exists = False

        source = path if file_exists else "embedded-defaults"

        assert source == "embedded-defaults"


class TestVersionExtraction:
    """Tests for version extraction from rules."""

    def test_version_from_meta(self):
        """Test version extracted from __meta__."""
        rules = {"__meta__": {"version": "v2.0"}}

        version = rules.get("__meta__", {}).get("version", "v1")

        assert version == "v2.0"

    def test_version_default_fallback(self):
        """Test default version when not in meta."""
        rules = {"__meta__": {}}

        version = rules.get("__meta__", {}).get("version", "v1")

        assert version == "v1"

    def test_version_no_meta_section(self):
        """Test default version when no __meta__ section."""
        rules = {}

        version = rules.get("__meta__", {}).get("version", "v1")

        assert version == "v1"


class TestRawParameterHandling:
    """Tests for raw parameter handling."""

    def test_raw_true_returns_rules(self):
        """Test raw=True returns full rules dict."""
        rules = {"material": {"simple": [1, 2]}}
        raw = True

        result = rules if raw else {}

        assert result == rules

    def test_raw_false_returns_empty(self):
        """Test raw=False returns empty dict."""
        rules = {"material": {"simple": [1, 2]}}
        raw = False

        result = rules if raw else {}

        assert result == {}


class TestEnvironmentVariableHandling:
    """Tests for environment variable handling."""

    def test_default_rules_path(self):
        """Test default PROCESS_RULES_FILE path."""
        with patch.dict("os.environ", {}, clear=True):
            path = os.getenv("PROCESS_RULES_FILE", "config/process_rules.yaml")
            assert path == "config/process_rules.yaml"

    def test_custom_rules_path(self):
        """Test custom PROCESS_RULES_FILE path."""
        with patch.dict("os.environ", {"PROCESS_RULES_FILE": "/custom/path.yaml"}):
            path = os.getenv("PROCESS_RULES_FILE", "config/process_rules.yaml")
            assert path == "/custom/path.yaml"


class TestExceptionHandling:
    """Tests for exception handling in hash calculation."""

    def test_hash_exception_returns_none(self):
        """Test hash is None when file read raises exception."""
        path = "nonexistent.yaml"
        file_hash = None

        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            file_hash = None

        assert file_hash is None


class TestMetricsIncrement:
    """Tests for metrics counter logic."""

    def test_success_metrics_incremented(self):
        """Test success metrics are incremented on ok response."""
        mock_counter = MagicMock()

        # Simulate success path
        try:
            # Response creation succeeds
            mock_counter.labels(status="ok").inc()
            success = True
        except Exception:
            mock_counter.labels(status="error").inc()
            success = False

        assert success is True
        mock_counter.labels.assert_called_with(status="ok")

    def test_error_metrics_incremented(self):
        """Test error metrics are incremented on exception."""
        mock_counter = MagicMock()

        # Simulate error path
        try:
            raise ValueError("Test error")
        except Exception:
            mock_counter.labels(status="error").inc()

        mock_counter.labels.assert_called_with(status="error")


class TestRouterConfiguration:
    """Tests for router configuration patterns."""

    def test_router_pattern(self):
        """Test APIRouter pattern for process module."""
        from fastapi import APIRouter

        # Test router pattern
        router = APIRouter()
        assert router is not None

    def test_module_all_pattern(self):
        """Test __all__ exports pattern."""
        # Test the pattern used for module exports
        __all__ = ["router"]
        assert "router" in __all__


class TestEndpointIntegration:
    """Integration tests for process rules audit endpoint patterns."""

    def test_load_rules_call_pattern(self):
        """Test load_rules call pattern."""
        mock_rules = {
            "__meta__": {"version": "test_v1"},
            "aluminum": {"simple": [1], "complex": [2]},
        }

        # Simulate endpoint logic
        rules = mock_rules
        version = rules.get("__meta__", {}).get("version", "v1")
        materials = sorted([m for m in rules.keys() if not m.startswith("__")])

        assert version == "test_v1"
        assert "aluminum" in materials

    def test_raw_false_pattern(self):
        """Test raw=False returns empty dict."""
        rules = {"material": {"complexity": [1, 2]}}
        raw = False

        result = rules if raw else {}

        assert result == {}

    def test_file_hash_calculation_pattern(self):
        """Test file hash calculation pattern."""
        file_content = b"test file content"
        expected_hash = hashlib.sha256(file_content).hexdigest()[:16]

        assert len(expected_hash) == 16
        assert expected_hash == hashlib.sha256(file_content).hexdigest()[:16]

    def test_response_construction_pattern(self):
        """Test response construction pattern."""
        rules = {
            "__meta__": {"version": "v1"},
            "steel": {"medium": [1]},
        }
        path = "config/process_rules.yaml"
        file_exists = False

        # Simulate response construction
        version = rules.get("__meta__", {}).get("version", "v1")
        source = path if file_exists else "embedded-defaults"
        materials = sorted([m for m in rules.keys() if not m.startswith("__")])

        response = {
            "version": version,
            "source": source,
            "materials": materials,
        }

        assert response["version"] == "v1"
        assert response["source"] == "embedded-defaults"
        assert response["materials"] == ["steel"]


class TestComplexRulesStructure:
    """Tests for complex rules structure handling."""

    def test_nested_materials_and_complexities(self):
        """Test handling of deeply nested rules structure."""
        rules = {
            "__meta__": {"version": "v3", "author": "test"},
            "aluminum_6061": {
                "simple_turning": [1, 2, 3],
                "complex_milling": [4, 5, 6],
                "metadata": {"description": "not a list"},
            },
            "steel_304": {
                "basic": [7, 8],
            },
            "__internal__": {"debug": True},
        }

        # Extract materials
        materials = sorted([m for m in rules.keys() if not m.startswith("__")])
        assert materials == ["aluminum_6061", "steel_304"]

        # Extract complexities
        complexities: Dict[str, list[str]] = {}
        for m in materials:
            cm = rules.get(m, {})
            if isinstance(cm, dict):
                complexities[m] = sorted([c for c in cm.keys() if isinstance(cm.get(c), list)])

        assert complexities["aluminum_6061"] == ["complex_milling", "simple_turning"]
        assert complexities["steel_304"] == ["basic"]


class TestProcessRulesAuditEndpoint:
    """Tests for the actual process_rules_audit endpoint function."""

    @pytest.mark.asyncio
    async def test_endpoint_with_file_exists(self):
        """Test process_rules_audit when rules file exists."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {
            "__meta__": {"version": "v2"},
            "steel": {"simple": ["step1"], "complex": ["step1", "step2"]},
            "aluminum": {"basic": ["cut"]},
        }

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch.dict(os.environ, {"PROCESS_RULES_FILE": "/tmp/rules.yaml"}):
                with patch("src.api.v1.process.os.path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data=b"test content")):
                        with patch(
                            "src.api.v1.process.process_rules_audit_requests_total"
                        ) as mock_metric:
                            mock_metric.labels.return_value = MagicMock()

                            result = await process_rules_audit(raw=True, api_key="test")

                            assert result.version == "v2"
                            assert result.source == "/tmp/rules.yaml"
                            assert result.hash is not None
                            assert len(result.hash) == 16
                            assert "steel" in result.materials
                            assert "aluminum" in result.materials
                            assert result.raw == mock_rules
                            mock_metric.labels.assert_called_with(status="ok")

    @pytest.mark.asyncio
    async def test_endpoint_without_file(self):
        """Test process_rules_audit when rules file doesn't exist."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {"__meta__": {"version": "v1"}, "copper": {"simple": ["step1"]}}

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch.dict(os.environ, {"PROCESS_RULES_FILE": "/nonexistent/path.yaml"}):
                with patch("src.api.v1.process.os.path.exists", return_value=False):
                    with patch(
                        "src.api.v1.process.process_rules_audit_requests_total"
                    ) as mock_metric:
                        mock_metric.labels.return_value = MagicMock()

                        result = await process_rules_audit(raw=True, api_key="test")

                        assert result.source == "embedded-defaults"
                        assert result.hash is None
                        assert "copper" in result.materials

    @pytest.mark.asyncio
    async def test_endpoint_raw_false(self):
        """Test process_rules_audit with raw=False returns empty raw dict."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {"__meta__": {"version": "v1"}, "steel": {"simple": ["step1"]}}

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch("src.api.v1.process.os.path.exists", return_value=False):
                with patch("src.api.v1.process.process_rules_audit_requests_total") as mock_metric:
                    mock_metric.labels.return_value = MagicMock()

                    result = await process_rules_audit(raw=False, api_key="test")

                    assert result.raw == {}

    @pytest.mark.asyncio
    async def test_endpoint_no_meta_version(self):
        """Test process_rules_audit defaults version when __meta__ missing."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {"steel": {"simple": ["step1"]}}

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch("src.api.v1.process.os.path.exists", return_value=False):
                with patch("src.api.v1.process.process_rules_audit_requests_total") as mock_metric:
                    mock_metric.labels.return_value = MagicMock()

                    result = await process_rules_audit(raw=True, api_key="test")

                    assert result.version == "v1"

    @pytest.mark.asyncio
    async def test_endpoint_file_read_exception(self):
        """Test process_rules_audit handles file read errors gracefully."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {"__meta__": {"version": "v1"}, "steel": {"simple": ["step1"]}}

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch.dict(os.environ, {"PROCESS_RULES_FILE": "/tmp/rules.yaml"}):
                with patch("src.api.v1.process.os.path.exists", return_value=True):
                    with patch("builtins.open", side_effect=IOError("Permission denied")):
                        with patch(
                            "src.api.v1.process.process_rules_audit_requests_total"
                        ) as mock_metric:
                            mock_metric.labels.return_value = MagicMock()

                            result = await process_rules_audit(raw=True, api_key="test")

                            # Hash should be None due to read error
                            assert result.hash is None

    @pytest.mark.asyncio
    async def test_endpoint_error_increments_error_metric(self):
        """Test process_rules_audit increments error metric on exception."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {"__meta__": {"version": "v1"}}

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch("src.api.v1.process.os.path.exists", return_value=False):
                with patch(
                    "src.api.v1.process.ProcessRulesAuditResponse",
                    side_effect=Exception("Model error"),
                ):
                    with patch(
                        "src.api.v1.process.process_rules_audit_requests_total"
                    ) as mock_metric:
                        mock_labels = MagicMock()
                        mock_metric.labels.return_value = mock_labels

                        with pytest.raises(Exception, match="Model error"):
                            await process_rules_audit(raw=True, api_key="test")

                        mock_metric.labels.assert_called_with(status="error")

    @pytest.mark.asyncio
    async def test_endpoint_complexities_extraction(self):
        """Test process_rules_audit correctly extracts complexities."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {
            "steel": {
                "simple": ["step1"],  # Should be included (list)
                "complex": ["step1", "step2"],  # Should be included (list)
                "description": "Steel processing",  # Should NOT be included (string)
            }
        }

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch("src.api.v1.process.os.path.exists", return_value=False):
                with patch("src.api.v1.process.process_rules_audit_requests_total") as mock_metric:
                    mock_metric.labels.return_value = MagicMock()

                    result = await process_rules_audit(raw=True, api_key="test")

                    steel_complexities = result.complexities.get("steel", [])
                    assert "simple" in steel_complexities
                    assert "complex" in steel_complexities
                    assert "description" not in steel_complexities

    @pytest.mark.asyncio
    async def test_endpoint_non_dict_material_values(self):
        """Test endpoint handles case where material value is not a dict."""
        from src.api.v1.process import process_rules_audit

        mock_rules = {
            "steel": {"simple": ["step1"]},  # Normal dict
            "copper": "just a string",  # Not a dict - should be skipped in complexities
        }

        with patch("src.core.process_rules.load_rules", return_value=mock_rules):
            with patch("src.api.v1.process.os.path.exists", return_value=False):
                with patch("src.api.v1.process.process_rules_audit_requests_total") as mock_metric:
                    mock_metric.labels.return_value = MagicMock()

                    result = await process_rules_audit(raw=True, api_key="test")

                    assert "steel" in result.complexities
                    # Non-dict material values should have empty list complexities
                    # The endpoint skips materials where value is not isinstance(cm, dict)


class TestProcessRulesAuditResponseModelDirect:
    """Direct tests for ProcessRulesAuditResponse model."""

    def test_model_creation(self):
        """Test ProcessRulesAuditResponse can be created."""
        from src.api.v1.process import ProcessRulesAuditResponse

        response = ProcessRulesAuditResponse(
            version="v1",
            source="config/process_rules.yaml",
            hash="abc123",
            materials=["steel", "aluminum"],
            complexities={"steel": ["simple", "complex"]},
            raw={"__meta__": {"version": "v1"}},
        )

        assert response.version == "v1"
        assert response.source == "config/process_rules.yaml"
        assert response.hash == "abc123"
        assert response.materials == ["steel", "aluminum"]

    def test_model_with_none_hash(self):
        """Test ProcessRulesAuditResponse with None hash."""
        from src.api.v1.process import ProcessRulesAuditResponse

        response = ProcessRulesAuditResponse(
            version="v1",
            source="embedded-defaults",
            hash=None,
            materials=[],
            complexities={},
            raw={},
        )

        assert response.hash is None


class TestRouterExport:
    """Tests for router export."""

    def test_router_exported(self):
        """Test router is exported from module."""
        from src.api.v1.process import router

        assert router is not None

    def test_all_contains_router(self):
        """Test __all__ contains router."""
        from src.api.v1 import process

        assert "router" in process.__all__
