"""Tests for src/api/v1/model.py to improve coverage.

Covers:
- model_reload endpoint with all status branches
- get_model_version endpoint
- get_opcode_audit endpoint
- Error handling paths
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestModelReloadSuccess:
    """Tests for model_reload success path."""

    @pytest.mark.asyncio
    async def test_model_reload_success(self):
        """Test successful model reload."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/model.pkl", force=False)

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "success",
                "model_version": "1.0.0",
                "hash": "abc123",
            }
            with patch("src.ml.classifier.get_opcode_audit_snapshot") as mock_audit:
                mock_audit.return_value = {}
                result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "success"
        assert result.model_version == "1.0.0"
        assert result.hash == "abc123"


class TestModelReloadNotFound:
    """Tests for model_reload not_found path."""

    @pytest.mark.asyncio
    async def test_model_reload_not_found(self):
        """Test model reload when file not found."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/nonexistent/model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "not_found",
                "error": {"message": "File not found"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "not_found"
        assert result.error is not None


class TestModelReloadVersionMismatch:
    """Tests for model_reload version_mismatch path."""

    @pytest.mark.asyncio
    async def test_model_reload_version_mismatch(self):
        """Test model reload with version mismatch."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(
            path="/path/to/model.pkl",
            expected_version="2.0.0",
        )

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "version_mismatch",
                "actual_version": "1.0.0",
                "error": {"message": "Version mismatch"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "version_mismatch"


class TestModelReloadSizeExceeded:
    """Tests for model_reload size_exceeded path."""

    @pytest.mark.asyncio
    async def test_model_reload_size_exceeded(self):
        """Test model reload when size limit exceeded."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/large_model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "size_exceeded",
                "error": {
                    "message": "Model too large",
                    "context": {"size_mb": 500, "max_mb": 100},
                },
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "size_exceeded"


class TestModelReloadSecurityErrors:
    """Tests for model_reload security error paths."""

    @pytest.mark.asyncio
    async def test_model_reload_magic_invalid(self):
        """Test model reload with invalid magic bytes."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/bad_model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "magic_invalid",
                "error": {"message": "Invalid magic bytes"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "magic_invalid"

    @pytest.mark.asyncio
    async def test_model_reload_hash_mismatch(self):
        """Test model reload with hash mismatch."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/tampered_model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "hash_mismatch",
                "error": {"message": "Hash verification failed"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "hash_mismatch"

    @pytest.mark.asyncio
    async def test_model_reload_opcode_blocked(self):
        """Test model reload with blocked opcode."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/malicious_model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "opcode_blocked",
                "error": {"message": "Dangerous opcode detected"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "opcode_blocked"

    @pytest.mark.asyncio
    async def test_model_reload_opcode_scan_error(self):
        """Test model reload with opcode scan error."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "opcode_scan_error",
                "error": {"message": "Failed to scan opcodes"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "opcode_scan_error"


class TestModelReloadRollback:
    """Tests for model_reload rollback path (lines 116-122)."""

    @pytest.mark.asyncio
    async def test_model_reload_rollback(self):
        """Test model reload triggers rollback."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/broken_model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "rollback",
                "rollback_version": "0.9.0",
                "rollback_hash": "rollback_hash_123",
                "error": {"message": "Model failed validation, rolled back"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "rollback"
        assert result.model_version == "0.9.0"
        assert result.hash == "rollback_hash_123"
        assert result.error is not None


class TestModelReloadUnknownStatus:
    """Tests for model_reload unknown status path."""

    @pytest.mark.asyncio
    async def test_model_reload_unknown_status(self):
        """Test model reload with unknown status."""
        from src.api.v1.model import model_reload, ModelReloadRequest

        payload = ModelReloadRequest(path="/path/to/model.pkl")

        with patch("src.ml.classifier.reload_model") as mock_reload:
            mock_reload.return_value = {
                "status": "unexpected_status",
                "error": {"message": "Something unexpected happened"},
            }
            result = await model_reload(payload, api_key="test", admin_token="admin")

        assert result.status == "error"


class TestGetModelVersion:
    """Tests for get_model_version endpoint (lines 137-145)."""

    @pytest.mark.asyncio
    async def test_get_model_version_full_info(self):
        """Test get_model_version returns full info."""
        from src.api.v1.model import get_model_version

        with patch("src.ml.classifier.get_model_info") as mock_info:
            mock_info.return_value = {
                "version": "1.2.3",
                "hash": "hash456",
                "loaded_at": "2024-01-01T00:00:00Z",
                "path": "/models/current.pkl",
            }
            result = await get_model_version(api_key="test")

        assert result["model_version"] == "1.2.3"
        assert result["model_hash"] == "hash456"
        assert result["loaded_at"] == "2024-01-01T00:00:00Z"
        assert result["path"] == "/models/current.pkl"

    @pytest.mark.asyncio
    async def test_get_model_version_partial_info(self):
        """Test get_model_version with partial info."""
        from src.api.v1.model import get_model_version

        with patch("src.ml.classifier.get_model_info") as mock_info:
            mock_info.return_value = {
                "version": "1.0.0",
            }
            result = await get_model_version(api_key="test")

        assert result["model_version"] == "1.0.0"
        assert result["model_hash"] is None
        assert result["loaded_at"] is None
        assert result["path"] is None


class TestGetOpcodeAudit:
    """Tests for get_opcode_audit endpoint."""

    @pytest.mark.asyncio
    async def test_get_opcode_audit(self):
        """Test get_opcode_audit returns audit info."""
        from src.api.v1.model import get_opcode_audit

        with patch("src.ml.classifier.get_opcode_audit_snapshot") as mock_audit:
            mock_audit.return_value = {
                "enabled": True,
                "scanned_models": 5,
                "blocked_opcodes": ["GLOBAL", "REDUCE"],
            }
            result = await get_opcode_audit(api_key="test", admin_token="admin")

        assert result["enabled"] is True
        assert result["scanned_models"] == 5

    @pytest.mark.asyncio
    async def test_get_opcode_audit_empty(self):
        """Test get_opcode_audit when no audit data."""
        from src.api.v1.model import get_opcode_audit

        with patch("src.ml.classifier.get_opcode_audit_snapshot") as mock_audit:
            mock_audit.return_value = {}
            result = await get_opcode_audit(api_key="test", admin_token="admin")

        assert result == {}


class TestModelReloadRequest:
    """Tests for ModelReloadRequest model."""

    def test_model_reload_request_defaults(self):
        """Test ModelReloadRequest default values."""
        from src.api.v1.model import ModelReloadRequest

        req = ModelReloadRequest()

        assert req.path is None
        assert req.expected_version is None
        assert req.force is False

    def test_model_reload_request_with_values(self):
        """Test ModelReloadRequest with values."""
        from src.api.v1.model import ModelReloadRequest

        req = ModelReloadRequest(
            path="/path/to/model.pkl",
            expected_version="2.0.0",
            force=True,
        )

        assert req.path == "/path/to/model.pkl"
        assert req.expected_version == "2.0.0"
        assert req.force is True


class TestModelReloadResponse:
    """Tests for ModelReloadResponse model."""

    def test_model_reload_response_minimal(self):
        """Test ModelReloadResponse with minimal fields."""
        from src.api.v1.model import ModelReloadResponse

        resp = ModelReloadResponse(status="success")

        assert resp.status == "success"
        assert resp.model_version is None
        assert resp.hash is None
        assert resp.error is None
        assert resp.opcode_audit is None

    def test_model_reload_response_full(self):
        """Test ModelReloadResponse with all fields."""
        from src.api.v1.model import ModelReloadResponse

        resp = ModelReloadResponse(
            status="success",
            model_version="1.0.0",
            hash="abc123",
            error=None,
            opcode_audit={"scanned": True},
        )

        assert resp.status == "success"
        assert resp.model_version == "1.0.0"
        assert resp.hash == "abc123"
        assert resp.opcode_audit == {"scanned": True}


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_exists(self):
        """Test router is exported."""
        from src.api.v1.model import router

        assert router is not None

    def test_router_has_reload_route(self):
        """Test router has reload route."""
        from src.api.v1.model import router

        routes = [r.path for r in router.routes]
        assert "/reload" in routes

    def test_router_has_version_route(self):
        """Test router has version route."""
        from src.api.v1.model import router

        routes = [r.path for r in router.routes]
        assert "/version" in routes

    def test_router_has_opcode_audit_route(self):
        """Test router has opcode-audit route."""
        from src.api.v1.model import router

        routes = [r.path for r in router.routes]
        assert "/opcode-audit" in routes
