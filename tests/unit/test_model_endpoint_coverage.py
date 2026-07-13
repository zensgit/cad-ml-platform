"""Tests for src/api/v1/model.py to improve coverage.

Covers:
- model_reload endpoint (SEALED, L3 Phase A: always 403, loader never reached)
- get_model_version endpoint
- get_opcode_audit endpoint
- Error handling paths
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException


class TestModelReloadSuccess:
    """model_reload is sealed: a payload that would formerly succeed still gets 403."""

    @pytest.mark.asyncio
    async def test_model_reload_success(self):
        """Formerly: successful model reload. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/model.pkl", force=False)

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


class TestModelReloadNotFound:
    """model_reload is sealed: a not-found-shaped payload still gets 403."""

    @pytest.mark.asyncio
    async def test_model_reload_not_found(self):
        """Formerly: model reload when file not found. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/nonexistent/model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


class TestModelReloadVersionMismatch:
    """model_reload is sealed: a version-mismatch-shaped payload still gets 403."""

    @pytest.mark.asyncio
    async def test_model_reload_version_mismatch(self):
        """Formerly: model reload with version mismatch. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(
            path="/path/to/model.pkl",
            expected_version="2.0.0",
        )

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


class TestModelReloadSizeExceeded:
    """model_reload is sealed: a size-exceeded-shaped payload still gets 403."""

    @pytest.mark.asyncio
    async def test_model_reload_size_exceeded(self):
        """Formerly: model reload when size limit exceeded. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/large_model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


class TestModelReloadSecurityErrors:
    """model_reload is sealed: security-error-shaped payloads still get 403."""

    @pytest.mark.asyncio
    async def test_model_reload_magic_invalid(self):
        """Formerly: model reload with invalid magic bytes. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/bad_model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403

    @pytest.mark.asyncio
    async def test_model_reload_hash_mismatch(self):
        """Formerly: model reload with hash mismatch. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/tampered_model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403

    @pytest.mark.asyncio
    async def test_model_reload_opcode_blocked(self):
        """Formerly: model reload with blocked opcode. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/malicious_model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403

    @pytest.mark.asyncio
    async def test_model_reload_opcode_scan_error(self):
        """Formerly: model reload with opcode scan error. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


class TestModelReloadRollback:
    """model_reload is sealed: a rollback-shaped payload still gets 403 (lines 116-122 are gone)."""

    @pytest.mark.asyncio
    async def test_model_reload_rollback(self):
        """Formerly: model reload triggers rollback. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/broken_model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


class TestModelReloadUnknownStatus:
    """model_reload is sealed: there is no more status-mapping branch to hit — always 403."""

    @pytest.mark.asyncio
    async def test_model_reload_unknown_status(self):
        """Formerly: model reload with unknown status. Now: sealed fail-closed 403."""
        from src.api.v1.model import ModelReloadRequest, model_reload

        payload = ModelReloadRequest(path="/path/to/model.pkl")

        with pytest.raises(HTTPException) as e:
            await model_reload(payload, api_key="test", admin_token="test")
        assert e.value.status_code == 403


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
