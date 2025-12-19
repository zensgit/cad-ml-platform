"""Unit tests for dedup2d Redis job store - Phase 4 rolling upgrade compatibility."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict
from unittest import mock

import pytest


class TestDedup2DPayloadConfig:
    """Tests for Dedup2DPayloadConfig."""

    def test_default_config(self) -> None:
        """Default config should have bytes_b64 disabled."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DPayloadConfig

        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.include_bytes_b64 is False
        assert cfg.bytes_b64_max_bytes == 10 * 1024 * 1024  # 10MB default

    def test_enable_bytes_b64(self) -> None:
        """Enable bytes_b64 via env var."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DPayloadConfig

        env = {"DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.include_bytes_b64 is True

    def test_enable_bytes_b64_true_string(self) -> None:
        """Enable bytes_b64 via 'true' string."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DPayloadConfig

        env = {"DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64": "true"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.include_bytes_b64 is True

    def test_custom_max_bytes(self) -> None:
        """Custom max bytes threshold."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DPayloadConfig

        env = {"DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES": "5242880"}  # 5MB
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.bytes_b64_max_bytes == 5 * 1024 * 1024

    def test_negative_max_bytes_clamped_to_zero(self) -> None:
        """Negative max bytes should be clamped to 0."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DPayloadConfig

        env = {"DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES": "-100"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.bytes_b64_max_bytes == 0


class TestBase64Encoding:
    """Tests for base64 encoding/decoding utilities."""

    def test_encode_bytes_b64(self) -> None:
        """Encode bytes to base64."""
        from src.core.dedupcad_2d_jobs_redis import _encode_bytes_b64

        data = b"hello world"
        encoded = _encode_bytes_b64(data)
        assert encoded == base64.b64encode(data).decode("ascii")

    def test_encode_empty_bytes(self) -> None:
        """Encode empty bytes."""
        from src.core.dedupcad_2d_jobs_redis import _encode_bytes_b64

        encoded = _encode_bytes_b64(b"")
        assert encoded == ""

    def test_encode_binary_data(self) -> None:
        """Encode binary data with non-ASCII bytes."""
        from src.core.dedupcad_2d_jobs_redis import _encode_bytes_b64

        data = bytes(range(256))
        encoded = _encode_bytes_b64(data)
        # Verify roundtrip
        assert base64.b64decode(encoded) == data


class TestWorkerBackwardCompatibility:
    """Tests for worker backward compatibility with file_bytes_b64."""

    def test_decode_bytes_b64(self) -> None:
        """Decode base64 string to bytes."""
        from src.core.dedupcad_2d_worker import _decode_bytes_b64

        original = b"test file content"
        encoded = base64.b64encode(original).decode("ascii")
        decoded = _decode_bytes_b64(encoded)
        assert decoded == original

    def test_decode_empty_b64(self) -> None:
        """Decode empty base64 string."""
        from src.core.dedupcad_2d_worker import _decode_bytes_b64

        decoded = _decode_bytes_b64("")
        assert decoded == b""

    @pytest.mark.asyncio
    async def test_load_file_bytes_from_payload_with_file_ref(self) -> None:
        """Load file bytes from payload using file_ref (new format)."""
        from src.core.dedupcad_2d_worker import _load_file_bytes_from_payload

        expected_bytes = b"test file content from storage"

        mock_storage = mock.AsyncMock()
        mock_storage.load_bytes = mock.AsyncMock(return_value=expected_bytes)

        payload = {
            "file_ref": {"backend": "local", "path": "job123/file.dxf"},
        }

        with mock.patch(
            "src.core.dedup2d_file_storage.create_dedup2d_file_storage",
            return_value=mock_storage,
        ):
            file_bytes, file_ref, storage = await _load_file_bytes_from_payload(payload)

        assert file_bytes == expected_bytes
        assert file_ref is not None
        assert file_ref.backend == "local"
        assert storage is mock_storage

    @pytest.mark.asyncio
    async def test_load_file_bytes_from_payload_with_b64_fallback(self) -> None:
        """Load file bytes from payload using file_bytes_b64 (legacy format)."""
        from src.core.dedupcad_2d_worker import _load_file_bytes_from_payload

        original_content = b"legacy file content"
        encoded_content = base64.b64encode(original_content).decode("ascii")

        payload = {
            "file_bytes_b64": encoded_content,
            # No file_ref
        }

        file_bytes, file_ref, storage = await _load_file_bytes_from_payload(payload)

        assert file_bytes == original_content
        assert file_ref is None
        assert storage is None

    @pytest.mark.asyncio
    async def test_load_file_bytes_from_payload_prefers_file_ref(self) -> None:
        """When both file_ref and file_bytes_b64 exist, prefer file_ref."""
        from src.core.dedupcad_2d_worker import _load_file_bytes_from_payload

        file_ref_content = b"content from file_ref"
        b64_content = b"content from b64"

        mock_storage = mock.AsyncMock()
        mock_storage.load_bytes = mock.AsyncMock(return_value=file_ref_content)

        payload = {
            "file_ref": {"backend": "local", "path": "job123/file.dxf"},
            "file_bytes_b64": base64.b64encode(b64_content).decode("ascii"),
        }

        with mock.patch(
            "src.core.dedup2d_file_storage.create_dedup2d_file_storage",
            return_value=mock_storage,
        ):
            file_bytes, file_ref, storage = await _load_file_bytes_from_payload(payload)

        assert file_bytes == file_ref_content  # file_ref takes precedence
        assert file_ref is not None

    @pytest.mark.asyncio
    async def test_load_file_bytes_from_payload_missing_both(self) -> None:
        """Raise ValueError when neither file_ref nor file_bytes_b64 exists."""
        from src.core.dedupcad_2d_worker import _load_file_bytes_from_payload

        payload: Dict[str, Any] = {}

        with pytest.raises(ValueError, match="Missing file_ref and file_bytes_b64"):
            await _load_file_bytes_from_payload(payload)

    @pytest.mark.asyncio
    async def test_load_file_bytes_from_payload_empty_b64(self) -> None:
        """Empty file_bytes_b64 string should fall through to error."""
        from src.core.dedupcad_2d_worker import _load_file_bytes_from_payload

        payload = {
            "file_bytes_b64": "",  # Empty string
        }

        with pytest.raises(ValueError, match="Missing file_ref and file_bytes_b64"):
            await _load_file_bytes_from_payload(payload)


class TestDualWritePayload:
    """Tests for dual-write payload functionality."""

    @pytest.mark.asyncio
    async def test_payload_without_bytes_b64_by_default(self) -> None:
        """By default, payload should not include file_bytes_b64."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DPayloadConfig

        # Default config
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.include_bytes_b64 is False

    @pytest.mark.asyncio
    async def test_payload_includes_bytes_b64_when_enabled_small_file(self) -> None:
        """When enabled and file is small, payload should include file_bytes_b64."""
        from src.core.dedupcad_2d_jobs_redis import (
            Dedup2DPayloadConfig,
            _encode_bytes_b64,
        )

        env = {
            "DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64": "1",
            "DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES": "1000",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.include_bytes_b64 is True
        assert cfg.bytes_b64_max_bytes == 1000

        # Simulate payload creation logic
        file_bytes = b"small file content"
        assert len(file_bytes) <= cfg.bytes_b64_max_bytes

        # This simulates what submit_dedup2d_job does
        payload: Dict[str, Any] = {}
        if cfg.include_bytes_b64 and len(file_bytes) <= cfg.bytes_b64_max_bytes:
            payload["file_bytes_b64"] = _encode_bytes_b64(file_bytes)

        assert "file_bytes_b64" in payload
        assert base64.b64decode(payload["file_bytes_b64"]) == file_bytes

    @pytest.mark.asyncio
    async def test_payload_excludes_bytes_b64_when_file_too_large(self) -> None:
        """When file exceeds max size, payload should not include file_bytes_b64."""
        from src.core.dedupcad_2d_jobs_redis import (
            Dedup2DPayloadConfig,
            _encode_bytes_b64,
        )

        env = {
            "DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64": "1",
            "DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES": "10",  # 10 bytes max
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DPayloadConfig.from_env()

        assert cfg.include_bytes_b64 is True
        assert cfg.bytes_b64_max_bytes == 10

        # File larger than threshold
        file_bytes = b"this is a larger file content that exceeds 10 bytes"
        assert len(file_bytes) > cfg.bytes_b64_max_bytes

        # This simulates what submit_dedup2d_job does
        payload: Dict[str, Any] = {}
        if cfg.include_bytes_b64 and len(file_bytes) <= cfg.bytes_b64_max_bytes:
            payload["file_bytes_b64"] = _encode_bytes_b64(file_bytes)

        assert "file_bytes_b64" not in payload


class TestDedup2DRedisJobConfig:
    """Tests for Dedup2DRedisJobConfig."""

    def test_default_config(self) -> None:
        """Default config from env."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DRedisJobConfig

        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = Dedup2DRedisJobConfig.from_env()

        assert cfg.redis_url == "redis://localhost:6379/0"
        assert cfg.key_prefix == "dedup2d"
        assert cfg.queue_name == "dedup2d:queue"
        assert cfg.ttl_seconds >= 60
        assert cfg.max_jobs >= 1
        assert cfg.job_timeout_seconds >= 1

    def test_custom_config(self) -> None:
        """Custom config from env."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DRedisJobConfig

        env = {
            "DEDUP2D_REDIS_URL": "redis://custom:6380/1",
            "DEDUP2D_REDIS_KEY_PREFIX": "myprefix",
            "DEDUP2D_ARQ_QUEUE_NAME": "custom:queue",
            "DEDUP2D_ASYNC_TTL_SECONDS": "7200",
            "DEDUP2D_ASYNC_MAX_JOBS": "500",
            "DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS": "600",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DRedisJobConfig.from_env()

        assert cfg.redis_url == "redis://custom:6380/1"
        assert cfg.key_prefix == "myprefix"
        assert cfg.queue_name == "custom:queue"
        assert cfg.ttl_seconds == 7200
        assert cfg.max_jobs == 500
        assert cfg.job_timeout_seconds == 600

    def test_fallback_to_redis_url(self) -> None:
        """Fallback to REDIS_URL when DEDUP2D_REDIS_URL not set."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DRedisJobConfig

        env = {"REDIS_URL": "redis://fallback:6379/2"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DRedisJobConfig.from_env()

        assert cfg.redis_url == "redis://fallback:6379/2"

    def test_ttl_minimum_clamped(self) -> None:
        """TTL should be clamped to minimum 60 seconds."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DRedisJobConfig

        env = {"DEDUP2D_ASYNC_TTL_SECONDS": "10"}  # Too low
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DRedisJobConfig.from_env()

        assert cfg.ttl_seconds == 60  # Clamped to minimum

    def test_max_jobs_minimum_clamped(self) -> None:
        """Max jobs should be clamped to minimum 1."""
        from src.core.dedupcad_2d_jobs_redis import Dedup2DRedisJobConfig

        env = {"DEDUP2D_ASYNC_MAX_JOBS": "0"}  # Too low
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DRedisJobConfig.from_env()

        assert cfg.max_jobs == 1  # Clamped to minimum
