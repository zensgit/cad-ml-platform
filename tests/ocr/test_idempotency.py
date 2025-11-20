"""Tests for Idempotency-Key support in OCR endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.idempotency import (
    IDEMPOTENCY_TTL_SECONDS,
    build_idempotency_key,
    check_idempotency,
    store_idempotency,
)


class TestIdempotencyKeyBuilder:
    """Test idempotency key construction."""

    def test_builds_correct_format(self):
        """Test: Idempotency key follows expected format."""
        key = build_idempotency_key("abc123", endpoint="ocr")
        assert key == "idempotency:ocr:abc123"

    def test_different_endpoints(self):
        """Test: Different endpoints produce different keys."""
        key_ocr = build_idempotency_key("xyz", endpoint="ocr")
        key_vision = build_idempotency_key("xyz", endpoint="vision")
        assert key_ocr != key_vision
        assert "ocr" in key_ocr
        assert "vision" in key_vision

    def test_same_input_same_output(self):
        """Test: Same input produces same key."""
        key1 = build_idempotency_key("test-key-123", endpoint="ocr")
        key2 = build_idempotency_key("test-key-123", endpoint="ocr")
        assert key1 == key2


class TestIdempotencyCheck:
    """Test idempotency cache lookup."""

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.get_cache")
    async def test_returns_cached_response_on_hit(self, mock_get_cache):
        """Test: Returns cached response when idempotency key exists."""
        cached_data = {
            "provider": "paddle",
            "confidence": 0.85,
            "fallback_level": None,
            "processing_time_ms": 150,
            "dimensions": [],
            "symbols": [],
            "title_block": {},
        }
        mock_get_cache.return_value = cached_data

        result = await check_idempotency("test-key", endpoint="ocr")

        assert result == cached_data
        mock_get_cache.assert_called_once_with("idempotency:ocr:test-key")

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.get_cache")
    async def test_returns_none_on_miss(self, mock_get_cache):
        """Test: Returns None when idempotency key not found."""
        mock_get_cache.return_value = None

        result = await check_idempotency("new-key", endpoint="ocr")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_key(self):
        """Test: Returns None for empty idempotency key."""
        result = await check_idempotency("", endpoint="ocr")
        assert result is None

        result = await check_idempotency(None, endpoint="ocr")
        assert result is None


class TestIdempotencyStore:
    """Test idempotency cache storage."""

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.set_cache")
    async def test_stores_response_with_ttl(self, mock_set_cache):
        """Test: Stores response with correct TTL."""
        response_data = {
            "provider": "deepseek_hf",
            "confidence": 0.92,
            "fallback_level": None,
            "processing_time_ms": 200,
            "dimensions": [{"type": "diameter", "value": 20.0}],
            "symbols": [],
            "title_block": {},
        }

        await store_idempotency("store-key", response_data, endpoint="ocr")

        mock_set_cache.assert_called_once_with(
            "idempotency:ocr:store-key", response_data, IDEMPOTENCY_TTL_SECONDS
        )

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.set_cache")
    async def test_custom_ttl(self, mock_set_cache):
        """Test: Allows custom TTL override."""
        response_data = {"test": "data"}
        custom_ttl = 3600

        await store_idempotency("key", response_data, endpoint="ocr", ttl_seconds=custom_ttl)

        mock_set_cache.assert_called_once_with("idempotency:ocr:key", response_data, custom_ttl)

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.set_cache")
    async def test_skips_storage_for_empty_key(self, mock_set_cache):
        """Test: Does not store if idempotency key is empty."""
        await store_idempotency("", {"data": "value"}, endpoint="ocr")
        mock_set_cache.assert_not_called()

        await store_idempotency(None, {"data": "value"}, endpoint="ocr")
        mock_set_cache.assert_not_called()


class TestIdempotencyIntegration:
    """Integration tests for idempotency flow."""

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.set_cache")
    @patch("src.utils.idempotency.get_cache")
    async def test_full_flow_store_and_retrieve(self, mock_get_cache, mock_set_cache):
        """Test: Full flow - first request stores, second retrieves."""
        response_data = {
            "provider": "paddle",
            "confidence": 0.88,
            "fallback_level": None,
            "processing_time_ms": 120,
            "dimensions": [],
            "symbols": [],
            "title_block": {},
        }

        # First request - cache miss
        mock_get_cache.return_value = None
        first_check = await check_idempotency("flow-key", endpoint="ocr")
        assert first_check is None

        # Store response
        await store_idempotency("flow-key", response_data, endpoint="ocr")
        mock_set_cache.assert_called_once()

        # Second request - cache hit
        mock_get_cache.return_value = response_data
        second_check = await check_idempotency("flow-key", endpoint="ocr")
        assert second_check == response_data

    @pytest.mark.asyncio
    @patch("src.utils.idempotency.get_cache")
    async def test_different_keys_independent(self, mock_get_cache):
        """Test: Different idempotency keys are independent."""
        # Key A has cached response
        mock_get_cache.side_effect = [
            {"provider": "paddle", "confidence": 0.9},  # Key A
            None,  # Key B
        ]

        result_a = await check_idempotency("key-a", endpoint="ocr")
        result_b = await check_idempotency("key-b", endpoint="ocr")

        assert result_a is not None
        assert result_b is None
