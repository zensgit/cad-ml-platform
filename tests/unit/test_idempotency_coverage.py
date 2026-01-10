"""Tests for src/utils/idempotency.py to improve coverage.

Covers:
- build_idempotency_key function
- check_idempotency function
- store_idempotency function
- TTL constants
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestIdempotencyConstants:
    """Tests for idempotency constants."""

    def test_idempotency_ttl_seconds(self):
        """Test IDEMPOTENCY_TTL_SECONDS is 24 hours."""
        from src.utils.idempotency import IDEMPOTENCY_TTL_SECONDS

        assert IDEMPOTENCY_TTL_SECONDS == 86400  # 24 * 60 * 60


class TestBuildIdempotencyKey:
    """Tests for build_idempotency_key function."""

    def test_default_endpoint(self):
        """Test build_idempotency_key with default endpoint."""
        from src.utils.idempotency import build_idempotency_key

        key = build_idempotency_key("request-123")

        assert key == "idempotency:ocr:request-123"

    def test_custom_endpoint(self):
        """Test build_idempotency_key with custom endpoint."""
        from src.utils.idempotency import build_idempotency_key

        key = build_idempotency_key("request-456", endpoint="vision")

        assert key == "idempotency:vision:request-456"

    def test_key_format(self):
        """Test key format follows pattern."""
        from src.utils.idempotency import build_idempotency_key

        key = build_idempotency_key("abc123", endpoint="api")

        assert key.startswith("idempotency:")
        assert ":api:" in key
        assert key.endswith("abc123")


class TestCheckIdempotency:
    """Tests for check_idempotency function."""

    @pytest.mark.asyncio
    async def test_empty_key_returns_none(self):
        """Test check_idempotency returns None for empty key."""
        from src.utils.idempotency import check_idempotency

        result = await check_idempotency("")

        assert result is None

    @pytest.mark.asyncio
    async def test_none_key_returns_none(self):
        """Test check_idempotency returns None for None key."""
        from src.utils.idempotency import check_idempotency

        result = await check_idempotency(None)  # type: ignore

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test check_idempotency returns cached response."""
        from src.utils.idempotency import check_idempotency

        cached_response = {"result": "cached_data", "status": "success"}

        with patch(
            "src.utils.idempotency.get_cache", new_callable=AsyncMock, return_value=cached_response
        ):
            result = await check_idempotency("request-123")

        assert result == cached_response

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test check_idempotency returns None on cache miss."""
        from src.utils.idempotency import check_idempotency

        with patch("src.utils.idempotency.get_cache", new_callable=AsyncMock, return_value=None):
            result = await check_idempotency("request-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_logs_info(self):
        """Test check_idempotency logs on cache hit."""
        from src.utils.idempotency import check_idempotency

        cached_response = {"result": "cached"}

        with patch(
            "src.utils.idempotency.get_cache", new_callable=AsyncMock, return_value=cached_response
        ):
            with patch("src.utils.idempotency.logger") as mock_logger:
                await check_idempotency("request-123", endpoint="ocr")

                mock_logger.info.assert_called_once()
                call_args = mock_logger.info.call_args
                assert "idempotency.cache_hit" in call_args[0]

    @pytest.mark.asyncio
    async def test_custom_endpoint(self):
        """Test check_idempotency with custom endpoint."""
        from src.utils.idempotency import check_idempotency

        with patch(
            "src.utils.idempotency.get_cache", new_callable=AsyncMock, return_value=None
        ) as mock_get:
            await check_idempotency("request-123", endpoint="vision")

            mock_get.assert_called_once_with("idempotency:vision:request-123")


class TestStoreIdempotency:
    """Tests for store_idempotency function."""

    @pytest.mark.asyncio
    async def test_empty_key_no_store(self):
        """Test store_idempotency does nothing for empty key."""
        from src.utils.idempotency import store_idempotency

        with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock) as mock_set:
            await store_idempotency("", {"result": "data"})

            mock_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_key_no_store(self):
        """Test store_idempotency does nothing for None key."""
        from src.utils.idempotency import store_idempotency

        with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock) as mock_set:
            await store_idempotency(None, {"result": "data"})  # type: ignore

            mock_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_response(self):
        """Test store_idempotency stores response in cache."""
        from src.utils.idempotency import IDEMPOTENCY_TTL_SECONDS, store_idempotency

        response = {"result": "success", "data": [1, 2, 3]}

        with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock) as mock_set:
            await store_idempotency("request-123", response)

            mock_set.assert_called_once_with(
                "idempotency:ocr:request-123", response, IDEMPOTENCY_TTL_SECONDS
            )

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        """Test store_idempotency with custom TTL."""
        from src.utils.idempotency import store_idempotency

        response = {"result": "success"}
        custom_ttl = 3600  # 1 hour

        with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock) as mock_set:
            await store_idempotency("request-123", response, ttl_seconds=custom_ttl)

            mock_set.assert_called_once_with("idempotency:ocr:request-123", response, custom_ttl)

    @pytest.mark.asyncio
    async def test_custom_endpoint(self):
        """Test store_idempotency with custom endpoint."""
        from src.utils.idempotency import IDEMPOTENCY_TTL_SECONDS, store_idempotency

        response = {"result": "success"}

        with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock) as mock_set:
            await store_idempotency("request-123", response, endpoint="vision")

            mock_set.assert_called_once_with(
                "idempotency:vision:request-123", response, IDEMPOTENCY_TTL_SECONDS
            )

    @pytest.mark.asyncio
    async def test_logs_stored_info(self):
        """Test store_idempotency logs on successful store."""
        from src.utils.idempotency import store_idempotency

        response = {"result": "success"}

        with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock):
            with patch("src.utils.idempotency.logger") as mock_logger:
                await store_idempotency("request-123", response, endpoint="ocr")

                mock_logger.info.assert_called_once()
                call_args = mock_logger.info.call_args
                assert "idempotency.stored" in call_args[0]


class TestIdempotencyFlow:
    """Tests for idempotency flow integration."""

    @pytest.mark.asyncio
    async def test_check_then_store_flow(self):
        """Test check -> miss -> process -> store flow."""
        from src.utils.idempotency import check_idempotency, store_idempotency

        idempotency_key = "unique-request-id"
        response = {"result": "processed"}

        with patch("src.utils.idempotency.get_cache", new_callable=AsyncMock, return_value=None):
            with patch("src.utils.idempotency.set_cache", new_callable=AsyncMock) as mock_set:
                # Check first - cache miss
                cached = await check_idempotency(idempotency_key)
                assert cached is None

                # Store after processing
                await store_idempotency(idempotency_key, response)
                mock_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_hit_skips_processing(self):
        """Test check -> hit -> return cached (no processing)."""
        from src.utils.idempotency import check_idempotency

        idempotency_key = "duplicate-request"
        cached_response = {"result": "already_processed"}

        with patch(
            "src.utils.idempotency.get_cache", new_callable=AsyncMock, return_value=cached_response
        ):
            result = await check_idempotency(idempotency_key)

        assert result == cached_response


class TestIdempotencyKeyFormats:
    """Tests for various idempotency key formats."""

    def test_uuid_style_key(self):
        """Test UUID-style idempotency key."""
        from src.utils.idempotency import build_idempotency_key

        key = build_idempotency_key("550e8400-e29b-41d4-a716-446655440000")

        assert "550e8400-e29b-41d4-a716-446655440000" in key

    def test_timestamp_style_key(self):
        """Test timestamp-style idempotency key."""
        from src.utils.idempotency import build_idempotency_key

        key = build_idempotency_key("req_1702233600_abc123")

        assert "req_1702233600_abc123" in key

    def test_special_characters_in_key(self):
        """Test special characters in idempotency key."""
        from src.utils.idempotency import build_idempotency_key

        key = build_idempotency_key("user:123:action:submit")

        assert "user:123:action:submit" in key
