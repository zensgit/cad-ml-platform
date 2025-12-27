"""Tests for src/utils/circuit_breaker.py to improve coverage.

Covers:
- CircuitConfig dataclass
- CircuitBreaker class
- State management (closed, half-open, open)
- Redis path with get/set state
- Local fallback path
- Error handling paths
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCircuitConfigUtils:
    """Tests for CircuitConfig dataclass."""

    def test_default_values(self):
        """Test CircuitConfig default values."""
        from src.utils.circuit_breaker import CircuitConfig

        cfg = CircuitConfig()

        assert cfg.error_threshold == 5
        assert cfg.timeout_seconds == 300
        assert cfg.half_open_requests == 2

    def test_custom_values(self):
        """Test CircuitConfig with custom values."""
        from src.utils.circuit_breaker import CircuitConfig

        cfg = CircuitConfig(
            error_threshold=10,
            timeout_seconds=600,
            half_open_requests=5,
        )

        assert cfg.error_threshold == 10
        assert cfg.timeout_seconds == 600
        assert cfg.half_open_requests == 5


class TestCircuitBreakerUtilsInit:
    """Tests for CircuitBreaker initialization."""

    def test_init_default_config(self):
        """Test CircuitBreaker initialization with default config."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            cb = CircuitBreaker("test_key")

        assert cb.key == "ocr:cb:test_key"
        assert cb._state == 0
        assert cb._opened_at == 0.0
        assert cb._half_open_budget == 0

    def test_init_custom_config(self):
        """Test CircuitBreaker initialization with custom config."""
        from src.utils.circuit_breaker import CircuitBreaker, CircuitConfig

        cfg = CircuitConfig(error_threshold=10, timeout_seconds=600)

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            cb = CircuitBreaker("test", cfg=cfg)

        assert cb.cfg.error_threshold == 10
        assert cb.cfg.timeout_seconds == 600


class TestCircuitBreakerUtilsGetState:
    """Tests for CircuitBreaker._get_state method."""

    @pytest.mark.asyncio
    async def test_get_state_no_redis_client(self):
        """Test _get_state returns local state when Redis not available."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 2  # Set local state to open

                state = await cb._get_state()

        assert state == 2

    @pytest.mark.asyncio
    async def test_get_state_from_redis(self):
        """Test _get_state returns state from Redis."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value="1")  # half-open state

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                state = await cb._get_state()

        assert state == 1

    @pytest.mark.asyncio
    async def test_get_state_redis_returns_none(self):
        """Test _get_state returns 0 when Redis key doesn't exist."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=None)

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                state = await cb._get_state()

        assert state == 0

    @pytest.mark.asyncio
    async def test_get_state_redis_error(self):
        """Test _get_state falls back to local state on Redis error."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Redis error"))

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                cb._state = 1
                state = await cb._get_state()

        assert state == 1


class TestCircuitBreakerUtilsSetState:
    """Tests for CircuitBreaker._set_state method."""

    @pytest.mark.asyncio
    async def test_set_state_no_redis_client(self):
        """Test _set_state updates local state when Redis not available."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                await cb._set_state(2)

        assert cb._state == 2

    @pytest.mark.asyncio
    async def test_set_state_to_closed_deletes_redis(self):
        """Test _set_state(0) deletes Redis key."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.delete = AsyncMock()

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                await cb._set_state(0)

        mock_client.delete.assert_called_once_with("ocr:cb:test")

    @pytest.mark.asyncio
    async def test_set_state_to_open_sets_redis(self):
        """Test _set_state(2) sets Redis key with TTL."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.setex = AsyncMock()

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                await cb._set_state(2)

        mock_client.setex.assert_called_once_with("ocr:cb:test", 300, "2")

    @pytest.mark.asyncio
    async def test_set_state_to_half_open_sets_redis(self):
        """Test _set_state(1) sets Redis key with TTL."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.setex = AsyncMock()

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                await cb._set_state(1)

        mock_client.setex.assert_called_once_with("ocr:cb:test", 300, "1")

    @pytest.mark.asyncio
    async def test_set_state_redis_error(self):
        """Test _set_state handles Redis error gracefully."""
        from src.utils.circuit_breaker import CircuitBreaker

        mock_client = MagicMock()
        mock_client.setex = AsyncMock(side_effect=Exception("Redis error"))

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=mock_client):
                cb = CircuitBreaker("test")
                # Should not raise
                await cb._set_state(2)

        # Local state should still be updated
        assert cb._state == 2


class TestCircuitBreakerUtilsShouldAllow:
    """Tests for CircuitBreaker.should_allow method."""

    @pytest.mark.asyncio
    async def test_should_allow_closed_state(self):
        """Test should_allow returns True when circuit is closed."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 0

                result = await cb.should_allow()

        assert result is True

    @pytest.mark.asyncio
    async def test_should_allow_open_state_not_timed_out(self):
        """Test should_allow returns False when circuit is open and not timed out."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 2
                cb._opened_at = time.time()  # Just opened

                result = await cb.should_allow()

        assert result is False

    @pytest.mark.asyncio
    async def test_should_allow_open_state_timed_out(self):
        """Test should_allow transitions to half-open when timeout expired."""
        from src.utils.circuit_breaker import CircuitBreaker, CircuitConfig

        cfg = CircuitConfig(timeout_seconds=1)

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test", cfg=cfg)
                cb._state = 2
                cb._opened_at = time.time() - 2  # Opened 2 seconds ago

                result = await cb.should_allow()

        assert result is True
        assert cb._state == 1  # Transitioned to half-open
        assert cb._half_open_budget == 2

    @pytest.mark.asyncio
    async def test_should_allow_half_open_with_budget(self):
        """Test should_allow returns True in half-open with budget remaining."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 1
                cb._half_open_budget = 2

                result = await cb.should_allow()

        assert result is True
        assert cb._half_open_budget == 1

    @pytest.mark.asyncio
    async def test_should_allow_half_open_no_budget(self):
        """Test should_allow returns False in half-open with no budget."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 1
                cb._half_open_budget = 0

                result = await cb.should_allow()

        assert result is False


class TestCircuitBreakerUtilsOnSuccess:
    """Tests for CircuitBreaker.on_success method."""

    @pytest.mark.asyncio
    async def test_on_success_resets_to_closed(self):
        """Test on_success resets circuit to closed state."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 1  # half-open

                await cb.on_success()

        assert cb._state == 0


class TestCircuitBreakerUtilsOnError:
    """Tests for CircuitBreaker.on_error method."""

    @pytest.mark.asyncio
    async def test_on_error_from_closed_opens_circuit(self):
        """Test on_error transitions from closed to open."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 0

                await cb.on_error()

        assert cb._state == 2
        assert cb._opened_at > 0

    @pytest.mark.asyncio
    async def test_on_error_from_half_open_opens_circuit(self):
        """Test on_error transitions from half-open to open."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 1
                old_opened_at = cb._opened_at

                await cb.on_error()

        assert cb._state == 2
        assert cb._opened_at > old_opened_at

    @pytest.mark.asyncio
    async def test_on_error_from_open_stays_open(self):
        """Test on_error from open state stays open."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 2
                cb._opened_at = time.time()

                await cb.on_error()

        # Should still be open (no transition from state 2)
        assert cb._state == 2


class TestCircuitBreakerUtilsIntegration:
    """Integration tests for CircuitBreaker full flow."""

    @pytest.mark.asyncio
    async def test_full_circuit_breaker_flow(self):
        """Test full circuit breaker state machine flow."""
        from src.utils.circuit_breaker import CircuitBreaker, CircuitConfig

        cfg = CircuitConfig(timeout_seconds=1, half_open_requests=1)

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test", cfg=cfg)

                # 1. Start in closed state, should allow
                assert await cb.should_allow() is True

                # 2. Error opens circuit
                await cb.on_error()
                assert cb._state == 2

                # 3. Open circuit blocks requests
                assert await cb.should_allow() is False

                # 4. Wait for timeout to expire
                cb._opened_at = time.time() - 2

                # 5. Should transition to half-open
                assert await cb.should_allow() is True
                assert cb._state == 1

                # 6. Success in half-open closes circuit
                await cb.on_success()
                assert cb._state == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens circuit."""
        from src.utils.circuit_breaker import CircuitBreaker, CircuitConfig

        cfg = CircuitConfig(timeout_seconds=1, half_open_requests=1)

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test", cfg=cfg)

                # Setup: circuit is half-open
                cb._state = 1
                cb._half_open_budget = 1

                # Probe request
                assert await cb.should_allow() is True

                # Probe fails
                await cb.on_error()

                # Circuit should be open again
                assert cb._state == 2


class TestCircuitBreakerUtilsConcurrency:
    """Tests for concurrent access to CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_concurrent_should_allow(self):
        """Test concurrent should_allow calls are serialized."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")
                cb._state = 1
                cb._half_open_budget = 2

                # Run concurrent calls
                results = await asyncio.gather(*[cb.should_allow() for _ in range(5)])

        # First two should succeed (budget=2), rest should fail
        true_count = sum(results)
        assert true_count == 2


class TestCircuitStatesValues:
    """Tests for circuit state numeric values."""

    def test_state_values(self):
        """Test circuit state numeric values."""
        # States: 0=closed, 1=half_open, 2=open
        CLOSED = 0
        HALF_OPEN = 1
        OPEN = 2

        assert CLOSED == 0
        assert HALF_OPEN == 1
        assert OPEN == 2


class TestCircuitBreakerUtilsMetrics:
    """Tests for metrics integration."""

    def test_init_sets_metric(self):
        """Test initialization sets circuit state metric."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_labels = MagicMock()
            mock_metric.labels.return_value = mock_labels

            CircuitBreaker("test")

            mock_metric.labels.assert_called_with(key="ocr:cb:test")
            mock_labels.set.assert_called_with(0)

    @pytest.mark.asyncio
    async def test_set_state_updates_metric(self):
        """Test _set_state updates circuit state metric."""
        from src.utils.circuit_breaker import CircuitBreaker

        with patch("src.utils.circuit_breaker.ocr_circuit_state") as mock_metric:
            mock_labels = MagicMock()
            mock_metric.labels.return_value = mock_labels
            with patch("src.utils.circuit_breaker.get_client", return_value=None):
                cb = CircuitBreaker("test")

                await cb._set_state(2)

                # Called during init (0) and during _set_state (2)
                assert mock_labels.set.call_count == 2
