"""Additional tests for BaseProvider to improve coverage.

Targets uncovered code paths in src/core/providers/base.py:
- Line 68: name property fallback to __class__.__name__
- Line 75: provider_type property returning None
- Lines 107, 135-136, 139-140: health check edge cases
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.core.providers.base import BaseProvider, ProviderConfig, ProviderStatus


# --- Test Fixtures ---


@dataclass
class MinimalConfig:
    """Config without name or provider_type attributes."""

    some_field: str = "value"


@dataclass
class ConfigWithNoneName:
    """Config with name=None."""

    name: str | None = None
    provider_type: str | None = None


@dataclass
class ConfigWithEmptyName:
    """Config with empty string name."""

    name: str = ""
    provider_type: str = ""


class ConcreteProvider(BaseProvider):
    """Concrete implementation for testing."""

    async def _process_impl(self, request: Any, **kwargs: Any) -> dict:
        return {"processed": True}


class FailingHealthCheckProvider(BaseProvider):
    """Provider that fails health check with exception."""

    async def _process_impl(self, request: Any, **kwargs: Any) -> dict:
        return {"processed": True}

    async def _health_check_impl(self) -> bool:
        raise RuntimeError("Health check failed")


class SlowHealthCheckProvider(BaseProvider):
    """Provider with slow health check for timeout testing."""

    async def _process_impl(self, request: Any, **kwargs: Any) -> dict:
        return {"processed": True}

    async def _health_check_impl(self) -> bool:
        await asyncio.sleep(10)  # Will timeout
        return True


class UnhealthyProvider(BaseProvider):
    """Provider that returns False from health check."""

    async def _process_impl(self, request: Any, **kwargs: Any) -> dict:
        return {"processed": True}

    async def _health_check_impl(self) -> bool:
        return False


# --- Name Property Tests (Line 68) ---


class TestNamePropertyFallback:
    """Tests for name property fallback to __class__.__name__."""

    def test_name_fallback_when_config_has_no_name_attr(self):
        """Line 68: When config has no 'name' attribute, use class name."""
        config = MinimalConfig()
        provider = ConcreteProvider(config)
        assert provider.name == "ConcreteProvider"

    def test_name_fallback_when_config_name_is_none(self):
        """Line 68: When config.name is None, use class name."""
        config = ConfigWithNoneName(name=None)
        provider = ConcreteProvider(config)
        assert provider.name == "ConcreteProvider"

    def test_name_fallback_when_config_name_is_empty(self):
        """Line 68: When config.name is empty string, use class name."""
        config = ConfigWithEmptyName(name="")
        provider = ConcreteProvider(config)
        assert provider.name == "ConcreteProvider"

    def test_name_uses_config_when_valid(self):
        """Verify name uses config.name when it's a valid string."""
        config = ProviderConfig(name="my-provider", provider_type="test")
        provider = ConcreteProvider(config)
        assert provider.name == "my-provider"


# --- Provider Type Property Tests (Line 75) ---


class TestProviderTypeProperty:
    """Tests for provider_type property returning None."""

    def test_provider_type_none_when_config_has_no_attr(self):
        """Line 75: When config has no 'provider_type' attribute, return None."""
        config = MinimalConfig()
        provider = ConcreteProvider(config)
        assert provider.provider_type is None

    def test_provider_type_none_when_config_value_is_none(self):
        """Line 75: When config.provider_type is None, return None."""
        config = ConfigWithNoneName(provider_type=None)
        provider = ConcreteProvider(config)
        assert provider.provider_type is None

    def test_provider_type_none_when_config_value_is_empty(self):
        """Line 75: When config.provider_type is empty string, return None."""
        config = ConfigWithEmptyName(provider_type="")
        provider = ConcreteProvider(config)
        assert provider.provider_type is None

    def test_provider_type_uses_config_when_valid(self):
        """Verify provider_type uses config value when it's a valid string."""
        config = ProviderConfig(name="test", provider_type="classifier")
        provider = ConcreteProvider(config)
        assert provider.provider_type == "classifier"


# --- Health Check Edge Cases (Lines 107, 135-140) ---


class TestHealthCheckEdgeCases:
    """Tests for health check edge cases."""

    @pytest.mark.asyncio
    async def test_health_check_with_zero_timeout_uses_default(self):
        """Line 107: When timeout_seconds <= 0, use 0.5 as minimum."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        # Zero timeout should be replaced with 0.5
        ok = await provider.health_check(timeout_seconds=0)
        assert ok is True
        assert provider.status == ProviderStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_with_negative_timeout_uses_default(self):
        """Line 107: Negative timeout should be replaced with 0.5."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        ok = await provider.health_check(timeout_seconds=-5)
        assert ok is True
        assert provider.status == ProviderStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_timeout_capped_at_10(self):
        """Line 108: Timeout should be capped at 10 seconds."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        # Even with large timeout, should work (capped internally)
        ok = await provider.health_check(timeout_seconds=100)
        assert ok is True

    @pytest.mark.asyncio
    async def test_health_check_timeout_triggers_down_status(self):
        """Lines 120-123: Timeout should set status to DOWN and error to 'timeout'."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = SlowHealthCheckProvider(config)

        ok = await provider.health_check(timeout_seconds=0.1)
        assert ok is False
        assert provider.status == ProviderStatus.DOWN
        assert provider.last_error == "timeout"

    @pytest.mark.asyncio
    async def test_health_check_exception_sets_last_error(self):
        """Lines 124-127: Exception should set status to DOWN and capture error message."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = FailingHealthCheckProvider(config)

        ok = await provider.health_check()
        assert ok is False
        assert provider.status == ProviderStatus.DOWN
        assert "Health check failed" in provider.last_error

    @pytest.mark.asyncio
    async def test_health_check_false_sets_unhealthy_error(self):
        """Lines 117-118: When health check returns False, set 'unhealthy' error."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = UnhealthyProvider(config)

        ok = await provider.health_check()
        assert ok is False
        assert provider.status == ProviderStatus.DOWN
        assert provider.last_error == "unhealthy"


# --- Mark Degraded/Healthy Tests (Lines 135-136, 139-140) ---


class TestMarkDegradedHealthy:
    """Tests for mark_degraded and mark_healthy methods."""

    def test_mark_degraded_sets_status_and_error(self):
        """Lines 135-136: mark_degraded should set DEGRADED status and error reason."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        provider.mark_degraded("high latency")
        assert provider.status == ProviderStatus.DEGRADED
        assert provider.last_error == "high latency"

    def test_mark_healthy_clears_error(self):
        """Lines 139-140: mark_healthy should set HEALTHY status and clear error."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        # First mark as degraded
        provider.mark_degraded("some error")
        assert provider.status == ProviderStatus.DEGRADED
        assert provider.last_error == "some error"

        # Then mark healthy
        provider.mark_healthy()
        assert provider.status == ProviderStatus.HEALTHY
        assert provider.last_error is None

    def test_mark_degraded_after_healthy(self):
        """Verify mark_degraded works after mark_healthy."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        provider.mark_healthy()
        provider.mark_degraded("new issue")
        assert provider.status == ProviderStatus.DEGRADED
        assert provider.last_error == "new issue"


# --- Status Snapshot Tests ---


class TestStatusSnapshot:
    """Tests for status_snapshot method."""

    def test_status_snapshot_includes_all_fields(self):
        """Verify status_snapshot returns all expected fields."""
        config = ProviderConfig(name="snapshot-test", provider_type="test-type")
        provider = ConcreteProvider(config)

        snapshot = provider.status_snapshot()
        assert snapshot["name"] == "snapshot-test"
        assert snapshot["provider_type"] == "test-type"
        assert snapshot["status"] == "unknown"
        assert snapshot["last_error"] is None
        assert snapshot["last_health_check_at"] is None
        assert snapshot["last_health_check_latency_ms"] is None

    @pytest.mark.asyncio
    async def test_status_snapshot_after_health_check(self):
        """Verify status_snapshot updates after health check."""
        config = ProviderConfig(name="snapshot-test", provider_type="test-type")
        provider = ConcreteProvider(config)

        await provider.health_check()
        snapshot = provider.status_snapshot()

        assert snapshot["status"] == "healthy"
        assert snapshot["last_health_check_at"] is not None
        assert snapshot["last_health_check_latency_ms"] is not None
        assert snapshot["last_health_check_latency_ms"] >= 0


# --- Lifecycle Methods Tests ---


class TestLifecycleMethods:
    """Tests for warmup and shutdown lifecycle methods."""

    @pytest.mark.asyncio
    async def test_warmup_default_implementation(self):
        """Verify default warmup does nothing but completes."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        # Should not raise
        await provider.warmup()

    @pytest.mark.asyncio
    async def test_shutdown_default_implementation(self):
        """Verify default shutdown does nothing but completes."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        # Should not raise
        await provider.shutdown()


# --- Process Method Tests ---


class TestProcessMethod:
    """Tests for process method delegation."""

    @pytest.mark.asyncio
    async def test_process_delegates_to_impl(self):
        """Verify process calls _process_impl."""
        config = ProviderConfig(name="test", provider_type="test")
        provider = ConcreteProvider(config)

        result = await provider.process({"input": "data"})
        assert result == {"processed": True}

    @pytest.mark.asyncio
    async def test_process_passes_kwargs(self):
        """Verify process passes kwargs to _process_impl."""

        class KwargsProvider(BaseProvider):
            async def _process_impl(self, request: Any, **kwargs: Any) -> dict:
                return {"request": request, "kwargs": kwargs}

        config = ProviderConfig(name="test", provider_type="test")
        provider = KwargsProvider(config)

        result = await provider.process("input", extra="value")
        assert result["request"] == "input"
        assert result["kwargs"] == {"extra": "value"}
