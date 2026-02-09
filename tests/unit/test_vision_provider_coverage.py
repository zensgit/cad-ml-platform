"""Additional tests for vision provider to improve coverage.

Targets uncovered code paths in src/core/providers/vision.py:
- Line 49: empty image bytes validation
- Lines 62-65: _health_check_impl with wrapped provider health_check
- Lines 86-91: StubVisionCoreProvider default config
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.providers.registry import ProviderRegistry
from src.core.providers.vision import (
    VisionProviderAdapter,
    VisionProviderConfig,
    bootstrap_core_vision_providers,
)
from src.core.vision.base import VisionDescription


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


# --- VisionProviderAdapter Tests ---


class TestVisionProviderAdapter:
    """Tests for VisionProviderAdapter."""

    @pytest.mark.asyncio
    async def test_process_rejects_non_bytes(self):
        """process rejects non-bytes input."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock()
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        with pytest.raises(TypeError, match="expects raw image bytes"):
            await adapter.process({"not": "bytes"})

    @pytest.mark.asyncio
    async def test_process_rejects_empty_bytes(self):
        """Line 49: process rejects empty image bytes."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock()
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        with pytest.raises(ValueError, match="image bytes cannot be empty"):
            await adapter.process(b"")

    @pytest.mark.asyncio
    async def test_process_accepts_bytearray(self):
        """process accepts bytearray input."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock()
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(summary="test summary", confidence=0.9)
        )
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        result = await adapter.process(bytearray(b"image data"))
        assert result.summary == "test summary"

    @pytest.mark.asyncio
    async def test_health_check_with_sync_health_check_method(self):
        """Lines 62-65: health_check_impl calls wrapped provider's sync health_check."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock()
        mock_provider.health_check = MagicMock(return_value=True)
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is True
        mock_provider.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_with_async_health_check_method(self):
        """Lines 63-64: health_check_impl awaits async health_check."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is True
        mock_provider.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_returns_false_from_provider(self):
        """health_check_impl returns False when provider returns False."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock()
        mock_provider.health_check = MagicMock(return_value=False)
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_fallback_to_analyze_image(self):
        """Lines 66-70: health_check_impl falls back to analyze_image when no health_check method."""
        config = VisionProviderConfig(
            name="test", provider_type="vision", provider_name="stub"
        )
        mock_provider = MagicMock(spec=[])  # No health_check attribute
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(summary="probe", confidence=0.9)
        )
        adapter = VisionProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is True
        mock_provider.analyze_image.assert_called_once()


# --- Bootstrap Tests ---


class TestBootstrapCoreVisionProviders:
    """Tests for bootstrap_core_vision_providers."""

    def test_bootstrap_registers_stub(self):
        """Lines 81-94: bootstrap registers vision/stub provider."""
        bootstrap_core_vision_providers()
        assert ProviderRegistry.exists("vision", "stub")

    def test_bootstrap_registers_deepseek_stub(self):
        """Lines 96-109: bootstrap registers vision/deepseek_stub provider."""
        bootstrap_core_vision_providers()
        assert ProviderRegistry.exists("vision", "deepseek_stub")

    def test_bootstrap_is_idempotent(self):
        """bootstrap can be called multiple times without error."""
        bootstrap_core_vision_providers()
        first_stub = ProviderRegistry.get_provider_class("vision", "stub")

        bootstrap_core_vision_providers()  # Second call
        second_stub = ProviderRegistry.get_provider_class("vision", "stub")

        assert first_stub is second_stub

    def test_stub_provider_default_config(self):
        """Lines 86-91: StubVisionCoreProvider uses default config when None."""
        bootstrap_core_vision_providers()
        provider = ProviderRegistry.get("vision", "stub")

        assert provider.config.name == "stub"
        assert provider.config.provider_type == "vision"
        assert provider.config.provider_name == "stub"

    def test_deepseek_stub_provider_default_config(self):
        """Lines 101-105: DeepSeekStubVisionCoreProvider uses default config."""
        bootstrap_core_vision_providers()
        provider = ProviderRegistry.get("vision", "deepseek_stub")

        assert provider.config.name == "deepseek_stub"
        assert provider.config.provider_type == "vision"
        assert provider.config.provider_name == "deepseek_stub"


# --- VisionProviderConfig Tests ---


class TestVisionProviderConfig:
    """Tests for VisionProviderConfig dataclass."""

    def test_default_values(self):
        """VisionProviderConfig has correct defaults."""
        config = VisionProviderConfig(name="test", provider_type="vision")
        assert config.provider_name == "stub"
        assert config.include_description_default is True
        assert config.provider_kwargs == {}

    def test_custom_values(self):
        """VisionProviderConfig accepts custom values."""
        config = VisionProviderConfig(
            name="custom",
            provider_type="vision",
            provider_name="deepseek",
            include_description_default=False,
            provider_kwargs={"api_key": "test"},
        )
        assert config.provider_name == "deepseek"
        assert config.include_description_default is False
        assert config.provider_kwargs == {"api_key": "test"}
