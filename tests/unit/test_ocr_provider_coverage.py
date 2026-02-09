"""Additional tests for OCR provider to improve coverage.

Targets uncovered code paths in src/core/providers/ocr.py:
- Line 47: unsupported provider error
- Line 56: empty image bytes validation
- Line 69: health_check returns True when no health_check method
- Lines 73-77: warmup method
- Line 83: extract compatibility method
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.ocr.base import OcrResult
from src.core.providers.ocr import (
    OcrProviderAdapter,
    OcrProviderConfig,
    bootstrap_core_ocr_providers,
)
from src.core.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


# --- OcrProviderAdapter Tests ---


class TestOcrProviderAdapter:
    """Tests for OcrProviderAdapter."""

    @pytest.mark.asyncio
    async def test_process_rejects_non_bytes(self):
        """process rejects non-bytes input."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        with pytest.raises(TypeError, match="expects raw image bytes"):
            await adapter.process({"not": "bytes"})

    @pytest.mark.asyncio
    async def test_process_rejects_empty_bytes(self):
        """Line 56: process rejects empty image bytes."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        with pytest.raises(ValueError, match="image bytes cannot be empty"):
            await adapter.process(b"")

    @pytest.mark.asyncio
    async def test_process_accepts_bytearray(self):
        """process accepts bytearray input."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        mock_provider.extract = AsyncMock(
            return_value=OcrResult(text="extracted", confidence=0.9)
        )
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        result = await adapter.process(bytearray(b"image data"))
        assert result.text == "extracted"

    @pytest.mark.asyncio
    async def test_health_check_with_sync_health_check_method(self):
        """health_check_impl calls wrapped provider's sync health_check."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        mock_provider.health_check = MagicMock(return_value=True)
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is True
        mock_provider.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_with_async_health_check_method(self):
        """health_check_impl awaits async health_check."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is True
        mock_provider.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_returns_true_without_health_check_method(self):
        """Line 69: health_check_impl returns True when no health_check method."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock(spec=[])  # No health_check attribute
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        ok = await adapter.health_check()
        assert ok is True

    @pytest.mark.asyncio
    async def test_warmup_calls_wrapped_sync_warmup(self):
        """Lines 73-77: warmup calls wrapped provider's sync warmup."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        mock_provider.warmup = MagicMock(return_value=None)
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        await adapter.warmup()
        mock_provider.warmup.assert_called_once()

    @pytest.mark.asyncio
    async def test_warmup_awaits_wrapped_async_warmup(self):
        """Lines 75-77: warmup awaits async warmup."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        mock_provider.warmup = AsyncMock(return_value=None)
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        await adapter.warmup()
        mock_provider.warmup.assert_called_once()

    @pytest.mark.asyncio
    async def test_warmup_does_nothing_without_warmup_method(self):
        """warmup does nothing when wrapped provider has no warmup."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock(spec=[])  # No warmup attribute
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        # Should not raise
        await adapter.warmup()

    @pytest.mark.asyncio
    async def test_extract_compatibility_method(self):
        """Line 83: extract method delegates to process."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="paddle"
        )
        mock_provider = MagicMock()
        mock_provider.extract = AsyncMock(
            return_value=OcrResult(text="extracted", confidence=0.9)
        )
        adapter = OcrProviderAdapter(config, wrapped_provider=mock_provider)

        result = await adapter.extract(b"image data", trace_id="trace-123")
        assert result.text == "extracted"


# --- _build_default_provider Tests ---


class TestBuildDefaultProvider:
    """Tests for OcrProviderAdapter._build_default_provider."""

    def test_unsupported_provider_raises_error(self):
        """Line 47: unsupported provider raises ValueError."""
        config = OcrProviderConfig(
            name="test", provider_type="ocr", provider_name="unsupported_provider"
        )

        with pytest.raises(ValueError, match="Unsupported OCR provider"):
            OcrProviderAdapter(config)


# --- Bootstrap Tests ---


class TestBootstrapCoreOcrProviders:
    """Tests for bootstrap_core_ocr_providers."""

    def test_bootstrap_registers_paddle(self):
        """bootstrap registers ocr/paddle provider."""
        bootstrap_core_ocr_providers()
        assert ProviderRegistry.exists("ocr", "paddle")

    def test_bootstrap_registers_deepseek_hf(self):
        """bootstrap registers ocr/deepseek_hf provider."""
        bootstrap_core_ocr_providers()
        assert ProviderRegistry.exists("ocr", "deepseek_hf")

    def test_bootstrap_is_idempotent(self):
        """bootstrap can be called multiple times without error."""
        bootstrap_core_ocr_providers()
        first_paddle = ProviderRegistry.get_provider_class("ocr", "paddle")

        bootstrap_core_ocr_providers()  # Second call
        second_paddle = ProviderRegistry.get_provider_class("ocr", "paddle")

        assert first_paddle is second_paddle


# --- OcrProviderConfig Tests ---


class TestOcrProviderConfig:
    """Tests for OcrProviderConfig dataclass."""

    def test_default_values(self):
        """OcrProviderConfig has correct defaults."""
        config = OcrProviderConfig(name="test", provider_type="ocr")
        assert config.provider_name == "paddle"
        assert config.trace_id is None
        assert config.provider_kwargs == {}

    def test_custom_values(self):
        """OcrProviderConfig accepts custom values."""
        config = OcrProviderConfig(
            name="custom",
            provider_type="ocr",
            provider_name="deepseek_hf",
            trace_id="trace-123",
            provider_kwargs={"model": "custom"},
        )
        assert config.provider_name == "deepseek_hf"
        assert config.trace_id == "trace-123"
        assert config.provider_kwargs == {"model": "custom"}
