"""Unit tests for vision providers.

Tests the vision provider implementations and factory.
"""

import os
import pytest
from unittest.mock import patch

from src.core.vision import (
    VisionDescription,
    VisionProviderError,
    create_vision_provider,
    get_available_providers,
    create_stub_provider,
)
from src.core.vision.providers import (
    DeepSeekStubProvider,
    DeepSeekVisionProvider,
    OpenAIVisionProvider,
    AnthropicVisionProvider,
)


# Sample image data (minimal PNG header)
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)
SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"


class TestDeepSeekStubProvider:
    """Tests for the stub provider."""

    @pytest.mark.asyncio
    async def test_analyze_image_returns_description(self):
        """Test that stub returns fixed description."""
        provider = DeepSeekStubProvider(simulate_latency_ms=0)
        result = await provider.analyze_image(SAMPLE_PNG)

        assert isinstance(result, VisionDescription)
        assert result.summary != ""
        assert len(result.details) > 0
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_image_ocr_only_mode(self):
        """Test OCR-only mode returns minimal description."""
        provider = DeepSeekStubProvider(simulate_latency_ms=0)
        result = await provider.analyze_image(SAMPLE_PNG, include_description=False)

        assert result.summary == "Image processed (OCR-only mode)"
        assert result.details == []
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_analyze_image_empty_raises(self):
        """Test that empty image data raises error."""
        provider = DeepSeekStubProvider(simulate_latency_ms=0)
        with pytest.raises(ValueError, match="empty"):
            await provider.analyze_image(b"")

    def test_provider_name(self):
        """Test provider name property."""
        provider = DeepSeekStubProvider()
        assert provider.provider_name == "deepseek_stub"


class TestDeepSeekVisionProvider:
    """Tests for DeepSeek Vision provider."""

    def test_init_without_api_key_raises(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            with pytest.raises(VisionProviderError, match="API key required"):
                DeepSeekVisionProvider()

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            provider = DeepSeekVisionProvider()
            assert provider.api_key == "test-key"
            assert provider.provider_name == "deepseek"

    def test_detect_image_type_png(self):
        """Test PNG detection."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            provider = DeepSeekVisionProvider()
            assert provider._detect_image_type(SAMPLE_PNG) == "image/png"

    def test_detect_image_type_jpeg(self):
        """Test JPEG detection."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            provider = DeepSeekVisionProvider()
            assert provider._detect_image_type(SAMPLE_JPEG) == "image/jpeg"

    @pytest.mark.asyncio
    async def test_analyze_image_ocr_only_mode(self):
        """Test OCR-only mode returns minimal response."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            provider = DeepSeekVisionProvider()
            result = await provider.analyze_image(SAMPLE_PNG, include_description=False)
            assert result.summary == "Image processed (OCR-only mode)"


class TestOpenAIVisionProvider:
    """Tests for OpenAI Vision provider."""

    def test_init_without_api_key_raises(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(VisionProviderError, match="API key required"):
                OpenAIVisionProvider()

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIVisionProvider()
            assert provider.api_key == "test-key"
            assert provider.provider_name == "openai"
            assert provider.model == "gpt-4o"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIVisionProvider(model="gpt-4-turbo")
            assert provider.model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_analyze_image_ocr_only_mode(self):
        """Test OCR-only mode returns minimal response."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIVisionProvider()
            result = await provider.analyze_image(SAMPLE_PNG, include_description=False)
            assert result.summary == "Image processed (OCR-only mode)"


class TestAnthropicVisionProvider:
    """Tests for Anthropic Vision provider."""

    def test_init_without_api_key_raises(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(VisionProviderError, match="API key required"):
                AnthropicVisionProvider()

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicVisionProvider()
            assert provider.api_key == "test-key"
            assert provider.provider_name == "anthropic"

    def test_detect_media_type(self):
        """Test media type detection."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicVisionProvider()
            assert provider._detect_media_type(SAMPLE_PNG) == "image/png"
            assert provider._detect_media_type(SAMPLE_JPEG) == "image/jpeg"

    @pytest.mark.asyncio
    async def test_analyze_image_ocr_only_mode(self):
        """Test OCR-only mode returns minimal response."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicVisionProvider()
            result = await provider.analyze_image(SAMPLE_PNG, include_description=False)
            assert result.summary == "Image processed (OCR-only mode)"


class TestVisionProviderFactory:
    """Tests for vision provider factory."""

    def test_create_stub_provider(self):
        """Test creating stub provider."""
        provider = create_vision_provider("stub")
        assert isinstance(provider, DeepSeekStubProvider)

    def test_create_stub_provider_explicit(self):
        """Test creating stub provider with explicit type."""
        provider = create_stub_provider()
        assert isinstance(provider, DeepSeekStubProvider)

    def test_create_deepseek_with_key(self):
        """Test creating DeepSeek provider."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            provider = create_vision_provider("deepseek")
            assert isinstance(provider, DeepSeekVisionProvider)

    def test_create_openai_with_key(self):
        """Test creating OpenAI provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = create_vision_provider("openai")
            assert isinstance(provider, OpenAIVisionProvider)

    def test_create_anthropic_with_key(self):
        """Test creating Anthropic provider."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = create_vision_provider("anthropic")
            assert isinstance(provider, AnthropicVisionProvider)

    def test_auto_detect_deepseek(self):
        """Test auto-detection selects DeepSeek when key available."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            provider = create_vision_provider("auto")
            assert isinstance(provider, DeepSeekVisionProvider)

    def test_auto_detect_openai(self):
        """Test auto-detection selects OpenAI when key available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            provider = create_vision_provider("auto")
            assert isinstance(provider, OpenAIVisionProvider)

    def test_auto_detect_anthropic(self):
        """Test auto-detection selects Anthropic when key available."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            provider = create_vision_provider("auto")
            assert isinstance(provider, AnthropicVisionProvider)

    def test_auto_detect_fallback_to_stub(self):
        """Test auto-detection falls back to stub when no keys."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all API keys
            for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            provider = create_vision_provider("auto")
            assert isinstance(provider, DeepSeekStubProvider)

    def test_fallback_to_stub_on_error(self):
        """Test fallback to stub when provider creation fails."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            # Request OpenAI without key should fall back to stub
            provider = create_vision_provider("openai", fallback_to_stub=True)
            assert isinstance(provider, DeepSeekStubProvider)

    def test_no_fallback_raises_error(self):
        """Test that disabling fallback raises error."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            with pytest.raises(VisionProviderError):
                create_vision_provider("openai", fallback_to_stub=False)

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises error."""
        with pytest.raises(VisionProviderError, match="Unknown provider"):
            create_vision_provider("unknown_provider", fallback_to_stub=False)


class TestGetAvailableProviders:
    """Tests for get_available_providers."""

    def test_returns_all_providers(self):
        """Test that all providers are listed."""
        providers = get_available_providers()
        assert "stub" in providers
        assert "deepseek" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_stub_always_available(self):
        """Test that stub is always available."""
        providers = get_available_providers()
        assert providers["stub"]["available"] is True
        assert providers["stub"]["requires_key"] is False

    def test_key_set_detection(self):
        """Test that key_set reflects environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            providers = get_available_providers()
            assert providers["openai"]["key_set"] is True

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            providers = get_available_providers()
            assert providers["openai"]["key_set"] is False
