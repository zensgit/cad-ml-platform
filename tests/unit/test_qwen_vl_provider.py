"""Tests for Qwen-VL (通义千问视觉) Vision Provider.

Tests the Alibaba Cloud DashScope integration for vision analysis.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision.base import VisionDescription, VisionProviderError
from src.core.vision.factory import (
    FACTORY_REGISTRY,
    PROVIDER_REGISTRY,
    _auto_detect_provider,
    create_vision_provider,
    get_available_providers,
)
from src.core.vision.providers.qwen_vl import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_EN,
    QwenVLProvider,
    create_qwen_vl_provider,
)


class TestQwenVLProviderInit:
    """Tests for QwenVLProvider initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        provider = QwenVLProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "qwen-vl-max"
        assert provider.provider_name == "qwen_vl"

    def test_init_with_dashscope_env(self) -> None:
        """Test initialization with DASHSCOPE_API_KEY env var."""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}):
            provider = QwenVLProvider()
            assert provider.api_key == "env-key"

    def test_init_with_qwen_env(self) -> None:
        """Test initialization with QWEN_API_KEY env var."""
        with patch.dict(os.environ, {"QWEN_API_KEY": "qwen-env-key"}, clear=True):
            # Clear DASHSCOPE_API_KEY to test fallback
            os.environ.pop("DASHSCOPE_API_KEY", None)
            provider = QwenVLProvider()
            assert provider.api_key == "qwen-env-key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DASHSCOPE_API_KEY", None)
            os.environ.pop("QWEN_API_KEY", None)
            with pytest.raises(VisionProviderError) as exc_info:
                QwenVLProvider()
            assert "API key required" in str(exc_info.value)

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        provider = QwenVLProvider(api_key="test-key", model="qwen-vl-plus")
        assert provider.model == "qwen-vl-plus"

    def test_init_with_chinese_prompt(self) -> None:
        """Test Chinese prompt is used by default."""
        provider = QwenVLProvider(api_key="test-key", use_chinese_prompt=True)
        assert "工程图纸分析专家" in provider.system_prompt

    def test_init_with_english_prompt(self) -> None:
        """Test English prompt option."""
        provider = QwenVLProvider(api_key="test-key", use_chinese_prompt=False)
        assert "engineering drawing analyzer" in provider.system_prompt

    def test_init_with_custom_prompt(self) -> None:
        """Test custom system prompt."""
        custom_prompt = "Custom analysis prompt"
        provider = QwenVLProvider(api_key="test-key", system_prompt=custom_prompt)
        assert provider.system_prompt == custom_prompt

    def test_init_domestic_endpoint(self) -> None:
        """Test domestic (China) endpoint is used by default."""
        provider = QwenVLProvider(api_key="test-key")
        assert "dashscope.aliyuncs.com" in provider.base_url

    def test_init_international_endpoint(self) -> None:
        """Test international endpoint option."""
        provider = QwenVLProvider(api_key="test-key", international=True)
        assert "dashscope-intl.aliyuncs.com" in provider.base_url

    def test_init_custom_base_url(self) -> None:
        """Test custom base URL."""
        provider = QwenVLProvider(api_key="test-key", base_url="https://custom.api.com/v1")
        assert provider.base_url == "https://custom.api.com/v1"

    def test_init_timeout_configuration(self) -> None:
        """Test timeout configuration."""
        provider = QwenVLProvider(api_key="test-key", timeout_seconds=180.0)
        assert provider.timeout_seconds == 180.0


class TestQwenVLProviderAnalyze:
    """Tests for QwenVLProvider.analyze_image method."""

    @pytest.mark.asyncio
    async def test_analyze_image_success(self) -> None:
        """Test successful image analysis."""
        provider = QwenVLProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "机械零件图纸",
                                "details": ["尺寸: 100x50mm", "公差: ±0.1"],
                                "confidence": 0.95,
                            }
                        )
                    }
                }
            ]
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # PNG image magic bytes
            image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

            result = await provider.analyze_image(image_data)

            assert isinstance(result, VisionDescription)
            assert result.summary == "机械零件图纸"
            assert len(result.details) == 2
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_analyze_image_ocr_only_mode(self) -> None:
        """Test OCR-only mode returns placeholder."""
        provider = QwenVLProvider(api_key="test-key")

        result = await provider.analyze_image(b"test", include_description=False)

        assert result.summary == "Image processed (OCR-only mode)"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_analyze_image_empty_raises(self) -> None:
        """Test empty image data raises error."""
        provider = QwenVLProvider(api_key="test-key")

        with pytest.raises(ValueError, match="image_data cannot be empty"):
            await provider.analyze_image(b"")

    @pytest.mark.asyncio
    async def test_analyze_image_api_error(self) -> None:
        """Test API error handling."""
        provider = QwenVLProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(VisionProviderError) as exc_info:
                await provider.analyze_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_image_json_wrapped_in_markdown(self) -> None:
        """Test handling of JSON wrapped in markdown code blocks."""
        provider = QwenVLProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
    "summary": "测试图纸",
    "details": ["细节1"],
    "confidence": 0.8
}
```"""
                    }
                }
            ]
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.analyze_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            assert result.summary == "测试图纸"
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_analyze_image_chinese_field_names(self) -> None:
        """Test handling of Chinese field names in response."""
        provider = QwenVLProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"概述": "零件图", "详细信息": ["尺寸信息"], "置信度": 0.9})}}
            ]
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.analyze_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            assert result.summary == "零件图"
            assert result.confidence == 0.9


class TestQwenVLProviderImageDetection:
    """Tests for image type detection."""

    def test_detect_png(self) -> None:
        """Test PNG detection."""
        provider = QwenVLProvider(api_key="test-key")
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/png"

    def test_detect_jpeg(self) -> None:
        """Test JPEG detection."""
        provider = QwenVLProvider(api_key="test-key")
        image_data = b"\xff\xd8" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/jpeg"

    def test_detect_gif(self) -> None:
        """Test GIF detection."""
        provider = QwenVLProvider(api_key="test-key")
        image_data = b"GIF89a" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/gif"

    def test_detect_webp(self) -> None:
        """Test WebP detection."""
        provider = QwenVLProvider(api_key="test-key")
        image_data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/webp"

    def test_detect_unknown_defaults_png(self) -> None:
        """Test unknown format defaults to PNG."""
        provider = QwenVLProvider(api_key="test-key")
        image_data = b"unknown format data"
        assert provider._detect_image_type(image_data) == "image/png"


class TestQwenVLProviderJsonExtraction:
    """Tests for JSON extraction from response content."""

    def test_extract_plain_json(self) -> None:
        """Test extracting plain JSON."""
        provider = QwenVLProvider(api_key="test-key")
        content = '{"key": "value"}'
        assert provider._extract_json(content) == '{"key": "value"}'

    def test_extract_json_from_markdown(self) -> None:
        """Test extracting JSON from markdown code block."""
        provider = QwenVLProvider(api_key="test-key")
        content = '```json\n{"key": "value"}\n```'
        assert provider._extract_json(content) == '{"key": "value"}'

    def test_extract_json_from_generic_code_block(self) -> None:
        """Test extracting JSON from generic code block."""
        provider = QwenVLProvider(api_key="test-key")
        content = '```\n{"key": "value"}\n```'
        assert provider._extract_json(content) == '{"key": "value"}'

    def test_extract_json_with_surrounding_text(self) -> None:
        """Test extracting JSON with surrounding text."""
        provider = QwenVLProvider(api_key="test-key")
        content = 'Here is the result: {"key": "value"} Done.'
        result = provider._extract_json(content)
        assert result == '{"key": "value"}'


class TestQwenVLFactoryFunction:
    """Tests for create_qwen_vl_provider factory function."""

    def test_create_with_defaults(self) -> None:
        """Test factory with default settings."""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            provider = create_qwen_vl_provider()
            assert provider.model == "qwen-vl-max"
            assert provider.use_chinese_prompt is True

    def test_create_with_custom_model(self) -> None:
        """Test factory with custom model."""
        provider = create_qwen_vl_provider(api_key="test-key", model="qwen-vl-plus")
        assert provider.model == "qwen-vl-plus"

    def test_create_with_english_prompt(self) -> None:
        """Test factory with English prompt."""
        provider = create_qwen_vl_provider(api_key="test-key", use_chinese_prompt=False)
        assert "engineering drawing" in provider.system_prompt

    def test_create_international(self) -> None:
        """Test factory with international endpoint."""
        provider = create_qwen_vl_provider(api_key="test-key", international=True)
        assert "dashscope-intl" in provider.base_url


class TestFactoryIntegration:
    """Tests for Qwen-VL integration in factory.py."""

    def test_qwen_in_provider_registry(self) -> None:
        """Test Qwen-VL is in provider registry."""
        assert "qwen" in PROVIDER_REGISTRY
        assert "qwen_vl" in PROVIDER_REGISTRY
        assert "qwen-vl" in PROVIDER_REGISTRY
        assert "tongyi" in PROVIDER_REGISTRY
        assert "dashscope" in PROVIDER_REGISTRY

    def test_qwen_in_factory_registry(self) -> None:
        """Test Qwen-VL is in factory registry."""
        assert "qwen" in FACTORY_REGISTRY
        assert "qwen_vl" in FACTORY_REGISTRY
        assert "qwen-vl" in FACTORY_REGISTRY
        assert "tongyi" in FACTORY_REGISTRY
        assert "dashscope" in FACTORY_REGISTRY

    def test_create_qwen_via_factory(self) -> None:
        """Test creating Qwen-VL via main factory."""
        provider = create_vision_provider("qwen", api_key="test-key")
        assert provider.provider_name == "qwen_vl"

    def test_create_qwen_vl_via_factory(self) -> None:
        """Test creating Qwen-VL via qwen_vl alias."""
        provider = create_vision_provider("qwen_vl", api_key="test-key")
        assert provider.provider_name == "qwen_vl"

    def test_create_tongyi_via_factory(self) -> None:
        """Test creating Qwen-VL via tongyi alias."""
        provider = create_vision_provider("tongyi", api_key="test-key")
        assert provider.provider_name == "qwen_vl"

    def test_auto_detect_with_dashscope_key(self) -> None:
        """Test auto-detection with DASHSCOPE_API_KEY."""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            # Clear other keys
            for key in ["QWEN_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            detected = _auto_detect_provider()
            assert detected == "qwen_vl"

    def test_auto_detect_with_qwen_key(self) -> None:
        """Test auto-detection with QWEN_API_KEY."""
        with patch.dict(os.environ, {"QWEN_API_KEY": "test-key"}, clear=True):
            # Clear other keys
            for key in [
                "DASHSCOPE_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ]:
                os.environ.pop(key, None)
            detected = _auto_detect_provider()
            assert detected == "qwen_vl"

    def test_get_available_providers_includes_qwen(self) -> None:
        """Test get_available_providers includes Qwen-VL."""
        providers = get_available_providers()
        assert "qwen_vl" in providers
        assert providers["qwen_vl"]["description"]
        assert "通义千问" in providers["qwen_vl"]["description"]
        assert providers["qwen_vl"]["default_model"] == "qwen-vl-max"


class TestQwenVLProviderClose:
    """Tests for provider cleanup."""

    @pytest.mark.asyncio
    async def test_close_without_client(self) -> None:
        """Test close when client not created."""
        provider = QwenVLProvider(api_key="test-key")
        await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        """Test close when client exists."""
        provider = QwenVLProvider(api_key="test-key")

        # Create client
        client = await provider._get_client()
        assert provider._client is not None

        # Close
        await provider.close()
        assert provider._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
