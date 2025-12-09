"""Tests for GLM-4V (智谱清言视觉) Vision Provider.

Tests the Zhipu AI BigModel integration for vision analysis.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision.providers.glm4v import (
    GLM4VProvider,
    create_glm4v_provider,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_EN,
)
from src.core.vision.base import VisionDescription, VisionProviderError
from src.core.vision.factory import (
    create_vision_provider,
    get_available_providers,
    _auto_detect_provider,
    PROVIDER_REGISTRY,
    FACTORY_REGISTRY,
)


class TestGLM4VProviderInit:
    """Tests for GLM4VProvider initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        provider = GLM4VProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "glm-4v-plus"
        assert provider.provider_name == "glm4v"

    def test_init_with_zhipuai_env(self) -> None:
        """Test initialization with ZHIPUAI_API_KEY env var."""
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "zhipu-env-key"}):
            provider = GLM4VProvider()
            assert provider.api_key == "zhipu-env-key"

    def test_init_with_glm_env(self) -> None:
        """Test initialization with GLM_API_KEY env var."""
        with patch.dict(os.environ, {"GLM_API_KEY": "glm-env-key"}, clear=True):
            # Clear other keys to test fallback
            os.environ.pop("ZHIPUAI_API_KEY", None)
            os.environ.pop("ZHIPU_API_KEY", None)
            provider = GLM4VProvider()
            assert provider.api_key == "glm-env-key"

    def test_init_with_zhipu_env(self) -> None:
        """Test initialization with ZHIPU_API_KEY env var."""
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "zhipu-alt-key"}, clear=True):
            # Clear other keys to test fallback
            os.environ.pop("ZHIPUAI_API_KEY", None)
            os.environ.pop("GLM_API_KEY", None)
            provider = GLM4VProvider()
            assert provider.api_key == "zhipu-alt-key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ZHIPUAI_API_KEY", None)
            os.environ.pop("GLM_API_KEY", None)
            os.environ.pop("ZHIPU_API_KEY", None)
            with pytest.raises(VisionProviderError) as exc_info:
                GLM4VProvider()
            assert "API key required" in str(exc_info.value)

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        provider = GLM4VProvider(api_key="test-key", model="glm-4v-flash")
        assert provider.model == "glm-4v-flash"

    def test_init_with_chinese_prompt(self) -> None:
        """Test Chinese prompt is used by default."""
        provider = GLM4VProvider(api_key="test-key", use_chinese_prompt=True)
        assert "工程图纸分析专家" in provider.system_prompt

    def test_init_with_english_prompt(self) -> None:
        """Test English prompt option."""
        provider = GLM4VProvider(api_key="test-key", use_chinese_prompt=False)
        assert "engineering drawing analyzer" in provider.system_prompt

    def test_init_with_custom_prompt(self) -> None:
        """Test custom system prompt."""
        custom_prompt = "Custom analysis prompt"
        provider = GLM4VProvider(api_key="test-key", system_prompt=custom_prompt)
        assert provider.system_prompt == custom_prompt

    def test_init_default_base_url(self) -> None:
        """Test default base URL is BigModel endpoint."""
        provider = GLM4VProvider(api_key="test-key")
        assert "open.bigmodel.cn" in provider.base_url

    def test_init_custom_base_url(self) -> None:
        """Test custom base URL."""
        provider = GLM4VProvider(api_key="test-key", base_url="https://custom.api.com/v1")
        assert provider.base_url == "https://custom.api.com/v1"

    def test_init_timeout_configuration(self) -> None:
        """Test timeout configuration."""
        provider = GLM4VProvider(api_key="test-key", timeout_seconds=180.0)
        assert provider.timeout_seconds == 180.0

    def test_init_temperature_configuration(self) -> None:
        """Test temperature configuration."""
        provider = GLM4VProvider(api_key="test-key", temperature=0.5)
        assert provider.temperature == 0.5


class TestGLM4VProviderAnalyze:
    """Tests for GLM4VProvider.analyze_image method."""

    @pytest.mark.asyncio
    async def test_analyze_image_success(self) -> None:
        """Test successful image analysis."""
        provider = GLM4VProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "summary": "机械零件图纸",
                            "details": ["尺寸: 100x50mm", "公差: ±0.1"],
                            "confidence": 0.95
                        })
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
        provider = GLM4VProvider(api_key="test-key")

        result = await provider.analyze_image(b"test", include_description=False)

        assert result.summary == "Image processed (OCR-only mode)"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_analyze_image_empty_raises(self) -> None:
        """Test empty image data raises error."""
        provider = GLM4VProvider(api_key="test-key")

        with pytest.raises(ValueError, match="image_data cannot be empty"):
            await provider.analyze_image(b"")

    @pytest.mark.asyncio
    async def test_analyze_image_api_error(self) -> None:
        """Test API error handling."""
        provider = GLM4VProvider(api_key="test-key")

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
        provider = GLM4VProvider(api_key="test-key")

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
        provider = GLM4VProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "概述": "零件图",
                            "详细信息": ["尺寸信息"],
                            "置信度": 0.9
                        })
                    }
                }
            ]
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.analyze_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            assert result.summary == "零件图"
            assert result.confidence == 0.9


class TestGLM4VProviderImageDetection:
    """Tests for image type detection."""

    def test_detect_png(self) -> None:
        """Test PNG detection."""
        provider = GLM4VProvider(api_key="test-key")
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/png"

    def test_detect_jpeg(self) -> None:
        """Test JPEG detection."""
        provider = GLM4VProvider(api_key="test-key")
        image_data = b"\xff\xd8" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/jpeg"

    def test_detect_gif(self) -> None:
        """Test GIF detection."""
        provider = GLM4VProvider(api_key="test-key")
        image_data = b"GIF89a" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/gif"

    def test_detect_webp(self) -> None:
        """Test WebP detection."""
        provider = GLM4VProvider(api_key="test-key")
        image_data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
        assert provider._detect_image_type(image_data) == "image/webp"

    def test_detect_unknown_defaults_png(self) -> None:
        """Test unknown format defaults to PNG."""
        provider = GLM4VProvider(api_key="test-key")
        image_data = b"unknown format data"
        assert provider._detect_image_type(image_data) == "image/png"


class TestGLM4VProviderJsonExtraction:
    """Tests for JSON extraction from response content."""

    def test_extract_plain_json(self) -> None:
        """Test extracting plain JSON."""
        provider = GLM4VProvider(api_key="test-key")
        content = '{"key": "value"}'
        assert provider._extract_json(content) == '{"key": "value"}'

    def test_extract_json_from_markdown(self) -> None:
        """Test extracting JSON from markdown code block."""
        provider = GLM4VProvider(api_key="test-key")
        content = '```json\n{"key": "value"}\n```'
        assert provider._extract_json(content) == '{"key": "value"}'

    def test_extract_json_from_generic_code_block(self) -> None:
        """Test extracting JSON from generic code block."""
        provider = GLM4VProvider(api_key="test-key")
        content = '```\n{"key": "value"}\n```'
        assert provider._extract_json(content) == '{"key": "value"}'

    def test_extract_json_with_surrounding_text(self) -> None:
        """Test extracting JSON with surrounding text."""
        provider = GLM4VProvider(api_key="test-key")
        content = 'Here is the result: {"key": "value"} Done.'
        result = provider._extract_json(content)
        assert result == '{"key": "value"}'


class TestGLM4VFactoryFunction:
    """Tests for create_glm4v_provider factory function."""

    def test_create_with_defaults(self) -> None:
        """Test factory with default settings."""
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-key"}):
            provider = create_glm4v_provider()
            assert provider.model == "glm-4v-plus"
            assert provider.use_chinese_prompt is True

    def test_create_with_custom_model(self) -> None:
        """Test factory with custom model."""
        provider = create_glm4v_provider(api_key="test-key", model="glm-4v-flash")
        assert provider.model == "glm-4v-flash"

    def test_create_with_english_prompt(self) -> None:
        """Test factory with English prompt."""
        provider = create_glm4v_provider(api_key="test-key", use_chinese_prompt=False)
        assert "engineering drawing" in provider.system_prompt


class TestFactoryIntegration:
    """Tests for GLM-4V integration in factory.py."""

    def test_glm_in_provider_registry(self) -> None:
        """Test GLM-4V is in provider registry."""
        assert "glm" in PROVIDER_REGISTRY
        assert "glm4v" in PROVIDER_REGISTRY
        assert "glm-4v" in PROVIDER_REGISTRY
        assert "zhipu" in PROVIDER_REGISTRY
        assert "chatglm" in PROVIDER_REGISTRY

    def test_glm_in_factory_registry(self) -> None:
        """Test GLM-4V is in factory registry."""
        assert "glm" in FACTORY_REGISTRY
        assert "glm4v" in FACTORY_REGISTRY
        assert "glm-4v" in FACTORY_REGISTRY
        assert "zhipu" in FACTORY_REGISTRY
        assert "chatglm" in FACTORY_REGISTRY

    def test_create_glm_via_factory(self) -> None:
        """Test creating GLM-4V via main factory."""
        provider = create_vision_provider("glm", api_key="test-key")
        assert provider.provider_name == "glm4v"

    def test_create_glm4v_via_factory(self) -> None:
        """Test creating GLM-4V via glm4v alias."""
        provider = create_vision_provider("glm4v", api_key="test-key")
        assert provider.provider_name == "glm4v"

    def test_create_zhipu_via_factory(self) -> None:
        """Test creating GLM-4V via zhipu alias."""
        provider = create_vision_provider("zhipu", api_key="test-key")
        assert provider.provider_name == "glm4v"

    def test_auto_detect_with_zhipuai_key(self) -> None:
        """Test auto-detection with ZHIPUAI_API_KEY."""
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-key"}, clear=True):
            # Clear other keys
            for key in ["GLM_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY",
                       "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            detected = _auto_detect_provider()
            assert detected == "glm4v"

    def test_auto_detect_with_glm_key(self) -> None:
        """Test auto-detection with GLM_API_KEY."""
        with patch.dict(os.environ, {"GLM_API_KEY": "test-key"}, clear=True):
            # Clear other keys
            for key in ["ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY",
                       "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            detected = _auto_detect_provider()
            assert detected == "glm4v"

    def test_qwen_takes_priority_over_glm(self) -> None:
        """Test that Qwen-VL takes priority over GLM-4V in auto-detection."""
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "qwen-key",
            "ZHIPUAI_API_KEY": "glm-key"
        }, clear=True):
            # Clear other keys
            for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            detected = _auto_detect_provider()
            assert detected == "qwen_vl"  # Qwen should take priority

    def test_get_available_providers_includes_glm(self) -> None:
        """Test get_available_providers includes GLM-4V."""
        providers = get_available_providers()
        assert "glm4v" in providers
        assert providers["glm4v"]["description"]
        assert "智谱清言" in providers["glm4v"]["description"]
        assert providers["glm4v"]["default_model"] == "glm-4v-plus"


class TestGLM4VProviderClose:
    """Tests for provider cleanup."""

    @pytest.mark.asyncio
    async def test_close_without_client(self) -> None:
        """Test close when client not created."""
        provider = GLM4VProvider(api_key="test-key")
        await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        """Test close when client exists."""
        provider = GLM4VProvider(api_key="test-key")

        # Create client
        client = await provider._get_client()
        assert provider._client is not None

        # Close
        await provider.close()
        assert provider._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
