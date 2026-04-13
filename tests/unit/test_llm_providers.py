"""Tests for LLM provider integrations."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.assistant.llm_providers import (
    BaseLLMProvider,
    ClaudeProvider,
    LLMConfig,
    OfflineProvider,
    OllamaProvider,
    OpenAIProvider,
    QwenProvider,
    get_best_available_provider,
    get_provider,
)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:

    def test_defaults(self):
        config = LLMConfig()
        assert config.api_key is None
        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.timeout == 30

    def test_custom_values(self):
        config = LLMConfig(api_key="test-key", model_name="test-model", temperature=0.5)
        assert config.api_key == "test-key"
        assert config.model_name == "test-model"
        assert config.temperature == 0.5


# ---------------------------------------------------------------------------
# OfflineProvider
# ---------------------------------------------------------------------------

class TestOfflineProvider:

    def test_always_available(self):
        provider = OfflineProvider()
        assert provider.is_available() is True

    def test_generate_with_knowledge(self):
        provider = OfflineProvider()
        response = provider.generate(
            "system prompt",
            "用户问题\n参考知识:材料Q235B是碳素钢\n请基于以上知识回答"
        )
        assert "Q235B" in response or "碳素钢" in response

    def test_generate_without_knowledge(self):
        provider = OfflineProvider()
        response = provider.generate("system", "普通问题")
        assert "离线模式" in response

    def test_generate_no_relevant_knowledge(self):
        provider = OfflineProvider()
        response = provider.generate("system", "参考知识:未找到相关知识。\n请基于以上知识回答")
        assert "离线模式" in response

    def test_inherits_base(self):
        provider = OfflineProvider()
        assert isinstance(provider, BaseLLMProvider)


# ---------------------------------------------------------------------------
# ClaudeProvider (mocked)
# ---------------------------------------------------------------------------

class TestClaudeProvider:

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.core.assistant.llm_providers.ClaudeProvider._init_client")
    def test_not_available_without_key(self, mock_init):
        mock_init.return_value = None
        provider = ClaudeProvider()
        provider._client = None
        assert provider.is_available() is False

    @patch("src.core.assistant.llm_providers.ClaudeProvider._init_client")
    def test_available_with_client(self, mock_init):
        mock_init.return_value = None
        provider = ClaudeProvider()
        provider._client = MagicMock()
        assert provider.is_available() is True

    @patch("src.core.assistant.llm_providers.ClaudeProvider._init_client")
    def test_generate_raises_without_client(self, mock_init):
        mock_init.return_value = None
        provider = ClaudeProvider()
        provider._client = None
        with pytest.raises(RuntimeError, match="Anthropic client not initialized"):
            provider.generate("system", "user")

    @patch("src.core.assistant.llm_providers.ClaudeProvider._init_client")
    def test_generate_calls_api(self, mock_init):
        mock_init.return_value = None
        provider = ClaudeProvider()
        mock_client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "test response"
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        provider._client = mock_client

        result = provider.generate("system prompt", "user prompt")
        assert result == "test response"
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAIProvider (mocked)
# ---------------------------------------------------------------------------

class TestOpenAIProvider:

    @patch("src.core.assistant.llm_providers.OpenAIProvider._init_client")
    def test_default_model_override(self, mock_init):
        mock_init.return_value = None
        config = LLMConfig(model_name="claude-3-sonnet-20240229")
        provider = OpenAIProvider(config)
        assert provider.config.model_name.startswith("gpt")

    @patch("src.core.assistant.llm_providers.OpenAIProvider._init_client")
    def test_gpt_model_preserved(self, mock_init):
        mock_init.return_value = None
        config = LLMConfig(model_name="gpt-4o")
        provider = OpenAIProvider(config)
        assert provider.config.model_name == "gpt-4o"

    @patch("src.core.assistant.llm_providers.OpenAIProvider._init_client")
    def test_not_available_without_client(self, mock_init):
        mock_init.return_value = None
        provider = OpenAIProvider()
        provider._client = None
        assert provider.is_available() is False

    @patch("src.core.assistant.llm_providers.OpenAIProvider._init_client")
    def test_generate_raises_without_client(self, mock_init):
        mock_init.return_value = None
        provider = OpenAIProvider()
        provider._client = None
        with pytest.raises(RuntimeError, match="OpenAI client not initialized"):
            provider.generate("system", "user")

    @patch("src.core.assistant.llm_providers.OpenAIProvider._init_client")
    def test_generate_calls_api(self, mock_init):
        mock_init.return_value = None
        provider = OpenAIProvider()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "gpt response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.generate("system", "user")
        assert result == "gpt response"


# ---------------------------------------------------------------------------
# QwenProvider (mocked)
# ---------------------------------------------------------------------------

class TestQwenProvider:

    @patch.dict(os.environ, {}, clear=True)
    def test_default_model(self):
        config = LLMConfig(model_name="claude-3-sonnet-20240229")
        provider = QwenProvider(config)
        assert "qwen" in provider.config.model_name.lower()

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        provider = QwenProvider()
        # Without dashscope import or key, should not be available
        assert provider.is_available() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_generate_raises_without_key(self):
        provider = QwenProvider()
        provider._api_key = None
        with pytest.raises(RuntimeError, match="DashScope API key not set"):
            provider.generate("system", "user")


# ---------------------------------------------------------------------------
# OllamaProvider (mocked)
# ---------------------------------------------------------------------------

class TestOllamaProvider:

    def test_default_model(self):
        provider = OllamaProvider()
        assert provider.config.model_name == "llama3"

    def test_custom_model_preserved(self):
        config = LLMConfig(model_name="mistral")
        provider = OllamaProvider(config)
        assert provider.config.model_name == "mistral"

    @patch("requests.get")
    def test_available_when_running(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        provider = OllamaProvider()
        assert provider.is_available() is True

    @patch("requests.get")
    def test_not_available_when_down(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        provider = OllamaProvider()
        assert provider.is_available() is False

    @patch("requests.post")
    def test_generate_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"message": {"content": "ollama response"}}),
        )
        provider = OllamaProvider()
        result = provider.generate("system", "user")
        assert result == "ollama response"

    @patch("requests.post")
    def test_generate_error(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="internal error")
        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="Ollama error"):
            provider.generate("system", "user")


# ---------------------------------------------------------------------------
# get_provider / get_best_available_provider
# ---------------------------------------------------------------------------

class TestGetProvider:

    def test_get_claude_provider(self):
        with patch("src.core.assistant.llm_providers.ClaudeProvider._init_client"):
            provider = get_provider("claude")
            assert isinstance(provider, ClaudeProvider)

    def test_get_openai_provider(self):
        with patch("src.core.assistant.llm_providers.OpenAIProvider._init_client"):
            provider = get_provider("openai")
            assert isinstance(provider, OpenAIProvider)

    def test_get_offline_provider(self):
        provider = get_provider("offline")
        assert isinstance(provider, OfflineProvider)

    def test_unknown_name_returns_offline(self):
        provider = get_provider("nonexistent_provider")
        assert isinstance(provider, OfflineProvider)

    def test_aliases(self):
        with patch("src.core.assistant.llm_providers.ClaudeProvider._init_client"):
            p = get_provider("anthropic")
            assert isinstance(p, ClaudeProvider)

        with patch("src.core.assistant.llm_providers.OpenAIProvider._init_client"):
            p = get_provider("gpt")
            assert isinstance(p, OpenAIProvider)

        p = get_provider("local")
        assert isinstance(p, OllamaProvider)


class TestGetBestAvailableProvider:

    @patch.dict(os.environ, {}, clear=True)
    def test_fallback_to_offline(self):
        """Without any API keys or services, should fall back to offline."""
        with patch("src.core.assistant.llm_providers.ClaudeProvider._init_client"):
            with patch("src.core.assistant.llm_providers.OpenAIProvider._init_client"):
                provider = get_best_available_provider()
                # Should eventually reach OfflineProvider
                assert provider.is_available() is True

    def test_returns_base_provider(self):
        provider = get_best_available_provider()
        assert isinstance(provider, BaseLLMProvider)
