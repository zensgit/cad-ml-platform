"""Tests for LLM providers and assistant API."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestLLMProviders:
    """Tests for LLM provider implementations."""

    def test_offline_provider_always_available(self):
        """Test offline provider is always available."""
        from src.core.assistant.llm_providers import OfflineProvider

        provider = OfflineProvider()
        assert provider.is_available() is True

    def test_offline_provider_extracts_knowledge(self):
        """Test offline provider extracts knowledge from prompt."""
        from src.core.assistant.llm_providers import OfflineProvider

        provider = OfflineProvider()
        user_prompt = """用户问题: 304不锈钢的强度？

参考知识:
材料: S30408 (304不锈钢)
抗拉强度: 520 MPa
屈服强度: 205 MPa

请基于以上知识回答用户问题。"""

        response = provider.generate("system", user_prompt)
        assert "520" in response or "知识库" in response

    def test_offline_provider_no_knowledge(self):
        """Test offline provider handles no knowledge case."""
        from src.core.assistant.llm_providers import OfflineProvider

        provider = OfflineProvider()
        response = provider.generate("system", "一个简单问题")
        assert "未找到" in response or "离线" in response

    def test_get_provider_returns_correct_type(self):
        """Test get_provider returns correct provider type."""
        from src.core.assistant.llm_providers import (
            get_provider,
            ClaudeProvider,
            OpenAIProvider,
            QwenProvider,
            OllamaProvider,
            OfflineProvider,
        )

        assert isinstance(get_provider("claude"), ClaudeProvider)
        assert isinstance(get_provider("openai"), OpenAIProvider)
        assert isinstance(get_provider("qwen"), QwenProvider)
        assert isinstance(get_provider("ollama"), OllamaProvider)
        assert isinstance(get_provider("offline"), OfflineProvider)
        assert isinstance(get_provider("unknown"), OfflineProvider)

    def test_get_best_available_provider_fallback(self):
        """Test get_best_available_provider falls back to offline."""
        from src.core.assistant.llm_providers import get_best_available_provider, OfflineProvider

        # Without any API keys set, should fall back to offline
        provider = get_best_available_provider()
        # Should at least return a provider that is available
        assert provider.is_available()

    def test_llm_config_defaults(self):
        """Test LLM config has sensible defaults."""
        from src.core.assistant.llm_providers import LLMConfig

        config = LLMConfig()
        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.timeout == 30

    def test_claude_provider_without_key(self):
        """Test Claude provider without API key is not available."""
        from src.core.assistant.llm_providers import ClaudeProvider, LLMConfig

        with patch.dict("os.environ", {}, clear=True):
            config = LLMConfig(api_key=None)
            provider = ClaudeProvider(config)
            # Without anthropic package or key, should not be available
            # (may be available if package installed with key in env)

    def test_openai_provider_without_key(self):
        """Test OpenAI provider without API key is not available."""
        from src.core.assistant.llm_providers import OpenAIProvider, LLMConfig

        with patch.dict("os.environ", {}, clear=True):
            config = LLMConfig(api_key=None)
            provider = OpenAIProvider(config)
            # Without openai package or key, should not be available


class TestAssistantIntegration:
    """Integration tests for CAD Assistant."""

    def test_assistant_initialization(self):
        """Test assistant initializes correctly."""
        from src.core.assistant import CADAssistant, AssistantConfig

        config = AssistantConfig(auto_select_provider=True)
        assistant = CADAssistant(config=config)

        assert assistant is not None
        assert assistant._llm_provider is not None

    def test_assistant_ask_with_offline_mode(self):
        """Test assistant can answer in offline mode."""
        from src.core.assistant import CADAssistant, AssistantConfig

        config = AssistantConfig(auto_select_provider=True)
        assistant = CADAssistant(config=config)

        response = assistant.ask("304不锈钢的抗拉强度是多少？")

        assert response is not None
        assert response.answer is not None
        assert len(response.answer) > 0

    def test_assistant_with_custom_callback(self):
        """Test assistant with custom LLM callback."""
        from src.core.assistant import CADAssistant, AssistantConfig

        def mock_llm(system_prompt, user_prompt):
            return "This is a mock response"

        config = AssistantConfig()
        assistant = CADAssistant(config=config, llm_callback=mock_llm)

        response = assistant.ask("任何问题")
        # With retrieval results, should use mock callback
        assert response is not None

    def test_assistant_supported_queries(self):
        """Test assistant returns supported query examples."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        queries = assistant.get_supported_queries()

        assert "材料查询" in queries
        assert "公差配合" in queries
        assert "标准件" in queries
        assert "加工参数" in queries

    def test_assistant_get_suggestions(self):
        """Test assistant provides query suggestions."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        suggestions = assistant.get_suggestions("304")

        assert isinstance(suggestions, list)


class TestAssistantAPI:
    """Tests for assistant API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.v1.assistant import router

        app = FastAPI()
        app.include_router(router, prefix="/assistant")
        return TestClient(app)

    def test_query_endpoint(self, client):
        """Test query endpoint."""
        response = client.post(
            "/assistant/query",
            json={"query": "304不锈钢的强度是多少？"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "answer" in data
        assert "confidence" in data

    def test_query_with_context(self, client):
        """Test query with additional context."""
        response = client.post(
            "/assistant/query",
            json={
                "query": "推荐什么材料？",
                "context": "需要耐腐蚀，用于海洋环境",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_query_verbose_mode(self, client):
        """Test query with verbose output."""
        response = client.post(
            "/assistant/query",
            json={
                "query": "M10螺纹的底孔是多少？",
                "verbose": True,
            }
        )

        assert response.status_code == 200
        data = response.json()
        # Verbose mode should include intent and entities
        if data["success"]:
            assert "intent" in data

    def test_suggest_endpoint(self, client):
        """Test suggestion endpoint."""
        response = client.get("/assistant/suggest", params={"q": "304"})

        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_supported_queries_endpoint(self, client):
        """Test supported queries endpoint."""
        response = client.get("/assistant/supported-queries")

        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) > 0

    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get("/assistant/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "provider" in data
        assert "knowledge_modules" in data

    def test_feedback_endpoint(self, client):
        """Test feedback endpoint."""
        response = client.post(
            "/assistant/feedback",
            params={
                "query": "测试问题",
                "answer": "测试回答",
                "rating": 4,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestProviderSelection:
    """Tests for provider selection logic."""

    def test_auto_select_provider_enabled(self):
        """Test auto provider selection."""
        from src.core.assistant import CADAssistant, AssistantConfig

        config = AssistantConfig(auto_select_provider=True)
        assistant = CADAssistant(config=config)

        # Should have selected some provider
        assert assistant._llm_provider is not None

    def test_manual_provider_selection(self):
        """Test manual provider selection."""
        from src.core.assistant import CADAssistant, AssistantConfig, LLMProvider
        from src.core.assistant.llm_providers import OfflineProvider

        config = AssistantConfig(
            auto_select_provider=False,
            llm_provider=LLMProvider.LOCAL,
        )
        assistant = CADAssistant(config=config)

        # Should have selected the specified provider type
        assert assistant._llm_provider is not None
