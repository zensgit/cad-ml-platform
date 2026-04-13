"""
Unit tests for VLLMProvider.

Tests cover initialization, generation, streaming, health checks,
fallback behavior, and token counting.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.assistant.llm_providers import (
    LLMConfig,
    VLLMProvider,
    get_provider,
    get_best_available_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    """Create a VLLMProvider with default config."""
    return VLLMProvider()


@pytest.fixture
def provider_custom():
    """Create a VLLMProvider with custom config."""
    config = LLMConfig(temperature=0.7, max_tokens=512, timeout=10)
    with patch.dict(
        "os.environ",
        {
            "VLLM_ENDPOINT": "http://gpu-server:8100",
            "VLLM_MODEL": "custom-model-awq",
            "VLLM_TIMEOUT": "15",
        },
    ):
        return VLLMProvider(config)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestVLLMProviderInit:
    def test_default_init(self, provider):
        assert provider._base_url == "http://localhost:8100"
        assert provider._model == "deepseek-coder-6.7b-awq"
        assert provider._timeout == 30  # default from LLMConfig

    def test_custom_env_init(self, provider_custom):
        assert provider_custom._base_url == "http://gpu-server:8100"
        assert provider_custom._model == "custom-model-awq"
        assert provider_custom._timeout == 15

    def test_config_values_preserved(self, provider_custom):
        assert provider_custom.config.temperature == 0.7
        assert provider_custom.config.max_tokens == 512

    def test_get_provider_by_name(self):
        p = get_provider("vllm")
        assert isinstance(p, VLLMProvider)


# ---------------------------------------------------------------------------
# is_available tests
# ---------------------------------------------------------------------------

class TestVLLMProviderAvailability:
    @patch("requests.get")
    def test_available_via_health(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        assert provider.is_available() is True
        mock_get.assert_called_once_with(
            "http://localhost:8100/health", timeout=2
        )

    @patch("requests.get")
    def test_available_via_v1_models_fallback(self, mock_get, provider):
        """When /health fails, should try /v1/models."""
        health_resp = MagicMock()
        health_resp.status_code = 404

        models_resp = MagicMock()
        models_resp.status_code = 200

        mock_get.side_effect = [health_resp, models_resp]

        assert provider.is_available() is True
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_unavailable_when_both_fail(self, mock_get, provider):
        mock_get.side_effect = ConnectionError("refused")
        assert provider.is_available() is False

    @patch("requests.get")
    def test_unavailable_on_timeout(self, mock_get, provider):
        import requests
        mock_get.side_effect = requests.Timeout("timed out")
        assert provider.is_available() is False


# ---------------------------------------------------------------------------
# generate tests
# ---------------------------------------------------------------------------

class TestVLLMProviderGenerate:
    @patch("requests.post")
    def test_generate_success(self, mock_post, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": "304 stainless steel has UTS of 515 MPa."}}
            ]
        }
        mock_post.return_value = mock_resp

        result = provider.generate("You are an engineer.", "What is 304 UTS?")

        assert result == "304 stainless steel has UTS of 515 MPa."
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:8100/v1/chat/completions"
        payload = call_args[1]["json"]
        assert payload["model"] == "deepseek-coder-6.7b-awq"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2

    @patch("requests.post")
    def test_generate_error_raises(self, mock_post, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="vLLM error: 500"):
            provider.generate("system", "user")

    @patch("requests.post")
    def test_generate_timeout(self, mock_post, provider):
        import requests
        mock_post.side_effect = requests.Timeout("timed out")

        with pytest.raises(requests.Timeout):
            provider.generate("system", "user")


# ---------------------------------------------------------------------------
# generate_stream tests
# ---------------------------------------------------------------------------

class TestVLLMProviderStream:
    @patch("requests.post")
    def test_stream_success(self, mock_post, provider):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_resp

        chunks = list(provider.generate_stream("system", "user"))
        assert chunks == ["Hello", " world"]

    @patch("requests.post")
    def test_stream_error_raises(self, mock_post, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="vLLM stream error: 503"):
            list(provider.generate_stream("system", "user"))

    @patch("requests.post")
    def test_stream_skips_empty_and_malformed(self, mock_post, provider):
        lines = [
            "",
            "data: {malformed json",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_resp

        chunks = list(provider.generate_stream("system", "user"))
        assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# health_check tests
# ---------------------------------------------------------------------------

class TestVLLMProviderHealthCheck:
    @patch("requests.get")
    def test_healthy(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"id": "deepseek-coder-6.7b-awq"}]
        }
        mock_get.return_value = mock_resp

        result = provider.health_check()
        assert result["status"] == "healthy"
        assert "deepseek-coder-6.7b-awq" in result["models_available"]

    @patch("requests.get")
    def test_unhealthy_non_200(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_get.return_value = mock_resp

        result = provider.health_check()
        assert result["status"] == "unhealthy"

    @patch("requests.get")
    def test_connection_refused(self, mock_get, provider):
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")

        result = provider.health_check()
        assert result["status"] == "unavailable"
        assert result["error"] == "connection_refused"

    @patch("requests.get")
    def test_timeout(self, mock_get, provider):
        import requests
        mock_get.side_effect = requests.Timeout("timed out")

        result = provider.health_check()
        assert result["status"] == "unavailable"
        assert result["error"] == "timeout"


# ---------------------------------------------------------------------------
# token counting tests
# ---------------------------------------------------------------------------

class TestVLLMProviderTokenCounting:
    def test_empty_string(self, provider):
        assert provider.count_tokens("") == 0

    def test_english_text(self, provider):
        text = "Hello world, this is a test."
        tokens = provider.count_tokens(text)
        assert tokens > 0
        assert tokens == int(len(text) / 3.5)

    def test_cjk_text(self, provider):
        text = "304不锈钢的抗拉强度是515兆帕"
        tokens = provider.count_tokens(text)
        assert tokens > 0

    def test_minimum_one_token(self, provider):
        assert provider.count_tokens("a") == 1


# ---------------------------------------------------------------------------
# Fallback / integration with get_best_available_provider
# ---------------------------------------------------------------------------

class TestVLLMProviderFallback:
    @patch("requests.get")
    def test_fallback_when_unavailable(self, mock_get):
        """When vLLM is down, get_best_available_provider skips it."""
        mock_get.side_effect = ConnectionError("refused")

        # All providers will be unavailable except offline
        provider = get_best_available_provider()
        # Should not be VLLMProvider since it's not available
        assert not isinstance(provider, VLLMProvider)
