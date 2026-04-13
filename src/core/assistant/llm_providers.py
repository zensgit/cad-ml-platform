"""
LLM Provider Integrations.

Supports multiple LLM providers for response generation.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    api_key: Optional[str] = None
    model_name: str = "claude-3-sonnet-20240229"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 30  # seconds


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response from LLM.

        Args:
            system_prompt: System/context prompt
            user_prompt: User query with context

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured."""
        pass


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic

            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._client = Anthropic(api_key=api_key)
        except ImportError:
            pass

    def is_available(self) -> bool:
        """Check if Claude is available."""
        return self._client is not None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Claude."""
        if not self._client:
            raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY.")

        message = self._client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        return message.content[0].text


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if not self.config.model_name.startswith("gpt"):
            self.config.model_name = "gpt-4-turbo-preview"
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self._client = OpenAI(api_key=api_key)
        except ImportError:
            pass

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self._client is not None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using GPT."""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        response = self._client.chat.completions.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content


class QwenProvider(BaseLLMProvider):
    """Alibaba Qwen/Tongyi provider (通义千问)."""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if "qwen" not in self.config.model_name.lower():
            self.config.model_name = "qwen-turbo"
        self._api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")

    def is_available(self) -> bool:
        """Check if Qwen is available."""
        try:
            import dashscope
            return self._api_key is not None
        except ImportError:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Qwen."""
        if not self._api_key:
            raise RuntimeError("DashScope API key not set. Set DASHSCOPE_API_KEY.")

        import dashscope
        from dashscope import Generation

        dashscope.api_key = self._api_key

        response = Generation.call(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            result_format="message",
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError(f"Qwen API error: {response.code} - {response.message}")


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if not self.config.model_name or "claude" in self.config.model_name:
            self.config.model_name = "llama3"
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            response = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Ollama."""
        import requests

        response = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
                "stream": False,
            },
            timeout=self.config.timeout,
        )

        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise RuntimeError(f"Ollama error: {response.status_code} - {response.text}")


class VLLMProvider(BaseLLMProvider):
    """
    vLLM local inference provider.

    Connects to a vLLM server via its OpenAI-compatible HTTP API.
    Designed for low-latency (<100ms) local inference with quantized models
    such as DeepSeek-Coder-6.7B AWQ.

    Configuration via environment variables:
        VLLM_ENDPOINT: Base URL of the vLLM server (default: http://localhost:8100)
        VLLM_MODEL: Model name served by vLLM (default: deepseek-coder-6.7b-awq)
        VLLM_TIMEOUT: Request timeout in seconds (default: 30)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._base_url = os.getenv("VLLM_ENDPOINT", "http://localhost:8100")
        self._model = os.getenv("VLLM_MODEL", "deepseek-coder-6.7b-awq")
        self._timeout = int(os.getenv("VLLM_TIMEOUT", str(self.config.timeout)))

    def is_available(self) -> bool:
        """Check if vLLM server is reachable via its health endpoint."""
        try:
            import requests

            # Try /health first (vLLM native), fall back to /v1/models (OpenAI compat)
            for path in ("/health", "/v1/models"):
                try:
                    resp = requests.get(
                        f"{self._base_url}{path}", timeout=2
                    )
                    if resp.status_code == 200:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using vLLM's OpenAI-compatible chat completions API."""
        import requests

        response = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False,
            },
            timeout=self._timeout,
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(
                f"vLLM error: {response.status_code} - {response.text}"
            )

    def generate_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        """Generate streaming response using server-sent events (SSE).

        Yields:
            Text chunks as they arrive from vLLM.
        """
        import requests

        response = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
            },
            timeout=self._timeout,
            stream=True,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM stream error: {response.status_code} - {response.text}"
            )

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def count_tokens(self, text: str) -> int:
        """Estimate token count.

        Uses a simple heuristic (characters / 3.5 for mixed CJK/English text).
        For precise counts, the vLLM /v1/tokenize endpoint can be used when
        available but is not required for basic operation.
        """
        if not text:
            return 0
        # CJK-heavy text averages ~1.5 chars/token; English ~4 chars/token.
        # Blend for mixed CAD domain text.
        return max(1, int(len(text) / 3.5))

    def health_check(self) -> dict:
        """Return detailed health information from the vLLM server.

        Returns:
            Dict with keys: status, endpoint, model, and optionally models_available.
        """
        import requests

        result = {
            "status": "unavailable",
            "endpoint": self._base_url,
            "model": self._model,
        }

        try:
            resp = requests.get(f"{self._base_url}/v1/models", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                result["status"] = "healthy"
                result["models_available"] = models
            else:
                result["status"] = "unhealthy"
                result["error"] = f"HTTP {resp.status_code}"
        except requests.ConnectionError:
            result["error"] = "connection_refused"
        except requests.Timeout:
            result["error"] = "timeout"
        except Exception as e:
            result["error"] = str(e)

        return result


class OfflineProvider(BaseLLMProvider):
    """
    Offline provider - uses knowledge base only without external LLM.

    Useful for testing or environments without LLM access.
    """

    def is_available(self) -> bool:
        """Always available."""
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response from knowledge context only."""
        # Extract knowledge section from user prompt
        if "参考知识:" in user_prompt:
            parts = user_prompt.split("参考知识:")
            if len(parts) > 1:
                knowledge = parts[1].split("请基于以上知识")[0].strip()
                if knowledge and knowledge != "未找到相关知识。":
                    return f"根据知识库查询结果:\n\n{knowledge}\n\n[离线模式 - 仅展示知识库内容]"

        return "知识库中未找到相关信息。[离线模式]"


def get_provider(provider_name: str, config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """
    Get LLM provider by name.

    Args:
        provider_name: Provider name (claude, openai, qwen, ollama, offline)
        config: Optional LLM configuration

    Returns:
        Configured LLM provider instance

    Example:
        >>> provider = get_provider("claude")
        >>> if provider.is_available():
        ...     response = provider.generate(system_prompt, user_prompt)
    """
    providers = {
        "claude": ClaudeProvider,
        "anthropic": ClaudeProvider,
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,
        "gpt4": OpenAIProvider,
        "qwen": QwenProvider,
        "tongyi": QwenProvider,
        "vllm": VLLMProvider,
        "ollama": OllamaProvider,
        "local": OllamaProvider,
        "offline": OfflineProvider,
    }

    provider_class = providers.get(provider_name.lower(), OfflineProvider)
    return provider_class(config)


def get_best_available_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """
    Get the best available LLM provider.

    Checks providers in priority order: Claude > OpenAI > Qwen > vLLM > Ollama > Offline

    Returns:
        Best available provider instance
    """
    priority_order = ["claude", "openai", "qwen", "vllm", "ollama", "offline"]

    for provider_name in priority_order:
        provider = get_provider(provider_name, config)
        if provider.is_available():
            return provider

    return OfflineProvider(config)
