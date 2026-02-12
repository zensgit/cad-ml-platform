"""
LLM Provider Integrations.

Supports multiple LLM providers for response generation.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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
        "ollama": OllamaProvider,
        "local": OllamaProvider,
        "offline": OfflineProvider,
    }

    provider_class = providers.get(provider_name.lower(), OfflineProvider)
    return provider_class(config)


def get_best_available_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """
    Get the best available LLM provider.

    Checks providers in priority order: Claude > OpenAI > Qwen > Ollama > Offline

    Returns:
        Best available provider instance
    """
    priority_order = ["claude", "openai", "qwen", "ollama", "offline"]

    for provider_name in priority_order:
        provider = get_provider(provider_name, config)
        if provider.is_available():
            return provider

    return OfflineProvider(config)
