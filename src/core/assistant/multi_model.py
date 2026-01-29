"""
Multi-Model Support Module for CAD Assistant.

Provides model switching, load balancing, and fallback capabilities.
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ModelProvider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    CLAUDE = "claude"
    QWEN = "qwen"
    OLLAMA = "ollama"
    OFFLINE = "offline"


class ModelStatus(Enum):
    """Model availability status."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"


@dataclass
class ModelConfig:
    """Configuration for a model."""

    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    priority: int = 1  # Lower = higher priority
    weight: float = 1.0  # For load balancing
    timeout: float = 30.0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "priority": self.priority,
            "weight": self.weight,
            "timeout": self.timeout,
            "enabled": self.enabled,
        }


@dataclass
class ModelHealth:
    """Health status of a model."""

    provider: ModelProvider
    status: ModelStatus
    latency_ms: float = 0
    error_rate: float = 0
    last_check: float = field(default_factory=time.time)
    last_error: Optional[str] = None

    def is_healthy(self) -> bool:
        """Check if model is healthy enough to use."""
        return self.status in [ModelStatus.AVAILABLE, ModelStatus.DEGRADED]


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_LATENCY = "least_latency"
    PRIORITY = "priority"
    RANDOM = "random"


class ModelSelector:
    """
    Selects models based on configured strategy.

    Supports multiple load balancing strategies and automatic failover.
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.PRIORITY,
    ):
        """
        Initialize model selector.

        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self._models: Dict[ModelProvider, ModelConfig] = {}
        self._health: Dict[ModelProvider, ModelHealth] = {}
        self._round_robin_index = 0

    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self._models[config.provider] = config
        self._health[config.provider] = ModelHealth(
            provider=config.provider,
            status=ModelStatus.AVAILABLE,
        )

    def unregister_model(self, provider: ModelProvider) -> None:
        """Unregister a model."""
        self._models.pop(provider, None)
        self._health.pop(provider, None)

    def update_health(
        self,
        provider: ModelProvider,
        status: ModelStatus,
        latency_ms: float = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update model health status."""
        if provider in self._health:
            health = self._health[provider]
            health.status = status
            health.latency_ms = latency_ms
            health.last_check = time.time()
            if error:
                health.last_error = error
                health.error_rate = min(health.error_rate + 0.1, 1.0)
            else:
                health.error_rate = max(health.error_rate - 0.05, 0.0)

    def select_model(self) -> Optional[ModelConfig]:
        """
        Select a model based on the configured strategy.

        Returns:
            Selected model config or None if no models available
        """
        available = self._get_available_models()
        if not available:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._select_weighted(available)
        elif self.strategy == LoadBalancingStrategy.LEAST_LATENCY:
            return self._select_least_latency(available)
        elif self.strategy == LoadBalancingStrategy.PRIORITY:
            return self._select_priority(available)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available)

        return available[0] if available else None

    def select_with_fallback(self) -> List[ModelConfig]:
        """
        Get ordered list of models with fallbacks.

        Returns:
            List of models in order of preference
        """
        available = self._get_available_models()
        return sorted(available, key=lambda m: m.priority)

    def _get_available_models(self) -> List[ModelConfig]:
        """Get list of available models."""
        available = []
        for provider, config in self._models.items():
            if not config.enabled:
                continue
            health = self._health.get(provider)
            if health and health.is_healthy():
                available.append(config)
        return available

    def _select_round_robin(self, models: List[ModelConfig]) -> ModelConfig:
        """Round-robin selection."""
        self._round_robin_index = (self._round_robin_index + 1) % len(models)
        return models[self._round_robin_index]

    def _select_weighted(self, models: List[ModelConfig]) -> ModelConfig:
        """Weighted random selection."""
        total_weight = sum(m.weight for m in models)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for model in models:
            cumulative += model.weight
            if r <= cumulative:
                return model
        return models[-1]

    def _select_least_latency(self, models: List[ModelConfig]) -> ModelConfig:
        """Select model with lowest latency."""
        return min(
            models,
            key=lambda m: self._health.get(m.provider, ModelHealth(m.provider, ModelStatus.AVAILABLE)).latency_ms
        )

    def _select_priority(self, models: List[ModelConfig]) -> ModelConfig:
        """Select highest priority (lowest number) model."""
        return min(models, key=lambda m: m.priority)

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all models."""
        return {
            provider.value: {
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "error_rate": health.error_rate,
                "last_check": health.last_check,
                "last_error": health.last_error,
            }
            for provider, health in self._health.items()
        }


class MultiModelAssistant:
    """
    Assistant with multi-model support.

    Provides automatic model switching and failover.

    Example:
        >>> assistant = MultiModelAssistant()
        >>> assistant.add_model(ModelConfig(
        ...     provider=ModelProvider.OPENAI,
        ...     model_name="gpt-4",
        ...     api_key="sk-...",
        ...     priority=1
        ... ))
        >>> assistant.add_model(ModelConfig(
        ...     provider=ModelProvider.CLAUDE,
        ...     model_name="claude-3",
        ...     api_key="sk-...",
        ...     priority=2
        ... ))
        >>> result = await assistant.ask("304不锈钢的强度")
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.PRIORITY,
        max_retries: int = 3,
    ):
        """
        Initialize multi-model assistant.

        Args:
            strategy: Load balancing strategy
            max_retries: Maximum retry attempts
        """
        self.selector = ModelSelector(strategy)
        self.max_retries = max_retries
        self._providers: Dict[ModelProvider, Any] = {}

    def add_model(self, config: ModelConfig) -> None:
        """Add a model configuration."""
        self.selector.register_model(config)

        # Initialize provider
        provider = self._create_provider(config)
        if provider:
            self._providers[config.provider] = provider

    def remove_model(self, provider: ModelProvider) -> None:
        """Remove a model."""
        self.selector.unregister_model(provider)
        self._providers.pop(provider, None)

    def _create_provider(self, config: ModelConfig) -> Optional[Any]:
        """Create provider instance based on config."""
        # Lazy import to avoid circular dependencies
        try:
            from .llm_providers import (
                OpenAIProvider,
                ClaudeProvider,
                QwenProvider,
                OllamaProvider,
                OfflineProvider,
                LLMConfig,
            )

            llm_config = LLMConfig(
                model=config.model_name,
                api_key=config.api_key or "",
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            providers = {
                ModelProvider.OPENAI: OpenAIProvider,
                ModelProvider.CLAUDE: ClaudeProvider,
                ModelProvider.QWEN: QwenProvider,
                ModelProvider.OLLAMA: OllamaProvider,
                ModelProvider.OFFLINE: OfflineProvider,
            }

            provider_class = providers.get(config.provider)
            if provider_class:
                return provider_class(llm_config)
        except ImportError:
            pass

        return None

    async def ask(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Tuple[str, ModelProvider]:
        """
        Ask a question with automatic failover.

        Args:
            query: User query
            context: Optional context

        Returns:
            Tuple of (response, provider_used)
        """
        models = self.selector.select_with_fallback()

        for attempt, model in enumerate(models[:self.max_retries]):
            provider = self._providers.get(model.provider)
            if not provider:
                continue

            try:
                start_time = time.time()

                # Call provider
                if hasattr(provider, 'generate_async'):
                    response = await provider.generate_async(
                        query,
                        context=context,
                    )
                else:
                    response = provider.generate(
                        query,
                        context=context,
                    )

                latency = (time.time() - start_time) * 1000

                # Update health
                self.selector.update_health(
                    model.provider,
                    ModelStatus.AVAILABLE,
                    latency_ms=latency,
                )

                return response, model.provider

            except Exception as e:
                # Update health on failure
                self.selector.update_health(
                    model.provider,
                    ModelStatus.UNAVAILABLE,
                    error=str(e),
                )

        raise RuntimeError("All models failed")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        return {
            "models": [
                {
                    **config.to_dict(),
                    "health": self.selector._health.get(config.provider, {})
                }
                for config in self.selector._models.values()
            ],
            "strategy": self.selector.strategy.value,
        }
