"""
Unit tests for multi_model.py - Multi-Model Support Module.

Tests cover:
- ModelConfig creation and serialization
- ModelHealth status tracking
- ModelSelector with different strategies
- MultiModelAssistant failover behavior
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import time

from src.core.assistant.multi_model import (
    ModelProvider,
    ModelStatus,
    ModelConfig,
    ModelHealth,
    LoadBalancingStrategy,
    ModelSelector,
    MultiModelAssistant,
)


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_model_config_creation(self):
        """Test basic config creation."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            api_key="sk-xxx",
            max_tokens=2000,
        )
        assert config.provider == ModelProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.max_tokens == 2000
        assert config.enabled is True

    def test_model_config_defaults(self):
        """Test default values."""
        config = ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        )
        assert config.api_key is None
        assert config.temperature == 0.7
        assert config.priority == 1
        assert config.weight == 1.0
        assert config.timeout == 30.0

    def test_model_config_to_dict(self):
        """Test conversion to dict excludes sensitive data."""
        config = ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude-3",
            api_key="sk-secret",
        )
        result = config.to_dict()

        assert "api_key" not in result
        assert result["provider"] == "claude"
        assert result["model_name"] == "claude-3"

    def test_all_providers(self):
        """Test all provider types."""
        providers = [
            ModelProvider.OPENAI,
            ModelProvider.CLAUDE,
            ModelProvider.QWEN,
            ModelProvider.OLLAMA,
            ModelProvider.OFFLINE,
        ]
        for provider in providers:
            config = ModelConfig(provider=provider, model_name="test")
            assert config.provider == provider


class TestModelHealth:
    """Tests for ModelHealth class."""

    def test_model_health_creation(self):
        """Test health status creation."""
        health = ModelHealth(
            provider=ModelProvider.OPENAI,
            status=ModelStatus.AVAILABLE,
        )
        assert health.provider == ModelProvider.OPENAI
        assert health.status == ModelStatus.AVAILABLE
        assert health.latency_ms == 0
        assert health.error_rate == 0

    def test_is_healthy_available(self):
        """Test is_healthy for available status."""
        health = ModelHealth(
            provider=ModelProvider.OPENAI,
            status=ModelStatus.AVAILABLE,
        )
        assert health.is_healthy() is True

    def test_is_healthy_degraded(self):
        """Test is_healthy for degraded status."""
        health = ModelHealth(
            provider=ModelProvider.OPENAI,
            status=ModelStatus.DEGRADED,
        )
        assert health.is_healthy() is True

    def test_is_healthy_unavailable(self):
        """Test is_healthy for unavailable status."""
        health = ModelHealth(
            provider=ModelProvider.OPENAI,
            status=ModelStatus.UNAVAILABLE,
        )
        assert health.is_healthy() is False

    def test_is_healthy_rate_limited(self):
        """Test is_healthy for rate limited status."""
        health = ModelHealth(
            provider=ModelProvider.OPENAI,
            status=ModelStatus.RATE_LIMITED,
        )
        assert health.is_healthy() is False

    def test_all_statuses(self):
        """Test all status types."""
        statuses = [
            ModelStatus.AVAILABLE,
            ModelStatus.DEGRADED,
            ModelStatus.UNAVAILABLE,
            ModelStatus.RATE_LIMITED,
        ]
        for status in statuses:
            health = ModelHealth(provider=ModelProvider.OPENAI, status=status)
            assert health.status == status


class TestModelSelector:
    """Tests for ModelSelector class."""

    @pytest.fixture
    def selector(self):
        """Create selector with test models."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.PRIORITY)

        # Register multiple models
        selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            priority=1,
        ))
        selector.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude-3",
            priority=2,
        ))
        selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
            priority=3,
        ))

        return selector

    def test_register_model(self):
        """Test model registration."""
        selector = ModelSelector()
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
        )
        selector.register_model(config)

        assert ModelProvider.OPENAI in selector._models
        assert ModelProvider.OPENAI in selector._health

    def test_unregister_model(self, selector):
        """Test model unregistration."""
        selector.unregister_model(ModelProvider.OPENAI)

        assert ModelProvider.OPENAI not in selector._models
        assert ModelProvider.OPENAI not in selector._health

    def test_update_health(self, selector):
        """Test health status update."""
        selector.update_health(
            ModelProvider.OPENAI,
            ModelStatus.DEGRADED,
            latency_ms=500,
        )

        health = selector._health[ModelProvider.OPENAI]
        assert health.status == ModelStatus.DEGRADED
        assert health.latency_ms == 500

    def test_update_health_with_error(self, selector):
        """Test health update with error increases error rate."""
        initial_rate = selector._health[ModelProvider.OPENAI].error_rate

        selector.update_health(
            ModelProvider.OPENAI,
            ModelStatus.UNAVAILABLE,
            error="Connection timeout",
        )

        health = selector._health[ModelProvider.OPENAI]
        assert health.error_rate > initial_rate
        assert health.last_error == "Connection timeout"

    def test_update_health_success_decreases_error_rate(self, selector):
        """Test successful health update decreases error rate."""
        # First set some error rate
        selector._health[ModelProvider.OPENAI].error_rate = 0.5

        selector.update_health(
            ModelProvider.OPENAI,
            ModelStatus.AVAILABLE,
            latency_ms=100,
        )

        assert selector._health[ModelProvider.OPENAI].error_rate < 0.5

    def test_select_model_priority_strategy(self, selector):
        """Test priority-based selection."""
        selected = selector.select_model()

        # Should select OpenAI (priority 1)
        assert selected.provider == ModelProvider.OPENAI

    def test_select_model_with_unavailable(self, selector):
        """Test selection skips unavailable models."""
        selector.update_health(ModelProvider.OPENAI, ModelStatus.UNAVAILABLE)

        selected = selector.select_model()

        # Should select Claude (priority 2) since OpenAI is unavailable
        assert selected.provider == ModelProvider.CLAUDE

    def test_select_model_round_robin(self):
        """Test round-robin selection."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.ROUND_ROBIN)

        for i in range(3):
            selector.register_model(ModelConfig(
                provider=list(ModelProvider)[i],
                model_name=f"model-{i}",
            ))

        # Should cycle through models
        selections = [selector.select_model().provider for _ in range(6)]
        # Round robin should hit each model multiple times
        assert len(set(selections)) >= 2

    def test_select_model_weighted(self):
        """Test weighted selection."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.WEIGHTED)

        selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            weight=9.0,  # Heavy weight
        ))
        selector.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude",
            weight=1.0,  # Light weight
        ))

        # Run multiple selections
        selections = [selector.select_model().provider for _ in range(100)]
        openai_count = selections.count(ModelProvider.OPENAI)

        # OpenAI should be selected much more often
        assert openai_count > 60

    def test_select_model_least_latency(self):
        """Test least latency selection."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.LEAST_LATENCY)

        selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
        ))
        selector.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude",
        ))

        # Set different latencies
        selector.update_health(ModelProvider.OPENAI, ModelStatus.AVAILABLE, latency_ms=500)
        selector.update_health(ModelProvider.CLAUDE, ModelStatus.AVAILABLE, latency_ms=100)

        selected = selector.select_model()
        assert selected.provider == ModelProvider.CLAUDE

    def test_select_model_random(self):
        """Test random selection."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.RANDOM)

        selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
        ))
        selector.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude",
        ))

        # Should select randomly (both should appear over many tries)
        selections = {selector.select_model().provider for _ in range(50)}
        assert len(selections) >= 1  # At least one provider selected

    def test_select_model_no_available(self, selector):
        """Test selection when no models available."""
        for provider in [ModelProvider.OPENAI, ModelProvider.CLAUDE, ModelProvider.OFFLINE]:
            selector.update_health(provider, ModelStatus.UNAVAILABLE)

        selected = selector.select_model()
        assert selected is None

    def test_select_model_disabled(self, selector):
        """Test selection skips disabled models."""
        selector._models[ModelProvider.OPENAI].enabled = False

        selected = selector.select_model()
        assert selected.provider == ModelProvider.CLAUDE

    def test_select_with_fallback(self, selector):
        """Test fallback list generation."""
        fallbacks = selector.select_with_fallback()

        assert len(fallbacks) == 3
        # Should be ordered by priority
        assert fallbacks[0].provider == ModelProvider.OPENAI
        assert fallbacks[1].provider == ModelProvider.CLAUDE
        assert fallbacks[2].provider == ModelProvider.OFFLINE

    def test_get_all_health(self, selector):
        """Test getting all health statuses."""
        selector.update_health(ModelProvider.OPENAI, ModelStatus.AVAILABLE, latency_ms=100)

        health = selector.get_all_health()

        assert "openai" in health
        assert health["openai"]["status"] == "available"
        assert health["openai"]["latency_ms"] == 100


class TestMultiModelAssistant:
    """Tests for MultiModelAssistant class."""

    @pytest.fixture
    def assistant(self):
        """Create multi-model assistant."""
        return MultiModelAssistant(
            strategy=LoadBalancingStrategy.PRIORITY,
            max_retries=3,
        )

    def test_add_model(self, assistant):
        """Test adding a model."""
        config = ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        )
        # Mock the provider creation
        with patch.object(assistant, '_create_provider', return_value=MagicMock()):
            assistant.add_model(config)

        assert ModelProvider.OFFLINE in assistant.selector._models

    def test_remove_model(self, assistant):
        """Test removing a model."""
        config = ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        )
        with patch.object(assistant, '_create_provider', return_value=MagicMock()):
            assistant.add_model(config)
        assistant.remove_model(ModelProvider.OFFLINE)

        assert ModelProvider.OFFLINE not in assistant.selector._models

    @pytest.mark.asyncio
    async def test_ask_success(self, assistant):
        """Test successful ask."""
        # Add offline model (always available)
        config = ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        )
        # Register model in selector directly
        assistant.selector.register_model(config)

        # Mock the provider - use spec to avoid MagicMock auto-creating generate_async
        mock_provider = MagicMock(spec=['generate'])
        mock_provider.generate.return_value = "Test response"
        assistant._providers[ModelProvider.OFFLINE] = mock_provider

        response, provider = await assistant.ask("What is steel?")

        assert response == "Test response"
        assert provider == ModelProvider.OFFLINE

    @pytest.mark.asyncio
    async def test_ask_with_context(self, assistant):
        """Test ask with context."""
        config = ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        )
        assistant.selector.register_model(config)

        mock_provider = MagicMock(spec=['generate'])
        mock_provider.generate.return_value = "Response with context"
        assistant._providers[ModelProvider.OFFLINE] = mock_provider

        response, _ = await assistant.ask("Query", context="Some context")

        mock_provider.generate.assert_called_with("Query", context="Some context")

    @pytest.mark.asyncio
    async def test_ask_failover(self, assistant):
        """Test failover to next model on failure."""
        # Add two models
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            priority=1,
        ))
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
            priority=2,
        ))

        # First provider fails
        mock_openai = MagicMock(spec=['generate'])
        mock_openai.generate.side_effect = Exception("API Error")
        assistant._providers[ModelProvider.OPENAI] = mock_openai

        # Second provider succeeds
        mock_offline = MagicMock(spec=['generate'])
        mock_offline.generate.return_value = "Fallback response"
        assistant._providers[ModelProvider.OFFLINE] = mock_offline

        response, provider = await assistant.ask("Query")

        assert response == "Fallback response"
        assert provider == ModelProvider.OFFLINE

    @pytest.mark.asyncio
    async def test_ask_all_fail(self, assistant):
        """Test when all models fail."""
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
        ))

        mock_provider = MagicMock(spec=['generate'])
        mock_provider.generate.side_effect = Exception("Error")
        assistant._providers[ModelProvider.OPENAI] = mock_provider

        with pytest.raises(RuntimeError, match="All models failed"):
            await assistant.ask("Query")

    @pytest.mark.asyncio
    async def test_ask_updates_health_on_success(self, assistant):
        """Test health is updated on successful call."""
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        ))

        mock_provider = MagicMock(spec=['generate'])
        mock_provider.generate.return_value = "Response"
        assistant._providers[ModelProvider.OFFLINE] = mock_provider

        await assistant.ask("Query")

        health = assistant.selector._health[ModelProvider.OFFLINE]
        assert health.status == ModelStatus.AVAILABLE
        assert health.latency_ms > 0

    @pytest.mark.asyncio
    async def test_ask_updates_health_on_failure(self, assistant):
        """Test health is updated on failed call."""
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
        ))
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
            priority=2,
        ))

        # OpenAI fails
        mock_openai = MagicMock(spec=['generate'])
        mock_openai.generate.side_effect = Exception("Failed")
        assistant._providers[ModelProvider.OPENAI] = mock_openai

        # Offline succeeds
        mock_offline = MagicMock(spec=['generate'])
        mock_offline.generate.return_value = "OK"
        assistant._providers[ModelProvider.OFFLINE] = mock_offline

        await assistant.ask("Query")

        health = assistant.selector._health[ModelProvider.OPENAI]
        assert health.status == ModelStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_ask_async_provider(self, assistant):
        """Test with async provider."""
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        ))

        mock_provider = MagicMock(spec=['generate_async'])
        mock_provider.generate_async = AsyncMock(return_value="Async response")
        assistant._providers[ModelProvider.OFFLINE] = mock_provider

        response, _ = await assistant.ask("Query")

        assert response == "Async response"
        mock_provider.generate_async.assert_called_once()

    def test_get_status(self, assistant):
        """Test status retrieval."""
        with patch.object(assistant, '_create_provider', return_value=None):
            assistant.add_model(ModelConfig(
                provider=ModelProvider.OFFLINE,
                model_name="offline",
            ))

        status = assistant.get_status()

        assert "models" in status
        assert "strategy" in status
        assert status["strategy"] == "priority"
        assert len(status["models"]) == 1


class TestLoadBalancingStrategies:
    """Tests for all load balancing strategies."""

    def test_all_strategies_exist(self):
        """Test all strategy types exist."""
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.WEIGHTED,
            LoadBalancingStrategy.LEAST_LATENCY,
            LoadBalancingStrategy.PRIORITY,
            LoadBalancingStrategy.RANDOM,
        ]
        assert len(strategies) == 5

    def test_strategy_values(self):
        """Test strategy string values."""
        assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancingStrategy.WEIGHTED.value == "weighted"
        assert LoadBalancingStrategy.LEAST_LATENCY.value == "least_latency"
        assert LoadBalancingStrategy.PRIORITY.value == "priority"
        assert LoadBalancingStrategy.RANDOM.value == "random"
