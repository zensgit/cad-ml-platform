"""Additional tests for knowledge providers to improve coverage.

Targets uncovered code paths in src/core/providers/knowledge.py:
- Lines 43-44: get_tolerance_value returns None
- Lines 47-48: health_check catches exception
- Lines 83-85: get_thread_spec returns None or exception
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.providers.knowledge import (
    KnowledgeProviderConfig,
    StandardsKnowledgeProviderAdapter,
    ToleranceKnowledgeProviderAdapter,
    bootstrap_core_knowledge_providers,
)
from src.core.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


# --- ToleranceKnowledgeProviderAdapter Tests ---


class TestToleranceKnowledgeProviderAdapter:
    """Tests for ToleranceKnowledgeProviderAdapter."""

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_tolerance_value_none(self):
        """Lines 43-44: health_check returns False when get_tolerance_value returns None."""
        config = KnowledgeProviderConfig(
            name="tolerance", provider_type="knowledge", provider_name="tolerance"
        )
        adapter = ToleranceKnowledgeProviderAdapter(config)

        with patch(
            "src.core.knowledge.tolerance.get_tolerance_value", return_value=None
        ):
            ok = await adapter.health_check()
            assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_exception(self):
        """Lines 47-48: health_check returns False on exception."""
        config = KnowledgeProviderConfig(
            name="tolerance", provider_type="knowledge", provider_name="tolerance"
        )
        adapter = ToleranceKnowledgeProviderAdapter(config)

        with patch(
            "src.core.knowledge.tolerance.get_tolerance_value",
            side_effect=RuntimeError("Test error"),
        ):
            ok = await adapter.health_check()
            assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_deviations_none(self):
        """Line 46: health_check returns False when get_limit_deviations returns None."""
        config = KnowledgeProviderConfig(
            name="tolerance", provider_type="knowledge", provider_name="tolerance"
        )
        adapter = ToleranceKnowledgeProviderAdapter(config)

        with patch(
            "src.core.knowledge.tolerance.get_tolerance_value", return_value=0.015
        ):
            with patch(
                "src.core.knowledge.tolerance.get_limit_deviations", return_value=None
            ):
                ok = await adapter.health_check()
                assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_all_checks_pass(self):
        """health_check returns True when all probes succeed."""
        config = KnowledgeProviderConfig(
            name="tolerance", provider_type="knowledge", provider_name="tolerance"
        )
        adapter = ToleranceKnowledgeProviderAdapter(config)

        # Use real implementation - should pass
        ok = await adapter.health_check()
        assert ok is True

    @pytest.mark.asyncio
    async def test_process_returns_status(self):
        """process returns status dict."""
        config = KnowledgeProviderConfig(
            name="tolerance", provider_type="knowledge", provider_name="tolerance"
        )
        adapter = ToleranceKnowledgeProviderAdapter(config)

        result = await adapter.process(None)
        assert result["status"] == "ok"
        assert "counts" in result
        assert "examples" in result


# --- StandardsKnowledgeProviderAdapter Tests ---


class TestStandardsKnowledgeProviderAdapter:
    """Tests for StandardsKnowledgeProviderAdapter."""

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_thread_spec_none(self):
        """Lines 83-84: health_check returns False when get_thread_spec returns None."""
        config = KnowledgeProviderConfig(
            name="standards", provider_type="knowledge", provider_name="standards"
        )
        adapter = StandardsKnowledgeProviderAdapter(config)

        with patch(
            "src.core.knowledge.standards.get_thread_spec", return_value=None
        ):
            ok = await adapter.health_check()
            assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_exception(self):
        """Lines 84-85: health_check returns False on exception."""
        config = KnowledgeProviderConfig(
            name="standards", provider_type="knowledge", provider_name="standards"
        )
        adapter = StandardsKnowledgeProviderAdapter(config)

        with patch(
            "src.core.knowledge.standards.get_thread_spec",
            side_effect=RuntimeError("Test error"),
        ):
            ok = await adapter.health_check()
            assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_all_checks_pass(self):
        """health_check returns True when probe succeeds."""
        config = KnowledgeProviderConfig(
            name="standards", provider_type="knowledge", provider_name="standards"
        )
        adapter = StandardsKnowledgeProviderAdapter(config)

        # Use real implementation - should pass
        ok = await adapter.health_check()
        assert ok is True

    @pytest.mark.asyncio
    async def test_process_returns_status(self):
        """process returns status dict."""
        config = KnowledgeProviderConfig(
            name="standards", provider_type="knowledge", provider_name="standards"
        )
        adapter = StandardsKnowledgeProviderAdapter(config)

        result = await adapter.process(None)
        assert result["status"] == "ok"
        assert "counts" in result
        assert "examples" in result


# --- Bootstrap Tests ---


class TestBootstrapCoreKnowledgeProviders:
    """Tests for bootstrap_core_knowledge_providers."""

    def test_bootstrap_registers_tolerance(self):
        """bootstrap registers knowledge/tolerance provider."""
        bootstrap_core_knowledge_providers()
        assert ProviderRegistry.exists("knowledge", "tolerance")

    def test_bootstrap_registers_standards(self):
        """bootstrap registers knowledge/standards provider."""
        bootstrap_core_knowledge_providers()
        assert ProviderRegistry.exists("knowledge", "standards")

    def test_bootstrap_is_idempotent(self):
        """bootstrap can be called multiple times without error."""
        bootstrap_core_knowledge_providers()
        first_tolerance = ProviderRegistry.get_provider_class("knowledge", "tolerance")

        bootstrap_core_knowledge_providers()  # Second call
        second_tolerance = ProviderRegistry.get_provider_class("knowledge", "tolerance")

        assert first_tolerance is second_tolerance

    def test_tolerance_provider_default_config_wiring(self):
        """Lines 115-120: default config for knowledge/tolerance."""
        bootstrap_core_knowledge_providers()
        provider = ProviderRegistry.get("knowledge", "tolerance")

        assert provider.config.name == "tolerance"
        assert provider.config.provider_type == "knowledge"
        assert provider.config.provider_name == "tolerance"

    def test_standards_provider_default_config_wiring(self):
        """Lines 127-132: default config for knowledge/standards."""
        bootstrap_core_knowledge_providers()
        provider = ProviderRegistry.get("knowledge", "standards")

        assert provider.config.name == "standards"
        assert provider.config.provider_type == "knowledge"
        assert provider.config.provider_name == "standards"


# --- KnowledgeProviderConfig Tests ---


class TestKnowledgeProviderConfig:
    """Tests for KnowledgeProviderConfig dataclass."""

    def test_default_values(self):
        """KnowledgeProviderConfig has correct defaults."""
        config = KnowledgeProviderConfig(name="test", provider_type="knowledge")
        assert config.provider_name == "unknown"

    def test_custom_values(self):
        """KnowledgeProviderConfig accepts custom values."""
        config = KnowledgeProviderConfig(
            name="custom",
            provider_type="knowledge",
            provider_name="tolerance",
        )
        assert config.provider_name == "tolerance"
