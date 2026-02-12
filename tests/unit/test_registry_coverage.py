"""Additional tests for registry to improve coverage.

Targets uncovered code paths in src/core/providers/registry.py:
- Line 96: unregister returns False when provider not found
- Line 102: unregister removes instance when exists
- Line 104: unregister removes domain from instances when empty
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class DummyProvider(BaseProvider[ProviderConfig, dict]):
    """Dummy provider for testing."""

    def __init__(self, config: ProviderConfig | None = None):
        super().__init__(config or ProviderConfig(name="dummy", provider_type="test"))

    async def _process_impl(self, request, **kwargs):
        return {"status": "ok"}


# --- Unregister Tests (Lines 96, 102, 104) ---


class TestUnregister:
    """Tests for ProviderRegistry.unregister method."""

    def test_unregister_nonexistent_domain_returns_false(self):
        """Line 96: unregister returns False when domain doesn't exist."""
        result = ProviderRegistry.unregister("nonexistent", "provider")
        assert result is False

    def test_unregister_nonexistent_provider_returns_false(self):
        """Line 96: unregister returns False when provider doesn't exist in domain."""
        @ProviderRegistry.register("test", "exists")
        class ExistsProvider(DummyProvider):
            pass

        result = ProviderRegistry.unregister("test", "nonexistent")
        assert result is False

    def test_unregister_existing_provider_returns_true(self):
        """unregister returns True when provider is removed."""
        @ProviderRegistry.register("test", "to_remove")
        class ToRemoveProvider(DummyProvider):
            pass

        assert ProviderRegistry.exists("test", "to_remove")
        result = ProviderRegistry.unregister("test", "to_remove")
        assert result is True
        assert not ProviderRegistry.exists("test", "to_remove")

    def test_unregister_removes_cached_instance(self):
        """Line 102: unregister removes cached instance."""
        @ProviderRegistry.register("test", "cached")
        class CachedProvider(DummyProvider):
            pass

        # Create a cached instance
        instance = ProviderRegistry.get("test", "cached")
        assert instance is not None

        # Unregister should remove the instance
        result = ProviderRegistry.unregister("test", "cached")
        assert result is True

        # Verify domain is removed from instances (Line 104)
        with ProviderRegistry._lock:
            assert "test" not in ProviderRegistry._instances

    def test_unregister_removes_empty_domain(self):
        """Lines 98-99: unregister removes domain when empty."""
        @ProviderRegistry.register("to_empty", "provider")
        class Provider(DummyProvider):
            pass

        assert "to_empty" in ProviderRegistry.list_domains()

        ProviderRegistry.unregister("to_empty", "provider")

        assert "to_empty" not in ProviderRegistry.list_domains()

    def test_unregister_keeps_domain_with_other_providers(self):
        """unregister keeps domain when other providers exist."""
        @ProviderRegistry.register("multi", "first")
        class FirstProvider(DummyProvider):
            pass

        @ProviderRegistry.register("multi", "second")
        class SecondProvider(DummyProvider):
            pass

        ProviderRegistry.unregister("multi", "first")

        assert "multi" in ProviderRegistry.list_domains()
        assert not ProviderRegistry.exists("multi", "first")
        assert ProviderRegistry.exists("multi", "second")

    def test_unregister_removes_instance_domain_when_empty(self):
        """Line 104: unregister removes instance domain when empty."""
        @ProviderRegistry.register("inst_test", "provider")
        class InstProvider(DummyProvider):
            pass

        # Create cached instance
        ProviderRegistry.get("inst_test", "provider")

        # Verify instance exists
        with ProviderRegistry._lock:
            assert "inst_test" in ProviderRegistry._instances

        # Unregister
        ProviderRegistry.unregister("inst_test", "provider")

        # Verify instance domain is removed
        with ProviderRegistry._lock:
            assert "inst_test" not in ProviderRegistry._instances


# --- Cache Behavior Tests ---


class TestCacheBehavior:
    """Tests for cache-related behavior."""

    def test_cache_disabled_creates_new_instances(self):
        """get creates new instances when cache disabled."""
        @ProviderRegistry.register("test", "no_cache")
        class NoCacheProvider(DummyProvider):
            pass

        with patch.dict(os.environ, {"PROVIDER_REGISTRY_CACHE_ENABLED": "false"}):
            inst1 = ProviderRegistry.get("test", "no_cache")
            inst2 = ProviderRegistry.get("test", "no_cache")
            assert inst1 is not inst2

    def test_cache_enabled_returns_same_instance(self):
        """get returns same instance when cache enabled."""
        @ProviderRegistry.register("test", "cached")
        class CachedProvider(DummyProvider):
            pass

        with patch.dict(os.environ, {"PROVIDER_REGISTRY_CACHE_ENABLED": "true"}):
            inst1 = ProviderRegistry.get("test", "cached")
            inst2 = ProviderRegistry.get("test", "cached")
            assert inst1 is inst2

    def test_get_with_args_bypasses_cache(self):
        """get with args always creates new instance."""
        @ProviderRegistry.register("test", "with_args")
        class WithArgsProvider(DummyProvider):
            def __init__(self, config=None):
                super().__init__(config or ProviderConfig(name="test", provider_type="test"))

        inst1 = ProviderRegistry.get("test", "with_args")
        inst2 = ProviderRegistry.get("test", "with_args", None)  # With arg
        assert inst1 is not inst2

    def test_get_with_kwargs_bypasses_cache(self):
        """get with kwargs always creates new instance."""
        @ProviderRegistry.register("test", "with_kwargs")
        class WithKwargsProvider(DummyProvider):
            def __init__(self, config=None):
                super().__init__(config or ProviderConfig(name="test", provider_type="test"))

        inst1 = ProviderRegistry.get("test", "with_kwargs")
        inst2 = ProviderRegistry.get("test", "with_kwargs", config=None)  # With kwarg
        assert inst1 is not inst2


# --- Clear Methods Tests ---


class TestClearMethods:
    """Tests for clear methods."""

    def test_clear_instances_keeps_registrations(self):
        """clear_instances keeps provider registrations."""
        @ProviderRegistry.register("test", "clear_test")
        class ClearTestProvider(DummyProvider):
            pass

        # Create instance
        ProviderRegistry.get("test", "clear_test")

        # Clear instances
        ProviderRegistry.clear_instances()

        # Provider class still registered
        assert ProviderRegistry.exists("test", "clear_test")

        # But instances are cleared (new instance created)
        with ProviderRegistry._lock:
            assert "test" not in ProviderRegistry._instances

    def test_clear_removes_everything(self):
        """clear removes both registrations and instances."""
        @ProviderRegistry.register("test", "full_clear")
        class FullClearProvider(DummyProvider):
            pass

        ProviderRegistry.get("test", "full_clear")

        ProviderRegistry.clear()

        assert not ProviderRegistry.exists("test", "full_clear")
        assert ProviderRegistry.list_domains() == []


# --- Registration Tests ---


class TestRegistration:
    """Tests for provider registration."""

    def test_duplicate_registration_raises_error(self):
        """Registering same provider twice raises ValueError."""
        @ProviderRegistry.register("test", "duplicate")
        class FirstProvider(DummyProvider):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @ProviderRegistry.register("test", "duplicate")
            class SecondProvider(DummyProvider):
                pass

    def test_get_nonexistent_raises_keyerror(self):
        """get_provider_class raises KeyError for nonexistent provider."""
        with pytest.raises(KeyError, match="Provider not found"):
            ProviderRegistry.get_provider_class("nonexistent", "provider")

    def test_register_rejects_non_base_provider_class(self):
        """Registration should fail fast when class is not BaseProvider."""

        with pytest.raises(TypeError, match="inherit BaseProvider"):

            @ProviderRegistry.register("test", "invalid")
            class _NotProvider:
                pass

    def test_register_rejects_empty_domain_or_provider_name(self):
        """Registration should reject empty identifiers."""
        with pytest.raises(ValueError, match="domain must be a non-empty string"):
            ProviderRegistry.register("", "valid")
        with pytest.raises(ValueError, match="provider_name must be a non-empty string"):
            ProviderRegistry.register("valid", " ")

    def test_register_rejects_separator_characters(self):
        """Registration should reject reserved separators for provider IDs."""
        with pytest.raises(ValueError, match="cannot contain '/' or ':'"):
            ProviderRegistry.register("test/domain", "provider")
        with pytest.raises(ValueError, match="cannot contain '/' or ':'"):
            ProviderRegistry.register("test", "provider:name")


# --- List Methods Tests ---


class TestListMethods:
    """Tests for list methods."""

    def test_list_domains_empty(self):
        """list_domains returns empty list when no providers."""
        assert ProviderRegistry.list_domains() == []

    def test_list_domains_sorted(self):
        """list_domains returns sorted domains."""
        @ProviderRegistry.register("zebra", "p")
        class ZebraProvider(DummyProvider):
            pass

        @ProviderRegistry.register("alpha", "p")
        class AlphaProvider(DummyProvider):
            pass

        domains = ProviderRegistry.list_domains()
        assert domains == ["alpha", "zebra"]

    def test_list_providers_empty_domain(self):
        """list_providers returns empty list for nonexistent domain."""
        assert ProviderRegistry.list_providers("nonexistent") == []

    def test_list_providers_sorted(self):
        """list_providers returns sorted provider names."""
        @ProviderRegistry.register("test", "zebra")
        class ZebraProvider(DummyProvider):
            pass

        @ProviderRegistry.register("test", "alpha")
        class AlphaProvider(DummyProvider):
            pass

        providers = ProviderRegistry.list_providers("test")
        assert providers == ["alpha", "zebra"]
