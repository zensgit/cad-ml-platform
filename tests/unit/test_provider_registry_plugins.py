from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry
from src.core.providers.bootstrap import reset_core_provider_plugins_state


@pytest.fixture(autouse=True)
def _reset_plugin_cache():
    reset_core_provider_plugins_state()
    yield
    reset_core_provider_plugins_state()


def test_bootstrap_loads_core_provider_plugins(monkeypatch) -> None:
    ProviderRegistry.clear()

    monkeypatch.setenv(
        "CORE_PROVIDER_PLUGINS",
        "tests.fixtures.provider_plugin_example:bootstrap",
    )
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    snapshot = bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("test", "example") is True

    plugins = snapshot.get("plugins") or {}
    assert plugins.get("enabled") is True
    assert "tests.fixtures.provider_plugin_example:bootstrap" in plugins.get("loaded", [])
    cache = plugins.get("cache") or {}
    assert cache.get("reused") is False
    assert cache.get("reason") == "first_load"


def test_bootstrap_plugin_non_strict_captures_errors(monkeypatch) -> None:
    ProviderRegistry.clear()

    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", "does.not.exist")
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "false")

    snapshot = bootstrap_core_provider_registry()
    plugins = snapshot.get("plugins") or {}
    assert plugins.get("enabled") is True
    assert plugins.get("loaded") == []
    assert plugins.get("errors")


def test_bootstrap_plugin_strict_raises(monkeypatch) -> None:
    ProviderRegistry.clear()

    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", "does.not.exist.strict")
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    with pytest.raises(ModuleNotFoundError):
        bootstrap_core_provider_registry()


def test_bootstrap_plugin_reloads_after_registry_clear(monkeypatch) -> None:
    ProviderRegistry.clear()

    plugin = "tests.fixtures.provider_plugin_example:bootstrap"
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", plugin)
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    first = bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("test", "example") is True
    first_plugins = first.get("plugins") or {}
    first_registered = first_plugins.get("registered") or {}
    assert "test/example" in first_registered.get(plugin, [])

    # Simulate test/runtime registry reset without resetting plugin cache.
    ProviderRegistry.clear()

    second = bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("test", "example") is True
    second_plugins = second.get("plugins") or {}
    assert plugin in second_plugins.get("loaded", [])
    second_cache = second_plugins.get("cache") or {}
    assert second_cache.get("reused") is False
    assert second_cache.get("reason") == "missing_registered_provider"
    assert "test/example" in (second_cache.get("missing_registered") or [])


def test_bootstrap_plugin_cache_hit_when_registry_intact(monkeypatch) -> None:
    ProviderRegistry.clear()

    plugin = "tests.fixtures.provider_plugin_example:bootstrap"
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", plugin)
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    first = bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("test", "example") is True
    first_cache = (first.get("plugins") or {}).get("cache") or {}
    assert first_cache.get("reused") is False

    second = bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("test", "example") is True
    second_cache = (second.get("plugins") or {}).get("cache") or {}
    assert second_cache.get("reused") is True
    assert second_cache.get("reason") == "config_match"
