from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry


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

