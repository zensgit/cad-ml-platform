from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry
from src.core.providers.classifier import ClassifierRequest
from src.core.providers.bootstrap import reset_core_provider_plugins_state


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    reset_core_provider_plugins_state()
    yield
    ProviderRegistry.clear()
    reset_core_provider_plugins_state()


def test_example_plugin_bootstrap_registers_provider(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORE_PROVIDER_PLUGINS",
        "src.core.provider_plugins.example_classifier:bootstrap",
    )
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    snapshot = bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("classifier", "example_rules") is True

    plugins = snapshot.get("plugins") or {}
    assert "src.core.provider_plugins.example_classifier:bootstrap" in plugins.get(
        "loaded", []
    )


@pytest.mark.asyncio
async def test_example_plugin_provider_process(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORE_PROVIDER_PLUGINS",
        "src.core.provider_plugins.example_classifier:bootstrap",
    )
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    bootstrap_core_provider_registry()
    provider = ProviderRegistry.get("classifier", "example_rules")

    payload = await provider.process(ClassifierRequest(filename="x.dxf"))
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"
    assert payload.get("label") == "mechanical_drawing"
    assert payload.get("source") == "example_rules"
