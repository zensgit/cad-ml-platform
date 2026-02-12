"""Tests for provider plugin bootstrap Prometheus metrics exposure."""

from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry
from src.core.providers.bootstrap import reset_core_provider_plugins_state


@pytest.fixture(autouse=True)
def _plugin_state_isolation():
    ProviderRegistry.clear()
    reset_core_provider_plugins_state()
    yield
    ProviderRegistry.clear()
    reset_core_provider_plugins_state()


def test_provider_plugin_bootstrap_emits_reload_and_cache_metrics(
    require_metrics_enabled, metrics_text, monkeypatch
):
    plugin = "tests.fixtures.provider_plugin_example:bootstrap"
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", plugin)
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    # First call performs load; second call should hit cache.
    bootstrap_core_provider_registry()
    bootstrap_core_provider_registry()

    text = metrics_text()
    if not text:
        pytest.skip("metrics not available in this environment")

    assert 'core_provider_plugin_bootstrap_total{result="reload_ok"}' in text
    assert 'core_provider_plugin_bootstrap_total{result="cache_hit"}' in text
    reload_bucket_lines = [
        line
        for line in text.splitlines()
        if line.startswith("core_provider_plugin_bootstrap_duration_seconds_bucket{")
        and 'result="reload_ok"' in line
    ]
    cache_hit_bucket_lines = [
        line
        for line in text.splitlines()
        if line.startswith("core_provider_plugin_bootstrap_duration_seconds_bucket{")
        and 'result="cache_hit"' in line
    ]
    assert reload_bucket_lines
    assert cache_hit_bucket_lines


def test_provider_plugin_bootstrap_emits_strict_error_metric(
    require_metrics_enabled, metrics_text, monkeypatch
):
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", "does.not.exist.strict")
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")

    with pytest.raises(ModuleNotFoundError):
        bootstrap_core_provider_registry()

    text = metrics_text()
    if not text:
        pytest.skip("metrics not available in this environment")

    assert 'core_provider_plugin_bootstrap_total{result="strict_error"}' in text
    strict_error_bucket_lines = [
        line
        for line in text.splitlines()
        if line.startswith("core_provider_plugin_bootstrap_duration_seconds_bucket{")
        and 'result="strict_error"' in line
    ]
    assert strict_error_bucket_lines
