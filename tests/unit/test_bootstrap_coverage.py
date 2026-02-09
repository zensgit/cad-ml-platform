"""Additional tests for bootstrap module to improve coverage.

Targets uncovered code paths in src/core/providers/bootstrap.py:
- Lines 28-30: _build_snapshot catches exception from get_provider_class
- Line 63: get_core_provider_registry_snapshot without lazy bootstrap
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.providers import bootstrap as bootstrap_module
from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.bootstrap import (
    bootstrap_core_provider_registry,
    get_core_provider_registry_snapshot,
)
from src.core.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def _clear_registry_and_reset_bootstrap():
    """Clear registry and reset bootstrap state."""
    ProviderRegistry.clear()
    bootstrap_module._BOOTSTRAPPED = False
    bootstrap_module._BOOTSTRAP_TS = None
    yield
    ProviderRegistry.clear()
    bootstrap_module._BOOTSTRAPPED = False
    bootstrap_module._BOOTSTRAP_TS = None


class DummyProvider(BaseProvider[ProviderConfig, dict]):
    """Dummy provider for testing."""

    def __init__(self, config: ProviderConfig | None = None):
        super().__init__(config or ProviderConfig(name="dummy", provider_type="test"))

    async def _process_impl(self, request, **kwargs):
        return {"status": "ok"}


# --- bootstrap_core_provider_registry Tests ---


class TestBootstrapCoreProviderRegistry:
    """Tests for bootstrap_core_provider_registry."""

    def test_bootstrap_returns_snapshot(self):
        """bootstrap_core_provider_registry returns a snapshot dict."""
        snapshot = bootstrap_core_provider_registry()

        assert snapshot["bootstrapped"] is True
        assert snapshot["bootstrap_timestamp"] is not None
        assert "total_domains" in snapshot
        assert "total_providers" in snapshot
        assert "domains" in snapshot
        assert "providers" in snapshot
        assert "provider_classes" in snapshot

    def test_bootstrap_sets_global_flags(self):
        """bootstrap_core_provider_registry sets global _BOOTSTRAPPED flag."""
        assert bootstrap_module._BOOTSTRAPPED is False
        assert bootstrap_module._BOOTSTRAP_TS is None

        bootstrap_core_provider_registry()

        assert bootstrap_module._BOOTSTRAPPED is True
        assert bootstrap_module._BOOTSTRAP_TS is not None

    def test_bootstrap_only_sets_flags_once(self):
        """bootstrap_core_provider_registry only sets timestamp on first call."""
        bootstrap_core_provider_registry()
        first_ts = bootstrap_module._BOOTSTRAP_TS

        bootstrap_core_provider_registry()
        second_ts = bootstrap_module._BOOTSTRAP_TS

        assert first_ts == second_ts


# --- get_core_provider_registry_snapshot Tests ---


class TestGetCoreProviderRegistrySnapshot:
    """Tests for get_core_provider_registry_snapshot."""

    def test_snapshot_with_lazy_bootstrap_true(self):
        """get_core_provider_registry_snapshot triggers bootstrap when lazy=True."""
        assert bootstrap_module._BOOTSTRAPPED is False

        snapshot = get_core_provider_registry_snapshot(lazy_bootstrap=True)

        assert bootstrap_module._BOOTSTRAPPED is True
        assert snapshot["bootstrapped"] is True

    def test_snapshot_with_lazy_bootstrap_false(self):
        """Line 63: get_core_provider_registry_snapshot skips bootstrap when lazy=False."""
        assert bootstrap_module._BOOTSTRAPPED is False

        snapshot = get_core_provider_registry_snapshot(lazy_bootstrap=False)

        # Should NOT trigger bootstrap
        assert bootstrap_module._BOOTSTRAPPED is False
        assert snapshot["bootstrapped"] is False
        assert snapshot["bootstrap_timestamp"] is None

    def test_snapshot_after_bootstrap(self):
        """get_core_provider_registry_snapshot returns snapshot after manual bootstrap."""
        bootstrap_core_provider_registry()

        snapshot = get_core_provider_registry_snapshot(lazy_bootstrap=False)

        assert snapshot["bootstrapped"] is True
        assert snapshot["bootstrap_timestamp"] is not None


# --- _build_snapshot Exception Handling Tests ---


class TestBuildSnapshotExceptionHandling:
    """Tests for _build_snapshot exception handling."""

    def test_snapshot_handles_get_provider_class_exception(self):
        """Lines 28-30: _build_snapshot catches exception from get_provider_class."""

        @ProviderRegistry.register("test", "error_provider")
        class ErrorProvider(DummyProvider):
            pass

        # Manually mark as bootstrapped to avoid real bootstrap
        bootstrap_module._BOOTSTRAPPED = True
        bootstrap_module._BOOTSTRAP_TS = 12345.0

        # Patch get_provider_class to raise for this specific provider
        original_get_class = ProviderRegistry.get_provider_class

        def mock_get_class(domain, name):
            if domain == "test" and name == "error_provider":
                raise RuntimeError("Test error")
            return original_get_class(domain, name)

        with patch.object(ProviderRegistry, "get_provider_class", side_effect=mock_get_class):
            snapshot = get_core_provider_registry_snapshot(lazy_bootstrap=False)

        # Should not raise, and should have "unknown" for the error provider
        assert "test" in snapshot["provider_classes"]
        assert snapshot["provider_classes"]["test"]["error_provider"] == "unknown"


# --- Snapshot Content Tests ---


class TestSnapshotContent:
    """Tests for snapshot content structure."""

    def test_snapshot_contains_all_domains(self):
        """Snapshot contains all registered domains."""
        bootstrap_core_provider_registry()

        snapshot = get_core_provider_registry_snapshot()

        # Should contain vision, ocr, classifier, knowledge domains
        assert "vision" in snapshot["domains"]
        assert "ocr" in snapshot["domains"]
        assert "classifier" in snapshot["domains"]
        assert "knowledge" in snapshot["domains"]

    def test_snapshot_provider_classes_format(self):
        """Snapshot provider_classes has correct format."""
        bootstrap_core_provider_registry()

        snapshot = get_core_provider_registry_snapshot()

        # Each domain should have provider name -> class path mapping
        for domain in snapshot["domains"]:
            assert domain in snapshot["provider_classes"]
            for provider_name, class_path in snapshot["provider_classes"][domain].items():
                assert isinstance(provider_name, str)
                assert isinstance(class_path, str)
                # Class path should contain module and class name
                if class_path != "unknown":
                    assert "." in class_path

    def test_snapshot_total_counts_correct(self):
        """Snapshot total counts are accurate."""
        bootstrap_core_provider_registry()

        snapshot = get_core_provider_registry_snapshot()

        # total_domains should match len(domains)
        assert snapshot["total_domains"] == len(snapshot["domains"])

        # total_providers should match sum of providers per domain
        total = sum(len(providers) for providers in snapshot["providers"].values())
        assert snapshot["total_providers"] == total
