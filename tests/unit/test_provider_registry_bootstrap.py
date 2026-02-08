from __future__ import annotations

from src.core.providers import (
    ProviderRegistry,
    bootstrap_core_provider_registry,
    get_core_provider_registry_snapshot,
)


def test_bootstrap_registers_core_domains_and_providers() -> None:
    ProviderRegistry.clear()
    snapshot = bootstrap_core_provider_registry()

    assert snapshot["bootstrapped"] is True
    assert "vision" in snapshot["domains"]
    assert "ocr" in snapshot["domains"]
    assert "classifier" in snapshot["domains"]
    assert "stub" in snapshot["providers"]["vision"]
    assert "deepseek_stub" in snapshot["providers"]["vision"]
    assert "paddle" in snapshot["providers"]["ocr"]
    assert "deepseek_hf" in snapshot["providers"]["ocr"]
    assert "hybrid" in snapshot["providers"]["classifier"]
    assert "graph2d" in snapshot["providers"]["classifier"]
    assert "graph2d_ensemble" in snapshot["providers"]["classifier"]
    assert "v16" in snapshot["providers"]["classifier"]
    assert "v6" in snapshot["providers"]["classifier"]


def test_bootstrap_is_idempotent() -> None:
    ProviderRegistry.clear()
    first = bootstrap_core_provider_registry()
    second = bootstrap_core_provider_registry()

    assert first["total_domains"] == second["total_domains"]
    assert first["total_providers"] == second["total_providers"]
    assert second["bootstrapped"] is True


def test_registry_snapshot_reflects_runtime_state() -> None:
    ProviderRegistry.clear()
    bootstrap_core_provider_registry()
    snapshot = get_core_provider_registry_snapshot()

    assert snapshot["bootstrapped"] is True
    assert snapshot["total_domains"] >= 2
    assert snapshot["total_providers"] >= 4
    assert isinstance(snapshot["providers"], dict)
    assert isinstance(snapshot.get("provider_classes"), dict)
    assert "classifier" in snapshot["provider_classes"]
    assert "hybrid" in snapshot["provider_classes"]["classifier"]
    assert isinstance(snapshot["provider_classes"]["classifier"]["hybrid"], str)
