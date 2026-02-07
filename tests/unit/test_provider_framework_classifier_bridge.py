from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry
from src.core.providers.classifier import (
    ClassifierRequest,
    bootstrap_core_classifier_providers,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


def test_bootstrap_core_classifier_providers_registers_expected_entries() -> None:
    bootstrap_core_classifier_providers()
    assert ProviderRegistry.exists("classifier", "hybrid") is True
    assert ProviderRegistry.exists("classifier", "graph2d") is True
    assert ProviderRegistry.exists("classifier", "graph2d_ensemble") is True


@pytest.mark.asyncio
async def test_hybrid_provider_process_and_health_check() -> None:
    bootstrap_core_classifier_providers()
    provider = ProviderRegistry.get("classifier", "hybrid")

    result = await provider.process(ClassifierRequest(filename="J2925001-01人孔v2.dxf"))
    assert isinstance(result, dict)
    assert {"label", "confidence", "source"}.issubset(result.keys())

    ok = await provider.health_check()
    assert ok is True
    assert provider.status.value == "healthy"


@pytest.mark.asyncio
async def test_graph2d_provider_process_returns_status_dict() -> None:
    bootstrap_core_classifier_providers()
    provider = ProviderRegistry.get("classifier", "graph2d")

    result = await provider.process(
        ClassifierRequest(filename="x.dxf", file_bytes=b"0")
    )
    assert isinstance(result, dict)
    assert "status" in result

    ok = await provider.health_check()
    assert isinstance(ok, bool)
    assert provider.status.value in {"healthy", "down"}


@pytest.mark.asyncio
async def test_graph2d_provider_rejects_invalid_request_type() -> None:
    bootstrap_core_classifier_providers()
    provider = ProviderRegistry.get("classifier", "graph2d")

    with pytest.raises(TypeError, match="expects ClassifierRequest"):
        await provider.process({"not": "request"})
