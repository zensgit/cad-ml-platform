from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry
from src.core.providers.base import ProviderStatus


@pytest.mark.asyncio
async def test_tolerance_knowledge_provider_health_and_process() -> None:
    ProviderRegistry.clear()
    bootstrap_core_provider_registry()

    provider = ProviderRegistry.get("knowledge", "tolerance")
    ok = await provider.health_check(timeout_seconds=0.5)
    assert ok is True
    assert provider.status == ProviderStatus.HEALTHY
    assert provider.last_error is None

    payload = await provider.process(request={})
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"
    counts = payload.get("counts") or {}
    assert counts.get("common_fits", 0) > 0
    assert counts.get("size_ranges", 0) > 0
    assert counts.get("tolerance_grade_tables", 0) > 0


@pytest.mark.asyncio
async def test_standards_knowledge_provider_health_and_process() -> None:
    ProviderRegistry.clear()
    bootstrap_core_provider_registry()

    provider = ProviderRegistry.get("knowledge", "standards")
    ok = await provider.health_check(timeout_seconds=0.5)
    assert ok is True
    assert provider.status == ProviderStatus.HEALTHY
    assert provider.last_error is None

    payload = await provider.process(request={})
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"
    counts = payload.get("counts") or {}
    assert counts.get("threads", 0) > 0
    assert counts.get("bearings", 0) > 0
    assert counts.get("orings", 0) > 0

